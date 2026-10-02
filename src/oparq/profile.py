"""Fast, deterministic profiling for sort-key selection."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pyarrow as pa
import pyarrow.compute as pc

from .models import ColumnProfile


DEFAULT_SAMPLE_ROWS = 250_000


def as_table(value: pa.Table | pa.RecordBatch) -> pa.Table:
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    raise TypeError("expected a pyarrow.Table or pyarrow.RecordBatch")


def validate_unique_names(table: pa.Table) -> None:
    names = table.column_names
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate column names are not supported: {duplicates}")


def sample_table(
    table: pa.Table,
    sample_rows: int | None = DEFAULT_SAMPLE_ROWS,
    *,
    strata: int = 16,
) -> pa.Table:
    """Return a deterministic, evenly-spaced sample without random state.

    Contiguous windows are taken across the whole table instead of just using
    its head.  This is both cheap (Arrow slices are zero-copy) and resistant to
    common time/file ordering bias.
    """

    if sample_rows is None or sample_rows >= table.num_rows:
        return table
    if sample_rows <= 0:
        raise ValueError("sample_rows must be positive or None")
    if table.num_rows == 0:
        return table

    strata = max(1, min(strata, sample_rows))
    base, extra = divmod(sample_rows, strata)
    pieces: list[pa.Table] = []
    for index in range(strata):
        length = base + (1 if index < extra else 0)
        if length == 0:
            continue
        if strata == 1:
            start = (table.num_rows - length) // 2
        else:
            start = round(index * (table.num_rows - length) / (strata - 1))
        pieces.append(table.slice(start, length))
    return pa.concat_tables(pieces)


def decode_dictionary(column: pa.ChunkedArray) -> pa.ChunkedArray:
    if pa.types.is_dictionary(column.type):
        return pc.dictionary_decode(column)
    return column


def sortable_type(data_type: pa.DataType) -> bool:
    """Whether oparq treats a type as a safe scalar sort key."""

    if pa.types.is_dictionary(data_type):
        return sortable_type(data_type.value_type)
    return any(
        predicate(data_type)
        for predicate in (
            pa.types.is_boolean,
            pa.types.is_integer,
            pa.types.is_floating,
            pa.types.is_decimal,
            pa.types.is_temporal,
            pa.types.is_string,
            pa.types.is_large_string,
            pa.types.is_binary,
            pa.types.is_large_binary,
            pa.types.is_fixed_size_binary,
        )
    )


def _looks_like_json(column: pa.ChunkedArray, *, checks: int = 24) -> bool:
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        return False
    try:
        values = column.drop_null().slice(0, checks).to_pylist()
    except UnicodeDecodeError:
        # Some externally produced STRING columns contain opaque byte values.
        # Arrow can reorder/write those bytes without decoding them as JSON.
        return False
    candidates = 0
    parsed = 0
    for value in values:
        stripped = value.lstrip()
        if not stripped.startswith(("{", "[")):
            continue
        candidates += 1
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(decoded, (dict, list)):
            parsed += 1
    required = max(3, len(values) // 2)
    return candidates >= required and parsed / candidates >= 0.8


def _distribution(column: pa.ChunkedArray) -> tuple[int, float, float]:
    """Return distinct count, dominant fraction, and Gini impurity."""

    if len(column) == 0:
        return 0, 0.0, 0.0
    values = pc.value_counts(decode_dictionary(column))
    counts = values.field("counts")
    distinct = len(counts)
    dominant = int(pc.max(counts).as_py()) / len(column)
    counts_f64 = pc.cast(counts, pa.float64())
    squared = pc.multiply(counts_f64, counts_f64)
    sum_squared = float(pc.sum(squared).as_py())
    gini = max(0.0, 1.0 - sum_squared / (len(column) ** 2))
    return distinct, dominant, gini


def _run_count(column: pa.ChunkedArray) -> int:
    """Count equal-value runs, treating adjacent nulls/NaNs as equal."""

    if len(column) == 0:
        return 0
    if len(column) == 1:
        return 1
    column = decode_dictionary(column)
    left = column.slice(0, len(column) - 1)
    right = column.slice(1)
    equal = pc.fill_null(pc.equal(left, right), False)
    both_null = pc.and_(pc.is_null(left), pc.is_null(right))
    same = pc.or_(equal, both_null)
    if pa.types.is_floating(column.type):
        both_nan = pc.and_(
            pc.fill_null(pc.is_nan(left), False),
            pc.fill_null(pc.is_nan(right), False),
        )
        same = pc.or_(same, both_nan)
    same_count = int(pc.sum(pc.cast(same, pa.int64())).as_py())
    return len(column) - same_count


def _order_sensitive_bytes(column: pa.ChunkedArray) -> int:
    """Approximate value/definition bytes whose order can help on disk.

    Arrow allocates fixed-width value slots even for nulls, while Parquet does
    not store those meaningless values. Discounting them prevents a 99.99%-null
    decimal from looking like a huge compression target.
    """

    rows = len(column)
    if rows == 0:
        return 0
    valid = rows - column.null_count
    validity_bytes = (rows + 7) // 8 if column.null_count else 0
    if pa.types.is_dictionary(column.type):
        return round(column.nbytes * valid / rows) + validity_bytes
    data_type = column.type
    if pa.types.is_string(data_type) or pa.types.is_binary(data_type):
        offsets = sum((len(chunk) + 1) * 4 for chunk in column.chunks)
        payload = max(0, column.nbytes - offsets - validity_bytes)
        return payload + validity_bytes
    if pa.types.is_large_string(data_type) or pa.types.is_large_binary(data_type):
        offsets = sum((len(chunk) + 1) * 8 for chunk in column.chunks)
        payload = max(0, column.nbytes - offsets - validity_bytes)
        return payload + validity_bytes
    value_bytes = max(0, column.nbytes - validity_bytes)
    return round(value_bytes * valid / rows) + validity_bytes


def profile_table(
    value: pa.Table | pa.RecordBatch,
    *,
    sample_rows: int | None = DEFAULT_SAMPLE_ROWS,
    exclude: Iterable[str] = (),
    detect_json: bool = True,
) -> tuple[ColumnProfile, ...]:
    """Profile columns using exact histograms over a stratified sample.

    Arrow has no approximate-NDV Python kernel.  oparq therefore uses the
    compiled exact ``value_counts`` kernel on a bounded sample.  Passing
    ``sample_rows=None`` profiles the entire table exactly.
    """

    table = as_table(value)
    validate_unique_names(table)
    excluded = set(exclude)
    unknown = excluded.difference(table.column_names)
    if unknown:
        raise ValueError(f"excluded columns not in table: {sorted(unknown)}")
    sampled = sample_table(table, sample_rows)

    profiles: list[ColumnProfile] = []
    for field, full_column, sample_column in zip(
        table.schema, table.columns, sampled.columns, strict=True
    ):
        eligible = True
        reason: str | None = None
        if field.name in excluded:
            eligible, reason = False, "excluded"
        elif not sortable_type(field.type):
            eligible, reason = False, "nested, extension, or unsupported type"
        elif detect_json and _looks_like_json(decode_dictionary(sample_column)):
            eligible, reason = False, "JSON-like string"

        distinct = 0
        runs = 0
        dominant = 0.0
        gini = 0.0
        if eligible and sampled.num_rows:
            try:
                distinct, dominant, gini = _distribution(sample_column)
                runs = _run_count(sample_column)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
                eligible, reason = False, "cardinality kernel unsupported"

        repeatability = (
            max(0.0, 1.0 - distinct / sampled.num_rows)
            if sampled.num_rows
            else 0.0
        )
        cluster_gain = max(0.0, (runs - distinct) / runs) if runs else 0.0
        sensitive_bytes = _order_sensitive_bytes(full_column)
        # A key only has room to improve runs that are not already clustered.
        # This avoids selecting a large timestamp/block column whose input is
        # already nearly sorted. Byte weight keeps the objective storage-first.
        potential = float(sensitive_bytes) * cluster_gain * gini
        if eligible and distinct <= 1:
            reason = "constant in planning sample"

        profiles.append(
            ColumnProfile(
                name=field.name,
                arrow_type=str(field.type),
                byte_size=full_column.nbytes,
                order_sensitive_bytes=sensitive_bytes,
                null_count=full_column.null_count,
                sampled_rows=sampled.num_rows,
                sample_distinct=distinct,
                sample_runs=runs,
                dominant_fraction=dominant,
                gini_impurity=gini,
                repeatability=repeatability,
                cluster_gain=cluster_gain,
                potential_bytes=potential,
                eligible=eligible,
                reason=reason,
            )
        )
    return tuple(profiles)
