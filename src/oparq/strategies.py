"""Additional compression-oriented ordering strategies.

The functions in this module deliberately separate two decisions:

* :func:`entropy_order` and :func:`payload_benefit_order` choose ordinary
  lexicographic sort keys.  Those keys retain their natural value ordering and
  can therefore be recorded in Parquet sorting metadata.
* :func:`frequency_sort_indices` changes the order of values within each key,
  placing frequent values first.  It can create long runs, but the ranked keys
  are *not* naturally sorted.  Only ``prefix_keys`` may be advertised as
  sorted in Parquet metadata.

Frequency ordering is the practical, Arrow-native relative of the
frequency-aware row orders described by Lemire, Kaser, and Gutarra in
"Reordering Rows for Better Compression: Beyond the Lexicographic Order".
It is intentionally simpler than Vortex: it remains a conventional
lexicographic sort after replacing values with deterministic frequency ranks.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2

import pyarrow as pa
import pyarrow.compute as pc

from .models import ColumnProfile
from .profile import as_table, decode_dictionary, sortable_type
from .sorting import sort_options


def effective_cardinality(profile: ColumnProfile) -> float:
    """Return the inverse-Simpson effective cardinality of a profile.

    ``ColumnProfile.gini_impurity`` is ``1 - sum(p**2)``.  Therefore
    ``1 / (1 - gini)`` is the Hill number of order two: the number of equally
    likely values that would have the same collision probability.  Unlike raw
    distinct count, it recognizes that a highly skewed 100-value column can
    behave like a two-value column for run formation.
    """

    if profile.sample_distinct <= 0 or profile.sampled_rows <= 0:
        return 0.0
    # The exact lower bound prevents floating-point cancellation from making
    # an effective cardinality exceed the number of observed rows.
    collision_probability = max(
        1.0 / profile.sampled_rows,
        1.0 - profile.gini_impurity,
    )
    estimate = 1.0 / collision_probability
    return min(float(profile.sample_distinct), estimate)


def entropy_order(
    candidates: Sequence[ColumnProfile],
    slots: int,
) -> list[str]:
    """Order keys from low to high effective cardinality.

    This is an entropy-aware version of ascending-cardinality ordering.  Raw
    distinct count is retained as a deterministic secondary key, followed by
    estimated byte benefit and the column name.
    """

    if slots <= 0:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            effective_cardinality(item),
            item.sample_distinct,
            -item.potential_bytes,
            item.name,
        ),
    )
    return [item.name for item in ordered[:slots]]


def payload_benefit(profile: ColumnProfile) -> float:
    """Estimate useful byte benefit per bit of key grain consumed.

    ``potential_bytes`` already discounts payload that is constant, already
    clustered, or unlikely to form useful runs.  Dividing by the information
    needed to distinguish the column's effective values stops a large but
    nearly unique identifier from automatically consuming the first key slot.
    """

    if profile.potential_bytes <= 0:
        return 0.0
    grain_bits = max(1.0, log2(1.0 + effective_cardinality(profile)))
    return profile.potential_bytes / grain_bits


def payload_benefit_order(
    candidates: Sequence[ColumnProfile],
    slots: int,
) -> list[str]:
    """Order keys by modeled payload/run benefit per unit of key entropy."""

    if slots <= 0:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            -payload_benefit(item),
            effective_cardinality(item),
            item.sample_distinct,
            -item.potential_bytes,
            item.name,
        ),
    )
    return [item.name for item in ordered[:slots]]


def frequency_metadata_keys(prefix_keys: Sequence[str] = ()) -> tuple[str, ...]:
    """Return the only keys safe to publish as naturally sorted.

    Frequency ranks are data-dependent and do not follow the physical type's
    natural order.  A reader may rely on the natural prefix, but must not be
    told that frequency-ranked keys are ascending even when a particular data
    set happens to give the same order.
    """

    return tuple(prefix_keys)


def _validated_keys(
    table: pa.Table,
    keys: Sequence[str],
    prefix_keys: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ranked = tuple(keys)
    prefix = tuple(prefix_keys)
    if len(set(ranked)) != len(ranked):
        raise ValueError("keys contains duplicate columns")
    if len(set(prefix)) != len(prefix):
        raise ValueError("prefix_keys contains duplicate columns")
    overlap = set(ranked).intersection(prefix)
    if overlap:
        raise ValueError(
            f"frequency keys and prefix_keys must be disjoint: {sorted(overlap)}"
        )
    requested = set(ranked).union(prefix)
    missing = requested.difference(table.column_names)
    if missing:
        raise ValueError(f"sort columns not in table: {sorted(missing)}")
    unsupported = [
        name for name in (*prefix, *ranked) if not sortable_type(table[name].type)
    ]
    if unsupported:
        raise ValueError(f"unsupported scalar sort columns: {unsupported}")
    return ranked, prefix


def _frequency_rank(column: pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Map non-null values to frequency-descending, value-ascending ranks."""

    values = decode_dictionary(column)
    non_null = values.drop_null()
    if len(non_null) == 0:
        return pa.nulls(len(values), type=pa.int64())

    histogram = pc.value_counts(non_null)
    distinct_values = histogram.field("values")
    counts = histogram.field("counts")
    histogram_table = pa.table(
        [counts, distinct_values],
        names=["__frequency", "__value"],
    )
    order = pc.sort_indices(
        histogram_table,
        **sort_options([("__frequency", "descending"), ("__value", "ascending")], "at_end"),
    )
    ranked_values = pc.take(distinct_values, order)
    # Since null is absent from ranked_values, index_in leaves input nulls as
    # null.  The final sort can then honor the caller's null placement while
    # still clustering every null into a single run.
    return pc.cast(pc.index_in(values, value_set=ranked_values), pa.int64())


def frequency_sort_indices(
    table: pa.Table,
    keys: Sequence[str],
    *,
    prefix_keys: Sequence[str] = (),
    null_placement: str = "at_end",
) -> pa.Array:
    """Return a frequency-ranked row permutation using Arrow kernels.

    Rows are first ordered naturally by ``prefix_keys``.  Within equal-prefix
    ranges, every column in ``keys`` is ordered by its *full-table* value
    frequency (most frequent first).  Equal-frequency values use natural
    ascending value order as a deterministic tie-break.

    The implementation materializes only temporary integer rank columns and a
    sort permutation; it does not convert rows to Python objects.  The ranked
    keys are not naturally sorted, so callers writing Parquet sorting metadata
    must publish only :func:`frequency_metadata_keys`.
    """

    arrow_table = as_table(table)
    if null_placement not in {"at_start", "at_end"}:
        raise ValueError("null_placement must be 'at_start' or 'at_end'")
    ranked, prefix = _validated_keys(arrow_table, keys, prefix_keys)

    if not ranked and not prefix:
        return pa.array(range(arrow_table.num_rows), type=pa.uint64())

    temporary_columns: list[pa.Array | pa.ChunkedArray] = []
    temporary_names: list[str] = []
    sort_keys: list[tuple[str, str]] = []

    for index, name in enumerate(prefix):
        temporary_name = f"__oparq_prefix_{index}"
        temporary_columns.append(decode_dictionary(arrow_table[name]))
        temporary_names.append(temporary_name)
        sort_keys.append((temporary_name, "ascending"))

    for index, name in enumerate(ranked):
        temporary_name = f"__oparq_frequency_rank_{index}"
        temporary_columns.append(_frequency_rank(arrow_table[name]))
        temporary_names.append(temporary_name)
        sort_keys.append((temporary_name, "ascending"))

    key_table = pa.table(temporary_columns, names=temporary_names)
    return pc.sort_indices(key_table, **sort_options(sort_keys, null_placement))


__all__ = [
    "effective_cardinality",
    "entropy_order",
    "frequency_metadata_keys",
    "frequency_sort_indices",
    "payload_benefit",
    "payload_benefit_order",
]
