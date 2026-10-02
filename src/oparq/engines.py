"""Optional bounded-memory sorting through DuckDB's local execution engine.

The input can be an Arrow scanner backed by a local or remote filesystem.
DuckDB consumes its batches and spills global sorting to ``temp_directory``;
it does not need a downloaded copy of the original Parquet objects.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa


def require_duckdb() -> Any:
    """Import the optional wheel, rejecting an unrelated namespace directory."""

    try:
        import duckdb
    except ImportError as error:
        raise ImportError("DuckDB sorting requires pip install 'oparq[duckdb]'") from error
    if not hasattr(duckdb, "connect"):
        raise ImportError("DuckDB sorting requires pip install 'oparq[duckdb]'")
    return duckdb


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def natural_order_sql(schema: pa.Schema, sort_keys: Sequence[str], null_placement: str) -> str:
    """DuckDB ordering matching Arrow, including NaNs with nulls at start."""

    nulls = "FIRST" if null_placement == "at_start" else "LAST"
    expressions = []
    for name in sort_keys:
        identifier = quote_identifier(name)
        dtype = schema.field(name).type
        if pa.types.is_dictionary(dtype):
            dtype = dtype.value_type
        if null_placement == "at_start" and pa.types.is_floating(dtype):
            expressions.append(
                f"CASE WHEN {identifier} IS NULL THEN 0 "
                f"WHEN isnan({identifier}) THEN 1 ELSE 2 END ASC"
            )
        expressions.append(f"{identifier} ASC NULLS {nulls}")
    return ", ".join(expressions)


def _validate_type(dtype: pa.DataType, path: str) -> None:
    # DuckDB's zoned timestamps use microseconds, and its TIME/INTERVAL types
    # cannot preserve arbitrary nanoseconds. Fail before executing a query.
    if pa.types.is_timestamp(dtype) and dtype.tz and dtype.unit == "ns":
        raise TypeError(f"DuckDB cannot preserve nanosecond zoned timestamp {path}")
    if pa.types.is_time64(dtype) and dtype.unit == "ns":
        raise TypeError(f"DuckDB cannot preserve nanosecond time {path}")
    if pa.types.is_interval(dtype):
        raise TypeError(f"DuckDB INTERVAL cannot preserve nanoseconds in {path}")
    if pa.types.is_duration(dtype) or isinstance(dtype, pa.ExtensionType):
        raise TypeError(f"DuckDB cannot guarantee lossless conversion of {path}: {dtype}")
    if pa.types.is_union(dtype) or pa.types.is_run_end_encoded(dtype):
        raise TypeError(f"DuckDB cannot guarantee lossless conversion of {path}: {dtype}")
    # DuckDB cannot import these Arrow types or restore them from its results.
    if (pa.types.is_null(dtype) or pa.types.is_float16(dtype) or pa.types.is_decimal256(dtype)
            or pa.types.is_list_view(dtype) or pa.types.is_large_list_view(dtype)):
        raise TypeError(f"DuckDB cannot round-trip {path}: {dtype}; use engine='arrow'")
    if pa.types.is_dictionary(dtype):
        _validate_type(dtype.value_type, path)
    elif pa.types.is_struct(dtype):
        for field in dtype:
            _validate_type(field.type, f"{path}.{field.name}")
    elif pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        _validate_type(dtype.value_type, f"{path}[]")
    elif pa.types.is_map(dtype):
        _validate_type(dtype.key_type, f"{path}.key")
        _validate_type(dtype.item_type, f"{path}.value")


def validate_duckdb_schema(schema: pa.Schema) -> None:
    """Reject schemas DuckDB cannot round-trip before any file is published.

    Known value-lossy types are rejected explicitly. An empty-table round trip
    then catches any other type the installed DuckDB cannot import or return.
    """

    if len(set(schema.names)) != len(schema):
        raise ValueError("DuckDB sorting requires unique column names")
    for field in schema:
        _validate_type(field.type, field.name)
    connection = require_duckdb().connect()
    try:
        connection.register("oparq_schema_probe", pa.Table.from_batches([], schema=schema))
        result = connection.execute("SELECT * FROM oparq_schema_probe").to_arrow_table()
        result.cast(schema, safe=True)
    except Exception as error:
        raise TypeError(f"DuckDB cannot round-trip this schema losslessly: {error}") from error
    finally:
        connection.close()


def restore_batch_schema(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    """Restore Arrow widths, dictionary values, timezone and nullability.

    Safe casts reject fractional timestamps and out-of-range numerics. This
    function does not silently infer a new output schema from SQL results.
    """

    table = pa.Table.from_batches([batch])
    if table.column_names != schema.names:
        raise ValueError("DuckDB returned different columns from the input schema")
    try:
        table = table.cast(schema, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as error:
        raise TypeError(f"DuckDB result cannot be restored losslessly: {error}") from error
    for field, column in zip(schema, table.columns):
        if not field.nullable and column.null_count:
            raise ValueError(f"DuckDB introduced nulls in non-nullable column {field.name}")
    batches = table.combine_chunks().to_batches()
    return batches[0] if batches else pa.RecordBatch.from_pylist([], schema=schema)


@contextmanager
def duckdb_sorted_batches(
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
    sort_keys: Sequence[str],
    *,
    temp_directory: str | Path,
    memory_limit: str = "6GB",
    max_temp_directory_size: str = "18GB",
    batch_size: int = 262_144,
    threads: int = 4,
    null_placement: str = "at_end",
) -> Iterator[Iterator[pa.RecordBatch]]:
    """Globally sort a streaming Arrow source with stable input-order ties.

    Use this as a context manager so the SQL connection and any spilled sort
    data are released even if writing fails. Times spent fetching batches
    include scan, sort, and gather; DuckDB does not expose an Arrow-style
    permutation/gather split through this interface.

    Arrow STRING buffers are validated as UTF-8 before crossing the engine
    boundary. DuckDB can silently replace malformed strings when importing an
    Arrow batch; this backend rejects such input instead of altering its bytes.
    """

    if not sort_keys:
        # Batches stream straight to the writer; DuckDB never converts them.
        yield iter(batches)
        return
    validate_duckdb_schema(schema)
    if len(set(sort_keys)) != len(sort_keys):
        raise ValueError("sort_keys contains duplicate columns")
    unknown = set(sort_keys).difference(schema.names)
    if unknown:
        raise ValueError(f"sort keys not in input: {sorted(unknown)}")
    if null_placement not in {"at_start", "at_end"}:
        raise ValueError("null_placement must be 'at_start' or 'at_end'")
    if batch_size <= 0 or threads <= 0:
        raise ValueError("batch_size and threads must be positive")
    directory = Path(temp_directory)
    directory.mkdir(parents=True, exist_ok=True)
    duckdb = require_duckdb()
    connection = duckdb.connect(config={
        "memory_limit": memory_limit,
        "temp_directory": str(directory.resolve()),
        "max_temp_directory_size": max_temp_directory_size,
        "threads": threads,
        "preserve_insertion_order": True,
    })
    row_id = "__oparq_input_row_number"
    while row_id in schema.names:
        row_id += "_"
    columns = ", ".join(quote_identifier(name) for name in schema.names)
    ordering = natural_order_sql(schema, sort_keys, null_placement)
    validation_error: ValueError | None = None

    def validated_batches() -> Iterator[pa.RecordBatch]:
        nonlocal validation_error
        for index, batch in enumerate(batches):
            if not isinstance(batch, pa.RecordBatch):
                validation_error = ValueError(f"DuckDB input batch {index} is not an Arrow RecordBatch")
                raise validation_error
            if not batch.schema.equals(schema, check_metadata=False):
                validation_error = ValueError(f"DuckDB input batch {index} changed the source schema")
                raise validation_error
            try:
                # Full validation recursively checks strings, dictionaries and
                # nested arrays. Structural validation alone omits UTF-8 checks.
                batch.validate(full=True)
            except pa.ArrowException as error:
                validation_error = ValueError(
                    f"DuckDB refuses invalid Arrow input batch {index}; "
                    f"lossless conversion is not guaranteed: {error}"
                )
                raise validation_error from error
            yield batch

    reader = pa.RecordBatchReader.from_batches(schema, validated_batches())
    try:
        connection.register("oparq_input", reader)
        query = (
            f"SELECT {columns} FROM (SELECT *, row_number() OVER () AS "
            f"{quote_identifier(row_id)} FROM oparq_input) "
            f"ORDER BY {ordering}, {quote_identifier(row_id)}"
        )
        result = connection.execute(query).to_arrow_reader(batch_size)
        yield (restore_batch_schema(batch, schema) for batch in result)
    except Exception as error:
        # The Arrow C-stream callback wraps Python iterator errors in an engine
        # exception. Recover our actionable validation error for the caller.
        if validation_error is not None and error is not validation_error:
            raise validation_error from error
        raise
    finally:
        connection.close()
