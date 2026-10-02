#!/usr/bin/env python3
"""Checkpointed full-row, same-PyArrow-writer benchmarks with external sort.

Only key planning uses a bounded sample. Every result scans/writes/verifies all
input rows. Ordered, disjoint key ranges can bound external-sort disk space;
concatenating their globally sorted rows yields the full global lexicographic
order. Range boundaries use a separate uniform, full-input key-only reservoir,
not the potentially skewed leading rows used by fast key planning. The stable
file/row identity is included so even identical sort keys can be split safely.
Source compression levels are not recoverable from Parquet metadata:
ZSTD level 1 is an explicit controlled comparison, not a claimed source level.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from time import perf_counter, time
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import oparq
from oparq.engines import (
    natural_order_sql, quote_identifier, require_duckdb, restore_batch_schema,
    validate_duckdb_schema,
)
from oparq.planning import plan_sort

if __package__:
    from . import run_suite as suite
else:
    import run_suite as suite


REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPOSITORY / "local" / "data" / "source"
DEFAULT_OUTPUT = REPOSITORY / "benchmarks" / "results" / "full-corpus-2026-09-30.json"
RANGE_SAMPLE_SEED = 20260930
IDENTITY_COLUMNS = ("filename", "file_row_number")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_sql(files: Sequence[Path], *, identity: bool = False) -> str:
    paths = ", ".join(_sql_string(str(path.resolve())) for path in files)
    options = ", filename=true, file_row_number=true" if identity else ""
    return f"read_parquet([{paths}], hive_partitioning=false{options})"


def _connection(directory: Path, args: argparse.Namespace):
    return require_duckdb().connect(config={
        "memory_limit": args.memory_limit,
        "temp_directory": str(directory.resolve()),
        "max_temp_directory_size": args.max_temp_size,
        "threads": args.threads,
        "preserve_insertion_order": True,
    })


def _snapshot(files: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "size": path.stat().st_size,
         "mtime_ns": path.stat().st_mtime_ns}
        for path in files
    ]


def _stable_inventory(source: Path, min_age_seconds: float) -> dict[str, Any]:
    files = suite._parquet_files(source)
    if not files:
        raise ValueError(f"no Parquet files under {source}")
    before = _snapshot(files)
    age = time() - max(path.stat().st_mtime for path in files)
    if age < min_age_seconds:
        raise RuntimeError(f"source was modified {age:.1f}s ago; wait for stable downloads")
    inventory = suite._inventory(source)
    if before != _snapshot(files):
        raise RuntimeError("input changed while reading its Parquet footers")
    schema = pq.read_schema(files[0])
    for path in files[1:]:
        if not pq.read_schema(path).equals(schema, check_metadata=False):
            raise TypeError(f"input schemas differ: {path}")
    validate_duckdb_schema(schema)
    inventory["snapshot"] = before
    inventory["minimum_age_seconds"] = age
    inventory["source_compression_level"] = None
    inventory["source_compression_level_provenance"] = "not encoded in Parquet metadata"
    return inventory


def _iter_source(files: Sequence[Path], schema: pa.Schema, batch_size: int) -> Iterator[pa.RecordBatch]:
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=True):
            yield restore_batch_schema(batch, schema)


def _planning_sample(inventory: dict[str, Any], rows: int, sample_files: int) -> tuple[pa.Table, list[dict[str, Any]]]:
    file_plan = suite._sample_file_plan(inventory["files"], rows, sample_files)
    schema = pq.read_schema(inventory["files"][0]["path"])
    tables = []
    for entry in file_plan:
        remaining = entry["sampled_rows"]
        batches = []
        for batch in pq.ParquetFile(entry["path"]).iter_batches(batch_size=min(8192, remaining)):
            selected = batch.slice(0, remaining)
            batches.append(restore_batch_schema(selected, schema))
            remaining -= selected.num_rows
            if remaining <= 0:
                break
        tables.append(pa.Table.from_batches(batches, schema=schema))
    return pa.concat_tables(tables), file_plan


def _binary_view_type(dtype: pa.DataType) -> pa.DataType:
    """Preserve string buffers as opaque bytes only for malformed-source runs."""

    if pa.types.is_string(dtype):
        return pa.binary()
    if pa.types.is_large_string(dtype):
        return pa.large_binary()
    if pa.types.is_string_view(dtype):
        return pa.binary_view()
    if pa.types.is_dictionary(dtype):
        return pa.dictionary(dtype.index_type, _binary_view_type(dtype.value_type), ordered=dtype.ordered)
    if pa.types.is_struct(dtype):
        return pa.struct([field.with_type(_binary_view_type(field.type)) for field in dtype])
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        field = dtype.value_field.with_type(_binary_view_type(dtype.value_type))
        if pa.types.is_large_list(dtype):
            return pa.large_list(field)
        return pa.list_(field, dtype.list_size) if pa.types.is_fixed_size_list(dtype) else pa.list_(field)
    if pa.types.is_map(dtype):
        return pa.map_(_binary_view_type(dtype.key_type), _binary_view_type(dtype.item_type),
                       keys_sorted=dtype.keys_sorted)
    return dtype


def _binary_view_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([field.with_type(_binary_view_type(field.type)) for field in schema],
                     metadata=schema.metadata)


def _binary_view_batch(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [column.view(field.type) for column, field in zip(batch.columns, _binary_view_schema(schema))],
        schema=_binary_view_schema(schema),
    )


def _restore_binary_view_batch(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    batch = restore_batch_schema(batch, _binary_view_schema(schema))
    return pa.RecordBatch.from_arrays(
        [column.view(field.type) for column, field in zip(batch.columns, schema)], schema=schema,
    )


def _register_binary_view_source(connection: Any, files: Sequence[Path], schema: pa.Schema,
                                 batch_size: int, *, columns: Sequence[str] | None = None,
                                 identity: bool = False) -> str:
    """Bridge Parquet through Arrow without decoding/replacing invalid UTF-8.

    STRING buffers must explicitly be viewed as BINARY: passing a malformed
    Arrow STRING directly through the SQL bridge is not guaranteed lossless.
    This benchmark-only compatibility path does not alter input files or the
    production library's strict reader behavior.
    """

    selected = pa.schema([schema.field(name) for name in columns], metadata=schema.metadata) if columns else schema
    output_schema = _binary_view_schema(selected)
    if identity:
        output_schema = output_schema.append(pa.field("filename", pa.string()))
        output_schema = output_schema.append(pa.field("file_row_number", pa.int64()))

    def batches() -> Iterator[pa.RecordBatch]:
        for path in files:
            row_offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
                batch = _binary_view_batch(restore_batch_schema(batch, selected), selected)
                if identity:
                    arrays = [*batch.columns,
                              pa.repeat(pa.scalar(str(path.resolve())), batch.num_rows),
                              pa.array(range(row_offset, row_offset + batch.num_rows), type=pa.int64())]
                    batch = pa.RecordBatch.from_arrays(arrays, schema=output_schema)
                row_offset += batch.num_rows
                yield batch

    relation = "oparq_binary_view_source"
    connection.register(relation, pa.RecordBatchReader.from_batches(output_schema, batches()))
    return quote_identifier(relation)


def _fingerprint(connection: Any, files: Sequence[Path], schema: pa.Schema, *,
                 opaque_strings: bool = False, batch_size: int = 262_144) -> dict[str, Any]:
    columns = ", ".join(quote_identifier(name) for name in schema.names)
    # Two independent seeds, a full integer sum and XOR make accidental row
    # changes extremely unlikely. This is a probabilistic all-row check, not
    # a mathematical proof of multiset equality.
    source = (_register_binary_view_source(connection, files, schema, batch_size)
              if opaque_strings else _read_sql(files))
    query = (
        "SELECT count(*), sum(h1::HUGEINT)::VARCHAR, bit_xor(h1)::VARCHAR, "
        "sum(h2::HUGEINT)::VARCHAR, bit_xor(h2)::VARCHAR FROM ("
        f"SELECT hash(123456789, {columns}) AS h1, hash(987654321, {columns}) AS h2 "
        f"FROM {source})"
    )
    values = connection.execute(query).fetchone()
    return dict(zip(("rows", "sum_hash_1", "xor_hash_1", "sum_hash_2", "xor_hash_2"), values))


def _same_bound(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    for x, y in zip(left, right):
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


def _range_bounds(sample: pa.Table, keys: Sequence[str], ranges: int) -> list[tuple[Any, ...]]:
    if ranges <= 1 or not keys or sample.num_rows == 0:
        return []
    key_table = sample.select(keys)
    indices = pc.sort_indices(key_table, sort_keys=[(key, "ascending") for key in keys])
    ordered = key_table.take(indices)
    bounds: list[tuple[Any, ...]] = []
    for index in range(1, ranges):
        offset = (index * sample.num_rows) // ranges
        row = tuple(ordered[name][offset].as_py() for name in keys)
        if not bounds or not _same_bound(bounds[-1], row):
            bounds.append(row)
    return bounds


def _full_input_key_sample(files: Sequence[Path], keys: Sequence[str],
                           temporary: Path, args: argparse.Namespace, *,
                           schema: pa.Schema | None = None,
                           opaque_strings: bool = False) -> pa.Table:
    """Scan all input keys into a bounded, reproducible uniform reservoir.

    Projection precedes sampling, so wide payload columns are never decoded
    merely to determine ranges. DuckDB's global reservoir retains at most the
    requested number of rows. A seeded reservoir is repeatable only when the
    source iteration order is repeatable: this one key-only scan uses a single
    thread, while the actual range sorts keep the configured thread count.
    """

    if set(IDENTITY_COLUMNS).intersection(keys):
        raise ValueError("sort keys cannot use benchmark identity column names")
    columns = ", ".join(quote_identifier(name) for name in (*keys, *IDENTITY_COLUMNS))
    sample_rows = min(getattr(args, "sample_rows", 250_000), 250_000)
    connection = _connection(temporary, args)
    try:
        connection.execute("SET threads=1")
        source = (_register_binary_view_source(
            connection, files, schema or pq.read_schema(files[0]), args.batch_size,
            columns=keys, identity=True,
        ) if opaque_strings else _read_sql(files, identity=True))
        query = (
            f"SELECT {columns} FROM (SELECT {columns} FROM {source}) "
            f"USING SAMPLE {sample_rows} ROWS (reservoir, {RANGE_SAMPLE_SEED})"
        )
        return connection.execute(query).to_arrow_table()
    finally:
        connection.close()


def _range_predicate(keys: Sequence[str], lower: tuple[Any, ...] | None, upper: tuple[Any, ...] | None) -> tuple[str, list[Any]]:
    key_tuple = "row(" + ", ".join(quote_identifier(key) for key in keys) + ")"
    # An untyped None parameter inside ROW may bind as BLOB. STRUCT-to-STRUCT
    # comparisons then fail against VARCHAR fields. Bind every parameter to
    # its corresponding physical input type, including null bounds.
    placeholder = "row(" + ", ".join(
        f"cast_to_type(?, {quote_identifier(key)})" for key in keys
    ) + ")"
    conditions = []
    parameters: list[Any] = []
    if lower is not None:
        conditions.append(f"{key_tuple} >= {placeholder}")
        parameters.extend(lower)
    if upper is not None:
        conditions.append(f"{key_tuple} < {placeholder}")
        parameters.extend(upper)
    # DuckDB1.5 can rewrite two STRUCT bounds into unsupported BETWEEN.
    # CASE preserves the disjoint range condition without that rewrite.
    if len(conditions) == 2:
        return f"CASE WHEN {conditions[0]} THEN {conditions[1]} ELSE false END", parameters
    return " AND ".join(conditions) or "true", parameters


def _sort_queries(files: Sequence[Path], schema: pa.Schema, keys: Sequence[str],
                  bounds: Sequence[tuple[Any, ...]], *,
                  range_keys: Sequence[str] | None = None,
                  source_relation: str | None = None) -> Iterator[tuple[str, list[Any]]]:
    if "filename" in schema.names or "file_row_number" in schema.names:
        raise ValueError("benchmark input cannot have filename/file_row_number metadata column names")
    columns = ", ".join(quote_identifier(name) for name in schema.names)
    order = natural_order_sql(schema, keys, "at_end")
    source = source_relation or _read_sql(files, identity=True)
    for lower, upper in zip([None, *bounds], [*bounds, None]):
        predicate, parameters = _range_predicate(range_keys or keys, lower, upper)
        yield (
            f"SELECT {columns} FROM {source} WHERE {predicate} "
            f"ORDER BY {order}, filename ASC, file_row_number ASC",
            parameters,
        )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_case(
    files: Sequence[Path], schema: pa.Schema, sample: pa.Table, plan: oparq.SortPlan,
    output: Path, temporary: Path, args: argparse.Namespace, *, codec: str,
    opaque_strings: bool = False,
) -> dict[str, Any]:
    level = args.compression_level if pa.Codec.supports_compression_level(codec) else None
    keys = plan.sort_keys
    if keys and set(IDENTITY_COLUMNS).intersection(schema.names):
        raise ValueError("benchmark input cannot have filename/file_row_number metadata column names")
    range_keys = (*keys, *IDENTITY_COLUMNS)
    range_sampling_seconds = 0.0
    key_sample_rows = 0
    bounds = []
    if keys and args.sort_ranges > 1:
        print("  sampling full-input sort keys for balanced ranges", file=sys.stderr, flush=True)
        started = perf_counter()
        key_sample = _full_input_key_sample(files, keys, temporary, args,
                                           schema=schema, opaque_strings=opaque_strings)
        key_sample_rows = key_sample.num_rows
        bounds = _range_bounds(key_sample, range_keys, args.sort_ranges)
        range_sampling_seconds = perf_counter() - started
        del key_sample
    sorting_columns = (
        pq.SortingColumn.from_ordering(schema, [(key, "ascending") for key in keys])
        if keys else None
    )
    metadata = dict(schema.metadata or {})
    metadata[b"oparq.sort_plan"] = json.dumps(plan.as_dict(include_profiles=False), sort_keys=True).encode()
    output_schema = schema.with_metadata(metadata)
    engine_seconds = range_sampling_seconds
    writing_seconds = 0.0
    restore_seconds = 0.0
    peak_spill_bytes = 0
    rows = 0
    started = perf_counter()
    writer = pq.ParquetWriter(
        output, output_schema, compression=codec, compression_level=level,
        use_dictionary=True, write_statistics=True, sorting_columns=sorting_columns,
    )
    writing_seconds += perf_counter() - started
    pending: list[pa.RecordBatch] = []
    pending_rows = 0

    def consume(batch: pa.RecordBatch) -> None:
        nonlocal rows, pending_rows, writing_seconds, restore_seconds
        started = perf_counter()
        batch = (_restore_binary_view_batch(batch, schema) if opaque_strings and keys
                 else restore_batch_schema(batch, schema))
        restore_seconds += perf_counter() - started
        offset = 0
        rows += batch.num_rows
        while offset < batch.num_rows:
            count = min(args.row_group_size - pending_rows, batch.num_rows - offset)
            pending.append(batch.slice(offset, count))
            pending_rows += count
            offset += count
            if pending_rows == args.row_group_size:
                started = perf_counter()
                writer.write_table(pa.Table.from_batches(pending, schema=output_schema), row_group_size=args.row_group_size)
                writing_seconds += perf_counter() - started
                pending.clear()
                pending_rows = 0

    try:
        if keys:
            source_relation = quote_identifier("oparq_binary_view_source") if opaque_strings else None
            for index, (query, parameters) in enumerate(_sort_queries(
                    files, schema, keys, bounds, range_keys=range_keys,
                    source_relation=source_relation)):
                print(f"  sort range {index + 1}/{len(bounds) + 1}", file=sys.stderr, flush=True)
                connection = _connection(temporary, args)
                try:
                    started = perf_counter()
                    if opaque_strings:
                        _register_binary_view_source(connection, files, schema, args.batch_size, identity=True)
                    reader = connection.execute(query, parameters).to_arrow_reader(args.batch_size)
                    engine_seconds += perf_counter() - started
                    iterator = iter(reader)
                    while True:
                        started = perf_counter()
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            engine_seconds += perf_counter() - started
                            break
                        engine_seconds += perf_counter() - started
                        peak_spill_bytes = max(peak_spill_bytes, _directory_bytes(temporary))
                        consume(batch)
                finally:
                    connection.close()
        else:
            iterator = iter(_iter_source(files, schema, args.batch_size))
            while True:
                started = perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    engine_seconds += perf_counter() - started
                    break
                engine_seconds += perf_counter() - started
                consume(batch)
        if pending:
            started = perf_counter()
            writer.write_table(pa.Table.from_batches(pending, schema=output_schema), row_group_size=args.row_group_size)
            writing_seconds += perf_counter() - started
    finally:
        started = perf_counter()
        writer.close()
        writing_seconds += perf_counter() - started
    return {
        "rows": rows,
        "bytes": output.stat().st_size,
        "scan_sort_gather_seconds": engine_seconds,
        "schema_restore_seconds": restore_seconds,
        "writing_seconds": writing_seconds,
        "permutation_seconds": None,
        "gathering_seconds": None,
        "stage_timing_note": "DuckDB scan/sort/gather are inseparable; baseline stage is Arrow scan",
        "sort_ranges": len(bounds) + 1 if keys else 0,
        "range_bounds": [[str(value) if value is not None else None for value in bound] for bound in bounds],
        "range_sample_method": "full_input_key_reservoir" if key_sample_rows else "none",
        "range_sampling_seconds": range_sampling_seconds,
        "key_sample_rows": key_sample_rows,
        "key_sampling_source_scans": int(bool(keys) and args.sort_ranges > 1),
        "range_sampling_seed": RANGE_SAMPLE_SEED if key_sample_rows else None,
        "range_keys": list(range_keys) if keys else [],
        "range_sampling_note": "Uniform full-input keys plus stable file/row identity; key-only scan included in execution time.",
        "full_width_source_scans_for_sort": len(bounds) + 1 if keys else 1,
        "source_scans_for_sort": len(bounds) + 1 + int(args.sort_ranges > 1) if keys else 1,
        "peak_observed_spill_bytes": peak_spill_bytes,
        "input_reader": ("arrow_parquet" if not keys else
                         "arrow_binary_views" if opaque_strings else "duckdb_parquet"),
        "source_invalid_utf8_preserved": opaque_strings,
    }


def _is_monotonic(files: Sequence[Path], keys: Sequence[str], batch_size: int) -> bool:
    previous: pa.Table | None = None
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(keys)):
            table = pa.Table.from_batches([batch])
            if previous is not None:
                table = pa.concat_tables([previous, table])
            indices = pc.sort_indices(table, sort_keys=[(key, "ascending") for key in keys])
            # Stable Arrow sort returns identity iff every key is monotonic.
            expected = pa.array(range(table.num_rows), type=indices.type)
            if not indices.equals(expected):
                return False
            previous = table.slice(table.num_rows - 1)
    return True


def _checkpoint(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".checkpoint")
    staging.write_text(json.dumps(report, indent=2, default=str) + "\n")
    staging.replace(output)


def _annotate_legacy_range_sampling(report: dict[str, Any]) -> None:
    """Label old validated timings without rerunning or changing their values."""

    for dataset in report.get("datasets", ()):
        for result in dataset.get("results", ()):
            if "range_sample_method" in result:
                continue
            keys = result.get("plan", {}).get("sort_keys", [])
            used_sample = bool(keys) and result.get("sort_ranges", 0) > 1
            result.update({
                "range_sample_method": "planning_sample" if used_sample else "none",
                "range_sampling_seconds": 0.0,
                "key_sample_rows": result.get("plan", {}).get("sampled_rows", 0) if used_sample else 0,
                "key_sampling_source_scans": 0,
                "range_sampling_seed": None,
                "range_keys": keys,
                "full_width_source_scans_for_sort": result.get("source_scans_for_sort", 1),
                "range_sampling_note": "Historical case reused the existing planning sample; no additional source-key scan.",
            })


def _sources(root: Path) -> list[tuple[str, Path]]:
    sources = [(path.name, path) for path in sorted((root / "clickhouse").iterdir())
               if path.is_dir() and any(path.rglob("*.parquet"))]
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name != "clickhouse" and any(path.rglob("*.parquet")):
            sources.append((path.name, path))
    return sources


def benchmark_source(name: str, source: Path, args: argparse.Namespace, report: dict[str, Any]) -> None:
    print(f"{name}: inventory full input", file=sys.stderr, flush=True)
    inventory = _stable_inventory(source, args.min_stable_age)
    files = [Path(entry["path"]) for entry in inventory["files"]]
    schema = pq.read_schema(files[0])
    codecs = inventory["source_codecs"]
    if len(codecs) != 1:
        raise ValueError(f"mixed source codecs need an explicit policy: {codecs}")
    codec = codecs[0].lower()
    if codec == "uncompressed":
        codec = "none"
    record: dict[str, Any] = {
        "name": name, "inventory": inventory,
        "writer": {"compression": codec, "compression_level": args.compression_level if codec == "zstd" else None,
                   "row_group_size": args.row_group_size, "use_dictionary": True, "write_statistics": True},
        "results": [], "status": "running",
    }
    old = next((item for item in report["datasets"] if item["name"] == name), None)
    if old and old.get("status") == "pass" and old["inventory"]["snapshot"] == inventory["snapshot"]:
        print(f"{name}: already complete; checkpoint reused", file=sys.stderr, flush=True)
        return
    if old and old.get("inventory", {}).get("snapshot") == inventory["snapshot"]:
        # Completed, validated cases are safe to resume even if a later case
        # was interrupted. Never reuse a case from a changed source snapshot.
        record["results"] = [item for item in old.get("results", []) if item.get("status") == "pass"]
    if old:
        report["datasets"].remove(old)
    report["datasets"].append(record)
    _checkpoint(report, args.output)
    started = perf_counter()
    sample, file_plan = _planning_sample(inventory, args.sample_rows, args.sample_files)
    record["sample_read_seconds"] = perf_counter() - started
    record["planning_sample"] = {"rows": sample.num_rows, "files": file_plan}
    with tempfile.TemporaryDirectory(prefix="oparq-full-", dir=args.temp_root) as temporary_name:
        temporary = Path(temporary_name)
        spill = temporary / "spill"
        spill.mkdir()
        connection = _connection(spill, args)
        try:
            started = perf_counter()
            opaque_strings = False
            try:
                source_fingerprint = _fingerprint(connection, files, schema)
            except require_duckdb().InvalidInputException as error:
                if "Invalid string encoding found in Parquet" not in str(error):
                    raise
                opaque_strings = True
                record["strict_utf8_failed_fingerprint_seconds"] = perf_counter() - started
                record["data_quality"] = {
                    "invalid_utf8_in_source_string": True,
                    "strict_reader_error": str(error),
                    "handling": "Benchmark-only Arrow STRING-to-BINARY buffer views through SQL; restore original STRING schema without decoding or replacing bytes.",
                    "quality_note": "Malformed source STRING bytes remain malformed, byte-for-byte. No cleaning or repair is performed; the original logical schema is retained and remains nonconforming.",
                    "fingerprint_mode": "all string fields viewed as opaque binary for both source and output",
                }
                print(f"{name}: invalid source UTF-8; using explicit lossless binary views", file=sys.stderr, flush=True)
                source_fingerprint = _fingerprint(connection, files, schema,
                                                  opaque_strings=True, batch_size=args.batch_size)
            record["source_fingerprint_seconds"] = perf_counter() - started
            record["source_fingerprint"] = source_fingerprint
            record["source_fingerprint_mode"] = "binary_string_views" if opaque_strings else "strict_parquet"
        finally:
            connection.close()
        baseline = next((item for item in record["results"] if item["requested_algorithm"] == "none"), None)
        for algorithm in args.algorithms:
            if any(item["requested_algorithm"] == algorithm for item in record["results"]):
                print(f"{name}: {algorithm} validated checkpoint reused", file=sys.stderr, flush=True)
                continue
            print(f"{name}: {algorithm}; {inventory['rows']:,} full rows", file=sys.stderr, flush=True)
            started = perf_counter()
            plan = plan_sort(
                sample, algorithm=algorithm, sample_rows=None,
                trial_sample_rows=args.sample_rows,
                trial_compression=codec,
                trial_compression_level=args.compression_level if codec == "zstd" else None,
            )
            plan = replace(plan, total_rows=inventory["rows"])
            planning_seconds = perf_counter() - started
            # Identical key orders have identical data/writer/row-group output.
            # Reuse only after a full write and validation, and label timings.
            reused = next((item for item in record["results"] if item["plan"]["sort_keys"] == list(plan.sort_keys)
                           and item.get("status") == "pass"), None)
            if reused:
                result = {**reused, "requested_algorithm": algorithm, "plan": plan.as_dict(include_profiles=False),
                          "planning_seconds": planning_seconds, "reused_full_result_of": reused["requested_algorithm"]}
            else:
                output = temporary / f"{algorithm}.parquet"
                result = _write_case(files, schema, sample, plan, output, spill, args,
                                     codec=codec, opaque_strings=opaque_strings)
                result.update({"requested_algorithm": algorithm, "plan": plan.as_dict(include_profiles=False),
                               "planning_seconds": planning_seconds})
                print(f"{name}: verifying {algorithm} across every row", file=sys.stderr, flush=True)
                started = perf_counter()
                connection = _connection(spill, args)
                try:
                    actual_fingerprint = _fingerprint(connection, [output], schema,
                                                      opaque_strings=opaque_strings, batch_size=args.batch_size)
                finally:
                    connection.close()
                actual_schema = pq.read_schema(output)
                schema_matches = actual_schema.equals(schema, check_metadata=False)
                monotonic = _is_monotonic([output], plan.sort_keys, args.batch_size) if plan.sort_keys else None
                checks = {
                    "row_count": result["rows"] == inventory["rows"] == actual_fingerprint["rows"],
                    "schema_preserved": schema_matches,
                    "all_row_multiset_fingerprint": actual_fingerprint == source_fingerprint,
                    "global_keys_monotonic": monotonic,
                    "stable_tie_order": "explicit filename/file_row_number SQL order; exact fixtures tested",
                }
                result["verification_seconds"] = perf_counter() - started
                result["checks"] = checks
                result["status"] = "pass" if all(value is not False for value in checks.values()) else "fail"
                result["full_row_fingerprint"] = actual_fingerprint
                output.unlink()
                if _snapshot(suite._parquet_files(source)) != inventory["snapshot"]:
                    raise RuntimeError("input files changed during the benchmark")
            if algorithm == "none":
                baseline = result
            if baseline:
                result["savings_vs_no_sort"] = (baseline["bytes"] - result["bytes"]) / baseline["bytes"]
            result["bytes_vs_downloaded_source"] = result["bytes"] / inventory["file_size_bytes"] - 1
            record["results"].append(result)
            _checkpoint(report, args.output)
            if result["status"] != "pass":
                raise RuntimeError(f"full-row verification failed for {name}/{algorithm}")
        record["status"] = "pass"
        _checkpoint(report, args.output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", type=suite._csv)
    parser.add_argument("--algorithms", type=suite._csv, default=("none", "codec_fast", "portfolio"))
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--row-group-size", type=suite._positive, default=1_000_000)
    parser.add_argument("--sample-rows", type=suite._positive, default=250_000)
    parser.add_argument("--sample-files", type=suite._positive, default=32)
    parser.add_argument("--batch-size", type=suite._positive, default=262_144)
    parser.add_argument("--sort-ranges", type=suite._positive, default=16)
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--max-temp-size", default="18GB")
    parser.add_argument("--temp-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--threads", type=suite._positive, default=4)
    parser.add_argument("--min-stable-age", type=float, default=120)
    args = parser.parse_args(argv)
    if "none" not in args.algorithms or args.algorithms[0] != "none":
        parser.error("none must be the first algorithm for a controlled baseline")
    if any(algorithm not in oparq.ALGORITHMS for algorithm in args.algorithms):
        parser.error("unknown algorithm")
    if "frequency" in args.algorithms:
        parser.error("this runner implements natural-order plans; frequency needs separate value ranks")
    if shutil.disk_usage(args.temp_root).free < 8 * 1024**3:
        parser.error("at least8GiB free disk required for bounded external sort")
    corpus_sources = _sources(args.data_root)
    sources = corpus_sources
    if args.datasets:
        requested = set(args.datasets)
        sources = [(name, path) for name, path in sources if name in requested]
        missing = requested.difference(name for name, _ in sources)
        if missing:
            parser.error(f"datasets missing: {sorted(missing)}")
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key not in {"output", "datasets"}
    }
    configuration = json.loads(json.dumps(configuration))
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report["configuration"] != configuration:
            parser.error("existing report settings differ; select a new --output")
        _annotate_legacy_range_sampling(report)
    else:
        report = {
            "format_version": 1, "started_at_utc": datetime.now(UTC).isoformat(),
            "configuration": configuration,
            "versions": {"python": platform.python_version(), "pyarrow": pa.__version__,
                         "duckdb": require_duckdb().__version__, "oparq": oparq.__version__},
            "method": "ALL source rows; only planning sample is bounded. Same PyArrow writer and row groups.",
            "fingerprint_note": "Two seeded64-bit hashes with HUGEINT sums+XOR scan every row; probabilistic content check.",
            "compression_level_note": "ZSTD1 controlled assumption; source level unknown. Codec preserved for each dataset.",
            "datasets": [],
        }
    report["corpus_datasets"] = [name for name, _ in corpus_sources]
    report["requested_datasets"] = [name for name, _ in sources]
    report["status"] = "running"
    report.pop("completed_at_utc", None)
    _checkpoint(report, args.output)
    for name, source in sources:
        try:
            benchmark_source(name, source, args, report)
        except Exception as error:
            record = next((item for item in report["datasets"] if item["name"] == name), None)
            if record is None:
                record = {"name": name, "results": []}
                report["datasets"].append(record)
            record["status"] = "fail"
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            _checkpoint(report, args.output)
            print(f"{name}: FAILED {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    # New datasets arriving during the run are pending, never silently counted
    # as fully covered. Existing datasets are inventoried again on resume.
    final_sources = _sources(args.data_root)
    report["corpus_datasets"] = [name for name, _ in final_sources]
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    report["status"] = "pass" if all(item["status"] == "pass" for item in report["datasets"]) else "fail"
    complete_names = {item["name"] for item in report["datasets"] if item["status"] == "pass"}
    report["missing_corpus_datasets"] = [name for name, _ in final_sources if name not in complete_names]
    report["full_corpus_complete"] = not report["missing_corpus_datasets"]
    _checkpoint(report, args.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
