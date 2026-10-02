#!/usr/bin/env python3
"""Compare Arrow and DuckDB using saved keys and every row of each source file.

This deliberately follows ``rewrite_dataset`` semantics: sort within each
physical file, retain file boundaries, and reuse fixed keys learned by the
full-corpus benchmark. It is not a global sort across an entire partition.
All paths use the same PyArrow writer settings and a forced no-sort rewrite.
Only one source file is decoded for exact validation at a time. Timings are
hardware/cache dependent; an optional SQL engine is not inherently faster.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import oparq
from oparq.dataset import rewrite_file
from oparq.engines import require_duckdb
from oparq.sorting import sort_options
from oparq.storage import inspect_compression, resolve_compression

if __package__:
    from .full_corpus import _checkpoint, _snapshot, _sources, _stable_inventory
    from . import run_suite as suite
else:
    from full_corpus import _checkpoint, _snapshot, _sources, _stable_inventory
    import run_suite as suite


REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPOSITORY / "benchmarks/results/full-corpus-2026-09-30.json"
DEFAULT_OUTPUT = REPOSITORY / "benchmarks/results/saved-plan-engines-2026-09-30.json"


def _array_equal(left: pa.Array, right: pa.Array) -> bool:
    """Exact logical equality, with matching NaNs and nulls treated as equal."""

    if not left.type.equals(right.type) or len(left) != len(right):
        return False
    if left.equals(right):
        return True
    dtype = left.type
    if pa.types.is_dictionary(dtype):
        return _array_equal(left.dictionary_decode(), right.dictionary_decode())
    nulls = pc.is_null(left)
    if not nulls.equals(pc.is_null(right)):
        return False
    if pa.types.is_floating(dtype):
        values_equal = pc.fill_null(pc.equal(left, right), False)
        nans_equal = pc.and_(pc.fill_null(pc.is_nan(left), False),
                             pc.fill_null(pc.is_nan(right), False))
        return bool(pc.all(pc.or_(pc.or_(values_equal, nans_equal), nulls)).as_py())
    # Ignore children hidden by a parent null, exactly as Arrow equality does.
    # Filtering also normalizes sliced list offsets before comparing children.
    if pa.types.is_nested(dtype):
        valid = pc.invert(nulls)
        left = pc.filter(left, valid)
        right = pc.filter(right, valid)
        if pa.types.is_struct(dtype):
            return all(_array_equal(left.field(index), right.field(index))
                       for index in range(dtype.num_fields))
        if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
            return left.offsets.equals(right.offsets) and _array_equal(left.values, right.values)
        if pa.types.is_fixed_size_list(dtype):
            return _array_equal(left.values, right.values)
        if pa.types.is_map(dtype):
            return (left.offsets.equals(right.offsets)
                    and _array_equal(left.keys, right.keys)
                    and _array_equal(left.items, right.items))
    return False


def _validate_output(source: Path, output: Path, keys: Sequence[str],
                     null_placement: str, row_group_size: int,
                     batch_size: int) -> dict[str, Any]:
    """Compare every output value and stable tie against independent Arrow sort.

    No probabilistic content hash is used. The input file stays in memory, but
    the expected reordered table is gathered only one output batch at a time.
    Validation time is explicitly excluded from rewrite performance timings.
    """

    started = perf_counter()
    original = pq.ParquetFile(source).read()
    input_read_seconds = perf_counter() - started
    started = perf_counter()
    indices = (pc.sort_indices(original.select(keys),
                               **sort_options([(key, "ascending") for key in keys], null_placement))
               if keys else None)
    expected_permutation_seconds = perf_counter() - started
    footer = pq.read_metadata(output)
    actual_schema = pq.read_schema(output)
    schema_matches = actual_schema.equals(original.schema, check_metadata=False)
    offset = 0
    unequal_columns: set[str] = set()
    started = perf_counter()
    if schema_matches:
        for batch in pq.ParquetFile(output).iter_batches(batch_size=batch_size):
            expected = (original.take(indices.slice(offset, batch.num_rows)) if indices is not None
                        else original.slice(offset, batch.num_rows))
            for index, name in enumerate(original.schema.names):
                if not _array_equal(expected.column(index).combine_chunks(), batch.column(index)):
                    unequal_columns.add(name)
            offset += batch.num_rows
    expected_groups = [min(row_group_size, original.num_rows - position)
                       for position in range(0, original.num_rows, row_group_size)]
    actual_groups = [footer.row_group(index).num_rows for index in range(footer.num_row_groups)]
    sorting_columns = (pq.SortingColumn.from_ordering(original.schema,
                                                    [(key, "ascending") for key in keys],
                                                    null_placement=null_placement) if keys else ())
    sorting_metadata_matches = all(footer.row_group(index).sorting_columns == sorting_columns
                                   for index in range(footer.num_row_groups))
    checks = {
        "row_count": offset == original.num_rows == footer.num_rows,
        "schema_preserved": schema_matches,
        "every_column_matches_exact_stable_arrow_order": not unequal_columns and schema_matches,
        "row_group_geometry": actual_groups == expected_groups,
        "sorting_metadata": sorting_metadata_matches,
    }
    return {
        "checks": checks,
        "unequal_columns": sorted(unequal_columns),
        "status": "pass" if all(checks.values()) else "fail",
        "validation_input_read_seconds": input_read_seconds,
        "validation_expected_permutation_seconds": expected_permutation_seconds,
        "validation_compare_seconds": perf_counter() - started,
        "row_group_rows": actual_groups,
        "compressed_column_bytes": sum(footer.row_group(group).column(column).total_compressed_size
                                       for group in range(footer.num_row_groups)
                                       for column in range(footer.num_columns)),
    }


def _saved_plan(checkpoint: dict[str, Any], name: str, algorithm: str,
                schema: pa.Schema) -> oparq.RewritePlan:
    dataset = next((item for item in checkpoint["datasets"] if item["name"] == name), None)
    if dataset is None:
        raise ValueError(f"no full-corpus result for {name!r}")
    result = next((item for item in dataset["results"]
                   if item["requested_algorithm"] == algorithm and item.get("status") == "pass"), None)
    if result is None:
        raise ValueError(f"no validated {algorithm!r} plan for {name!r}")
    payload = result["plan"]
    if payload.get("value_order", "natural") != "natural":
        raise ValueError("this comparison requires a natural-order saved plan")
    return oparq.RewritePlan(
        algorithm=payload["algorithm"], sort_keys=tuple(payload["sort_keys"]),
        prefix_keys=tuple(payload.get("prefix_keys", ())),
        column_types=tuple((field.name, str(field.type)) for field in schema),
        schema_base64=base64.b64encode(schema.serialize().to_pybytes()).decode("ascii"),
        null_placement=payload.get("null_placement", "at_end"),
        sampled_rows=payload.get("sampled_rows", 0),
    )


@contextmanager
def _measure_input_reads(engine: str, timings: dict[str, float]):
    """Measure Arrow decoding without pretending DuckDB execution is unfused."""

    if engine == "arrow":
        import oparq.dataset as dataset_module

        original = dataset_module.read_parquet_source

        def read(*args: Any, **kwargs: Any):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings["input_read_seconds"] += perf_counter() - started

        with patch.object(dataset_module, "read_parquet_source", side_effect=read):
            yield
    else:
        import oparq.engines as engines_module

        original = engines_module.duckdb_sorted_batches

        def timed_batches(batches: Any) -> Iterator[pa.RecordBatch]:
            iterator = iter(batches)
            while True:
                started = perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    timings["input_read_seconds"] += perf_counter() - started
                    return
                timings["input_read_seconds"] += perf_counter() - started
                yield batch

        @contextmanager
        def measured(schema: pa.Schema, batches: Any, *args: Any, **kwargs: Any):
            with original(schema, timed_batches(batches), *args, **kwargs) as result:
                yield result

        with patch.object(engines_module, "duckdb_sorted_batches", side_effect=measured):
            yield


def _write_case(source: Path, output: Path, plan: oparq.RewritePlan, engine: str,
                args: argparse.Namespace, temporary: Path) -> dict[str, Any]:
    timings = {"input_read_seconds": 0.0}
    started = perf_counter()
    with _measure_input_reads(engine, timings):
        result = rewrite_file(
            source, output, plan=plan, engine=engine, skip_unchanged=False,
            compression="preserve", compression_level=args.compression_level,
            row_group_size=args.row_group_size, temp_directory=temporary,
            memory_limit=args.memory_limit, max_temp_directory_size=args.max_temp_size,
            use_dictionary=True, write_statistics=True,
        )
    wall_seconds = perf_counter() - started
    record = {
        "engine": engine, "bytes": result.file_size,
        "rows": result.plan.total_rows, "planning_seconds": result.planning_seconds,
        "input_read_seconds": timings["input_read_seconds"],
        "sort_seconds": result.sort_seconds,
        "permutation_seconds": result.permutation_seconds,
        "gathering_seconds": result.gathering_seconds,
        "writing_seconds": result.write_seconds,
        "rewrite_wall_seconds": wall_seconds,
        "input_read_is_component_of_sort_seconds": engine == "duckdb",
        "timing_note": ("Arrow input read separate; permutation and gathering measured independently."
                        if engine == "arrow" else
                        "DuckDB scan/sort/gather is combined; input batch decoding is a measured component, not additive."),
    }
    record.update(_validate_output(source, output, plan.sort_keys, plan.null_placement,
                                   args.row_group_size, args.batch_size))
    return record


def benchmark_source(name: str, source: Path, args: argparse.Namespace,
                     report: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    source = source.resolve()
    inventory = _stable_inventory(source, args.min_stable_age)
    paths = [Path(entry["path"]) for entry in inventory["files"]]
    schema = pq.read_schema(paths[0])
    plan = _saved_plan(checkpoint, name, args.algorithm, schema)
    baseline = oparq.RewritePlan(
        algorithm="none", sort_keys=(), prefix_keys=(),
        column_types=plan.column_types, schema_base64=plan.schema_base64,
        null_placement=plan.null_placement,
    )
    old = next((item for item in report["datasets"] if item["name"] == name), None)
    record: dict[str, Any] = {
        "name": name, "inventory": inventory, "fixed_plan": plan.as_dict(),
        "scope": "ALL rows of ALL physical files; sorting separately within each file",
        "files": [], "status": "running",
    }
    if old and old["inventory"]["snapshot"] == inventory["snapshot"] and old["fixed_plan"] == plan.as_dict():
        if old.get("status") == "pass":
            print(f"{name}: complete checkpoint reused", file=sys.stderr, flush=True)
            return
        record["files"] = old.get("files", [])
    if old:
        report["datasets"].remove(old)
    report["datasets"].append(record)
    _checkpoint(report, args.output)
    with tempfile.TemporaryDirectory(prefix="oparq-engines-", dir=args.temp_root) as directory:
        temporary = Path(directory)
        for file_index, path in enumerate(paths):
            relative = path.relative_to(source).as_posix()
            file_record = next((item for item in record["files"] if item["relative_path"] == relative), None)
            if file_record is None:
                settings = resolve_compression(inspect_compression(path), compression="preserve",
                                               compression_level=args.compression_level)
                file_record = {
                    "relative_path": relative, "source_bytes": path.stat().st_size,
                    "writer": {**settings.writer_options(), "row_group_size": args.row_group_size,
                               "use_dictionary": True, "write_statistics": True},
                    "cases": [], "status": "running",
                }
                record["files"].append(file_record)
            # Alternating engine order avoids always giving one engine the
            # later, warmer cache. The forced baseline always runs first.
            sorted_engines = ("arrow", "duckdb") if file_index % 2 == 0 else ("duckdb", "arrow")
            cases = [("none_arrow", "arrow", baseline),
                     *((f"{args.algorithm}_{engine}", engine, plan) for engine in sorted_engines)]
            for case, engine, case_plan in cases:
                if any(item["case"] == case and item.get("status") == "pass" for item in file_record["cases"]):
                    continue
                print(f"{name}/{relative}: {case}; ALL rows", file=sys.stderr, flush=True)
                output = temporary / "output.parquet"
                try:
                    result = _write_case(path, output, case_plan, engine, args, temporary)
                finally:
                    output.unlink(missing_ok=True)
                result["case"] = case
                result["sort_keys"] = list(case_plan.sort_keys)
                file_record["cases"] = [item for item in file_record["cases"] if item["case"] != case]
                file_record["cases"].append(result)
                _checkpoint(report, args.output)
                if result["status"] != "pass":
                    raise RuntimeError(f"exact validation failed: {name}/{relative}/{case}")
                if _snapshot(paths) != inventory["snapshot"]:
                    raise RuntimeError("input changed during comparison")
            file_record["status"] = "pass"
            _checkpoint(report, args.output)
    aggregates = []
    for case in ("none_arrow", f"{args.algorithm}_arrow", f"{args.algorithm}_duckdb"):
        values = [next(item for item in file["cases"] if item["case"] == case) for file in record["files"]]
        aggregate = {"case": case, "files": len(values), "rows": sum(item["rows"] for item in values),
                     "status": "pass" if all(item["status"] == "pass" for item in values) else "fail"}
        for key in ("bytes", "compressed_column_bytes", "planning_seconds", "input_read_seconds",
                    "sort_seconds", "writing_seconds", "rewrite_wall_seconds",
                    "permutation_seconds", "gathering_seconds"):
            aggregate[key] = (sum(item[key] for item in values)
                              if all(item[key] is not None for item in values) else None)
        aggregates.append(aggregate)
    baseline_bytes = aggregates[0]["bytes"]
    for aggregate in aggregates:
        aggregate["savings_vs_no_sort"] = 1 - aggregate["bytes"] / baseline_bytes
    record["results"] = aggregates
    record["status"] = "pass"
    _checkpoint(report, args.output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY / "local/data/source")
    parser.add_argument("--plan-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", type=suite._csv, default=("stock", "hits", "solana"))
    parser.add_argument("--algorithm", choices=("codec_fast", "portfolio"), default="portfolio")
    parser.add_argument("--compression-level", type=int, default=1,
                        help="Controlled level for level-bearing codecs (ZSTD1 by default); NOT a source-level claim")
    parser.add_argument("--row-group-size", type=suite._positive, default=1_000_000)
    parser.add_argument("--batch-size", type=suite._positive, default=262_144)
    parser.add_argument("--memory-limit", default="6GB")
    parser.add_argument("--max-temp-size", default="18GB")
    parser.add_argument("--temp-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--min-stable-age", type=float, default=120)
    args = parser.parse_args(argv)
    checkpoint = json.loads(args.plan_checkpoint.read_text())
    sources = [(name, source) for name, source in _sources(args.data_root) if name in args.datasets]
    missing = set(args.datasets).difference(name for name, _ in sources)
    if missing:
        parser.error(f"datasets missing: {sorted(missing)}")
    configuration = json.loads(json.dumps({key: str(value) if isinstance(value, Path) else value
                                          for key, value in vars(args).items() if key != "output"}))
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report["configuration"] != configuration:
            parser.error("existing report settings differ; select a new --output")
    else:
        report = {
            "format_version": 1, "started_at_utc": datetime.now(UTC).isoformat(),
            "configuration": configuration,
            "versions": {"python": platform.python_version(), "pyarrow": pa.__version__,
                         "duckdb": require_duckdb().__version__, "oparq": oparq.__version__},
            "method": "Saved keys, all rows, per-file ordering. Same PyArrow writer/level/row groups. Exact Arrow-order validation.",
            "compression_level_note": "Explicit controlled level; Parquet footers do not record the source level. Snappy has no level.",
            "bytes_note": "Per-engine sort-plan metadata formatting may differ slightly. compressed_column_bytes compares the encoded data without footer metadata.",
            "timing_note": "No planning search repeated. Validation excluded. Warm-cache effects remain; engines alternate order by file. Do not run alongside heavy jobs.",
            "datasets": [],
        }
    report["status"] = "running"
    _checkpoint(report, args.output)
    for name, source in sources:
        try:
            benchmark_source(name, source, args, report, checkpoint)
        except Exception as error:
            record = next((item for item in report["datasets"] if item["name"] == name), None)
            if record is None:
                record = {"name": name, "files": []}
                report["datasets"].append(record)
            record["status"] = "fail"
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            _checkpoint(report, args.output)
            print(f"{name}: FAILED {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    report["status"] = "pass" if all(item["status"] == "pass" for item in report["datasets"]) else "fail"
    report["requested_datasets_complete"] = all(
        any(item["name"] == name and item["status"] == "pass" for item in report["datasets"])
        for name in args.datasets)
    _checkpoint(report, args.output)
    return 0 if report["status"] == "pass" and report["requested_datasets_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
