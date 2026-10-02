#!/usr/bin/env python3
"""Run reproducible, controlled oparq benchmarks over multiple datasets.

The default run is intentionally bounded to 100,000 rows per dataset.  Use
``--full`` explicitly to benchmark every row.  Progress is written to stderr;
the JSON report is written to stdout or to ``--output``.

Examples
--------
Run a quick comparison over every bundled ClickHouse dataset::

    uv run python benchmarks/run_suite.py > /tmp/oparq-suite.json

Run selected datasets and algorithms with a controlled output row-group size::

    uv run python benchmarks/run_suite.py \
      --datasets pypi,cell_towers,opensky \
      --algorithms none,cardinality,weighted,runs \
      --rows 250000 --row-group-size 125000 \
      --output benchmarks/results/clickhouse-bounded.json

Run the full inputs (potentially expensive)::

    uv run python benchmarks/run_suite.py --full --algorithms none,weighted

Sample across a sharded dataset without loading every file::

    uv run python benchmarks/run_suite.py \
      --datasets large_dataset --sample-files 12 --rows 100000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from itertools import accumulate
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import oparq


REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPOSITORY / "local" / "data" / "source" / "clickhouse"
DEFAULT_ALGORITHMS = ("none", "cardinality", "weighted")
DEFAULT_ROWS = 100_000
DEFAULT_ROW_GROUP_SIZE = 100_000


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return items


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _optional_level(value: str) -> int | None:
    if value.lower() in {"default", "none"}:
        return None
    return int(value)


def _json_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return float(value)


def _first_attr(value: object, names: Iterable[str]) -> int | float | None:
    """Read a timing field across current and newer oparq result models."""

    for name in names:
        if hasattr(value, name):
            return _json_number(getattr(value, name))
    return None


def _timings(result: object) -> dict[str, int | float | None]:
    """Normalize both the current combined timer and upcoming split timers."""

    planning = _first_attr(
        result,
        ("planning_seconds", "plan_seconds", "profile_and_plan_seconds"),
    )
    sorting = _first_attr(result, ("sorting_seconds", "sort_seconds"))
    permutation = _first_attr(result, ("permutation_seconds",))
    gathering = _first_attr(result, ("gathering_seconds",))
    combined = _first_attr(
        result,
        ("planning_and_sort_seconds", "plan_and_sort_seconds"),
    )
    writing = _first_attr(result, ("write_seconds", "writing_seconds"))

    # Do not invent a planning/sorting split when the installed oparq only
    # exposes the legacy combined value.  Retaining it explicitly makes old
    # and new reports comparable without presenting guessed measurements.
    if combined is None and planning is not None and sorting is not None:
        combined = planning + sorting
    measured = [item for item in (combined, writing) if item is not None]
    return {
        "planning_seconds": planning,
        "sorting_seconds": sorting,
        "permutation_seconds": permutation,
        "gathering_seconds": gathering,
        "planning_and_sort_seconds": combined,
        "writing_seconds": writing,
        "measured_total_seconds": sum(measured) if measured else None,
    }


def _parquet_files(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".parquet":
        return [source]
    if source.is_dir():
        return sorted(source.rglob("*.parquet"))
    raise FileNotFoundError(f"Parquet source not found: {source}")


def _dataset_schema(files: Sequence[Path]) -> pa.Schema:
    return ds.dataset([str(path) for path in files], format="parquet").schema


def _inventory(source: Path) -> dict[str, Any]:
    """Read only Parquet footers and return reproducibility metadata."""

    files = _parquet_files(source)
    schema = _dataset_schema(files)
    total_rows = 0
    row_groups = 0
    compressed_bytes = 0
    uncompressed_bytes = 0
    created_by: set[str] = set()
    codecs: set[str] = set()
    columns: dict[str, dict[str, int]] = {}
    file_entries: list[dict[str, Any]] = []

    for path in files:
        metadata = pq.read_metadata(path)
        total_rows += metadata.num_rows
        row_groups += metadata.num_row_groups
        if metadata.created_by:
            created_by.add(metadata.created_by)
        file_entries.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
            }
        )
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                compressed_bytes += column.total_compressed_size
                uncompressed_bytes += column.total_uncompressed_size
                codecs.add(column.compression)
                sizes = columns.setdefault(
                    column.path_in_schema,
                    {"compressed_bytes": 0, "uncompressed_bytes": 0},
                )
                sizes["compressed_bytes"] += column.total_compressed_size
                sizes["uncompressed_bytes"] += column.total_uncompressed_size

    largest_columns = [
        {"path": name, **sizes}
        for name, sizes in sorted(
            columns.items(),
            key=lambda item: (-item[1]["compressed_bytes"], item[0]),
        )
    ]
    return {
        "source": str(source.resolve()),
        "files": file_entries,
        "file_size_bytes": sum(item["size_bytes"] for item in file_entries),
        "rows": total_rows,
        "row_groups": row_groups,
        "column_chunks_compressed_bytes": compressed_bytes,
        "column_chunks_uncompressed_bytes": uncompressed_bytes,
        "source_compression_ratio": (
            uncompressed_bytes / compressed_bytes if compressed_bytes else None
        ),
        "source_codecs": sorted(codecs),
        "created_by": sorted(created_by),
        "schema": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
        "columns_by_compressed_size": largest_columns,
    }


def _read_table(source: Path, row_limit: int | None) -> pa.Table:
    """Read the full dataset or stop once a deterministic leading bound is met."""

    files = _parquet_files(source)
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    if row_limit is None:
        return dataset.to_table()
    # Dataset.head avoids materializing the rest of a large input.  A leading
    # bound is intentional: it is deterministic and makes repeated benchmark
    # reports byte-for-byte comparable for a fixed source.
    return dataset.head(row_limit)


def _sample_file_plan(
    file_entries: Sequence[dict[str, Any]], row_limit: int, sample_files: int
) -> list[dict[str, Any]]:
    """Select file strata by row position, then divide the row budget between them.

    Footer row counts locate equally spaced points in the sorted file list.
    A large file can contain several points, so fill any duplicate selections
    from the largest remaining gaps in file order. Each selected file contributes
    its leading rows, making the exact sample independent of scanner scheduling.
    """

    nonempty = [entry for entry in file_entries if entry["rows"] > 0]
    count = min(sample_files, row_limit, len(nonempty))
    if not count:
        return []

    cumulative = list(accumulate(entry["rows"] for entry in nonempty))
    total_rows = cumulative[-1]
    selected = {
        bisect_right(cumulative, ((2 * index + 1) * total_rows) // (2 * count))
        for index in range(count)
    }
    while len(selected) < count:
        boundaries = [-1, *sorted(selected), len(nonempty)]
        left, right = max(
            zip(boundaries, boundaries[1:]),
            key=lambda gap: (gap[1] - gap[0] - 1, -gap[0]),
        )
        if left == -1:
            selected.add(0)
        elif right == len(nonempty):
            selected.add(right - 1)
        else:
            selected.add((left + right) // 2)

    indices = sorted(selected)
    capacities = [nonempty[index]["rows"] for index in indices]
    allocations = [0] * count
    remaining = min(row_limit, sum(capacities))
    active = list(range(count))
    while remaining and active:
        share, extra = divmod(remaining, len(active))
        next_active = []
        for position, allocation_index in enumerate(active):
            available = capacities[allocation_index] - allocations[allocation_index]
            take = min(available, share + (position < extra))
            allocations[allocation_index] += take
            remaining -= take
            if take < available:
                next_active.append(allocation_index)
        active = next_active

    return [
        {
            "path": nonempty[index]["path"],
            "source_rows": capacities[position],
            "row_offset": 0,
            "sampled_rows": allocations[position],
        }
        for position, index in enumerate(indices)
    ]


def _read_sampled_table(
    file_entries: Sequence[dict[str, Any]], plan: Sequence[dict[str, Any]]
) -> pa.Table:
    """Read only the requested leading rows from each selected Parquet file."""

    schema = _dataset_schema([Path(entry["path"]) for entry in file_entries])
    if not plan:
        return pa.Table.from_batches([], schema=schema)
    tables = [
        ds.dataset([entry["path"]], format="parquet", schema=schema).head(
            entry["sampled_rows"]
        )
        for entry in plan
    ]
    table = pa.concat_tables(tables)
    expected_rows = sum(entry["sampled_rows"] for entry in plan)
    if table.num_rows != expected_rows:
        raise RuntimeError(
            f"sampled Parquet files changed during the read: expected "
            f"{expected_rows} rows, got {table.num_rows}"
        )
    return table


def _discover(root: Path, selected: Sequence[str] | None) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    available: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.rglob("*.parquet")):
            available[child.name] = child
        elif child.is_file() and child.suffix.lower() == ".parquet":
            available[child.stem] = child

    if selected is None:
        return sorted(available.items())
    missing = sorted(set(selected).difference(available))
    if missing:
        choices = ", ".join(sorted(available))
        raise ValueError(f"unknown datasets {missing}; available: {choices}")
    return [(name, available[name]) for name in selected]


def _algorithm_result(result: object) -> dict[str, Any]:
    return {
        "algorithm": getattr(result, "algorithm"),
        "resolved_algorithm": getattr(result, "resolved_algorithm"),
        "sort_keys": list(getattr(result, "sort_keys")),
        "size_bytes": int(getattr(result, "size_bytes")),
        "savings_fraction": float(getattr(result, "savings_fraction")),
        "timings": _timings(result),
    }


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary_name, output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"directory containing named datasets (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--datasets",
        type=_csv,
        help="comma-separated dataset names; default discovers every dataset",
    )
    parser.add_argument(
        "--algorithms",
        type=_csv,
        default=DEFAULT_ALGORITHMS,
        help="comma-separated algorithms (default: none,cardinality,weighted)",
    )
    bounds = parser.add_mutually_exclusive_group()
    bounds.add_argument(
        "--rows",
        type=_positive,
        default=DEFAULT_ROWS,
        help=f"maximum rows loaded per dataset (default: {DEFAULT_ROWS})",
    )
    bounds.add_argument(
        "--full",
        action="store_true",
        help="load and benchmark every row; may consume substantial time and memory",
    )
    parser.add_argument(
        "--sample-files",
        type=_positive,
        metavar="N",
        help="sample up to N files across each dataset, using at most --rows rows",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--compression-level",
        type=_optional_level,
        default=None,
        metavar="N|default",
    )
    parser.add_argument(
        "--row-group-size",
        type=_positive,
        default=DEFAULT_ROW_GROUP_SIZE,
        help=f"output row-group rows (default: {DEFAULT_ROW_GROUP_SIZE})",
    )
    parser.add_argument(
        "--sample-rows",
        type=_positive,
        default=100_000,
        help="bounded rows used for column profiles (default: 100000)",
    )
    parser.add_argument(
        "--exact-profile",
        action="store_true",
        help="profile every loaded row instead of the bounded profile sample",
    )
    parser.add_argument("--run-sample-rows", type=_positive, default=25_000)
    parser.add_argument("--trial-sample-rows", type=_positive, default=50_000)
    parser.add_argument("--max-keys", type=_positive, default=8)
    parser.add_argument("--candidate-pool-size", type=_positive, default=12)
    parser.add_argument(
        "--fast-candidates",
        "--fast-candidate-count",
        dest="fast_candidate_count",
        type=_positive,
        default=4,
    )
    parser.add_argument(
        "--max-trials",
        "--max-trial-evaluations",
        dest="max_trial_evaluations",
        type=_positive,
        default=24,
    )
    parser.add_argument(
        "--sort-backend",
        choices=("auto", "arrow", "rank"),
        default="auto",
        help="Arrow scalar sort, integer-rank sort, or automatic selection",
    )
    parser.add_argument(
        "--gather-threads",
        type=_non_negative,
        default=0,
        help="parallel column-gather workers; 0 selects oparq automatic mode",
    )
    parser.add_argument(
        "--in-memory-sort",
        action="store_true",
        help="materialize the full reordered table instead of gathering row groups",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON atomically to this path instead of stdout",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first dataset failure instead of recording it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.full and args.sample_files is not None:
        parser.error("--sample-files cannot be combined with --full")
    unknown_algorithms = sorted(set(args.algorithms).difference(oparq.ALGORITHMS))
    if unknown_algorithms:
        choices = ", ".join(oparq.ALGORITHMS)
        raise SystemExit(
            f"unknown algorithms {unknown_algorithms}; available: {choices}"
        )
    datasets = _discover(args.root, args.datasets)
    if not datasets:
        raise SystemExit(f"no Parquet datasets found under {args.root}")

    row_limit = None if args.full else args.rows
    sample_rows = None if args.exact_profile else args.sample_rows
    started_at = datetime.now(UTC)
    report: dict[str, Any] = {
        "format_version": 1,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyarrow": pa.__version__,
            "oparq": oparq.__version__,
        },
        "configuration": {
            "root": str(args.root.resolve()),
            "datasets": [name for name, _ in datasets],
            "algorithms": list(args.algorithms),
            "row_limit": row_limit,
            "sample_files": args.sample_files,
            "profile_sample_rows": sample_rows,
            "run_sample_rows": args.run_sample_rows,
            "trial_sample_rows": args.trial_sample_rows,
            "max_sort_columns": args.max_keys,
            "candidate_pool_size": args.candidate_pool_size,
            "fast_candidate_count": args.fast_candidate_count,
            "max_trial_evaluations": args.max_trial_evaluations,
            "sort_backend": args.sort_backend,
            "gather_threads": args.gather_threads,
            "stream_sort": not args.in_memory_sort,
            "compression": args.compression,
            "compression_level": args.compression_level,
            "row_group_size": args.row_group_size,
            "use_dictionary": True,
            "write_statistics": True,
        },
        "datasets": [],
    }

    suite_started = perf_counter()
    for index, (name, source) in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] {name}: inventory", file=sys.stderr)
        entry: dict[str, Any] = {"name": name}
        try:
            entry["source"] = _inventory(source)
            print(f"[{index}/{len(datasets)}] {name}: read", file=sys.stderr)
            read_started = perf_counter()
            if args.sample_files is None:
                table = _read_table(source, row_limit)
            else:
                plan = _sample_file_plan(
                    entry["source"]["files"], row_limit, args.sample_files
                )
                table = _read_sampled_table(entry["source"]["files"], plan)
            read_seconds = perf_counter() - read_started
            entry["input"] = {
                "benchmarked_rows": table.num_rows,
                "columns": table.num_columns,
                "arrow_nbytes": table.nbytes,
                "read_seconds": read_seconds,
            }
            if args.sample_files is not None:
                entry["input"]["sampling"] = {
                    "mode": "files",
                    "files": plan,
                }

            print(
                f"[{index}/{len(datasets)}] {name}: benchmark "
                f"{','.join(args.algorithms)}",
                file=sys.stderr,
            )
            benchmark_started = perf_counter()
            results = oparq.benchmark(
                table,
                algorithms=args.algorithms,
                compression=args.compression,
                compression_level=args.compression_level,
                row_group_size=args.row_group_size,
                sample_rows=sample_rows,
                run_sample_rows=args.run_sample_rows,
                trial_sample_rows=args.trial_sample_rows,
                max_sort_columns=args.max_keys,
                candidate_pool_size=args.candidate_pool_size,
                fast_candidate_count=args.fast_candidate_count,
                max_trial_evaluations=args.max_trial_evaluations,
                sort_backend=args.sort_backend,
                gather_threads=args.gather_threads,
                stream_sort=not args.in_memory_sort,
            )
            entry["benchmark_wall_seconds"] = perf_counter() - benchmark_started
            entry["results"] = [_algorithm_result(item) for item in results]
            entry["status"] = "ok"
        except Exception as error:  # keep long suites useful after one bad input
            entry["status"] = "error"
            entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            if args.fail_fast:
                report["datasets"].append(entry)
                report["finished_at"] = datetime.now(UTC).isoformat()
                report["suite_wall_seconds"] = perf_counter() - suite_started
                _write_report(report, args.output)
                raise
        report["datasets"].append(entry)

    report["finished_at"] = datetime.now(UTC).isoformat()
    report["suite_wall_seconds"] = perf_counter() - suite_started
    _write_report(report, args.output)
    return 1 if any(item["status"] == "error" for item in report["datasets"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
