#!/usr/bin/env python3
"""Verify complete Parquet row order on a bounded, multi-file source sample.

This is a separate correctness check; its writes do not contribute to the
timings in ``run_suite.py``. The default verifies ``none``, ``codec_fast``, and
``portfolio`` on at most 100,000 rows from six files. For example::

    uv run python -m benchmarks.verify_order data/source/clickhouse/taxi \
      --rows 100000 --sample-files 6 --output benchmarks/results/taxi-order.json
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import oparq
from oparq.core import plan_permutation
from oparq.sorting import sort_indices, use_chunked_gather

if __package__:
    from . import run_suite as suite
else:
    import run_suite as suite

_csv = suite._csv
_inventory = suite._inventory
_optional_level = suite._optional_level
_positive = suite._positive
_read_sampled_table = suite._read_sampled_table
_sample_file_plan = suite._sample_file_plan
_write_report = suite._write_report


DEFAULT_ALGORITHMS = ("none", "codec_fast", "portfolio")
DEFAULT_ROWS = 100_000
DEFAULT_SAMPLE_FILES = 6


def _same_rows(actual: pa.Table, expected: pa.Table) -> bool:
    """Compare every column and row while ignoring schema metadata."""

    return actual.equals(expected, check_metadata=False)


def _verify_case(
    source: pa.Table,
    output: Path,
    result: oparq.WriteResult,
    *,
    row_group_size: int,
    sort_backend: str,
) -> dict[str, Any]:
    plan = result.plan
    permutation = plan_permutation(source, plan, sort_backend=sort_backend)
    expected = source if permutation is None else source.take(permutation)
    actual = pq.read_table(output)
    metadata = pq.read_metadata(output)
    expected_groups = (source.num_rows + row_group_size - 1) // row_group_size

    checks: dict[str, bool | None] = {
        "row_count": actual.num_rows == source.num_rows,
        "all_rows_match_plan": _same_rows(actual, expected),
        "row_groups": metadata.num_row_groups == expected_groups,
        "input_order_preserved": (
            _same_rows(actual, source) if permutation is None else None
        ),
        "natural_permutation_matches_arrow": None,
        "natural_keys_monotonic": None,
    }
    if plan.sorting_keys and plan.value_order == "natural":
        arrow_permutation = sort_indices(
            source,
            plan.sorting_keys,
            null_placement=plan.null_placement,
            sort_backend="arrow",
        )
        checks["natural_permutation_matches_arrow"] = (
            permutation is not None and permutation.equals(arrow_permutation)
        )
        output_indices = sort_indices(
            actual,
            plan.sorting_keys,
            null_placement=plan.null_placement,
            sort_backend="arrow",
        )
        identity = pa.array(range(actual.num_rows), type=output_indices.type)
        checks["natural_keys_monotonic"] = output_indices.equals(identity)

    passed = all(value is not False for value in checks.values())
    return {
        "requested_algorithm": plan.requested_algorithm,
        "resolved_algorithm": plan.algorithm,
        "sort_keys": list(plan.sort_keys),
        "natural_sort_keys": list(plan.sorting_keys),
        "rows": actual.num_rows,
        "bytes": result.file_size,
        "row_groups": metadata.num_row_groups,
        "expected_row_groups": expected_groups,
        "stream_sort_exercised": permutation is not None and source.num_rows > row_group_size,
        "chunked_gather_expected": (
            permutation is not None
            and source.num_rows > row_group_size
            and use_chunked_gather(source, row_group_size)
        ),
        "checks": checks,
        "status": "pass" if passed else "fail",
    }


def verify_source(
    source: Path,
    *,
    rows: int = DEFAULT_ROWS,
    sample_files: int = DEFAULT_SAMPLE_FILES,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    row_group_size: int | None = None,
    compression: str = "zstd",
    compression_level: int | None = None,
    sort_backend: str = "auto",
) -> dict[str, Any]:
    """Write and check each result using the same bounded input rows."""

    inventory = _inventory(source)
    file_plan = _sample_file_plan(inventory["files"], rows, sample_files)
    table = _read_sampled_table(inventory["files"], file_plan)
    if table.num_rows == 0:
        raise ValueError(f"no nonempty Parquet rows under {source}")
    group_rows = row_group_size or max(1, min(16_384, table.num_rows // 4))
    report: dict[str, Any] = {
        "format_version": 1,
        "source": str(source.resolve()),
        "source_rows": inventory["rows"],
        "selected_files": file_plan,
        "sampled_rows": table.num_rows,
        "configuration": {
            "requested_rows": rows,
            "requested_sample_files": sample_files,
            "algorithms": list(algorithms),
            "row_group_size": group_rows,
            "compression": compression,
            "compression_level": compression_level,
            "sort_backend": sort_backend,
        },
        "results": [],
    }
    write_options = {
        "row_group_size": group_rows,
        "stream_sort": True,
        "compression": compression,
        "compression_level": compression_level,
        "sort_backend": sort_backend,
    }
    with tempfile.TemporaryDirectory(prefix="oparq-order-") as directory:
        for algorithm in algorithms:
            output = Path(directory) / f"{algorithm}.parquet"
            try:
                result = oparq.write(table, output, algorithm=algorithm, **write_options)
                case = _verify_case(
                    table,
                    output,
                    result,
                    row_group_size=group_rows,
                    sort_backend=sort_backend,
                )
            except Exception as error:
                case = {
                    "requested_algorithm": algorithm,
                    "status": "fail",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            report["results"].append(case)
        if not any(item.get("stream_sort_exercised") for item in report["results"]):
            if table.num_rows <= group_rows:
                report["streaming_probe"] = {
                    "status": "skipped",
                    "reason": "sample fits within one row group",
                }
            else:
                try:
                    eligible = [
                        profile for profile in oparq.profile_table(table)
                        if profile.eligible
                    ]
                    if eligible:
                        key = min(
                            eligible,
                            key=lambda profile: (
                                profile.sample_distinct <= 1,
                                profile.sample_distinct,
                                profile.byte_size,
                            ),
                        ).name
                        output = Path(directory) / "forced-stream-sort.parquet"
                        result = oparq.write(
                            table, output, algorithm="none", prefix=[key], **write_options
                        )
                        probe = _verify_case(
                            table, output, result,
                            row_group_size=group_rows, sort_backend=sort_backend,
                        )
                        probe["forced_prefix_key"] = key
                        report["streaming_probe"] = probe
                    else:
                        report["streaming_probe"] = {
                            "status": "skipped",
                            "reason": "source has no sortable scalar column",
                        }
                except Exception as error:
                    report["streaming_probe"] = {
                        "status": "fail",
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
    report["streaming_checked"] = any(
        item.get("status") == "pass" and item.get("stream_sort_exercised")
        for item in [*report["results"], report.get("streaming_probe", {})]
    )
    report["status"] = (
        "pass" if (
            all(item["status"] == "pass" for item in report["results"])
            and report.get("streaming_probe", {}).get("status") != "fail"
        )
        else "fail"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Parquet file or dataset directory")
    parser.add_argument("--rows", type=_positive, default=DEFAULT_ROWS)
    parser.add_argument("--sample-files", type=_positive, default=DEFAULT_SAMPLE_FILES)
    parser.add_argument("--algorithms", type=_csv, default=DEFAULT_ALGORITHMS)
    parser.add_argument("--row-group-size", type=_positive)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--compression-level", type=_optional_level, default=None, metavar="N|default"
    )
    parser.add_argument("--sort-backend", choices=("auto", "arrow", "rank"), default="auto")
    parser.add_argument("--output", type=Path, help="write JSON atomically here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    unknown = sorted(set(args.algorithms).difference(oparq.ALGORITHMS))
    if unknown:
        parser.error(f"unknown algorithms: {', '.join(unknown)}")
    report = verify_source(
        args.source,
        rows=args.rows,
        sample_files=args.sample_files,
        algorithms=args.algorithms,
        row_group_size=args.row_group_size,
        compression=args.compression,
        compression_level=args.compression_level,
        sort_backend=args.sort_backend,
    )
    _write_report(report, args.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
