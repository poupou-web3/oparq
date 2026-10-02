"""Controlled, same-writer comparisons of oparq algorithms."""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa

from .io import write
from .models import BenchmarkResult
from .profile import as_table


def benchmark(
    value: pa.Table | pa.RecordBatch,
    *,
    algorithms: Iterable[str] = (
        "none",
        "cardinality",
        "weighted",
        "entropy",
        "payload",
        "frequency",
        "portfolio",
        "codec_fast",
        "codec",
        "auto",
    ),
    compression: str | None = "zstd",
    compression_level: int | None = None,
    row_group_size: int | None = None,
    sample_rows: int | None = 250_000,
    run_sample_rows: int = 50_000,
    trial_sample_rows: int = 250_000,
    max_sort_columns: int = 8,
    candidate_pool_size: int = 12,
    fast_candidate_count: int = 4,
    max_trial_evaluations: int = 24,
    sort_backend: str = "auto",
    full: bool = False,
    gather_threads: int = 0,
    stream_sort: bool = True,
) -> tuple[BenchmarkResult, ...]:
    """Compare algorithms with identical data, writer, codec, and row groups."""

    table = as_table(value)
    raw: list[BenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="oparq-benchmark-") as directory:
        root = Path(directory)
        for index, algorithm in enumerate(algorithms):
            result = write(
                table,
                root / f"{index:02d}-{algorithm}.parquet",
                algorithm=algorithm,
                compression=compression,
                compression_level=compression_level,
                row_group_size=row_group_size,
                sample_rows=sample_rows,
                run_sample_rows=run_sample_rows,
                trial_sample_rows=trial_sample_rows,
                max_sort_columns=max_sort_columns,
                candidate_pool_size=candidate_pool_size,
                fast_candidate_count=fast_candidate_count,
                max_trial_evaluations=max_trial_evaluations,
                sort_backend=sort_backend,
                full=full,
                gather_threads=gather_threads,
                stream_sort=stream_sort,
            )
            raw.append(
                BenchmarkResult(
                    algorithm=algorithm,
                    resolved_algorithm=result.plan.algorithm,
                    sort_keys=result.plan.sort_keys,
                    size_bytes=result.file_size,
                    planning_seconds=result.planning_seconds,
                    sort_seconds=result.sort_seconds,
                    write_seconds=result.write_seconds,
                    permutation_seconds=result.permutation_seconds,
                    gathering_seconds=result.gathering_seconds,
                )
            )
    if not raw:
        return ()
    # ``none`` is the controlled input-order baseline regardless of where the
    # caller places it. Keep the historical first-result fallback for custom
    # comparisons that deliberately omit ``none``.
    baseline_result = next(
        (item for item in raw if item.resolved_algorithm == "none"),
        raw[0],
    )
    baseline = baseline_result.size_bytes
    return tuple(
        BenchmarkResult(
            algorithm=item.algorithm,
            resolved_algorithm=item.resolved_algorithm,
            sort_keys=item.sort_keys,
            size_bytes=item.size_bytes,
            planning_seconds=item.planning_seconds,
            sort_seconds=item.sort_seconds,
            write_seconds=item.write_seconds,
            savings_fraction=(baseline - item.size_bytes) / baseline if baseline else 0.0,
            permutation_seconds=item.permutation_seconds,
            gathering_seconds=item.gathering_seconds,
        )
        for item in raw
    )
