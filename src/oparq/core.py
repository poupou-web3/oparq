"""In-memory table optimization."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import pyarrow as pa

from .models import OptimizationResult, SortPlan
from .planning import plan_sort, sort_indices
from .profile import DEFAULT_SAMPLE_ROWS, as_table
from .sorting import take_table, unreorderable_columns
from .strategies import frequency_sort_indices


def plan_permutation(
    value: pa.Table | pa.RecordBatch,
    plan: SortPlan,
    *,
    null_placement: str | None = None,
    sort_backend: str = "auto",
) -> pa.Array | None:
    """Compute the stable permutation without materializing reordered values."""

    table = as_table(value)
    if table.num_rows != plan.total_rows:
        raise ValueError(
            f"plan has {plan.total_rows} rows but table has {table.num_rows}"
        )
    if not plan.sort_keys or table.num_rows <= 1:
        return None
    effective_null_placement = (
        plan.null_placement if null_placement is None else null_placement
    )
    if plan.value_order == "frequency":
        frequency_keys = plan.sort_keys[len(plan.prefix_keys) :]
        permutation = frequency_sort_indices(
            table,
            frequency_keys,
            prefix_keys=plan.prefix_keys,
            null_placement=effective_null_placement,
        )
    else:
        permutation = sort_indices(
            table,
            plan.sort_keys,
            null_placement=effective_null_placement,
            sort_backend=sort_backend,
        )
    return permutation


def apply_plan(
    value: pa.Table | pa.RecordBatch,
    plan: SortPlan,
    *,
    null_placement: str | None = None,
    sort_backend: str = "auto",
    gather_threads: int = 0,
) -> pa.Table:
    """Apply a plan using Arrow's compiled stable sort and take kernels."""

    if gather_threads < 0:
        raise ValueError("gather_threads must be non-negative")
    table = as_table(value)
    if plan.sort_keys and table.num_rows > 1:
        blocked = unreorderable_columns(table.schema)
        if blocked:
            raise ValueError(f"cannot apply the plan: Arrow cannot reorder {blocked}")
    permutation = plan_permutation(
        table, plan, null_placement=null_placement, sort_backend=sort_backend
    )
    if permutation is None:
        return table
    return take_table(table, permutation, gather_threads=gather_threads)


def optimize(
    value: pa.Table | pa.RecordBatch,
    *,
    algorithm: str = "auto",
    prefix: Iterable[str] = (),
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    sample_rows: int | None = DEFAULT_SAMPLE_ROWS,
    run_sample_rows: int = 50_000,
    trial_sample_rows: int = 250_000,
    max_sort_columns: int = 8,
    candidate_pool_size: int = 12,
    fast_candidate_count: int = 4,
    max_trial_evaluations: int = 24,
    min_run_improvement: float = 0.005,
    min_trial_improvement: float = 0.005,
    trial_compression: str | None = "zstd",
    trial_compression_level: int | None = None,
    detect_json: bool = True,
    null_placement: str = "at_end",
    sort_backend: str = "auto",
    full: bool = False,
    gather_threads: int = 0,
) -> OptimizationResult:
    """Plan and reorder an in-memory Arrow table for Parquet compression."""

    table = as_table(value)
    if gather_threads < 0:
        raise ValueError("gather_threads must be non-negative")
    planning_started = perf_counter()
    plan = plan_sort(
        table,
        algorithm=algorithm,
        prefix=prefix,
        include=include,
        exclude=exclude,
        sample_rows=sample_rows,
        run_sample_rows=run_sample_rows,
        trial_sample_rows=trial_sample_rows,
        max_sort_columns=max_sort_columns,
        candidate_pool_size=candidate_pool_size,
        fast_candidate_count=fast_candidate_count,
        max_trial_evaluations=max_trial_evaluations,
        min_run_improvement=min_run_improvement,
        min_trial_improvement=min_trial_improvement,
        trial_compression=trial_compression,
        trial_compression_level=trial_compression_level,
        detect_json=detect_json,
        null_placement=null_placement,
        sort_backend=sort_backend,
        full=full,
    )
    planning_seconds = perf_counter() - planning_started

    sort_started = perf_counter()
    output = apply_plan(
        table,
        plan,
        null_placement=null_placement,
        sort_backend=sort_backend,
        gather_threads=gather_threads,
    )
    sort_seconds = perf_counter() - sort_started
    return OptimizationResult(
        table=output,
        plan=plan,
        planning_seconds=planning_seconds,
        sort_seconds=sort_seconds,
    )
