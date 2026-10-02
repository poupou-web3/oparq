"""Parquet readers and writers with truthful sorting metadata."""

from __future__ import annotations

import json
import os
import posixpath
import uuid
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

from .core import plan_permutation
from .models import RewritePlan, SortPlan, WriteResult
from .planning import plan_sort
from .profile import DEFAULT_SAMPLE_ROWS, as_table
from .sorting import ChunkedTableGather, take_table, unreorderable_columns, use_chunked_gather

_UNSET = object()


def read_parquet(source: str | Path, *, filesystem: fs.FileSystem | None = None) -> pa.Table:
    """Read local/S3/GCS Parquet through Arrow without a local staging file."""

    from .storage import read_parquet_source
    return read_parquet_source(source, filesystem)


def _metadata(table: pa.Table, plan_payload: dict[str, Any]) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    metadata[b"oparq.sort_plan"] = json.dumps(
        plan_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return table.replace_schema_metadata(metadata)


def _output_file(destination: str | Path) -> Path | str:
    value = str(destination)
    if "://" in value:
        return value if value.lower().endswith(".parquet") else value.rstrip("/") + "/part-00000.parquet"
    path = Path(destination)
    return path if path.suffix.lower() == ".parquet" else path / "part-00000.parquet"


def write(
    value: pa.Table | pa.RecordBatch,
    destination: str | Path,
    *,
    algorithm: str = "auto",
    plan: RewritePlan | SortPlan | None = None,
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
    detect_json: bool = True,
    null_placement: str = "at_end",
    sort_backend: str = "auto",
    full: bool = False,
    gather_threads: int = 0,
    stream_sort: bool = True,
    compression: Any = _UNSET,
    compression_level: Any = _UNSET,
    row_group_size: int | None = None,
    use_dictionary: bool | list[str] = True,
    write_statistics: bool | list[str] = True,
    write_page_index: bool = False,
    overwrite: bool = False,
    filesystem: fs.FileSystem | None = None,
    **parquet_options: Any,
) -> WriteResult:
    """Optimize and write an in-memory Arrow table.

    Local destinations are replaced atomically. Cloud publication uses a
    temporary object followed by a move, which is not an atomic transaction.

    A new in-memory table uses Zstandard's Arrow codec default. A reusable
    ``plan`` skips profiling and key selection; its codec settings are used
    unless the caller explicitly overrides them. File rewrites preserve source
    settings through ``rewrite``. Statistics remain enabled for pruning.
    """

    output = _output_file(destination)
    from .storage import hidden_sibling, resolve_location
    location = resolve_location(output, filesystem)
    if location.filesystem.get_file_info(location.path).type != fs.FileType.NotFound and not overwrite:
        raise FileExistsError(f"output already exists: {output}")

    table = as_table(value)
    reused_plan = plan is not None
    learned_plan = plan if isinstance(plan, RewritePlan) else None
    if compression is _UNSET:
        compression = learned_plan.compression if learned_plan else "zstd"
    if compression_level is _UNSET:
        compression_level = learned_plan.compression_level if learned_plan else None
    if gather_threads < 0:
        raise ValueError("gather_threads must be non-negative")
    if row_group_size is not None and row_group_size <= 0:
        raise ValueError("row_group_size must be positive or None")
    # Match PyArrow's default and maximum Parquet row-group geometry.
    group_rows = min(row_group_size or 1_048_576, 64 * 1_048_576)
    planning_started = perf_counter()
    if plan is None:
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
            trial_compression=compression,
            trial_compression_level=compression_level,
            detect_json=detect_json,
            null_placement=null_placement,
            sort_backend=sort_backend,
            full=full,
        )
    else:
        if prefix and tuple(prefix) != plan.prefix_keys:
            raise ValueError("prefix conflicts with the supplied plan")
        plan = plan.for_table(table) if isinstance(plan, RewritePlan) else plan
        blocked = unreorderable_columns(table.schema) if plan.sort_keys else []
        if blocked and plan.prefix_keys:
            raise ValueError(f"cannot sort by the required prefix: Arrow cannot reorder {blocked}")
        if blocked:
            # Rows move as a unit; keep input order and advertise no sorting.
            plan = replace(plan, sort_keys=(), note=f"input order kept: Arrow cannot reorder {blocked}")
    planning_seconds = perf_counter() - planning_started
    if reused_plan:
        planning_seconds = 0.0
    sort_started = perf_counter()
    permutation = plan_permutation(table, plan, sort_backend=sort_backend)
    permutation_seconds = perf_counter() - sort_started
    gathering_seconds = 0.0
    parent = posixpath.dirname(location.path)
    if parent:
        location.filesystem.create_dir(parent, recursive=True)

    compact_plan = plan.as_dict(include_profiles=False)
    annotated_table = _metadata(table, compact_plan)
    sorting_columns = None
    if plan.sorting_keys:
        sorting_columns = pq.SortingColumn.from_ordering(
            annotated_table.schema,
            plan.sort_order,
            null_placement=plan.null_placement,
        )

    # A hidden sibling keeps readers from treating a partial file as data.
    # Arrow creates it with the process umask (mkstemp would force 0600).
    temporary_path = hidden_sibling(location.path, f"oparq-{uuid.uuid4().hex}.tmp")
    if location.is_local:
        temporary: Any = Path(temporary_path)
        stream = None
    else:
        stream = location.filesystem.open_output_stream(temporary_path)
        temporary = stream
    write_seconds = 0.0
    try:
        from .storage import compression_metadata, effective_compression_level

        effective_level = effective_compression_level(compression, compression_level)
        recorded_metadata = dict(annotated_table.schema.metadata or {})
        recorded_metadata[b"oparq.compression"] = compression_metadata(compression, effective_level)
        annotated_table = annotated_table.replace_schema_metadata(recorded_metadata)
        writer_options = dict(
            compression=compression,
            compression_level=effective_level,
            use_dictionary=use_dictionary,
            write_statistics=write_statistics,
            write_page_index=write_page_index,
            sorting_columns=sorting_columns,
            **parquet_options,
        )
        if stream_sort and permutation is not None and table.num_rows > group_rows:
            # Gather only the next sorted row group. Keeping the input plus a
            # second full reordered table can trigger paging even when the
            # input alone comfortably fits in memory.
            started = perf_counter()
            chunked_gather = (
                ChunkedTableGather(annotated_table, gather_threads=gather_threads)
                if use_chunked_gather(annotated_table, group_rows)
                else None
            )
            gathering_seconds += perf_counter() - started
            started = perf_counter()
            writer = pq.ParquetWriter(temporary, annotated_table.schema, **writer_options)
            write_seconds += perf_counter() - started
            try:
                for offset in range(0, table.num_rows, group_rows):
                    started = perf_counter()
                    group_permutation = permutation.slice(offset, group_rows)
                    group = (
                        chunked_gather.take(group_permutation)
                        if chunked_gather is not None
                        else take_table(
                            annotated_table,
                            group_permutation,
                            gather_threads=gather_threads,
                        )
                    )
                    gathering_seconds += perf_counter() - started
                    started = perf_counter()
                    writer.write_table(group, row_group_size=group_rows)
                    write_seconds += perf_counter() - started
                    del group
            finally:
                started = perf_counter()
                writer.close()
                write_seconds += perf_counter() - started
        else:
            started = perf_counter()
            ordered = (
                take_table(annotated_table, permutation, gather_threads=gather_threads)
                if permutation is not None else annotated_table
            )
            gathering_seconds += perf_counter() - started
            started = perf_counter()
            pq.write_table(ordered, temporary, row_group_size=row_group_size, **writer_options)
            write_seconds += perf_counter() - started
        started = perf_counter()
        if stream is not None:
            stream.close()
        if location.is_local:
            os.replace(temporary_path, location.path)
        else:
            location.filesystem.move(temporary_path, location.path)
        write_seconds += perf_counter() - started
    finally:
        if stream is not None and not stream.closed:
            stream.close()
        if location.filesystem.get_file_info(temporary_path).type == fs.FileType.File:
            location.filesystem.delete_file(temporary_path)
    return WriteResult(
        path=output,
        file_size=location.filesystem.get_file_info(location.path).size,
        plan=plan,
        planning_seconds=planning_seconds,
        sort_seconds=permutation_seconds + gathering_seconds,
        write_seconds=write_seconds,
        permutation_seconds=permutation_seconds,
        gathering_seconds=gathering_seconds,
    )


def rewrite(
    source: str | Path,
    destination: str | Path,
    **options: Any,
) -> Any:
    """Rewrite files while preserving known source codec/level by default.

    Use ``rewrite_dataset`` with a learned plan to retain a partitioned tree.
    A directory plus an explicit .parquet output still consolidates its rows.
    """

    from .dataset import rewrite_file, rewrite_dataset
    from .storage import (
        inspect_compression, inventory_parquet, resolve_compression, source_index_options,
    )

    source_filesystem = options.pop("source_filesystem", options.pop("filesystem", None))
    destination_filesystem = options.pop("destination_filesystem", None)
    inventory = inventory_parquet(source, source_filesystem)
    if not inventory.is_directory:
        return rewrite_file(source, _output_file(destination), source_filesystem=source_filesystem,
                            destination_filesystem=destination_filesystem, **options)
    if not str(destination).lower().endswith(".parquet"):
        if not isinstance(options.get("plan"), RewritePlan):
            raise ValueError("a partition-preserving rewrite requires a reusable plan; call fit_dataset first")
        return rewrite_dataset(source, destination, source_filesystem=source_filesystem,
                               destination_filesystem=destination_filesystem, **options)
    # Explicit filenames do not inject Hive columns, so consolidating would
    # discard each row's directory-encoded partition value.
    physical = set(pq.read_schema(inventory.files[0], filesystem=inventory.filesystem).names)
    dropped = sorted({
        part.split("=", 1)[0]
        for path in inventory.files
        for part in inventory.relative_path(path).split("/")[:-1]
        if "=" in part and part.split("=", 1)[0] not in physical
    })
    if dropped:
        raise ValueError(
            "consolidating would drop Hive partition values that are not physical "
            f"columns: {dropped}; use rewrite_dataset to keep the directory layout"
        )
    compression = options.pop("compression", "preserve")
    level = options.pop("compression_level", "preserve")
    manifest = options.pop("compression_manifest", None)
    settings = resolve_compression(inspect_compression(inventory, manifest=manifest),
                                   compression=compression, compression_level=level)
    options.pop("skip_unchanged", None)
    if options.pop("engine", "arrow") != "arrow":
        raise ValueError("use rewrite_dataset with a learned plan for DuckDB partition rewrites")
    for name in ("memory_limit", "max_temp_directory_size", "temp_directory"):
        options.pop(name, None)
    footers = [pq.read_metadata(path, filesystem=inventory.filesystem) for path in inventory.files]
    group_rows = min(options.get("row_group_size") or 1_048_576,
                     max(1, sum(footer.num_rows for footer in footers)))
    index_options = {key: value for key, value in
                     source_index_options(footers, output_group_rows=group_rows).items()
                     if key not in options}
    return write(read_parquet(source, filesystem=source_filesystem), destination,
                 filesystem=destination_filesystem, **settings.writer_options(),
                 **index_options, **options)
