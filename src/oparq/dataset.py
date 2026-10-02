"""Rewrite local/cloud file trees with a fixed plan and unchanged partitions."""

from __future__ import annotations

import json
import posixpath
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.parquet as pq

from .models import DatasetRewriteResult, RewritePlan, SortPlan, WriteResult
from .sorting import unreorderable_columns
from .storage import (
    compression_metadata, copy_file, hidden_sibling, inspect_compression,
    inventory_parquet, read_parquet_source, resolve_compression, resolve_location,
    source_index_options,
)


_WRITER_OPTIONS = {
    "use_dictionary", "write_statistics", "write_page_index", "data_page_size",
    "data_page_version", "version", "dictionary_pagesize_limit", "write_batch_size",
    "column_encoding", "use_byte_stream_split", "bloom_filter_options",
}


def _preserves_writer_settings(compression: Any, compression_level: Any,
                               row_group_size: int | None, options: dict[str, Any]) -> bool:
    return (compression == "preserve" and compression_level == "preserve"
            and row_group_size is None and not _WRITER_OPTIONS.intersection(options))


def _file_manifest(manifest: dict[str, Any] | None, relative_path: str) -> dict[str, Any] | None:
    """Resolve tree-relative overrides before examining a single file."""

    if manifest is None:
        return None
    result = {key: value for key, value in manifest.items() if key != "files"}
    result.update(manifest.get("files", {}).get(relative_path, {}))
    return result


def _copy_or_skip(source: str, destination: str, source_fs: fs.FileSystem,
                  destination_fs: fs.FileSystem, plan: SortPlan, overwrite: bool) -> WriteResult:
    same = source_fs.equals(destination_fs) and source == destination
    started = perf_counter()
    if not same:
        copy_file(source, destination, source_filesystem=source_fs,
                  destination_filesystem=destination_fs, overwrite=overwrite)
    return WriteResult(
        path=destination, file_size=source_fs.get_file_info(source).size, plan=plan,
        planning_seconds=0.0, sort_seconds=0.0,
        write_seconds=0.0 if same else perf_counter() - started,
        permutation_seconds=0.0, gathering_seconds=0.0,
        action="skipped" if same else "copied",
    )


def _finish_candidate(result: WriteResult, source: str, destination: str,
                      source_fs: fs.FileSystem, destination_fs: fs.FileSystem,
                      overwrite: bool) -> WriteResult:
    """Publish a smaller candidate, otherwise retain the original bytes."""

    candidate = str(result.path)
    try:
        source_bytes = source_fs.get_file_info(source).size
        if result.file_size >= source_bytes:
            plan = replace(
                result.plan, sort_keys=(), prefix_keys=(),
                note=(f"sorted candidate was {result.file_size:,} bytes versus "
                      f"{source_bytes:,} source bytes; kept original bytes"),
            )
            started = perf_counter()
            destination_fs.delete_file(candidate)
            cleanup_seconds = perf_counter() - started
            fallback = _copy_or_skip(source, destination, source_fs,
                                     destination_fs, plan, overwrite)
            return replace(
                fallback, planning_seconds=result.planning_seconds,
                sort_seconds=result.sort_seconds,
                write_seconds=result.write_seconds + fallback.write_seconds + cleanup_seconds,
                permutation_seconds=result.permutation_seconds,
                gathering_seconds=result.gathering_seconds,
            )
        started = perf_counter()
        destination_fs.move(candidate, destination)
        return replace(result, path=destination,
                       write_seconds=result.write_seconds + perf_counter() - started)
    finally:
        if destination_fs.get_file_info(candidate).type == fs.FileType.File:
            destination_fs.delete_file(candidate)


def _write_stream(schema: pa.Schema, batches: Any, destination: str,
                  destination_fs: fs.FileSystem, plan: SortPlan, settings: Any,
                  row_group_size: int, overwrite: bool, options: dict[str, Any]) -> WriteResult:
    """Write exact row groups, charging upstream fetch work to one engine timer."""

    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    if destination_fs.get_file_info(destination).type != fs.FileType.NotFound and not overwrite:
        raise FileExistsError(destination)
    parent = posixpath.dirname(destination)
    if parent:
        destination_fs.create_dir(parent, recursive=True)
    temporary = hidden_sibling(destination, f"oparq-{uuid.uuid4().hex}.tmp")
    metadata = dict(schema.metadata or {})
    metadata[b"oparq.sort_plan"] = json.dumps(plan.as_dict(include_profiles=False)).encode()
    metadata[b"oparq.compression"] = compression_metadata(settings.compression, settings.compression_level)
    schema = schema.with_metadata(metadata)
    sorting_columns = (
        pq.SortingColumn.from_ordering(schema, plan.sort_order, null_placement=plan.null_placement)
        if plan.sorting_keys else None
    )
    writing = 0.0
    fetching = 0.0
    total_rows = 0
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    stream = None
    writer = None
    try:
        started = perf_counter()
        stream = destination_fs.open_output_stream(temporary)
        writer = pq.ParquetWriter(stream, schema, sorting_columns=sorting_columns,
                                 **settings.writer_options(), **options)
        writing += perf_counter() - started
        iterator = iter(batches)
        while True:
            started = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                fetching += perf_counter() - started
                break
            fetching += perf_counter() - started
            pending.append(batch)
            pending_rows += batch.num_rows
            total_rows += batch.num_rows
            if pending_rows >= row_group_size:
                table = pa.Table.from_batches(pending, schema=schema)
                while table.num_rows >= row_group_size:
                    started = perf_counter()
                    writer.write_table(table.slice(0, row_group_size), row_group_size=row_group_size)
                    writing += perf_counter() - started
                    table = table.slice(row_group_size)
                pending = table.to_batches()
                pending_rows = table.num_rows
        if pending:
            started = perf_counter()
            writer.write_table(pa.Table.from_batches(pending, schema=schema), row_group_size=row_group_size)
            writing += perf_counter() - started
        if total_rows != plan.total_rows:
            raise RuntimeError(f"source changed during rewrite: expected {plan.total_rows} rows, received {total_rows}")
        started = perf_counter()
        writer.close()
        writer = None
        stream.close()
        destination_fs.move(temporary, destination)
        writing += perf_counter() - started
    finally:
        if writer is not None:
            writer.close()
        if stream is not None and not stream.closed:
            stream.close()
        if destination_fs.get_file_info(temporary).type == fs.FileType.File:
            destination_fs.delete_file(temporary)
    return WriteResult(path=destination, file_size=destination_fs.get_file_info(destination).size,
                       plan=plan, planning_seconds=0.0, sort_seconds=fetching,
                       write_seconds=writing)


def rewrite_file(
    source: str | Path, destination: str | Path, *,
    plan: RewritePlan | SortPlan | None = None,
    source_filesystem: fs.FileSystem | None = None,
    destination_filesystem: fs.FileSystem | None = None,
    compression: Any = "preserve", compression_level: Any = "preserve",
    compression_manifest: dict[str, Any] | None = None,
    skip_unchanged: bool = True, overwrite: bool = False,
    engine: str = "arrow", row_group_size: int | None = None,
    temp_directory: str | Path | None = None,
    memory_limit: str = "6GB", max_temp_directory_size: str = "18GB",
    **options: Any,
) -> WriteResult:
    """Rewrite one file, or retain original bytes when sorting cannot help.

    A no-sort plan with preserved writer settings does not decode/re-encode
    the source. At a different destination it is copied; in place it is skipped.
    With preserved settings and no mandatory prefix, a sorted candidate is
    published only when it is smaller than the source. ``skip_unchanged=False``
    disables both guards; explicit writer changes and mandatory prefixes are
    honored even if the output grows.
    DuckDB supports natural fixed plans and may spill locally during sorting.
    """

    from .io import write

    if engine not in {"arrow", "duckdb"}:
        raise ValueError("engine must be 'arrow' or 'duckdb'")
    origin = resolve_location(source, source_filesystem)
    target = resolve_location(destination, destination_filesystem)
    footer = pq.read_metadata(origin.path, filesystem=origin.filesystem)
    schema = pq.read_schema(origin.path, filesystem=origin.filesystem)
    empty = pa.Table.from_batches([], schema=schema)
    bound = plan.for_table(empty) if isinstance(plan, RewritePlan) else plan
    if bound is not None:
        bound = replace(bound, total_rows=footer.num_rows)
    blocked = unreorderable_columns(schema) if engine == "arrow" and bound and bound.sort_keys else []
    if blocked and bound.prefix_keys:
        raise ValueError(f"cannot sort by the required prefix: Arrow cannot reorder {blocked}")
    if blocked:
        # Rows move as a unit, so the Arrow engine keeps this file's order.
        bound = replace(bound, sort_keys=(), note=f"input order kept: Arrow cannot reorder {blocked}")
    unchanged_settings = _preserves_writer_settings(compression, compression_level,
                                                    row_group_size, options)
    if bound is not None and not bound.sort_keys and skip_unchanged and unchanged_settings:
        return _copy_or_skip(origin.path, target.path, origin.filesystem, target.filesystem, bound, overwrite)
    target_info = target.filesystem.get_file_info(target.path)
    if bound is not None and target_info.type != fs.FileType.NotFound:
        if target_info.type != fs.FileType.File or not overwrite:
            raise FileExistsError(f"output already exists: {destination}")
    settings = resolve_compression(
        inspect_compression(origin.path, origin.filesystem, manifest=compression_manifest),
        compression=compression, compression_level=compression_level,
    )
    # Rebuild the source's page index and bloom filters for the new row order
    # unless the caller chose these writer options explicitly.
    group_rows = min(row_group_size or 1_048_576, max(1, footer.num_rows))
    index_options = {key: value for key, value in
                     source_index_options([footer], output_group_rows=group_rows).items()
                     if key not in options}
    if engine == "arrow":
        table = read_parquet_source(origin.path, origin.filesystem)
        planning_seconds = 0.0
        if bound is None:
            from .planning import plan_sort
            plan_options = {key: value for key, value in options.items() if key in {
                "algorithm", "prefix", "include", "exclude", "sample_rows", "run_sample_rows",
                "trial_sample_rows", "max_sort_columns", "candidate_pool_size", "fast_candidate_count",
                "max_trial_evaluations", "min_run_improvement", "min_trial_improvement", "detect_json",
                "null_placement", "sort_backend", "full",
            }}
            started = perf_counter()
            bound = plan_sort(table, trial_compression=settings.compression,
                              trial_compression_level=settings.compression_level, **plan_options)
            planning_seconds = perf_counter() - started
            if not bound.sort_keys and skip_unchanged and unchanged_settings:
                return replace(_copy_or_skip(origin.path, target.path, origin.filesystem,
                                              target.filesystem, bound, overwrite),
                               planning_seconds=planning_seconds)
        target_info = target.filesystem.get_file_info(target.path)
        if target_info.type != fs.FileType.NotFound:
            if target_info.type != fs.FileType.File or not overwrite:
                raise FileExistsError(f"output already exists: {destination}")
        guarded = skip_unchanged and unchanged_settings and not bound.prefix_keys
        output = (hidden_sibling(target.path, f"oparq-candidate-{uuid.uuid4().hex}.parquet")
                  if guarded else target.path)
        try:
            result = write(table, output, filesystem=target.filesystem, plan=bound,
                           overwrite=overwrite if not guarded else False,
                           row_group_size=row_group_size,
                           **settings.writer_options(), **index_options, **options)
            result = replace(result, planning_seconds=result.planning_seconds + planning_seconds)
            return (_finish_candidate(result, origin.path, target.path, origin.filesystem,
                                       target.filesystem, overwrite) if guarded else result)
        finally:
            if guarded and target.filesystem.get_file_info(output).type == fs.FileType.File:
                target.filesystem.delete_file(output)
    if bound is None:
        raise ValueError("DuckDB rewrites require a learned plan; call fit_dataset first")
    if bound.value_order != "natural":
        raise ValueError("DuckDB currently supports natural-order learned plans; use engine='arrow' for frequency")
    from .engines import duckdb_sorted_batches

    dataset = ds.dataset([origin.path], filesystem=origin.filesystem, format="parquet")
    scanner = dataset.scanner(batch_size=262_144)
    writer_options = {**index_options,
                      **{key: value for key, value in options.items() if key in _WRITER_OPTIONS}}
    guarded = skip_unchanged and unchanged_settings and not bound.prefix_keys
    output = (hidden_sibling(target.path, f"oparq-candidate-{uuid.uuid4().hex}.parquet")
              if guarded else target.path)
    try:
        with tempfile.TemporaryDirectory(prefix="oparq-spill-", dir=temp_directory) as spill:
            started = perf_counter()
            with duckdb_sorted_batches(schema, scanner.to_batches(), bound.sort_keys,
                                       temp_directory=spill, memory_limit=memory_limit,
                                       max_temp_directory_size=max_temp_directory_size,
                                       null_placement=bound.null_placement) as batches:
                initial_sort = perf_counter() - started
                result = _write_stream(schema, batches, output, target.filesystem, bound,
                                       settings, row_group_size or 1_048_576,
                                       overwrite if not guarded else False, writer_options)
            result = replace(result, sort_seconds=result.sort_seconds + initial_sort)
            return (_finish_candidate(result, origin.path, target.path, origin.filesystem,
                                       target.filesystem, overwrite) if guarded else result)
    finally:
        if guarded and target.filesystem.get_file_info(output).type == fs.FileType.File:
            target.filesystem.delete_file(output)


def rewrite_dataset(
    source: str | Path, destination: str | Path, *, plan: RewritePlan,
    source_filesystem: fs.FileSystem | None = None,
    destination_filesystem: fs.FileSystem | None = None,
    **options: Any,
) -> DatasetRewriteResult:
    """Reuse one plan per input file, retaining every relative partition path.

    Sorting is global within each physical file. File boundaries and Hive
    directories are preserved; this does not coalesce an entire partition.
    Dataset-level aggregate metadata files are not copied with stale offsets.
    """

    inventory = inventory_parquet(source, source_filesystem)
    target = resolve_location(destination, destination_filesystem)
    if not isinstance(plan, RewritePlan):
        raise TypeError("rewrite_dataset requires a reusable RewritePlan")
    engine = options.get("engine", "arrow")
    if engine not in {"arrow", "duckdb"}:
        raise ValueError("engine must be 'arrow' or 'duckdb'")
    if engine == "duckdb" and plan.value_order != "natural":
        raise ValueError("DuckDB currently supports natural-order learned plans; use engine='arrow' for frequency")
    if inventory.filesystem.equals(target.filesystem):
        source_root = inventory.root.rstrip("/")
        destination_root = target.path.rstrip("/")
        if destination_root.startswith(source_root + "/"):
            raise ValueError("destination must not be inside the source tree")
    compression = options.get("compression", "preserve")
    compression_level = options.get("compression_level", "preserve")
    preserved = (options.get("skip_unchanged", True)
                 and _preserves_writer_settings(compression, compression_level,
                                                options.get("row_group_size"), options))
    manifest = options.get("compression_manifest")
    overwrite = options.get("overwrite", False)
    prepared = []
    source_paths = set(inventory.files)
    same_filesystem = inventory.filesystem.equals(target.filesystem)
    root_info = target.filesystem.get_file_info(target.path)
    if root_info.type not in {fs.FileType.Directory, fs.FileType.NotFound}:
        raise FileExistsError(f"dataset destination is not a directory: {destination}")
    # Validate all schemas, compression provenance and destination paths before
    # publishing anything. A no-sort byte copy does not need encoder provenance.
    for file in inventory.files:
        schema = pq.read_schema(file, filesystem=inventory.filesystem)
        plan.for_table(pa.Table.from_batches([], schema=schema))
        blocked = unreorderable_columns(schema) if engine == "arrow" and plan.sort_keys else []
        if blocked and plan.prefix_keys:
            raise ValueError(f"cannot sort {file} by the required prefix: Arrow cannot reorder {blocked}")
        # A no-key plan, or keys the Arrow engine cannot apply, keeps original bytes.
        copies_original = preserved and (not plan.sort_keys or bool(blocked))
        if engine == "duckdb" and plan.sort_keys:
            from .engines import validate_duckdb_schema
            validate_duckdb_schema(schema)
        relative = inventory.relative_path(file)
        output = posixpath.join(target.path, relative)
        same = same_filesystem and file == output
        if same_filesystem and output in source_paths and not same:
            raise ValueError(f"destination would overwrite another source file: {output}")
        output_info = target.filesystem.get_file_info(output)
        if output_info.type != fs.FileType.NotFound:
            if output_info.type != fs.FileType.File or (not overwrite and not (same and copies_original)):
                raise FileExistsError(f"output already exists: {output}")
        parent = posixpath.dirname(output)
        while parent:
            info = target.filesystem.get_file_info(parent)
            if info.type not in {fs.FileType.Directory, fs.FileType.NotFound}:
                raise FileExistsError(f"output parent is not a directory: {parent}")
            next_parent = posixpath.dirname(parent)
            if next_parent == parent:
                break
            parent = next_parent
        file_manifest = _file_manifest(manifest, relative)
        if not copies_original:
            resolve_compression(
                inspect_compression(file, inventory.filesystem, manifest=file_manifest),
                compression=compression, compression_level=compression_level,
            )
        prepared.append((file, output, file_manifest))
    results = []
    for file, output, file_manifest in prepared:
        file_options = {**options, "compression_manifest": file_manifest}
        results.append(rewrite_file(file, output, plan=plan,
                                   source_filesystem=inventory.filesystem,
                                   destination_filesystem=target.filesystem, **file_options))
    return DatasetRewriteResult(str(source), str(destination), tuple(results))
