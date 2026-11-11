"""Partition-scope rewrites: sort across a directory's files, write even files.

``rewrite_dataset(..., scope="partition")`` treats each leaf directory as one
unit. Its rows are sorted together (ties keep file order, then row order) and
written as ``part-00000.parquet``... files of roughly ``target_file_size``. Hive directories are kept, so readers see the same
partitions with fewer, better-compressed files.

Publication is a set of files, so it cannot be atomic on an object store. The
source and destination trees must therefore not overlap: compaction always
writes a new tree. A checkpoint marker is written twice per partition: a
``publishing`` marker naming its outputs before they are moved into place,
and a ``done`` marker afterwards. A rerun deletes the outputs of an
interrupted publication before redoing that partition.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import posixpath
import re
import tempfile
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.parquet as pq

from .execution import TASK_OVERHEAD_BYTES, Prefetcher, Task, in_shard, parse_size, run_tasks
from .models import DatasetRewriteResult, PartitionResult, RewritePlan, SortPlan, WriteResult
from .sorting import ChunkedTableGather, take_table, unreorderable_columns, use_chunked_gather
from .storage import (
    ParquetInventory, compression_metadata, hidden_sibling, inspect_compression, read_footers,
    resolve_compression, source_index_options,
)

OUTPUT_NAME = "part-{index:05d}.parquet"
DEFAULT_TARGET_FILE_SIZE = 512 * 1024 * 1024
# A partition is split only when its estimated output exceeds the target by
# this factor, so files stay below about 1.25x the target (unless a single
# row group is larger) instead of producing an extra small file.
TARGET_TOLERANCE = 1.25
# Row groups shrink only when one, at this margin over the densest recent
# group, could not fit under max_file_size.
RECENT_GROUPS = 4
PREDICTION_MARGIN = 1.1
# Cap decisions: a group fits without measurement if its Arrow size times
# RAW_BOUND fits; otherwise its density is probed on PROBE_FRACTION-th slices
# (at least PROBE_MIN_ROWS rows) of both ends. Within a factor UNCERTAINTY of
# that estimate either way, the group is encoded alone to measure it exactly.
RAW_BOUND = 1.1
PROBE_FRACTION = 64
PROBE_MIN_ROWS = 512
UNCERTAINTY = 1.5
# Row groups are not shrunk or split below this (or the configured size).
MIN_GROUP_ROWS = 1024
# Share of the cap that estimate-based and measured decisions may fill. The
# measured share leaves room for page-index statistics, which grow with the
# number of pages and are written at close.
ESTIMATE_HEADROOM = 0.97
MEASURED_HEADROOM = 0.99
# Bytes a Parquet bloom filter header adds to its bitset (measured: 26).
BLOOM_HEADER_BYTES = 64
_TEMPORARY = re.compile(r"^\..+\.oparq-(?:candidate-)?[0-9a-f]{32}\.(?:tmp|parquet)$")


def _is_data_file(name: str) -> bool:
    return name.lower().endswith(".parquet") and not name.startswith((".", "_"))


@dataclass(slots=True)
class _Partition:
    relative: str
    key: str
    files: list[str]
    relatives: list[str]
    dest_dir: str
    identity: list[list[Any]]
    fingerprint: str
    existing: list[str] = field(default_factory=list)
    temporaries: list[str] = field(default_factory=list)
    replaceable: set[str] = field(default_factory=set)
    footers: list[Any] = field(default_factory=list)
    schema: pa.Schema | None = None
    bound: SortPlan | None = None
    settings: Any = None
    copies_original: bool = False
    streaming: bool = False

    @property
    def rows(self) -> int:
        return sum(footer.num_rows for footer in self.footers)

    @property
    def source_bytes(self) -> int:
        return sum(item[1] or 0 for item in self.identity)


class _RollingWriter:
    """Write a sorted stream into consecutive files of about ``target_bytes``,
    each at most ``max_bytes``.

    Target: after every row group, the bytes written so far give compressed
    bytes per row; with the rows not yet in a closed file this estimates the
    remaining output, how many files it should form, and each file's share.
    A file closes at the row-group boundary nearest that share.

    Cap: before a row group is written, the file's size with it is projected.
    Close-time bytes are the footer and row-group metadata, measured once by
    encoding one row as one and as two row groups, plus bloom filters, which
    Parquet also writes at close. Arrow sizes a filter by the distinct values
    it holds, at most ``min(ndv, rows)``, so each row group reserves the
    filter size for that bound (``_bloom_bound``), which cannot be exceeded.
    The group's own data size is bounded by its uncompressed Arrow size, or else
    estimated by encoding small slices from both of its ends. If the
    estimate leaves the decision ambiguous, the group is encoded alone to
    measure it exactly. The file closes first when the group does not fit,
    and a group too large for any file is split (``reduced_group_rows``).
    The estimate is the recent groups' measured density, or the end slices'
    density discounted by ``UNCERTAINTY`` when that is higher (slices are
    noisy, so they act as a jump detector). Estimated decisions fill at most
    97% of the cap, measured ones 99% (page-index statistics grow with the
    data and are written at close). A file can exceed the cap only if a row
    group is more than 1.5 times denser than both the recent groups and its
    end slices, if a row group's pages exceed ``RAW_BOUND`` times its Arrow
    size (the shortcut taken far below the cap), or if a row group of
    ``MIN_GROUP_ROWS`` rows (or the configured size, if smaller) is larger
    than the cap.
    Files are hidden temporaries until the caller publishes them.
    """

    def __init__(self, filesystem: fs.FileSystem, directory: str, schema: pa.Schema,
                 writer_options: dict[str, Any], *, target_bytes: int, total_rows: int,
                 group_rows: int, max_bytes: int | None = None,
                 prior_bytes_per_row: float | None = None) -> None:
        self.filesystem = filesystem
        self.directory = directory
        self.schema = schema
        self.writer_options = writer_options
        self.target_bytes = target_bytes
        self.max_bytes = max_bytes or int(TARGET_TOLERANCE * target_bytes)
        if self.max_bytes < target_bytes:
            raise ValueError("max_file_size must not be smaller than target_file_size")
        self.total_rows = total_rows
        self.group_rows = group_rows
        self.prior_bytes_per_row = prior_bytes_per_row
        self.reduced_group_rows: int | None = None
        # Row groups are never shrunk or split below this many rows.
        self._min_group_rows = max(1, min(group_rows, MIN_GROUP_ROWS))
        self.measured_groups = 0  # Groups encoded alone to measure them exactly.
        self.files: list[tuple[str, str, int]] = []
        self.written_rows = 0
        self.write_seconds = 0.0
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self._writer: pq.ParquetWriter | None = None
        self._stream: Any = None
        self._current: tuple[str, str] | None = None
        self._current_rows = 0
        self._current_groups = 0
        self._closed_rows = 0
        self._closed_bytes = 0
        self._recent: list[float] = []  # Data bytes per row of the latest row groups.
        self._per_group: int | None = None  # Close-time bytes added by each row group.
        self._base = 0  # Bytes of a file holding no row group.
        # Pages are measured without bloom filters; those are reserved apart.
        self._data_options = {key: value for key, value in writer_options.items()
                              if key != "bloom_filter_options"}
        self._blooms = [(int(item.get("ndv", 0)), float(item.get("fpp", 0.05)))
                        for item in (writer_options.get("bloom_filter_options") or {}).values()]
        self._current_bloom = 0  # Bloom bytes reserved for the open file's groups.

    # -- measurement -------------------------------------------------------

    def _bloom_bound(self, rows: int) -> int:
        """Largest bloom filter bytes a row group of ``rows`` rows can carry."""

        total = 0
        for ndv, fpp in self._blooms:
            distinct = max(1, min(ndv or rows, rows))
            bits = -8.0 * distinct / math.log(1 - fpp ** (1 / 8))
            bits = int(min(max(bits, 8 * 32), 8 * (128 << 20)))
            if bits & (bits - 1):
                bits = 1 << bits.bit_length()
            total += bits // 8 + BLOOM_HEADER_BYTES
        return total

    def _encode(self, table: pa.Table, groups: int = 1) -> int:
        sink = pa.BufferOutputStream()
        started = perf_counter()
        pq.write_table(table, sink, row_group_size=max(1, table.num_rows // groups), **self._data_options)
        self.write_seconds += perf_counter() - started
        return sink.getvalue().size

    def _measure_overheads(self, sample: pa.Table) -> None:
        row = sample.slice(0, 1)
        one = self._encode(row)
        two = self._encode(pa.concat_tables([row, row]), groups=2)
        self._per_group = max(0, two - one)
        self._base = max(0, one - self._per_group)

    def _data_size(self, table: pa.Table) -> float:
        """Bytes the rows' pages add to a file (overheads excluded)."""

        return max(0.0, self._encode(table) - self._base - self._per_group)

    def _probe(self, group: pa.Table) -> float:
        """Bytes per row estimated from both ends of ``group``."""

        rows = max(PROBE_MIN_ROWS, group.num_rows // PROBE_FRACTION)
        if 2 * rows >= group.num_rows:
            return self._data_size(group) / group.num_rows
        head = self._data_size(group.slice(0, rows)) / rows
        tail = self._data_size(group.slice(group.num_rows - rows)) / rows
        return max(head, tail)

    def _projected(self, data_bytes: float, rows: int) -> float:
        """Size of the current file, if closed after adding a row group of
        ``rows`` rows whose pages take ``data_bytes``."""

        written = self._stream.tell() if self._writer is not None else 0
        return (written + data_bytes + (self._current_groups + 1) * self._per_group + self._base
                + self._current_bloom + self._bloom_bound(rows))

    # -- writing -----------------------------------------------------------

    def _densest(self) -> float | None:
        return max(self._recent) if self._recent else self.prior_bytes_per_row

    def _next_group_rows(self) -> int:
        """Configured row-group size, reduced only if one group could cross the cap."""

        rows = self.group_rows
        densest = self._densest()
        if densest:
            # Sized so that, with the cap's uncertainty margin and the group's
            # bloom filter reserve, it fits an empty file without being
            # encoded twice.
            per_row = (densest * PREDICTION_MARGIN * UNCERTAINTY
                       + self._bloom_bound(self.group_rows) / self.group_rows)
            room = self.max_bytes * ESTIMATE_HEADROOM - self._base - (self._per_group or 0)
            fitting = max(self._min_group_rows, int(room / per_row))
            if fitting < rows:
                rows = fitting
                self.reduced_group_rows = min(self.reduced_group_rows or rows, rows)
        return rows

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self._pending.append(table)
        self._pending_rows += table.num_rows
        while self._pending_rows >= (rows := self._next_group_rows()):
            self._flush(rows)

    def _take(self, rows: int) -> pa.Table:
        table = pa.concat_tables(self._pending) if len(self._pending) > 1 else self._pending[0]
        head, tail = table.slice(0, rows), table.slice(rows)
        self._pending = [tail] if tail.num_rows else []
        self._pending_rows = tail.num_rows
        return head

    def _open(self) -> None:
        final = posixpath.join(self.directory, OUTPUT_NAME.format(index=len(self.files)))
        temporary = hidden_sibling(final, f"oparq-{uuid.uuid4().hex}.tmp")
        started = perf_counter()
        self.filesystem.create_dir(self.directory, recursive=True)
        self._stream = self.filesystem.open_output_stream(temporary)
        self._writer = pq.ParquetWriter(self._stream, self.schema, **self.writer_options)
        self.write_seconds += perf_counter() - started
        self._current = (temporary, final)
        self._current_rows = 0
        self._current_groups = 0
        self._current_bloom = 0

    def _close_file(self) -> None:
        started = perf_counter()
        self._writer.close()
        self._stream.close()
        temporary, final = self._current
        size = self.filesystem.get_file_info(temporary).size
        self.write_seconds += perf_counter() - started
        self.files.append((temporary, final, self._current_rows))
        self._closed_rows += self._current_rows
        self._closed_bytes += size
        self._writer = self._stream = self._current = None
        self._current_rows = self._current_groups = self._current_bloom = 0

    def _fit(self, group: pa.Table) -> pa.Table:
        """Close the file or split ``group`` until it fits under the cap."""

        exact: float | None = None
        # Each pass either decides, closes the open file (at most once per
        # group, since the new file is empty), or halves the group.
        while True:
            empty_file = self._writer is None or not self._current_rows
            rows = group.num_rows
            if exact is None:
                if self._projected(group.nbytes * RAW_BOUND, rows) <= self.max_bytes:
                    return group  # Fits even uncompressed: nothing to measure.
                # Estimated decisions keep extra headroom below the cap.
                budget = self.max_bytes * ESTIMATE_HEADROOM
                # End slices are noisy, so they only override the recent
                # groups' measured density when they show a real jump.
                estimate = max(self._densest() or 0.0, self._probe(group) / UNCERTAINTY) * rows
                if self._projected(UNCERTAINTY * estimate, rows) <= budget:
                    return group
                if not empty_file and self._projected(estimate / UNCERTAINTY, rows) > self.max_bytes:
                    self._close_file()  # Clearly does not fit: no need to measure.
                    continue
                exact = self._data_size(group)
                self.measured_groups += 1
            if self._projected(exact, rows) <= self.max_bytes * MEASURED_HEADROOM:
                return group
            if not empty_file:
                self._close_file()
                continue
            if group.num_rows <= self._min_group_rows:
                return group  # A minimal row group larger than the cap is written alone.
            half = max(self._min_group_rows, group.num_rows // 2)
            self._pending.insert(0, group.slice(half))
            self._pending_rows += group.num_rows - half
            group = group.slice(0, half)
            self.reduced_group_rows = min(self.reduced_group_rows or half, half)
            exact = None

    def _flush(self, rows: int) -> None:
        group = self._take(rows)
        if self._per_group is None:
            self._measure_overheads(group)
        group = self._fit(group)
        if self._writer is None:
            self._open()
        before = self._stream.tell()
        started = perf_counter()
        self._writer.write_table(group, row_group_size=group.num_rows)
        self.write_seconds += perf_counter() - started
        self._current_rows += group.num_rows
        self._current_groups += 1
        self._current_bloom += self._bloom_bound(group.num_rows)
        self.written_rows += group.num_rows
        current_bytes = self._stream.tell()
        self._recent = (self._recent + [(current_bytes - before) / group.num_rows])[-RECENT_GROUPS:]
        unwritten = self.total_rows - self.written_rows
        if not unwritten:
            return
        # Target: close at the boundary nearest this file's share.
        next_rows = min(self._next_group_rows(), unwritten)
        per_row = (self._closed_bytes + current_bytes) / (self._closed_rows + self._current_rows)
        remaining_bytes = per_row * (self.total_rows - self._closed_rows)
        files_left = max(1, math.ceil(remaining_bytes / (TARGET_TOLERANCE * self.target_bytes)))
        if files_left > 1 and current_bytes + per_row * next_rows / 2 >= remaining_bytes / files_left:
            self._close_file()

    def finish(self) -> list[tuple[str, str, int]]:
        while self._pending_rows:
            self._flush(min(self._pending_rows, self._next_group_rows()))
        if self._writer is None and not self.files:
            self._open()  # An empty partition still has a schema.
        if self._writer is not None:
            self._close_file()
        return self.files

    def abort(self) -> None:
        if self._writer is not None:
            with contextlib.suppress(Exception):
                self._writer.close()
        if self._stream is not None and not self._stream.closed:
            with contextlib.suppress(Exception):
                self._stream.close()
        temporaries = [item[0] for item in self.files]
        if self._current is not None:
            temporaries.append(self._current[0])
        for path in temporaries:
            if self.filesystem.get_file_info(path).type == fs.FileType.File:
                self.filesystem.delete_file(path)


def _source_batches(path: str, filesystem: fs.FileSystem, buffer: pa.Buffer | None,
                    batch_size: int = 262_144) -> Iterator[pa.RecordBatch]:
    if buffer is not None:
        yield from pq.ParquetFile(pa.BufferReader(buffer)).iter_batches(batch_size=batch_size)
        return
    dataset = ds.dataset([path], filesystem=filesystem, format="parquet")
    yield from dataset.scanner(batch_size=batch_size).to_batches()


def _partition_fingerprint(base: str, target_bytes: int, max_bytes: int) -> str:
    payload = {"base": base, "scope": "partition", "target_bytes": target_bytes,
               "max_bytes": max_bytes, "name": OUTPUT_NAME}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def rewrite_partitions(
    *, source: Any, destination: Any, inventory: ParquetInventory, target: Any, plan: RewritePlan,
    options: dict[str, Any], engine: str, workers: int, budget: int | None,
    shard: tuple[int, int] | None, store: Any, listing: list[fs.FileInfo],
    existing: dict[str, fs.FileInfo], preserved: bool, overwrite: bool,
    progress: Callable[[PartitionResult, int, int], None] | None,
    io_threads: int | None, prefetch: int | None, target_bytes: int, max_bytes: int,
    run_fingerprint: Callable[..., str], file_manifest: Callable[..., Any],
    source_identity: Callable[[fs.FileInfo], dict[str, Any]],
) -> DatasetRewriteResult:
    """Implementation of ``rewrite_dataset(..., scope="partition")``."""

    from .dataset import _WRITER_OPTIONS, _copy_or_skip, _preserves_writer_settings

    source_fs, target_fs = inventory.filesystem, target.filesystem
    if source_fs.equals(target_fs):
        source_root = inventory.root.rstrip("/")
        destination_root = target.path.rstrip("/")
        if (destination_root == source_root or source_root.startswith(destination_root + "/")
                or not destination_root):
            raise ValueError("scope='partition' writes a new tree: the destination must not "
                             "be the source tree or contain it")
    compression = options.get("compression", "preserve")
    compression_level = options.get("compression_level", "preserve")
    manifest = options.get("compression_manifest")
    unchanged_settings = _preserves_writer_settings(compression, compression_level,
                                                    options.get("row_group_size"), options)
    guard = options.get("skip_unchanged", True) and unchanged_settings

    groups: dict[str, list[str]] = {}
    for file in inventory.files:
        relative_dir = posixpath.dirname(inventory.relative_path(file))
        groups.setdefault(relative_dir, []).append(file)
    partitions: list[_Partition] = []
    by_dir: dict[str, list[fs.FileInfo]] = {}
    for info in listing:
        by_dir.setdefault(posixpath.dirname(info.path), []).append(info)
    for relative_dir, files in groups.items():
        if not in_shard(relative_dir or ".", shard):
            continue
        relatives = [inventory.relative_path(file) for file in files]
        dest_dir = posixpath.join(target.path, relative_dir) if relative_dir else target.path.rstrip("/")
        base = run_fingerprint(plan, options, {rel: file_manifest(manifest, rel) for rel in relatives})
        identity = [[rel, *source_identity(inventory.info(file)).values()]
                    for rel, file in zip(relatives, files)]
        part = _Partition(relative=relative_dir, key=f"partition:{relative_dir}", files=files,
                          relatives=relatives, dest_dir=dest_dir, identity=identity,
                          fingerprint=_partition_fingerprint(base, target_bytes, max_bytes))
        for info in by_dir.get(dest_dir, []):
            name = posixpath.basename(info.path)
            if info.type == fs.FileType.File and _is_data_file(name):
                part.existing.append(info.path)
            elif info.type == fs.FileType.File and _TEMPORARY.match(name):
                part.temporaries.append(info.path)
        partitions.append(part)
    markers = store.load(part.key for part in partitions) if store else {}

    resumed: dict[str, PartitionResult] = {}
    remaining: list[_Partition] = []
    for part in partitions:
        marker = markers.get(part.key)
        if marker and marker.get("state") == "publishing":
            # Outputs of an interrupted publication are ours to replace.
            part.replaceable.update(marker.get("outputs", ()))
        outputs = marker.get("outputs", []) if marker else []
        complete = (marker and marker.get("state") == "done"
                    and marker.get("fingerprint") == part.fingerprint
                    and marker.get("source") == part.identity and outputs
                    and all((info := existing.get(item["path"])) is not None
                            and info.type == fs.FileType.File and info.size == item["size"]
                            for item in outputs))
        if complete:
            resumed[part.key] = _resumed(plan, part, marker)
        else:
            remaining.append(part)

    footers = read_footers(source_fs, [file for part in remaining for file in part.files])
    position = 0
    for part in remaining:
        part.footers = footers[position:position + len(part.files)]
        position += len(part.files)
        _prepare(part, plan=plan, engine=engine, preserved=preserved, overwrite=overwrite,
                 inventory=inventory, existing=existing, target_root=target.path, manifest=manifest,
                 compression=compression, compression_level=compression_level)
    removed = 0
    for part in remaining:
        for path in part.temporaries:
            target_fs.delete_file(path)
            removed += 1

    remote = not isinstance(source_fs, fs.LocalFileSystem)
    ordered = remaining if workers == 1 or len(remaining) <= 1 else sorted(
        remaining, key=lambda part: (-part.source_bytes, part.relative))
    depth = prefetch if prefetch is not None else (max(2, workers) if remote else 0)
    decode_order = [file for part in ordered if not part.copies_original for file in part.files]
    sizes = {file: item[1] for part in remaining for file, item in zip(part.files, part.identity)}
    prefetcher = (Prefetcher(source_fs, decode_order, depth=depth, sizes=sizes)
                  if depth and decode_order else None)
    row_group_size = options.get("row_group_size")
    group_rows = min(row_group_size or 1_048_576, 64 * 1_048_576)

    def run(part: _Partition, observe: Callable[[pa.Table], None] | None = None) -> PartitionResult:
        if part.copies_original:
            result = _publish_originals(part, source_fs, target_fs, store, _copy_or_skip)
        else:
            result = _compact(part, source_fs=source_fs, target_fs=target_fs, engine=engine,
                              options=options, writer_keys=_WRITER_OPTIONS, guard=guard,
                              target_bytes=target_bytes, max_bytes=max_bytes,
                              group_rows=group_rows, store=store,
                              observe=observe, prefetcher=prefetcher, copy=_copy_or_skip)
        published = {item.path for item in result.outputs}
        for path in part.existing:
            if path not in published and target_fs.get_file_info(path).type == fs.FileType.File:
                target_fs.delete_file(path)  # Permitted by overwrite, or our own partial output.
        if store is not None:
            store.record(part.key, {
                "state": "done", "source": part.identity, "fingerprint": part.fingerprint,
                "action": result.action, "rows": part.rows, "sort_keys": list(result.plan.sort_keys),
                "prefix_keys": list(result.plan.prefix_keys), "note": result.plan.note,
                "outputs": [{"path": item.path, "size": item.file_size,
                             "rows": item.plan.total_rows} for item in result.outputs],
            })
        return result

    produced: dict[str, PartitionResult] = {}
    completed = 0

    def finished(part: _Partition, result: PartitionResult) -> None:
        nonlocal completed
        completed += 1
        produced[part.key] = result
        if progress is not None:
            progress(result, completed, len(remaining))

    previous_io = pa.io_thread_count()
    try:
        if workers == 1 or len(remaining) <= 1:
            for part in remaining:
                finished(part, run(part))
        else:
            wanted_io = io_threads or (4 * workers if remote else None)
            if wanted_io and wanted_io > previous_io:
                pa.set_io_thread_count(wanted_io)
            duckdb_reserve = int(parse_size(options.get("memory_limit", "6GB")) * 1.25)
            tasks = []
            for index, part in enumerate(ordered):
                kind = "copy" if part.copies_original else engine
                stream_estimate = _streaming_estimate(part, group_rows)
                fixed = (duckdb_reserve + stream_estimate if kind == "duckdb" and not part.streaming
                         else stream_estimate if part.streaming and kind != "copy" else None)
                tasks.append(Task(
                    index=index, kind=kind, rows=part.rows, size=part.source_bytes,
                    schema_key=tuple((item.name, str(item.type)) for item in part.schema),
                    run=(lambda observe, part=part: run(part, observe)), fixed_estimate=fixed,
                ))
            run_tasks(tasks, workers=workers, budget=budget,
                      on_done=lambda task, result: finished(ordered[task.index], result))
    finally:
        if pa.io_thread_count() != previous_io:
            pa.set_io_thread_count(previous_io)
        if prefetcher is not None:
            prefetcher.close()
    results = tuple(resumed[part.key] if part.key in resumed else produced[part.key]
                    for part in partitions)
    files = tuple(item for result in results for item in result.outputs)
    return DatasetRewriteResult(str(source), str(destination), files, removed_temporaries=removed,
                                shard=shard, partitions=results)


def _streaming_estimate(part: _Partition, group_rows: int) -> int:
    """Memory for a stream that holds about one row group at a time.

    Decoded size per row is not in the footer; eight times its uncompressed
    size per row is a deliberately generous stand-in, and the scheduler also
    checks live allocations.
    """

    uncompressed = sum(footer.row_group(index).total_byte_size
                       for footer in part.footers for index in range(footer.num_row_groups))
    per_row = uncompressed / max(1, part.rows)
    return int(8 * per_row * min(group_rows, max(1, part.rows))) + 2 * TASK_OVERHEAD_BYTES


def _prepare(part: _Partition, *, plan: RewritePlan, engine: str, preserved: bool, overwrite: bool,
             inventory: ParquetInventory, existing: dict[str, fs.FileInfo], target_root: str,
             manifest: Any, compression: Any, compression_level: Any) -> None:
    schemas = [footer.schema.to_arrow_schema() for footer in part.footers]
    part.schema = schemas[0]
    for path, schema in zip(part.files[1:], schemas[1:]):
        if not schema.equals(part.schema, check_metadata=False):
            raise ValueError(f"partition {part.relative or '.'!r} mixes physical schemas "
                             f"({part.files[0]} and {path}); it cannot be compacted into one order")
    bound = plan.for_table(pa.Table.from_batches([], schema=part.schema))
    blocked = unreorderable_columns(part.schema) if engine == "arrow" and bound.sort_keys else []
    if blocked and bound.prefix_keys:
        raise ValueError(f"cannot sort partition {part.relative or '.'!r} by the required prefix: "
                         f"Arrow cannot reorder {blocked}")
    if blocked:
        bound = replace(bound, sort_keys=(), note=f"input order kept: Arrow cannot reorder {blocked}")
    if engine == "duckdb" and bound.sort_keys:
        from .engines import validate_duckdb_schema
        validate_duckdb_schema(part.schema)
    part.bound = replace(bound, total_rows=part.rows)
    part.streaming = not part.bound.sort_keys
    part.copies_original = preserved and len(part.files) == 1 and part.streaming
    if not part.copies_original:
        subset = ParquetInventory(inventory.filesystem, inventory.root, tuple(part.files),
                                  inventory.is_directory, inventory.infos)
        part.settings = resolve_compression(
            inspect_compression(subset, manifest=manifest, footers=dict(zip(part.files, part.footers))),
            compression=compression, compression_level=compression_level,
        )
    info = existing.get(part.dest_dir)
    if info is not None and info.type != fs.FileType.Directory:
        raise FileExistsError(f"output partition is not a directory: {part.dest_dir}")
    parent = posixpath.dirname(part.dest_dir)
    while parent.startswith(target_root.rstrip("/") + "/"):
        info = existing.get(parent)
        if info is not None and info.type != fs.FileType.Directory:
            raise FileExistsError(f"output parent is not a directory: {parent}")
        parent = posixpath.dirname(parent)
    foreign = [path for path in part.existing if path not in part.replaceable]
    if foreign and not overwrite:
        raise FileExistsError(
            f"destination partition {part.dest_dir} already contains Parquet files "
            f"({posixpath.basename(foreign[0])}...); pass overwrite=True to replace "
            "that partition's data files")


def _publish_originals(part: _Partition, source_fs: fs.FileSystem, target_fs: fs.FileSystem,
                       store: Any, copy: Callable, note: str | None = None,
                       timings: dict[str, Any] | None = None) -> PartitionResult:
    outputs = [posixpath.join(part.dest_dir, posixpath.basename(path)) for path in part.files]
    if store is not None:
        store.record(part.key, {"state": "publishing", "outputs": outputs})
    plan = replace(part.bound, sort_keys=(), prefix_keys=(), note=note or part.bound.note)
    results = []
    for path, output, footer, item in zip(part.files, outputs, part.footers, part.identity):
        copied = copy(path, output, source_fs, target_fs, replace(plan, total_rows=footer.num_rows),
                      True, item[1])
        results.append(copied)
    timings = dict(timings or {})
    timings["write_seconds"] = timings.get("write_seconds", 0.0) + sum(r.write_seconds for r in results)
    return PartitionResult(partition=part.relative, path=part.dest_dir, sources=tuple(part.files),
                           outputs=tuple(results), action="copied", source_bytes=part.source_bytes,
                           plan=plan, **timings)


def _compact(part: _Partition, *, source_fs: fs.FileSystem, target_fs: fs.FileSystem, engine: str,
             options: dict[str, Any], writer_keys: set[str], guard: bool, target_bytes: int,
             max_bytes: int, group_rows: int, store: Any, observe: Callable | None, prefetcher: Prefetcher | None,
             copy: Callable) -> PartitionResult:
    bound = part.bound
    settings = part.settings
    index_options = {key: value for key, value in
                     source_index_options(part.footers, output_group_rows=min(
                         group_rows, max(1, part.rows))).items() if key not in options}
    writer_options = {**index_options, **{key: value for key, value in options.items()
                                          if key in writer_keys}}
    metadata = dict(part.schema.metadata or {})
    metadata[b"oparq.sort_plan"] = json.dumps(bound.as_dict(include_profiles=False)).encode()
    metadata[b"oparq.compression"] = compression_metadata(settings.compression, settings.compression_level)
    schema = part.schema.with_metadata(metadata)
    sorting_columns = (pq.SortingColumn.from_ordering(schema, bound.sort_order,
                                                      null_placement=bound.null_placement)
                       if bound.sorting_keys else None)
    writer = _RollingWriter(target_fs, part.dest_dir, schema,
                            {**settings.writer_options(), **writer_options,
                             "sorting_columns": sorting_columns},
                            target_bytes=target_bytes, max_bytes=max_bytes, total_rows=part.rows,
                            group_rows=group_rows,
                            prior_bytes_per_row=part.source_bytes / part.rows if part.rows else None)
    timings: dict[str, Any] = {"planning_seconds": 0.0}
    buffers = (lambda path: prefetcher.take(path)) if prefetcher is not None else (lambda path: None)
    try:
        if part.streaming:
            started = perf_counter()
            for path in part.files:
                for batch in _source_batches(path, source_fs, buffers(path)):
                    writer.write(pa.Table.from_batches([batch]).replace_schema_metadata(schema.metadata))
            timings["sort_seconds"] = perf_counter() - started - writer.write_seconds
        elif engine == "arrow":
            _arrow_sorted(part, source_fs, schema, writer, group_rows, options, observe, buffers, timings)
        else:
            _duckdb_sorted(part, source_fs, schema, writer, options, buffers, timings)
        files = writer.finish()
    except BaseException:
        writer.abort()
        raise
    if writer.written_rows != part.rows:
        writer.abort()
        raise RuntimeError(f"source changed during rewrite of partition {part.relative or '.'!r}: "
                           f"expected {part.rows} rows, received {writer.written_rows}")
    timings["write_seconds"] = writer.write_seconds
    sizes = [target_fs.get_file_info(temporary).size for temporary, _, _ in files]
    if guard and not bound.prefix_keys and sum(sizes) >= part.source_bytes:
        writer.abort()
        note = (f"sorted partition was {sum(sizes):,} bytes versus {part.source_bytes:,} "
                "source bytes; kept original files")
        return _publish_originals(part, source_fs, target_fs, store, copy, note=note, timings=timings)
    outputs = []
    started = perf_counter()
    try:
        if store is not None:
            store.record(part.key, {"state": "publishing", "outputs": [final for _, final, _ in files]})
        for (temporary, final, rows), size in zip(files, sizes):
            target_fs.move(temporary, final)
            outputs.append(WriteResult(path=final, file_size=size, plan=replace(bound, total_rows=rows),
                                       planning_seconds=0.0, sort_seconds=0.0, write_seconds=0.0,
                                       action="compacted"))
    except BaseException:
        # Unpublished temporaries are removed now; published outputs are
        # named by the publishing marker and replaced by the next run.
        writer.abort()
        raise
    timings["write_seconds"] += perf_counter() - started
    note = f"sorted across {len(part.files)} source file(s) into {len(outputs)}"
    if writer.reduced_group_rows is not None:
        note += (f"; row groups reduced to as few as {writer.reduced_group_rows:,} rows "
                 f"to keep files under max_file_size")
    return PartitionResult(partition=part.relative, path=part.dest_dir, sources=tuple(part.files),
                           outputs=tuple(outputs), action="compacted", source_bytes=part.source_bytes,
                           plan=replace(bound, note=note), **timings)


def _arrow_sorted(part: _Partition, source_fs: fs.FileSystem, schema: pa.Schema,
                  writer: _RollingWriter, group_rows: int, options: dict[str, Any],
                  observe: Callable | None, buffers: Callable, timings: dict[str, Any]) -> None:
    from .core import plan_permutation
    from .storage import read_parquet_source

    tables = []
    for path in part.files:
        buffer = buffers(path)
        table = (read_parquet_source(path, source_fs) if buffer is None
                 else pq.read_table(pa.BufferReader(buffer)))
        del buffer
        tables.append(table.replace_schema_metadata(schema.metadata))
    # File order, then row order: the stable sort keeps ties in that order.
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    del tables
    if observe is not None:
        observe(table)
    started = perf_counter()
    permutation = plan_permutation(table, part.bound, sort_backend=options.get("sort_backend", "auto"))
    timings["permutation_seconds"] = perf_counter() - started
    gather_threads = options.get("gather_threads", 0)
    started = perf_counter()
    chunked = (ChunkedTableGather(table, gather_threads=gather_threads)
               if use_chunked_gather(table, group_rows) else None)
    gathering = perf_counter() - started
    for offset in range(0, table.num_rows, group_rows):
        started = perf_counter()
        indices = permutation.slice(offset, group_rows)
        group = (chunked.take(indices) if chunked is not None
                 else take_table(table, indices, gather_threads=gather_threads))
        gathering += perf_counter() - started
        writer.write(group)
        del group
    timings["gathering_seconds"] = gathering
    timings["sort_seconds"] = timings["permutation_seconds"] + gathering


def _duckdb_sorted(part: _Partition, source_fs: fs.FileSystem, schema: pa.Schema,
                   writer: _RollingWriter, options: dict[str, Any], buffers: Callable,
                   timings: dict[str, Any]) -> None:
    from .engines import duckdb_sorted_batches

    def inputs() -> Iterator[pa.RecordBatch]:
        for path in part.files:
            yield from _source_batches(path, source_fs, buffers(path))

    with tempfile.TemporaryDirectory(prefix="oparq-spill-", dir=options.get("temp_directory")) as spill:
        started = perf_counter()
        with duckdb_sorted_batches(part.schema, inputs(), part.bound.sort_keys, temp_directory=spill,
                                   memory_limit=options.get("memory_limit", "6GB"),
                                   max_temp_directory_size=options.get("max_temp_directory_size", "18GB"),
                                   null_placement=part.bound.null_placement) as batches:
            for batch in batches:
                writer.write(pa.Table.from_batches([batch]).replace_schema_metadata(schema.metadata))
        # DuckDB fuses scan, sort and gather; only their combined time is known.
        timings["sort_seconds"] = perf_counter() - started - writer.write_seconds


def _resumed(plan: RewritePlan, part: _Partition, marker: dict[str, Any]) -> PartitionResult:
    bound = SortPlan(
        requested_algorithm=plan.algorithm, algorithm=plan.algorithm,
        sort_keys=tuple(marker.get("sort_keys", ())), prefix_keys=tuple(marker.get("prefix_keys", ())),
        profiles=(), total_rows=int(marker.get("rows", 0)), sampled_rows=plan.sampled_rows,
        value_order=plan.value_order, null_placement=plan.null_placement,
        note=marker.get("note") or "completed by an earlier run; resumed from checkpoint",
    )
    outputs = tuple(WriteResult(path=item["path"], file_size=int(item["size"]),
                                plan=replace(bound, total_rows=int(item.get("rows", 0))),
                                planning_seconds=0.0, sort_seconds=0.0, write_seconds=0.0,
                                permutation_seconds=0.0, gathering_seconds=0.0,
                                action=marker["action"], resumed=True)
                    for item in marker["outputs"])
    return PartitionResult(partition=part.relative, path=part.dest_dir, sources=tuple(part.files),
                           outputs=outputs, action=marker["action"], source_bytes=part.source_bytes,
                           plan=bound, permutation_seconds=0.0, gathering_seconds=0.0, resumed=True)
