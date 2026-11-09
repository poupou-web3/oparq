"""Parallel, memory-bounded and resumable execution of per-file rewrites.

Parquet encoding and Arrow sorting each use about one core per file, and
both release the GIL, so a dataset rewrite runs files concurrently on a
thread pool. Threads share one Arrow memory pool, which lets one budget
account for every in-flight file and be checked against real allocations.

Decoded size cannot be predicted from a Parquet footer: dictionary-encoded
strings can decode to many times their "uncompressed" page size. The
estimator therefore learns decoded bytes per row from files it has read,
runs the first file of each schema alone, and also refuses new work while
the pool's live allocation is above the budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import pyarrow as pa
import pyarrow.fs as fs

# Peak Arrow allocation of a streamed in-memory rewrite relative to the
# decoded table: the table, its permutation and sort temporaries, one gathered
# row group and the writer's buffers. Measured near 2.2-2.5x; kept above that.
PEAK_FACTOR = 3.0
# Fixed allowance per running file for small buffers and copy streams.
TASK_OVERHEAD_BYTES = 16 * 1024 * 1024
DEFAULT_BUDGET_FRACTION = 0.7
# Prefetch downloads larger objects as concurrent ranged reads.
PREFETCH_PART_BYTES = 16 * 1024 * 1024
# Total concurrent range streams across all prefetched files.
PREFETCH_STREAMS = 16
CHECKPOINT_FORMAT = 1
CHECKPOINT_DIRECTORY = "_oparq/checkpoint"

_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?)b?\s*$", re.IGNORECASE)
_TEMPORARY = re.compile(r"^\.(?P<name>.+)\.oparq-(?:candidate-)?[0-9a-f]{32}\.(?:tmp|parquet)$")


def parse_size(value: int | str) -> int:
    """Parse ``6GB`` (decimal), ``6GiB`` (binary) or a byte count."""

    if isinstance(value, bool):
        raise TypeError("size must be an integer or a string such as '6GB'")
    if isinstance(value, int):
        number = value
    else:
        match = _SIZE.match(str(value))
        if not match:
            raise ValueError(f"invalid size: {value!r}; use bytes or a value such as '24GB'")
        base = 1024 if match[3] else 1000
        power = " kmgt".index(match[2].lower() or " ")
        number = int(float(match[1]) * base ** power)
    if number <= 0:
        raise ValueError("size must be positive")
    return number


def available_memory() -> int | None:
    """Physical memory, lowered to a cgroup (container) limit when one is set."""

    limits = []
    try:
        limits.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as handle:
                text = handle.read().strip()
        except OSError:
            continue
        # cgroup v1 reports "no limit" as a huge sentinel value.
        if text.isdigit() and 0 < int(text) < 1 << 60:
            limits.append(int(text))
    return min(limits) if limits else None


def resolve_budget(memory_budget: int | str | None) -> int:
    if memory_budget is not None:
        return parse_size(memory_budget)
    total = available_memory()
    if total is None:
        raise ValueError("cannot detect available memory; pass memory_budget explicitly")
    return int(total * DEFAULT_BUDGET_FRACTION)


def resolve_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a non-negative integer (0 = one per CPU)")
    return workers or (os.cpu_count() or 1)


def parse_shard(value: str | Sequence[int] | None) -> tuple[int, int] | None:
    """Accept ``"i/N"`` or ``(i, N)`` with ``0 <= i < N``."""

    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split("/")
        if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
            raise ValueError("shard must look like INDEX/COUNT, for example 0/4")
        index, count = (int(part) for part in parts)
    else:
        index, count = value
    if isinstance(index, bool) or isinstance(count, bool) or count <= 0 or not 0 <= index < count:
        raise ValueError("shard index must satisfy 0 <= index < count")
    return index, count


def in_shard(relative_path: str, shard: tuple[int, int] | None) -> bool:
    """Stable assignment by path hash: every machine computes the same split.

    Shards are disjoint and cover every file, without any coordination.
    """

    if shard is None:
        return True
    index, count = shard
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count == index


def stale_temporaries(listing: Iterable[fs.FileInfo], outputs: Iterable[str]) -> list[str]:
    """Hidden oparq temporaries left beside these outputs by an interrupted run."""

    wanted = {(posixpath.dirname(path), posixpath.basename(path)) for path in outputs}
    found = []
    for info in listing:
        if info.type != fs.FileType.File:
            continue
        match = _TEMPORARY.match(posixpath.basename(info.path))
        if match and (posixpath.dirname(info.path), match["name"]) in wanted:
            found.append(info.path)
    return sorted(found)


@dataclass(slots=True)
class CheckpointStore:
    """One small JSON marker per completed file.

    Object stores cannot append, so a single journal would be rewritten for
    every file. Per-file markers cost one PUT each, never conflict between
    shards or workers, and are loaded with one listing on resume. The default
    location is ``<destination>/_oparq/checkpoint``; readers skip ``_`` paths.
    """

    filesystem: fs.FileSystem
    root: str
    _created: bool = field(default=False, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def marker(self, relative_path: str) -> str:
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:40]
        return posixpath.join(self.root, f"{digest}.json")

    def load(self, relative_paths: Iterable[str], *, threads: int = 32) -> dict[str, dict[str, Any]]:
        selector = fs.FileSelector(self.root, allow_not_found=True, recursive=False)
        present = {info.path for info in self.filesystem.get_file_info(selector)
                   if info.type == fs.FileType.File}
        wanted = {self.marker(path): path for path in relative_paths}
        paths = [path for path in wanted if path in present]
        if not paths:
            return {}

        def read(path: str) -> dict[str, Any] | None:
            try:
                with self.filesystem.open_input_stream(path) as stream:
                    payload = json.loads(stream.read())
            except (OSError, ValueError):
                return None  # A corrupt marker is treated as absent.
            if (not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT
                    or payload.get("relative_path") != wanted[path]):
                return None
            return payload

        with ThreadPoolExecutor(max_workers=min(threads, len(paths)),
                                thread_name_prefix="oparq-checkpoint") as executor:
            payloads = list(executor.map(read, paths))
        return {wanted[path]: payload for path, payload in zip(paths, payloads) if payload}

    def record(self, relative_path: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if not self._created:
                self.filesystem.create_dir(self.root, recursive=True)
                self._created = True
        data = json.dumps({"format": CHECKPOINT_FORMAT, "relative_path": relative_path,
                           **payload}, sort_keys=True).encode()
        path = self.marker(relative_path)
        if isinstance(self.filesystem, fs.LocalFileSystem):
            # A local rename is atomic; an object PUT already is.
            temporary = posixpath.join(self.root, f".{uuid.uuid4().hex}.tmp")
            with self.filesystem.open_output_stream(temporary) as stream:
                stream.write(data)
            self.filesystem.move(temporary, path)
        else:
            with self.filesystem.open_output_stream(path) as stream:
                stream.write(data)


class MemoryEstimator:
    """Learn decoded Arrow bytes per row, per physical schema."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._bytes_per_row: dict[Any, float] = {}

    def observe(self, schema_key: Any, table: pa.Table) -> None:
        if table.num_rows <= 0:
            return
        value = table.nbytes / table.num_rows
        with self._lock:
            # The maximum keeps estimates conservative across heterogeneous files.
            self._bytes_per_row[schema_key] = max(value, self._bytes_per_row.get(schema_key, 0.0))

    def estimate(self, schema_key: Any, rows: int) -> int | None:
        with self._lock:
            bytes_per_row = self._bytes_per_row.get(schema_key)
        if bytes_per_row is None:
            return None if rows > 0 else TASK_OVERHEAD_BYTES
        return int(rows * bytes_per_row * PEAK_FACTOR) + TASK_OVERHEAD_BYTES


@dataclass(slots=True)
class Task:
    """One file of work. ``kind`` is ``copy``, ``arrow`` or ``duckdb``."""

    index: int
    kind: str
    rows: int
    size: int
    schema_key: Any
    run: Callable[[Callable[[pa.Table], None]], Any]
    fixed_estimate: int | None = None


def run_tasks(
    tasks: Sequence[Task], *, workers: int, budget: int,
    estimator: MemoryEstimator | None = None,
    on_done: Callable[[Task, Any], None] | None = None,
    memory_pool: pa.MemoryPool | None = None,
    poll_seconds: float = 0.05,
) -> dict[int, Any]:
    """Run tasks on ``workers`` threads without exceeding the memory budget.

    Admission uses the larger of the reserved estimates and the pool's live
    allocation. A task with an unknown decoded size calibrates its schema
    while no other decoding task runs. A task whose estimate exceeds the
    whole budget runs alone (with a warning) rather than never. On the first
    failure nothing new starts; running tasks finish and the error is raised.
    """

    estimator = estimator or MemoryEstimator()
    pool = memory_pool or pa.default_memory_pool()
    # Largest first reduces the tail; copies hold almost no memory and go last.
    pending = sorted(tasks, key=lambda task: (task.kind == "copy", -task.size, task.index))
    results: dict[int, Any] = {}
    running: dict[Future, Task] = {}
    reserved: dict[int, int] = {}
    calibrating: set[Any] = set()
    errors: list[BaseException] = []
    lock = Lock()

    def observer(task: Task) -> Callable[[pa.Table], None]:
        def observe(table: pa.Table) -> None:
            estimator.observe(task.schema_key, table)
            refined = estimator.estimate(task.schema_key, task.rows)
            with lock:
                reserved[task.index] = max(reserved.get(task.index, 0), refined or 0)
                calibrating.discard(task.schema_key)
        return observe

    def estimate(task: Task) -> int | None:
        if task.kind == "copy":
            return TASK_OVERHEAD_BYTES
        if task.fixed_estimate is not None:
            return task.fixed_estimate
        return estimator.estimate(task.schema_key, task.rows)

    def admissible(task: Task) -> int | None:
        """Return the reservation if the task may start now, else ``None``."""

        with lock:
            decoding = [item for item in running.values() if item.kind != "copy"]
            in_use = max(sum(reserved.values()), pool.bytes_allocated())
            wanted = estimate(task)
            if wanted is None:
                # Unknown size: calibrate alone, one file per schema at a time.
                if decoding or task.schema_key in calibrating:
                    return None
                calibrating.add(task.schema_key)
                return 0
            if task.kind != "copy" and calibrating:
                return None  # A file of unknown size is decoding; wait for its size.
            if task.kind == "copy" or not decoding:
                if task.kind != "copy" and wanted > budget:
                    warnings.warn(
                        f"file needs an estimated {wanted:,} bytes, above the "
                        f"{budget:,}-byte memory budget; running it alone",
                        RuntimeWarning, stacklevel=3,
                    )
                return wanted
            return wanted if in_use + wanted <= budget else None

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="oparq-rewrite") as executor:
        while pending or running:
            while not errors and pending and len(running) < workers:
                for position, task in enumerate(pending):
                    grant = admissible(task)
                    if grant is not None:
                        break
                else:
                    break
                pending.pop(position)
                with lock:
                    reserved[task.index] = grant
                running[executor.submit(task.run, observer(task))] = task
            if errors and not running:
                break
            if not running:
                if pending:
                    # Only reachable if no task can be admitted with nothing
                    # running, which admissible() rules out; guard anyway.
                    raise RuntimeError("scheduler could not admit any task")
                break
            done, _ = wait(running, timeout=poll_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                with lock:
                    reserved.pop(task.index, None)
                    calibrating.discard(task.schema_key)
                try:
                    value = future.result()
                except BaseException as error:  # noqa: BLE001 - re-raised below
                    errors.append(error)
                    continue
                results[task.index] = value
                if on_done is not None:
                    try:
                        on_done(task, value)
                    except BaseException as error:  # noqa: BLE001
                        errors.append(error)
    if errors:
        first = errors[0]
        if len(errors) > 1:
            first.add_note(f"{len(errors) - 1} other file(s) also failed in this run")
        raise first
    return results


class Prefetcher:
    """Download whole source objects ahead of the workers that decode them.

    A worker otherwise waits for its file's bytes before it can use a core.
    Prefetching overlaps those downloads with other files' CPU work, holding
    compressed bytes (not a decoded-size reservation) for at most ``depth``
    files. Buffers come from Arrow's pool, so the scheduler's live-allocation
    check counts them. A file that was not prefetched, or whose download
    failed, is read directly by its worker, which reports any real error.
    """

    def __init__(self, filesystem: fs.FileSystem, paths: Sequence[str], *, depth: int,
                 sizes: Mapping[str, int] | None = None) -> None:
        if depth <= 0:
            raise ValueError("prefetch depth must be positive")
        self._filesystem = filesystem
        self._sizes = dict(sizes or {})
        self._queue = list(paths)
        self._depth = depth
        self._futures: dict[str, Future] = {}
        self._lock = Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=depth, thread_name_prefix="oparq-prefetch")
        self._fill()

    def _download(self, path: str) -> pa.Buffer:
        size = self._sizes.get(path)
        if size is None or size <= PREFETCH_PART_BYTES:
            with self._filesystem.open_input_file(path) as handle:
                return handle.read_buffer()
        # One object-store stream is bandwidth-limited; fetch ranges
        # concurrently, each on its own handle, into one Arrow buffer.
        buffer = pa.allocate_buffer(size)
        view = memoryview(buffer).cast("B")

        def fetch(offset: int) -> int:
            length = min(PREFETCH_PART_BYTES, size - offset)
            with self._filesystem.open_input_file(path) as handle:
                handle.seek(offset)
                data = handle.read(length)
            if len(data) != length:
                raise OSError(f"source changed while prefetching: {path}")
            view[offset:offset + length] = data
            return length

        offsets = range(0, size, PREFETCH_PART_BYTES)
        streams = max(1, PREFETCH_STREAMS // self._depth)
        with ThreadPoolExecutor(max_workers=min(streams, len(offsets)),
                                thread_name_prefix="oparq-range") as executor:
            list(executor.map(fetch, offsets))
        return buffer

    def _fill(self) -> None:
        with self._lock:
            while not self._closed and self._queue and len(self._futures) < self._depth:
                path = self._queue.pop(0)
                self._futures[path] = self._executor.submit(self._download, path)

    def take(self, path: str) -> pa.Buffer | None:
        with self._lock:
            future = self._futures.pop(path, None)
            if future is None and path in self._queue:
                self._queue.remove(path)  # Its worker reads it directly.
        self._fill()
        if future is None:
            return None
        try:
            return future.result()
        except Exception:  # noqa: BLE001 - the direct read reports the real error
            return None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            futures = list(self._futures.values())
            self._futures.clear()
            self._queue.clear()
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=True)

