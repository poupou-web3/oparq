"""Native Arrow sort backends, including reusable integer-ranked keys."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc

from .profile import decode_dictionary


SORT_BACKENDS = ("auto", "arrow", "rank")
PARALLEL_TAKE_MIN_BYTES = 64 * 1024**2
CHUNKED_TAKE_MIN_REPEATED_BYTES = 256 * 1024**2


def _per_key_null_placement() -> bool:
    try:
        pc.SortOptions(sort_keys=[("", "ascending", "at_end")])
    except (TypeError, ValueError):
        return False
    return True


# PyArrow 25 deprecates the global null_placement option, and releases before
# it only accept that global form. Every oparq sort uses one placement.
PER_KEY_NULL_PLACEMENT = _per_key_null_placement()


def sort_options(
    keys: Sequence[tuple[str, str]], null_placement: str
) -> dict[str, object]:
    """``sort_indices`` keyword arguments for (name, order) keys."""

    if PER_KEY_NULL_PLACEMENT:
        return {"sort_keys": [(name, order, null_placement) for name, order in keys]}
    return {"sort_keys": list(keys), "null_placement": null_placement}


def natural_rank(
    column: pa.ChunkedArray, *, null_placement: str = "at_end"
) -> pa.ChunkedArray:
    """Replace scalar values with their natural-order ranks, keeping nulls.

    Comparisons then touch small integers instead of long string payloads.
    Dictionary encoding and sorting distinct values both run in Arrow C++.
    """

    values = decode_dictionary(column)
    # Chunked STRING/BINARY inputs may exceed the 32-bit offsets supported by
    # one array. Widen the temporary key storage before combining them.
    if values.nbytes >= 2**31:
        if pa.types.is_string(values.type):
            values = pc.cast(values, pa.large_string())
        elif pa.types.is_binary(values.type):
            values = pc.cast(values, pa.large_binary())
    values = values.combine_chunks()
    encoded = pc.dictionary_encode(values)
    dictionary = encoded.dictionary
    if len(dictionary) == 0:
        return pa.chunked_array([encoded.indices])
    options = (
        sort_options([("", "ascending")], null_placement)
        if PER_KEY_NULL_PLACEMENT
        else {"sort_keys": "ascending", "null_placement": null_placement}
    )
    ranks = pc.cast(pc.rank(dictionary, tiebreaker="dense", **options), pa.int32())
    return pa.chunked_array([pc.take(ranks, encoded.indices)])


def key_table(
    table: pa.Table,
    keys: Sequence[str],
    *,
    sort_backend: str = "auto",
    null_placement: str = "at_end",
) -> pa.Table:
    if sort_backend not in SORT_BACKENDS:
        raise ValueError(f"sort_backend must be one of {SORT_BACKENDS}")
    arrays = []
    for name in keys:
        column = decode_dictionary(table[name])
        data_type = column.type
        is_bytes = any(
            check(data_type)
            for check in (
                pa.types.is_string,
                pa.types.is_large_string,
                pa.types.is_binary,
                pa.types.is_large_binary,
                pa.types.is_fixed_size_binary,
            )
        )
        # A bounded NDV check avoids constructing huge dictionaries for unique
        # IDs. The rank backend explicitly opts into that overhead.
        use_rank = sort_backend == "rank"
        if sort_backend == "auto" and is_bytes and len(column) >= 100_000:
            probe_rows = min(16_384, len(column))
            # Evenly distributed zero-copy windows avoid a clustered head.
            windows = [
                column.slice(i * (len(column) - probe_rows // 4) // 3, probe_rows // 4)
                for i in range(4)
            ]
            probe = pa.chunked_array(
                [chunk for piece in windows for chunk in piece.chunks]
            )
            distinct = pc.count_distinct(probe).as_py()
            use_rank = distinct / len(probe) <= 0.75
        if use_rank and len(column):
            column = natural_rank(column, null_placement=null_placement)
        arrays.append(column)
    return pa.table(arrays, names=list(keys))


def sort_key_indices(
    keys_table: pa.Table,
    keys: Sequence[str],
    *,
    null_placement: str = "at_end",
) -> pa.Array:
    return pc.sort_indices(
        keys_table, **sort_options([(name, "ascending") for name in keys], null_placement)
    )


def sort_indices(
    table: pa.Table,
    keys: Sequence[str],
    *,
    null_placement: str = "at_end",
    sort_backend: str = "auto",
) -> pa.Array:
    return sort_key_indices(
        key_table(table, keys, sort_backend=sort_backend, null_placement=null_placement),
        keys,
        null_placement=null_placement,
    )


def unreorderable_columns(schema: pa.Schema) -> list[str]:
    """Columns whose type Arrow's take kernel cannot gather.

    Reordering moves every column together, so one such column (for example
    ``string_view`` or run-end encoded data in PyArrow 25) prevents the Arrow
    engine from sorting the whole table. The kernel is probed with an empty
    column, so newly supported types are accepted automatically.
    """

    empty = pa.Table.from_batches([], schema=schema)
    indices = pa.array([], type=pa.int64())
    names = []
    for field, column in zip(schema, empty.columns):
        try:
            pc.take(column, indices)
        except pa.ArrowNotImplementedError:
            names.append(field.name)
    return names


def take_table(
    table: pa.Table, permutation: pa.Array, *, gather_threads: int = 0
) -> pa.Table:
    """Gather independent columns concurrently with native Arrow kernels.

    This helper only accepts the in-range permutation emitted by our sort
    backends. A small table stays serial; automatic mode caps workers at four
    to avoid oversubscribing a memory-bandwidth-bound operation.
    """

    if gather_threads < 0:
        raise ValueError("gather_threads must be non-negative")
    workers = gather_threads or min(4, pa.cpu_count())
    if workers <= 1 or table.num_columns <= 1 or table.nbytes < PARALLEL_TAKE_MIN_BYTES:
        return table.take(permutation)

    def gather(column: pa.ChunkedArray) -> pa.ChunkedArray:
        return pc.take(column, permutation, boundscheck=False)

    with ThreadPoolExecutor(max_workers=min(workers, table.num_columns)) as executor:
        columns = list(executor.map(gather, table.columns))
    return pa.Table.from_arrays(columns, schema=table.schema)


def use_chunked_gather(table: pa.Table, group_rows: int) -> bool:
    """Choose the chunk-aware path when repeated full-column copies are costly.

    Arrow's take of a ChunkedArray concatenates its entire source on *each*
    call, even if the requested indices cover just one output row group.
    The chunk-aware path sorts each output group's indices, so keep the
    native kernel for small inputs and lightly chunked columns.
    """

    groups = (table.num_rows + group_rows - 1) // group_rows
    chunked = [column for column in table.columns if column.num_chunks > 1]
    return (
        groups >= 4
        and any(column.num_chunks >= 4 for column in chunked)
        and sum(column.nbytes for column in chunked) * (groups - 1)
        >= CHUNKED_TAKE_MIN_REPEATED_BYTES
    )


def _permutation_view(permutation: pa.Array) -> memoryview:
    """Read sort indices without making Python integers or a NumPy copy."""

    formats = {
        pa.int32(): "i",
        pa.uint32(): "I",
        pa.int64(): "q",
        pa.uint64(): "Q",
    }
    try:
        format_code = formats[permutation.type]
    except KeyError as error:
        raise TypeError("chunked gather needs 32- or 64-bit integer indices") from error
    if permutation.null_count:
        raise ValueError("chunked gather needs a non-null sort permutation")
    values = memoryview(permutation.buffers()[1]).cast(format_code)
    return values[permutation.offset : permutation.offset + len(permutation)]


def _partition_sorted_indices(
    sorted_indices: pa.Array, lengths: tuple[int, ...]
) -> list[tuple[int, pa.Array]]:
    """Find each source chunk's contiguous window in the sorted indices."""

    values = _permutation_view(sorted_indices)
    spans = []
    start = 0
    lower = 0
    for chunk_index, length in enumerate(lengths):
        end = start + length
        upper = bisect_left(values, end, lo=lower)
        if upper > lower:
            local = sorted_indices.slice(lower, upper - lower)
            if start:
                local = pc.subtract(local, pa.scalar(start, type=sorted_indices.type))
            spans.append((chunk_index, local))
        start = end
        lower = upper
    return spans


class ChunkedTableGather:
    """Gather row groups without repeatedly concatenating the whole input.

    Column chunk layouts can differ. Sort each group permutation by source
    position in Arrow C++, split it at each column's chunk boundaries, then
    restore output order with one group-sized take. Dictionary arrays are
    unified once so each group retains the same dictionary as a full-source
    Arrow take, even when some input chunks contribute no rows.
    """

    def __init__(self, table: pa.Table, *, gather_threads: int = 0) -> None:
        if gather_threads < 0:
            raise ValueError("gather_threads must be non-negative")
        self.schema = table.schema
        self.nbytes = table.nbytes
        self.columns = [
            column.unify_dictionaries()
            if pa.types.is_dictionary(column.type) and column.num_chunks > 1
            else column
            for column in table.columns
        ]
        self.layouts = [
            tuple(len(chunk) for chunk in column.chunks) for column in self.columns
        ]
        self.gather_threads = gather_threads

    def take(self, permutation: pa.Array) -> pa.Table:
        _permutation_view(permutation)
        source_order = pc.sort_indices(permutation)
        sorted_indices = pc.take(permutation, source_order, boundscheck=False)
        output_order = pc.sort_indices(source_order)
        partitions = {
            layout: _partition_sorted_indices(sorted_indices, layout)
            for layout in set(self.layouts)
            if len(layout) > 1
        }

        def gather(item: tuple[pa.ChunkedArray, tuple[int, ...]]) -> pa.Array:
            column, layout = item
            if len(layout) == 1:
                return pc.take(column.chunk(0), permutation, boundscheck=False)
            spans = partitions[layout]
            pieces = [
                pc.take(column.chunk(chunk_index), local, boundscheck=False)
                for chunk_index, local in spans
            ]
            if not pieces:
                return column.slice(0, 0).combine_chunks()
            combined = pieces[0] if len(pieces) == 1 else pa.concat_arrays(pieces)
            return pc.take(combined, output_order, boundscheck=False)

        items = list(zip(self.columns, self.layouts))
        workers = self.gather_threads or min(4, pa.cpu_count())
        if workers <= 1 or len(items) <= 1 or self.nbytes < PARALLEL_TAKE_MIN_BYTES:
            columns = [gather(item) for item in items]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
                columns = list(executor.map(gather, items))
        return pa.Table.from_arrays(columns, schema=self.schema)
