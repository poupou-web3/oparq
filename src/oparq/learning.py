"""Learn an ordering once, then reuse it without profiling future files."""

from __future__ import annotations

import tempfile
import base64
import json
import random
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.fs as fs

from .models import RewritePlan
from .profile import as_table


def fit(
    value: pa.Table | pa.RecordBatch,
    *,
    algorithms: Iterable[str] = ("codec_fast", "portfolio"),
    prefix: Iterable[str] = (),
    compression: Any = "zstd",
    compression_level: Any = None,
    row_group_size: int | None = None,
    min_improvement: float = 0.005,
    **planning_options: Any,
) -> RewritePlan:
    """Compare strategies on supplied sample rows and freeze the winning keys.

    The input-order (or mandatory-prefix) baseline is always included. The
    resulting plan is a heuristic trained on these samples, not a proof of
    the globally smallest file. Codec settings remain fixed during comparison.
    """

    from .io import write
    from .storage import compression_metadata

    if not 0 <= min_improvement < 1:
        raise ValueError("min_improvement must be in [0, 1)")
    table = as_table(value)
    # Freeze effective defaults now, rather than relying on a future encoder
    # version to give None the same meaning as during training.
    writer_settings = json.loads(compression_metadata(compression, compression_level))
    compression = writer_settings["compression"]
    compression_level = writer_settings["compression_level"]
    prefix_keys = tuple(prefix)
    names = tuple(dict.fromkeys(("none", *algorithms)))
    evaluations = []
    results = []
    with tempfile.TemporaryDirectory(prefix="oparq-fit-") as directory:
        for index, algorithm in enumerate(names):
            result = write(
                table, Path(directory) / f"{index}.parquet",
                algorithm=algorithm, prefix=prefix_keys,
                compression=compression, compression_level=compression_level,
                row_group_size=row_group_size, **planning_options,
            )
            results.append(result)
            evaluations.append({
                "algorithm": algorithm,
                "resolved_algorithm": result.plan.algorithm,
                "sort_keys": list(result.plan.sort_keys),
                "size_bytes": result.file_size,
                "planning_seconds": result.planning_seconds,
            })
    baseline = results[0]
    winner = min(results, key=lambda item: item.file_size)
    if baseline.file_size and (baseline.file_size - winner.file_size) / baseline.file_size < min_improvement:
        winner = baseline
    selected = winner.plan
    return RewritePlan(
        algorithm=selected.algorithm, sort_keys=selected.sort_keys,
        prefix_keys=selected.prefix_keys,
        column_types=tuple((field.name, str(field.type)) for field in table.schema),
        value_order=selected.value_order, null_placement=selected.null_placement,
        compression=compression, compression_level=compression_level,
        sampled_rows=table.num_rows, evaluations=tuple(evaluations),
        schema_base64=base64.b64encode(table.schema.serialize().to_pybytes()).decode("ascii"),
    )


SAMPLE_METHODS = ("row_groups", "head")


def _read_window(path: str, filesystem: fs.FileSystem, footer: Any, rows: int,
                 method: str, rng: random.Random) -> pa.Table:
    """Read ``rows`` consecutive rows from one file.

    ``head`` starts at the first row. ``row_groups`` starts at the first row
    of a seeded random row group, chosen in proportion to its rows, and
    continues into later groups if needed. Pages are streamed, so only the
    window's pages (not whole files) are read. Consecutive rows keep the
    file's existing clustering, which compression comparisons depend on.
    """

    import pyarrow.parquet as pq

    groups = footer.num_row_groups
    first = 0
    if method == "row_groups" and groups > 1 and footer.num_rows > 0:
        target = rng.randrange(footer.num_rows)
        for first in range(groups):
            target -= footer.row_group(first).num_rows
            if target < 0:
                break
    with filesystem.open_input_file(path) as handle:
        reader = pq.ParquetFile(handle, metadata=footer, buffer_size=1 << 20, pre_buffer=False)
        schema = reader.schema_arrow
        batches, collected = [], 0
        for batch in reader.iter_batches(batch_size=min(rows, 65_536) or 1,
                                         row_groups=list(range(first, groups)), use_threads=True):
            batches.append(batch.slice(0, rows - collected))
            collected += batches[-1].num_rows
            if collected >= rows:
                break
        if collected < rows and first > 0:
            # Too few rows after the chosen group: take the rest from the start.
            for batch in reader.iter_batches(batch_size=min(rows, 65_536),
                                             row_groups=list(range(first)), use_threads=True):
                batches.append(batch.slice(0, rows - collected))
                collected += batches[-1].num_rows
                if collected >= rows:
                    break
    return pa.Table.from_batches(batches, schema=schema)


def _sample(inventory: Any, indices: list[int], footers: dict[str, Any], sample_rows: int,
            method: str, seed: int) -> pa.Table:
    if method not in SAMPLE_METHODS:
        raise ValueError(f"sample_method must be one of {SAMPLE_METHODS}")
    share, extra = divmod(sample_rows, len(indices))
    tables = []
    for position, file_index in enumerate(indices):
        path = inventory.files[file_index]
        # A per-file generator keeps each window independent of the others.
        rng = random.Random(f"{seed}:{inventory.relative_path(path)}")
        tables.append(_read_window(path, inventory.filesystem, footers[path],
                                   share + (position < extra), method, rng))
    schema = tables[0].schema
    if any(not table.schema.equals(schema, check_metadata=False) for table in tables):
        raise ValueError("training files have different physical schemas; fit one plan per schema")
    return pa.concat_tables(tables)


def sample_dataset(
    source: str | Path,
    *, filesystem: fs.FileSystem | None = None,
    sample_rows: int = 250_000,
    sample_files: int = 32,
    sample_method: str = "row_groups",
    seed: int = 0,
) -> pa.Table:
    """Read bounded rows from deterministic files spread across a source tree.

    ``sample_files`` files are chosen evenly across the sorted inventory and
    contribute equal consecutive-row windows; see ``sample_method``. The same
    inputs and ``seed`` always give the same sample.
    """

    from .storage import inventory_parquet, read_footers, spread_indices

    if sample_rows <= 0 or sample_files <= 0:
        raise ValueError("sample_rows and sample_files must be positive")
    inventory = inventory_parquet(source, filesystem)
    indices = spread_indices(len(inventory.files), min(sample_files, sample_rows))
    paths = [inventory.files[index] for index in indices]
    footers = dict(zip(paths, read_footers(inventory.filesystem, paths)))
    return _sample(inventory, indices, footers, sample_rows, sample_method, seed)


def validate_plan(
    windows: Iterable[tuple[str, pa.Table]], plan: RewritePlan,
    *, row_group_size: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Measure a fixed plan against input order on held-out row windows.

    Both orders are written with the plan's codec and level. The input-order
    baseline keeps the plan's mandatory prefix, matching ``fit``'s baseline.
    """

    from .io import write

    records = []
    with tempfile.TemporaryDirectory(prefix="oparq-validate-") as directory:
        for index, (name, table) in enumerate(windows):
            settings = {"compression": plan.compression, "compression_level": plan.compression_level,
                        "row_group_size": row_group_size}
            baseline = write(table, Path(directory) / f"{index}-none.parquet", algorithm="none",
                             prefix=plan.prefix_keys, **settings)
            ordered = write(table, Path(directory) / f"{index}-plan.parquet",
                            plan=plan.for_table(table), **settings)
            records.append({
                "path": name, "rows": table.num_rows,
                "baseline_bytes": baseline.file_size, "plan_bytes": ordered.file_size,
                "savings": (1 - ordered.file_size / baseline.file_size) if baseline.file_size else 0.0,
            })
    return tuple(records)


def fit_dataset(
    source: str | Path,
    *, filesystem: fs.FileSystem | None = None,
    sample_rows: int = 250_000,
    sample_files: int = 32,
    sample_method: str = "row_groups",
    seed: int = 0,
    footer_files: int | None = 256,
    validation_files: int = 4,
    validation_rows: int = 100_000,
    compression: Any = "preserve",
    compression_level: Any = "preserve",
    compression_manifest: dict[str, Any] | None = None,
    **options: Any,
) -> RewritePlan:
    """Learn a reusable plan from local/cloud files with explicit codec provenance.

    Codec/level provenance is checked on ``footer_files`` footers spread across
    the inventory plus every training file (``None`` inspects every footer).
    Each rewrite still checks every file it re-encodes, so this sample only
    sets the codec used to compare orders here. Footers are read in parallel.

    After fitting, up to ``validation_files`` held-out files (never used for
    training) are each sampled for ``validation_rows`` rows and written in
    input order and in plan order; per-file savings are stored in
    ``RewritePlan.validation``. No held-out files remain when the inventory
    is not larger than ``sample_files``.
    """

    from .storage import (
        ParquetInventory, inspect_compression, inventory_parquet, read_footers,
        resolve_compression, spread_indices,
    )

    if sample_rows <= 0 or sample_files <= 0:
        raise ValueError("sample_rows and sample_files must be positive")
    if footer_files is not None and footer_files <= 0:
        raise ValueError("footer_files must be positive, or None for every file")
    if validation_files < 0 or validation_rows <= 0:
        raise ValueError("validation_files must be non-negative and validation_rows positive")
    inventory = inventory_parquet(source, filesystem)
    total = len(inventory.files)
    training = spread_indices(total, min(sample_files, sample_rows))
    used = set(training)
    unused = [index for index in range(total) if index not in used]
    held_out = [unused[index] for index in spread_indices(len(unused), validation_files)] if unused else []
    inspected = set(range(total)) if footer_files is None else set(spread_indices(total, footer_files))
    paths = sorted({inventory.files[index] for index in inspected | used | set(held_out)})
    footers = dict(zip(paths, read_footers(inventory.filesystem, paths)))
    checked = sorted({inventory.files[index] for index in inspected | used})
    subset = ParquetInventory(inventory.filesystem, inventory.root, tuple(checked),
                              inventory.is_directory, inventory.infos)
    settings = resolve_compression(
        inspect_compression(subset, manifest=compression_manifest, footers=footers),
        compression=compression, compression_level=compression_level,
    )
    table = _sample(inventory, training, footers, sample_rows, sample_method, seed)
    plan = fit(table, **settings.writer_options(), **options)
    if not held_out:
        return plan
    windows = []
    for index in held_out:
        path = inventory.files[index]
        rng = random.Random(f"{seed}:validate:{inventory.relative_path(path)}")
        window = _read_window(path, inventory.filesystem, footers[path], validation_rows,
                              sample_method, rng)
        windows.append((inventory.relative_path(path), window))
    return replace(plan, validation=validate_plan(
        windows, plan, row_group_size=options.get("row_group_size")))
