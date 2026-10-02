"""Learn an ordering once, then reuse it without profiling future files."""

from __future__ import annotations

import tempfile
import base64
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
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


def sample_dataset(
    source: str | Path,
    *, filesystem: fs.FileSystem | None = None,
    sample_rows: int = 250_000,
    sample_files: int = 8,
) -> pa.Table:
    """Read bounded rows from deterministic files spread across a source tree."""

    from .storage import inventory_parquet

    if sample_rows <= 0 or sample_files <= 0:
        raise ValueError("sample_rows and sample_files must be positive")
    inventory = inventory_parquet(source, filesystem)
    count = min(sample_files, len(inventory.files), sample_rows)
    indices = [((2 * index + 1) * len(inventory.files)) // (2 * count) for index in range(count)]
    tables = []
    share, extra = divmod(sample_rows, count)
    for index, file_index in enumerate(indices):
        dataset = ds.dataset([inventory.files[file_index]], filesystem=inventory.filesystem, format="parquet")
        tables.append(dataset.head(share + (index < extra)))
    schema = tables[0].schema
    if any(not table.schema.equals(schema, check_metadata=False) for table in tables):
        raise ValueError("training files have different physical schemas; fit one plan per schema")
    return pa.concat_tables(tables)


def fit_dataset(
    source: str | Path,
    *, filesystem: fs.FileSystem | None = None,
    sample_rows: int = 250_000,
    sample_files: int = 8,
    compression: Any = "preserve",
    compression_level: Any = "preserve",
    compression_manifest: dict[str, Any] | None = None,
    **options: Any,
) -> RewritePlan:
    """Learn a reusable plan from local/cloud files with explicit codec provenance."""

    from .storage import inspect_compression, resolve_compression

    settings = resolve_compression(
        inspect_compression(source, filesystem, manifest=compression_manifest),
        compression=compression, compression_level=compression_level,
    )
    table = sample_dataset(
        source, filesystem=filesystem, sample_rows=sample_rows, sample_files=sample_files,
    )
    return fit(table, **settings.writer_options(), **options)
