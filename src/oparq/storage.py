"""Arrow filesystem access and explicit Parquet compression provenance.

Parquet records a codec for each leaf column, but does not record its encoder
level. Preservation therefore uses an explicit manifest or oparq metadata for
levels; it never guesses a level from compressed bytes or a writer name.
"""

from __future__ import annotations

import inspect
import json
import math
import posixpath
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.parquet as pq


COMPRESSION_METADATA_KEY = b"oparq.compression"


@dataclass(frozen=True, slots=True)
class StorageLocation:
    filesystem: fs.FileSystem
    path: str

    @property
    def is_local(self) -> bool:
        return isinstance(self.filesystem, fs.LocalFileSystem)


@dataclass(frozen=True, slots=True)
class ParquetInventory:
    filesystem: fs.FileSystem
    root: str
    files: tuple[str, ...]
    is_directory: bool

    def relative_path(self, path: str) -> str:
        base = self.root if self.is_directory else posixpath.dirname(self.root)
        relative = posixpath.relpath(path, base or ".")
        if relative == ".." or relative.startswith("../"):
            raise ValueError(f"file is outside the source inventory: {path}")
        return relative


@dataclass(frozen=True, slots=True)
class SourceCompression:
    """Codecs observed in footers and levels established by provenance.

    A missing ``levels`` entry means unknown, while a present ``None`` entry
    explicitly means the encoder's default was requested. If provenance names
    conflicting levels, the column remains unknown and appears in ``conflicts``.
    """

    codecs: dict[str, tuple[str, ...]]
    levels: dict[str, int | None]
    level_sources: dict[str, str]
    conflicts: tuple[str, ...] = ()

    @property
    def unknown_level_columns(self) -> tuple[str, ...]:
        return tuple(
            column for column, codecs in self.codecs.items()
            if any(supports_compression_level(codec) for codec in codecs)
            and column not in self.levels
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "codecs": {name: list(values) for name, values in self.codecs.items()},
            "levels": self.levels,
            "level_sources": self.level_sources,
            "unknown_level_columns": list(self.unknown_level_columns),
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True, slots=True)
class CompressionSettings:
    compression: str | dict[str, str]
    compression_level: int | None | dict[str, int]
    provenance: str

    def writer_options(self) -> dict[str, Any]:
        return {
            "compression": self.compression,
            "compression_level": self.compression_level,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.writer_options(), "provenance": self.provenance}


def resolve_location(
    source: str | Path, filesystem: fs.FileSystem | None = None,
) -> StorageLocation:
    """Resolve local paths and cloud URIs without staging a local copy.

    Supplying a filesystem supports credentials, custom endpoints, and testing.
    In that case a URI is reduced to the filesystem's bucket/key path.
    """

    value = str(source)
    parsed = urlsplit(value)
    if filesystem is None:
        if not parsed.scheme:
            value = str(Path(value).absolute())
        resolved_fs, path = fs.FileSystem.from_uri(value)
        return StorageLocation(resolved_fs, path)
    if parsed.scheme:
        if parsed.query or parsed.fragment:
            raise ValueError("filesystem URIs must not contain a query or fragment")
        if parsed.scheme == "file":
            if parsed.netloc not in ("", "localhost"):
                raise ValueError("file URI must refer to the local host")
            path = unquote(parsed.path)
        else:
            path = parsed.netloc + unquote(parsed.path)
            if not parsed.netloc:
                path = path.lstrip("/")
    else:
        path = value
    return StorageLocation(filesystem, filesystem.normalize_path(path))


def hidden_sibling(path: str, suffix: str) -> str:
    """Name a temporary object beside ``path`` that dataset readers ignore.

    Arrow, Spark, Hive, and Trino skip names starting with ``.``, so an
    interrupted rewrite cannot leave an object that is read as table data.
    """

    parent, name = posixpath.split(path)
    return posixpath.join(parent, f".{name}.{suffix}")


def _is_hidden(relative_path: str) -> bool:
    return any(part.startswith((".", "_")) for part in relative_path.split("/"))


def inventory_parquet(
    source: str | Path, filesystem: fs.FileSystem | None = None,
) -> ParquetInventory:
    """List Parquet files, skipping hidden and ``_``-prefixed paths.

    Like Arrow and Spark dataset discovery, components such as ``_temporary``,
    ``_delta_log`` or ``.staging`` below the source root are not table data.
    """

    location = resolve_location(source, filesystem)
    info = location.filesystem.get_file_info(location.path)
    if info.type == fs.FileType.File:
        files = (info.path,)
        is_directory = False
    elif info.type == fs.FileType.Directory:
        selected = location.filesystem.get_file_info(
            fs.FileSelector(location.path, recursive=True)
        )
        files = tuple(sorted(
            item.path for item in selected
            if item.type == fs.FileType.File and item.path.lower().endswith(".parquet")
            and not _is_hidden(posixpath.relpath(item.path, location.path or "."))
        ))
        is_directory = True
    else:
        raise FileNotFoundError(f"source not found: {source}")
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {source}")
    return ParquetInventory(location.filesystem, location.path, files, is_directory)


def read_parquet_source(
    source: str | Path | ParquetInventory,
    filesystem: fs.FileSystem | None = None,
    **scanner_options: Any,
) -> pa.Table:
    """Read source rows directly through Arrow's filesystem interface.

    Explicit filenames preserve the physical file schema and do not add Hive
    partition columns. Partition directory names remain in the inventory.
    Every file must share one physical schema: Arrow would otherwise infer the
    first file's schema and silently drop columns that only later files have.
    """

    inventory = source if isinstance(source, ParquetInventory) else inventory_parquet(
        source, filesystem
    )
    files = list(inventory.files)
    if len(files) > 1:
        schema = pq.read_schema(files[0], filesystem=inventory.filesystem)
        for path in files[1:]:
            if not pq.read_schema(path, filesystem=inventory.filesystem).equals(
                schema, check_metadata=False
            ):
                raise ValueError(
                    f"Parquet files have different physical schemas: {files[0]} and "
                    f"{path}; read or rewrite each schema separately"
                )
    return ds.dataset(
        files, filesystem=inventory.filesystem, format="parquet"
    ).to_table(**scanner_options)


def open_input(source: str | Path, filesystem: fs.FileSystem | None = None) -> pa.NativeFile:
    location = resolve_location(source, filesystem)
    return location.filesystem.open_input_file(location.path)


def open_output(
    destination: str | Path, filesystem: fs.FileSystem | None = None,
    *, overwrite: bool = False,
) -> pa.NativeFile:
    """Open an output stream; remote object publication is not atomic here."""

    location = resolve_location(destination, filesystem)
    info = location.filesystem.get_file_info(location.path)
    if info.type != fs.FileType.NotFound and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    parent = posixpath.dirname(location.path)
    if parent:
        location.filesystem.create_dir(parent, recursive=True)
    return location.filesystem.open_output_stream(location.path)


def copy_file(
    source: str | Path, destination: str | Path,
    *, source_filesystem: fs.FileSystem | None = None,
    destination_filesystem: fs.FileSystem | None = None,
    overwrite: bool = False,
) -> None:
    """Copy one object without decoding it or creating a local staging file."""

    origin = resolve_location(source, source_filesystem)
    target = resolve_location(destination, destination_filesystem)
    if origin.filesystem.get_file_info(origin.path).type != fs.FileType.File:
        raise FileNotFoundError(f"source file not found: {source}")
    if target.filesystem.get_file_info(target.path).type != fs.FileType.NotFound and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    if origin.filesystem.equals(target.filesystem) and origin.path == target.path:
        raise ValueError("source and destination must be different files")
    parent = posixpath.dirname(target.path)
    if parent:
        target.filesystem.create_dir(parent, recursive=True)
    fs.copy_files(
        origin.path, target.path, source_filesystem=origin.filesystem,
        destination_filesystem=target.filesystem,
    )


def _codec_name(codec: str | None) -> str:
    value = "none" if codec is None else codec.lower()
    return {"uncompressed": "none", "lz4_raw": "lz4"}.get(value, value)


def supports_compression_level(codec: str | None) -> bool:
    normalized = _codec_name(codec)
    if normalized == "none":
        return False
    try:
        return pa.Codec.supports_compression_level(normalized)
    except (TypeError, ValueError):
        return False


def effective_compression_level(
    compression: str | Mapping[str, str] | None,
    compression_level: int | Mapping[str, int] | None,
) -> int | dict[str, int] | None:
    """Remove levels for codecs that do not accept an encoder level.

    With mixed codecs, even a scalar level must become a per-column mapping:
    Arrow otherwise tries to apply that level to Snappy/uncompressed columns.
    """

    if isinstance(compression, Mapping):
        levels = {
            name: _column_value(compression_level, name)
            for name, codec in compression.items()
            if supports_compression_level(codec)
        }
        return {name: level for name, level in levels.items() if level is not None} or None
    return compression_level if supports_compression_level(compression) else None


def _column_value(value: Any, column: str, *, default: Any = None) -> Any:
    return value.get(column, default) if isinstance(value, Mapping) else value


def _provenance_payload(metadata: Mapping[bytes, bytes] | None) -> Mapping[str, Any] | None:
    for key in (COMPRESSION_METADATA_KEY, b"oparq.writer"):
        raw = (metadata or {}).get(key)
        if raw is not None:
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(value, Mapping):
                return value
    return None


def inspect_compression(
    source: str | Path | ParquetInventory,
    filesystem: fs.FileSystem | None = None,
    *, manifest: Mapping[str, Any] | None = None,
) -> SourceCompression:
    """Read footer codecs and optional explicit encoder-level provenance.

    ``manifest`` may have ``compression`` / ``compression_level`` settings and
    a ``files`` mapping of relative filenames to overrides. A manifest level
    is trusted only if its declared codec matches the actual footer codec.
    """

    inventory = source if isinstance(source, ParquetInventory) else inventory_parquet(
        source, filesystem
    )
    codecs: dict[str, set[str]] = {}
    observed_levels: dict[str, list[tuple[bool, int | None, str]]] = {}
    for path in inventory.files:
        footer = pq.read_metadata(path, filesystem=inventory.filesystem)
        relative = inventory.relative_path(path)
        payload = _provenance_payload(footer.metadata)
        origin = "oparq metadata"
        if manifest is not None:
            payload = {**(payload or {}), **manifest}
            file_payload = manifest.get("files", {}).get(relative, {})
            if file_payload:
                payload.update(file_payload)
            origin = "manifest"
        file_codecs: dict[str, set[str]] = {}
        for group in range(footer.num_row_groups):
            row_group = footer.row_group(group)
            for index in range(row_group.num_columns):
                column = row_group.column(index)
                codec = _codec_name(column.compression)
                codecs.setdefault(column.path_in_schema, set()).add(codec)
                file_codecs.setdefault(column.path_in_schema, set()).add(codec)
        # Empty files still have a schema, but no codec in their absent chunks.
        for column, actual in file_codecs.items():
            for codec in actual:
                if not supports_compression_level(codec):
                    continue
                known = False
                level = None
                if payload is not None and "compression_level" in payload:
                    declared = _column_value(payload.get("compression"), column)
                    levels = payload["compression_level"]
                    default_level = payload.get("default_compression_level")
                    level_present = (
                        not isinstance(levels, Mapping) or column in levels
                        or "default_compression_level" in payload
                    )
                    if declared is not None and _codec_name(declared) == codec and level_present:
                        level = _column_value(levels, column, default=default_level)
                        if level is None or (isinstance(level, int) and not isinstance(level, bool)):
                            known = True
                observed_levels.setdefault(column, []).append((known, level, origin))
    levels: dict[str, int | None] = {}
    level_sources: dict[str, str] = {}
    conflicts: list[str] = []
    for column, observations in observed_levels.items():
        known_values = {level for known, level, _ in observations if known}
        if len(known_values) > 1:
            conflicts.append(column)
        elif observations and all(known for known, _, _ in observations):
            levels[column] = observations[0][1]
            level_sources[column] = ", ".join(sorted({origin for _, _, origin in observations}))
    return SourceCompression(
        {name: tuple(sorted(values)) for name, values in sorted(codecs.items())},
        levels, level_sources, tuple(sorted(conflicts)),
    )


def resolve_compression(
    source: SourceCompression,
    *, compression: str | Mapping[str, str] | None = "preserve",
    compression_level: int | Mapping[str, int | None] | None | str = "preserve",
) -> CompressionSettings:
    """Resolve writer settings without silently changing source level.

    An explicit ``compression_level=None`` intentionally selects the codec
    default. With ``'preserve'``, unknown or conflicting levels require the
    caller to supply a level (or deliberately select the default).
    """

    if compression == "preserve":
        conflicts = [name for name, values in source.codecs.items() if len(values) != 1]
        if conflicts:
            raise ValueError(
                "source codecs differ within these columns; choose an explicit output "
                "codec or rewrite files separately: " + ", ".join(conflicts)
            )
        if not source.codecs:
            raise ValueError("empty source has no footer codec; choose compression explicitly")
        column_codecs = {name: values[0] for name, values in source.codecs.items()}
        provenance = "source footer"
    elif isinstance(compression, Mapping):
        missing = set(source.codecs) - set(compression)
        if missing:
            raise ValueError("compression mapping is missing columns: " + ", ".join(sorted(missing)))
        column_codecs = {name: _codec_name(compression[name]) for name in source.codecs}
        provenance = "explicit output settings"
    else:
        column_codecs = {name: _codec_name(compression) for name in source.codecs}
        provenance = "explicit output settings"
    all_codecs = set(column_codecs.values())
    writer_codec: str | dict[str, str] = (
        next(iter(all_codecs)) if len(all_codecs) == 1 else column_codecs
    )
    if not column_codecs:
        writer_codec = _codec_name(compression) if not isinstance(compression, Mapping) else dict(compression)
    levels: dict[str, int | None] = {}
    unknown: list[str] = []
    for name, codec in column_codecs.items():
        if not supports_compression_level(codec):
            continue
        if compression_level == "preserve":
            if source.codecs.get(name) == (codec,) and name in source.levels:
                levels[name] = source.levels[name]
            else:
                unknown.append(name)
        elif isinstance(compression_level, Mapping):
            if name not in compression_level:
                unknown.append(name)
            else:
                levels[name] = compression_level[name]
        elif compression_level is None or (isinstance(compression_level, int) and not isinstance(compression_level, bool)):
            levels[name] = compression_level
        else:
            raise ValueError("compression_level must be an integer, None, a column mapping, or 'preserve'")
    if unknown:
        raise ValueError(
            "Parquet metadata does not record compression levels. Supply "
            "compression_level explicitly (or None to intentionally use the codec "
            "default), or provide level provenance in a manifest. Unknown columns: "
            + ", ".join(unknown)
        )
    for name, level in levels.items():
        if level is not None:
            if not isinstance(level, int) or isinstance(level, bool):
                raise ValueError(f"invalid compression level for {name}: {level!r}")
            pa.Codec(column_codecs[name], compression_level=level)
    unique_levels = set(levels.values())
    writer_level: int | None | dict[str, int] = (
        next(iter(unique_levels)) if len(unique_levels) == 1 else
        {name: level for name, level in levels.items() if level is not None}
    )
    if not levels:
        writer_level = None
    if compression_level == "preserve" and levels:
        provenance += "; level from recorded source provenance"
    elif levels:
        provenance += "; explicit level"
    writer_level = effective_compression_level(writer_codec, writer_level)
    return CompressionSettings(writer_codec, writer_level, provenance)


# Parquet's common default (parquet-mr, Spark). Arrow folds each filter down
# to the distinct values actually written at this false-positive target.
_BLOOM_FPP = 0.01
# Rebuilding needs footer bloom offsets (PyArrow 25+) and bloom writing
# (PyArrow 24+). Older releases still rebuild the page index.
REBUILDS_BLOOM_FILTERS = (
    hasattr(pq.ColumnChunkMetaData, "bloom_filter_offset")
    and "bloom_filter_options" in inspect.signature(pq.write_table).parameters
)


def _bloom_ndv(bits: float) -> int:
    """Declared NDV that makes Arrow start from ``bits``, a power of two.

    The declared NDV only bounds the filter Arrow allocates before folding,
    so a rebuilt filter never exceeds the corresponding source filter.
    """

    target = max(32 * 8, 2 ** round(math.log2(max(bits, 1.0))))
    scale = -math.log(1 - _BLOOM_FPP ** 0.125)
    return max(1, math.floor(target * scale / 8 * (1 - 1e-9)))


def source_index_options(
    footers: Iterable[pq.FileMetaData], *, output_group_rows: int,
) -> dict[str, Any]:
    """Writer options that rebuild the sources' page index and bloom filters.

    Both structures are computed by the writer from the rows it writes, so they
    remain correct after reordering. Bloom filters are rebuilt on the same
    columns with a 1% false-positive target; each column's source bits per row
    bound the rebuilt size, including when row groups are resized.
    """

    page_index = False
    bits: dict[str, float] = {}
    rows: dict[str, int] = {}
    unsized: set[str] = set()
    for footer in footers:
        for group in range(footer.num_row_groups):
            row_group = footer.row_group(group)
            for index in range(row_group.num_columns):
                column = row_group.column(index)
                page_index = page_index or bool(
                    getattr(column, "has_column_index", False)
                    or getattr(column, "has_offset_index", False)
                )
                if getattr(column, "bloom_filter_offset", None) is None:
                    continue
                path = column.path_in_schema
                length = getattr(column, "bloom_filter_length", None)
                if not length or length < 32:
                    unsized.add(path)
                    continue
                # The recorded length includes a small header; the bitset is
                # the largest power of two it contains.
                bits[path] = bits.get(path, 0.0) + 8 * 2 ** math.floor(math.log2(length))
                rows[path] = rows.get(path, 0) + row_group.num_rows
    options: dict[str, Any] = {}
    if page_index:
        options["write_page_index"] = True
    blooms = {
        path: {"ndv": _bloom_ndv(total * output_group_rows / max(rows[path], 1)),
               "fpp": _BLOOM_FPP}
        for path, total in bits.items()
    }
    # Without a recorded size, use Arrow's documented default: NDV = rows.
    blooms.update({path: {"ndv": max(1, output_group_rows), "fpp": _BLOOM_FPP}
                   for path in unsized.difference(blooms)})
    if blooms and REBUILDS_BLOOM_FILTERS:
        options["bloom_filter_options"] = blooms
    return options


def compression_metadata(
    compression: str | Mapping[str, str] | None,
    compression_level: int | Mapping[str, int] | None,
) -> bytes:
    """Record effective levels so subsequent rewrites can preserve them.

    An encoder-default level is resolved now using the installed Arrow codec,
    avoiding an ambiguous ``None`` when rewriting in a different environment.
    """

    if isinstance(compression, Mapping):
        codecs = {name: _codec_name(codec) for name, codec in compression.items()}
        levels = {}
        for name, codec in codecs.items():
            if supports_compression_level(codec):
                level = _column_value(compression_level, name)
                levels[name] = pa.Codec.default_compression_level(codec) if level is None else level
        recorded_codec: str | dict[str, str] = codecs
        recorded_level: int | None | dict[str, int] = levels
    else:
        recorded_codec = _codec_name(compression)
        recorded_level = compression_level
        if supports_compression_level(recorded_codec) and compression_level is None:
            recorded_level = pa.Codec.default_compression_level(recorded_codec)
    payload = {"version": 1, "compression": recorded_codec, "compression_level": recorded_level}
    if not isinstance(compression, Mapping) and isinstance(compression_level, Mapping):
        if supports_compression_level(recorded_codec):
            payload["default_compression_level"] = pa.Codec.default_compression_level(recorded_codec)
    return json.dumps(
        payload,
        sort_keys=True, separators=(",", ":"),
    ).encode()
