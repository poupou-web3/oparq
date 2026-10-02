# Compression settings and source provenance

Parquet stores a codec per column chunk, but it does not store the encoder's
compression level. A ZSTD footer does not tell you whether the exporter used
level 1, 9, or another level. Snappy has no encoder-level parameter.

## New table versus existing-file defaults

| Operation | Codec default | Level default |
| --- | --- | --- |
| `write(table, ...)` without a learned plan | ZSTD | Arrow codec default; currently 1 |
| `write(table, ..., plan=rewrite_plan)` | Plan's training setting | Plan's setting unless overridden |
| `rewrite` / `rewrite_dataset` | Preserve source footer codecs | Preserve known source level; reject unknown level |
| `fit_dataset` | Preserve source footer codecs | Require source provenance or explicit level |

`compression_level=None` deliberately selects the installed codec default;
it does not mean "preserve". `fit` resolves that default immediately and
stores the effective concrete level in its learned plan (currently ZSTD1),
so later application does not silently use a different encoder default.
A manually constructed `RewritePlan` with level `None` still means the
installed default rather than a fixed integer.

## Supply trusted exporter settings

Use this only if the exporter configuration actually establishes level 1:

```python
source_settings = {"compression": "zstd", "compression_level": 1}
oparq.rewrite(
    "input.parquet", "optimized.parquet",
    compression_manifest=source_settings,
)
```

A tree manifest can include file-specific overrides:

```json
{
  "compression": "zstd",
  "compression_level": 1,
  "files": {
    "year=2026/part-00002.parquet": {
      "compression": "zstd",
      "compression_level": 3
    }
  }
}
```

The declared codec must match the actual footer codec before a manifest
level is trusted. Dictionary mappings can specify codec/level by leaf column.
Across a mixed-codec tree, fit compatible cohorts deliberately; differing
codecs within one column across files require an explicit training policy
or separate plans. Per-file rewrite preservation can retain each file's
own codecs.

## Intentionally change settings

```python
oparq.rewrite(
    "input.parquet", "zstd9.parquet",
    compression="zstd", compression_level=9,
)
```

This is an explicit output change, so it is honored even if the result grows.
To intentionally use the source codec's installed default, supply
`compression_level=None`. CLI forms are `--compression-level 9` and
`--compression-level default`.

File rewrites also rebuild a source page index and, with PyArrow 25+,
bloom filters (same columns, 1% false-positive target, no larger than the
source filters) for the new row order. Explicit `write_page_index` or `bloom_filter_options`
values override this.

New oparq files record effective codec/level provenance in
`oparq.compression` metadata. Later preservation can use this metadata.
Byte-copy and in-place no-op decisions do not need an unknown level because
they do not invoke an encoder.

## Nulls are already compressed by Parquet

Parquet represents nulls using definition levels and omits null-value
payloads. This happens automatically, including when no row sorting is used.
The profiler counts null as a distinct bucket because clustering it can
improve definition-level runs. It is not eliminating an otherwise stored
full null payload. See the [Parquet null specification](https://parquet.apache.org/docs/file-format/nulls/).

## Storage and query trade-offs

Higher levels can spend more CPU for fewer bytes; row sorting can trade
existing query clustering for compression. Keep codec, level, dictionaries,
statistics, row-group geometry, writer, and input rows identical when measuring
ordering. Comparing to the downloaded file also includes exporter/layout
differences. Smaller files do not guarantee faster queries or useful pruning
for every filter. See [benchmark evidence](benchmarks.md).
