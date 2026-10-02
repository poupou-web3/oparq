# oparq — smaller Parquet files for S3 and GCS

`oparq` is a Python library for reducing Parquet storage bytes by choosing
compression-friendly row orders. Learn sort keys from samples once, save the
plan, and reuse it across a local, Amazon S3, or Google Cloud Storage dataset.
PyArrow writes the files; optional DuckDB sorting can spill to disk.

It changes physical row order, not the logical row values. Compression gains
depend on the data and existing clustering; smaller files and faster queries
are not guaranteed.

## Install

The library is MIT-licensed and requires Python 3.12+ and PyArrow 16+.

```bash
pip install oparq
pip install "oparq[duckdb]"  # optional larger-than-memory DuckDB sorting
```

## Write an Arrow table

```python
import pyarrow as pa
import oparq

table = pa.table({
    "account": ["b", "a", "b", "a"],
    "payload": ["repeat-b", "repeat-a", "repeat-b", "repeat-a"],
})
result = oparq.write(table, "optimized.parquet", algorithm="auto")
print(result.plan.sort_keys, result.file_size)
print(result.planning_seconds, result.permutation_seconds,
      result.gathering_seconds, result.write_seconds)
```

New-table writes use Zstandard's Arrow codec default, currently level 1,
with dictionaries and statistics enabled. `auto` uses bounded `codec_fast`
key selection. Use `algorithm="none"` with the same writer settings for a
controlled baseline.

## Learn once, rewrite a bucket

```python
import json
from pathlib import Path
import oparq

# An exporter manifest must record the actual source codec/level.
source_settings = json.loads(Path("source-compression.json").read_text())
plan = oparq.fit_dataset(
    "s3://source-bucket/events",
    algorithms=("codec_fast", "portfolio"),
    prefix=("event_date",),  # Optional physical leading key; discover its suffix.
    compression_manifest=source_settings,
)
plan.save("events-order.json")

result = oparq.rewrite_dataset(
    "s3://source-bucket/events",
    "gs://destination-bucket/events",
    plan=oparq.RewritePlan.load("events-order.json"),
    compression_manifest=source_settings,
    engine="duckdb",  # Install the extra; "arrow" is the default.
    memory_limit="6GB",
)
print(result.rewritten_files, result.copied_files, result.skipped_files)
```

The saved plan skips future profiling and algorithm selection. Rewrites retain
Parquet relative paths, Hive partition directories, and file boundaries,
sorting within each physical file. They do not coalesce partitions, copy
arbitrary sidecars, or provide a bucket-wide transaction.
Prefix keys must be physical columns; virtual Hive columns are not injected.

## Defaults that protect your data

- File rewrites preserve source codecs and known encoder levels. Parquet
  footers do **not** store compression levels: unknown ZSTD levels require
  trusted provenance or an explicit output level, never a silent jump to 9.
- With preserved settings, no mandatory prefix, and
  `skip_unchanged=True`, no-key plans copy/skip the original. A sorted output
  that is not smaller is discarded before publication. The attempted work
  still costs time.
- Explicit codec/level/layout changes and required prefixes are honored even
  if files grow. Existing destinations are protected unless
  `overwrite=True`.
- A source page index and bloom filters (PyArrow 25+) are rebuilt for the
  new row order, so rewritten files keep the same pruning structures.
- Remote reads avoid a staging copy of source files, but data still travels
  through the worker and DuckDB may spill locally. Run near your buckets.
- Nested and detected JSON columns are not automatic sort candidates but are
  carried through sorting. Parquet already omits null-value payloads.
- Sorting can improve or weaken query clustering. Statistics and advertised
  natural sort metadata remain truthful; no universal pruning win is promised.

## Measured evidence

The September 30, 2026 full-corpus comparison processed every row of 21
datasets: 967,938,981 rows in 986 source files, sorting each dataset
globally across its files. Under controlled identical writer settings,
`portfolio` saved 16.28% in aggregate; the best measured method per dataset
saved 18.28%. That best mixture was selected **after** full results, not
predicted by `auto`. Some sample-chosen orders regressed sharply.

`rewrite_dataset` sorts within each file, so its savings differ from that
global sort. The separate Arrow/DuckDB comparison applied the same
`portfolio` keys file by file to all 26,015,956 rows in 40 stock, Hits, and
Solana files, with exact per-file value/order checks: stock saved 20.61% and
Solana 29.64% (33.54% and 43.00% when sorted globally). Arrow was faster in
that run; DuckDB is retained for bounded-memory execution, not a
demonstrated universal speed advantage.

The source repository contains the quickstart and API reference in `docs/`,
and benchmark runners plus machine-readable results in `benchmarks/`.
The measurements above are from
historical development runs, not a claim that the original inputs are public.
