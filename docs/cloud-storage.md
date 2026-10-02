# S3 and GCS bucket rewrites

`oparq` reads and writes cloud URIs through Arrow filesystems without first
staging copies of the source Parquet files. It does not eliminate network
transfer: source bytes still enter the worker, rows are decoded and sorted,
and encoded output is sent to the destination. DuckDB can also spill
temporary sort data locally.

For a bucket-wide migration, run the worker near the buckets rather than
routing the dataset through a laptop. Installing an embedded engine does
not relocate computation or make a sorting rewrite into a server-side
object copy. Account for network/egress, requests, temporary storage, and
compute as well as saved output bytes.

## Learn keys from representative files

Freeze the source snapshot while learning and rewriting. Fit one plan per
compatible physical schema. An exporter manifest must record actual encoder
settings; do not guess a level from the Parquet footer.

```python
import json
from pathlib import Path
import oparq

source_settings = json.loads(Path("source-compression.json").read_text())
plan = oparq.fit_dataset(
    "s3://source-bucket/events",
    algorithms=("codec_fast", "portfolio", "weighted"),
    sample_rows=250_000,
    sample_files=8,
    compression_manifest=source_settings,
)
plan.save("events-order.json")
```

The sample is deterministic leading-row quotas from files spread across
the inventory, not a random sample of every row. Source footer inspection
still covers the inventory. If this is not representative, construct your
own sample table and call `fit`. `fit` compares row-order algorithms while
holding codec and writer settings fixed; it does not auto-select a codec.

Add `prefix=("event_date", "tenant_id")` when those natural ascending keys
must lead the order. The planner discovers only the suffix. That requirement
is honored even when it makes a file larger.
Prefix keys must exist in the physical schema, not only in Hive directory names.

## Reuse the plan across the tree

```python
result = oparq.rewrite_dataset(
    "s3://source-bucket/events",
    "gs://destination-bucket/events",
    plan=oparq.RewritePlan.load("events-order.json"),
    compression_manifest=source_settings,
    engine="duckdb",
    memory_limit="6GB",
    max_temp_directory_size="18GB",
    temp_directory="/tmp",
)
print(result.rewritten_files, result.copied_files, result.skipped_files)
```

Use the optional DuckDB extra for this example. The default `engine="arrow"`
loads and sorts each physical file in memory. DuckDB consumes Arrow batches,
supports natural fixed-key plans, and may spill when sorting. It is not
universally faster; provision enough worker memory, spill space, and network.

## What partition preservation means

An input such as `year=2026/month=08/part-00001.parquet` keeps the same
relative path under the destination. File boundaries remain unchanged;
sorting is global within each file, not across an entire partition or
dataset. Partition directories are not injected into the physical schema.

Only inventoried Parquet files are migrated. Like Arrow and Spark dataset
discovery, paths below the source root with a component starting with `.`
or `_` (such as `_temporary/`, `_delta_log/`, or hidden files) are skipped.
Aggregate `_metadata` files with stale offsets are not copied, and arbitrary
non-Parquet sidecars are not mirrored. This is not an atomic bucket transaction: earlier files can
be published before a later runtime failure. Schemas, compression provenance,
and destination collisions are preflighted, but source immutability and
operational recovery remain the caller's responsibilities.

Destinations are protected by default. `overwrite=True` permits replacing
existing target files; writing to a different bucket retains the source.
A target nested inside the source tree is rejected.

## Avoid paying storage for unsuccessful candidates

With preserved codec/level settings, no explicit writer-layout changes,
no mandatory prefix, and `skip_unchanged=True`:

- A fixed no-key plan copies the original bytes to a new destination or
  skips an in-place rewrite without decoding the file.
- A sorted candidate is published only if its actual bytes are smaller
  than the original file. Otherwise it is discarded and the source is
  copied or skipped. The attempted plan/sort/write cost has already been paid.

Explicit codec, level, layout, or prefix requirements bypass this storage
guard. `skip_unchanged=False` forces re-encoding. Inspect each
`WriteResult.action` and plan note; do not assume all files were sorted.

## Credentials and explicit filesystems

Arrow's S3/GCS filesystem implementations provide native credentials.
You can pass `filesystem=` to `fit_dataset` and `read_parquet`, and
`source_filesystem=` / `destination_filesystem=` to `rewrite_dataset`.
Follow your cloud's least-privilege policy; no credentials are embedded in
plans or examples. Actual cloud integration requires credentials and a
destination selected by the caller.

Primary references: [Arrow filesystems](https://arrow.apache.org/docs/python/filesystems.html),
[DuckDB out-of-core execution](https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads#larger-than-memory-workloads-out-of-core-processing),
and [DuckDB S3 export](https://duckdb.org/docs/current/guides/network_cloud_storage/s3_export).
