# S3 and GCS bucket rewrites

`oparq` reads and writes cloud URIs through Arrow filesystems without
staging source Parquet files on local disk. It does not eliminate network
transfer: source bytes still enter the worker (prefetching holds a few
compressed files in memory), rows are decoded and sorted, and encoded output
is sent to the destination. DuckDB can also spill temporary sort data locally.

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
    sample_files=32,
    compression_manifest=source_settings,
)
plan.save("events-order.json")
for check in plan.validation:
    print(check["path"], f"{check['savings']:+.1%}")
```

The sample is consecutive-row windows from 32 files spread evenly across the
sorted inventory. Each window starts at a seeded random row group, not only
at a file's first row, and reads only the window's pages. Codec provenance
is checked on 256 spread footers plus the training files, read concurrently
(`footer_files=None` inspects all). Afterwards, held-out files that were not
used for training are written in input order and in plan order; a plan
whose held-out savings are much lower than expected, or negative, did not
generalize. If the sample is still not representative, construct your own
table and call `fit`. `fit` compares row-order algorithms while holding codec
and writer settings fixed; it does not auto-select a codec.

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

## Compact each partition into fewer, sorted files

`rewrite_dataset` normally sorts within each file, keeping file boundaries.
Sorting across files groups similar rows from the whole partition, which
compresses further, and it replaces many small files with a few large ones.
`scope="partition"` sorts all files of each leaf directory together and
writes them as `part-00000.parquet`, `part-00001.parquet`, ... of about
`target_file_size` (default 512 MiB):

```python
result = oparq.rewrite_dataset(
    "s3://source-bucket/events", "s3://compacted-bucket/events",
    plan=oparq.RewritePlan.load("events-order.json"),
    compression_manifest=source_settings,
    scope="partition", target_file_size="512MB", workers=0,
)
for partition in result.partitions:
    print(partition.partition, len(partition.sources), "->", len(partition.outputs))
```

- **A new tree.** Replacing a set of files cannot be atomic on an object
  store, so the destination must not be, or contain, the source tree. Write
  to a new prefix, check it, then point readers at it.
- **Hive directories are kept.** Each leaf directory is one partition, and
  its files are merged only with each other. Every file in a partition must
  share one physical schema and one codec/level.
- **Order.** Ties keep file order, then row order. Each output file holds a
  consecutive range of the partition's order and advertises its sort keys.
- **Sizes: a target and a cap.** Files aim at `target_file_size`: output
  size is learned while writing and files close at the row-group boundary
  nearest an even share, so the last file absorbs estimation error and can
  be smaller. `max_file_size` (default 1.25 × the target) is a cap: before
  each row group is written, the file's size with it is projected, counting
  bloom filters and the footer that Parquet writes at close; a group whose
  size is uncertain is encoded once in memory to measure it, and the file
  closes first if it does not fit. A row group too large for any file is
  split (the partition's `plan.note` says so). A file can exceed the cap
  only if a row group is more than 1.5 times denser than the recent groups
  and its own end slices, or a row group of 1,024 rows (or the configured
  `row_group_size`, if smaller) is larger than the cap. On the
  benchmark data (targets 32–512 MB, two row-group sizes) the largest file
  was 0.98 × the cap. Raise `max_file_size` rather than accept smaller row
  groups; measuring costs time only for groups near a file boundary.
- **Guards.** With preserved settings, a partition whose result is not
  smaller keeps its original files (copied with their names). A single file
  with no keys is copied. `skip_unchanged=False` always compacts.
- **Memory.** One worker handles one partition. The Arrow engine decodes the
  whole partition, so budget about three times its decoded size; use
  `engine="duckdb"` (with `memory_limit`) for partitions larger than memory.
  Encoding a partition uses one core, so a single very large partition is
  not parallelized; many partitions are.
- **Existing data.** A destination partition that already holds Parquet
  files is refused unless `overwrite=True`, which replaces that partition's
  data files (others in the directory are deleted once the new set is
  published).
- **Resume.** Checkpoints work per partition. A `publishing` marker names
  the outputs before they are moved into place; after a crash, a rerun
  replaces that partial set without needing `overwrite`.

## Use every core of the worker

Decoding, sorting and Parquet encoding each use about one core per file, so a
single-file-at-a-time rewrite leaves most of a large VM idle. `workers`
rewrites several files at once on threads (Arrow releases the GIL):

```python
result = oparq.rewrite_dataset(
    "s3://source-bucket/events", "s3://destination-bucket/events",
    plan=oparq.RewritePlan.load("events-order.json"),
    compression_manifest=source_settings,
    workers=0,              # one per CPU
    memory_budget="48GB",   # default: 70% of physical or container memory
)
```

The memory budget bounds the estimated decoded size of files in flight. A
Parquet footer cannot predict that size (dictionary strings can decode to
many times their page size), so the first file of each schema runs alone and
its decoded bytes per row become the estimate; the live Arrow allocation is
also checked before each new file starts. A single file estimated above the
whole budget runs alone with a warning; consider `engine="duckdb"` for it.

For remote sources, upcoming files are downloaded while other files are
being decoded (`prefetch`, default `max(2, workers)` files), and Arrow's I/O
thread pool is raised for the run. Output bytes do not depend on `workers`,
`prefetch` or `shard`.

## Split a bucket across machines

`shard="i/N"` (CLI `--shard i/N`) selects files by a stable hash of their
relative path. Shards are disjoint and cover every file, so N machines given
`0/N` … `N-1/N` never process the same file and need no coordinator:

```bash
# On machine i of N, close to both buckets:
oparq rewrite s3://source-bucket/events s3://destination-bucket/events \
  --plan events-order.json --compression-manifest source-settings.json \
  --workers 0 --shard "$INDEX/$COUNT" --progress
```

Keep the plan, manifest and options identical on every machine. A batch job
array that passes its index works the same way.

## Resume after an interruption

Each finished file writes a small marker under
`<destination>/_oparq/checkpoint/` (readers skip `_` paths). Rerun the same
command after a crash or a failed file: files whose source size and
modification time, plan keys, writer settings and output size all still
match are reported as resumed without being read again, and hidden
temporaries the interrupted run left beside its outputs are removed. The
retry unit is one file, and every shard can resume independently.

A file whose source or plan changed is processed again; if its old output
exists, pass `overwrite=True` (`--overwrite`), which then replaces only
outputs not covered by a matching marker. A file published just before a
crash, but before its marker, needs the same flag. Use `checkpoint=False`
(`--no-checkpoint`) to write no markers, or `checkpoint="s3://state/run-1"`
to keep them elsewhere. Markers prove what this tool wrote; they are not a
bucket-wide transaction, so keep the source snapshot frozen until all shards
finish.

## Where to run it

Run workers in the same region as the buckets (a VM, or a batch job array):
every byte is downloaded, decoded, sorted, re-encoded and uploaded. Size a
machine by cores for throughput and by memory for concurrency: roughly
three times the largest file's decoded size per concurrent file. Cross-cloud
or cross-region rewrites pay egress on the full source size.

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
