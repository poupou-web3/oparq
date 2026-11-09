# Changelog

## Unreleased

- Rewrite dataset files concurrently: `workers=` / `--workers` runs files on a thread pool within a `memory_budget` that learns decoded bytes per row and checks live Arrow allocations. Outputs are byte-identical to serial runs.
- Split a dataset across machines without overlap with `shard="i/N"` / `--shard`, using a stable hash of each relative path.
- Resume interrupted rewrites from per-file checkpoint markers in `<destination>/_oparq/checkpoint` (on by default; `checkpoint=False` / `--no-checkpoint`). A marker is honored only when the source, plan keys, writer settings and output still match; leftover hidden temporaries are removed.
- Prefetch remote source files ahead of the workers that decode them (`prefetch=`), fetching larger files as concurrent ranged reads.
- Read Parquet footers concurrently in dataset preflight, `fit_dataset`, `inspect_compression`, directory reads and consolidation; preflight lists the destination once instead of checking each output path, and rewrites reuse footers instead of reading them again.
- `fit_dataset` inspects 256 spread footers plus the training files by default (`footer_files=None` for all) and samples 32 files instead of 8.
- `fit_dataset` and `sample_dataset` take windows starting at a seeded random row group (`sample_method="row_groups"`) instead of only each file's first rows; `"head"` keeps the previous behavior. Plans fitted with defaults may therefore differ from 0.3.0.
- `fit_dataset` checks the plan on held-out files and stores per-file results in `RewritePlan.validation`; `oparq fit` prints a summary.
- `rewrite_dataset` adds `progress`, `io_threads`, `DatasetRewriteResult.resumed_files`, `removed_temporaries`, `shard`, and `WriteResult.resumed`.

## 0.3.0 — 2026-10-02

- Learn and serialize a fixed row-order plan, then apply it without repeating planning.
- Preserve file-tree paths and partition directories when rewriting local, S3, and GCS datasets.
- Preserve source codecs and known encoder levels; reject guesses about unknown source levels.
- Retain original bytes for no-sort plans and reject non-improving storage-only rewrites.
- Support mandatory leading keys while discovering compression-friendly suffix keys.
- Add an optional DuckDB external-sort backend; retain Arrow as the default.
- Separate native planning, permutation, gathering, and writing timings.
- Remove algorithm aliases; accept only documented canonical strategy names.
- Reject malformed Arrow strings at the DuckDB import boundary to prevent byte loss.
- Validate full-row compression results for 21 datasets and exact engine outputs for three datasets.
- Add distribution checks, documentation, and benchmark-results publication tools.
- Reject differing file schemas and unrecoverable Hive partition values instead of silently dropping columns when reading or consolidating a directory.
- Skip `.`- and `_`-prefixed paths during inventory, and give temporary and candidate outputs hidden names so interrupted rewrites cannot add readable data files.
- Create local outputs with the process umask rather than owner-only permissions.
- Leave tables with columns Arrow cannot reorder (such as `string_view`) unsearched and unsorted instead of failing; a required prefix on them is an error.
- Reject DuckDB-incompatible types (intervals, float16, decimal256, null, list views) during preflight instead of mid-rewrite.
- Skip the `runs` search when no key can be selected; report CLI user errors without tracebacks.
- Add the `all` algorithm, which compares every planner's proposal, and `full=True` / `--full` planning on every row.
- Rebuild a source page index and bloom filters when rewriting files.
- Let `oparq plan` and `oparq benchmark` compare orders when the source level is unknown, using one stated codec setting for every order.
- Support PyArrow 16+ (18+ on Python 3.13); CI also tests the oldest supported releases.

This is an early release. Cloud access is covered by injected filesystem tests,
not an integration run against a production account. Heuristic sort selection
does not promise the smallest possible file.
