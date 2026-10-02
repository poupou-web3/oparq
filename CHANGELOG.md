# Changelog

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
