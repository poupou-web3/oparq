# Python API reference

This reference describes oparq 0.3.0. Inputs are `pyarrow.Table` or
`pyarrow.RecordBatch` unless a function accepts a file/dataset source.
Paths can be local strings/`Path` objects or supported remote URIs.

## `write(value, destination, **options) -> WriteResult`

Plan, stably reorder, and write an in-memory table using PyArrow. A saved
`RewritePlan` skips profiling/selection; a table-specific `SortPlan` is also
accepted. Without a learned plan, new-table defaults are ZSTD with the Arrow
codec-default level, currently 1.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `algorithm` | `"auto"` | Canonical key-selection algorithm |
| `plan` | `None` | Fixed `RewritePlan` or `SortPlan`; overrides planning |
| `prefix` | `()` | Required natural ascending leading physical columns |
| `include` / `exclude` | `None` / `()` | Limit/remove automatic candidates |
| `sample_rows` | `250_000` | Profile rows; `None` profiles the full table |
| `trial_sample_rows` | `250_000` | Maximum rows used by codec trials |
| `run_sample_rows` | `50_000` | Run-proxy sample budget |
| `max_sort_columns` | `8` | Maximum total keys including prefix |
| `candidate_pool_size` | `12` | Broader planner candidate pool |
| `fast_candidate_count` | `4` | Shortlisted codec-fast candidates/step |
| `max_trial_evaluations` | `24` | Codec-fast encode budget |
| `min_trial_improvement` / `min_run_improvement` | `0.005` | Minimum planner-step benefit |
| `detect_json` | `True` | Heuristic JSON-string candidate exclusion |
| `null_placement` | `"at_end"` | `"at_start"` or `"at_end"` |
| `sort_backend` | `"auto"` | Arrow-native or naturally ranked key implementation; also `"arrow"` / `"rank"` |
| `full` | `False` | Plan with every row: no profile, run, or trial sampling |
| `gather_threads` | `0` | Automatic, at most four workers; `1` is serial |
| `stream_sort` | `True` | Gather/write sorted row groups rather than a second full table |
| `compression` / `compression_level` | New-table defaults or learned-plan settings | Explicit codec/level override; mappings are accepted by the writer path |
| `row_group_size` | Arrow default | `1_048_576` rows by default, capped at Arrow's maximum |
| `use_dictionary` / `write_statistics` | `True` / `True` | Writer settings; can select columns |
| `write_page_index` | `False` | PyArrow writer option |
| `overwrite` | `False` | Reject existing destinations unless enabled |
| `filesystem` | Inferred | Explicit Arrow output filesystem |

Additional `**parquet_options` are forwarded to PyArrow. A conflicting prefix
with a supplied plan is rejected. `write` does not compare against an original
source file: the non-growth safeguard belongs to file rewrite operations.
A destination without a `.parquet` suffix is treated as a directory:
`write(table, "out")` creates `out/part-00000.parquet`.

## `fit(value, **options) -> RewritePlan`

Compare strategies on supplied sample rows, always including `none` or the
mandatory-prefix baseline. Return the smallest measured plan if it exceeds
`min_improvement=0.005`; otherwise retain the baseline.

Defaults: `algorithms=("codec_fast", "portfolio")`, `prefix=()`,
`compression="zstd"`, `compression_level=None`, and `row_group_size=None`.
Planning controls accepted by `write` can be passed through. Codec and writer
settings are held fixed. This is not an automatic codec search.

## `fit_dataset(source, **options) -> RewritePlan`

Read footer codec/provenance information from a spread sample of files,
take consecutive-row windows from deterministic files spread across the
source, call `fit`, then check the plan on held-out files.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `sample_rows` | `250_000` | Total training rows, split evenly across training files |
| `sample_files` | `32` | Training files spread evenly across the sorted inventory |
| `sample_method` | `"row_groups"` | Window start: a seeded random row group per file (weighted by rows), or `"head"` (each file's first row, the 0.3.0 behavior) |
| `seed` | `0` | Sampling seed; equal inputs and seed give the same sample |
| `footer_files` | `256` | Footers inspected for codec provenance, plus every training file; `None` inspects all |
| `validation_files` | `4` | Held-out files (never trained on) checked after fitting; `0` disables |
| `validation_rows` | `100_000` | Rows per held-out window |
| `compression` / `compression_level` | `"preserve"` / `"preserve"` | Codec used to compare orders |

Accepts `filesystem`, `compression_manifest`, `algorithms`, `prefix`, and
other fitting options. Unknown source levels on a level-sensitive codec
require explicit settings or trusted provenance. Footers are read
concurrently. Sampling footers only decides the codec used to compare orders
here; every rewrite still checks each file it re-encodes.

Windows are consecutive rows, so they keep each file's existing clustering;
only the window's pages are read, not whole files. Held-out results are stored
in `RewritePlan.validation` as `path`, `rows`, `baseline_bytes`, `plan_bytes`
and `savings` per file. They compare the plan with input order on windows
written with the plan's codec. Windows are smaller than files, so savings
usually differ from whole-file savings: use them to spot plans that do not
generalize, not as a forecast. When the inventory has no more files than
`sample_files`, nothing is held out and `validation` is empty.

`sample_dataset(source, filesystem=None, sample_rows=250_000,
sample_files=32, sample_method="row_groups", seed=0) -> Table` exposes the
same sampling separately. `validate_plan(windows, plan, row_group_size=None)`
runs the held-out comparison on `(name, table)` pairs you provide.

## `RewritePlan`

A row-count-independent artifact storing algorithm, fixed keys/prefix,
value/null order, learned schema, sample evaluation evidence, and training
codec settings. New artifacts include a serialized Arrow schema so structural
type/nullability expectations can be checked without relying only on display
strings.

- `plan.save(destination, overwrite=False)` writes JSON locally or to a
  supported remote URI.
- `RewritePlan.load(source)` loads and validates the artifact.
- `plan.as_dict()` returns JSON-compatible values.
- `plan.for_table(table)` validates learned columns/types/nullability and
  binds a `SortPlan` to the table's row count.
- `plan.validation` holds the held-out check written by `fit_dataset`;
  plans saved before it existed load with an empty tuple.

`fit` resolves a requested codec-default level into the effective concrete
level before saving the plan, so fitted defaults are frozen. A manually
constructed plan with `compression_level=None` still means installed codec
default, not a fixed level. Additional columns are not automatically optimized
by an older plan. Refit when schema or distributions materially change.

## `rewrite(source, destination, **options)`

For a source file, return `WriteResult`. For a tree destination with a learned
plan, route to `rewrite_dataset` and return `DatasetRewriteResult`.
A directory source with an explicit `.parquet` output instead consolidates
in memory through the Arrow path; do not use that form to preserve file
boundaries or process a larger-than-RAM tree. Consolidation requires one
physical schema and rejects Hive `key=value` directories whose key is not a
physical column, because those partition values would be lost.

Core options are:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `plan` | `None` | Arrow can plan one file; DuckDB requires a fixed plan |
| `compression` / `compression_level` | `"preserve"` / `"preserve"` | Preserve known source settings |
| `compression_manifest` | `None` | Trusted exporter codec/level provenance |
| `engine` | `"arrow"` | Optional `"duckdb"` natural-order sorting |
| `skip_unchanged` | `True` | Copy/skip no-key originals and guard non-smaller sorted candidates |
| `overwrite` | `False` | Protect existing destination files |
| `row_group_size` | Arrow default | Explicit change bypasses the source-byte guard |
| `memory_limit` | `"6GB"` | DuckDB memory setting, not a total-process memory guarantee |
| `max_temp_directory_size` | `"18GB"` | DuckDB local spill cap |
| `temp_directory` | System temp location | Parent location for isolated spill directories |
| `source_filesystem` / `destination_filesystem` | Inferred | Explicit Arrow filesystems |

Writer/planning controls can be passed through for the applicable path.
A source page index and bloom filters are rebuilt for the new row order:
bloom filters (PyArrow 25+) on the same columns, with a 1% false-positive
target and no larger than the source filters. Pass `write_page_index` or
`bloom_filter_options` to choose them explicitly; that counts as a writer
change. With preserved settings, no required prefix, and default skip
behavior, a candidate is published only if smaller than the original. Explicit
codec/level/layout changes or required prefixes are honored even if it grows.
See [compression policy](compression.md) and [cloud safety](cloud-storage.md).

## `rewrite_dataset(source, destination, *, plan, **options)`

Requires a `RewritePlan`. Preflight schemas, source-level provenance when
needed, and target collisions before publishing files; then rewrite each
physical file using fixed keys. Relative paths and file boundaries are
retained. The operation is not globally sorted across files or atomic across
the dataset. It accepts the file rewrite controls above, plus:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `workers` | `1` | Files rewritten concurrently on threads; `0` is one per CPU |
| `memory_budget` | 70% of physical or container memory | Bytes or `"24GB"`; bounds estimated in-flight decoded memory when `workers > 1` |
| `shard` | `None` | `"i/N"` or `(i, N)`: only files whose relative-path hash falls in shard `i` |
| `checkpoint` | `True` | Markers in `<destination>/_oparq/checkpoint`; a path/URI elsewhere; `False` disables |
| `progress` | `None` | `progress(result, completed, total)` after each processed file |
| `prefetch` | `max(2, workers)` remote, `0` local | Source files downloaded ahead of the workers that decode them |
| `io_threads` | `4 * workers` for remote sources | Arrow's process-wide I/O pool size during the run; restored afterwards |

Preflight reads footers concurrently and lists the destination once instead
of checking each output path. Threads scale because Arrow decoding, sorting
and Parquet encoding release the GIL; one thread pool also lets one memory
budget see every in-flight file. Decoded size is learned per schema from the
first file (which runs alone) and the pool's live allocation is also checked;
a file estimated above the whole budget runs alone with a `RuntimeWarning`.
DuckDB files reserve 1.25 × `memory_limit` each.

Shards are disjoint and cover every file; machines need no coordination.
A checkpoint marker records the source size and modification time, a
fingerprint of the plan keys and writer settings, and the output size. A
rerun skips a file only when all of these still match and the output exists;
it reads no footer for such files. A changed source or plan, or a missing
output, is processed again (an existing stale output needs `overwrite=True`).
Hidden temporaries left beside this run's outputs by an interrupted run are
deleted. On a failure, no new file starts, running files finish (and are
checkpointed), and the first error is raised.

Prefetched files are fetched whole, larger ones as concurrent 16 MiB ranges,
and decoded from memory; their buffers count toward the live allocation the
scheduler checks. A failed prefetch falls back to a direct read.

## Planning and in-memory helpers

- `read_parquet(source, filesystem=None) -> Table` loads physical Parquet
  columns into memory using Arrow. It does not inject virtual Hive columns.
  Files in a directory must share one physical schema; differing schemas are
  rejected rather than silently dropping columns.
- `plan_sort(value, **planning_options) -> SortPlan` selects keys. Codec trial
  controls use `trial_compression` and `trial_compression_level`; it does not
  write a permanent output. A `SortPlan` is bound to one table row count.
- `profile_table(value, sample_rows=250_000, exclude=(), detect_json=True)`
  returns `ColumnProfile` tuples; sample distinct/run counts are exact for
  selected rows, not estimates presented as exact full-input cardinality.
- `apply_plan(value, sort_plan, sort_backend="auto", gather_threads=0)`
  returns a fully materialized reordered table; `null_placement=None` uses
  the plan's placement. The table row count must match the `SortPlan`.
- `optimize(value, **options) -> OptimizationResult` plans and returns that
  fully materialized reordered table. It can use more memory than streamed
  `write`, because both input and output remain resident.
- `benchmark(value, algorithms=..., compression="zstd", compression_level=None,
  row_group_size=None, **planning_options)` returns `BenchmarkResult` tuples
  with controlled identical writer settings and temporary outputs.

## Results and timings

`WriteResult` exposes `path`, `file_size`, `plan`, and `action`
(`"rewritten"`, `"copied"`, or `"skipped"`). Its stage fields are
`planning_seconds`, `permutation_seconds`, `gathering_seconds`,
`sort_seconds`, and `write_seconds`. On Arrow,
`sort_seconds = permutation_seconds + gathering_seconds`.

DuckDB reports combined upstream scan/sort/output work in `sort_seconds`;
independent permutation/gather times are unavailable, not fabricated zeros.
Writer time includes destination publication, which is atomic locally but
not a cloud bucket transaction. Guarded candidates retain attempted-stage
costs even when their output is discarded. Reused plans report zero search
time, not zero validation or sorting work.

`DatasetRewriteResult.files` contains each file result in inventory order;
convenience counts are `rewritten_files`, `copied_files`, `skipped_files`,
and `resumed_files`. A resumed `WriteResult` has `resumed=True`, the action
recorded by the earlier run, and zero timings. `removed_temporaries` counts
deleted leftovers of an interrupted run, and `shard` echoes the selection.
`planning_and_sort_seconds` remains a result convenience property.

`BenchmarkResult` uses `size_bytes`, `savings_fraction`, `algorithm`,
`resolved_algorithm`, `sort_keys`, and corresponding stage fields.
