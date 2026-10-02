# Command-line reference

The installed entry point is `oparq`; `python -m oparq` also works. From the
checkout, prefix commands with `uv run` and add `--extra duckdb` when needed.

## Inspect a plan

```bash
oparq plan input.parquet
oparq plan input.parquet --compression-level 3 --json
oparq plan input.parquet --algorithm all --full
```

`plan` only compares orders, so it never refuses an unknown source level:
it falls back to the codec's default level (or ZSTD when codecs differ),
uses those settings for every compared order including no sorting, and
prints them as `trial codec`. `benchmark` follows the same rule. Explicit
`--compression`/`--compression-level` values are used as given. `plan` loads
the source table into memory; use `fit` for bounded training rows.

## Learn and save one plan

```bash
oparq fit input/tree order.json \
  --algorithms codec_fast,portfolio,weighted \
  --sample-rows 250000 --sample-files 8 \
  --prefix event_date \
  --compression-manifest source-settings.json
```

`fit` always includes the input/prefix baseline and saves a reusable artifact.
Its bounded `--sample-rows` must be positive. `--overwrite` permits replacing
the plan artifact. Algorithm names are canonical, without aliases.

## Rewrite a file or a partitioned tree

```bash
oparq rewrite input.parquet optimized.parquet \
  --compression-manifest source-settings.json

oparq rewrite s3://source-bucket/events gs://destination-bucket/events \
  --plan order.json --engine duckdb --memory-limit 6GB \
  --compression-manifest source-settings.json
```

A tree destination requires a saved plan. DuckDB also requires fixed keys
and the optional extra. The default Arrow engine can select keys for one
file. `--force-rewrite` disables copy/skip and non-growth guards;
`--overwrite` permits replacing destination files.

Writer changes include `--compression CODEC`,
`--compression-level INTEGER|default|preserve`, and `--row-group-size ROWS`.
Both compression options default to `preserve`. Explicit settings/prefixes
can grow the output. No-key plans with preserved settings normally copy or
skip rather than re-encode.

## Control planning and execution cost

| Option | Purpose |
| --- | --- |
| `--algorithm NAME` | Single-file plan/rewrite strategy |
| `--prefix COL,COL` | Fixed ascending leading keys |
| `--include COL,COL` / `--exclude COL,COL` | Candidate selection |
| `--max-keys N` | Maximum total keys |
| `--fast-candidates N` / `--max-trials N` | Codec-fast search budget |
| `--sample-rows N` | Profile budget; `0` profiles all rows on applicable commands |
| `--full` | Plan with every row instead of profile, run, and trial samples |
| `--trial-sample-rows N` / `--run-sample-rows N` | Trial/proxy budgets |
| `--sort-backend auto|arrow|rank` | Native Arrow sort implementation |
| `--gather-threads N` | `0` automatic; `1` serial |
| `--in-memory-sort` | Materialize the full reordered Arrow table |
| `--memory-limit SIZE` / `--temp-directory PATH` | DuckDB execution controls |

Execution controls do not make a byte-copy decision into a sort. Saved plans
skip algorithm search; a command-line search option does not relearn them.
Use `oparq COMMAND --help` for each command's supported options.

## Controlled comparison

```bash
oparq benchmark input.parquet \
  --algorithms none,codec_fast,portfolio \
  --compression-level 1 --row-group-size 1000000 --rows 500000
```

`--rows 0` uses all loaded rows; bounded `--rows` results are not full-input
claims. This command loads input into memory. Repository benchmark runners
provide full-corpus bounded external sorting and separate exact per-file
engine comparisons; see [benchmarks](benchmarks.md).
