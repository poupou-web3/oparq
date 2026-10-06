# Benchmark runners

`results/` contains the path-sanitized September 30, 2026 development
measurements. [Methodology and limitations](../docs/benchmarks.md) explains
their coverage, correctness checks, regressions, and input availability.
These records are not benchmarks of every subsequent package revision.
Generate narrative summaries from JSON; do not commit dated duplicate reports.
The published bundle is the [`Poupou/oparq-benchmarks`](https://huggingface.co/datasets/Poupou/oparq-benchmarks) dataset.

For a new comparison on your own stable input files:

```bash
uv run --extra duckdb python -m benchmarks.full_corpus \
  --data-root /path/to/inputs \
  --algorithms none,codec_fast,portfolio --compression-level 1 \
  --row-group-size 1000000 \
  --output local/results/full-corpus-rerun.json
uv run --extra duckdb python -m benchmarks.compare_saved_plan \
  --data-root /path/to/inputs --datasets stock,hits --algorithm portfolio \
  --plan-checkpoint local/results/full-corpus-rerun.json \
  --compression-level 1 --row-group-size 1000000 \
  --output local/results/engines-rerun.json
```

Input layout follows the inventories: `clickhouse/<dataset>/*.parquet`,
plus an optional `solana/` tree. Runners default to ignored
`local/data/source/`. Only the `ontime`, `trips`, `lineorder`, `cell_towers`,
and `covid` inputs are republished, in [`Poupou/oparq-benchmark-inputs`](https://huggingface.co/datasets/Poupou/oparq-benchmark-inputs)
(see `scripts/prepare_inputs.py`); the other inputs' terms do not allow it.

Hold rows, codec, level, row groups, dictionaries, and writer settings fixed.
ZSTD1 is an explicit experimental choice, not an inferred source level.
The no-sort baseline is mandatory. Only planning uses bounded samples;
full-corpus output covers every row. Global corpus sorting and per-file engine
sorting have different scopes and their savings are not interchangeable.

Use a new output path for a new run. Checkpoints resume compatible validated
cases; committed checkpoints use portable `inputs/` identifiers and cannot
locate a local source snapshot. Recorded timings are single-run,
hardware/cache-dependent measurements. See the JSON for separate planning,
permutation, gathering, writing, and fused DuckDB stage fields.

The source repository retains runners and machine-readable evidence. Detailed
summary Markdown is generated when preparing Hugging Face result bundles.
Original Parquet inputs, local owner notes, generated bundles, and archived
experiments are not committed or included in Python distributions.
