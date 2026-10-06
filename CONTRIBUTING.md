# Contributing to oparq

Use Python 3.12 or newer and keep runtime dependencies separate from release
tooling. PyArrow 16 is the oldest supported release; CI runs the suite
against it, so use feature detection for newer PyArrow APIs.

```bash
uv sync --extra duckdb --group release
uv run python -m unittest discover -s tests -v
uv build
uv run python scripts/check_distribution.py
```

The `src/oparq` directory is the installable package. `docs` contains the
maintained user documentation as plain Markdown read on GitHub; there is no
generated documentation site. Benchmark reproduction code and current
reports live in `benchmarks`; historical experiments are archived in `local/`.
Local datasets, generated output, and upstream research checkouts are ignored
and must not be committed or uploaded with the package.

The public source is `src/`, `tests/`, `docs/`, `benchmarks/`, `scripts/`,
the CI workflows, package configuration, and standard project documentation.
`local/` holds private inputs, research clones, prototypes, archived reports,
and prepared publication bundles. `.venv/`, `dist/`, and caches are ignored
generated output. Do not force-add these paths.

Before committing, review `git status --short --untracked-files=all` and
`git diff --cached`. Benchmark runners default to `local/data/source/`;
use `--data-root` and a new output path for your own inputs. Historical
checkpoints are evidence, not portable resume caches.

Changes to sorting must preserve all logical values, schema, stable tie order,
null/NaN behavior, row-group geometry, and truthful sorting metadata. Test both
native Arrow and optional DuckDB where applicable. Report fused engine timings
as fused; do not invent separate permutation or gathering values.

Compare compression against a no-sort output using the same input rows,
codec, level, dictionaries, writer, and row groups. Identify sampled outputs
explicitly. Do not relabel sample savings as full-dataset savings or claim
that an after-the-fact best combination is the automatic selector's result.

Do not add raw datasets to a public benchmark bundle without verified source
provenance and redistribution permission. Public bundles must not contain
credentials, machine-local paths, spill files, or arbitrary workspace files.

Before a public release, check the built artifacts and install the wheel in
an isolated environment, with and without the DuckDB extra.
Neither unit tests nor this repository's CI publishes to PyPI automatically.

## Documentation and publication assets

Tests check that every maintained Markdown file is listed, that relative
links resolve, and that the quickstart examples run. The public guides describe
library use; owner checklists and dated narrative reports belong in ignored
`local/` storage. The Python distributions contain the runtime source, README,
and license—not the documentation, benchmark records, release tooling, or
local notes.

Benchmark-result bundles are prepared with `scripts/prepare_benchmarks.py`
and checked with `scripts/publish_benchmarks.py --help`. Preparation is local;
upload requires authentication, a named target, and explicit `--confirm-public`.
Input bundles come from `scripts/prepare_inputs.py`, which copies only the
datasets cleared in its `PUBLISHED` list, with each source's licence and
required attribution; record a new dataset's terms there before publishing it.
Never treat a public download URL as a blanket data license, or claim that a
prepared bundle has already been uploaded. Generate a new bundle after code
or docs change; do not mutate an existing checksummed snapshot.
