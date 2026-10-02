# Benchmarks and limitations

The September 30, 2026 development benchmark covers every row of
21 datasets: **967,938,981 rows in 986 source files**. Only decision samples
are bounded; output rows are not sampled. The separate engine comparison
covers **26,015,956 rows in 40 files** from stock, Hits, and Solana, not all
21 datasets.

Source inventories, algorithms, keys, writer settings, checks, and timings are
recorded in `benchmarks/results/`. These are historical measurements made
during development, not a fresh benchmark of each subsequent package version.
The recorded implementation versions and method changes remain in the JSON.

## Full-corpus controlled comparison

All methods use the same PyArrow writer, dictionaries, statistics, rows, and
1,000,000-row groups. Source codecs are retained by dataset: ZSTD sources
use explicitly controlled **level 1**, and Snappy stays Snappy. Source ZSTD
levels were unknown, not inferred to be 1 or 9.

| Method | Aggregate output bytes | Saved against no-sort |
| --- | ---: | ---: |
| `none` | 11,778,560,511 | Baseline |
| `codec_fast` | 13,171,760,772 | −11.83%: larger |
| `portfolio` | 9,861,456,670 | 16.28% |
| Best full-result choice per dataset, including `none` | 9,625,971,945 | 18.28% |

The best mixture is an **after-the-fact selection**, not a measured outcome
of `auto` or a sample-only chooser. No global optimum or uniform saving is
claimed. Important dataset results include:

| Dataset / strategy | Full rows | Saved against identical no-sort output |
| --- | ---: | ---: |
| ontime / `portfolio` | 33,000,000 | 65.94% |
| Solana / `codec_fast` | 6,105,615 | 52.67% |
| stock / `portfolio` | 14,910,341 | 33.54% |
| Hits / `codec_fast` | 5,000,000 | 0.20% |
| PyPI / both search methods | 40,000,000 | 0%; input order retained |
| cisco_umbrella / `codec_fast` | 95,000,000 | −129.00%: larger |
| noaa_v2 / `codec_fast` | 364,000,000 | −198.76%: larger |

These regressions are why sample estimates are not promised savings and
production preserved-settings file rewrites compare actual candidate bytes
with the original. The raw runner intentionally measures bad candidates
rather than hiding them behind that guard.

### Full execution and checks

The runner streams the Arrow no-sort baseline and uses DuckDB external sorting
for natural-key candidates. Ordered disjoint ranges yield a global stable
order across each dataset. Legacy boundaries came from leading planning
samples; later boundaries use a seeded full-input key/identity reservoir,
adding a narrow key scan. Actual ranges/scans and sampling time are recorded.
The mixed checkpoint versions and single local runs are not replicated engine
speed comparisons.

Every newly encoded output is checked for row count and original schema,
plus two seeded 64-bit row hashes with integer sums and XOR over all rows.
This is a probabilistic multiset check, not mathematical identity proof.
Sorted outputs additionally check all keys for monotonicity across batches.
Stable ties use filename/source-row identity in SQL and exact fixtures.
Identical orders reuse already validated output/stages rather than repeating
encoding; only their planning is newly timed.

This runner sorts globally across source files. Production `rewrite_dataset`
sorts within each file and retains boundaries; those storage savings need not
match global benchmark savings.

## Arrow/DuckDB comparison

All 40 stock/Hits/Solana physical files were rewritten with the same saved
`portfolio` keys and writer settings, without another planning search.
Every output column and row order was compared exactly with an independently
computed stable Arrow permutation of that source file. Null/NaN logical values
and nested values are included. This is not the probabilistic corpus hash check.

| Path | Summed rewrite wall time | Output bytes |
| --- | ---: | ---: |
| Arrow | 38.00 seconds | 1,929,051,066 |
| DuckDB path | 69.96 seconds | 1,929,029,129 |

Arrow was faster here; DuckDB was close on stock and about 2.50× slower on
Solana. Hits selected no keys: its DuckDB-named path streams Arrow batches
without SQL sorting. These are single runs with cache effects, not a universal
speed ranking. DuckDB is retained for bounded-memory/spilling and framework
integration.

Small encoded-column/footer differences occurred despite matching logical
order and row-group sizes. They are not all attributed to metadata and do
not establish a material compression advantage for either engine.

## Source data quality

PyPI contains a field declared STRING with malformed UTF-8 bytes. The strict
DuckDB Parquet reader rejected them. The benchmark used an explicit,
benchmark-only STRING/BINARY buffer-view bridge for byte-preserving source
and output fingerprints; all full PyPI plans chose no keys, so writes used
Arrow directly. The sorted binary bridge is covered by exact fixtures.

Original bytes and the logical schema were retained; **malformed STRING
bytes remain malformed**. No cleaning, replacement, or repair was done.
Production DuckDB sorting fully validates Arrow batches and rejects such
input rather than silently replacing strings. Native Arrow can preserve the
bytes, but that does not make the STRING column conforming.

## Input availability and reproducibility

The original inputs are not included in the source repository or Python
distributions. The intended public input corpus comprises the 20
ClickHouse-derived datasets: 961,833,366 rows in 966 files. Solana is excluded
from original-input publication; its historical measurements remain labeled
above. Publishing original ClickHouse inputs requires source attribution and
the applicable upstream redistribution terms, not oparq's MIT license.

No Hugging Face upload is confirmed. The existing results-bundle tool publishes
measurements, hashes, provenance, and reproduction code only; it does not yet
implement original-input upload. Exact acquisition/export provenance and
immutable public input revisions have not been established. Until matching
inputs are available, an independent full rerun is not publicly reproducible.
Historical cases also lack separately archived executable snapshots; the
current code and recorded method/version annotations must be distinguished.

## Reproduce from the source checkout

```bash
uv run --extra duckdb python -m benchmarks.full_corpus \
  --algorithms none,codec_fast,portfolio --compression-level 1 \
  --row-group-size 1000000 \
  --output benchmarks/results/full-corpus-rerun.json

uv run --extra duckdb python -m benchmarks.compare_saved_plan \
  --plan-checkpoint benchmarks/results/full-corpus-rerun.json \
  --datasets stock,hits,solana --algorithm portfolio \
  --compression-level 1 --row-group-size 1000000 \
  --output benchmarks/results/saved-plan-engines-rerun.json
```

Use the same stable input snapshot and enough memory/spill space. The runners
checkpoint compatible completed cases; choose a new output when changing
settings. Temporary outputs are removed after checks; original source files
are not changed. Consult current report JSON for complete stage times rather
than comparing unrelated historical runs.
