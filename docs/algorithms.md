# Sorting algorithms

No heuristic guarantees the globally smallest Parquet file. Savings depend
on correlations across the whole schema, not just the largest sort column.
An apparently compressible URL column can lose more bytes in other columns
when it destroys their existing clustering.

## Canonical names

| Algorithm | Decision | Typical trade-off |
| --- | --- | --- |
| `auto` | Uses `codec_fast` | Default bounded compression/speed balance |
| `all` | Run every planner below, then encode each distinct proposal | Most planning CPU; best of the built-in proposals, not of all possible orders |
| `codec_fast` | Greedy shortlisted Parquet-byte trials | Four candidates/step and at most 24 trial encodes by default |
| `codec` | Broader greedy Parquet-byte trials | Up to 12 candidates/step; more planning CPU |
| `portfolio` | Compare complete cardinality/weighted/entropy/payload proposals | Up to five encodes including the input/prefix baseline; proposals can miss shorter useful prefixes |
| `runs` | Byte-weighted equal-value run proxy | Experimental cheap approximation of codec behavior |
| `weighted` | Prefer large repeatable non-degenerate columns, then increasing cardinality | Histogram/profiling path without codec trials |
| `cardinality` | Increasing sampled distinct count | Simple low-cardinality-first baseline |
| `entropy` | Inverse-Simpson effective cardinality | Accounts for highly skewed distributions |
| `payload` | Potential saved bytes per bit of effective cardinality | Prioritizes large repeated payloads |
| `frequency` | Entropy keys with frequency-ranked values | Experimental; non-natural suffix order |
| `none` | Retain input order, or sort only a mandatory prefix | Controlled no-extra-key baseline |

Only these canonical names are accepted. `auto` resolves
to `codec_fast`, not a learned universal selector.

`all` collects the proposals of `cardinality`, `weighted`, `entropy`,
`payload`, `frequency`, `runs`, `codec_fast`, `codec`, and the bounded
`portfolio` pool, then encodes each distinct proposal once on the same rows
with the same codec settings. The input/prefix order stays unless the
smallest proposal beats it by `min_trial_improvement`. The plan note names
the planners that proposed the winner. A `frequency` winner is not
supported by the DuckDB engine.

## Plan with every row

```python
oparq.write(table, "optimized.parquet", algorithm="all", full=True)
```

`full=True` disables planning samples: profiles, run proxies, and every
trial encode use the whole input table. With `fit_dataset`, that whole
input is the sampled training table. Full planning is exact for that input
but can cost many full-table encodes; use it when planning time matters
less than choosing among the built-in orders.

## Columns that are never sort keys

Nested, map, list, struct, extension, view-layout, and detected JSON
columns are not automatic candidates; their values still move with each
row. When no key can be selected, the search is skipped and input order is
kept. PyArrow (through 25) cannot reorder `string_view`, `binary_view`, or
run-end encoded columns: a table containing one is neither searched nor
sorted by the Arrow engine, and its plan note names the columns. A required prefix
on such a table is an error.

## Bound the cost of deciding keys

```python
oparq.write(
    table, "optimized.parquet",
    algorithm="codec_fast",
    sample_rows=250_000,
    trial_sample_rows=250_000,
    fast_candidate_count=4,
    max_trial_evaluations=24,
    max_sort_columns=8,
)
```

The profiler computes distinct/run statistics exactly for its selected rows,
not an exact full-input cardinality when sampling is active. Null counts as
a bucket; constants and nearly unique columns receive little or no weight.
The payload/byte scores are ranking proxies, not promised savings.

JSON detection is heuristic; use `exclude=("opaque_json",)` when you know a
string column is opaque. `include` limits automatic candidates.

## Require leading keys, then discover a suffix

```python
plan = oparq.fit(
    sample,
    algorithms=("codec_fast", "portfolio"),
    prefix=("event_date", "tenant_id"),
    compression="zstd", compression_level=1,
)
```

Prefix order is fixed, natural, and ascending. Prefix keys count toward
`max_sort_columns`. The baseline uses that required prefix, and the search
adds only a suffix. There is no descending-key API in 0.3.0. Required prefixes
are honored even when compression worsens; they are application constraints.
Keys must be physical columns; virtual Hive partition columns are not injected.

`frequency` sorts suffix values by frequency rather than natural order, so
Parquet sort metadata advertises only its natural prefix. Row-group statistics
remain enabled. The optional DuckDB backend supports natural learned plans,
not frequency-ranked suffixes.

## Separate selection from application

`fit` compares strategies on supplied sample rows and freezes the winning
keys; `fit_dataset` reads a bounded file-spread sample. The default minimum
sample improvement is 0.5% over input/prefix order. Reusing `RewritePlan`
avoids another profiling/search pass but still requires sorting and writing.
Codec settings are held fixed: these are row-order algorithms, not codec
selection algorithms.

Research context: [Reordering Rows for Better Compression](https://arxiv.org/abs/1207.2189),
[Reordering Columns for Smaller Indexes](https://arxiv.org/abs/0909.1346),
and ClickHouse's [optimize_row_order](https://clickhouse.com/blog/clickhouse-release-24-06).
The benchmark regressions demonstrate why full-output safeguards matter;
see [measured limitations](benchmarks.md).
