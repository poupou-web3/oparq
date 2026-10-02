# Installation and quickstart

## Install

Python 3.12+ and PyArrow 16+ are required. The optional DuckDB extra adds an
external-memory sort backend.

```bash
pip install oparq
# Include the optional engine:
pip install "oparq[duckdb]"
```

From a source checkout, `uv sync --extra duckdb` installs the same
dependencies for development.

## Write an in-memory Arrow table

```python
import pyarrow as pa
import oparq

table = pa.table({
    "account": ["b", "a", "b", "a"],
    "payload": ["repeat-b", "repeat-a", "repeat-b", "repeat-a"],
})
result = oparq.write(table, "optimized.parquet", algorithm="auto")
print(result.plan.explain())
print(result.file_size)
```

This writes a new table using Zstandard's Arrow codec default, currently
level 1, dictionaries, statistics, and up to eight keys. It does not inherit
an encoder level from the table. New files record effective codec/level
provenance for future rewrites.

An existing destination is rejected unless `overwrite=True`. Physical row
order can change; do not use this if unsorted arrival order is semantic.
`prefix=("event_date",)` supplies required natural ascending leading keys;
it does not preserve arbitrary arrival order or request descending order.

## Compare against no sorting

```python
results = oparq.benchmark(
    table,
    algorithms=("none", "codec_fast", "portfolio"),
    compression="zstd",
    compression_level=1,
    row_group_size=100_000,
)
for result in results:
    print(result.algorithm, result.size_bytes, result.savings_fraction)
```

The same writer settings and rows are used for every candidate. Temporary
outputs are removed after measuring. Comparing a sorted ZSTD9 rewrite with
an original Snappy file would not isolate row-order savings.

## Learn once instead of planning every write

```python
plan = oparq.fit(
    table,
    algorithms=("codec_fast", "portfolio", "weighted"),
    compression="zstd",
    compression_level=1,
)
plan.save("order.json")

future = pa.table({"account": ["c", "a"], "payload": ["repeat-c", "repeat-a"]})
result = oparq.write(future, "future.parquet", plan=oparq.RewritePlan.load("order.json"))
assert result.planning_seconds == 0
```

The fixed keys are reused with a new row count. Learned columns, types, and
recorded nullability are validated. Refit deliberately when distributions
change; sample wins are not guarantees for all future files.

For file rewrites, defaults differ: source codec/level preservation is the
policy. Continue with [compression provenance](compression.md) and
[the bucket workflow](cloud-storage.md).
