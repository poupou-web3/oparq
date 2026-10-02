from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from oparq.engines import duckdb_sorted_batches, restore_batch_schema, validate_duckdb_schema
from oparq.sorting import sort_options
from oparq.dataset import rewrite_file
from oparq.models import RewritePlan
from oparq.storage import COMPRESSION_METADATA_KEY, compression_metadata


try:
    import duckdb
except ImportError:
    duckdb = None


def _matches_arrow_and_preserves_schema(tmp_path):
    schema = pa.schema([
        pa.field("key", pa.float64()),
        pa.field('quoted"name', pa.uint64(), nullable=False),
        pa.field("binary", pa.binary(3)),
        pa.field("stamp", pa.timestamp("ms", tz="UTC")),
        pa.field("decimal", pa.decimal128(11, 5)),
        pa.field("nested", pa.list_(pa.field("element", pa.uint32(), nullable=False))),
        pa.field("dictionary", pa.dictionary(pa.int8(), pa.string())),
    ], metadata={b"custom": b"original"})
    table = pa.Table.from_arrays([
        pa.array([2.0, None, float("nan"), -1.0, 2.0, None]),
        pa.array([2**64 - 1, 1, 2, 3, 4, 5], type=pa.uint64()),
        pa.array([b"abc", b"def", None, b"xyz", b"aaa", b"bbb"], type=pa.binary(3)),
        pa.array([datetime(2026, 1, 1, tzinfo=UTC)] * 6, type=pa.timestamp("ms", tz="UTC")),
        pa.array([Decimal("1.01234")] * 6, type=pa.decimal128(11, 5)),
        pa.array([[1, 2], [], None, [3], [4], [5]], type=schema.field("nested").type),
        pa.array(["b", "a", None, "c", "b", "a"]).dictionary_encode().cast(schema.field("dictionary").type),
    ], schema=schema)
    with duckdb_sorted_batches(
        schema, table.to_batches(max_chunksize=2), ["key"],
        temp_directory=tmp_path, batch_size=2, threads=2,
    ) as batches:
        actual = pa.Table.from_batches(list(batches), schema=schema)
    permutation = pc.sort_indices(table.select(["key"]), sort_keys=[("key", "ascending")])
    expected = table.take(permutation)
    assert actual.schema.equals(schema, check_metadata=True)
    assert actual['quoted"name'].equals(expected['quoted"name'])
    # Arrow equality regards NaN != NaN; compare all other exact values.
    assert actual.drop(["key", "dictionary"]).equals(expected.drop(["key", "dictionary"]), check_metadata=True)
    assert actual["dictionary"].cast(pa.string()).equals(expected["dictionary"].cast(pa.string()))
    assert actual["key"].to_pylist()[:3] == [-1.0, 2.0, 2.0]
    assert pc.is_nan(actual["key"]).to_pylist()[3] is True
    assert actual["key"].null_count == 2


def _stable_across_batches_with_nulls_first(tmp_path):
    table = pa.table({"key": [None, 1.0, 1.0, None, 0.0, 1.0, float("nan"), float("nan")], "row": range(8)})
    with duckdb_sorted_batches(
        table.schema, table.to_batches(max_chunksize=1), ["key"],
        temp_directory=tmp_path, batch_size=2, null_placement="at_start",
    ) as batches:
        actual = pa.Table.from_batches(list(batches))
    expected = table.take(pc.sort_indices(
        table.select(["key"]), **sort_options([("key", "ascending")], "at_start")
    ))
    assert actual["row"].equals(expected["row"])


@unittest.skipUnless(hasattr(duckdb, "connect"), "DuckDB optional wheel not installed")
class DuckDBEngineTests(unittest.TestCase):
    def test_matches_arrow_and_preserves_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            _matches_arrow_and_preserves_schema(directory)

    def test_stable_across_batches_with_nulls_first(self):
        with tempfile.TemporaryDirectory() as directory:
            _stable_across_batches_with_nulls_first(directory)

    def test_rejects_lossy_nanosecond_zone(self):
        with self.assertRaisesRegex(TypeError, "nanosecond zoned"):
            validate_duckdb_schema(pa.schema([("time", pa.timestamp("ns", tz="UTC"))]))

    def test_preflight_rejects_types_duckdb_truncates_or_cannot_restore(self):
        # Parquet can store all but the interval; each previously failed only
        # after earlier dataset files had been published, or lost nanoseconds.
        for dtype in (pa.month_day_nano_interval(), pa.float16(), pa.decimal256(40, 2),
                      pa.null(), pa.list_view(pa.int64()), pa.struct([("half", pa.float16())])):
            with self.subTest(type=dtype), self.assertRaises(TypeError):
                validate_duckdb_schema(pa.schema([("key", pa.int64()), ("value", dtype)]))
        validate_duckdb_schema(pa.schema([("key", pa.int64()), ("text", pa.large_string()),
                                          ("amount", pa.decimal128(5, 2))]))

    def test_empty_result_batch_restores_the_input_schema(self):
        schema = pa.schema([("key", pa.int64()), ("label", pa.dictionary(pa.int8(), pa.string()))])
        batch = restore_batch_schema(pa.record_batch([pa.array([], pa.int64()), pa.array([], pa.string())],
                                                     names=["key", "label"]), schema)
        self.assertEqual((batch.num_rows, batch.schema), (0, schema))

    def test_rejects_invalid_utf8_before_engine_import_including_nested_and_dictionary(self):
        malformed = pa.array([b"\xff\x90"], type=pa.binary()).view(pa.string())
        nested = pa.ListArray.from_arrays(pa.array([0, 1], type=pa.int32()), malformed)
        dictionary = pa.DictionaryArray.from_arrays(pa.array([0], type=pa.int8()), malformed)
        for column in (malformed, nested, dictionary):
            batch = pa.record_batch([pa.array([1]), column], names=["key", "payload"])
            with self.subTest(type=column.type), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "invalid Arrow input batch 0.*Invalid UTF8"):
                    with duckdb_sorted_batches(batch.schema, [batch], ["key"],
                                               temp_directory=directory) as batches:
                        list(batches)

    def test_bad_later_batch_prevents_output_and_preserves_existing_destination(self):
        valid = pa.record_batch([pa.array([2]), pa.array(["valid"])], names=["key", "payload"])
        malformed = pa.array([b"\xff\x90"], type=pa.binary()).view(pa.string())
        invalid = pa.record_batch([pa.array([1]), malformed], schema=valid.schema)
        table = pa.Table.from_batches([valid, invalid])
        metadata = {COMPRESSION_METADATA_KEY: compression_metadata("zstd", 1)}
        table = table.replace_schema_metadata(metadata)
        plan = RewritePlan(algorithm="weighted", sort_keys=("key",), prefix_keys=(),
                           column_types=tuple((field.name, str(field.type)) for field in table.schema))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            pq.write_table(table, source, compression="zstd", compression_level=1)
            scanner = ds.dataset(str(source), format="parquet").scanner(batch_size=1)
            # Use one-row source batches to prove valid earlier data does not
            # hide a malformed later batch. The SQL global sort consumes all
            # batches before any writer is opened.
            for existing in (False, True):
                destination = Path(directory) / f"output-{existing}.parquet"
                if existing:
                    pq.write_table(pa.table({"untouched": [1]}), destination)
                    original = destination.read_bytes()
                with self.subTest(existing=existing), patch("oparq.dataset.ds.dataset") as dataset_factory:
                    dataset_factory.return_value.scanner.return_value = scanner
                    with self.assertRaisesRegex(ValueError, "invalid Arrow input batch 1.*Invalid UTF8"):
                        rewrite_file(source, destination, plan=plan, engine="duckdb", overwrite=existing)
                if existing:
                    self.assertEqual(destination.read_bytes(), original)
                else:
                    self.assertFalse(destination.exists())
                self.assertFalse(any("oparq-" in path.name for path in Path(directory).iterdir()))

    def test_arrow_rewrite_preserves_malformed_string_bytes_without_cleaning(self):
        malformed = pa.array([b"\xff\x90", b"valid"], type=pa.binary()).view(pa.string())
        table = pa.table({"key": [2, 1], "payload": malformed})
        table = table.replace_schema_metadata({COMPRESSION_METADATA_KEY: compression_metadata("zstd", 1)})
        plan = RewritePlan(algorithm="weighted", sort_keys=("key",), prefix_keys=("key",),
                           column_types=tuple((field.name, str(field.type)) for field in table.schema))
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source.parquet", Path(directory) / "output.parquet"
            pq.write_table(table, source, compression="zstd", compression_level=1)
            rewrite_file(source, destination, plan=plan, engine="arrow")
            actual = pq.read_table(destination)
            self.assertEqual(actual["key"].to_pylist(), [1, 2])
            self.assertEqual(actual["payload"].cast(pa.binary()).to_pylist(), [b"valid", b"\xff\x90"])

    def test_rejects_changed_batch_schema_before_engine_import(self):
        schema = pa.schema([("key", pa.int64())])
        batch = pa.record_batch([pa.array([1], type=pa.int32())], names=["key"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "changed the source schema"):
                with duckdb_sorted_batches(schema, [batch], ["key"], temp_directory=directory) as batches:
                    list(batches)
