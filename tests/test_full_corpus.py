import argparse
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from benchmarks.full_corpus import (
    _annotate_legacy_range_sampling, _connection, _fingerprint, _full_input_key_sample, _is_monotonic,
    _range_bounds, _sort_queries, _write_case, benchmark_source,
)
from oparq.models import SortPlan

try:
    import duckdb
except ImportError:
    duckdb = None


@unittest.skipUnless(hasattr(duckdb, "connect"), "DuckDB optional wheel not installed")
class FullCorpusTests(unittest.TestCase):
    def test_malformed_string_buffers_are_sorted_and_hashed_losslessly_as_binary_views(self):
        raw = [b"z", b"\xff\x90", b"a", None, b"\xff\x90", b"\x00ok", b"b"]
        source = pa.table({
            "key": pa.array(raw, type=pa.binary()).view(pa.string()),
            "identity": range(7),
            "nested": pa.array([[b"\xff"], None, [], [b"ok"], [None], [b"\x90"], [b"z"]],
                               type=pa.list_(pa.binary())).view(pa.list_(pa.string())),
        })
        plan = SortPlan("portfolio", "portfolio", ("key",), (), (), 7, 7)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            files = []
            for index, offset in enumerate(range(0, 7, 3)):
                path = directory / f"part-{index}.parquet"
                pq.write_table(source.slice(offset, 3), path)
                files.append(path)
            args = argparse.Namespace(compression_level=1, sort_ranges=4, sample_rows=100,
                                      batch_size=2, row_group_size=3,
                                      memory_limit="64MB", max_temp_size="1GB", threads=2)
            output = directory / "output.parquet"
            result = _write_case(files, source.schema, source, plan, output, directory, args,
                                 codec="zstd", opaque_strings=True)
            actual = pq.ParquetFile(output).read()
            expected_indices = pc.sort_indices(pa.table({"key": pa.array(raw, type=pa.binary())}),
                                               sort_keys=[("key", "ascending")])
            expected = source.take(expected_indices)
            self.assertEqual(actual.schema.remove_metadata(), source.schema)
            self.assertEqual(actual["key"].combine_chunks().view(pa.binary()).to_pylist(),
                             expected["key"].combine_chunks().view(pa.binary()).to_pylist())
            self.assertEqual(actual["nested"].combine_chunks().view(pa.list_(pa.binary())).to_pylist(),
                             expected["nested"].combine_chunks().view(pa.list_(pa.binary())).to_pylist())
            self.assertTrue(actual["identity"].equals(expected["identity"]))
            self.assertTrue(_is_monotonic([output], ("key",), 2))
            self.assertEqual(result["input_reader"], "arrow_binary_views")
            connection = _connection(directory, args)
            try:
                with self.assertRaises(duckdb.InvalidInputException):
                    _fingerprint(connection, files, source.schema)
                self.assertEqual(_fingerprint(connection, files, source.schema, opaque_strings=True),
                                 _fingerprint(connection, [output], source.schema, opaque_strings=True))
            finally:
                connection.close()

    def test_invalid_source_full_benchmark_uses_explicit_fallback_and_keeps_quality_caveat(self):
        source = pa.table({"hash": pa.array([b"ok", b"\xff", None], type=pa.binary()).view(pa.string()),
                           "identity": [0, 1, 2]})
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            inputs = directory / "input"
            inputs.mkdir()
            pq.write_table(source, inputs / "part.parquet", compression="zstd")
            args = argparse.Namespace(compression_level=1, sort_ranges=4, sample_rows=100,
                                      sample_files=2, algorithms=("none", "codec_fast", "portfolio"), min_stable_age=0,
                                      batch_size=2, row_group_size=3, output=directory / "results.json",
                                      temp_root=directory, memory_limit="64MB", max_temp_size="1GB", threads=2)
            report = {"datasets": []}
            benchmark_source("bad_strings", inputs, args, report)
            record = report["datasets"][0]
            self.assertEqual(record["status"], "pass")
            self.assertTrue(record["data_quality"]["invalid_utf8_in_source_string"])
            self.assertIn("No cleaning or repair", record["data_quality"]["quality_note"])
            self.assertEqual(record["source_fingerprint_mode"], "binary_string_views")
            self.assertEqual(len(record["results"]), 3)
            self.assertEqual(record["results"][0]["rows"], 3)
            self.assertTrue(record["results"][0]["checks"]["all_row_multiset_fingerprint"])
            self.assertTrue(record["results"][0]["source_invalid_utf8_preserved"])

    def test_old_validated_results_keep_timings_and_gain_explicit_range_provenance(self):
        case = {"status": "pass", "plan": {"sort_keys": ["key"], "sampled_rows": 100},
                "sort_ranges": 4, "source_scans_for_sort": 4,
                "scan_sort_gather_seconds": 12.5}
        report = {"datasets": [{"results": [case]}]}
        _annotate_legacy_range_sampling(report)
        self.assertEqual(case["scan_sort_gather_seconds"], 12.5)
        self.assertEqual(case["source_scans_for_sort"], 4)
        self.assertEqual(case["range_sample_method"], "planning_sample")
        self.assertEqual(case["range_sampling_seconds"], 0)
        self.assertEqual(case["key_sampling_source_scans"], 0)

    def test_full_key_reservoir_avoids_biased_heads_and_splits_duplicate_keys(self):
        # Every planning-file head contains only key=0. Full input is mostly
        # key=1, with many ties that must be split by stable file/row identity.
        source = pa.table({"key": [0] * 100 + [1] * 1900,
                           "identity": range(2000), "payload": ["unused" * 30] * 2000})
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            files = []
            for index in range(4):
                path = directory / f"part-{index}.parquet"
                pq.write_table(source, path)
                files.append(path)
            args = argparse.Namespace(memory_limit="64MB", max_temp_size="1GB", threads=2,
                                      sample_rows=1000)
            first = _full_input_key_sample(files, ("key",), directory, args)
            second = _full_input_key_sample(files, ("key",), directory, args)
            self.assertEqual(first.num_rows, 1000)
            self.assertEqual(first.schema.names, ["key", "filename", "file_row_number"])
            self.assertTrue(first.equals(second))
            self.assertGreater(sum(first["key"].to_pylist()), 850)
            range_keys = ("key", "filename", "file_row_number")
            bounds = _range_bounds(first, range_keys, 4)
            self.assertEqual(len(bounds), 3)
            connection = _connection(directory, args)
            try:
                counts = [connection.execute(f"SELECT count(*) FROM ({query})", parameters).fetchone()[0]
                          for query, parameters in _sort_queries(
                              files, source.schema, ("key",), bounds, range_keys=range_keys)]
            finally:
                connection.close()
            self.assertEqual(sum(counts), 8000)
            self.assertTrue(all(1500 < count < 2500 for count in counts), counts)

    def test_null_string_range_bounds_are_typed(self):
        source = pa.table({"key": ["b", None, "a", None, "b"], "row": [0, 1, 2, 3, 4]})
        plan = SortPlan("portfolio", "portfolio", ("key",), (), (), 5, 5)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            file = root / "source.parquet"
            pq.write_table(source, file)
            args = argparse.Namespace(compression_level=1, sort_ranges=5,
                                      batch_size=2, row_group_size=3,
                                      memory_limit="64MB", max_temp_size="1GB", threads=2)
            output = root / "sorted.parquet"
            _write_case([file], source.schema, source, plan, output, root, args, codec="zstd")
            expected = source.take(pc.sort_indices(source, sort_keys=[("key", "ascending")]))
            self.assertTrue(pq.read_table(output).equals(expected))

    def test_ranges_produce_full_global_stable_arrow_order(self):
        source = pa.table({
            "key": [1.0, None, 0.0, float("nan"), 1.0, -1.0, None, 0.0, float("nan"), 1.0],
            "secondary": [2, 0, 3, 2, 1, 4, 0, 1, 2, 1],
            "identity": range(10),
            "nested": [[1], None, [], [2, 3], [4], [5], None, [7], [8], [9]],
        })
        keys = ("key", "secondary")
        plan = SortPlan("portfolio", "portfolio", keys, (), (), source.num_rows, source.num_rows)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            files = []
            for index, offset in enumerate(range(0, 10, 3)):
                path = directory / f"part-{index:03d}.parquet"
                pq.write_table(source.slice(offset, 3), path)
                files.append(path)
            spill = directory / "spill"
            spill.mkdir()
            args = argparse.Namespace(
                compression_level=1, sort_ranges=5, batch_size=3,
                row_group_size=4, memory_limit="64MB", max_temp_size="1GB", threads=2,
            )
            output = directory / "output.parquet"
            result = _write_case(files, source.schema, source, plan, output, spill, args, codec="zstd")
            actual = pq.read_table(output)
            expected = source.take(pc.sort_indices(source, sort_keys=[(key, "ascending") for key in keys]))
            self.assertTrue(actual.drop(["key"]).equals(expected.drop(["key"])))
            self.assertEqual(actual.num_rows, source.num_rows)
            self.assertEqual(pq.read_metadata(output).num_row_groups, 3)
            self.assertTrue(_is_monotonic([output], keys, 3))
            connection = _connection(spill, args)
            try:
                self.assertEqual(_fingerprint(connection, files, source.schema),
                                 _fingerprint(connection, [output], source.schema))
            finally:
                connection.close()
            self.assertGreater(result["sort_ranges"], 1)
            self.assertEqual(result["range_sample_method"], "full_input_key_reservoir")
            self.assertEqual(result["key_sample_rows"], 10)
            self.assertEqual(result["key_sampling_source_scans"], 1)
            self.assertEqual(result["source_scans_for_sort"], result["sort_ranges"] + 1)
            self.assertGreaterEqual(result["scan_sort_gather_seconds"], result["range_sampling_seconds"])

    def test_no_sort_scans_all_shards_in_source_order(self):
        source = pa.table({"key": [4, 0, 3, 2, 1], "nested": [[1], None, [], [2], [3]]})
        plan = SortPlan("none", "none", (), (), (), 5, 5)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            files = []
            for index, offset in enumerate(range(0, 5, 2)):
                path = directory / f"part-{index:03d}.parquet"
                pq.write_table(source.slice(offset, 2), path)
                files.append(path)
            args = argparse.Namespace(
                compression_level=1, sort_ranges=3, batch_size=2,
                row_group_size=3, memory_limit="64MB", max_temp_size="1GB", threads=2,
            )
            output = directory / "output.parquet"
            result = _write_case(files, source.schema, source, plan, output, directory, args, codec="zstd")
            self.assertTrue(pq.read_table(output).equals(source))
            self.assertEqual(result["rows"], 5)
            self.assertEqual(result["source_scans_for_sort"], 1)
