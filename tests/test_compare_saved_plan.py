import argparse
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from benchmarks.compare_saved_plan import (
    _array_equal, _saved_plan, _validate_output, benchmark_source,
)
from oparq.engines import require_duckdb


def _fixture():
    return pa.table({
        "key": [1.0, None, 0.0, float("nan"), 1.0, -1.0, None, 0.0, float("nan"), 1.0],
        "secondary": [2, 0, 3, 2, 1, 4, 0, 1, 2, 1],
        "identity": range(10),
        "nested": [[1.0], None, [], [float("nan"), None], [4.0], [5.0], None, [7.0], [8.0], [9.0]],
        "struct": [{"number": float("nan")}, {"number": 2.0}, None, {"number": None},
                   {"number": 4.0}, None, {"number": 6.0}, {"number": 7.0},
                   {"number": 8.0}, {"number": 9.0}],
    })


def _checkpoint(name="tiny"):
    return {"datasets": [{"name": name, "results": [{
        "requested_algorithm": "portfolio", "status": "pass",
        "plan": {"algorithm": "portfolio", "sort_keys": ["key", "secondary"],
                 "prefix_keys": ["key"], "sampled_rows": 5,
                 "value_order": "natural", "null_placement": "at_end"},
    }]}]}


class ExactValidationTests(unittest.TestCase):
    def test_nested_nan_equality_is_exact_and_ignores_masked_children(self):
        left = pa.array([[float("nan"), None], None, []], type=pa.list_(pa.float64()))
        right = pa.array([[float("nan"), None], None, []], type=pa.list_(pa.float64()))
        self.assertTrue(_array_equal(left, right))
        self.assertFalse(_array_equal(left, pa.array([[float("nan"), 3.0], None, []], type=left.type)))
        struct_type = pa.struct([("value", pa.float64())])
        self.assertTrue(_array_equal(
            pa.array([None, {"value": float("nan")}], type=struct_type),
            pa.array([None, {"value": float("nan")}], type=struct_type)))

    def test_validator_checks_all_payload_values_and_stable_ties(self):
        table = _fixture()
        keys = ("key", "secondary")
        expected = table.take(pc.sort_indices(table, sort_keys=[(key, "ascending") for key in keys]))
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            source = directory / "source.parquet"
            output = directory / "output.parquet"
            pq.write_table(table, source)
            ordering = pq.SortingColumn.from_ordering(table.schema, [(key, "ascending") for key in keys])
            pq.write_table(expected, output, row_group_size=3, sorting_columns=ordering)
            result = _validate_output(source, output, keys, "at_end", 3, 2)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["row_group_rows"], [3, 3, 3, 1])
            # Corrupt a non-key value while preserving the keys and schema.
            altered = expected.set_column(2, "identity", pa.array([-1] * expected.num_rows))
            pq.write_table(altered, output, row_group_size=3, sorting_columns=ordering)
            result = _validate_output(source, output, keys, "at_end", 3, 2)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["unequal_columns"], ["identity"])

    def test_saved_plan_requires_validated_checkpoint_case(self):
        plan = _saved_plan(_checkpoint(), "tiny", "portfolio", _fixture().schema)
        self.assertEqual(plan.sort_keys, ("key", "secondary"))
        self.assertEqual(plan.prefix_keys, ("key",))
        with self.assertRaisesRegex(ValueError, "no validated"):
            _saved_plan(_checkpoint(), "tiny", "codec_fast", _fixture().schema)


try:
    duckdb_available = hasattr(require_duckdb(), "connect")
except ImportError:
    duckdb_available = False


@unittest.skipUnless(duckdb_available, "optional DuckDB wheel not installed")
class SavedPlanComparisonTests(unittest.TestCase):
    def test_all_physical_files_exact_engines_and_checkpoint_resume(self):
        table = _fixture()
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            source = directory / "input"
            for index in range(2):
                partition = source / f"day={index + 1}"
                partition.mkdir(parents=True)
                pq.write_table(table.slice(index * 5, 5), partition / "part.parquet",
                               compression="zstd", compression_level=3)
            args = argparse.Namespace(
                algorithm="portfolio", compression_level=1,
                row_group_size=3, batch_size=2, memory_limit="64MB",
                max_temp_size="1GB", temp_root=directory, min_stable_age=0,
                output=directory / "results.json",
            )
            report = {"datasets": []}
            benchmark_source("tiny", source, args, report, _checkpoint())
            record = report["datasets"][0]
            self.assertEqual(record["status"], "pass")
            self.assertEqual(len(record["files"]), 2)
            self.assertEqual({file["relative_path"] for file in record["files"]},
                             {"day=1/part.parquet", "day=2/part.parquet"})
            self.assertEqual({result["rows"] for result in record["results"]}, {10})
            self.assertTrue(all(result["planning_seconds"] == 0 for result in record["results"]))
            for file in record["files"]:
                self.assertEqual(file["writer"]["compression_level"], 1)
                self.assertEqual(len(file["cases"]), 3)
                self.assertTrue(all(case["status"] == "pass" for case in file["cases"]))
                duckdb_case = next(case for case in file["cases"] if case["engine"] == "duckdb")
                arrow_case = next(case for case in file["cases"] if case["case"] == "portfolio_arrow")
                self.assertTrue(duckdb_case["input_read_is_component_of_sort_seconds"])
                self.assertFalse(arrow_case["input_read_is_component_of_sort_seconds"])
                self.assertIsNone(duckdb_case["permutation_seconds"])
                self.assertIsNotNone(arrow_case["permutation_seconds"])
                self.assertEqual(arrow_case["compressed_column_bytes"], duckdb_case["compressed_column_bytes"])
            # A completed report is reused without changing case timings.
            before = args.output.read_text()
            benchmark_source("tiny", source, args, report, _checkpoint())
            self.assertEqual(args.output.read_text(), before)
            self.assertFalse(any(directory.glob("oparq-engines-*")))


if __name__ == "__main__":
    unittest.main()
