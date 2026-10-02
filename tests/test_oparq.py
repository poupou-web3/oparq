from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import oparq


class ProfileTests(unittest.TestCase):
    def test_nested_and_json_are_excluded_but_null_scalar_is_eligible(self) -> None:
        table = pa.table(
            {
                "category": pa.array(["a", None, "a", "b"]),
                "json_payload": [
                    json.dumps({"x": 1}),
                    json.dumps({"x": 2}),
                    json.dumps({"x": 3}),
                    json.dumps({"x": 4}),
                ],
                "nested": pa.array([[1], [2], None, [3]]),
            }
        )
        profiles = {item.name: item for item in oparq.profile_table(table)}
        self.assertTrue(profiles["category"].eligible)
        self.assertEqual(profiles["category"].sample_distinct, 3)
        self.assertFalse(profiles["json_payload"].eligible)
        self.assertEqual(profiles["json_payload"].reason, "JSON-like string")
        self.assertFalse(profiles["nested"].eligible)

    def test_exact_mode_profiles_all_rows(self) -> None:
        table = pa.table({"x": [1, 1, 2, 3]})
        profile = oparq.profile_table(table, sample_rows=None)[0]
        self.assertEqual(profile.sampled_rows, 4)
        self.assertEqual(profile.sample_distinct, 3)


class PlanningTests(unittest.TestCase):
    def test_cardinality_orders_low_to_high_and_drops_constant(self) -> None:
        table = pa.table(
            {
                "constant": [1] * 8,
                "low": [0, 1] * 4,
                "high": list(range(4)) * 2,
            }
        )
        plan = oparq.plan_sort(
            table,
            algorithm="cardinality",
            sample_rows=None,
            max_sort_columns=8,
        )
        self.assertEqual(plan.sort_keys, ("low", "high"))

    def test_weighted_prefers_heavy_repeatable_column(self) -> None:
        rows = 100
        table = pa.table(
            {
                "tiny": [index % 5 for index in range(rows)],
                "heavy": [("x" * 200) + str(index % 10) for index in range(rows)],
                "unique": [("u" * 200) + str(index) for index in range(rows)],
            }
        )
        plan = oparq.plan_sort(
            table,
            algorithm="weighted",
            sample_rows=None,
            max_sort_columns=1,
        )
        self.assertEqual(plan.sort_keys, ("heavy",))

    def test_prefix_is_always_first(self) -> None:
        table = pa.table({"day": [2, 1, 2, 1], "kind": [2, 2, 1, 1]})
        plan = oparq.plan_sort(
            table,
            algorithm="cardinality",
            prefix=["day"],
            sample_rows=None,
        )
        self.assertEqual(plan.sort_keys[0], "day")

    def test_auto_uses_bounded_codec_search(self) -> None:
        table = pa.table(
            {
                "group": [index % 5 for index in range(500)],
                "payload": ["payload-" + str(index % 5) for index in range(500)],
            }
        )
        plan = oparq.plan_sort(
            table,
            algorithm="auto",
            sample_rows=None,
            trial_sample_rows=500,
        )
        self.assertEqual(plan.algorithm, "codec_fast")
        self.assertTrue(plan.sort_keys)
        self.assertEqual(plan.score_kind, "Parquet bytes")

    def test_runs_skips_scoring_when_no_key_can_be_selected(self) -> None:
        table = pa.table({"key": [1, 2, 1, 2], "nested": [[1], [2], [3], [4]]})
        for options in ({"max_sort_columns": 0}, {"include": []}):
            with self.subTest(**options), patch(
                "oparq.planning._weighted_run_cost", side_effect=AssertionError("scored")
            ):
                plan = oparq.plan_sort(table, algorithm="runs", **options)
                self.assertEqual(plan.sort_keys, ())
                self.assertIsNone(plan.score_kind)
                self.assertIsNone(plan.baseline_score)

    def test_dictionary_key_is_decoded_for_table_sort(self) -> None:
        dictionary = pa.array(["b", "a"])
        values = pa.DictionaryArray.from_arrays([0, 1, 0, 1], dictionary)
        table = pa.table({"key": values, "row": [0, 1, 2, 3]})
        result = oparq.optimize(
            table,
            algorithm="none",
            prefix=["key"],
            sample_rows=None,
        )
        self.assertEqual(result.table["key"].to_pylist(), ["a", "a", "b", "b"])


class SortAndWriteTests(unittest.TestCase):
    def test_apply_plan_sorts_and_preserves_rows(self) -> None:
        table = pa.table({"key": [2, 1, 2, 1], "value": ["b", "a", "d", "c"]})
        result = oparq.optimize(
            table,
            algorithm="none",
            prefix=["key", "value"],
            sample_rows=None,
        )
        self.assertEqual(result.table["key"].to_pylist(), [1, 1, 2, 2])
        self.assertCountEqual(
            zip(table["key"].to_pylist(), table["value"].to_pylist()),
            zip(result.table["key"].to_pylist(), result.table["value"].to_pylist()),
        )

    def test_write_records_sorting_metadata_and_plan(self) -> None:
        table = pa.table({"key": [2, 1, 2, 1], "value": ["b", "a", "d", "c"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            result = oparq.write(
                table,
                path,
                algorithm="none",
                prefix=["key"],
                compression="snappy",
            )
            self.assertEqual(result.path, path)
            metadata = pq.read_metadata(path)
            self.assertEqual(metadata.row_group(0).sorting_columns[0].column_index, 0)
            restored = pq.read_table(path)
            self.assertIn(b"oparq.sort_plan", restored.schema.metadata)
            self.assertEqual(restored["key"].to_pylist(), [1, 1, 2, 2])

    @unittest.skipIf(os.name != "posix", "POSIX permission bits")
    def test_local_output_permissions_follow_umask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.parquet"
            pq.write_table(pa.table({"key": [1]}), reference)
            path = Path(directory) / "data.parquet"
            oparq.write(pa.table({"key": [2, 1]}), path, algorithm="none")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),
                             stat.S_IMODE(reference.stat().st_mode))

    def test_write_refuses_overwrite_by_default(self) -> None:
        table = pa.table({"key": [2, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            oparq.write(table, path, algorithm="none")
            with self.assertRaises(FileExistsError):
                oparq.write(table, path, algorithm="none")

    def test_benchmark_uses_first_algorithm_as_baseline(self) -> None:
        table = pa.table(
            {
                "key": [index % 4 for index in range(200)],
                "payload": ["large-payload-" + str(index % 4) for index in range(200)],
            }
        )
        results = oparq.benchmark(
            table,
            algorithms=["none", "weighted"],
            compression="snappy",
            sample_rows=None,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].savings_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
