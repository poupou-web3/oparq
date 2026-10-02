from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa

import oparq
from oparq.planning import _encoded_size
from oparq.profile import sample_table


def _table(rows: int = 600) -> pa.Table:
    return pa.table({
        "group": [index % 3 for index in range(rows)],
        "label": ["label-" + str(index % 7) * 20 for index in range(rows)],
        "code": [index % 11 for index in range(rows)],
        "unique": list(range(rows)),
    })


class AllAlgorithmTests(unittest.TestCase):
    def test_selects_the_smallest_proposal_from_every_planner(self):
        table = _table()
        plan = oparq.plan_sort(table, algorithm="all", sample_rows=None, trial_sample_rows=600,
                               trial_compression="zstd", trial_compression_level=1)
        self.assertEqual(plan.algorithm, "all")
        self.assertEqual(plan.score_kind, "Parquet bytes")
        self.assertIn("all compared", plan.note)
        sample = sample_table(table, 600)
        for algorithm in ("cardinality", "weighted", "entropy", "payload", "runs",
                          "codec_fast", "codec", "portfolio"):
            with self.subTest(algorithm=algorithm):
                other = oparq.plan_sort(table, algorithm=algorithm, sample_rows=None,
                                        trial_sample_rows=600)
                size = _encoded_size(sample, other.sort_keys, compression="zstd",
                                     compression_level=1, null_placement="at_end")
                self.assertLessEqual(plan.estimated_score, size)

    def test_encodes_each_distinct_order_once_and_names_every_proposer(self):
        seen = []

        def size(table, keys, **options):
            seen.append((tuple(keys), options.get("permutation") is not None))
            return 50 if tuple(keys) == ("group", "label") else 100

        with patch("oparq.planning._encoded_size", side_effect=size):
            plan = oparq.plan_sort(_table(), algorithm="all", sample_rows=None, max_sort_columns=2)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(plan.trial_evaluations, len(seen))
        self.assertEqual(plan.sort_keys, ("group", "label"))
        self.assertEqual((plan.baseline_score, plan.estimated_score), (100, 50))
        self.assertRegex(plan.note, r"selected .*cardinality")

    def test_keeps_input_order_without_a_material_win(self):
        with patch("oparq.planning._encoded_size",
                   side_effect=lambda table, keys, **options: 999 if keys else 1000):
            plan = oparq.plan_sort(_table(), algorithm="all")
        self.assertEqual(plan.sort_keys, ())
        self.assertIn("input/prefix order retained", plan.note)

    def test_frequency_winner_advertises_only_its_natural_prefix(self):
        def size(table, keys, *, permutation=None, **options):
            return 10 if permutation is not None else 100

        with patch("oparq.planning._encoded_size", side_effect=size):
            plan = oparq.plan_sort(_table(), algorithm="all", prefix=["group"])
        self.assertEqual(plan.value_order, "frequency")
        self.assertEqual(plan.sort_keys[0], "group")
        self.assertGreater(len(plan.sort_keys), 1)
        self.assertEqual(plan.sorting_keys, ("group",))
        self.assertIn("frequency", plan.note)

    def test_saved_all_plan_round_trips(self):
        plan = oparq.fit(_table(), algorithms=("all",), compression="zstd", compression_level=1,
                         min_improvement=0)
        self.assertEqual(oparq.RewritePlan.from_dict(plan.as_dict()), plan)


class FullPlanningTests(unittest.TestCase):
    def test_full_profiles_and_encodes_every_row(self):
        rows_seen = []

        def size(table, keys, **options):
            rows_seen.append(table.num_rows)
            return 100 - len(keys)

        table = _table(1200)
        with patch("oparq.planning._encoded_size", side_effect=size):
            plan = oparq.plan_sort(table, algorithm="codec", full=True, sample_rows=10,
                                   trial_sample_rows=10, run_sample_rows=10)
        self.assertEqual(plan.sampled_rows, table.num_rows)
        self.assertEqual(set(rows_seen), {table.num_rows})

    def test_full_is_forwarded_by_write(self):
        with patch("oparq.io.plan_sort", wraps=oparq.plan_sort) as planner, \
                tempfile.TemporaryDirectory() as directory:
            oparq.write(_table(), Path(directory) / "out.parquet", algorithm="cardinality", full=True)
        self.assertTrue(planner.call_args.kwargs["full"])


if __name__ == "__main__":
    unittest.main()
