from __future__ import annotations

import unittest
from unittest.mock import patch

import pyarrow as pa

import oparq


def _table() -> pa.Table:
    return pa.table({
        "prefix": [index % 2 for index in range(120)],
        "small": [index % 3 for index in range(120)],
        "heavy": ["payload-" * 20 + str(index % 7) for index in range(120)],
        "other": [index % 11 for index in range(120)],
    })


class PortfolioTests(unittest.TestCase):
    def test_compares_distinct_complete_plans_and_records_winner(self):
        seen = []

        def size(table, keys, **options):
            seen.append(tuple(keys))
            return 50 if keys and keys[0] == "heavy" else 100

        with patch("oparq.planning._encoded_size", side_effect=size):
            plan = oparq.plan_sort(
                _table(), algorithm="portfolio", max_sort_columns=2,
                sample_rows=None, trial_sample_rows=120,
            )
        self.assertEqual(plan.sort_keys[0], "heavy")
        self.assertIn("portfolio selected", plan.note)
        self.assertEqual(plan.trial_evaluations, len(seen))
        self.assertEqual(len(seen), len(set(seen)))
        self.assertLessEqual(len(seen), 5)
        self.assertEqual(plan.baseline_score, 100)
        self.assertEqual(plan.estimated_score, 50)

    def test_keeps_input_order_when_trials_are_worse(self):
        with patch("oparq.planning._encoded_size",
                   side_effect=lambda table, keys, **options: 110 if keys else 100):
            plan = oparq.plan_sort(_table(), algorithm="portfolio")
        self.assertEqual(plan.sort_keys, ())
        self.assertEqual(plan.baseline_score, plan.estimated_score)
        self.assertIn("retained", plan.note)

    def test_budget_one_only_encodes_baseline(self):
        with patch("oparq.planning._encoded_size", return_value=100) as encode:
            plan = oparq.plan_sort(
                _table(), algorithm="portfolio", max_trial_evaluations=1,
            )
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(plan.trial_evaluations, 1)
        self.assertEqual(plan.sort_keys, ())

    def test_prefix_include_exclude_and_key_budget_are_respected(self):
        with patch("oparq.planning._encoded_size",
                   side_effect=lambda table, keys, **options: 50 if len(keys) > 1 else 100):
            plan = oparq.plan_sort(
                _table(), algorithm="portfolio", prefix=["prefix"],
                include=["heavy", "small"], exclude=["heavy"],
                max_sort_columns=2,
            )
        self.assertEqual(plan.sort_keys, ("prefix", "small"))
        self.assertEqual(plan.sorting_keys, plan.sort_keys)

    def test_requires_material_sample_improvement(self):
        with patch("oparq.planning._encoded_size",
                   side_effect=lambda table, keys, **options: 999 if keys else 1000):
            plan = oparq.plan_sort(_table(), algorithm="portfolio")
        self.assertEqual(plan.sort_keys, ())

    def test_auto_wide_schema_still_compares_with_input_order(self):
        table = pa.table({str(i): [j % (i + 2) for j in range(500)] for i in range(70)})
        with patch("oparq.planning._encoded_size", return_value=100):
            plan = oparq.plan_sort(table, algorithm="auto", max_trial_evaluations=2)
        self.assertEqual(plan.algorithm, "codec_fast")
        self.assertEqual(plan.sort_keys, ())
        self.assertEqual(plan.trial_evaluations, 2)


if __name__ == "__main__":
    unittest.main()
