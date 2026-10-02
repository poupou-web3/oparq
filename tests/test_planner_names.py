from __future__ import annotations

import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import oparq
from oparq.planning import _encoded_size


class AlgorithmNameTests(unittest.TestCase):
    def test_removed_aliases_are_rejected(self) -> None:
        table = pa.table({"key": [2, 1, 2, 1]})
        for name in ("clickhouse", "size", "greedy", "trial", "fast", "input", "codec-fast"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unknown algorithm"):
                oparq.plan_sort(table, algorithm=name)

    def test_canonical_names_remain_case_insensitive(self) -> None:
        plan = oparq.plan_sort(pa.table({"key": [2, 1, 2, 1]}), algorithm="CARDINALITY")
        self.assertEqual(plan.algorithm, "cardinality")
        self.assertEqual(plan.sort_keys, ("key",))


class TrialCompressionTests(unittest.TestCase):
    def test_mixed_codecs_receive_only_supported_explicit_levels(self) -> None:
        table = pa.table({"zstd_column": ["abc"] * 100, "snappy_column": ["def"] * 100})
        with patch("oparq.planning.pq.write_table", wraps=pq.write_table) as writer:
            size = _encoded_size(
                table, (), compression={"zstd_column": "zstd", "snappy_column": "snappy"},
                compression_level=5, null_placement="at_end",
            )
        self.assertGreater(size, 0)
        self.assertEqual(writer.call_args.kwargs["compression_level"], {"zstd_column": 5})

    def test_trial_preserves_per_column_levels(self) -> None:
        table = pa.table({"left": ["abc"] * 100, "right": ["def"] * 100})
        with patch("oparq.planning.pq.write_table", wraps=pq.write_table) as writer:
            _encoded_size(
                table, (), compression={"left": "zstd", "right": "zstd"},
                compression_level={"left": 1, "right": 7}, null_placement="at_end",
            )
        self.assertEqual(writer.call_args.kwargs["compression_level"], {"left": 1, "right": 7})

    def test_unspecified_trial_level_uses_codec_default(self) -> None:
        table = pa.table({"group": [index % 5 for index in range(100)], "payload": ["value"] * 100})
        with patch("oparq.planning.pq.write_table", wraps=pq.write_table) as writer:
            oparq.plan_sort(table, algorithm="codec", trial_sample_rows=100)
        self.assertGreater(writer.call_count, 0)
        self.assertTrue(all(call.kwargs["compression_level"] is None for call in writer.call_args_list))


if __name__ == "__main__":
    unittest.main()
