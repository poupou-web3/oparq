from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import oparq
from benchmarks import verify_order


class OrderVerificationTests(unittest.TestCase):
    def test_reports_bounded_shards_and_all_algorithm_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "shards"
            source.mkdir()
            for shard in range(5):
                pq.write_table(
                    pa.table({
                        "key": [index % 4 for index in range(80)],
                        "payload": ["long-payload-" + str(index % 4) for index in range(80)],
                        "row_id": [shard * 80 + index for index in range(80)],
                    }),
                    source / f"part-{shard}.parquet",
                )
            output = root / "order.json"
            exit_code = verify_order.main([
                str(source), "--rows", "60", "--sample-files", "3",
                "--row-group-size", "10", "--compression", "snappy",
                "--compression-level", "default", "--output", str(output),
            ])

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["sampled_rows"], 60)
            self.assertEqual(
                [Path(item["path"]).name for item in report["selected_files"]],
                ["part-0.parquet", "part-2.parquet", "part-4.parquet"],
            )
            self.assertEqual(
                [item["sampled_rows"] for item in report["selected_files"]],
                [20, 20, 20],
            )
            self.assertEqual(
                [item["requested_algorithm"] for item in report["results"]],
                ["none", "codec_fast", "portfolio"],
            )
            self.assertTrue(all(item["status"] == "pass" for item in report["results"]))
            self.assertTrue(all(item["row_groups"] == 6 for item in report["results"]))
            self.assertTrue(report["results"][0]["checks"]["input_order_preserved"])

    def test_arrow_sort_and_streaming_are_checked(self) -> None:
        table = pa.table({
            "key": [2, 1, 1, 2, 1, 2, 2, 1],
            "row_id": list(range(8)),
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sorted.parquet"
            result = oparq.write(
                table, path, algorithm="none", prefix=["key"],
                row_group_size=3, compression="snappy",
            )
            case = verify_order._verify_case(
                table, path, result, row_group_size=3, sort_backend="auto"
            )
            self.assertEqual(case["status"], "pass")
            self.assertTrue(case["stream_sort_exercised"])
            self.assertEqual(case["row_groups"], 3)
            self.assertTrue(case["checks"]["natural_permutation_matches_arrow"])
            self.assertTrue(case["checks"]["natural_keys_monotonic"])

            # Swapping equal-key rows preserves monotonic keys but breaks the
            # stable, complete row sequence promised by the plan.
            wrong = pq.read_table(path).take(pa.array([1, 0, 2, 3, 4, 5, 6, 7]))
            pq.write_table(wrong, path, row_group_size=3)
            corrupted = verify_order._verify_case(
                table, path, result, row_group_size=3, sort_backend="auto"
            )
            self.assertEqual(corrupted["status"], "fail")
            self.assertFalse(corrupted["checks"]["all_rows_match_plan"])
            self.assertTrue(corrupted["checks"]["natural_keys_monotonic"])

    def test_forced_prefix_checks_streaming_when_planners_keep_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for shard in range(3):
                pq.write_table(
                    pa.table({"constant": [0] * 20}),
                    source / f"part-{shard}.parquet",
                )
            report = verify_order.verify_source(
                source, rows=30, sample_files=3, row_group_size=5,
                compression="snappy", compression_level=None,
            )
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["streaming_checked"])
            self.assertEqual(report["streaming_probe"]["status"], "pass")
            self.assertTrue(report["streaming_probe"]["stream_sort_exercised"])


if __name__ == "__main__":
    unittest.main()
