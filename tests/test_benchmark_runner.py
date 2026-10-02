from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks import run_suite


class FileSamplingTests(unittest.TestCase):
    def test_split_sort_timings_are_reported_when_available(self) -> None:
        result = SimpleNamespace(
            planning_seconds=1.0,
            sorting_seconds=3.0,
            permutation_seconds=2.0,
            gathering_seconds=1.0,
            write_seconds=4.0,
        )
        timings = run_suite._timings(result)
        self.assertEqual(timings["permutation_seconds"], 2.0)
        self.assertEqual(timings["gathering_seconds"], 1.0)
        self.assertEqual(timings["sorting_seconds"], 3.0)
        self.assertEqual(timings["measured_total_seconds"], 8.0)

    def test_row_weighted_file_selection_and_balanced_budget(self) -> None:
        entries = [
            {"path": f"file-{index}.parquet", "rows": rows}
            for index, rows in enumerate((2, 10, 20))
        ]
        plan = run_suite._sample_file_plan(entries, row_limit=12, sample_files=3)
        self.assertEqual(
            plan,
            [
                {"path": "file-0.parquet", "source_rows": 2,
                 "row_offset": 0, "sampled_rows": 2},
                {"path": "file-1.parquet", "source_rows": 10,
                 "row_offset": 0, "sampled_rows": 5},
                {"path": "file-2.parquet", "source_rows": 20,
                 "row_offset": 0, "sampled_rows": 5},
            ],
        )

    def test_skewed_footer_counts_still_choose_distinct_spread_files(self) -> None:
        entries = [
            {"path": f"file-{index}.parquet", "rows": rows}
            for index, rows in enumerate((1, 1, 100, 1, 1))
        ]
        plan = run_suite._sample_file_plan(entries, row_limit=9, sample_files=3)
        self.assertEqual([item["path"] for item in plan], [
            "file-0.parquet", "file-2.parquet", "file-4.parquet"
        ])
        self.assertEqual([item["sampled_rows"] for item in plan], [1, 7, 1])

    def test_sample_file_count_cannot_exceed_row_budget(self) -> None:
        entries = [
            {"path": f"file-{index}.parquet", "rows": 10}
            for index in range(5)
        ]
        plan = run_suite._sample_file_plan(entries, row_limit=2, sample_files=5)
        self.assertEqual(len(plan), 2)
        self.assertEqual(sum(item["sampled_rows"] for item in plan), 2)
        self.assertTrue(all(item["sampled_rows"] == 1 for item in plan))

    def test_integration_records_exact_files_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "sharded"
            dataset.mkdir()
            for index in range(5):
                pq.write_table(
                    pa.table({"id": list(range(index * 10, index * 10 + 10))}),
                    dataset / f"part-{index}.parquet",
                )
            output = root / "report.json"
            with patch.object(run_suite.oparq, "benchmark", return_value=[]) as benchmark:
                exit_code = run_suite.main([
                    "--root", str(root), "--datasets", "sharded",
                    "--algorithms", "none", "--sample-files", "3",
                    "--rows", "9", "--output", str(output),
                ])

            self.assertEqual(exit_code, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["configuration"]["sample_files"], 3)
            entry = report["datasets"][0]
            self.assertEqual(entry["input"]["benchmarked_rows"], 9)
            self.assertEqual(entry["input"]["sampling"]["mode"], "files")
            files = entry["input"]["sampling"]["files"]
            self.assertEqual(
                [Path(item["path"]).name for item in files],
                ["part-0.parquet", "part-2.parquet", "part-4.parquet"],
            )
            self.assertEqual([item["sampled_rows"] for item in files], [3, 3, 3])
            self.assertEqual([item["row_offset"] for item in files], [0, 0, 0])
            self.assertEqual(
                benchmark.call_args.args[0]["id"].to_pylist(),
                [0, 1, 2, 20, 21, 22, 40, 41, 42],
            )

    def test_default_head_semantics_and_full_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for index in range(3):
                pq.write_table(
                    pa.table({"id": [index * 10, index * 10 + 1]}),
                    source / f"part-{index}.parquet",
                )
            self.assertEqual(
                run_suite._read_table(source, 3)["id"].to_pylist(),
                [0, 1, 10],
            )
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                run_suite.main(["--full", "--sample-files", "2"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
