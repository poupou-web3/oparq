from __future__ import annotations

import importlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import oparq
from oparq import cli
from oparq.models import BenchmarkResult, SortPlan, WriteResult


def _plan(algorithm: str = "none") -> SortPlan:
    return SortPlan(
        requested_algorithm=algorithm,
        algorithm=algorithm,
        sort_keys=(),
        prefix_keys=(),
        profiles=(),
        total_rows=3,
        sampled_rows=3,
    )


class TimingTests(unittest.TestCase):
    def test_optimize_measures_planning_and_sorting_separately(self) -> None:
        core = importlib.import_module("oparq.core")
        table = pa.table({"value": [3, 1, 2]})
        plan = _plan()
        events: list[str] = []
        ticks = iter((10.0, 11.25, 20.0, 23.5))

        def clock() -> float:
            events.append("clock")
            return next(ticks)

        def plan_table(*args: object, **kwargs: object) -> SortPlan:
            events.append("plan")
            return plan

        def sort_table(*args: object, **kwargs: object) -> pa.Table:
            events.append("sort")
            return table

        with (
            patch.object(core, "perf_counter", side_effect=clock),
            patch.object(core, "plan_sort", side_effect=plan_table),
            patch.object(core, "apply_plan", side_effect=sort_table),
        ):
            result = core.optimize(table)

        self.assertEqual(
            events,
            ["clock", "plan", "clock", "clock", "sort", "clock"],
        )
        self.assertEqual(result.planning_seconds, 1.25)
        self.assertEqual(result.sort_seconds, 3.5)
        self.assertEqual(result.planning_and_sort_seconds, 4.75)

    def test_benchmark_uses_none_as_baseline_when_it_is_not_first(self) -> None:
        benchmark_module = importlib.import_module("oparq.benchmark")
        table = pa.table({"value": [3, 1, 2]})

        def fake_write(
            value: pa.Table,
            destination: Path,
            **options: object,
        ) -> WriteResult:
            algorithm = str(options["algorithm"])
            return WriteResult(
                path=destination,
                file_size=80 if algorithm == "weighted" else 100,
                plan=_plan(algorithm),
                planning_seconds=1.0,
                sort_seconds=2.0,
                write_seconds=3.0,
            )

        with patch.object(benchmark_module, "write", side_effect=fake_write):
            results = benchmark_module.benchmark(
                table,
                algorithms=("weighted", "none"),
            )

        self.assertAlmostEqual(results[0].savings_fraction, 0.2)
        self.assertEqual(results[1].savings_fraction, 0.0)
        self.assertEqual(results[0].planning_seconds, 1.0)
        self.assertEqual(results[0].sort_seconds, 2.0)
        self.assertEqual(results[0].planning_and_sort_seconds, 3.0)

    def test_rewrite_cli_reports_the_split_timings(self) -> None:
        result = WriteResult(
            path=Path("output.parquet"),
            file_size=1024,
            plan=_plan(),
            planning_seconds=1.25,
            sort_seconds=2.5,
            write_seconds=3.75,
            permutation_seconds=1.0,
            gathering_seconds=1.5,
        )
        output = io.StringIO()
        with (
            patch.object(cli, "rewrite", return_value=result),
            redirect_stdout(output),
        ):
            cli.main(["rewrite", "input.parquet", "output.parquet"])

        rendered = output.getvalue()
        self.assertIn("plan: 1.25s", rendered)
        self.assertIn("permutation: 1.00s", rendered)
        self.assertIn("gather: 1.50s", rendered)
        self.assertIn("sort: 2.50s", rendered)
        self.assertIn("write: 3.75s", rendered)
        self.assertNotIn("plan + sort:", rendered)

    def test_benchmark_cli_reports_the_split_timings(self) -> None:
        table = pa.table({"value": [1, 2, 3]})
        result = BenchmarkResult(
            algorithm="none",
            resolved_algorithm="none",
            sort_keys=(),
            size_bytes=1024,
            planning_seconds=1.25,
            sort_seconds=2.5,
            write_seconds=3.75,
            permutation_seconds=1.0,
            gathering_seconds=1.5,
        )
        output = io.StringIO()
        with (
            patch.object(cli, "read_parquet", return_value=table),
            patch.object(cli, "benchmark", return_value=(result,)),
            redirect_stdout(output),
        ):
            cli.main(["benchmark", "input.parquet", "--algorithms", "none",
                      "--compression", "zstd", "--compression-level", "default"])

        rendered = output.getvalue()
        self.assertIn("plan=  1.25s", rendered)
        self.assertIn("perm=  1.00s", rendered)
        self.assertIn("gather=  1.50s", rendered)
        self.assertIn("sort=  2.50s", rendered)
        self.assertIn("write=  3.75s", rendered)
        self.assertNotIn("plan+sort=", rendered)

    def test_existing_output_is_rejected_before_optimization(self) -> None:
        io_module = importlib.import_module("oparq.io")
        table = pa.table({"value": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.parquet"
            path.touch()
            with (
                patch.object(io_module, "plan_sort") as optimize,
                self.assertRaises(FileExistsError),
            ):
                io_module.write(table, path)
        optimize.assert_not_called()

    def test_write_accepts_uncompressed_codec_spellings(self) -> None:
        io_module = importlib.import_module("oparq.io")
        table = pa.table({"value": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, compression in enumerate((None, "none")):
                path = root / f"uncompressed-{index}.parquet"
                io_module.write(
                    table,
                    path,
                    algorithm="none",
                    compression=compression,
                )
                metadata = pq.read_metadata(path)
                self.assertEqual(
                    metadata.row_group(0).column(0).compression,
                    "UNCOMPRESSED",
                )

    def test_apply_plan_uses_its_recorded_null_placement_by_default(self) -> None:
        core = importlib.import_module("oparq.core")
        table = pa.table({"value": [2, None, 1]})
        plan = SortPlan(
            requested_algorithm="none",
            algorithm="none",
            sort_keys=("value",),
            prefix_keys=("value",),
            profiles=(),
            total_rows=3,
            sampled_rows=3,
            null_placement="at_start",
        )

        default_order = core.apply_plan(table, plan)
        overridden_order = core.apply_plan(table, plan, null_placement="at_end")

        self.assertEqual(default_order["value"].to_pylist(), [None, 1, 2])
        self.assertEqual(overridden_order["value"].to_pylist(), [1, 2, None])

    def test_frequency_plan_only_advertises_its_natural_prefix(self) -> None:
        plan = SortPlan(
            requested_algorithm="frequency",
            algorithm="frequency",
            sort_keys=("partition", "value"),
            prefix_keys=("partition",),
            profiles=(),
            total_rows=3,
            sampled_rows=3,
            value_order="frequency",
        )

        self.assertEqual(plan.sorting_keys, ("partition",))
        self.assertEqual(plan.sort_order, [("partition", "ascending")])

    def test_codec_fast_strictly_respects_the_trial_budget(self) -> None:
        rows = 200
        table = pa.table(
            {
                f"key_{column}": [
                    (chr(97 + column) * 32) + str(index % (column + 2))
                    for index in range(rows)
                ]
                for column in range(6)
            }
        )

        plan = oparq.plan_sort(
            table,
            algorithm="codec_fast",
            sample_rows=None,
            trial_sample_rows=rows,
            fast_candidate_count=6,
            max_trial_evaluations=2,
            min_trial_improvement=0.0,
        )

        # The input-order baseline is one of the two allowed trial encodes.
        self.assertEqual(plan.trial_evaluations, 2)

    def test_failed_codec_trial_also_consumes_the_trial_budget(self) -> None:
        table = pa.table({
            "left": [index % 3 for index in range(100)],
            "right": [index % 5 for index in range(100)],
        })

        def encoded_size(table: pa.Table, keys: tuple[str, ...], **options: object) -> int:
            if keys:
                raise pa.ArrowInvalid("unsupported trial key")
            return 100

        with patch("oparq.planning._encoded_size", side_effect=encoded_size) as trial:
            plan = oparq.plan_sort(
                table, algorithm="codec_fast", sample_rows=None,
                max_trial_evaluations=2,
            )

        self.assertEqual(trial.call_count, 2)
        self.assertEqual(plan.trial_evaluations, 2)
        self.assertEqual(plan.sort_keys, ())

    def test_codec_fast_skips_trial_encoding_without_candidates(self) -> None:
        table = pa.table({"constant": ["same"] * 100})

        plan = oparq.plan_sort(
            table,
            algorithm="codec_fast",
            sample_rows=None,
            trial_sample_rows=100,
        )

        self.assertEqual(plan.sort_keys, ())
        self.assertEqual(plan.trial_evaluations, 0)
        self.assertIsNone(plan.baseline_score)
        self.assertIsNone(plan.estimated_score)


if __name__ == "__main__":
    unittest.main()
