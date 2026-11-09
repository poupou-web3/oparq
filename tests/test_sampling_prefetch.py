from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

import oparq
from oparq import execution
from oparq.cli import main
from oparq.engines import require_duckdb
from oparq.execution import Prefetcher
from oparq.learning import sample_dataset, validate_plan
from oparq.storage import (
    COMPRESSION_METADATA_KEY, compression_metadata, inspect_compression, inventory_parquet,
    open_output, read_footers, spread_indices,
)


def _write(filesystem, path, table, *, level=1, row_group_size=None, provenance=True):
    if provenance:
        table = table.replace_schema_metadata(
            {COMPRESSION_METADATA_KEY: compression_metadata("zstd", level)})
    with open_output(path, filesystem) as stream:
        pq.write_table(table, stream, compression="zstd", compression_level=level,
                       row_group_size=row_group_size)


def _numbered(start, rows):
    """Row ``i`` of the file holds ``start + i``: windows reveal where they began."""

    values = list(range(start, start + rows))
    return pa.table({"position": values, "group": [value % 7 for value in values],
                     "text": [f"value-{value % 11}" for value in values]})


class FooterTests(unittest.TestCase):
    def test_parallel_footers_keep_input_order_and_raise_the_first_error(self):
        filesystem = fs._MockFileSystem()
        paths = [f"b/part-{index:03d}.parquet" for index in range(40)]
        for index, path in enumerate(paths):
            _write(filesystem, path, _numbered(0, index + 1))
        footers = read_footers(filesystem, paths, threads=8)
        self.assertEqual([footer.num_rows for footer in footers], list(range(1, 41)))
        with self.assertRaises(FileNotFoundError):
            read_footers(filesystem, paths[:5] + ["b/missing.parquet"] + paths[5:], threads=8)

    def test_spread_indices_cover_the_range_evenly(self):
        self.assertEqual(spread_indices(10, 2), [2, 7])
        self.assertEqual(spread_indices(3, 10), [0, 1, 2])
        self.assertEqual(spread_indices(1000, 4), [125, 375, 625, 875])

    def test_inspect_compression_can_sample_and_reuse_footers(self):
        filesystem = fs._MockFileSystem()
        for index in range(20):
            _write(filesystem, f"b/day={index}/a.parquet", _numbered(0, 5),
                   provenance=index != 13)
        with self.assertRaisesRegex(ValueError, "compression levels"):
            oparq.storage.resolve_compression(inspect_compression("b", filesystem),
                                              compression="preserve", compression_level="preserve")
        # A spread sample of 4 files does not include day=13.
        sampled = inspect_compression("b", filesystem, max_files=4)
        self.assertEqual(sampled.levels["position"], 1)
        inventory = inventory_parquet("b", filesystem)
        footers = dict(zip(inventory.files, read_footers(filesystem, inventory.files)))
        with patch("oparq.storage.read_footers", side_effect=AssertionError("re-read")):
            inspect_compression(inventory, footers=footers)
        with self.assertRaises(ValueError):
            inspect_compression("b", filesystem, max_files=0)

    def test_inventory_keeps_listing_sizes(self):
        filesystem = fs._MockFileSystem()
        _write(filesystem, "b/a.parquet", _numbered(0, 50))
        inventory = inventory_parquet("b", filesystem)
        self.assertEqual(inventory.info("b/a.parquet").size,
                         filesystem.get_file_info("b/a.parquet").size)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        for index in range(40):
            _write(self.filesystem, f"tree/day={index:02d}/part.parquet",
                   _numbered(index * 10_000, 1000), row_group_size=100)

    def test_row_group_windows_are_consecutive_seeded_and_not_only_heads(self):
        head = sample_dataset("tree", filesystem=self.filesystem, sample_rows=320,
                              sample_files=8, sample_method="head")
        windows = sample_dataset("tree", filesystem=self.filesystem, sample_rows=320,
                                 sample_files=8)
        again = sample_dataset("tree", filesystem=self.filesystem, sample_rows=320, sample_files=8)
        other = sample_dataset("tree", filesystem=self.filesystem, sample_rows=320,
                               sample_files=8, seed=7)
        self.assertEqual(windows.num_rows, 320)
        self.assertTrue(windows.equals(again))
        self.assertFalse(windows.equals(other))
        offsets = [value % 10_000 for value in head["position"].to_pylist()[::40]]
        self.assertEqual(offsets, [0] * 8)
        starts = windows["position"].to_pylist()[::40]
        self.assertTrue(any(value % 10_000 for value in starts))
        for chunk in range(8):
            values = windows["position"].to_pylist()[chunk * 40:(chunk + 1) * 40]
            self.assertEqual(values, list(range(values[0], values[0] + 40)))
            self.assertEqual(values[0] % 100, 0)  # Windows start at a row-group boundary.

    def test_window_wraps_to_earlier_groups_when_a_late_group_is_short(self):
        filesystem = fs._MockFileSystem()
        _write(filesystem, "one/a.parquet", _numbered(0, 250), row_group_size=100)
        sample = sample_dataset("one", filesystem=filesystem, sample_rows=240, sample_files=1, seed=3)
        self.assertEqual(sample.num_rows, 240)
        self.assertEqual(len(set(sample["position"].to_pylist())), 240)

    def test_invalid_sampling_options(self):
        with self.assertRaises(ValueError):
            sample_dataset("tree", filesystem=self.filesystem, sample_method="random")
        for options in ({"footer_files": 0}, {"validation_files": -1}, {"validation_rows": 0},
                        {"sample_files": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                oparq.fit_dataset("tree", filesystem=self.filesystem, **options)

    def test_fit_samples_footers_and_validates_on_held_out_files(self):
        with patch("oparq.storage.read_footers", wraps=oparq.storage.read_footers) as footers:
            plan = oparq.fit_dataset("tree", filesystem=self.filesystem, sample_rows=2_000,
                                     sample_files=8, footer_files=6, validation_files=3,
                                     validation_rows=300, algorithms=("weighted",))
        read = sum(len(call.args[1]) for call in footers.call_args_list)
        self.assertLess(read, 40)
        self.assertEqual(len(plan.validation), 3)
        trained = {f"day={index:02d}/part.parquet" for index in spread_indices(40, 8)}
        for record in plan.validation:
            self.assertNotIn(record["path"], trained)
            self.assertEqual(record["rows"], 300)
            self.assertAlmostEqual(record["savings"],
                                   1 - record["plan_bytes"] / record["baseline_bytes"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan.save(path)
            self.assertEqual(oparq.RewritePlan.load(path).validation, plan.validation)

    def test_every_file_used_for_training_leaves_no_validation(self):
        plan = oparq.fit_dataset("tree", filesystem=self.filesystem, sample_rows=2_000,
                                 sample_files=64, algorithms=("weighted",))
        self.assertEqual(plan.validation, ())

    def test_plans_without_validation_still_load(self):
        plan = oparq.fit(_numbered(0, 500), algorithms=("weighted",), compression="zstd",
                         compression_level=1)
        payload = plan.as_dict()
        payload.pop("validation")
        self.assertEqual(oparq.RewritePlan.from_dict(payload).validation, ())

    def test_validate_plan_keeps_the_prefix_in_the_baseline(self):
        table = _numbered(0, 700)
        plan = oparq.fit(table, algorithms=("weighted",), prefix=("group",),
                         compression="zstd", compression_level=1, min_improvement=0)
        (record,) = validate_plan([("w", table)], plan)
        self.assertEqual(record["path"], "w")
        self.assertGreater(record["baseline_bytes"], 0)

    def test_cli_fit_reports_held_out_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            local = fs.LocalFileSystem()
            root = Path(directory)
            for index in range(12):
                _write(local, str(root / "src" / f"part-{index:02d}.parquet"),
                       _numbered(index * 1000, 400), row_group_size=100)
            out = io.StringIO()
            with redirect_stdout(out):
                main(["fit", str(root / "src"), str(root / "plan.json"), "--algorithms", "weighted",
                      "--sample-files", "4", "--validation-files", "2", "--validation-rows", "200",
                      "--footer-files", "0", "--sample-rows", "800"])
            self.assertIn("held-out windows: 2 files", out.getvalue())
            self.assertEqual(len(json.loads((root / "plan.json").read_text())["validation"]), 2)


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        self.paths = [f"src/day={index}/part.parquet" for index in range(6)]
        for index, path in enumerate(self.paths):
            table = _numbered(index * 1000, 3000).take(pa.array(list(range(2999, -1, -1))))
            _write(self.filesystem, path, table)
        schema = pq.read_schema(self.paths[0], filesystem=self.filesystem)
        self.plan = oparq.RewritePlan(
            algorithm="weighted", sort_keys=("group", "text"), prefix_keys=(),
            column_types=tuple((field.name, str(field.type)) for field in schema),
            compression="zstd", compression_level=1)

    def rewrite(self, destination, **options):
        return oparq.rewrite_dataset("s3://src", f"s3://{destination}", plan=self.plan,
                                     source_filesystem=self.filesystem,
                                     destination_filesystem=self.filesystem,
                                     skip_unchanged=False, checkpoint=False, **options)

    def outputs(self, root):
        return {path: self.filesystem.open_input_stream(path.replace("src", root, 1)).read()
                for path in self.paths}

    def test_prefetched_output_is_byte_identical(self):
        self.rewrite("direct", prefetch=0)
        with patch.object(execution, "PREFETCH_PART_BYTES", 1000):
            with patch.object(Prefetcher, "take", autospec=True,
                              side_effect=Prefetcher.take) as take:
                self.rewrite("ranged", prefetch=3, workers=3, memory_budget="1GB")
        self.assertEqual(take.call_count, len(self.paths))
        self.rewrite("serial")  # A mock filesystem is remote: default prefetch.
        self.assertEqual(self.outputs("ranged"), self.outputs("direct"))
        self.assertEqual(self.outputs("serial"), self.outputs("direct"))

    def test_ranged_download_assembles_the_exact_object(self):
        expected = self.filesystem.open_input_stream(self.paths[0]).read()
        with patch.object(execution, "PREFETCH_PART_BYTES", 777):
            prefetcher = Prefetcher(self.filesystem, self.paths[:1], depth=1,
                                    sizes={self.paths[0]: len(expected)})
            try:
                buffer = prefetcher.take(self.paths[0])
            finally:
                prefetcher.close()
        self.assertEqual(buffer.to_pybytes(), expected)

    def test_failed_or_unplanned_downloads_fall_back_to_direct_reads(self):
        prefetcher = Prefetcher(self.filesystem, ["src/missing.parquet", self.paths[0]], depth=1)
        try:
            self.assertIsNone(prefetcher.take("src/missing.parquet"))
            self.assertIsNone(prefetcher.take("src/not-queued.parquet"))
            self.assertIsNotNone(prefetcher.take(self.paths[0]))
        finally:
            prefetcher.close()
        with patch.object(Prefetcher, "_download", side_effect=OSError("reset")):
            result = self.rewrite("fallback", workers=2, memory_budget="1GB")
        self.assertEqual(result.rewritten_files, len(self.paths))
        self.rewrite("direct", prefetch=0)
        self.assertEqual(self.outputs("fallback"), self.outputs("direct"))

    def test_local_sources_do_not_prefetch_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            local = fs.LocalFileSystem()
            _write(local, f"{directory}/src/a.parquet", _numbered(0, 100))
            with patch("oparq.dataset.Prefetcher", side_effect=AssertionError("prefetched")):
                oparq.rewrite_dataset(f"{directory}/src", f"{directory}/out", plan=self.plan,
                                      workers=2, memory_budget="1GB", skip_unchanged=False)

    def test_invalid_prefetch(self):
        for value in (-1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.rewrite("never", prefetch=value)
        with self.assertRaises(ValueError):
            Prefetcher(self.filesystem, [], depth=0)

    def test_duckdb_reads_prefetched_bytes(self):
        try:
            require_duckdb()
        except ImportError:
            self.skipTest("DuckDB extra is not installed")
        with tempfile.TemporaryDirectory() as spill:
            self.rewrite("duck", engine="duckdb", prefetch=2, workers=2, memory_budget="64GB",
                         memory_limit="256MB", temp_directory=spill)
        self.rewrite("arrow", prefetch=0)
        for path in self.paths:
            self.assertEqual(
                oparq.read_parquet(path.replace("src", "duck", 1), filesystem=self.filesystem).to_pydict(),
                oparq.read_parquet(path.replace("src", "arrow", 1), filesystem=self.filesystem).to_pydict())


if __name__ == "__main__":
    unittest.main()
