from __future__ import annotations

import io
import json
import random
import tempfile
import threading
import time
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

import oparq
from oparq import dataset as dataset_module
from oparq.cli import main
from oparq.engines import require_duckdb
from oparq.execution import (
    CHECKPOINT_DIRECTORY, PEAK_FACTOR, TASK_OVERHEAD_BYTES, MemoryEstimator, Task,
    in_shard, parse_shard, parse_size, run_tasks, stale_temporaries,
)
from oparq.storage import (
    COMPRESSION_METADATA_KEY, compression_metadata, inventory_parquet, open_input, open_output,
)

PATHS = tuple(f"day={day}/hour={hour:02d}/part-{part}.parquet"
              for day in (1, 2) for hour in (0, 1, 2) for part in (0, 1))


def _table(seed, rows=600):
    order = list(range(rows))
    random.Random(seed).shuffle(order)
    return pa.table({
        "region": [index % 3 for index in order],
        "category": [index % 17 for index in order],
        "payload": [f"category-{index % 17:02d}-" + "x" * 90 for index in order],
        "row": order,
    })


def _plan(table, keys=("region", "category")):
    return oparq.RewritePlan(
        algorithm="weighted", sort_keys=tuple(keys), prefix_keys=(),
        column_types=tuple((field.name, str(field.type)) for field in table.schema),
        compression="zstd", compression_level=1, sampled_rows=10,
    )


def _write_source(filesystem, root, paths=PATHS, seed=0):
    for index, relative in enumerate(paths):
        table = _table(seed + index)
        table = table.replace_schema_metadata(
            {COMPRESSION_METADATA_KEY: compression_metadata("zstd", 1)})
        with open_output(f"{root}/{relative}", filesystem) as stream:
            pq.write_table(table, stream, compression="zstd", compression_level=1)


def _files(filesystem, root):
    infos = filesystem.get_file_info(fs.FileSelector(root, recursive=True, allow_not_found=True))
    found = {}
    for info in infos:
        if info.type == fs.FileType.File:
            with filesystem.open_input_stream(info.path) as stream:
                found[info.path[len(root) + 1:]] = stream.read()
    return found


def _data(filesystem, root):
    return {path: data for path, data in _files(filesystem, root).items()
            if not path.startswith("_oparq/")}


class HelperTests(unittest.TestCase):
    def test_sizes_accept_decimal_binary_and_byte_counts(self):
        self.assertEqual(parse_size("6GB"), 6_000_000_000)
        self.assertEqual(parse_size("6 GiB"), 6 * 1024 ** 3)
        self.assertEqual(parse_size("512mb"), 512_000_000)
        self.assertEqual(parse_size(4096), 4096)
        for bad in ("", "lots", "-1GB", 0, "0GB"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_size(bad)
        with self.assertRaises(TypeError):
            parse_size(True)

    def test_shards_are_disjoint_cover_every_path_and_are_stable(self):
        paths = [f"year=2026/day={day}/part-{part}.parquet" for day in range(40) for part in range(5)]
        owners = {path: [index for index in range(7) if in_shard(path, (index, 7))] for path in paths}
        self.assertTrue(all(len(found) == 1 for found in owners.values()))
        self.assertGreater(min(sum(found == [index] for found in owners.values())
                               for index in range(7)), 10)
        # A stable hash, not Python's per-process randomized hash(): these
        # values must never change, or a resumed shard would pick other files.
        self.assertEqual([next(index for index in range(4) if in_shard(name, (index, 4)))
                          for name in ("a.parquet", "b.parquet", "day=1/c.parquet")], [1, 3, 0])
        self.assertEqual(parse_shard("2/5"), (2, 5))
        self.assertEqual(parse_shard((0, 1)), (0, 1))
        for bad in ("5/5", "1", "a/b", "-1/3", (3, 2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_shard(bad)

    def test_stale_temporaries_match_only_named_outputs(self):
        names = [
            "d/.a.parquet.oparq-" + "0" * 32 + ".tmp",
            "d/.a.parquet.oparq-candidate-" + "f" * 32 + ".parquet",
            "d/.b.parquet.oparq-" + "1" * 32 + ".tmp",   # another file's output
            "d/.a.parquet.backup",                        # not an oparq temporary
        ]
        listing = [fs.FileInfo(name, fs.FileType.File) for name in names]
        self.assertEqual(stale_temporaries(listing, ["d/a.parquet"]), sorted(names[:2]))


class SchedulerTests(unittest.TestCase):
    class Pool:
        def bytes_allocated(self):
            return 0

    def _tasks(self, count, *, rows=100, duration=0.03, kind="arrow"):
        state = {"running": 0, "peak": 0, "at_first_observe": None}
        lock = threading.Lock()
        table = pa.table({"x": pa.array(range(rows), pa.int64())})

        def make(index):
            def run(observe):
                with lock:
                    state["running"] += 1
                    state["peak"] = max(state["peak"], state["running"])
                if kind != "copy":
                    with lock:
                        if state["at_first_observe"] is None:
                            state["at_first_observe"] = state["running"]
                    observe(table)
                time.sleep(duration)
                with lock:
                    state["running"] -= 1
                return index
            return Task(index=index, kind=kind, rows=rows, size=rows, schema_key="s", run=run)

        return [make(index) for index in range(count)], state, table

    def test_budget_bounds_concurrency_after_calibrating_alone(self):
        tasks, state, table = self._tasks(12)
        per_task = int(table.nbytes * PEAK_FACTOR) + TASK_OVERHEAD_BYTES
        estimator = MemoryEstimator()
        self.assertIsNone(estimator.estimate("s", table.num_rows))
        results = run_tasks(tasks, workers=8, budget=3 * per_task, estimator=estimator,
                            memory_pool=self.Pool(), poll_seconds=0.005)
        self.assertEqual(sorted(results), list(range(12)))
        # The unknown-size first file decoded with nothing else running.
        self.assertEqual(state["at_first_observe"], 1)
        self.assertEqual(estimator.estimate("s", table.num_rows), per_task)
        self.assertEqual(state["peak"], 3)

    def test_unlimited_budget_reaches_worker_count(self):
        tasks, state, _ = self._tasks(16)
        run_tasks(tasks, workers=4, budget=1 << 50, memory_pool=self.Pool(), poll_seconds=0.005)
        self.assertEqual(state["peak"], 4)

    def test_live_allocation_blocks_new_work(self):
        class Busy:
            def bytes_allocated(self):
                return 1 << 40

        tasks, state, _ = self._tasks(5)
        run_tasks(tasks, workers=4, budget=1 << 30, memory_pool=Busy(), poll_seconds=0.005)
        self.assertEqual(state["peak"], 1)

    def test_oversized_file_runs_alone_with_a_warning(self):
        tasks, state, _ = self._tasks(3, rows=10_000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_tasks(tasks, workers=3, budget=1, memory_pool=self.Pool(), poll_seconds=0.005)
        self.assertEqual(state["peak"], 1)
        self.assertTrue(any("running it alone" in str(item.message) for item in caught))

    def test_copies_do_not_wait_for_memory(self):
        tasks, state, _ = self._tasks(6, kind="copy")
        run_tasks(tasks, workers=6, budget=1, memory_pool=self.Pool(), poll_seconds=0.005)
        self.assertEqual(state["peak"], 6)

    def test_failure_stops_new_work_finishes_running_and_raises(self):
        started = []

        def make(index):
            def run(observe):
                started.append(index)
                time.sleep(0.02)
                if index == 0:
                    raise RuntimeError("boom")
                return index
            return Task(index=index, kind="copy", rows=1, size=100 - index, schema_key="s", run=run)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_tasks([make(index) for index in range(20)], workers=2, budget=1 << 40,
                      memory_pool=self.Pool(), poll_seconds=0.005)
        self.assertLess(len(started), 20)


class ParallelDatasetTests(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        _write_source(self.filesystem, "src")
        self.plan = _plan(_table(0))

    def rewrite(self, destination, **options):
        # Force re-encoding: on this small fixture a sorted file is not always
        # smaller, and the size guard would then (correctly) copy the source.
        options.setdefault("skip_unchanged", bool(not self.plan.sort_keys))
        return oparq.rewrite_dataset("s3://src", f"s3://{destination}", plan=self.plan,
                                     source_filesystem=self.filesystem,
                                     destination_filesystem=self.filesystem, **options)

    def test_parallel_output_is_byte_identical_to_serial(self):
        serial = self.rewrite("serial", checkpoint=False)
        parallel = self.rewrite("parallel", workers=4, memory_budget="1GB", checkpoint=False)
        self.assertEqual(serial.rewritten_files, len(PATHS))
        self.assertEqual([str(item.path).split("/", 1)[1] for item in parallel.files],
                         [str(item.path).split("/", 1)[1] for item in serial.files])
        self.assertEqual(_files(self.filesystem, "parallel"), _files(self.filesystem, "serial"))
        self.assertEqual(set(_files(self.filesystem, "serial")), set(PATHS))

    def test_shards_partition_the_tree_without_overlap(self):
        full = self.rewrite("full", checkpoint=False)
        counts = []
        for index in range(3):
            result = self.rewrite("sharded", shard=f"{index}/3", workers=2, memory_budget="1GB")
            self.assertEqual(result.shard, (index, 3))
            counts.append(len(result.files))
        self.assertEqual(sum(counts), len(full.files))
        self.assertEqual(_data(self.filesystem, "sharded"), _files(self.filesystem, "full"))

    def test_interrupted_run_resumes_without_redoing_finished_files(self):
        failing = PATHS[3]
        real = dataset_module.rewrite_file

        def flaky(source, destination, **options):
            if str(source).endswith(failing):
                raise OSError("network reset")
            return real(source, destination, **options)

        with patch("oparq.dataset.rewrite_file", side_effect=flaky):
            with self.assertRaisesRegex(OSError, "network reset"):
                self.rewrite("resume", workers=2, memory_budget="1GB")
        finished = set(_data(self.filesystem, "resume"))
        self.assertNotIn(failing, finished)
        self.assertTrue(finished)

        calls = []

        def counting(source, destination, **options):
            calls.append(str(source))
            return real(source, destination, **options)

        with patch("oparq.dataset.rewrite_file", side_effect=counting):
            result = self.rewrite("resume", workers=2, memory_budget="1GB")
        self.assertEqual(result.resumed_files, len(finished))
        self.assertEqual(len(calls), len(PATHS) - len(finished))
        self.assertTrue(all(item.resumed == (str(item.path)[len("resume/"):] in finished)
                            for item in result.files))
        self.assertTrue(all(item.action == "rewritten" for item in result.files))
        self.rewrite("reference", checkpoint=False)
        self.assertEqual(_data(self.filesystem, "resume"), _files(self.filesystem, "reference"))

        # A complete run resumes everything without reading any source footer.
        with patch("oparq.dataset.read_footers", return_value=[]) as footers:
            again = self.rewrite("resume")
        footers.assert_called_once_with(self.filesystem, [])
        self.assertEqual(again.resumed_files, len(PATHS))

    def test_changed_source_or_plan_invalidates_the_checkpoint(self):
        self.rewrite("out")
        changed = PATHS[0]
        with open_output(f"src/{changed}", self.filesystem, overwrite=True) as stream:
            pq.write_table(_table(99).replace_schema_metadata(
                {COMPRESSION_METADATA_KEY: compression_metadata("zstd", 1)}), stream,
                compression="zstd", compression_level=1)
        with self.assertRaisesRegex(FileExistsError, "no matching checkpoint"):
            self.rewrite("out")
        result = self.rewrite("out", overwrite=True)
        self.assertEqual(result.resumed_files, len(PATHS) - 1)
        self.assertEqual([item.resumed for item in result.files].count(False), 1)
        expected = oparq.read_parquet(f"src/{changed}", filesystem=self.filesystem).sort_by(
            [("region", "ascending"), ("category", "ascending")])
        self.assertEqual(oparq.read_parquet(f"out/{changed}", filesystem=self.filesystem).to_pydict(),
                         expected.to_pydict())

        self.plan = _plan(_table(0), keys=("category",))
        result = self.rewrite("out", overwrite=True)
        self.assertEqual(result.resumed_files, 0)

    def test_in_place_rewrite_resumes_instead_of_sorting_again(self):
        first = oparq.rewrite_dataset("s3://src", "s3://src", plan=self.plan, overwrite=True,
                                      source_filesystem=self.filesystem,
                                      destination_filesystem=self.filesystem, skip_unchanged=False)
        self.assertEqual(first.rewritten_files, len(PATHS))
        before = _data(self.filesystem, "src")
        with patch("oparq.dataset.rewrite_file", side_effect=AssertionError("rewritten again")):
            again = oparq.rewrite_dataset("s3://src", "s3://src", plan=self.plan,
                                          source_filesystem=self.filesystem,
                                          destination_filesystem=self.filesystem,
                                          skip_unchanged=False)
        self.assertEqual(again.resumed_files, len(PATHS))
        self.assertEqual(_data(self.filesystem, "src"), before)

    def test_missing_output_is_redone_even_with_a_marker(self):
        self.rewrite("out")
        self.filesystem.delete_file(f"out/{PATHS[1]}")
        result = self.rewrite("out")
        self.assertEqual(result.resumed_files, len(PATHS) - 1)
        self.assertIn(PATHS[1], _data(self.filesystem, "out"))

    def test_checkpoint_markers_are_hidden_from_readers_or_disabled(self):
        self.rewrite("marked")
        markers = [path for path in _files(self.filesystem, "marked") if path.startswith(CHECKPOINT_DIRECTORY)]
        self.assertEqual(len(markers), len(PATHS))
        payload = json.loads(_files(self.filesystem, "marked")[markers[0]])
        self.assertEqual(set(payload) >= {"source", "fingerprint", "action", "output_size"}, True)
        inventory = inventory_parquet("marked", self.filesystem)
        self.assertEqual(len(inventory.files), len(PATHS))
        self.rewrite("plain", checkpoint=False)
        self.assertEqual(set(_files(self.filesystem, "plain")), set(PATHS))

    def test_explicit_checkpoint_location(self):
        self.rewrite("out", checkpoint="s3://state/run-1")
        self.assertEqual(len(_files(self.filesystem, "state/run-1")), len(PATHS))
        self.assertEqual(set(_files(self.filesystem, "out")), set(PATHS))
        self.assertEqual(self.rewrite("out", checkpoint="s3://state/run-1").resumed_files, len(PATHS))

    def test_stale_temporaries_of_an_interrupted_run_are_removed(self):
        leftover = f"out/{PATHS[0].rsplit('/', 1)[0]}/.part-0.parquet.oparq-{'a' * 32}.tmp"
        unrelated = "out/.keep"
        for path in (leftover, unrelated):
            with open_output(path, self.filesystem) as stream:
                stream.write(b"partial")
        result = self.rewrite("out", workers=3, memory_budget="1GB")
        self.assertEqual(result.removed_temporaries, 1)
        self.assertEqual(self.filesystem.get_file_info(leftover).type, fs.FileType.NotFound)
        self.assertEqual(self.filesystem.get_file_info(unrelated).type, fs.FileType.File)

    def test_preflight_failure_in_parallel_mode_publishes_nothing(self):
        with open_output("src/day=3/bad.parquet", self.filesystem) as stream:
            pq.write_table(_table(5).drop_columns(["category"]), stream)
        with self.assertRaisesRegex(ValueError, "category"):
            self.rewrite("never", workers=4, memory_budget="1GB")
        self.assertEqual(self.filesystem.get_file_info("never").type, fs.FileType.NotFound)

    def test_progress_reports_every_processed_file(self):
        seen = []
        self.rewrite("out", workers=3, memory_budget="1GB",
                     progress=lambda result, done, total: seen.append((done, total)))
        self.assertEqual(seen, [(index, len(PATHS)) for index in range(1, len(PATHS) + 1)])

    def test_no_key_plan_copies_in_parallel(self):
        self.plan = _plan(_table(0), keys=())
        result = self.rewrite("copies", workers=4, memory_budget="1GB")
        self.assertEqual(result.copied_files, len(PATHS))
        self.assertEqual(_data(self.filesystem, "copies"), _files(self.filesystem, "src"))

    def test_invalid_execution_options_fail_before_reading(self):
        for options in ({"workers": -1}, {"shard": "3/3"}, {"io_threads": 0},
                        {"workers": 2, "memory_budget": "nope"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.rewrite("never", **options)
        self.assertEqual(self.filesystem.get_file_info("never").type, fs.FileType.NotFound)

    def test_single_file_rewrite_rejects_tree_only_options(self):
        with self.assertRaisesRegex(ValueError, "workers apply only"):
            oparq.rewrite(f"s3://src/{PATHS[0]}", "s3://x/out.parquet", plan=self.plan,
                          source_filesystem=self.filesystem, workers=2)


class DuckDBParallelTests(unittest.TestCase):
    def test_parallel_duckdb_matches_serial_arrow_values(self):
        try:
            require_duckdb()
        except ImportError:
            self.skipTest("DuckDB extra is not installed")
        filesystem = fs._MockFileSystem()
        _write_source(filesystem, "src", PATHS[:5])
        plan = _plan(_table(0))
        common = dict(plan=plan, source_filesystem=filesystem, destination_filesystem=filesystem)
        oparq.rewrite_dataset("s3://src", "s3://arrow", checkpoint=False, **common)
        with tempfile.TemporaryDirectory() as spill:
            oparq.rewrite_dataset("s3://src", "s3://duck", engine="duckdb", workers=3,
                                  memory_budget="64GB", memory_limit="256MB",
                                  temp_directory=spill, **common)
        for relative in PATHS[:5]:
            self.assertEqual(oparq.read_parquet(f"duck/{relative}", filesystem=filesystem).to_pydict(),
                             oparq.read_parquet(f"arrow/{relative}", filesystem=filesystem).to_pydict())


class LocalTreeTests(unittest.TestCase):
    def test_local_parallel_checkpoint_resume_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = fs.LocalFileSystem()
            _write_source(local, str(root / "src"), PATHS[:6])
            plan_path = root / "plan.json"
            _plan(_table(0)).save(plan_path)
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                main(["rewrite", str(root / "src"), str(root / "dst"), "--plan", str(plan_path),
                      "--workers", "3", "--memory-budget", "2GB", "--progress"])
            self.assertIn("files: 6;", out.getvalue())
            self.assertEqual(err.getvalue().count("\n"), 6)
            markers = list((root / "dst" / CHECKPOINT_DIRECTORY).iterdir())
            self.assertEqual(len(markers), 6)
            self.assertFalse([path for path in markers if path.name.endswith(".tmp")])
            out = io.StringIO()
            with redirect_stdout(out):
                main(["rewrite", str(root / "src"), str(root / "dst"), "--plan", str(plan_path),
                      "--workers", "0", "--memory-budget", "2GB"])
            self.assertIn("resumed=6", out.getvalue())
            out = io.StringIO()
            with redirect_stdout(out):
                for index in range(2):
                    main(["rewrite", str(root / "src"), str(root / "shards"), "--plan", str(plan_path),
                          "--shard", f"{index}/2", "--no-checkpoint"])
            self.assertEqual(sorted(path.relative_to(root / "shards").as_posix()
                                    for path in (root / "shards").rglob("*.parquet")), sorted(PATHS[:6]))
            self.assertFalse((root / "shards" / "_oparq").exists())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["rewrite", str(root / "src" / PATHS[0]), str(root / "one.parquet"),
                      "--plan", str(plan_path), "--workers", "2"])


if __name__ == "__main__":
    unittest.main()
