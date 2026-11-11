from __future__ import annotations

import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

import oparq
from oparq import compaction
from oparq.cli import main
from oparq.engines import require_duckdb
from oparq.storage import (
    COMPRESSION_METADATA_KEY, compression_metadata, inspect_compression, open_output,
)

PARTITIONS = ("day=1", "day=2", "day=3/hour=00")


def _table(seed, rows=400):
    rng = random.Random(seed)
    return pa.table({
        "region": [rng.randrange(4) for _ in range(rows)],
        "category": [rng.randrange(9) for _ in range(rows)],
        "payload": [f"text-{rng.randrange(30):02d}-" + "y" * 60 for _ in range(rows)],
        "origin": [f"{seed}:{index}" for index in range(rows)],
    })


def _write(filesystem, path, table, *, level=1, row_group_size=None):
    table = table.replace_schema_metadata({COMPRESSION_METADATA_KEY: compression_metadata("zstd", level)})
    with open_output(path, filesystem, overwrite=True) as stream:
        pq.write_table(table, stream, compression="zstd", compression_level=level,
                       row_group_size=row_group_size)


def _plan(table, keys=("region", "category"), prefix=()):
    return oparq.RewritePlan(
        algorithm="weighted", sort_keys=tuple(keys), prefix_keys=tuple(prefix),
        column_types=tuple((field.name, str(field.type)) for field in table.schema),
        compression="zstd", compression_level=1, sampled_rows=10)


def _data_files(filesystem, root):
    infos = filesystem.get_file_info(fs.FileSelector(root, recursive=True, allow_not_found=True))
    return sorted(info.path for info in infos if info.type == fs.FileType.File
                  and info.path.endswith(".parquet")
                  and not any(part.startswith((".", "_")) for part in info.path[len(root) + 1:].split("/")))


def _bytes(filesystem, root):
    return {path[len(root) + 1:]: filesystem.open_input_stream(path).read()
            for path in _data_files(filesystem, root)}


def _read(filesystem, paths):
    return pa.concat_tables([pq.read_table(path, filesystem=filesystem) for path in paths])


class CompactionFixture(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        self.sources = {}
        seed = 0
        for partition in PARTITIONS:
            self.sources[partition] = []
            for part in range(4):
                path = f"src/{partition}/part-{part}.parquet"
                _write(self.filesystem, path, _table(seed))
                self.sources[partition].append(path)
                seed += 1
        self.plan = _plan(_table(0))

    def compact(self, destination="dst", **options):
        options.setdefault("skip_unchanged", False)
        return oparq.rewrite_dataset("s3://src", f"s3://{destination}", plan=self.plan, scope="partition",
                                     source_filesystem=self.filesystem,
                                     destination_filesystem=self.filesystem, **options)

    def expected(self, partition, keys=("region", "category")):
        # Stable global order: file order, then row order, break key ties.
        combined = _read(self.filesystem, self.sources[partition])
        return combined.sort_by([(key, "ascending") for key in keys]) if keys else combined


class PartitionOrderTests(CompactionFixture):
    def test_each_partition_is_sorted_globally_across_its_files(self):
        result = self.compact()
        self.assertEqual(len(result.partitions), len(PARTITIONS))
        self.assertEqual(result.compacted_files, len(PARTITIONS))
        for partition in PARTITIONS:
            outputs = _data_files(self.filesystem, f"dst/{partition}")
            self.assertEqual(outputs, [f"dst/{partition}/part-00000.parquet"])
            actual = _read(self.filesystem, outputs)
            self.assertEqual(actual.to_pydict(), self.expected(partition).to_pydict())
        summary = {item.partition: item for item in result.partitions}
        self.assertEqual(summary["day=1"].sources, tuple(self.sources["day=1"]))
        self.assertEqual(summary["day=1"].action, "compacted")
        self.assertIn("4 source file(s) into 1", summary["day=1"].plan.note)
        self.assertGreater(summary["day=1"].permutation_seconds, -1)

    def test_target_size_splits_into_ordered_files_of_similar_size(self):
        whole = self.compact("one")
        size = whole.partitions[0].file_size
        result = self.compact("split", target_file_size=size // 4, row_group_size=100)
        for partition in result.partitions:
            outputs = [item.path for item in partition.outputs]
            self.assertGreaterEqual(len(outputs), 3)
            self.assertEqual(outputs, [f"{partition.path}/part-{index:05d}.parquet"
                                       for index in range(len(outputs))])
            actual = _read(self.filesystem, outputs)
            self.assertEqual(actual.to_pydict(), self.expected(partition.partition).to_pydict())
            sizes = [item.file_size for item in partition.outputs]
            # Tiny files are dominated by footers; the bound is tested below.
            self.assertLessEqual(max(sizes), 3 * (size // 4))
            for path in outputs:
                footer = pq.read_metadata(path, filesystem=self.filesystem)
                groups = [footer.row_group(index).num_rows for index in range(footer.num_row_groups)]
                self.assertTrue(all(rows == 100 for rows in groups[:-1]))
                self.assertEqual(footer.row_group(0).sorting_columns[0].column_index, 0)

    def test_rolling_writer_keeps_files_near_the_target(self):
        filesystem = fs._MockFileSystem()
        rng = random.Random(1)
        table = pa.table({"x": [rng.randrange(1 << 40) for _ in range(200_000)]})
        writer = compaction._RollingWriter(filesystem, "out", table.schema, {"compression": "none"},
                                           target_bytes=200_000, total_rows=table.num_rows,
                                           group_rows=5_000)
        for offset in range(0, table.num_rows, 7_000):
            writer.write(table.slice(offset, 7_000))
        files = writer.finish()
        sizes = [filesystem.get_file_info(temporary).size for temporary, _, _ in files]
        self.assertEqual(sum(rows for _, _, rows in files), table.num_rows)
        self.assertGreaterEqual(len(files), 6)
        self.assertLessEqual(max(sizes), 1.25 * 200_000 + 5_000 * 8)
        self.assertGreaterEqual(min(sizes[:-1]), 0.5 * 200_000)
        self.assertTrue(all(name.startswith("out/.part-") and name.endswith(".tmp")
                            for name, _, _ in files))
        writer.abort()
        self.assertFalse(filesystem.get_file_info(fs.FileSelector("out", allow_not_found=True)))

    def _roll(self, table, *, target, cap=None, group_rows=5_000, options=None, chunk=7_000):
        filesystem = fs._MockFileSystem()
        writer = compaction._RollingWriter(
            filesystem, "out", table.schema, options or {"compression": "zstd"},
            target_bytes=target, max_bytes=cap, total_rows=table.num_rows, group_rows=group_rows)
        for offset in range(0, table.num_rows, chunk):
            writer.write(table.slice(offset, chunk))
        files = writer.finish()
        sizes = [filesystem.get_file_info(temporary).size for temporary, _, _ in files]
        self.assertEqual(sum(rows for _, _, rows in files), table.num_rows)
        return writer, sizes

    def test_cap_holds_when_density_rises_along_the_order(self):
        # A long compressible run, then incompressible rows: early density
        # badly underestimates the rest, which used to oversize files.
        rng = random.Random(3)
        rows = 120_000
        table = pa.table({"x": [0] * (rows // 2) + [rng.randrange(1 << 62) for _ in range(rows // 2)]})
        writer, sizes = self._roll(table, target=100_000, group_rows=20_000)
        self.assertLessEqual(max(sizes), 125_000)
        self.assertGreater(writer.measured_groups, 0)
        self.assertGreaterEqual(len(sizes), 4)

    def test_far_from_the_cap_nothing_is_encoded_twice(self):
        table = pa.table({"x": list(range(10_000))})
        writer, sizes = self._roll(table, target=64 << 20)
        self.assertEqual(len(sizes), 1)
        self.assertEqual(writer.measured_groups, 0)

    def test_a_row_group_larger_than_the_cap_is_split(self):
        rng = random.Random(5)
        table = pa.table({"x": [rng.randrange(1 << 62) for _ in range(50_000)]})
        writer, sizes = self._roll(table, target=60_000, group_rows=50_000, chunk=50_000)
        self.assertLessEqual(max(sizes), 75_000)
        self.assertIsNotNone(writer.reduced_group_rows)
        self.assertLess(writer.reduced_group_rows, 50_000)

    def test_one_row_above_the_cap_is_written_rather_than_looping(self):
        table = pa.table({"x": ["z" * 5_000 for _ in range(4)]})
        writer, sizes = self._roll(table, target=100, cap=200, group_rows=1,
                                   options={"compression": "none"}, chunk=1)
        self.assertEqual(len(sizes), 4)

    def test_row_groups_are_not_shrunk_below_the_minimum(self):
        rng = random.Random(9)
        table = pa.table({"x": [rng.randrange(1 << 62) for _ in range(20_000)]})
        writer, sizes = self._roll(table, target=1_000, group_rows=10_000, chunk=10_000)
        self.assertEqual(writer.reduced_group_rows, compaction.MIN_GROUP_ROWS)
        self.assertLessEqual(len(sizes), 20_000 // compaction.MIN_GROUP_ROWS + 1)

    @unittest.skipUnless(oparq.storage.REBUILDS_BLOOM_FILTERS, "bloom filters need PyArrow 25+")
    def test_close_time_bloom_filters_count_toward_the_cap(self):
        rng = random.Random(8)
        rows = 200_000
        table = pa.table({"x": [rng.randrange(1 << 30) for _ in range(rows)]})
        options = {"compression": "zstd",
                   "bloom_filter_options": {"x": {"ndv": 100_000, "fpp": 0.001}}}
        writer, sizes = self._roll(table, target=500_000, group_rows=10_000, options=options)
        self.assertLessEqual(max(sizes), 625_000)

    def test_cap_must_not_be_below_target(self):
        with self.assertRaisesRegex(ValueError, "max_file_size"):
            compaction._RollingWriter(fs._MockFileSystem(), "out", pa.schema([("x", pa.int64())]), {},
                                      target_bytes=100, max_bytes=50, total_rows=1, group_rows=1)

    def test_settings_metadata_and_provenance_are_preserved(self):
        for partition, paths in self.sources.items():
            for path in paths:
                _write(self.filesystem, path, pq.read_table(path, filesystem=self.filesystem), level=5)
        result = self.compact()
        output = result.partitions[0].outputs[0].path
        profile = inspect_compression(output, self.filesystem)
        self.assertEqual(set(profile.levels.values()), {5})
        metadata = pq.read_schema(output, filesystem=self.filesystem).metadata
        self.assertIn(b"oparq.sort_plan", metadata)

    def test_no_keys_concatenates_in_file_order(self):
        self.plan = _plan(_table(0), keys=())
        result = self.compact(skip_unchanged=True)
        for partition in PARTITIONS:
            outputs = _data_files(self.filesystem, f"dst/{partition}")
            self.assertEqual(len(outputs), 1)
            self.assertEqual(_read(self.filesystem, outputs).to_pydict(),
                             self.expected(partition, keys=()).to_pydict())
        self.assertTrue(all(item.action == "compacted" for item in result.partitions))

    def test_single_file_without_keys_is_copied_byte_for_byte(self):
        self.plan = _plan(_table(0), keys=())
        filesystem = self.filesystem
        _write(filesystem, "lonely/day=9/only.parquet", _table(77))
        result = oparq.rewrite_dataset("s3://lonely", "s3://lonely-out", plan=self.plan, scope="partition",
                                       source_filesystem=filesystem, destination_filesystem=filesystem)
        self.assertEqual(result.partitions[0].action, "copied")
        self.assertEqual(_bytes(filesystem, "lonely-out"), _bytes(filesystem, "lonely"))

    def test_empty_partition_writes_one_empty_file(self):
        filesystem = fs._MockFileSystem()
        for index in range(2):
            _write(filesystem, f"empty/day=1/p{index}.parquet", _table(index).slice(0, 0))
        result = oparq.rewrite_dataset("s3://empty", "s3://empty-out", plan=self.plan, scope="partition",
                                       source_filesystem=filesystem, destination_filesystem=filesystem,
                                       skip_unchanged=False)
        (output,) = result.partitions[0].outputs
        self.assertEqual(pq.read_metadata(output.path, filesystem=filesystem).num_rows, 0)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        rng = random.Random(4)
        for part in range(3):
            # Several columns follow long runs and compress to almost nothing.
            # A stable sort by a 256-valued random key scatters every run, and
            # costs far more on those columns than the key column gains.
            rows = 20_000
            run = [index // 50 + part * 1000 for index in range(rows)]
            table = pa.table({
                "run": [f"value-{value:05d}" for value in run],
                "alpha": [f"alpha-{value * 7919 % 100003:06d}" for value in run],
                "beta": [value * 31 % 9973 for value in run],
                "gamma": [value / 3 for value in run],
                "noise": [rng.randrange(256) for _ in range(rows)],
            })
            _write(self.filesystem, f"src/day=1/f{part}.parquet", table)
        self.table = table.slice(0, 0)

    def test_a_larger_result_keeps_the_original_files(self):
        result = oparq.rewrite_dataset("s3://src", "s3://dst", plan=_plan(self.table, keys=("noise",)),
                                       scope="partition", source_filesystem=self.filesystem,
                                       destination_filesystem=self.filesystem)
        (partition,) = result.partitions
        self.assertEqual(partition.action, "copied")
        self.assertIn("kept original files", partition.plan.note)
        self.assertEqual(_bytes(self.filesystem, "dst"), _bytes(self.filesystem, "src"))
        leftovers = self.filesystem.get_file_info(fs.FileSelector("dst", recursive=True))
        self.assertFalse([info for info in leftovers if ".oparq-" in info.path])

    def test_a_required_prefix_is_honored_even_if_larger(self):
        plan = _plan(self.table, keys=("noise",), prefix=("noise",))
        result = oparq.rewrite_dataset("s3://src", "s3://dst", plan=plan, scope="partition",
                                       source_filesystem=self.filesystem,
                                       destination_filesystem=self.filesystem)
        self.assertEqual(result.partitions[0].action, "compacted")


class SafetyTests(CompactionFixture):
    def test_overlapping_trees_are_refused(self):
        for destination in ("s3://src", "s3://"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "new tree"):
                oparq.rewrite_dataset("s3://src", destination, plan=self.plan, scope="partition",
                                      source_filesystem=self.filesystem,
                                      destination_filesystem=self.filesystem)
        _write(self.filesystem, "outer/inner/day=1/a.parquet", _table(1))
        with self.assertRaisesRegex(ValueError, "new tree"):
            oparq.rewrite_dataset("s3://outer/inner", "s3://outer", plan=self.plan, scope="partition",
                                  source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        with self.assertRaisesRegex(ValueError, "inside the source"):
            self.compact("src/nested")

    def test_mixed_schemas_fail_before_anything_is_published(self):
        _write(self.filesystem, "src/day=2/odd.parquet", _table(50).drop_columns(["payload"]))
        with self.assertRaisesRegex(ValueError, "mixes physical schemas"):
            self.compact()
        self.assertEqual(self.filesystem.get_file_info("dst").type, fs.FileType.NotFound)

    def test_existing_destination_data_needs_overwrite_which_replaces_the_partition(self):
        stale = "dst/day=1/part-00007.parquet"
        _write(self.filesystem, stale, _table(99))
        with self.assertRaisesRegex(FileExistsError, "overwrite=True"):
            self.compact()
        self.assertEqual(self.filesystem.get_file_info("dst/day=2").type, fs.FileType.NotFound)
        self.compact(overwrite=True)
        self.assertEqual(self.filesystem.get_file_info(stale).type, fs.FileType.NotFound)
        self.assertEqual(_data_files(self.filesystem, "dst/day=1"), ["dst/day=1/part-00000.parquet"])

    def test_max_file_size_option(self):
        # Large enough that the cap is many times a footer on every PyArrow.
        for part in range(3):
            _write(self.filesystem, f"big/day=1/f{part}.parquet", _table(100 + part, rows=10_000))
        common = dict(plan=self.plan, scope="partition", skip_unchanged=False,
                      source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        whole = oparq.rewrite_dataset("s3://big", "s3://big-one", checkpoint=False, **common)
        size = whole.partitions[0].file_size
        capped = oparq.rewrite_dataset("s3://big", "s3://big-capped", target_file_size=size // 4,
                                       max_file_size=size // 3, row_group_size=500, **common)
        outputs = capped.partitions[0].outputs
        self.assertGreaterEqual(len(outputs), 3)
        self.assertLessEqual(max(item.file_size for item in outputs), size // 3)
        with self.assertRaisesRegex(ValueError, "must not be smaller"):
            self.compact("never", target_file_size="2MB", max_file_size="1MB")
        with self.assertRaisesRegex(ValueError, "only to scope='partition'"):
            oparq.rewrite_dataset("s3://src", "s3://dst", plan=self.plan, max_file_size="1GB",
                                  source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        # A different cap is a different output: checkpoints are not reused.
        again = oparq.rewrite_dataset("s3://big", "s3://big-capped", target_file_size=size // 4,
                                      max_file_size=size // 2, row_group_size=500, overwrite=True,
                                      **common)
        self.assertFalse(any(item.resumed for item in again.partitions))

    def test_invalid_scope_options(self):
        with self.assertRaisesRegex(ValueError, "scope must be"):
            oparq.rewrite_dataset("s3://src", "s3://dst", plan=self.plan, scope="bucket",
                                  source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        with self.assertRaisesRegex(ValueError, "only to scope='partition'"):
            oparq.rewrite_dataset("s3://src", "s3://dst", plan=self.plan, target_file_size="1GB",
                                  source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        with self.assertRaisesRegex(ValueError, "apply only to partition-preserving"):
            oparq.rewrite("s3://src/day=1/part-0.parquet", "s3://x/out.parquet", plan=self.plan,
                          source_filesystem=self.filesystem, scope="partition")


class ResumeTests(CompactionFixture):
    def test_rerun_resumes_every_partition_without_reading_sources(self):
        first = self.compact()
        with patch("oparq.compaction.read_footers", return_value=[]) as footers:
            again = self.compact()
        footers.assert_called_once_with(self.filesystem, [])
        self.assertTrue(all(item.resumed for item in again.partitions))
        self.assertEqual(again.resumed_files, len(first.files))
        self.assertEqual([item.file_size for item in again.files], [item.file_size for item in first.files])

    def test_changed_partition_is_redone_with_overwrite(self):
        self.compact()
        _write(self.filesystem, self.sources["day=2"][1], _table(500))
        with self.assertRaisesRegex(FileExistsError, "day=2"):
            self.compact()
        result = self.compact(overwrite=True)
        redone = [item.partition for item in result.partitions if not item.resumed]
        self.assertEqual(redone, ["day=2"])
        actual = _read(self.filesystem, _data_files(self.filesystem, "dst/day=2"))
        self.assertEqual(actual.to_pydict(), self.expected("day=2").to_pydict())

    def test_interrupted_publication_is_cleaned_up_on_rerun(self):
        real = compaction.WriteResult
        calls = []

        def crash_after_first_move(*args, **kwargs):
            calls.append(kwargs.get("path"))
            if len(calls) == 2:
                raise OSError("worker lost")
            return real(*args, **kwargs)

        whole = self.compact("probe", checkpoint=False)
        size = whole.partitions[0].file_size
        with patch("oparq.compaction.WriteResult", side_effect=crash_after_first_move):
            with self.assertRaisesRegex(OSError, "worker lost"):
                self.compact(target_file_size=size // 3, row_group_size=100)
        # The crash hit after the second move: two outputs are published, the
        # remaining temporaries were removed.
        self.assertEqual(_data_files(self.filesystem, "dst/day=1"),
                         ["dst/day=1/part-00000.parquet", "dst/day=1/part-00001.parquet"])
        hidden = self.filesystem.get_file_info(fs.FileSelector("dst/day=1"))
        self.assertFalse([info.path for info in hidden if ".oparq-" in info.path])
        with self.assertRaises(FileExistsError):
            self.compact("dst", checkpoint=False, target_file_size=size // 3, row_group_size=100)
        # A rerun without overwrite replaces the partial set it published.
        result = self.compact(target_file_size=size // 3, row_group_size=100)
        self.compact("reference", target_file_size=size // 3, row_group_size=100, checkpoint=False)
        self.assertGreater(len(_data_files(self.filesystem, "reference/day=1")), 2)
        self.assertEqual(_bytes(self.filesystem, "dst"), _bytes(self.filesystem, "reference"))
        self.assertFalse(any(item.resumed for item in result.partitions
                             if item.partition == result.partitions[0].partition))
        leftovers = self.filesystem.get_file_info(fs.FileSelector("dst", recursive=True))
        self.assertFalse([info.path for info in leftovers if ".oparq-" in info.path])


class ExecutionTests(CompactionFixture):
    def test_parallel_shards_and_prefetch_do_not_change_bytes(self):
        self.compact("serial", prefetch=0, checkpoint=False)
        self.compact("parallel", workers=3, memory_budget="1GB", checkpoint=False)
        counts = [len(self.compact("sharded", shard=f"{index}/2").partitions) for index in range(2)]
        self.assertEqual(sum(counts), len(PARTITIONS))
        reference = _bytes(self.filesystem, "serial")
        self.assertEqual(_bytes(self.filesystem, "parallel"), reference)
        self.assertEqual(_bytes(self.filesystem, "sharded"), reference)

    def test_progress_reports_partitions(self):
        seen = []
        self.compact(progress=lambda result, done, total: seen.append((result.partition, done, total)))
        self.assertEqual([item[1:] for item in seen], [(1, 3), (2, 3), (3, 3)])

    def test_duckdb_matches_arrow(self):
        try:
            require_duckdb()
        except ImportError:
            self.skipTest("DuckDB extra is not installed")
        self.compact("arrow", checkpoint=False)
        with tempfile.TemporaryDirectory() as spill:
            self.compact("duck", engine="duckdb", workers=2, memory_budget="64GB",
                         memory_limit="256MB", temp_directory=spill)
        for partition in PARTITIONS:
            self.assertEqual(
                _read(self.filesystem, _data_files(self.filesystem, f"duck/{partition}")).to_pydict(),
                _read(self.filesystem, _data_files(self.filesystem, f"arrow/{partition}")).to_pydict())

    def test_cli_partition_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = fs.LocalFileSystem()
            for index in range(5):
                _write(local, str(root / "src" / "day=1" / f"f{index}.parquet"), _table(index))
            self.plan.save(root / "plan.json")
            out = io.StringIO()
            with redirect_stdout(out):
                main(["rewrite", str(root / "src"), str(root / "dst"), "--plan", str(root / "plan.json"),
                      "--scope", "partition", "--target-file-size", "64MB",
                      "--max-file-size", "80MB", "--force-rewrite"])
            self.assertIn("partitions: 1; source files: 5 -> output files: 1", out.getvalue())
            self.assertEqual([path.name for path in (root / "dst" / "day=1").iterdir()], ["part-00000.parquet"])


if __name__ == "__main__":
    unittest.main()
