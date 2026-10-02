from __future__ import annotations

import json
import base64
import posixpath
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

import oparq
from oparq.dataset import rewrite_file
from oparq.engines import require_duckdb
from oparq.storage import (
    REBUILDS_BLOOM_FILTERS,
    COMPRESSION_METADATA_KEY,
    compression_metadata,
    effective_compression_level,
    inspect_compression,
    inventory_parquet,
    open_input,
    open_output,
)


def _fixture(rows=1200):
    indices = list(range(rows))
    random.Random(2026).shuffle(indices)
    return pa.table({
        "region": [index % 3 for index in indices],
        "category": [index % 19 for index in indices],
        "payload": [f"category-{index % 19:02d}-" + "x" * 140 for index in indices],
        "source_row": list(range(rows)),
    }).replace_schema_metadata({b"application": b"preserve this"})


def _fixed(table, *, keys=(), prefix=(), compression="zstd", level=1):
    return oparq.RewritePlan(
        algorithm="none" if not keys else "weighted",
        sort_keys=tuple(keys), prefix_keys=tuple(prefix),
        column_types=tuple((field.name, str(field.type)) for field in table.schema),
        compression=compression, compression_level=level, sampled_rows=10,
    )


def _source(table, path, filesystem, *, compression="zstd", level=1, provenance=True):
    metadata = dict(table.schema.metadata or {})
    if provenance:
        metadata[COMPRESSION_METADATA_KEY] = compression_metadata(compression, level)
    table = table.replace_schema_metadata(metadata)
    with open_output(path, filesystem) as stream:
        pq.write_table(table, stream, compression=compression,
                       compression_level=effective_compression_level(compression, level))


def _bytes(path, filesystem):
    with open_input(path, filesystem) as stream:
        return stream.read()


class ReusablePlanTests(unittest.TestCase):
    def test_learn_save_load_and_apply_to_new_rows_without_planning(self):
        sample = _fixture()
        plan = oparq.fit(
            sample, algorithms=("weighted",), prefix=("region",),
            compression="zstd", compression_level=1, row_group_size=256,
            min_improvement=0,
        )
        self.assertEqual(plan.prefix_keys, ("region",))
        self.assertEqual(plan.sort_keys[0], "region")
        self.assertEqual(plan.sampled_rows, sample.num_rows)
        self.assertGreaterEqual(len(plan.evaluations), 2)
        other = _fixture(1837)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "plans" / "order.json"
            plan.save(manifest)
            loaded = oparq.RewritePlan.load(manifest)
            self.assertEqual(loaded.as_dict(), plan.as_dict())
            with patch("oparq.io.plan_sort", side_effect=AssertionError("replanned")):
                result = oparq.write(other, Path(directory) / "output.parquet",
                                     plan=loaded, row_group_size=317)
            actual = pq.read_table(result.path)
        expected = oparq.apply_plan(other, loaded.for_table(other))
        self.assertEqual(actual.to_pydict(), expected.to_pydict())
        self.assertEqual(result.planning_seconds, 0)
        self.assertEqual(result.plan.total_rows, other.num_rows)
        self.assertEqual(actual.schema.metadata[b"application"], b"preserve this")

    def test_missing_column_and_type_drift_fail_before_output_publication(self):
        original = _fixture(10)
        plan = _fixed(original, keys=("region",))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "output.parquet"
            for changed in (
                original.drop(["category"]),
                original.set_column(0, "region", original["region"].cast(pa.string())),
            ):
                with self.subTest(schema=changed.schema):
                    with self.assertRaisesRegex(ValueError, "missing|changed type"):
                        oparq.write(changed, destination, plan=plan)
                    self.assertFalse(destination.exists())

    def test_dataset_schema_validation_precedes_any_file_publication(self):
        filesystem = fs._MockFileSystem()
        original = _fixture(10)
        _source(original, "bucket/day=1/a.parquet", filesystem)
        _source(original.drop(["category"]), "bucket/day=2/b.parquet", filesystem)
        with self.assertRaisesRegex(ValueError, "missing"):
            oparq.rewrite_dataset(
                "s3://bucket", "s3://new-bucket", plan=_fixed(original, keys=("region",)),
                source_filesystem=filesystem, destination_filesystem=filesystem,
            )
        self.assertEqual(filesystem.get_file_info("new-bucket").type, fs.FileType.NotFound)

    def test_serialized_schema_accepts_list_names_but_rejects_real_nested_changes(self):
        schema = pa.schema([
            pa.field("key", pa.int64()),
            pa.field("nested", pa.list_(pa.field("custom-name", pa.struct([
                pa.field("label", pa.dictionary(pa.int8(), pa.string(), ordered=True)),
            ]), nullable=False))),
        ])
        table = pa.Table.from_batches([], schema=schema)
        plan = replace(_fixed(table, keys=("key",)),
                       schema_base64=base64.b64encode(schema.serialize().to_pybytes()).decode("ascii"))
        cosmetic = pa.schema([
            schema.field("key"),
            pa.field("nested", pa.list_(pa.field("element", schema.field("nested").type.value_type,
                                                 nullable=False))),
        ])
        plan.for_table(pa.Table.from_batches([], schema=cosmetic))
        for changed_type in (
            pa.list_(pa.field("element", cosmetic.field("nested").type.value_type, nullable=True)),
            pa.list_(pa.field("element", pa.struct([
                pa.field("label", pa.dictionary(pa.int16(), pa.string(), ordered=True)),
            ]), nullable=False)),
            pa.list_(pa.field("element", pa.struct([
                pa.field("renamed", pa.dictionary(pa.int8(), pa.string(), ordered=True)),
            ]), nullable=False)),
            pa.list_(pa.field("element", pa.struct([
                pa.field("label", pa.dictionary(pa.int8(), pa.string(), ordered=False)),
            ]), nullable=False)),
        ):
            changed = cosmetic.set(1, pa.field("nested", changed_type))
            with self.subTest(type=changed_type), self.assertRaisesRegex(ValueError, "changed type"):
                plan.for_table(pa.Table.from_batches([], schema=changed))

    def test_legacy_nested_list_types_do_not_normalize_struct_field_names(self):
        original = pa.table({"key": [1], "value": [[{"label": [1]}]]})
        plan = _fixed(original, keys=("key",))
        element_type = pa.list_(pa.field("element", pa.struct([
            pa.field("label", pa.list_(pa.field("element", pa.int64()))),
        ])))
        compatible = pa.Table.from_batches([], schema=pa.schema([
            original.schema.field("key"), pa.field("value", element_type),
        ]))
        plan.for_table(compatible)
        for dtype in (pa.list_(pa.struct([("renamed", pa.list_(pa.int64()))])),
                      pa.large_list(pa.struct([("label", pa.list_(pa.int64()))])),
                      pa.list_(pa.struct([("label", pa.list_(pa.int32()))]))):
            changed = pa.Table.from_batches([], schema=compatible.schema.set(1, pa.field("value", dtype)))
            with self.subTest(type=dtype), self.assertRaisesRegex(ValueError, "changed type"):
                plan.for_table(changed)

    def test_malformed_plan_payloads_raise_value_errors(self):
        for payload in ([], {"format_version": 1, "algorithm": "none"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                oparq.RewritePlan.from_dict(payload)

    def test_saved_schema_payload_cannot_disagree_with_type_records(self):
        table = _fixture(5)
        plan = replace(_fixed(table), schema_base64="invalid-base64")
        with self.assertRaisesRegex(ValueError, "invalid serialized schema"):
            oparq.RewritePlan.from_dict(plan.as_dict())
        changed = table.schema.set(0, pa.field("region", pa.int32()))
        plan = replace(plan, schema_base64=base64.b64encode(changed.serialize().to_pybytes()).decode("ascii"))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            oparq.RewritePlan.from_dict(plan.as_dict())


class FileTreeRewriteTests(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        self.table = _fixture(36)

    def test_no_sort_plan_skips_in_place_without_requiring_unknown_source_level(self):
        path = "s3://bucket/day=1/a.parquet"
        _source(self.table, path, self.filesystem, provenance=False)
        before = _bytes(path, self.filesystem)
        with patch("oparq.dataset.read_parquet_source", side_effect=AssertionError("decoded")):
            result = rewrite_file(path, path, plan=_fixed(self.table),
                                  source_filesystem=self.filesystem,
                                  destination_filesystem=self.filesystem)
        self.assertEqual(result.action, "skipped")
        self.assertEqual(_bytes(path, self.filesystem), before)
        self.assertEqual(result.sort_seconds, 0)

    def test_no_sort_bucket_copy_keeps_exact_bytes_and_hive_directories(self):
        paths = ("gs://old/year=2025/month=01/a.parquet", "gs://old/year=2026/month=02/b.parquet")
        for path in paths:
            _source(self.table, path, self.filesystem, provenance=False)
        target_fs = fs._MockFileSystem()
        with patch("oparq.dataset.read_parquet_source", side_effect=AssertionError("decoded")):
            result = oparq.rewrite_dataset(
                "gs://old", "s3://new", plan=_fixed(self.table),
                source_filesystem=self.filesystem, destination_filesystem=target_fs,
            )
        self.assertEqual((result.copied_files, result.rewritten_files, result.skipped_files), (2, 0, 0))
        inventory = inventory_parquet("s3://new", target_fs)
        self.assertEqual(tuple(inventory.relative_path(path) for path in inventory.files), (
            "year=2025/month=01/a.parquet", "year=2026/month=02/b.parquet",
        ))
        for path in paths:
            self.assertEqual(_bytes(path, self.filesystem),
                             _bytes(path.replace("gs://old", "s3://new"), target_fs))

    def test_sorting_preserves_source_codec_and_recorded_level(self):
        codecs = {"region": "zstd", "category": "snappy", "payload": "zstd", "source_row": "snappy"}
        source = "s3://bucket/source.parquet"
        _source(self.table, source, self.filesystem, compression=codecs, level=1)
        result = rewrite_file(
            source, "s3://bucket/output.parquet", plan=_fixed(self.table, keys=("region", "category")),
            source_filesystem=self.filesystem, destination_filesystem=self.filesystem,
            skip_unchanged=False,
        )
        profile = inspect_compression(result.path, self.filesystem)
        self.assertEqual(profile.codecs, {name: (codec,) for name, codec in codecs.items()})
        self.assertEqual(profile.levels, {"region": 1, "payload": 1})
        actual = oparq.read_parquet(result.path, filesystem=self.filesystem)
        self.assertEqual(actual.to_pydict(), self.table.sort_by([("region", "ascending"), ("category", "ascending")]).to_pydict())

    def test_explicit_codec_or_level_change_forces_no_key_rewrite(self):
        source = "s3://bucket/source.parquet"
        _source(self.table, source, self.filesystem, compression="zstd", level=1)
        for name, settings in (("codec", {"compression": "snappy"}),
                               ("level", {"compression_level": 3})):
            with self.subTest(change=name):
                result = rewrite_file(
                    source, f"s3://bucket/{name}.parquet", plan=_fixed(self.table),
                    source_filesystem=self.filesystem, destination_filesystem=self.filesystem,
                    **settings,
                )
                self.assertEqual(result.action, "rewritten")
                self.assertNotEqual(_bytes(source, self.filesystem), _bytes(result.path, self.filesystem))
                self.assertEqual(oparq.read_parquet(result.path, filesystem=self.filesystem).to_pydict(), self.table.to_pydict())
        self.assertEqual(inspect_compression("bucket/codec.parquet", self.filesystem).codecs["region"], ("snappy",))
        self.assertEqual(inspect_compression("bucket/level.parquet", self.filesystem).levels["region"], 3)

    def test_unknown_source_level_blocks_a_real_sorted_rewrite_before_writing(self):
        source = "s3://bucket/source.parquet"
        _source(self.table, source, self.filesystem, provenance=False)
        with self.assertRaisesRegex(ValueError, "compression levels"):
            rewrite_file(source, "s3://bucket/new.parquet", plan=_fixed(self.table, keys=("region",)),
                         source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        self.assertEqual(self.filesystem.get_file_info("bucket/new.parquet").type, fs.FileType.NotFound)

    def test_preflight_unknown_level_in_later_file_publishes_nothing(self):
        _source(self.table, "bucket/day=1/a.parquet", self.filesystem)
        _source(self.table, "bucket/day=2/b.parquet", self.filesystem, provenance=False)
        with self.assertRaisesRegex(ValueError, "compression levels"):
            oparq.rewrite_dataset("s3://bucket", "s3://new", plan=_fixed(self.table, keys=("region",)),
                                  source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        self.assertEqual(self.filesystem.get_file_info("new").type, fs.FileType.NotFound)

    def test_preflight_collision_in_later_file_preserves_every_object(self):
        for path in ("bucket/day=1/a.parquet", "bucket/day=2/b.parquet"):
            _source(self.table, path, self.filesystem)
        existing = "new/day=2/b.parquet"
        _source(_fixture(5), existing, self.filesystem)
        original = _bytes(existing, self.filesystem)
        with self.assertRaises(FileExistsError):
            oparq.rewrite_dataset("s3://bucket", "s3://new", plan=_fixed(self.table, keys=("region",)),
                                  source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        self.assertEqual(_bytes(existing, self.filesystem), original)
        self.assertEqual(self.filesystem.get_file_info("new/day=1/a.parquet").type, fs.FileType.NotFound)

    def test_ancestor_destination_cannot_overwrite_another_input_file(self):
        paths = ("bucket/tree/a.parquet", "bucket/tree/tree/a.parquet")
        for path in paths:
            _source(self.table, path, self.filesystem)
        before = {path: _bytes(path, self.filesystem) for path in paths}
        with self.assertRaisesRegex(ValueError, "another source file"):
            oparq.rewrite_dataset("s3://bucket/tree", "s3://bucket",
                                  plan=_fixed(self.table, keys=("region",)),
                                  source_filesystem=self.filesystem, destination_filesystem=self.filesystem,
                                  overwrite=True)
        for path in paths:
            self.assertEqual(_bytes(path, self.filesystem), before[path])
        self.assertEqual(self.filesystem.get_file_info("bucket/a.parquet").type, fs.FileType.NotFound)

    def test_tree_relative_manifest_levels_are_applied_per_source_file(self):
        paths = ("bucket/day=1/a.parquet", "bucket/day=2/a.parquet")
        levels = (1, 5)
        for path, level in zip(paths, levels):
            _source(self.table, path, self.filesystem, level=level, provenance=False)
        manifest = {"compression": "zstd", "files": {
            "day=1/a.parquet": {"compression_level": 1},
            "day=2/a.parquet": {"compression_level": 5},
        }}
        result = oparq.rewrite_dataset("s3://bucket", "s3://new", plan=_fixed(self.table, keys=("region",)),
                                       source_filesystem=self.filesystem, destination_filesystem=self.filesystem,
                                       compression_manifest=manifest, skip_unchanged=False)
        self.assertEqual(result.rewritten_files, 2)
        for path, level in zip(paths, levels):
            actual = inspect_compression(path.replace("bucket/", "new/"), self.filesystem)
            self.assertEqual(actual.levels, {field.name: level for field in self.table.schema})

    def test_no_sort_copy_reports_measured_copy_time(self):
        _source(self.table, "bucket/source.parquet", self.filesystem, provenance=False)
        with patch("oparq.dataset.perf_counter", side_effect=(10.0, 10.25)):
            result = rewrite_file("s3://bucket/source.parquet", "s3://new/output.parquet",
                                  plan=_fixed(self.table), source_filesystem=self.filesystem,
                                  destination_filesystem=self.filesystem)
        self.assertEqual(result.action, "copied")
        self.assertEqual(result.write_seconds, 0.25)


class CompressionSizeGuardTests(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        self.table = _fixture(36)
        self.source = "bucket/source.parquet"
        _source(self.table, self.source, self.filesystem)
        self.original = _bytes(self.source, self.filesystem)
        self.plan = _fixed(self.table, keys=("region",))

    def _writer(self, payload):
        def write_candidate(table, destination, *, filesystem, plan, **options):
            with open_output(destination, filesystem, overwrite=options.get("overwrite", False)) as stream:
                stream.write(payload)
            return oparq.WriteResult(path=destination, file_size=len(payload), plan=plan,
                                     planning_seconds=1.0, sort_seconds=2.0, write_seconds=3.0,
                                     permutation_seconds=0.75, gathering_seconds=1.25)
        return write_candidate

    def _assert_no_candidates(self):
        paths = [info.path for info in self.filesystem.get_file_info(fs.FileSelector("bucket", recursive=True))
                 if info.type == fs.FileType.File]
        self.assertFalse(any("oparq-candidate" in path for path in paths), paths)

    def test_larger_or_equal_candidate_copies_original_and_reports_work(self):
        for extra in (0, 100):
            destination = f"bucket/output-{extra}.parquet"
            payload = b"x" * (len(self.original) + extra)
            with patch("oparq.io.write", side_effect=self._writer(payload)):
                result = rewrite_file(self.source, destination, plan=self.plan,
                                      source_filesystem=self.filesystem,
                                      destination_filesystem=self.filesystem)
            self.assertEqual(result.action, "copied")
            self.assertEqual(_bytes(destination, self.filesystem), self.original)
            self.assertEqual(result.plan.sort_keys, ())
            self.assertIn("kept original bytes", result.plan.note)
            self.assertEqual(result.planning_seconds, 1.0)
            self.assertEqual(result.sort_seconds, 2.0)
            self.assertGreaterEqual(result.write_seconds, 3.0)
            self.assertEqual(result.permutation_seconds, 0.75)
            self.assertEqual(result.gathering_seconds, 1.25)
            self._assert_no_candidates()

    def test_larger_in_place_candidate_never_replaces_original(self):
        with patch("oparq.io.write", side_effect=self._writer(b"x" * (len(self.original) + 100))):
            result = rewrite_file(self.source, self.source, plan=self.plan, overwrite=True,
                                  source_filesystem=self.filesystem,
                                  destination_filesystem=self.filesystem)
        self.assertEqual(result.action, "skipped")
        self.assertEqual(_bytes(self.source, self.filesystem), self.original)
        self.assertGreaterEqual(result.write_seconds, 3.0)
        self._assert_no_candidates()

    def test_candidate_name_is_hidden_from_dataset_readers(self):
        # An interrupted rewrite must not leave a *.parquet object that Arrow,
        # Spark or a later inventory would read as duplicate table rows.
        seen = []

        def record(table, destination, **options):
            seen.append(str(destination))
            return self._writer(b"small candidate")(table, destination, **options)

        with patch("oparq.io.write", side_effect=record):
            rewrite_file(self.source, "bucket/output.parquet", plan=self.plan,
                         source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        self.assertEqual(len(seen), 1)
        self.assertTrue(posixpath.basename(seen[0]).startswith("."), seen)
        self.assertEqual(posixpath.dirname(seen[0]), "bucket")

    def test_smaller_candidate_is_published(self):
        with patch("oparq.io.write", side_effect=self._writer(b"small candidate")):
            result = rewrite_file(self.source, "bucket/output.parquet", plan=self.plan,
                                  source_filesystem=self.filesystem,
                                  destination_filesystem=self.filesystem)
        self.assertEqual(result.action, "rewritten")
        self.assertEqual(result.path, "bucket/output.parquet")
        self.assertEqual(result.plan.sort_keys, ("region",))
        self.assertEqual(_bytes(result.path, self.filesystem), b"small candidate")
        self.assertEqual(_bytes(self.source, self.filesystem), self.original)
        self._assert_no_candidates()

    def test_mandatory_prefix_or_explicit_write_changes_bypass_guard(self):
        payload = b"x" * (len(self.original) + 100)
        cases = (
            (replace(self.plan, prefix_keys=("region",)), {}),
            (self.plan, {"skip_unchanged": False}),
            (self.plan, {"compression_level": 3}),
            (self.plan, {"compression": "snappy"}),
            (self.plan, {"row_group_size": 10}),
        )
        for index, (plan, options) in enumerate(cases):
            destination = f"bucket/output-{index}.parquet"
            with self.subTest(options=options, prefix=plan.prefix_keys):
                with patch("oparq.io.write", side_effect=self._writer(payload)):
                    result = rewrite_file(self.source, destination, plan=plan,
                                          source_filesystem=self.filesystem,
                                          destination_filesystem=self.filesystem, **options)
                self.assertEqual(result.action, "rewritten")
                self.assertEqual(_bytes(destination, self.filesystem), payload)
                self._assert_no_candidates()

    def test_candidate_failure_keeps_original_and_cleans_up(self):
        def fail(table, destination, *, filesystem, **options):
            with open_output(destination, filesystem) as stream:
                stream.write(b"unfinished candidate")
            raise RuntimeError("candidate failed")
        with patch("oparq.io.write", side_effect=fail), self.assertRaisesRegex(RuntimeError, "candidate failed"):
            rewrite_file(self.source, self.source, plan=self.plan, overwrite=True,
                          source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
        self.assertEqual(_bytes(self.source, self.filesystem), self.original)
        self._assert_no_candidates()

    def test_destination_collision_fails_before_candidate_write(self):
        destination = "bucket/output.parquet"
        _source(self.table, destination, self.filesystem)
        with patch("oparq.io.write", side_effect=AssertionError("candidate started")), self.assertRaises(FileExistsError):
            rewrite_file(self.source, destination, plan=self.plan,
                          source_filesystem=self.filesystem, destination_filesystem=self.filesystem)

    def test_duckdb_candidate_falls_back_or_publishes_without_temporary_leaks(self):
        try:
            require_duckdb()
        except ImportError:
            self.skipTest("optional DuckDB wheel not installed")
        for name, payload in (("larger", b"x" * (len(self.original) + 100)),
                              ("smaller", b"small candidate")):
            destination = f"bucket/{name}.parquet"
            def stream_candidate(schema, batches, output, filesystem, plan,
                                 settings, row_group_size, overwrite, options):
                actual = pa.Table.from_batches(list(batches), schema=schema)
                self.assertEqual(actual.to_pydict(), self.table.sort_by([("region", "ascending")]).to_pydict())
                with open_output(output, filesystem) as stream:
                    stream.write(payload)
                return oparq.WriteResult(path=output, file_size=len(payload), plan=plan,
                                         planning_seconds=0.0, sort_seconds=2.0, write_seconds=3.0)
            with self.subTest(candidate=name), patch("oparq.dataset._write_stream", side_effect=stream_candidate):
                result = rewrite_file(self.source, destination, plan=self.plan, engine="duckdb",
                                      source_filesystem=self.filesystem, destination_filesystem=self.filesystem)
            self.assertEqual(result.action, "copied" if name == "larger" else "rewritten")
            self.assertEqual(_bytes(destination, self.filesystem), self.original if name == "larger" else payload)
            self.assertGreaterEqual(result.sort_seconds, 2.0)
            self.assertGreaterEqual(result.write_seconds, 3.0)
            self.assertIsNone(result.permutation_seconds)
            self.assertIsNone(result.gathering_seconds)
            self._assert_no_candidates()


class IndexPreservationTests(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        self.table = _fixture(400)
        metadata = {**self.table.schema.metadata,
                    COMPRESSION_METADATA_KEY: compression_metadata("zstd", 1)}
        blooms = ({"bloom_filter_options": {"payload": {"ndv": 4096, "fpp": 0.01}}}
                  if REBUILDS_BLOOM_FILTERS else {})
        with open_output("bucket/source.parquet", self.filesystem) as stream:
            pq.write_table(self.table.replace_schema_metadata(metadata), stream,
                           compression="zstd", compression_level=1, write_page_index=True, **blooms)
        self.plan = _fixed(self.table, keys=("region", "category"))

    def _columns(self, path):
        footer = pq.read_metadata(str(path), filesystem=self.filesystem)
        return {footer.row_group(0).column(index).path_in_schema: footer.row_group(0).column(index)
                for index in range(footer.num_columns)}

    def test_rewrite_rebuilds_page_index_and_bloom_filters_for_the_new_order(self):
        engines = ["arrow"]
        try:
            require_duckdb()
            engines.append("duckdb")
        except ImportError:
            pass
        source = self._columns("bucket/source.parquet")
        expected = self.table.sort_by([("region", "ascending"), ("category", "ascending")])
        for engine in engines:
            with self.subTest(engine=engine):
                result = rewrite_file("bucket/source.parquet", f"bucket/{engine}.parquet", plan=self.plan,
                                      engine=engine, skip_unchanged=False,
                                      source_filesystem=self.filesystem,
                                      destination_filesystem=self.filesystem)
                columns = self._columns(result.path)
                self.assertTrue(all(c.has_column_index and c.has_offset_index for c in columns.values()))
                if REBUILDS_BLOOM_FILTERS:
                    self.assertEqual({name for name, c in columns.items()
                                      if getattr(c, "bloom_filter_offset", None) is not None},
                                     {"payload"})
                    self.assertLessEqual(columns["payload"].bloom_filter_length,
                                         source["payload"].bloom_filter_length)
                actual = oparq.read_parquet(result.path, filesystem=self.filesystem)
                self.assertEqual(actual.to_pydict(), expected.to_pydict())

    def test_explicit_writer_choices_replace_rebuilt_indexes(self):
        choices = {"write_page_index": False}
        if REBUILDS_BLOOM_FILTERS:
            choices["bloom_filter_options"] = None
        result = rewrite_file("bucket/source.parquet", "bucket/plain.parquet", plan=self.plan,
                              source_filesystem=self.filesystem, destination_filesystem=self.filesystem,
                              **choices)
        columns = self._columns(result.path)
        self.assertFalse(any(c.has_column_index or getattr(c, "bloom_filter_offset", None) is not None
                             for c in columns.values()))

    @unittest.skipUnless(REBUILDS_BLOOM_FILTERS, "this PyArrow cannot rebuild bloom filters")
    def test_resized_row_groups_keep_bloom_bits_per_row(self):
        from oparq.storage import source_index_options
        with open_output("bucket/groups.parquet", self.filesystem) as stream:
            pq.write_table(self.table, stream, row_group_size=200,
                           bloom_filter_options={"payload": {"ndv": 4096, "fpp": 0.01}})
        footer = pq.read_metadata("bucket/groups.parquet", filesystem=self.filesystem)
        same = source_index_options([footer], output_group_rows=200)["bloom_filter_options"]
        merged = source_index_options([footer], output_group_rows=400)["bloom_filter_options"]
        self.assertNotIn("write_page_index", source_index_options([footer], output_group_rows=200))
        self.assertEqual(set(same), {"payload"})
        self.assertAlmostEqual(merged["payload"]["ndv"] / same["payload"]["ndv"], 2, delta=0.01)


def _parquet_stores_string_view() -> bool:
    try:
        pq.write_table(pa.table({"text": pa.array([], pa.string_view())}), pa.BufferOutputStream())
    except pa.ArrowNotImplementedError:
        return False
    return True


@unittest.skipUnless(_parquet_stores_string_view(), "this PyArrow cannot write string_view to Parquet")
class UnreorderableFileTests(unittest.TestCase):
    def test_arrow_engine_copies_such_files_and_rejects_a_required_prefix_first(self):
        filesystem = fs._MockFileSystem()
        table = pa.table({"key": [2, 1, 2, 1], "text": pa.array(["b", "a", "b", "a"], pa.string_view())})
        _source(table, "bucket/day=1/a.parquet", filesystem)
        plan = _fixed(table, keys=("key",))
        result = oparq.rewrite_dataset("s3://bucket", "s3://new", plan=plan,
                                       source_filesystem=filesystem, destination_filesystem=filesystem)
        self.assertEqual((result.copied_files, result.rewritten_files), (1, 0))
        self.assertIn("Arrow cannot reorder ['text']", result.files[0].plan.note)
        self.assertEqual(_bytes("bucket/day=1/a.parquet", filesystem),
                         _bytes("new/day=1/a.parquet", filesystem))
        with self.assertRaisesRegex(ValueError, "required prefix"):
            oparq.rewrite_dataset("s3://bucket", "s3://other", plan=replace(plan, prefix_keys=("key",)),
                                  source_filesystem=filesystem, destination_filesystem=filesystem)
        self.assertEqual(filesystem.get_file_info("other").type, fs.FileType.NotFound)


class RemoteWriterTests(unittest.TestCase):
    def test_remote_write_keeps_values_metadata_ordering_and_row_group_geometry(self):
        table = _fixture(29)
        filesystem = fs._MockFileSystem()
        result = oparq.write(table, "s3://bucket/nested/output.parquet", filesystem=filesystem,
                             algorithm="none", prefix=("region",), row_group_size=7)
        actual = oparq.read_parquet(result.path, filesystem=filesystem)
        self.assertEqual(actual.to_pydict(), table.sort_by([("region", "ascending")]).to_pydict())
        self.assertEqual(actual.schema.metadata[b"application"], b"preserve this")
        recorded = json.loads(actual.schema.metadata[COMPRESSION_METADATA_KEY])
        self.assertEqual(recorded["compression_level"], pa.Codec.default_compression_level("zstd"))
        footer = pq.read_metadata("bucket/nested/output.parquet", filesystem=filesystem)
        self.assertEqual([footer.row_group(index).num_rows for index in range(footer.num_row_groups)], [7, 7, 7, 7, 1])
        self.assertEqual(footer.row_group(0).sorting_columns[0].column_index, 0)
        self.assertEqual(inventory_parquet("bucket", filesystem).files, ("bucket/nested/output.parquet",))

    def test_remote_failure_preserves_existing_object_and_removes_temporary_objects(self):
        table = _fixture(29)
        filesystem = fs._MockFileSystem()
        destination = "s3://bucket/output.parquet"
        oparq.write(table, destination, filesystem=filesystem, algorithm="none")
        original = _bytes(destination, filesystem)
        with patch("oparq.io.take_table", side_effect=RuntimeError("gather failed")):
            with self.assertRaisesRegex(RuntimeError, "gather failed"):
                oparq.write(table, destination, filesystem=filesystem,
                            algorithm="none", prefix=("region",), row_group_size=7, overwrite=True)
        self.assertEqual(_bytes(destination, filesystem), original)
        infos = filesystem.get_file_info(fs.FileSelector("bucket", recursive=True))
        self.assertEqual([info.path for info in infos if info.type == fs.FileType.File], ["bucket/output.parquet"])


class DuckDBRewriteTests(unittest.TestCase):
    def test_optional_duckdb_matches_arrow_values_ties_nested_data_and_groups(self):
        try:
            require_duckdb()
        except ImportError:
            self.skipTest("optional DuckDB wheel not installed")
        table = pa.table({
            "key": [2, None, 1, 2, 1, None, 1, 2, 1],
            "source_row": list(range(9)),
            "nested": [[1], None, [], [3, 4], [5], [], None, [8], [9]],
            "record": [{"label": "x"}, None, {"label": "y"}, {"label": "z"},
                       {"label": "a"}, None, {"label": "b"}, {"label": "c"}, {"label": "d"}],
        }).replace_schema_metadata({b"application": b"keep nested values"})
        filesystem = fs._MockFileSystem()
        source = "s3://bucket/source.parquet"
        _source(table, source, filesystem)
        for placement in ("at_start", "at_end"):
            with self.subTest(nulls=placement):
                plan = replace(_fixed(table, keys=("key",)), null_placement=placement)
                results = []
                for engine in ("arrow", "duckdb"):
                    results.append(rewrite_file(
                        source, f"s3://bucket/{engine}-{placement}.parquet", plan=plan,
                        source_filesystem=filesystem, destination_filesystem=filesystem,
                        engine=engine, row_group_size=4,
                    ))
                outputs = [oparq.read_parquet(result.path, filesystem=filesystem) for result in results]
                self.assertEqual(outputs[0].to_pydict(), outputs[1].to_pydict())
                self.assertTrue(outputs[0].schema.equals(outputs[1].schema, check_metadata=False))
                self.assertEqual(outputs[1].schema.metadata[b"application"], b"keep nested values")
                footer = pq.read_metadata(results[1].path.replace("s3://", ""), filesystem=filesystem)
                self.assertEqual([footer.row_group(index).num_rows for index in range(footer.num_row_groups)], [4, 4, 1])


if __name__ == "__main__":
    unittest.main()
