from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq

from oparq.storage import (
    COMPRESSION_METADATA_KEY,
    compression_metadata,
    copy_file,
    effective_compression_level,
    inspect_compression,
    inventory_parquet,
    open_input,
    open_output,
    read_parquet_source,
    resolve_compression,
)


class StorageFixture(unittest.TestCase):
    def setUp(self):
        self.filesystem = fs._MockFileSystem()
        self.table = pa.table({"key": [2, None, 1], "value": ["x", "x", "y"]})

    def write(self, path, *, compression="snappy", level=None, provenance=False):
        table = self.table
        if provenance:
            table = table.replace_schema_metadata({
                COMPRESSION_METADATA_KEY: compression_metadata(compression, level)
            })
        with open_output(path, self.filesystem) as output:
            pq.write_table(
                table, output, compression=compression,
                compression_level=effective_compression_level(compression, level),
            )


class StorageTests(StorageFixture):
    def test_injected_remote_filesystem_reads_and_preserves_directory_paths(self):
        self.write("s3://bucket/year=2026/part-b.parquet")
        self.write("s3://bucket/year=2025/part-a.parquet")
        with open_output("s3://bucket/notes.txt", self.filesystem) as output:
            output.write(b"ignored")
        inventory = inventory_parquet("s3://bucket", self.filesystem)
        self.assertEqual(inventory.files, (
            "bucket/year=2025/part-a.parquet", "bucket/year=2026/part-b.parquet",
        ))
        self.assertEqual(inventory.relative_path(inventory.files[0]), "year=2025/part-a.parquet")
        table = read_parquet_source(inventory)
        self.assertEqual(table.num_rows, 6)
        self.assertEqual(table.column_names, ["key", "value"])
        self.assertEqual(table.slice(0, 3).to_pydict(), self.table.to_pydict())

    def test_cross_filesystem_copy_keeps_exact_object_bytes_without_local_staging(self):
        self.write("gs://old/year=2026/data.parquet")
        destination_fs = fs._MockFileSystem()
        copy_file(
            "gs://old/year=2026/data.parquet", "s3://new/year=2026/data.parquet",
            source_filesystem=self.filesystem, destination_filesystem=destination_fs,
        )
        with open_input("gs://old/year=2026/data.parquet", self.filesystem) as source:
            original = source.read()
        with open_input("s3://new/year=2026/data.parquet", destination_fs) as copied:
            self.assertEqual(copied.read(), original)
        with self.assertRaises(FileExistsError):
            copy_file(
                "old/year=2026/data.parquet", "new/year=2026/data.parquet",
                source_filesystem=self.filesystem, destination_filesystem=destination_fs,
            )

    def test_local_uri_output_creates_parent_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "with spaces" / "nested" / "object.dat"
            with open_output(path.as_uri()) as output:
                output.write(b"original")
            self.assertEqual(path.read_bytes(), b"original")
            with self.assertRaises(FileExistsError):
                open_output(path)

    def test_inventory_skips_hidden_and_underscore_paths_like_arrow(self):
        for path in ("bucket/day=1/part.parquet", "bucket/_temporary/0/part.parquet",
                     "bucket/_delta_log/00000000000000000010.checkpoint.parquet",
                     "bucket/day=1/.part.parquet.oparq-candidate-1.parquet"):
            self.write(path)
        self.assertEqual(inventory_parquet("s3://bucket", self.filesystem).files,
                         ("bucket/day=1/part.parquet",))
        self.write("_staging/part.parquet")
        self.assertEqual(inventory_parquet("_staging", self.filesystem).files,
                         ("_staging/part.parquet",))

    def test_schema_drift_is_rejected_instead_of_dropping_columns(self):
        self.write("bucket/a.parquet")
        with open_output("bucket/b.parquet", self.filesystem) as output:
            pq.write_table(self.table.append_column("extra", pa.array([1, 2, 3])), output)
        with self.assertRaisesRegex(ValueError, "different physical schemas"):
            read_parquet_source("bucket", self.filesystem)

    def test_consolidating_hive_tree_cannot_drop_partition_values(self):
        import oparq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for day in (1, 2):
                (root / f"day={day}").mkdir(parents=True)
                pq.write_table(self.table, root / f"day={day}" / "part.parquet", compression="snappy")
            output = Path(directory) / "combined.parquet"
            with self.assertRaisesRegex(ValueError, r"partition values.*\['day'\]"):
                oparq.rewrite(root, output, algorithm="none")
            self.assertFalse(output.exists())
            physical = Path(directory) / "physical"
            (physical / "day=1").mkdir(parents=True)
            pq.write_table(self.table.append_column("day", pa.array([1, 1, 1])),
                           physical / "day=1" / "part.parquet", compression="snappy")
            result = oparq.rewrite(physical, output, algorithm="none")
            self.assertEqual(pq.read_table(result.path)["day"].to_pylist(), [1, 1, 1])

    def test_missing_and_empty_inventory_fail_before_reading(self):
        with self.assertRaises(FileNotFoundError):
            inventory_parquet("s3://missing", self.filesystem)
        self.filesystem.create_dir("empty")
        with self.assertRaisesRegex(FileNotFoundError, "no Parquet"):
            inventory_parquet("empty", self.filesystem)


class CompressionPreservationTests(StorageFixture):
    def test_codec_is_known_but_unrecorded_zstd_level_is_unknown(self):
        self.write("bucket/source.parquet", compression="zstd", level=1)
        profile = inspect_compression("bucket/source.parquet", self.filesystem)
        self.assertEqual(profile.codecs, {"key": ("zstd",), "value": ("zstd",)})
        self.assertEqual(profile.unknown_level_columns, ("key", "value"))
        with self.assertRaisesRegex(ValueError, "does not record compression levels"):
            resolve_compression(profile)
        settings = resolve_compression(profile, compression_level=1)
        self.assertEqual(settings.writer_options(), {"compression": "zstd", "compression_level": 1})
        self.assertIsNone(resolve_compression(profile, compression_level=None).compression_level)

    def test_snappy_has_no_encoder_level_to_preserve(self):
        self.write("bucket/source.parquet", compression="snappy")
        settings = resolve_compression(inspect_compression("bucket/source.parquet", self.filesystem))
        self.assertEqual(settings.writer_options(), {"compression": "snappy", "compression_level": None})

    def test_metadata_records_actual_default_for_future_preservation(self):
        self.write("bucket/source.parquet", compression="zstd", provenance=True)
        profile = inspect_compression("bucket/source.parquet", self.filesystem)
        actual_default = pa.Codec.default_compression_level("zstd")
        self.assertEqual(profile.levels, {"key": actual_default, "value": actual_default})
        self.assertEqual(resolve_compression(profile).compression_level, actual_default)

    def test_preserves_per_column_codecs_and_only_applies_valid_levels(self):
        codecs = {"key": "zstd", "value": "snappy"}
        self.write("bucket/source.parquet", compression=codecs, level=1, provenance=True)
        settings = resolve_compression(inspect_compression("bucket/source.parquet", self.filesystem))
        self.assertEqual(settings.writer_options(), {
            "compression": codecs, "compression_level": {"key": 1},
        })
        with open_output("bucket/rewrite.parquet", self.filesystem) as output:
            pq.write_table(self.table, output, **settings.writer_options())
        rewritten = inspect_compression("bucket/rewrite.parquet", self.filesystem)
        self.assertEqual(rewritten.codecs, {"key": ("zstd",), "value": ("snappy",)})

    def test_scalar_codec_with_per_column_level_records_default_for_unlisted_columns(self):
        self.write("bucket/source.parquet", compression="zstd", level={"key": 3}, provenance=True)
        profile = inspect_compression("bucket/source.parquet", self.filesystem)
        self.assertEqual(profile.levels, {"key": 3, "value": pa.Codec.default_compression_level("zstd")})
        self.assertEqual(resolve_compression(profile).compression_level, profile.levels)

    def test_manifest_level_must_name_the_actual_codec(self):
        self.write("bucket/source.parquet", compression="zstd", level=1)
        profile = inspect_compression("bucket/source.parquet", self.filesystem, manifest={
            "compression": "zstd", "compression_level": 1,
        })
        self.assertEqual(resolve_compression(profile).compression_level, 1)
        mismatched = inspect_compression("bucket/source.parquet", self.filesystem, manifest={
            "compression": "gzip", "compression_level": 1,
        })
        self.assertEqual(mismatched.unknown_level_columns, ("key", "value"))

    def test_per_file_manifest_and_conflicting_source_levels(self):
        self.write("bucket/day=1/part.parquet", compression="zstd", level=1)
        self.write("bucket/day=2/part.parquet", compression="zstd", level=3)
        profile = inspect_compression("bucket", self.filesystem, manifest={
            "compression": "zstd", "compression_level": 1,
            "files": {"day=2/part.parquet": {"compression_level": 3}},
        })
        self.assertEqual(profile.conflicts, ("key", "value"))
        with self.assertRaisesRegex(ValueError, "Unknown columns"):
            resolve_compression(profile)
        self.assertEqual(resolve_compression(profile, compression_level=1).compression_level, 1)

    def test_mixed_file_codecs_need_explicit_choice_or_per_file_rewrites(self):
        self.write("bucket/a.parquet", compression="snappy")
        self.write("bucket/b.parquet", compression="zstd", level=1, provenance=True)
        profile = inspect_compression("bucket", self.filesystem)
        with self.assertRaisesRegex(ValueError, "source codecs differ"):
            resolve_compression(profile)
        settings = resolve_compression(profile, compression="zstd", compression_level=1)
        self.assertEqual(settings.compression, "zstd")

    def test_one_file_without_level_provenance_prevents_dataset_level_guess(self):
        self.write("bucket/a.parquet", compression="zstd", level=1, provenance=True)
        self.write("bucket/b.parquet", compression="zstd", level=1)
        profile = inspect_compression("bucket", self.filesystem)
        self.assertEqual(profile.levels, {})
        with self.assertRaisesRegex(ValueError, "Unknown columns"):
            resolve_compression(profile)

    def test_explicit_new_codec_requires_intentional_new_level(self):
        self.write("bucket/source.parquet", compression="snappy")
        profile = inspect_compression("bucket/source.parquet", self.filesystem)
        with self.assertRaisesRegex(ValueError, "Unknown columns"):
            resolve_compression(profile, compression="zstd")
        self.assertEqual(resolve_compression(profile, compression="zstd", compression_level=1).compression_level, 1)


if __name__ == "__main__":
    unittest.main()
