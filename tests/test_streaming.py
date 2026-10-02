from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import oparq


def _input() -> pa.Table:
    return pa.table({
        "key": pa.chunked_array([[2, 1, None], [0, 2, 1], [None, 1]]),
        "payload": ["a", "z", "x", "b", "a", "z", "x", "a"],
        "nested": [[1], None, [], [2, 3], [4], [1], [2], []],
        "dictionary": pa.chunked_array([
            pa.array(["b", "a", "c"]).dictionary_encode(),
            pa.array(["c", "a", "b", "c", "a"]).dictionary_encode(),
        ]),
    }).replace_schema_metadata({b"application": b"keep me"})


def _sharded_input() -> pa.Table:
    return pa.table({
        "key": pa.chunked_array([[11, 10, 9, 8, 7], [6, 5], [4, 3, 2, 1, 0]]),
        "payload": pa.chunked_array([
            pa.array(["a", None]), pa.array(["b", "c", "d", "e"]),
            pa.array(["f"]), pa.array(["g", "h", None, "j", "k"]),
        ]),
        "nested": pa.chunked_array([
            pa.array([[1], [], None, [4, 5]]),
            pa.array([[6], [7], [], None, [10], [11], [12], []]),
        ]),
        "dictionary": pa.chunked_array([
            pa.array(values).dictionary_encode()
            for values in (
                ["a", "b"], ["c", None], ["d", "a"],
                ["e", "f"], [None, "g"], ["h", "a"],
            )
        ]),
    }).replace_schema_metadata({b"application": b"keep me"})


class StreamingWriteTests(unittest.TestCase):
    def test_stream_and_materialized_write_are_identical(self):
        for algorithm, prefix in (("none", ["key"]), ("frequency", [])):
            for placement in ("at_start", "at_end"):
                with self.subTest(algorithm=algorithm, nulls=placement):
                    with tempfile.TemporaryDirectory() as directory:
                        results = []
                        paths = [Path(directory) / f"{mode}.parquet" for mode in (True, False)]
                        for mode, path in zip((True, False), paths):
                            results.append(oparq.write(
                                _input(), path, algorithm=algorithm, prefix=prefix,
                                row_group_size=3, null_placement=placement,
                                stream_sort=mode,
                            ))
                        self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
                        written = pq.read_table(paths[0])
                        self.assertEqual(written.num_rows, 8)
                        self.assertEqual(written.schema.metadata[b"application"], b"keep me")
                        self.assertEqual(pq.read_metadata(paths[0]).num_row_groups, 3)
                        self.assertEqual(results[0].plan.sort_keys, results[1].plan.sort_keys)

    def test_only_gathers_one_row_group_at_a_time(self):
        from oparq.sorting import take_table
        sizes = []

        def gather(table, permutation, **options):
            sizes.append(len(permutation))
            return take_table(table, permutation, **options)

        with tempfile.TemporaryDirectory() as directory:
            with patch("oparq.io.take_table", side_effect=gather):
                result = oparq.write(
                    _input(), Path(directory) / "out.parquet", algorithm="none",
                    prefix=["key"], row_group_size=3,
                )
        self.assertEqual(sizes, [3, 3, 2])
        self.assertGreater(result.sort_seconds, 0)
        self.assertGreater(result.write_seconds, 0)

    def test_chunked_stream_is_byte_identical_with_dictionary_and_nested_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"output-{mode}.parquet"
                     for mode in ("chunked", "native", "materialized")]
            options = dict(algorithm="none", prefix=["key"], row_group_size=3)
            with patch("oparq.io.use_chunked_gather", return_value=True):
                chunked = oparq.write(_sharded_input(), paths[0], **options)
            with patch("oparq.io.use_chunked_gather", return_value=False):
                oparq.write(_sharded_input(), paths[1], **options)
            oparq.write(_sharded_input(), paths[2], stream_sort=False, **options)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertEqual(paths[0].read_bytes(), paths[2].read_bytes())
            self.assertEqual(pq.read_metadata(paths[0]).num_row_groups, 4)
            self.assertGreater(chunked.gathering_seconds, 0)
            self.assertAlmostEqual(
                chunked.sort_seconds,
                chunked.permutation_seconds + chunked.gathering_seconds,
            )

    def test_failed_group_keeps_existing_destination_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out.parquet"
            oparq.write(_input(), destination, algorithm="none")
            before = destination.read_bytes()
            with patch("oparq.io.take_table", side_effect=RuntimeError("gather failed")):
                with self.assertRaisesRegex(RuntimeError, "gather failed"):
                    oparq.write(
                        _input(), destination, algorithm="none", prefix=["key"],
                        row_group_size=3, overwrite=True,
                    )
            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_rejects_invalid_row_group_size_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "row_group_size"):
                oparq.write(_input(), Path(directory) / "out.parquet", row_group_size=0)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
