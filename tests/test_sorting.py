import unittest
from unittest.mock import patch

import pyarrow as pa

from oparq.sorting import ChunkedTableGather, sort_indices, use_chunked_gather


class RankedSortingTests(unittest.TestCase):
    def test_rank_and_arrow_backends_have_identical_stable_permutations(self):
        table = pa.table({
            "string": pa.chunked_array([["z", None, "b", "b"], ["a", "z", "b"]]),
            "number": [1, 3, 2, 2, 0, 1, 1],
        })
        for placement in ("at_start", "at_end"):
            expected = sort_indices(table, ["string", "number"],
                                    sort_backend="arrow", null_placement=placement)
            actual = sort_indices(table, ["string", "number"],
                                  sort_backend="rank", null_placement=placement)
            self.assertTrue(actual.equals(expected))

    def test_nan_and_null_match_natural_order(self):
        table = pa.table({"x": [float("nan"), 1., None, -1., float("nan"), 1.]})
        for placement in ("at_start", "at_end"):
            expected = sort_indices(table, ["x"], sort_backend="arrow",
                                    null_placement=placement)
            actual = sort_indices(table, ["x"], sort_backend="rank",
                                  null_placement=placement)
            self.assertTrue(actual.equals(expected))

    def test_invalid_utf8_is_profiled_without_python_decoding(self):
        from oparq import profile_table
        # STRING values from third-party Parquet writers can contain opaque
        # bytes; Arrow's native kernels can preserve and compare them.
        values = pa.array([b"\xff", b"\xff", b"a"]).view(pa.string())
        table = pa.table({"x": values})
        self.assertTrue(profile_table(table)[0].eligible)
        self.assertTrue(sort_indices(table, ["x"], sort_backend="rank").equals(
            sort_indices(table, ["x"], sort_backend="arrow")))

    def test_parallel_gather_preserves_rows_types_and_schema_metadata(self):
        from oparq.sorting import take_table
        table = pa.table({
            "x": pa.chunked_array([[2, 1], [2, 0]]),
            "list": [[1, 2], None, [], [4]],
            "dictionary": pa.array(["z", "a", "z", "b"]).dictionary_encode(),
        }).replace_schema_metadata({b"application": b"preserve"})
        permutation = sort_indices(table, ["x"])
        expected = table.take(permutation)
        with patch("oparq.sorting.PARALLEL_TAKE_MIN_BYTES", 0):
            actual = take_table(table, permutation, gather_threads=3)
        self.assertTrue(actual.equals(expected, check_metadata=True))

    def test_chunked_gather_matches_arrow_with_unequal_boundaries_and_types(self):
        table = pa.table({
            "key": pa.chunked_array([[3, None], [2, 4, 1], [0, 3, 2, 1]]),
            "payload": pa.chunked_array([
                pa.array(["a"]), pa.array([None, "b", "c"]),
                pa.array(["d", "e"]), pa.array(["f", None, "h"]),
            ]),
            "nested": pa.chunked_array([
                pa.array([[1, None], None, []]),
                pa.array([[4], [5, 6], None, [], [8], [9]]),
            ]),
            "dictionary": pa.chunked_array([
                pa.array(["red", None, "blue"]).dictionary_encode(),
                pa.array(["green", "red"]).dictionary_encode(),
                pa.array(["blue", "yellow", None, "green"]).dictionary_encode(),
            ]),
        }).replace_schema_metadata({b"app": b"keep"})
        gather = ChunkedTableGather(table, gather_threads=3)
        permutations = [
            pa.array([8, 0, 5, 1, 7, 3, 2, 4, 6], type=pa.uint64()),
            pa.array([0, 1, 2, 1], type=pa.uint64()),
            pa.array([8, 7], type=pa.uint64()),
        ]
        for permutation in permutations:
            with self.subTest(permutation=permutation.to_pylist()):
                expected = table.take(permutation)
                actual = gather.take(permutation)
                self.assertTrue(actual.equals(expected, check_metadata=True))
                self.assertEqual(actual["dictionary"].chunk(0).dictionary.to_pylist(),
                                 expected["dictionary"].chunk(0).dictionary.to_pylist())
                with patch("oparq.sorting.PARALLEL_TAKE_MIN_BYTES", 0):
                    parallel = gather.take(permutation)
                self.assertTrue(parallel.equals(expected, check_metadata=True))

    def test_tables_arrow_cannot_reorder_are_neither_searched_nor_sorted(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import pyarrow.parquet as pq

        import oparq
        from oparq.sorting import unreorderable_columns
        table = pa.table({
            "key": [3, 1, 3, 1, 2, 1],
            "text": pa.array(["c", None, "c", "a", "b", "a"], pa.string_view()),
            "nested": pa.array([["c"], None, [], ["a"], ["b"], []], pa.list_(pa.string_view())),
            "labels": pa.array(["x", "y", "x", "y", "x", "y"]).dictionary_encode(),
        })
        self.assertEqual(unreorderable_columns(table.schema), ["text", "nested"])
        with patch("oparq.planning.profile_table", side_effect=AssertionError("searched")):
            plan = oparq.plan_sort(table, algorithm="all", sample_rows=None)
        self.assertEqual(plan.sort_keys, ())
        self.assertIn("Arrow cannot reorder ['text', 'nested']", plan.note)
        with self.assertRaisesRegex(ValueError, "required prefix"):
            oparq.plan_sort(table, algorithm="none", prefix=["key"])
        fixed = oparq.RewritePlan(algorithm="weighted", sort_keys=("key",), prefix_keys=(),
                                  column_types=tuple((f.name, str(f.type)) for f in table.schema))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.parquet"
            try:
                result = oparq.write(table, path, plan=fixed)
            except pa.ArrowNotImplementedError:
                result = None  # This PyArrow cannot store string_view in Parquet.
            if result is not None:
                self.assertEqual(result.plan.sort_keys, ())
                self.assertTrue(pq.read_table(path).equals(table))
                self.assertEqual(pq.read_metadata(path).row_group(0).sorting_columns, ())
        with self.assertRaisesRegex(ValueError, "cannot apply the plan"):
            oparq.apply_plan(table, fixed.for_table(table))

    def test_chunked_gather_handles_offset_permutation_slice(self):
        table = pa.table({"x": pa.chunked_array([[10, 20], [], [30, 40], [50]])})
        permutation = pa.array([0, 4, 1, 3, 2], type=pa.uint64()).slice(1, 3)
        self.assertTrue(ChunkedTableGather(table).take(permutation).equals(
            table.take(permutation)))

    def test_chunked_gather_selection_needs_many_groups_and_chunks(self):
        table = pa.table({"x": pa.chunked_array([[1, 2, 3]] * 4)})
        with patch("oparq.sorting.CHUNKED_TAKE_MIN_REPEATED_BYTES", 0):
            self.assertTrue(use_chunked_gather(table, 3))
            self.assertFalse(use_chunked_gather(table, 4))
            self.assertFalse(use_chunked_gather(table.combine_chunks(), 3))


if __name__ == "__main__":
    unittest.main()
