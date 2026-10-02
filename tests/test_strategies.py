from __future__ import annotations

import unittest

import pyarrow as pa

from oparq.profile import profile_table
from oparq.strategies import (
    effective_cardinality,
    entropy_order,
    frequency_metadata_keys,
    frequency_sort_indices,
    payload_benefit_order,
)


class ScalarKeyStrategyTests(unittest.TestCase):
    def test_entropy_order_uses_effective_not_raw_cardinality(self) -> None:
        table = pa.table(
            {
                "uniform_two": [index % 2 for index in range(100)],
                "skewed_four": [0] * 97 + [1, 2, 3],
            }
        )
        profiles = {item.name: item for item in profile_table(table, sample_rows=None)}

        self.assertEqual(profiles["uniform_two"].sample_distinct, 2)
        self.assertEqual(profiles["skewed_four"].sample_distinct, 4)
        self.assertLess(
            effective_cardinality(profiles["skewed_four"]),
            effective_cardinality(profiles["uniform_two"]),
        )
        self.assertEqual(
            entropy_order(list(profiles.values()), slots=2),
            ["skewed_four", "uniform_two"],
        )

    def test_payload_benefit_prefers_large_repeatable_values(self) -> None:
        rows = 100
        table = pa.table(
            {
                "tiny": [index % 2 for index in range(rows)],
                "heavy": [
                    ("heavy-payload-" * 20) + str(index % 10)
                    for index in range(rows)
                ],
                "large_unique": [
                    ("unique-payload-" * 20) + str(index) for index in range(rows)
                ],
            }
        )
        profiles = profile_table(table, sample_rows=None)
        self.assertEqual(payload_benefit_order(profiles, slots=1), ["heavy"])
        self.assertEqual(payload_benefit_order(profiles, slots=0), [])


class FrequencyOrderingTests(unittest.TestCase):
    def test_frequency_descending_with_natural_value_ties(self) -> None:
        table = pa.table({"value": ["b", "a", "c", "a", "b", "a", "c"]})
        indices = frequency_sort_indices(table, ["value"])
        ordered = table.take(indices)
        self.assertEqual(
            ordered["value"].to_pylist(),
            ["a", "a", "a", "b", "b", "c", "c"],
        )

    def test_equal_frequencies_use_natural_numeric_order(self) -> None:
        table = pa.table({"value": [3, 2, 1, 3, 1, 2]})
        indices = frequency_sort_indices(table, ["value"])
        self.assertEqual(table.take(indices)["value"].to_pylist(), [1, 1, 2, 2, 3, 3])

    def test_prefix_is_natural_and_frequency_order_is_global(self) -> None:
        table = pa.table(
            {
                "group": [2, 1, 2, 1, 2, 1, 2, 1],
                "value": ["x", "y", "y", "x", "x", "z", "x", "y"],
            }
        )
        indices = frequency_sort_indices(
            table,
            ["value"],
            prefix_keys=["group"],
        )
        ordered = table.take(indices)
        self.assertEqual(ordered["group"].to_pylist(), [1, 1, 1, 1, 2, 2, 2, 2])
        self.assertEqual(
            ordered["value"].to_pylist(),
            ["x", "y", "y", "z", "x", "x", "x", "y"],
        )

    def test_nulls_form_one_run_at_requested_side(self) -> None:
        table = pa.table({"value": ["a", None, None, "b", "a", None]})
        at_end = table.take(
            frequency_sort_indices(table, ["value"], null_placement="at_end")
        )
        at_start = table.take(
            frequency_sort_indices(table, ["value"], null_placement="at_start")
        )
        self.assertEqual(at_end["value"].to_pylist(), ["a", "a", "b", None, None, None])
        self.assertEqual(
            at_start["value"].to_pylist(),
            [None, None, None, "a", "a", "b"],
        )

    def test_dictionary_values_are_ranked_by_decoded_value(self) -> None:
        dictionary = pa.array(["z", "a", "m"])
        values = pa.DictionaryArray.from_arrays([0, 1, 0, 2, 1, 0], dictionary)
        table = pa.table({"value": values})
        indices = frequency_sort_indices(table, ["value"])
        self.assertEqual(
            table.take(indices)["value"].to_pylist(),
            ["z", "z", "z", "a", "a", "m"],
        )

    def test_opaque_invalid_utf8_string_never_materializes_in_python(self) -> None:
        offsets = pa.py_buffer(
            b"".join(value.to_bytes(4, "little") for value in (0, 1, 2, 3))
        )
        values = pa.Array.from_buffers(
            pa.string(),
            3,
            [None, offsets, pa.py_buffer(b"\xffa\xff")],
        )
        table = pa.table({"value": values})

        # The invalid byte cannot be decoded by ``to_pylist``.  Arrow kernels
        # can nevertheless histogram, rank, and reorder it as opaque data.
        with self.assertRaises(UnicodeDecodeError):
            values.to_pylist()
        self.assertEqual(
            frequency_sort_indices(table, ["value"]).to_pylist(),
            [0, 2, 1],
        )

    def test_only_prefix_is_truthful_parquet_sort_metadata(self) -> None:
        self.assertEqual(frequency_metadata_keys(["day", "hour"]), ("day", "hour"))
        self.assertEqual(frequency_metadata_keys(), ())

    def test_identity_and_invalid_key_requests(self) -> None:
        table = pa.table({"value": [3, 1, 2], "nested": [[1], [2], [3]]})
        self.assertEqual(frequency_sort_indices(table, []).to_pylist(), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            frequency_sort_indices(table, ["value"], prefix_keys=["value"])
        with self.assertRaisesRegex(ValueError, "not in table"):
            frequency_sort_indices(table, ["missing"])
        with self.assertRaisesRegex(ValueError, "unsupported scalar"):
            frequency_sort_indices(table, ["nested"])
        with self.assertRaisesRegex(ValueError, "null_placement"):
            frequency_sort_indices(table, ["value"], null_placement="middle")


if __name__ == "__main__":
    unittest.main()
