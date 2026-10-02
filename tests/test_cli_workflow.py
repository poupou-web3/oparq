from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from oparq import cli, RewritePlan


class CLIWorkflowTests(unittest.TestCase):
    def test_fit_prefix_then_partition_preserving_rewrite(self):
        table = pa.table({"day": [2, 1, 2, 1], "payload": ["b", "a", "b", "a"]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input" / "year=2026"
            source.mkdir(parents=True)
            pq.write_table(table, source / "part.parquet", compression="zstd", compression_level=1)
            learned = root / "plan.json"
            output = io.StringIO()
            with redirect_stdout(output):
                cli.main(["fit", str(root / "input"), str(learned), "--prefix", "day",
                          "--algorithms", "weighted,cardinality", "--sample-rows", "4",
                          "--compression-level", "1"])
                cli.main(["rewrite", str(root / "input"), str(root / "output"),
                          "--plan", str(learned), "--compression-level", "1"])
            plan = RewritePlan.load(learned)
            self.assertEqual(plan.prefix_keys, ("day",))
            result = pq.ParquetFile(root / "output" / "year=2026" / "part.parquet").read()
            self.assertEqual(result["day"].to_pylist(), [1, 1, 2, 2])
            self.assertIn("rewritten=1", output.getvalue())

    def test_plan_and_benchmark_compare_orders_when_the_source_level_is_unknown(self):
        import json
        table = pa.table({"group": [index % 3 for index in range(300)],
                          "payload": ["x" * 30 + str(index % 3) for index in range(300)]})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.parquet"
            # A third-party ZSTD file: the footer names the codec, not the level.
            pq.write_table(table, source, compression="zstd", compression_level=7)
            text, payload = io.StringIO(), io.StringIO()
            with redirect_stdout(text):
                cli.main(["plan", str(source), "--algorithm", "all", "--full"])
                cli.main(["benchmark", str(source), "--algorithms", "none,cardinality"])
            with redirect_stdout(payload):
                cli.main(["plan", str(source), "--json"])
        default = pa.Codec.default_compression_level("zstd")
        self.assertIn(f"trial codec: zstd level {default} (source compression level unknown", text.getvalue())
        self.assertIn(f"codec: zstd level {default} (source compression level unknown", text.getvalue())
        self.assertIn("sample: 300 / 300 rows", text.getvalue())
        trial = json.loads(payload.getvalue())["trial_compression"]
        self.assertEqual((trial["compression"], trial["compression_level"]), ("zstd", None))

    def test_user_errors_exit_with_a_message_instead_of_a_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.parquet")
            stderr = io.StringIO()
            with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
                cli.main(["rewrite", missing, str(Path(directory) / "output.parquet")])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("oparq: error: source not found", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_fit_rejects_ignored_algorithm_switch(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.build_parser().parse_args(["fit", "input", "plan.json", "--algorithm", "weighted"])


if __name__ == "__main__":
    unittest.main()
