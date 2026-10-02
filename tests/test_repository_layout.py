"""The public source tree must stay independent of private local inputs."""

import json
from pathlib import Path
import subprocess
import unittest

from benchmarks.full_corpus import DEFAULT_ROOT
from benchmarks.run_suite import DEFAULT_DATA_ROOT


ROOT = Path(__file__).resolve().parent.parent


class RepositoryLayoutTests(unittest.TestCase):
    def test_benchmark_defaults_use_ignored_local_inputs(self):
        self.assertEqual(DEFAULT_ROOT, ROOT / "local/data/source")
        self.assertEqual(DEFAULT_DATA_ROOT, ROOT / "local/data/source/clickhouse")

    def test_public_results_have_no_original_machine_paths(self):
        for path in (ROOT / "benchmarks/results").glob("*.json"):
            text = path.read_text()
            self.assertNotIn("/Users/", text)
            self.assertNotIn("/var/folders/", text)
            report = json.loads(text)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["configuration"]["data_root"], "inputs")

    def test_local_directory_is_ignored_when_git_is_available(self):
        if not (ROOT / ".git").exists():
            self.skipTest("source archive has no Git checkout")
        result = subprocess.run(
            ["git", "check-ignore", "local/data/source", "local/research", "local/publication"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 3)
