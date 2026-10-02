from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
import tempfile
import tomllib
import unittest
from urllib.parse import unquote, urlsplit

import oparq


ROOT = Path(__file__).resolve().parents[1]
# Plain Markdown read on GitHub; there is no generated documentation site.
PAGES = ("quickstart.md", "cloud-storage.md", "compression.md", "algorithms.md",
         "api.md", "cli.md", "benchmarks.md")
MAINTAINED = ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "benchmarks/README.md",
              *("docs/" + name for name in PAGES))
LINK = re.compile(r"\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
# README links are absolute so they also work on PyPI.
REPOSITORY_FILE = re.compile(r"https://github\.com/poupou-web3/oparq/(?:blob|tree)/main/([^#?]+)")


class DocumentationTests(unittest.TestCase):
    def test_public_markdown_has_a_maintained_purpose(self):
        actual = {path.relative_to(ROOT).as_posix() for pattern in
                  ("*.md", "docs/**/*.md", "benchmarks/**/*.md") for path in ROOT.glob(pattern)}
        self.assertEqual(actual, set(MAINTAINED))

    def test_links_resolve_inside_the_repository(self):
        for name in MAINTAINED:
            source = ROOT / name
            text = source.read_text()
            self.assertNotIn("/Users/admin/", text)
            for link in LINK.findall(text):
                parsed = urlsplit(link)
                repository_file = REPOSITORY_FILE.fullmatch(link)
                if repository_file:
                    target = ROOT / unquote(repository_file.group(1))
                elif parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                else:
                    target = (source.parent / unquote(parsed.path)).resolve()
                with self.subTest(page=name, link=link):
                    self.assertTrue(target.is_relative_to(ROOT) and target.exists())

    def test_project_urls_name_existing_repository_files(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            urls = tomllib.load(stream)["project"]["urls"]
        self.assertIn("Repository", urls)
        for label, url in urls.items():
            repository_file = REPOSITORY_FILE.fullmatch(url)
            if repository_file:
                with self.subTest(label=label):
                    self.assertTrue((ROOT / repository_file.group(1)).exists())

    def test_quickstart_python_examples_execute_together(self):
        text = (ROOT / "docs/quickstart.md").read_text()
        snippets = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
        self.assertGreaterEqual(len(snippets), 3)
        namespace = {"__name__": "__documentation_example__"}
        original = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="oparq-docs-") as temporary:
            try:
                os.chdir(temporary)
                with redirect_stdout(io.StringIO()):
                    for snippet in snippets:
                        exec(compile(snippet, "docs/quickstart.md", "exec"), namespace)
                self.assertTrue(Path("optimized.parquet").is_file())
                self.assertTrue(Path("future.parquet").is_file())
                self.assertEqual(namespace["result"].planning_seconds, 0)
            finally:
                os.chdir(original)

    def test_algorithm_reference_matches_public_canonical_names(self):
        text = (ROOT / "docs/algorithms.md").read_text()
        names = re.findall(r"^\|\s*`([a-z_]+)`\s*\|", text, flags=re.MULTILINE)
        self.assertEqual(set(names), set(oparq.ALGORITHMS))
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
