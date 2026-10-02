from __future__ import annotations

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
import tempfile
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


class DocumentationTests(unittest.TestCase):
    def test_public_markdown_has_a_maintained_purpose(self):
        actual = {path.relative_to(ROOT).as_posix() for pattern in
                  ("*.md", "docs/**/*.md", "benchmarks/**/*.md") for path in ROOT.glob(pattern)}
        self.assertEqual(actual, set(MAINTAINED))

    def test_relative_links_resolve_inside_the_repository(self):
        for name in MAINTAINED:
            source = ROOT / name
            text = source.read_text()
            self.assertNotIn("/Users/admin/", text)
            for link in LINK.findall(text):
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = (source.parent / unquote(parsed.path)).resolve()
                with self.subTest(page=name, link=link):
                    self.assertTrue(target.is_relative_to(ROOT) and target.exists())

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
