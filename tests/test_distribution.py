from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from scripts.check_distribution import DistributionError, check_distribution, check_sdist, check_wheel


PROJECT = '''[project]
name = "oparq"
version = "0.3.0"
description = "Test fixture"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
license-files = ["LICENSE"]
dependencies = ["pyarrow>=25.0.1"]
[project.optional-dependencies]
duckdb = ["duckdb>=1.5.6,<2"]
[project.scripts]
oparq = "oparq.cli:main"
[tool.uv.build-backend]
source-include = ["docs/**", "scripts/check_distribution.py"]
'''


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        files = {
            "pyproject.toml": PROJECT,
            "README.md": "# oparq\n\nA fixture: café → compressed.\n",
            "LICENSE": "MIT fixture license\n",
            "src/oparq/__init__.py": '__version__ = "0.3.0"\n',
            "src/oparq/__main__.py": "from .cli import main\n",
            "src/oparq/cli.py": "def main(): pass\n",
            "src/oparq/py.typed": "",
            "docs/nested/guide.md": "# Guide\n",
            "scripts/check_distribution.py": "# Distribution fixture\n",
        }
        for name, contents in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        self.metadata = (
            "Metadata-Version: 2.4\nName: oparq\nVersion: 0.3.0\n"
            "Summary: Test fixture\n"
            "Requires-Python: >=3.12\nRequires-Dist: pyarrow>=25.0.1\n"
            "Requires-Dist: duckdb<2,>=1.5.6; extra == \"duckdb\"\n"
            "Provides-Extra: duckdb\nLicense-Expression: MIT\n"
            "Description-Content-Type: text/markdown\n\n"
        ).encode() + (self.root / "README.md").read_bytes()
        self.info = "oparq-0.3.0.dist-info"
        self.wheel_files = {
            **{"oparq/" + path.name: path.read_bytes() for path in (self.root / "src/oparq").iterdir()},
            self.info + "/METADATA": self.metadata,
            self.info + "/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            self.info + "/entry_points.txt": b"[console_scripts]\noparq = oparq.cli:main\n",
            self.info + "/licenses/LICENSE": (self.root / "LICENSE").read_bytes(),
        }
        self.sdist_files = {
            **{path.relative_to(self.root).as_posix(): path.read_bytes()
               for path in self.root.rglob("*") if path.is_file()},
            "PKG-INFO": self.metadata,
        }

    def wheel(self, files=None, *, corrupt_record=False):
        files = dict(self.wheel_files if files is None else files)
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        for filename, contents in files.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
            writer.writerow((filename, "sha256=" + ("wrong" if corrupt_record else digest), len(contents)))
        writer.writerow((self.info + "/RECORD", "", ""))
        files[self.info + "/RECORD"] = output.getvalue().encode()
        path = self.root / "oparq-0.3.0-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            for filename, contents in files.items():
                archive.writestr(filename, contents)
        return path

    def sdist(self, files=None):
        files = self.sdist_files if files is None else files
        path = self.root / "oparq-0.3.0.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for filename, contents in files.items():
                info = tarfile.TarInfo("oparq-0.3.0/" + filename)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        return path

    def test_clean_wheel_and_curated_sdist_pass(self):
        results = check_distribution([self.wheel(), self.sdist()], self.root)
        self.assertEqual([item["kind"] for item in results], ["wheel", "sdist"])
        self.assertTrue(all(item["bytes"] > 0 for item in results))

    def test_research_clone_or_data_cannot_leak_into_artifacts(self):
        with self.assertRaisesRegex(DistributionError, "unexpected wheel member"):
            check_wheel(self.wheel({**self.wheel_files, "arrow/private.cpp": b"large clone"}), self.root)
        with self.assertRaisesRegex(DistributionError, "unexpected sdist member"):
            check_sdist(self.sdist({**self.sdist_files, "data/source.parquet": b"dataset"}), self.root)

    def test_stale_package_sources_are_rejected(self):
        files = {**self.wheel_files, "oparq/cli.py": b"# stale build\n"}
        with self.assertRaisesRegex(DistributionError, "stale or changed package source"):
            check_wheel(self.wheel(files), self.root)

    def test_unsafe_archive_path_is_rejected_without_extraction(self):
        with self.assertRaisesRegex(DistributionError, "unsafe archive path"):
            check_sdist(self.sdist({**self.sdist_files, "../outside": b"unsafe"}), self.root)

    def test_dependencies_and_optional_extra_must_match(self):
        changed = self.metadata.replace(b"pyarrow>=25.0.1", b"pyarrow>=1")
        with self.assertRaisesRegex(DistributionError, "dependency metadata differs"):
            check_wheel(self.wheel({**self.wheel_files, self.info + "/METADATA": changed}), self.root)

    def test_typing_marker_is_required(self):
        files = {name: raw for name, raw in self.wheel_files.items() if name != "oparq/py.typed"}
        with self.assertRaisesRegex(DistributionError, "package modules missing"):
            check_wheel(self.wheel(files), self.root)

    def test_record_integrity_is_verified(self):
        with self.assertRaisesRegex(DistributionError, "invalid RECORD hash/size"):
            check_wheel(self.wheel(corrupt_record=True), self.root)

    def test_declared_nested_docs_and_scripts_are_required(self):
        files = {name: raw for name, raw in self.sdist_files.items() if name != "docs/nested/guide.md"}
        with self.assertRaisesRegex(DistributionError, "required source files missing"):
            check_sdist(self.sdist(files), self.root)

    def test_license_is_present_and_exact(self):
        files = {**self.wheel_files, self.info + "/licenses/LICENSE": b"wrong license"}
        with self.assertRaisesRegex(DistributionError, "license file missing or differs"):
            check_wheel(self.wheel(files), self.root)

    def test_public_version_must_match_metadata(self):
        (self.root / "src/oparq/__init__.py").write_text('__version__ = "0.2.0"\n')
        with self.assertRaisesRegex(DistributionError, "public __version__ differs"):
            check_wheel(self.wheel(), self.root)

    def test_audit_requires_both_distribution_types(self):
        with self.assertRaisesRegex(DistributionError, "both a wheel and a source distribution"):
            check_distribution([self.wheel()], self.root)


if __name__ == "__main__":
    unittest.main()
