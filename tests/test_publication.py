import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pyarrow.parquet as pq

from scripts.prepare_benchmarks import RESULT_SCHEMA, prepare_bundle
from scripts.publish_benchmarks import publish_bundle, verify_bundle


def _fixture(repository):
    for name, value in {"README.md": "Fixture", "LICENSE": "MIT License", "uv.lock": "version = 1",
                        "pyproject.toml": '[project]\nname="oparq"\nversion="0.3.0"',
                        "src/oparq/__init__.py": "__version__ = '0.3.0'", "src/oparq/py.typed": "",
                        "docs/quickstart.md": "# Fixture docs", "docs/draft.txt": "Not a guide"}.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    source = repository / "data/source/clickhouse/tiny/part.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"opaque source bytes never included")
    stat = source.stat()
    inventory = {"source": str(source.parent), "file_size_bytes": stat.st_size, "rows": 10,
                 "created_by": ["ClickHouse version fixture"], "source_codecs": ["ZSTD"],
                 "files": [{"path": str(source), "size_bytes": stat.st_size, "rows": 10, "row_groups": 1}],
                 "snapshot": [{"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]}
    configuration = {"data_root": str(repository / "data/source"), "temp_root": str(repository / "private-temp")}
    writer = {"compression": "zstd", "compression_level": 1, "row_group_size": 100}
    result = {"requested_algorithm": "none", "plan": {"sort_keys": []}, "rows": 10, "bytes": 100,
              "planning_seconds": 0, "writing_seconds": 1, "status": "pass"}
    full = {"configuration": configuration, "versions": {"oparq": "0.3.0"}, "status": "pass",
            "full_corpus_complete": True, "datasets": [{"name": "tiny", "inventory": inventory,
                                                         "writer": writer, "results": [result], "status": "pass"}]}
    engine_result = {**result, "case": "none_arrow", "sort_seconds": 0, "input_read_seconds": 0.1,
                     "permutation_seconds": 0, "gathering_seconds": 0}
    engines = {"configuration": configuration, "versions": {"oparq": "0.3.0"}, "status": "pass",
               "requested_datasets_complete": True,
               "datasets": [{"name": "tiny", "inventory": inventory, "fixed_plan": {"sort_keys": []},
                             "files": [{"writer": writer}], "results": [engine_result], "status": "pass"}]}
    for name, report in zip(("full-corpus-2026-09-30", "saved-plan-engines-2026-09-30"), (full, engines)):
        path = repository / "benchmarks/results" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
        summary = repository / "benchmarks" / (name + ".md")
        summary.write_text(f"[results](results/{name}.json) [runner](full_corpus.py)")
    # Unrelated and sensitive files must not enter the allowlisted snapshot.
    (repository / ".aws").mkdir()
    (repository / ".aws/credentials").write_text("secret fixture")
    return source


class PublicationTests(unittest.TestCase):
    def test_curated_bundle_full_hashes_relative_paths_and_viewer_tables(self):
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve() / "repo"
            repository.mkdir()
            source = _fixture(repository)
            bundle = repository / "bundle"
            result = prepare_bundle(repository, bundle, hash_source=True)
            self.assertTrue(result["full_source_sha256_complete"])
            files = verify_bundle(bundle)
            self.assertFalse(any("credentials" in path or path.startswith("inputs/") for path in files))
            self.assertTrue((bundle / "reproduction/src/oparq/py.typed").exists())
            self.assertTrue((bundle / "reproduction/docs/quickstart.md").exists())
            self.assertFalse((bundle / "reproduction/docs/draft.txt").exists())
            summary_name = "full-corpus-2026-09-30.md"
            self.assertIn("(../results/", (bundle / "summaries" / summary_name).read_text())
            self.assertIn("(../reproduction/benchmarks/full_corpus.py)", (bundle / "summaries" / summary_name).read_text())
            self.assertFalse((bundle / "reproduction/benchmarks" / summary_name).exists())
            self.assertIn("Generated from", (bundle / "summaries" / summary_name).read_text())
            self.assertNotIn("[results]", (bundle / "summaries" / summary_name).read_text())
            report = (bundle / "results/full-corpus-2026-09-30.json").read_text()
            self.assertNotIn(str(repository), report)
            self.assertIn("inputs/clickhouse/tiny/part.parquet", report)
            manifest = json.loads((bundle / "input-manifest.json").read_text())
            self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertTrue(manifest["files"][0]["matches_benchmark_snapshot"])
            for stem in ("full_corpus", "saved_plan_engines"):
                self.assertTrue(pq.read_schema(bundle / "tables" / (stem + ".parquet")).equals(RESULT_SCHEMA))
                self.assertEqual(pq.read_metadata(bundle / "tables" / (stem + ".parquet")).num_rows, 1)
            self.assertIn("not independently", (bundle / "README.md").read_text())
            self.assertIn("license: mit", (bundle / "README.md").read_text())
            with self.assertRaises(FileExistsError):
                prepare_bundle(repository, bundle)

    def test_unhashed_bundle_discloses_incompleteness_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve() / "repo"
            repository.mkdir()
            _fixture(repository)
            bundle = repository / "bundle"
            prepare_bundle(repository, bundle)
            marker = json.loads((bundle / "bundle-manifest.json").read_text())
            self.assertFalse(marker["full_source_sha256_complete"])
            verify_bundle(bundle)
            (bundle / "README.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_bundle(bundle)

    def test_publisher_requires_consent_and_authenticated_nonconflicting_target(self):
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve() / "repo"
            repository.mkdir()
            _fixture(repository)
            bundle = repository / "bundle"
            prepare_bundle(repository, bundle)
            api = Mock()
            api.repo_info.return_value = SimpleNamespace(private=False)
            api.list_repo_files.return_value = [".gitattributes"]
            api.upload_folder.return_value = SimpleNamespace(commit_url="https://example.test/commit")
            with self.assertRaisesRegex(ValueError, "confirm-public"):
                publish_bundle(bundle, "Poupou/oparq-benchmarks", api=api)
            api.whoami.assert_not_called()
            dry = publish_bundle(bundle, "Poupou/oparq-benchmarks", dry_run=True, api=api)
            self.assertFalse(dry["published"])
            api.whoami.assert_not_called()
            result = publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True, api=api)
            self.assertTrue(result["published"])
            api.whoami.assert_called_once()
            self.assertEqual(api.upload_folder.call_args.kwargs["repo_type"], "dataset")
            self.assertNotIn("delete_patterns", api.upload_folder.call_args.kwargs)
            api.repo_info.return_value = SimpleNamespace(private=True)
            with self.assertRaisesRegex(ValueError, "private"):
                publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True, api=api)
            api.repo_info.return_value = SimpleNamespace(private=False)
            api.list_repo_files.return_value = ["unrelated.csv"]
            with self.assertRaisesRegex(ValueError, "nonempty"):
                publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True, api=api)
            with self.assertRaisesRegex(ValueError, "unrelated"):
                publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True, update_existing=True, api=api)

    def test_changed_source_cannot_receive_historical_hash_claim(self):
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve() / "repo"
            repository.mkdir()
            source = _fixture(repository)
            source.write_bytes(b"changed source")
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                prepare_bundle(repository, repository / "bundle", hash_source=True)
            self.assertFalse((repository / "bundle").exists())


if __name__ == "__main__":
    unittest.main()
