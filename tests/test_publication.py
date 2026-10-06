import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pyarrow.parquet as pq

from scripts.prepare_benchmarks import CHECKSUM_FILE, RESULT_SCHEMA, prepare_bundle, sha256_file
from scripts.prepare_inputs import NOT_PUBLISHED, PUBLISHED, prepare_inputs
from scripts.publish_benchmarks import publish_bundle, verify_bundle


def _fixture(repository):
    for name, value in {"README.md": "Fixture", "LICENSE": "MIT License", "uv.lock": "version = 1",
                        "pyproject.toml": '[project]\nname="oparq"\nversion="0.3.0"\n[project.urls]\n'
                                          '"Benchmark inputs"="https://huggingface.co/datasets/example/inputs"',
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


def _inputs_fixture(repository, extra=("hits",)):
    """A passing checkpoint with every cleared dataset plus restricted ones."""

    (repository / "pyproject.toml").write_text(
        '[project]\nname="oparq"\n[project.urls]\nRepository="https://github.com/example/oparq"\n')
    datasets = []
    for name in (*PUBLISHED, *extra):
        recorded = f"inputs/clickhouse/{name}/part-00000.parquet"
        source = repository / "local/data/source" / recorded.removeprefix("inputs/")
        source.parent.mkdir(parents=True)
        source.write_bytes(f"{name} source bytes".encode())
        stat = source.stat()
        datasets.append({"name": name, "inventory": {
            "rows": 10, "files": [{"path": recorded, "size_bytes": stat.st_size, "rows": 10}],
            "snapshot": [{"path": recorded, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]}})
    report = repository / "benchmarks/results/full-corpus-2026-09-30.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "pass", "datasets": datasets}))


def _reseal(bundle):
    checksums = {path.relative_to(bundle).as_posix(): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(bundle.rglob("*")) if path.is_file() and path.name != CHECKSUM_FILE}
    (bundle / CHECKSUM_FILE).write_text(json.dumps({"format_version": 1, "files": checksums}))


class InputsPublicationTests(unittest.TestCase):
    def test_bundle_holds_only_cleared_datasets_with_their_terms(self):
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve()
            _inputs_fixture(repository)
            bundle = repository / "inputs-bundle"
            result = prepare_inputs(repository, bundle)
            self.assertEqual(result["datasets"], list(PUBLISHED))
            files = verify_bundle(bundle)
            data = {path for path in files if path.startswith("clickhouse/")}
            self.assertEqual(data, {f"clickhouse/{name}/part-00000.parquet" for name in PUBLISHED})
            self.assertEqual((bundle / "clickhouse/ontime/part-00000.parquet").read_bytes(), b"ontime source bytes")
            card = (bundle / "README.md").read_text()
            for terms in PUBLISHED.values():
                self.assertIn(terms["attribution"], card)
            self.assertIn("- cc-by-sa-4.0", card)
            self.assertIn(f"| `hits` | {NOT_PUBLISHED['hits']} |", card)
            self.assertIn("[source code](https://github.com/example/oparq)", card)
            (bundle / "clickhouse/covid/part-00000.parquet").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_bundle(bundle)

    def test_preparation_requires_decisions_snapshots_and_matching_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve()
            _inputs_fixture(repository, extra=("hits", "undecided"))
            with self.assertRaisesRegex(ValueError, "without a republication decision"):
                prepare_inputs(repository, repository / "bundle")
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve()
            _inputs_fixture(repository)
            manifest = repository / "input-manifest.json"
            manifest.write_text(json.dumps({"files": [
                {"path": "inputs/clickhouse/ontime/part-00000.parquet", "sha256": "0" * 64}]}))
            with self.assertRaisesRegex(RuntimeError, "published input manifest"):
                prepare_inputs(repository, repository / "bundle", input_manifest=manifest)
            (repository / "local/data/source/clickhouse/trips/part-00000.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                prepare_inputs(repository, repository / "bundle")
            self.assertFalse((repository / "bundle").exists())

    def test_publisher_rejects_restricted_datasets_even_with_consistent_checksums(self):
        with tempfile.TemporaryDirectory() as name:
            repository = Path(name).resolve()
            _inputs_fixture(repository)
            bundle = repository / "inputs-bundle"
            prepare_inputs(repository, bundle)
            restricted = bundle / "clickhouse/hits/part-00000.parquet"
            restricted.parent.mkdir()
            restricted.write_bytes(b"restricted rows")
            _reseal(bundle)
            with self.assertRaisesRegex(ValueError, "cleared datasets"):
                verify_bundle(bundle)
            restricted.unlink()
            marker = json.loads((bundle / "bundle-manifest.json").read_text())
            (bundle / "bundle-manifest.json").write_text(json.dumps({**marker, "datasets": [*marker["datasets"], "hits"]}))
            _reseal(bundle)
            with self.assertRaisesRegex(ValueError, "not cleared for republication"):
                verify_bundle(bundle)
            (bundle / "bundle-manifest.json").write_text(json.dumps(marker))
            _reseal(bundle)
            api = Mock()
            api.repo_info.return_value = SimpleNamespace(private=False)
            api.list_repo_files.return_value = ["bundle-manifest.json"]
            remote_marker = repository / "remote-marker.json"
            remote_marker.write_text(json.dumps({"kind": "oparq_results_only"}))
            api.hf_hub_download.return_value = str(remote_marker)
            with self.assertRaisesRegex(ValueError, "unrelated"):
                publish_bundle(bundle, "Poupou/oparq-benchmark-inputs", confirm_public=True,
                               update_existing=True, api=api)
            api.list_repo_files.return_value = [".gitattributes"]
            api.upload_folder.return_value = SimpleNamespace(commit_url="https://example.test/commit")
            publish_bundle(bundle, "Poupou/oparq-benchmark-inputs", confirm_public=True, api=api)
            self.assertIn("cleared for republication", api.upload_folder.call_args.kwargs["commit_message"])


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
            self.assertIn("[PyPI package](https://pypi.org/project/oparq/)", (bundle / "README.md").read_text())
            self.assertIn("[benchmark inputs](https://huggingface.co/datasets/example/inputs)",
                          (bundle / "README.md").read_text())
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
            self.assertIsNone(api.upload_folder.call_args.kwargs["delete_patterns"])
            api.repo_info.return_value = SimpleNamespace(private=True)
            with self.assertRaisesRegex(ValueError, "private"):
                publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True, api=api)
            api.repo_info.return_value = SimpleNamespace(private=False)
            api.list_repo_files.return_value = ["unrelated.csv"]
            with self.assertRaisesRegex(ValueError, "nonempty"):
                publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True, api=api)
            with self.assertRaisesRegex(ValueError, "unrelated"):
                publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True, update_existing=True, api=api)
            api.list_repo_files.return_value = [".gitattributes", "README.md", "bundle-manifest.json",
                                                "summaries/old[1].md"]
            api.hf_hub_download.return_value = str(bundle / "bundle-manifest.json")
            result = publish_bundle(bundle, "Poupou/oparq-benchmarks", confirm_public=True,
                                    update_existing=True, api=api)
            self.assertEqual(result["deleted"], ["summaries/old[1].md"])
            self.assertEqual(api.upload_folder.call_args.kwargs["delete_patterns"], ["summaries/old[[]1].md"])

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
