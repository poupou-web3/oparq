#!/usr/bin/env python3
"""Prepare an audited results-only benchmark bundle; never upload source rows."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import tomllib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
REPORT_NAMES = ("full-corpus-2026-09-30", "saved-plan-engines-2026-09-30")
CHECKSUM_FILE = "bundle-checksums.json"
RESULT_SCHEMA = pa.schema([
    ("dataset", pa.string()), ("algorithm", pa.string()), ("case", pa.string()), ("engine", pa.string()),
    ("scope", pa.string()), ("rows", pa.int64()), ("source_files", pa.int64()),
    ("source_bytes", pa.int64()), ("output_bytes", pa.int64()),
    ("compressed_column_bytes", pa.int64()), ("savings_vs_no_sort", pa.float64()),
    ("compression", pa.string()), ("compression_level", pa.int64()),
    ("row_group_size", pa.int64()), ("sort_keys_json", pa.string()),
    ("planning_seconds", pa.float64()), ("input_read_seconds", pa.float64()),
    ("permutation_seconds", pa.float64()), ("gathering_seconds", pa.float64()),
    ("combined_scan_sort_gather_seconds", pa.float64()), ("schema_restore_seconds", pa.float64()),
    ("writing_seconds", pa.float64()),
    ("rewrite_wall_seconds", pa.float64()), ("range_sampling_seconds", pa.float64()),
    ("key_sampling_source_scans", pa.int64()), ("source_scans_for_sort", pa.int64()),
    ("range_sample_method", pa.string()), ("reused_result_of", pa.string()),
    ("correctness_scope", pa.string()), ("status", pa.string()),
])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def _redact(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, replacements) for item in value]
    if isinstance(value, str):
        for origin, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
            value = value.replace(origin, target)
        return value
    return value


def _report_replacements(report: dict[str, Any], repository: Path) -> dict[str, str]:
    config = report["configuration"]
    replacements = {str(repository.resolve()): "reproduction"}
    for key, target in (("data_root", "inputs"), ("temp_root", "<runtime-temp>")):
        if config.get(key):
            original = Path(config[key])
            replacements[str(original)] = target
            replacements[str(original.resolve())] = target
    return replacements


def _normalized_rows(report: dict[str, Any], *, engines: bool) -> list[dict[str, Any]]:
    rows = []
    for dataset in report["datasets"]:
        baseline = next(result for result in dataset["results"]
                        if (result.get("case") == "none_arrow" if engines
                            else result["requested_algorithm"] == "none"))
        for result in dataset["results"]:
            plan = dataset["fixed_plan"] if engines else result["plan"]
            writer = dataset["files"][0]["writer"] if engines else dataset["writer"]
            no_sort = result.get("case") == "none_arrow" if engines else not plan["sort_keys"]
            engine = (result["case"].rsplit("_", 1)[-1] if engines
                      else "duckdb" if plan["sort_keys"] else "arrow")
            rows.append({
                "dataset": dataset["name"],
                "algorithm": result["case"].rsplit("_", 1)[0] if engines else result["requested_algorithm"],
                "case": result["case"] if engines else result["requested_algorithm"],
                "engine": engine, "scope": "per_physical_file" if engines else "global_dataset",
                "rows": result["rows"], "source_files": len(dataset["inventory"]["files"]),
                "source_bytes": dataset["inventory"]["file_size_bytes"], "output_bytes": result["bytes"],
                "compressed_column_bytes": result.get("compressed_column_bytes"),
                "savings_vs_no_sort": 1 - result["bytes"] / baseline["bytes"],
                "compression": writer["compression"], "compression_level": writer["compression_level"],
                "row_group_size": writer["row_group_size"],
                "sort_keys_json": json.dumps([] if no_sort else plan["sort_keys"]),
                "planning_seconds": result["planning_seconds"], "input_read_seconds": result.get("input_read_seconds"),
                "permutation_seconds": result.get("permutation_seconds"), "gathering_seconds": result.get("gathering_seconds"),
                "combined_scan_sort_gather_seconds": result.get("sort_seconds") if engines and engine == "duckdb"
                    else result.get("scan_sort_gather_seconds"),
                "schema_restore_seconds": result.get("schema_restore_seconds"),
                "writing_seconds": result["writing_seconds"], "rewrite_wall_seconds": result.get("rewrite_wall_seconds"),
                "range_sampling_seconds": result.get("range_sampling_seconds"),
                "key_sampling_source_scans": result.get("key_sampling_source_scans"),
                "source_scans_for_sort": result.get("source_scans_for_sort"),
                "range_sample_method": result.get("range_sample_method"),
                "reused_result_of": result.get("reused_full_result_of"),
                "correctness_scope": "exact_all_values_and_stable_arrow_order" if engines else "all_rows_probabilistic_hash_and_global_key_order",
                "status": result["status"],
            })
    return rows


def _input_manifest(report: dict[str, Any], data_root: Path, *, hash_source: bool) -> dict[str, Any]:
    recorded_root = Path(report["configuration"]["data_root"]).resolve()
    entries = []
    for dataset in report["datasets"]:
        snapshots = {str(Path(item["path"]).resolve()): item for item in dataset["inventory"]["snapshot"]}
        for entry in dataset["inventory"]["files"]:
            original = Path(entry["path"]).resolve()
            relative = original.relative_to(recorded_root)
            source = data_root.resolve() / relative
            snapshot = snapshots[str(original)]
            record = {"dataset": dataset["name"], "path": "inputs/" + relative.as_posix(),
                      "size_bytes": entry["size_bytes"], "rows": entry["rows"],
                      "row_groups": entry["row_groups"], "benchmark_mtime_ns": snapshot["mtime_ns"],
                      "sha256": None, "matches_benchmark_snapshot": None}
            if hash_source:
                before = source.stat()
                record["matches_benchmark_snapshot"] = (before.st_size == snapshot["size"]
                                                         and before.st_mtime_ns == snapshot["mtime_ns"])
                if not record["matches_benchmark_snapshot"]:
                    raise RuntimeError(f"source no longer matches benchmark snapshot: {relative}")
                record["sha256"] = sha256_file(source)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"source changed while hashing: {relative}")
            entries.append(record)
    return {"format_version": 1, "source_rows_included": False, "files": entries,
            "full_sha256_complete": all(item["sha256"] is not None for item in entries),
            "hash_note": "SHA256 covers each complete original Parquet file, not footer bytes or sampled rows."
                         if hash_source else "Source files were not hashed. Exact input-byte reproduction is not established."}


def _source_provenance(report: dict[str, Any]) -> dict[str, Any]:
    return {"source_rows_included": False, "datasets": [{
        "name": dataset["name"], "created_by": dataset["inventory"]["created_by"],
        "recorded_codec": dataset["inventory"]["source_codecs"], "original_compression_level": None,
        "upstream_url": None, "upstream_revision": None, "export_query": None,
        "redistribution_license": None,
        "provenance_status": "private_or_unavailable_source" if dataset["name"] == "solana" else "unverified_export_provenance",
        "note": ("User-provided Solana export; no public exact source or redistribution permission established."
                 if dataset["name"] == "solana" else
                 "ClickHouse-created export according to Parquet metadata. Dataset name is not proof of a specific upstream revision or export query."),
    } for dataset in report["datasets"]],
        "related_public_references_not_exact_sources": [
            "https://clickhouse.com/blog/announcing-the-new-sql-playground",
            "https://github.com/ClickHouse/ClickBench",
        ],
        "limitation": "Results only. Inputs are republished separately only where upstream terms allow it (scripts/prepare_inputs.py); supply the others yourself and verify every full-file SHA256. Exact upstream acquisition remains undocumented."}


def _copy_snapshot(repository: Path, destination: Path) -> None:
    # Strict allowlist excludes source Parquet, cloned projects, environments,
    # credentials, outputs, and unrelated user files.
    paths = [repository / name for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE")]
    for directory, pattern in (("src/oparq", "*.py"), ("benchmarks", "*.py"),
                               ("tests", "*.py"), ("scripts", "*.py")):
        paths.extend(sorted((repository / directory).glob(pattern)))
    for name in ("src/oparq/py.typed", "CHANGELOG.md", "CONTRIBUTING.md",
                 "benchmarks/README.md"):
        path = repository / name
        if path.exists():
            paths.append(path)
    paths.extend(path for path in sorted((repository / "docs").rglob("*.md")) if path.is_file())
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"snapshot input must be a regular file: {path.name}")
        target = destination / path.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat()
        shutil.copyfile(path, target)
        if path.parent == repository / "benchmarks" and path.suffix == ".md":
            # Reports stay alongside their local runners, but the curated
            # sanitized checkpoints live at the publication bundle root.
            target.write_text(target.read_text().replace("(results/", "(../../results/"))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source code changed during snapshot: {path.name}")


def _generated_summary(report: dict[str, Any], name: str, *, engines: bool) -> str:
    """Derive publication evidence from JSON, never from stale narrative files."""
    runner = "compare_saved_plan.py" if engines else "full_corpus.py"
    rows = _normalized_rows(report, engines=engines)
    title = "Saved-plan engine comparison" if engines else "Full-corpus compression comparison"
    lines = [
        f"# {title}", "",
        f"Generated from [checkpoint JSON](../results/{name}.json). "
        f"Reproduction runner: [{runner}](../reproduction/benchmarks/{runner}).",
        "", f"Recorded versions: `{json.dumps(report['versions'], sort_keys=True)}`.", "",
        "Historical development measurements, not a fresh test of the bundled current code. "
        "Original inputs are not included in this results bundle; inputs whose upstream "
        "terms allow it are published separately.", "",
        "Ordering scope: " + ("each physical file separately." if engines else "all rows globally within each dataset."), "",
        "Rows, codec/level, row-group size, and writer are held fixed against the no-sort baseline. "
        "Source encoder levels are not recoverable from Parquet footers. "
        "Codec levels here are explicit benchmark settings, not inferred source levels.", "",
        "Correctness: " + ("exact logical value/order comparisons with an independent stable Arrow permutation."
                           if engines else "all-row probabilistic fingerprints, schema/row counts, and sorted-key checks."), "",
        "Timings are single local measurements with cache effects. DuckDB scan/sort/gather is fused; "
        "unavailable component times are not zero. Reused cases and historical method changes "
        "are identified in the JSON. A best result selected after measurement is not an auto-planner result.", "",
        "| Dataset | Case | Rows | Output bytes | Saved vs no-sort | Planning s | Writing s | Keys |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        keys = row["sort_keys_json"].replace("|", "\\|").replace("`", "'")
        lines.append(f"| {row['dataset']} | {row['case']} | {row['rows']:,} | {row['output_bytes']:,} "
                     f"| {row['savings_vs_no_sort']:+.2%} | {row['planning_seconds']:.4f} "
                     f"| {row['writing_seconds']:.4f} | `{keys}` |")
    lines += ["", "See checkpoint JSON and normalized tables for complete stage times, "
              "source inventories, row-group geometry, data-quality caveats, and validations.", ""]
    return "\n".join(lines)


def _environment(reports: list[dict[str, Any]]) -> dict[str, Any]:
    versions = {}
    for name in ("pyarrow", "duckdb", "uv-build", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"prepared_at_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "installed_dependencies": versions, "reported_run_versions": [report["versions"] for report in reports],
            "historical_snapshot_note": "The bundled snapshot is the current preparation-time source. Earlier cases were run during development and were not individually archived; recorded versions/method changes remain in the checkpoints. Timings are hardware-dependent."}


def _project_links(repository: Path) -> str:
    """Link the card to its source repository, PyPI package, and republished inputs."""

    with (repository / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream).get("project", {})
    urls = project.get("urls", {})
    links = [f"[source code]({urls['Repository']})"] if "Repository" in urls else []
    if project.get("name"):
        links.append(f"[PyPI package](https://pypi.org/project/{project['name']}/)")
    if "Benchmark inputs" in urls:
        links.append(f"[benchmark inputs]({urls['Benchmark inputs']})")
    return " · ".join(links)


def _dataset_card(full: dict[str, Any], manifest: dict[str, Any], links: str = "") -> str:
    total_rows = sum(dataset["inventory"]["rows"] for dataset in full["datasets"])
    total_files = len(manifest["files"])
    return f"""---
license: mit
tags:
- benchmark
- parquet
- compression
configs:
- config_name: full_corpus
  data_files:
  - split: train
    path: tables/full_corpus.parquet
- config_name: saved_plan_engines
  data_files:
  - split: train
    path: tables/saved_plan_engines.parquet
---
# oparq benchmark results

{links + chr(10) + chr(10) if links else ""}Results only: **{len(full['datasets'])} datasets, {total_rows:,} input rows,
{total_files:,} source files**. No original source rows are distributed.
The viewer rows describe algorithm measurements, not individual source events.

Full-corpus results sort all input rows globally using fixed writer settings;
only key planning is sampled. The separate Arrow/DuckDB comparison sorts
within each physical file using saved keys, and must not be conflated with
the global comparison. ZSTD1 is an explicit controlled setting, not an
inferred source level; Snappy remains Snappy. Nulls are encoded automatically
by Parquet definition levels.

See `summaries/`, `results/` (path-sanitized checkpoints), `input-manifest.json`,
`source-provenance.json`, and `REPRODUCE.md`. Full-file SHA256 completeness:
**{manifest['full_sha256_complete']}**. Bundle files have SHA256 checksums.

Important limitations: the original export queries and upstream immutable
revisions were not established. Inputs whose upstream terms allow
republication are published separately with their licences; the others
are not. **The full corpus is not independently
reproducible from public inputs alone**, even when all local source hashes
are supplied. The snapshot records current code, not a separately archived
historical executable for each development-time case. Timings are single
local measurements with cache and hardware effects. Full-corpus content
verification uses probabilistic all-row fingerprints; the engine comparison
checks every value against an independent stable Arrow permutation.

The `pypi` source dataset contains invalid UTF-8 bytes in a STRING column. Its benchmark
preserves raw bytes through explicit binary views and restores the original
schema; no cleaning is performed, and the source remains nonconforming.
See the checkpoint's `data_quality` record.

The benchmark results, documentation, and bundled oparq source are MIT
licensed; see `reproduction/LICENSE`. This does not license the original
source data or imply source-data redistribution permission.
"""


REPRODUCE = """# Reproduction

This bundle excludes original input rows. Inputs whose terms allow it are
published separately, byte for byte; obtain the others yourself. Place files
under `reproduction/local/data/source/`, stripping the `inputs/` prefix from
each input-manifest path. Verify every complete-file SHA256 before claiming
input-byte reproduction. Sizes, schemas, and row counts alone do not identify
exact inputs. Missing upstream/export provenance and the unpublished inputs
prevent complete public reruns.

The bundled `reproduction/` tree is an allowlisted current source snapshot:
package code, runners, tests, publication scripts, pyproject.toml and uv.lock.
Check `bundle-checksums.json` first. Historical cases were generated during
development; use checkpoint method/version annotations when comparing them.

```sh
cd reproduction
uv sync --locked --extra duckdb
uv run --locked python -m unittest discover -s tests -q
uv run --locked python -m benchmarks.full_corpus \\
  --data-root local/data/source --compression-level 1 --row-group-size 1000000 \\
  --output benchmarks/results/full-corpus-rerun.json
uv run --locked python -m benchmarks.compare_saved_plan \\
  --data-root local/data/source \\
  --plan-checkpoint benchmarks/results/full-corpus-rerun.json \\
  --datasets stock,hits,solana --algorithm portfolio \\
  --compression-level 1 --row-group-size 1000000 \\
  --output benchmarks/results/saved-plan-engines-rerun.json
```

Use a new output file for new measurements; existing reports resume validated
cases. Full-corpus sorts use bounded external ranges and additional key scans;
read their timing/count fields. The engine run uses per-file ordering and
excludes independent correctness verification from execution timings. Run
without competing heavy processes. Memory, spill, scanner prefetch, dictionary
page/batch layout, OS caches and CPU hardware affect timings and sometimes
encoded byte counts despite equal rows and row-group geometry.

To reuse the published learned keys without rerunning the global benchmark,
run the engine command with
`--plan-checkpoint ../results/full-corpus-2026-09-30.json` instead. The sanitized
published checkpoint is a record, not a resumable local execution checkpoint:
its input/temp paths were deliberately changed to portable placeholders.
"""


def prepare_bundle(repository: Path, destination: Path, *, data_root: Path | None = None,
                   hash_source: bool = False) -> dict[str, Any]:
    repository = repository.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("choose a new bundle output directory")
    reports = [json.loads((repository / "benchmarks/results" / (name + ".json")).read_text())
               for name in REPORT_NAMES]
    full, engines = reports
    if full.get("status") != "pass" or not full.get("full_corpus_complete"):
        raise ValueError("full-corpus checkpoint is not complete and passing")
    if engines.get("status") != "pass" or not engines.get("requested_datasets_complete"):
        raise ValueError("engine checkpoint is not complete and passing")
    source_root = data_root or (repository / "local/data/source"
                               if (repository / "local/data/source").is_dir()
                               else Path(full["configuration"]["data_root"]))
    manifest = _input_manifest(full, source_root, hash_source=hash_source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="oparq-bundle-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for report, name, is_engines in zip(reports, REPORT_NAMES, (False, True)):
            replacements = _report_replacements(report, repository)
            _write_json(staging / "results" / (name + ".json"), _redact(report, replacements))
            summary = staging / "summaries" / (name + ".md")
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(_generated_summary(report, name, engines=is_engines))
            rows = _normalized_rows(report, engines=is_engines)
            table = pa.Table.from_pylist(rows, schema=RESULT_SCHEMA)
            table_name = "saved_plan_engines" if is_engines else "full_corpus"
            (staging / "tables").mkdir(exist_ok=True)
            pq.write_table(table, staging / "tables" / (table_name + ".parquet"), compression="zstd", compression_level=1)
            with (staging / "tables" / (table_name + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=RESULT_SCHEMA.names)
                writer.writeheader()
                writer.writerows(rows)
        _copy_snapshot(repository, staging / "reproduction")
        _write_json(staging / "input-manifest.json", manifest)
        _write_json(staging / "source-provenance.json", _source_provenance(full))
        _write_json(staging / "environment.json", _environment(reports))
        (staging / "README.md").write_text(_dataset_card(full, manifest, _project_links(repository)))
        (staging / "REPRODUCE.md").write_text(REPRODUCE)
        _write_json(staging / "bundle-manifest.json", {
            "format_version": 1, "kind": "oparq_results_only", "source_rows_included": False,
            "input_rows": sum(dataset["inventory"]["rows"] for dataset in full["datasets"]),
            "input_files": len(manifest["files"]), "full_source_sha256_complete": manifest["full_sha256_complete"],
            "independently_publicly_reproducible": False,
        })
        checksums = {path.relative_to(staging).as_posix(): {"size_bytes": path.stat().st_size,
                                                         "sha256": sha256_file(path)}
                     for path in sorted(staging.rglob("*")) if path.is_file()}
        _write_json(staging / CHECKSUM_FILE, {"format_version": 1, "files": checksums,
                                            "note": "This checksum file excludes itself to avoid a self-reference."})
        staging.rename(destination)
    return {"output": str(destination), "files": len(checksums) + 1,
            "source_files": len(manifest["files"]), "full_source_sha256_complete": manifest["full_sha256_complete"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "local/publication/oparq-benchmarks")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--hash-source", action="store_true", help="Read every complete source file for SHA256; never copy or upload source rows")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_bundle(args.repository, args.output, data_root=args.data_root,
                                   hash_source=args.hash_source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
