#!/usr/bin/env python3
"""Prepare a benchmark-inputs bundle from datasets whose terms allow republication.

Files are copied byte for byte and must still match the benchmark snapshot and,
when supplied, the published results' input manifest. The dataset card states
each dataset's source, licence and required attribution, and lists every other
benchmark input with the reason it is not republished. Upload the bundle with
``publish_benchmarks.py``; this script never uploads anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
from typing import Any

if __package__:
    from .prepare_benchmarks import CHECKSUM_FILE, ROOT, _write_json, sha256_file
else:
    from prepare_benchmarks import CHECKSUM_FILE, ROOT, _write_json, sha256_file


KIND = "oparq_benchmark_inputs"
REPORT = "benchmarks/results/full-corpus-2026-09-30.json"

# Upstream terms checked on 2026-10-05. Only these datasets may be republished.
PUBLISHED: dict[str, dict[str, str]] = {
    "ontime": {
        "title": "US airline on-time performance",
        "license": "other",
        "license_name": "U.S. public domain",
        "source": "U.S. Department of Transportation, Bureau of Transportation Statistics (BTS), "
                  "Reporting Carrier On-Time Performance",
        "source_url": "https://www.transtats.bts.gov/",
        "attribution": "Source: U.S. Department of Transportation, Bureau of Transportation Statistics.",
    },
    "trips": {
        "title": "New York City taxi trips",
        "license": "other",
        "license_name": "NYC Open Data (open by default)",
        "source": "New York City Taxi & Limousine Commission (TLC) trip records, with NYC "
                  "neighborhood tabulation areas and census tracts",
        "source_url": "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
        "attribution": "Source: New York City Taxi & Limousine Commission (TLC), via NYC Open Data.",
    },
    "lineorder": {
        "title": "Star Schema Benchmark LINEORDER (synthetic)",
        "license": "other",
        "license_name": "synthetic benchmark data",
        "source": "Star Schema Benchmark LINEORDER table generated with ssb-dbgen; synthetic, "
                  "not real business data",
        "source_url": "https://clickhouse.com/docs/get-started/sample-datasets/star-schema",
        "attribution": "Star Schema Benchmark, P. O'Neil, E. O'Neil and X. Chen.",
    },
    "cell_towers": {
        "title": "Cell towers",
        "license": "cc-by-sa-4.0",
        "license_name": "CC BY-SA 4.0",
        "source": "OpenCelliD, the open database of cell towers",
        "source_url": "https://opencellid.org/",
        "attribution": "OpenCelliD Project, licensed under CC BY-SA 4.0 "
                       "(https://creativecommons.org/licenses/by-sa/4.0/). Changes: a subset of "
                       "rows converted to Parquet; this subset is shared under the same licence.",
    },
    "covid": {
        "title": "COVID-19 epidemiology",
        "license": "cc-by-4.0",
        "license_name": "CC BY 4.0, with per-source terms",
        "source": "Google COVID-19 Open Data, epidemiology table",
        "source_url": "https://goo.gle/covid-19-open-data",
        "attribution": "Google COVID-19 Open Data (goo.gle/covid-19-open-data), licensed under CC BY 4.0.",
        "notice": "Rows produced by third parties remain under their original terms, listed in "
                  "the source's epidemiology table documentation: notably the New York Times US "
                  "data (attribution, non-commercial use), Sweden and Thailand (fair use), Chile "
                  "and Romania (custom terms), and Alaska (no licence specified).",
    },
}

# Every other benchmark input, and why its rows are not republished.
NOT_PUBLISHED: dict[str, str] = {
    "hits": "ClickBench web-analytics data is CC BY-NC-SA 4.0 (non-commercial only).",
    "noaa_v2": "Non-U.S. GHCN-Daily station data may not be redistributed for commercial "
               "activities (WMO Resolution 40).",
    "uk_price_paid": "The address fields carry Royal Mail and Ordnance Survey rights limited to "
                     "personal or non-commercial use.",
    "hackernews": "Hacker News posts and comments belong to their authors; no content licence.",
    "hackernews_history": "Hacker News posts and comments belong to their authors; no content licence.",
    "hackernews_top": "Hacker News data; no content licence.",
    "tranco": "No licence; the Tranco list merges sources including Cloudflare Radar (CC BY-NC 4.0).",
    "pypi": "PyPI file metadata from pypi-data / py-code.org; no licence could be found.",
    "workflow_jobs": "GitHub Actions job data collected by ClickHouse; no licence (rights reserved).",
    "opensky": "The OpenSky Network licence prohibits redistribution (non-profit research only).",
    "recipes": "RecipeNLG is licensed for non-commercial research only and may not be redistributed.",
    "forex": "HistData.com data has no open licence; redistribution needs HistData's permission.",
    "stock": "Source not identified; market data is normally licensed without redistribution rights.",
    "cisco_umbrella": "Copyright Cisco Umbrella; free to download but not licensed for republication.",
    "dns": "Source not identified.",
    "solana": "Private export; no public source or redistribution permission.",
}


def _project_urls(repository: Path) -> dict[str, str]:
    with (repository / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream).get("project", {})
    urls = dict(project.get("urls", {}))
    if project.get("name"):
        urls["PyPI"] = f"https://pypi.org/project/{project['name']}/"
    return urls


def _card(records: list[dict[str, Any]], urls: dict[str, str]) -> str:
    licenses = sorted({PUBLISHED[record["name"]]["license"] for record in records})
    lines = ["---", "license:", *(f"- {value}" for value in licenses),
             "pretty_name: oparq benchmark inputs", "tags:", "- parquet", "- benchmark", "configs:"]
    for record in records:
        lines += [f"- config_name: {record['name']}", "  data_files:", "  - split: train",
                  f"    path: clickhouse/{record['name']}/*.parquet"]
    links = [f"[{label}]({url})" for label, url in (
        ("source code", urls.get("Repository")), ("PyPI package", urls.get("PyPI")),
        ("benchmark results", urls.get("Benchmarks"))) if url]
    lines += ["---", "# oparq benchmark inputs", ""]
    if links:
        lines += [" · ".join(links), ""]
    lines += [
        "Exact input files, byte for byte, behind the oparq full-corpus benchmark for the",
        "datasets whose terms allow republication. `bundle-checksums.json` and the results",
        "dataset's `input-manifest.json` give the SHA-256 of every file; the manifest's",
        "`inputs/` prefix corresponds to this repository's root.",
        "",
        "The files were exported as Parquet from the ClickHouse public playground",
        "(play.clickhouse.com); some are subsets of the upstream tables. **There is no",
        "overall licence: each dataset keeps its own terms, stated below. Credit each",
        "source as required when you reuse its data.**",
        "",
        "## Published datasets",
        "",
    ]
    for record in records:
        terms = PUBLISHED[record["name"]]
        lines += [
            f"### `{record['name']}`: {terms['title']}",
            "",
            f"- Rows: {record['rows']:,} in {record['files']:,} files",
            f"- Source: [{terms['source']}]({terms['source_url']})",
            f"- Licence: {terms['license_name']}",
            f"- Required attribution: {terms['attribution']}",
        ]
        if "notice" in terms:
            lines.append(f"- Notice: {terms['notice']}")
        lines.append("")
    lines += [
        "## Not published",
        "",
        "These benchmark inputs are not republished because their terms do not allow it,",
        "or because no licence could be established. Their results, schemas and file",
        "hashes remain in the benchmark results dataset.",
        "",
        "| Dataset | Reason |",
        "| --- | --- |",
        *(f"| `{name}` | {reason} |" for name, reason in sorted(NOT_PUBLISHED.items())),
        "",
        "This page reports the terms published by each source; it is not legal advice.",
        "",
    ]
    return "\n".join(lines)


def prepare_inputs(repository: Path, destination: Path, *, data_root: Path | None = None,
                   input_manifest: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("choose a new bundle output directory")
    report = json.loads((repository / REPORT).read_text())
    if report.get("status") != "pass":
        raise ValueError("full-corpus checkpoint is not passing")
    records = {dataset["name"]: dataset for dataset in report["datasets"]}
    undecided = sorted(set(records) - set(PUBLISHED) - set(NOT_PUBLISHED))
    if undecided:
        raise ValueError(f"datasets without a republication decision: {undecided}")
    missing = sorted(set(PUBLISHED) - set(records))
    if missing:
        raise ValueError(f"published datasets missing from the checkpoint: {missing}")
    expected: dict[str, str] = {}
    if input_manifest is not None:
        expected = {item["path"]: item["sha256"]
                    for item in json.loads(input_manifest.read_text())["files"] if item["sha256"]}
    source_root = (data_root or repository / "local/data/source").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with tempfile.TemporaryDirectory(prefix="oparq-inputs-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        for name in PUBLISHED:
            inventory = records[name]["inventory"]
            snapshots = {item["path"]: item for item in inventory["snapshot"]}
            for entry in inventory["files"]:
                recorded = entry["path"]
                if not recorded.startswith("inputs/"):
                    raise ValueError(f"unexpected input path: {recorded}")
                relative = recorded.removeprefix("inputs/")
                source = source_root / relative
                snapshot = snapshots[recorded]
                before = source.stat()
                if (before.st_size, before.st_mtime_ns) != (snapshot["size"], snapshot["mtime_ns"]):
                    raise RuntimeError(f"source no longer matches the benchmark snapshot: {relative}")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                digest = sha256_file(target)
                if recorded in expected and expected[recorded] != digest:
                    raise RuntimeError(f"copy does not match the published input manifest: {relative}")
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"source changed while copying: {relative}")
            summaries.append({"name": name, "rows": inventory["rows"], "files": len(inventory["files"]),
                              "license": PUBLISHED[name]["license_name"]})
        (staging / "README.md").write_text(_card(summaries, _project_urls(repository)))
        _write_json(staging / "bundle-manifest.json", {
            "format_version": 1, "kind": KIND, "source_rows_included": True,
            "datasets": [item["name"] for item in summaries], "summary": summaries,
            "not_published": sorted(NOT_PUBLISHED),
            "matched_published_input_manifest": bool(expected),
        })
        checksums = {path.relative_to(staging).as_posix(): {"size_bytes": path.stat().st_size,
                                                         "sha256": sha256_file(path)}
                     for path in sorted(staging.rglob("*")) if path.is_file()}
        _write_json(staging / CHECKSUM_FILE, {"format_version": 1, "files": checksums,
                                            "note": "This checksum file excludes itself to avoid a self-reference."})
        staging.rename(destination)
    return {"output": str(destination), "files": len(checksums) + 1,
            "datasets": [item["name"] for item in summaries],
            "matched_published_input_manifest": bool(expected)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--input-manifest", type=Path,
                        help="the published results' input-manifest.json; every copied file must match it")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_inputs(args.repository, args.output, data_root=args.data_root,
                                    input_manifest=args.input_manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
