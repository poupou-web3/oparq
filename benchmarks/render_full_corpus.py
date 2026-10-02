"""Render a full-corpus checkpoint without rescanning any source data.

Run ``python -m benchmarks.render_full_corpus`` from the repository root.
Incomplete checkpoints deliberately retain pending and failed dataset rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
from typing import Any


REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPOSITORY / "benchmarks" / "results" / "full-corpus-2026-09-30.json"
DEFAULT_OUTPUT = REPOSITORY / "local" / "reports" / "full-corpus-2026-09-30.md"
MIB = 1024**2


def _code(value: Any) -> str:
    return "`" + str(value).replace("|", "\\|").replace("`", "'") + "`"


def _number(value: Any, *, decimals: int = 2) -> str:
    return "—" if value is None else f"{value:,.{decimals}f}"


def _passed(result: dict[str, Any] | None) -> bool:
    return bool(result and result.get("status") == "pass"
                and all(value is not False for value in result.get("checks", {}).values()))


def _failed(result: dict[str, Any] | None) -> bool:
    return bool(result and (result.get("status") == "fail"
                or any(value is False for value in result.get("checks", {}).values())))


def _cases(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result["requested_algorithm"]: result for result in record.get("results", [])}


def _complete(record: dict[str, Any] | None, algorithms: list[str]) -> bool:
    if not record or record.get("status") != "pass":
        return False
    cases = _cases(record)
    return all(_passed(cases.get(algorithm)) for algorithm in algorithms)


def _sort_input_scans(result: dict[str, Any]) -> int | str:
    """Count the narrow reservoir scan as well as full-width range scans."""

    if "source_scans_for_sort" in result:
        return result["source_scans_for_sort"]
    if result.get("plan", {}).get("sort_keys"):
        return result.get("sort_ranges", 0) + result.get("key_sampling_source_scans", 0)
    return 1


def _range_method(result: dict[str, Any]) -> str:
    if not result.get("plan", {}).get("sort_keys"):
        return "n/a — input order"
    return result.get("range_sample_method") or "leading_planning_sample (legacy)"


def _size_cell(result: dict[str, Any] | None, baseline: dict[str, Any] | None,
               *, show_savings: bool = True) -> str:
    if not _passed(result):
        return "failed" if _failed(result) else "pending"
    size = result.get("bytes")
    if size is None:
        return "—"
    cell = f"{size / MIB:,.2f}"
    if show_savings and _passed(baseline) and baseline.get("bytes"):
        saved = (baseline["bytes"] - size) / baseline["bytes"] * 100
        cell += f"; {saved:+.2f}%"
    return cell


def render(report: dict[str, Any], *, json_link: str = "results/full-corpus-2026-09-30.json") -> str:
    """Return deterministic Markdown for exactly the checkpointed results."""

    config = report.get("configuration", {})
    algorithms = list(config.get("algorithms", ("none", "codec_fast", "portfolio")))
    records = {record["name"]: record for record in report.get("datasets", [])}
    manifest = report.get("corpus_datasets")
    names = list(dict.fromkeys(manifest or report.get("requested_datasets") or records))
    names.extend(sorted(set(records).difference(names)))
    complete = [name for name in names if _complete(records.get(name), algorithms)]
    failed = [name for name in names if records.get(name, {}).get("status") == "fail"
              or any(_failed(item) for item in records.get(name, {}).get("results", []))]
    pending = [name for name in names if name not in complete and name not in failed]
    full_complete = bool(manifest) and len(complete) == len(names) and not failed
    title = "Complete" if full_complete else "Incomplete — failures present" if failed else "In progress"
    lines = [
        "# Full-corpus oparq comparison — 30 September 2026",
        "",
        f"Status: **{title}**. {len(complete)} / {len(names)} corpus datasets have every requested method validated; "
        f"{len(failed)} failed and {len(pending)} pending or running.",
        "",
        f"Source of truth: [checkpoint JSON]({json_link}). This document is generated from that checkpoint; "
        "it does not rescan the source files. A pending row is not a sampled benchmark result.",
        "",
    ]
    if not manifest:
        lines += ["The checkpoint has no complete-corpus manifest, so this report cannot establish whole-corpus coverage.", ""]
    if report.get("started_at_utc"):
        lines += [f"Run started: {_code(report['started_at_utc'])}. "
                  + (f"Runner finished: {_code(report['completed_at_utc'])}."
                     if report.get("completed_at_utc") else "The runner has not recorded completion."), ""]
    versions = report.get("versions", {})
    if versions:
        lines += ["Initial recorded environment: " + ", ".join(f"{name} {_code(value)}" for name, value in versions.items())
                  + ". These are historical checkpoint versions, not a claim that every resumed case used the initial package version.", ""]

    rows = sum(records[name].get("inventory", {}).get("rows", 0) for name in complete)
    files = sum(len(records[name].get("inventory", {}).get("files", [])) for name in complete)
    lines += [
        "## Controlled compression results",
        "",
        "Every reported output below contains **all rows of its dataset**. Bounded samples drive key selection "
        "and range boundaries, never the output row set. "
        "Sizes are MiB; signed percentages are bytes saved against that dataset's same-writer `none` output "
        "(negative means larger). Source-file bytes are not the controlled baseline.",
        "",
    ]
    columns = ["Dataset", "Source rows", "Source files", "Output codec / level", "Status"]
    columns += [f"{_code(algorithm)} MiB" + ("; saved" if algorithm != "none" else "") for algorithm in algorithms]
    columns += ["Best validated order"]
    lines += ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] + ["---:", "---:"] + ["---"] * (len(columns) - 3)) + " |"]
    for name in names:
        record = records.get(name, {})
        inventory = record.get("inventory", {})
        cases = _cases(record)
        baseline = cases.get("none")
        writer = record.get("writer", {})
        codec = writer.get("compression")
        level = writer.get("compression_level")
        codec_cell = "pending" if codec is None else _code(codec) + (f" / {level}" if level is not None else " / n/a")
        status = "complete" if name in complete else "failed" if name in failed else (
            record.get("status", "pending") if record.get("status", "pending") in {"pending", "running"} else "partial"
        )
        row = [_code(name), f"{inventory['rows']:,}" if "rows" in inventory else "pending",
               f"{len(inventory['files']):,}" if "files" in inventory else "pending", codec_cell, status]
        row += [_size_cell(cases.get(algorithm), baseline, show_savings=algorithm != "none") for algorithm in algorithms]
        valid = [item for item in cases.values() if _passed(item) and item.get("bytes") is not None]
        best = min(valid, key=lambda item: item["bytes"]) if valid else None
        row += [_code(best["requested_algorithm"]) if best else "pending"]
        lines += ["| " + " | ".join(row) + " |"]

    lines += ["", "Aggregates include **only datasets with all requested methods validated**; they never extrapolate pending inputs.", ""]
    if complete:
        totals = {algorithm: sum(_cases(records[name])[algorithm]["bytes"] for name in complete) for algorithm in algorithms}
        baseline_bytes = totals.get("none")
        lines += [f"Validated coverage: {rows:,} rows across {files:,} source files, {len(complete)} datasets.", "",
                  "| Method | Aggregate bytes | MiB | Saved vs no-sort |",
                  "| --- | ---: | ---: | ---: |"]
        for algorithm, size in totals.items():
            savings = f"{(baseline_bytes - size) / baseline_bytes * 100:+.2f}%" if baseline_bytes else "—"
            lines += [f"| {_code(algorithm)} | {size:,} | {size / MIB:,.2f} | {savings} |"]
        selected = sum(min(_cases(records[name])[algorithm]["bytes"] for algorithm in algorithms) for name in complete)
        savings = f"{(baseline_bytes - selected) / baseline_bytes * 100:+.2f}%" if baseline_bytes else "—"
        lines += [f"| Best measured method per dataset (including no-sort) | {selected:,} | {selected / MIB:,.2f} | {savings} |", "",
                  "The best-per-dataset aggregate is an **after-the-fact selection from full results**, not the outcome of "
                  "a proven sample-only selector. Its savings are an upper bound on this tested method set's chosen mixture, "
                  "not a claim of globally optimal ordering.", ""]
    else:
        lines += ["No complete dataset aggregate is available yet.", ""]

    lines += ["## Selected keys", "", "`(input order)` means no keys were selected. "
              "Identical key orders may share a validated execution result, as labeled in the timing table.", "",
              "| Dataset | Method | Fixed sort keys |", "| --- | --- | --- |"]
    for name in names:
        for algorithm in algorithms:
            case = _cases(records.get(name, {})).get(algorithm)
            if not _passed(case):
                continue
            keys = case.get("plan", {}).get("sort_keys", [])
            lines += [f"| {_code(name)} | {_code(algorithm)} | "
                      + (", ".join(_code(key) for key in keys) if keys else "(input order)") + " |"]

    lines += [
        "", "## Stage timings", "",
        "Seconds are single-run measurements, not replicated speedup estimates. Planning is separate from "
        "execution and writing. For sorted cases DuckDB's scan, sort, and output gathering are one combined "
        "upstream timer; a native Arrow permutation/gather decomposition is **unavailable**, not zero. "
        "For the no-sort baseline that same column measures Arrow input scanning. Schema restoration and "
        "PyArrow encoding/writing are timed separately. Verification is outside rewrite execution.", "",
        "A row marked `reuse X` inherits the already verified output's bytes and execution/verification stages "
        "from method X; only its planning time is newly measured. It was not independently written again. "
        "The rewrite-stage sum excludes sample reading, source fingerprinting, inventory, setup/cleanup, and verification.", "",
        "| Dataset | Method | Plan s | Scan/sort/gather s | Schema s | Write s | Rewrite-stage sum s | Verify s | Sort-input scans incl key sampling | Execution |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in names:
        for algorithm in algorithms:
            case = _cases(records.get(name, {})).get(algorithm)
            if not _passed(case):
                continue
            stage_names = ("planning_seconds", "scan_sort_gather_seconds", "schema_restore_seconds", "writing_seconds")
            stages = [case.get(key) for key in stage_names]
            total = sum(stages) if all(value is not None for value in stages) else None
            execution = "reuse " + _code(case["reused_full_result_of"]) if case.get("reused_full_result_of") else "measured"
            row = [_code(name), _code(algorithm), *[_number(value) for value in stages], _number(total),
                   _number(case.get("verification_seconds")), str(_sort_input_scans(case)), execution]
            lines += ["| " + " | ".join(row) + " |"]
    lines += ["", "### Range-boundary sampling", "",
              "The execution protocol changed during this checkpointed run to improve range balance. Earlier "
              "validated cases retain boundaries from their leading-row planning sample; newer cases use a "
              "deterministically seeded reservoir over all source rows, projecting only selected keys and "
              "stable filename/row-number identities. The latter adds **one narrow full-input scan**. "
              "Both protocols produce the same stable global row order for the same keys; only boundaries "
              "and execution cost differ. Old validated cases are not rerun. These mixed-version, single-pass "
              "timings are **not** a replicated speed comparison.", "",
              "For reservoir cases, boundary-sampling seconds below are already included in the combined "
              "scan/sort/gather timer above, not an additional charge. Legacy boundary construction was not "
              "separately timed (— means unrecorded). The scan count above includes both actual full-width sort "
              "ranges and any narrow key scan; it excludes sample reading and shared verification scans.", "",
              "| Dataset | Method | Boundary method | Key sample rows | Extra key-only scans | Full-width sort ranges | Boundary sampling s (included) |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        record = records.get(name, {})
        for algorithm in algorithms:
            case = _cases(record).get(algorithm)
            if not _passed(case):
                continue
            has_keys = bool(case.get("plan", {}).get("sort_keys"))
            sample_rows = case.get("key_sample_rows", record.get("planning_sample", {}).get("rows") if has_keys else 0)
            key_scans = case.get("key_sampling_source_scans", 0)
            ranges = case.get("full_width_source_scans_for_sort", case.get("sort_ranges", 0)) if has_keys else 0
            seconds = case.get("range_sampling_seconds", None if has_keys else 0.0)
            row = [_code(name), _code(algorithm), _code(_range_method(case)),
                   f"{sample_rows:,}" if sample_rows is not None else "—", str(key_scans), str(ranges), _number(seconds)]
            lines += ["| " + " | ".join(row) + " |"]
    lines += ["", "Shared per-dataset work is measured once, outside those method stage sums.", "",
              "| Dataset | Planning sample rows / files | Sample read s | Source fingerprint s |",
              "| --- | ---: | ---: | ---: |"]
    for name in names:
        record = records.get(name, {})
        sample = record.get("planning_sample", {})
        if not sample:
            continue
        lines += [f"| {_code(name)} | {sample.get('rows', 0):,} / {len(sample.get('files', [])):,} "
                  f"| {_number(record.get('sample_read_seconds'))} | {_number(record.get('source_fingerprint_seconds'))} |"]

    lines += [
        "", "## Method and correctness limits", "",
        f"Planning uses up to {config.get('sample_rows', 250_000):,} rows spread across up to "
        f"{config.get('sample_files', 32):,} files. All methods subsequently receive the same full input rows, "
        f"PyArrow writer, {config.get('row_group_size', 1_000_000):,}-row groups, dictionaries, and statistics.", "",
        "The footer codec is preserved for each dataset. ZSTD encoder levels are not recorded in the source "
        f"Parquet metadata: **ZSTD level {config.get('compression_level', 1)} is an explicit controlled setting**, "
        "not a claim about the original exporter level. Snappy remains Snappy and has no level parameter. "
        "Differences from downloaded source sizes also include writer/layout differences and do not isolate sorting.", "",
        f"Full-dataset natural ordering uses DuckDB with up to {config.get('sort_ranges', 16)} ordered, disjoint "
        "ranges. Legacy boundaries use the **entire sort-key tuple**; reservoir boundaries use that tuple "
        "plus stable filename/source-row identities, allowing heavily duplicated keys to be divided without "
        "changing tie order. Each range is globally sorted, and concatenating them retains global lexicographic "
        "order. Boundary duplicates can reduce the actual range count in legacy cases. The tables report "
        "actual full-width ranges and extra narrow key scans separately; both are included in the combined "
        "scan/sort/gather timer. Sorting can "
        f"spill locally (configured memory {config.get('memory_limit', '6GB')}, spill cap "
        f"{config.get('max_temp_size', '18GB')}, {config.get('threads', 4)} threads). "
        "The baseline streams input files in order using Arrow. This is not an Arrow-versus-DuckDB sort speed comparison.", "",
        "Every newly written output is checked for row count and the original physical Arrow schema, plus "
        "two independently seeded 64-bit row hashes aggregated with integer sums and XOR over **every row**. "
        "That is a probabilistic multiset check, not a mathematical proof. Sorted outputs additionally check "
        "all key rows for monotonicity, including batch boundaries. Stable tie order is enforced by filename "
        "and source row number in SQL and tested exactly on fixtures; it is not separately proven by a full "
        "output tie-identity scan. Source size/mtime snapshots are checked for changes during a measured case.", "",
        "Temporary benchmark outputs are deleted after verification. Identical validated orders can reuse "
        "one execution instead of repeating encoding; stored bytes/timings then describe that shared output, "
        "not a separately encoded per-algorithm metadata variation.", "",
        "**Scope distinction:** this benchmark sorts globally across each dataset's source files. The public "
        "`rewrite_dataset` API instead preserves each source file's relative partition path and file boundary "
        "and sorts within that file. Full-dataset gains reported here are not a guarantee of the same gains for a "
        "partition-preserving, per-file migration.", "",
        "These raw benchmarks deliberately measure candidates even when they grow the output. Production "
        "compression-only rewrites with preserved settings, no mandatory prefix, and `skip_unchanged=True` "
        "compare the actual candidate bytes with the original file before publication: an output that is not "
        "smaller is discarded and the original is copied or skipped. That per-file guard prevents a storage "
        "increase, but the planning/sorting/writing cost has already been paid. Explicit codec/level/layout "
        "changes or a mandatory prefix are honored even if they grow the file. The guard compares with the "
        "original file, not this benchmark's controlled no-sort rewrite.", "",
    ]
    quality_names = [name for name in names if records.get(name, {}).get("data_quality", {}).get("invalid_utf8_in_source_string")
                     or any(case.get("source_invalid_utf8_preserved") for case in records.get(name, {}).get("results", []))]
    if quality_names:
        lines += [
            "## Source-data quality: malformed UTF-8", "",
            "The source dataset(s) below contain byte sequences declared as Parquet/Arrow STRING that are not "
            "valid UTF-8. DuckDB's strict Parquet reader rejected them. The benchmark uses an explicitly "
            "**lossless, benchmark-only fallback**: Arrow exposes STRING data buffers as BINARY views where "
            "SQL execution is needed, then restores the original STRING buffer views and physical schema "
            "before writing. No-sort outputs use Arrow's Parquet reader directly; the table identifies the "
            "readers actually used for this corpus. "
            "It performs no text decoding, replacement, cleaning, or repair. **Malformed input bytes remain "
            "malformed byte-for-byte, and the retained logical STRING schema remains nonconforming.**", "",
            "Source and output all-row fingerprints use the same opaque-binary view mode so the comparison "
            "checks the original bytes rather than replacement text. An exact malformed-byte fixture checks "
            "the sorted output bytes, unchanged schema, and matching fingerprint mode. The corpus hash "
            "comparison remains probabilistic as described above.", "",
            "| Dataset | Fingerprint mode | Measured write-input readers | Original invalid bytes preserved | Failed strict attempt s (already included) |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for name in quality_names:
            record = records[name]
            valid_cases = [case for case in record.get("results", []) if _passed(case)]
            readers = sorted({case.get("input_reader", "unrecorded") for case in valid_cases})
            preserved = bool(valid_cases) and all(case.get("source_invalid_utf8_preserved") is True for case in valid_cases)
            row = [_code(name), _code(record.get("source_fingerprint_mode", "unrecorded")),
                   ", ".join(_code(reader) for reader in readers) if readers else "pending",
                   "yes; not repaired" if preserved else "pending",
                   _number(record.get("strict_utf8_failed_fingerprint_seconds"))]
            lines += ["| " + " | ".join(row) + " |"]
        lines += [
            "", "The failed strict-reader attempt is included in the shared source-fingerprint timer; it is not an extra charge.", "",
            "This fallback is **not silently enabled in production DuckDB rewrites**. The production Arrow-to-DuckDB "
            "sorting bridge fully validates Arrow batches (including nested/dictionary strings) and rejects "
            "malformed text before engine import instead of risking string replacement. The native Arrow path "
            "can preserve those bytes without cleaning, as covered by an exact fixture, but that does not "
            "make a malformed STRING column valid UTF-8. Prefer fixing the export schema to BINARY upstream "
            "when a field is genuinely arbitrary bytes.", "",
        ]
    if failed:
        lines += ["## Failures requiring attention", ""]
        for name in failed:
            error = records[name].get("error", {})
            message = (f"{error.get('type', 'Error')}: {error.get('message', 'method validation failed')}"
                       if error else "method validation failed")
            lines += [f"- {_code(name)}: {message.replace(chr(10), ' ')}"]
        lines += [""]
    if pending:
        lines += ["Pending/running datasets: " + ", ".join(_code(name) for name in pending) + ".", ""]

    benchmark_flags = [
        ("data-root", config.get("data_root", "data/source")),
        ("algorithms", ",".join(algorithms)),
        ("compression-level", config.get("compression_level", 1)),
        ("row-group-size", config.get("row_group_size", 1_000_000)),
        ("sample-rows", config.get("sample_rows", 250_000)),
        ("sample-files", config.get("sample_files", 32)),
        ("batch-size", config.get("batch_size", 262_144)),
        ("sort-ranges", config.get("sort_ranges", 16)),
        ("memory-limit", config.get("memory_limit", "6GB")),
        ("max-temp-size", config.get("max_temp_size", "18GB")),
        ("threads", config.get("threads", 4)),
        ("min-stable-age", config.get("min_stable_age", 120)),
    ]
    if config.get("temp_root"):
        benchmark_flags.append(("temp-root", config["temp_root"]))
    lines += ["## Reproduce", "",
              "Run from the repository root with the same stable source snapshot and locked environment. "
              "The runner checkpoints after each validated method and resumes matching snapshots/settings. "
              "Use a new `--output` if changing benchmark settings.", "", "```bash",
              "uv run --extra duckdb python -m benchmarks.full_corpus \\"]
    for index, (flag, value) in enumerate(benchmark_flags):
        continuation = " \\" if index < len(benchmark_flags) - 1 else ""
        lines += [f"  --{flag} {shlex.quote(str(value))}{continuation}"]
    lines += ["uv run python -m benchmarks.render_full_corpus", "```", "",
              "Generator: [render_full_corpus.py](render_full_corpus.py). Runner: [full_corpus.py](full_corpus.py).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = json.loads(args.input.read_text())
    link = os.path.relpath(args.input.resolve(), args.output.resolve().parent).replace(os.sep, "/")
    rendered = render(report, json_link=link)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
