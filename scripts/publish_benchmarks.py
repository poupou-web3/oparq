#!/usr/bin/env python3
"""Publish only an audited prepared results bundle, with explicit public consent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq

if __package__:
    from .prepare_benchmarks import CHECKSUM_FILE, RESULT_SCHEMA, sha256_file
else:
    from prepare_benchmarks import CHECKSUM_FILE, RESULT_SCHEMA, sha256_file


def verify_bundle(bundle: Path) -> list[str]:
    bundle = bundle.resolve()
    marker = json.loads((bundle / "bundle-manifest.json").read_text())
    if marker.get("kind") != "oparq_results_only" or marker.get("source_rows_included") is not False:
        raise ValueError("not an audited results-only bundle")
    manifest = json.loads((bundle / CHECKSUM_FILE).read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported checksum manifest")
    listed = set(manifest["files"])
    actual = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlinks are forbidden in publication bundles")
        if path.is_file():
            actual.add(path.relative_to(bundle).as_posix())
    if actual != listed | {CHECKSUM_FILE}:
        raise ValueError("bundle contains missing or unlisted files")
    allowed_roots = {"results", "summaries", "tables", "reproduction"}
    allowed_root_files = {"README.md", "REPRODUCE.md", "input-manifest.json", "source-provenance.json",
                          "environment.json", "bundle-manifest.json"}
    for relative in listed:
        parts = Path(relative).parts
        if Path(relative).is_absolute() or ".." in parts:
            raise ValueError("unsafe manifest path")
        if (len(parts) == 1 and relative not in allowed_root_files) or (len(parts) > 1 and parts[0] not in allowed_roots):
            raise ValueError(f"unexpected publication path: {relative}")
        if "data" in parts or ".venv" in parts or ".git" in parts:
            raise ValueError("source data or environment folders are forbidden")
        path = bundle / relative
        expected = manifest["files"][relative]
        if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"bundle checksum mismatch: {relative}")
        if path.suffix == ".parquet":
            if relative not in {"tables/full_corpus.parquet", "tables/saved_plan_engines.parquet"}:
                raise ValueError("only normalized measurement Parquet tables may be uploaded")
            if not pq.read_schema(path).equals(RESULT_SCHEMA, check_metadata=False):
                raise ValueError("Parquet table is not the normalized result schema")
    return sorted(actual)


def publish_bundle(bundle: Path, repo_id: str, *, confirm_public: bool = False,
                   dry_run: bool = False, update_existing: bool = False,
                   api: Any = None) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("repo_id must explicitly name owner/dataset")
    files = verify_bundle(bundle)
    if dry_run:
        return {"repo_id": repo_id, "visibility": "public", "files": len(files), "published": False}
    if not confirm_public:
        raise ValueError("public publication requires --confirm-public")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()  # Uses existing login/HF_TOKEN; never print credentials.
    api.whoami()  # An unauthenticated 404 does not establish repo absence.
    parent_commit = None
    # Never change an existing private repository's visibility implicitly.
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as error:
        if getattr(getattr(error, "response", None), "status_code", None) != 404:
            raise
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=False)
    else:
        if info.private is not False:
            raise ValueError("target dataset is private; do not change its visibility implicitly")
        parent_commit = getattr(info, "sha", None)
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset")) - {".gitattributes"}
        if remote_files:
            if not update_existing:
                raise ValueError("target is nonempty; explicitly authorize --update-existing for this benchmark repository")
            if "bundle-manifest.json" not in remote_files:
                raise ValueError("refusing to overwrite an unrelated dataset repository")
            remote_marker = json.loads(Path(api.hf_hub_download(
                repo_id=repo_id, filename="bundle-manifest.json", repo_type="dataset",
            )).read_text())
            if remote_marker.get("kind") != "oparq_results_only":
                raise ValueError("refusing to overwrite an unrelated dataset repository")
    commit = api.upload_folder(folder_path=str(bundle.resolve()), repo_id=repo_id,
                               repo_type="dataset", allow_patterns=files,
                               parent_commit=parent_commit,
                               commit_message="Publish audited oparq results-only benchmark bundle")
    return {"repo_id": repo_id, "visibility": "public", "files": len(files), "published": True,
            "commit_url": getattr(commit, "commit_url", str(commit))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--confirm-public", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-existing", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(publish_bundle(args.bundle, args.repo_id, confirm_public=args.confirm_public,
                                    dry_run=args.dry_run, update_existing=args.update_existing), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
