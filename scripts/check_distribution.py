"""Audit release artifacts and smoke-test an installed wheel.

Only the Python standard library is needed for the artifact audit. The smoke
mode must run in an isolated environment containing an installed oparq wheel,
not the checkout's editable installation. This script never publishes files.
"""

from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
from email.parser import BytesParser
import fnmatch
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


class DistributionError(ValueError):
    """A release artifact does not match the intended package."""


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DistributionError(message)


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts and "\\" not in value,
             f"unsafe archive path: {value}")
    return path


def _dependency(value: str) -> str:
    """Normalize formatting without imposing a new packaging dependency."""

    value = re.sub(r"\s+", "", value).replace('"', "'").lower()
    requirement, separator, marker = value.partition(";")
    match = re.match(r"([^<>=!~]+)(.*)", requirement)
    if match:
        requirement = _name(match[1]) + ",".join(sorted(match[2].split(",")))
    return requirement + (separator + marker if separator else "")


def project_settings(root: Path) -> dict:
    with (root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _metadata(raw: bytes, settings: dict, label: str) -> None:
    metadata = BytesParser().parsebytes(raw)
    for key, expected in (("Name", settings["name"]), ("Version", settings["version"]),
                          ("Requires-Python", settings["requires-python"]),
                          ("Summary", settings["description"])):
        _require(metadata.get(key) == expected, f"{label}: unexpected {key}: {metadata.get(key)!r}")
    expected = {_dependency(value) for value in settings.get("dependencies", ())}
    for extra, requirements in settings.get("optional-dependencies", {}).items():
        expected.update(_dependency(f"{value}; extra == '{extra}'") for value in requirements)
    actual = {_dependency(value) for value in metadata.get_all("Requires-Dist", ())}
    _require(actual == expected, f"{label}: dependency metadata differs: {actual} != {expected}")
    _require(set(metadata.get_all("Provides-Extra", ())) == set(settings.get("optional-dependencies", {})),
             f"{label}: optional extras differ from pyproject.toml")
    _require(metadata.get("Description-Content-Type") == "text/markdown",
             f"{label}: README is not marked as Markdown")
    if isinstance(settings.get("license"), str):
        _require(metadata.get("License-Expression") == settings["license"],
                 f"{label}: SPDX license metadata differs from pyproject.toml")


def _package_sources(root: Path) -> dict[str, bytes]:
    directory = root / "src" / "oparq"
    sources = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        _require(not path.is_symlink(), f"package sources must not be symlinks: {path}")
        relative = path.relative_to(directory).as_posix()
        _require(path.suffix in {".py", ".pyi"} or relative == "py.typed",
                 f"unexpected package source asset: {relative}")
        sources[relative] = path.read_bytes()
    _require("__init__.py" in sources and "__main__.py" in sources and "py.typed" in sources,
             "package sources must contain __init__.py, __main__.py and py.typed")
    versions = [node.value.value for node in ast.parse(sources["__init__.py"]).body
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    _require(versions == [project_settings(root)["version"]], "public __version__ differs from project metadata")
    return sources


def check_wheel(path: Path, root: Path) -> dict:
    settings = project_settings(root)
    sources = _package_sources(root)
    normalized = settings["name"].replace("-", "_")
    dist_info = f"{normalized}-{settings['version']}.dist-info"
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        _require(len(entries) == len(set(entries)), f"{path}: duplicate archive members")
        files = {value for value in entries if not value.endswith("/")}
        for value in entries:
            parts = _safe_path(value).parts
            _require(parts and parts[0] in {"oparq", dist_info},
                     f"{path}: unexpected wheel member: {value}")
            _require("__pycache__" not in parts and not value.endswith(".pyc"),
                     f"{path}: generated Python cache in wheel: {value}")
        expected = {f"oparq/{value}" for value in sources}
        allowed_dirs = {"oparq", dist_info, dist_info + "/licenses"}
        allowed_dirs.update(parent.as_posix() for value in expected
                            for parent in PurePosixPath(value).parents if parent.as_posix() != ".")
        _require(all(value.rstrip("/") in allowed_dirs for value in entries if value.endswith("/")),
                 f"{path}: unexpected wheel directory")
        actual = {value for value in files if value.startswith("oparq/")}
        _require(actual == expected, f"{path}: package modules missing or unexpectedly included: {actual ^ expected}")
        for relative, raw in sources.items():
            _require(archive.read(f"oparq/{relative}") == raw,
                     f"{path}: stale or changed package source: {relative}")
        required = {f"{dist_info}/{value}" for value in ("METADATA", "WHEEL", "entry_points.txt", "RECORD")}
        _require(required <= files, f"{path}: required wheel metadata is missing: {required - files}")
        _metadata(archive.read(f"{dist_info}/METADATA"), settings, str(path))
        allowed_metadata = required | {f"{dist_info}/licenses/{value}" for value in settings.get("license-files", ())}
        actual_metadata = {value for value in files if value.startswith(dist_info + "/")}
        _require(actual_metadata <= allowed_metadata, f"{path}: unexpected wheel metadata assets: {actual_metadata - allowed_metadata}")
        for filename in settings.get("license-files", ()):
            licensed = f"{dist_info}/licenses/{filename}"
            _require(licensed in files and archive.read(licensed) == (root / filename).read_bytes(),
                     f"{path}: license file missing or differs: {filename}")
        description = BytesParser().parsebytes(archive.read(f"{dist_info}/METADATA")).get_payload(decode=True)
        _require(description.rstrip(b"\n") == (root / "README.md").read_bytes().rstrip(b"\n"), f"{path}: stale wheel README")
        wheel = BytesParser().parsebytes(archive.read(f"{dist_info}/WHEEL"))
        _require(wheel.get("Root-Is-Purelib") == "true", f"{path}: expected pure-Python wheel")
        _require(wheel.get("Tag") == "py3-none-any", f"{path}: unexpected wheel compatibility tag")
        points = configparser.ConfigParser()
        points.read_string(archive.read(f"{dist_info}/entry_points.txt").decode())
        _require(dict(points["console_scripts"]) == settings.get("scripts", {}),
                 f"{path}: console entry points differ from pyproject.toml")
        rows = list(csv.reader(io.StringIO(archive.read(f"{dist_info}/RECORD").decode())))
        _require(len(rows) == len(files) and {row[0] for row in rows} == files,
                 f"{path}: RECORD does not enumerate every wheel file exactly once")
        for filename, digest, size in rows:
            if filename == f"{dist_info}/RECORD":
                _require(not digest and not size, f"{path}: RECORD must not hash itself")
                continue
            raw = archive.read(filename)
            calculated = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
            _require(digest == f"sha256={calculated}" and size == str(len(raw)),
                     f"{path}: invalid RECORD hash/size: {filename}")
    return {"artifact": str(path), "kind": "wheel", "files": len(files), "bytes": path.stat().st_size}


def check_sdist(path: Path, root: Path) -> dict:
    settings = project_settings(root)
    sources = _package_sources(root)
    prefix = f"{settings['name'].replace('-', '_')}-{settings['version']}"
    allowed = {"PKG-INFO", "pyproject.toml", "pyproject.toml.orig", "README.md"}
    allowed.update(item.name for item in root.iterdir()
                   if item.is_file() and (item.name.startswith("LICENSE") or item.name.startswith("NOTICE")))
    expected_sources = {f"src/oparq/{value}" for value in sources}
    curated = {"CHANGELOG.md", "CONTRIBUTING.md", "uv.lock", "scripts/check_distribution.py"}
    curated.update(item.relative_to(root).as_posix() for item in (root / "docs").rglob("*")
                   if item.is_file() and "__pycache__" not in item.parts
                   and item.suffix not in {".pyc", ".pyo"})
    curated = {value for value in curated if (root / value).is_file()}
    # Only declared curated files are required; other top-level directories
    # (research clones, data, benchmark output, site/dist) remain forbidden.
    with (root / "pyproject.toml").open("rb") as stream:
        includes = tomllib.load(stream).get("tool", {}).get("uv", {}).get("build-backend", {}).get("source-include", ())
    included = {value for value in curated if any(fnmatch.fnmatchcase(value, pattern) for pattern in includes)}
    allowed |= included
    allowed_dirs = {"."}
    allowed_dirs.update(parent.as_posix() for value in allowed | expected_sources
                        for parent in PurePosixPath(value).parents)
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        _require(len(members) == len({member.name for member in members}),
                 f"{path}: duplicate archive members")
        files = {}
        for member in members:
            parts = _safe_path(member.name).parts
            _require(parts and parts[0] == prefix, f"{path}: unexpected sdist root: {member.name}")
            _require(member.isdir() or member.isfile(), f"{path}: links/special files are not allowed: {member.name}")
            relative = PurePosixPath(*parts[1:]).as_posix()
            if member.isdir():
                _require(relative in allowed_dirs, f"{path}: unexpected sdist directory: {relative}")
                continue
            _require(relative in allowed or relative in expected_sources,
                     f"{path}: unexpected sdist member: {relative}")
            stream = archive.extractfile(member)
            _require(stream is not None, f"{path}: cannot read sdist member: {relative}")
            files[relative] = stream.read()
        required = expected_sources | included | {"PKG-INFO", "pyproject.toml", "README.md"}
        _require(required <= files.keys(), f"{path}: required source files missing: {required - files.keys()}")
        for relative, raw in sources.items():
            _require(files[f"src/oparq/{relative}"] == raw, f"{path}: stale package source: {relative}")
        _require(files["README.md"] == (root / "README.md").read_bytes(), f"{path}: stale README")
        for filename in included | set(settings.get("license-files", ())):
            _require(filename in files and files[filename] == (root / filename).read_bytes(),
                     f"{path}: curated source file missing or stale: {filename}")
        built_settings = tomllib.loads(files["pyproject.toml"].decode())["project"]
        _require(built_settings == settings, f"{path}: project metadata differs from checkout")
        _metadata(files["PKG-INFO"], settings, str(path))
    return {"artifact": str(path), "kind": "sdist", "files": len(files), "bytes": path.stat().st_size}


def check_distribution(paths: list[Path], root: Path) -> list[dict]:
    kinds = {"wheel" if path.name.endswith(".whl") else "sdist" if path.name.endswith(".tar.gz") else "unknown"
             for path in paths}
    _require(kinds == {"wheel", "sdist"}, "audit requires both a wheel and a source distribution")
    return [check_wheel(path, root) if path.name.endswith(".whl") else check_sdist(path, root) for path in paths]


def smoke_installed(root: Path, *, with_duckdb: bool = False) -> dict:
    import oparq
    import pyarrow as pa
    import pyarrow.parquet as pq

    settings = project_settings(root)
    package = Path(oparq.__file__).resolve()
    _require(not package.is_relative_to(root.resolve()), "smoke must use an installed wheel outside the checkout")
    _require(oparq.__version__ == settings["version"] == importlib.metadata.version("oparq"),
             "installed package and project versions differ")
    if with_duckdb:
        from oparq.engines import require_duckdb
        require_duckdb()
    else:
        _require(importlib.util.find_spec("duckdb") is None, "base-wheel smoke environment must not contain DuckDB")
    table = pa.table({"key": [2, 1, None, 1], "value": ["a", "b", "c", "d"]})
    with tempfile.TemporaryDirectory(prefix="oparq-wheel-smoke-") as directory:
        temporary = Path(directory)
        source = temporary / "source" / "day=1" / "part.parquet"
        original = oparq.write(table, source, algorithm="none", compression_level=1)
        plan = oparq.fit(table, algorithms=("weighted",), prefix=("key",), compression_level=1)
        saved = temporary / "plan.json"
        plan.save(saved)
        loaded = oparq.RewritePlan.load(saved)
        destination = temporary / "destination"
        result = oparq.rewrite_dataset(source.parent.parent, destination, plan=loaded,
                                       engine="duckdb" if with_duckdb else "arrow")
        actual = pq.read_table(destination / "day=1" / "part.parquet")
        _require(actual["key"].to_pylist() == [1, 1, 2, None], "wheel rewrite did not sort the required prefix")
        _require(actual["value"].to_pylist() == ["b", "d", "a", "c"], "wheel rewrite changed values or tie order")
        _require(result.rewritten_files == 1 and original.file_size > 0, "wheel rewrite result is incomplete")
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        commands = [[sys.executable, "-I", "-m", "oparq", "--help"],
                    [str(Path(sys.executable).parent / "oparq"), "--help"]]
        for command in commands:
            completed = subprocess.run(command, cwd=temporary, env=environment,
                                       capture_output=True, text=True, check=True)
            _require("rewrite" in completed.stdout and "fit" in completed.stdout,
                     "installed CLI entry point did not expose the expected commands")
    return {"smoke": "passed", "version": oparq.__version__, "package": str(package),
            "duckdb": with_duckdb, "python": sys.version.split()[0]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="*")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--smoke", action="store_true", help="test an installed wheel in a clean environment")
    parser.add_argument("--with-duckdb", action="store_true", help="include the optional DuckDB path in smoke mode")
    args = parser.parse_args(argv)
    if args.smoke:
        if args.artifacts:
            parser.error("--smoke does not accept artifact paths")
        result = smoke_installed(args.project_root, with_duckdb=args.with_duckdb)
    else:
        if args.with_duckdb:
            parser.error("--with-duckdb requires --smoke")
        settings = project_settings(args.project_root)
        stem = f"{settings['name'].replace('-', '_')}-{settings['version']}"
        paths = args.artifacts or [args.project_root / "dist" / (stem + "-py3-none-any.whl"),
                                   args.project_root / "dist" / (stem + ".tar.gz")]
        result = check_distribution(paths, args.project_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
