"""Command-line interface for oparq."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .benchmark import benchmark
from .io import read_parquet, rewrite
from .planning import ALGORITHMS, plan_sort
from .learning import fit_dataset
from .models import RewritePlan, DatasetRewriteResult


def _columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def _level(value: str) -> int | str | None:
    if value == "preserve":
        return value
    if value in {"default", "none"}:
        return None
    return int(value)


def _compression_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--compression", default="preserve", help="codec or preserve (default)")
    parser.add_argument("--compression-level", type=_level, default="preserve",
                        help="integer, default, or preserve (requires known source level)")
    parser.add_argument("--compression-manifest", type=Path,
                        help="JSON source codec/level provenance")


def _common(parser: argparse.ArgumentParser, *, select_algorithm: bool = True) -> None:
    if select_algorithm:
        parser.add_argument(
            "--algorithm",
            choices=ALGORITHMS,
            default="auto",
            help="key-selection strategy (default: auto)",
        )
    parser.add_argument(
        "--prefix",
        type=_columns,
        default=[],
        metavar="COL,COL",
        help="fixed leading keys that oparq must preserve",
    )
    parser.add_argument(
        "--include",
        type=_columns,
        metavar="COL,COL",
        help="limit automatically selected candidates",
    )
    parser.add_argument(
        "--exclude",
        type=_columns,
        default=[],
        metavar="COL,COL",
        help="columns that must never become sort keys",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=250_000,
        help="rows used for column profiles; 0 profiles all rows exactly",
    )
    parser.add_argument(
        "--run-sample-rows",
        type=int,
        default=50_000,
        help="rows used by the weighted-run planner",
    )
    parser.add_argument(
        "--trial-sample-rows",
        type=int,
        default=250_000,
        help="rows used by codec-guided planning",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=8,
        help="maximum total sort keys (default: 8)",
    )
    parser.add_argument(
        "--fast-candidates",
        type=int,
        default=4,
        help="candidates tested by fast codec planning (default: 4)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=24,
        help="maximum codec evaluations during planning (default: 24)",
    )
    parser.add_argument(
        "--sort-backend",
        choices=("auto", "arrow", "rank"),
        default="auto",
        help="sorting implementation (default: auto)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="plan with every row instead of bounded samples (slow, exact)",
    )


def _plan_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "algorithm": getattr(args, "algorithm", "auto"),
        "prefix": args.prefix,
        "include": args.include,
        "exclude": args.exclude,
        "sample_rows": None if args.sample_rows == 0 else args.sample_rows,
        "run_sample_rows": args.run_sample_rows,
        "trial_sample_rows": args.trial_sample_rows,
        "max_sort_columns": args.max_keys,
        "fast_candidate_count": args.fast_candidates,
        "max_trial_evaluations": args.max_trials,
        "sort_backend": args.sort_backend,
        "full": args.full,
    }


def _comparison_settings(
    source: str, manifest: dict[str, object] | None, compression: object, level: object,
) -> tuple[dict[str, object], str | None]:
    """Writer settings for read-only comparisons; never refuse an unknown source.

    Every compared order, including no sorting, is encoded with the returned
    settings, so savings stay controlled. Only ``preserve`` placeholders are
    replaced: an unknown level by the codec default, then mixed or missing
    source codecs by ZSTD. Explicit values are never overridden.
    """

    from .storage import inspect_compression, resolve_compression

    if compression != "preserve" and level != "preserve":
        return {"compression": compression, "compression_level": level}, None
    profile = inspect_compression(source, manifest=manifest)
    default_level = None if level == "preserve" else level
    attempts = (
        (compression, level, None),
        (compression, default_level, "source compression level unknown; every compared "
                                     "order uses the codec default level"),
        ("zstd" if compression == "preserve" else compression, default_level,
         "source codecs differ or are unknown; every compared order uses ZSTD "
         "at its default level"),
    )
    error: ValueError | None = None
    for codec, codec_level, note in attempts:
        try:
            settings = resolve_compression(profile, compression=codec, compression_level=codec_level)
        except ValueError as failure:
            error = error or failure
            continue
        return settings.writer_options(), note
    raise error


def _describe(settings: dict[str, object]) -> str:
    import pyarrow as pa

    codec, level = settings["compression"], settings["compression_level"]
    if isinstance(codec, dict) or isinstance(level, dict):
        return f"per-column codecs {codec} levels {level}"
    if level is None and codec not in (None, "none") and pa.Codec.supports_compression_level(codec):
        level = pa.Codec.default_compression_level(codec)
    return f"{codec}" + (f" level {level}" if level is not None else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oparq",
        description="Find compression-friendly sort keys and write smaller Parquet files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="profile data and print selected keys")
    plan.add_argument("source")
    plan.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    _common(plan)
    _compression_options(plan)

    learn = subparsers.add_parser("fit", allow_abbrev=False,
                                 help="learn and save keys for future partition-preserving rewrites")
    learn.add_argument("source")
    learn.add_argument("destination")
    learn.add_argument("--algorithms", type=_columns, default=["codec_fast", "portfolio"])
    learn.add_argument("--sample-files", type=int, default=8)
    learn.add_argument("--overwrite", action="store_true")
    _common(learn, select_algorithm=False)
    _compression_options(learn)

    rewrite_parser = subparsers.add_parser(
        "rewrite", help="load, reorder, and rewrite Parquet"
    )
    rewrite_parser.add_argument("source")
    rewrite_parser.add_argument("destination")
    _compression_options(rewrite_parser)
    rewrite_parser.add_argument("--plan", help="saved RewritePlan JSON; skips planning")
    rewrite_parser.add_argument("--engine", choices=("arrow", "duckdb"), default="arrow")
    rewrite_parser.add_argument("--force-rewrite", action="store_true", help="re-encode even when no sort is selected")
    rewrite_parser.add_argument("--memory-limit", default="6GB")
    rewrite_parser.add_argument("--temp-directory")
    rewrite_parser.add_argument("--row-group-size", type=int)
    rewrite_parser.add_argument("--overwrite", action="store_true")
    rewrite_parser.add_argument(
        "--in-memory-sort", action="store_true",
        help="materialize the full reordered table instead of gathering row groups",
    )
    rewrite_parser.add_argument(
        "--gather-threads",
        type=_nonnegative_int,
        default=0,
        help="parallel column gathers: 0 auto, 1 serial, or a positive limit",
    )
    _common(rewrite_parser)

    bench = subparsers.add_parser(
        "benchmark", help="controlled same-writer algorithm comparison"
    )
    bench.add_argument("source")
    bench.add_argument(
        "--algorithms",
        type=_columns,
        default=[
            "none",
            "cardinality",
            "weighted",
            "entropy",
            "payload",
            "frequency",
            "portfolio",
            "codec_fast",
            "codec",
            "auto",
        ],
    )
    bench.add_argument("--rows", type=int, default=0, help="0 uses all rows")
    _compression_options(bench)
    bench.add_argument("--row-group-size", type=int)
    bench.add_argument("--sample-rows", type=int, default=250_000)
    bench.add_argument("--run-sample-rows", type=int, default=50_000)
    bench.add_argument("--trial-sample-rows", type=int, default=250_000)
    bench.add_argument("--max-keys", type=int, default=8)
    bench.add_argument("--fast-candidates", type=int, default=4)
    bench.add_argument("--max-trials", type=int, default=24)
    bench.add_argument("--in-memory-sort", action="store_true")
    bench.add_argument(
        "--gather-threads",
        type=_nonnegative_int,
        default=0,
        help="parallel column gathers: 0 auto, 1 serial, or a positive limit",
    )
    bench.add_argument(
        "--sort-backend",
        choices=("auto", "arrow", "rank"),
        default="auto",
    )
    bench.add_argument(
        "--full",
        action="store_true",
        help="plan with every row instead of bounded samples (slow, exact)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _run(parser, args)
    except (OSError, ValueError, TypeError, ImportError) as error:
        # Missing files, refused overwrites, unknown source levels and lossy
        # engine conversions are user-actionable: report them without a traceback.
        parser.exit(1, f"{parser.prog}: error: {error}\n")


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    manifest = None
    if args.compression_manifest:
        manifest = json.loads(args.compression_manifest.read_text())

    if args.command == "fit":
        options = _plan_options(args)
        options.pop("algorithm")
        rows = options.pop("sample_rows")
        if rows is None:
            parser.error("fit requires a bounded positive --sample-rows")
        result = fit_dataset(
            args.source, algorithms=args.algorithms, sample_rows=rows,
            sample_files=args.sample_files, compression=args.compression,
            compression_level=args.compression_level, compression_manifest=manifest,
            **options,
        )
        result.save(args.destination, overwrite=args.overwrite)
        print(f"saved: {args.destination}")
        print(f"sort keys: {', '.join(result.sort_keys) or 'keep input order'}")
        return

    if args.command == "plan":
        settings, note = _comparison_settings(args.source, manifest, args.compression,
                                              args.compression_level)
        table = read_parquet(args.source)
        result = plan_sort(table, trial_compression=settings["compression"],
                           trial_compression_level=settings["compression_level"],
                           **_plan_options(args))
        if args.json:
            trial = {**settings, "note": note}
            print(json.dumps({**result.as_dict(), "trial_compression": trial}, indent=2))
            return
        print(result.explain())
        print(f"trial codec: {_describe(settings)}" + (f" ({note})" if note else ""))
        print("\nselected column profiles:")
        for name in result.sort_keys:
            profile = result.profile(name)
            print(
                f"  {name:<30} bytes={_bytes(profile.byte_size):>10} "
                f"sample_ndv={profile.sample_distinct:<10,} "
                f"repeat={profile.repeatability:>6.1%}"
            )
        return

    if args.command == "rewrite":
        options = _plan_options(args) if args.plan is None else {"plan": RewritePlan.load(args.plan)}
        # Execution controls have no effect on a byte-copy decision.
        if args.gather_threads:
            options["gather_threads"] = args.gather_threads
        if args.in_memory_sort:
            options["stream_sort"] = False
        result = rewrite(
            args.source,
            args.destination,
            **options,
            compression=args.compression,
            compression_level=args.compression_level,
            compression_manifest=manifest,
            row_group_size=args.row_group_size,
            engine=args.engine, memory_limit=args.memory_limit,
            temp_directory=args.temp_directory, skip_unchanged=not args.force_rewrite,
            overwrite=args.overwrite,
        )
        if isinstance(result, DatasetRewriteResult):
            print(f"files: {len(result.files)}; rewritten={result.rewritten_files} "
                  f"copied={result.copied_files} skipped={result.skipped_files}")
            print(f"output: {result.destination}")
            return
        print(result.plan.explain())
        print(f"output: {result.path}")
        print(f"action: {result.action}")
        print(f"size: {_bytes(result.file_size)}")
        print(f"plan: {result.planning_seconds:.2f}s")
        if result.permutation_seconds is not None:
            print(f"permutation: {result.permutation_seconds:.2f}s")
        if result.gathering_seconds is not None:
            print(f"gather: {result.gathering_seconds:.2f}s")
        print(f"sort: {result.sort_seconds:.2f}s")
        print(f"write: {result.write_seconds:.2f}s")
        return

    if args.command == "benchmark":
        writer_settings, note = _comparison_settings(args.source, manifest, args.compression,
                                                     args.compression_level)
        table = read_parquet(args.source)
        if args.rows:
            if args.rows < 0:
                parser.error("--rows must be non-negative")
            table = table.slice(0, args.rows)
        results = benchmark(
            table,
            algorithms=args.algorithms,
            **writer_settings,
            row_group_size=args.row_group_size,
            sample_rows=None if args.sample_rows == 0 else args.sample_rows,
            run_sample_rows=args.run_sample_rows,
            trial_sample_rows=args.trial_sample_rows,
            max_sort_columns=args.max_keys,
            fast_candidate_count=args.fast_candidates,
            max_trial_evaluations=args.max_trials,
            sort_backend=args.sort_backend,
            full=args.full,
            gather_threads=args.gather_threads,
            stream_sort=not args.in_memory_sort,
        )
        print(f"rows: {table.num_rows:,}")
        print(f"codec: {_describe(writer_settings)}" + (f" ({note})" if note else ""))
        for result in results:
            keys = ",".join(result.sort_keys) or "input order"
            detailed_sort = (
                f"perm={result.permutation_seconds:6.2f}s "
                f"gather={result.gathering_seconds:6.2f}s "
                if result.permutation_seconds is not None
                and result.gathering_seconds is not None
                else ""
            )
            print(
                f"{result.algorithm:<12} {_bytes(result.size_bytes):>10} "
                f"{result.savings_fraction:+7.1%}  "
                f"plan={result.planning_seconds:6.2f}s "
                f"{detailed_sort}"
                f"sort={result.sort_seconds:6.2f}s "
                f"write={result.write_seconds:6.2f}s  {keys}"
            )
        return

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
