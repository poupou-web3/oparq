"""Sort-key planning algorithms."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .sorting import SORT_BACKENDS, key_table, sort_key_indices, unreorderable_columns
from .sorting import sort_indices as native_sort_indices
from .storage import effective_compression_level

from .models import ColumnProfile, SortPlan
from .profile import (
    DEFAULT_SAMPLE_ROWS,
    as_table,
    decode_dictionary,
    profile_table,
    sample_table,
    validate_unique_names,
)


ALGORITHMS = (
    "auto", "all", "portfolio", "codec_fast", "codec", "runs", "weighted",
    "cardinality", "entropy", "payload", "frequency", "none",
)
def _normalize_algorithm(name: str) -> str:
    normalized = name.lower()
    if normalized not in ALGORITHMS:
        choices = ", ".join(ALGORITHMS)
        raise ValueError(f"unknown algorithm {name!r}; choose one of: {choices}")
    return normalized


def _candidate_profiles(
    profiles: Sequence[ColumnProfile],
    *,
    include: set[str] | None,
    prefix: set[str],
) -> list[ColumnProfile]:
    candidates = []
    for profile in profiles:
        if profile.name in prefix or not profile.eligible:
            continue
        if include is not None and profile.name not in include:
            continue
        if profile.sample_distinct <= 1:
            continue
        # A mostly-null or almost-constant column can technically produce a
        # run, but moving <0.01% of rows is noise and often adds sort work.
        if 1.0 - profile.dominant_fraction < 0.0001:
            continue
        if profile.repeatability <= 0.0001 or profile.potential_bytes <= 0:
            continue
        candidates.append(profile)
    return candidates


def _cardinality_order(
    candidates: Sequence[ColumnProfile], slots: int
) -> list[str]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.sample_distinct,
            -item.potential_bytes,
            item.name,
        ),
    )
    return [item.name for item in ordered[:slots]]


def _weighted_order(candidates: Sequence[ColumnProfile], slots: int) -> list[str]:
    # Select byte-heavy, repeatable, non-degenerate columns first, then apply
    # the proven ascending-cardinality lexicographic ordering to that subset.
    selected = sorted(
        candidates,
        key=lambda item: (-item.potential_bytes, item.sample_distinct, item.name),
    )[:slots]
    return _cardinality_order(selected, slots)


def sort_key_table(table: pa.Table, keys: Sequence[str] | None = None) -> pa.Table:
    """Return scalar key columns, decoding dictionaries for Arrow table sort."""

    names = list(keys) if keys is not None else table.column_names
    arrays = [decode_dictionary(table[name]) for name in names]
    return pa.table(arrays, names=names)


def sort_indices(
    table: pa.Table,
    keys: Sequence[str],
    *,
    null_placement: str = "at_end",
    sort_backend: str = "auto",
) -> pa.Array:
    return native_sort_indices(
        table, keys, null_placement=null_placement, sort_backend=sort_backend
    )


def _same_count(column: pa.ChunkedArray) -> int:
    if len(column) <= 1:
        return 0
    left = column.slice(0, len(column) - 1)
    right = column.slice(1)
    equal = pc.fill_null(pc.equal(left, right), False)
    both_null = pc.and_(pc.is_null(left), pc.is_null(right))
    same = pc.or_(equal, both_null)
    if pa.types.is_floating(column.type):
        both_nan = pc.and_(
            pc.fill_null(pc.is_nan(left), False),
            pc.fill_null(pc.is_nan(right), False),
        )
        same = pc.or_(same, both_nan)
    return int(pc.sum(pc.cast(same, pa.int64())).as_py())


def _weighted_run_cost(
    table: pa.Table,
    keys: Sequence[str],
    targets: Sequence[ColumnProfile],
    *,
    null_placement: str,
) -> float:
    if table.num_rows <= 1:
        return 0.0
    permutation = (
        sort_indices(table, keys, null_placement=null_placement) if keys else None
    )
    cost = 0.0
    for profile in targets:
        column = decode_dictionary(table[profile.name])
        if permutation is not None:
            column = pc.take(column, permutation)
        runs = table.num_rows - _same_count(column)
        cost += profile.order_sensitive_bytes * runs / table.num_rows
    return cost


def _runs_order(
    table: pa.Table,
    candidates: Sequence[ColumnProfile],
    profiles: Sequence[ColumnProfile],
    *,
    prefix: Sequence[str],
    slots: int,
    run_sample_rows: int,
    candidate_pool_size: int,
    min_run_improvement: float,
    null_placement: str,
) -> tuple[list[str], float, float]:
    planning_sample = sample_table(table, min(run_sample_rows, table.num_rows))
    if planning_sample.num_rows <= 1 or slots <= 0 or not candidates:
        base = _weighted_run_cost(
            planning_sample, prefix, profiles, null_placement=null_placement
        )
        return [], base, base

    pool = sorted(
        candidates,
        key=lambda item: (-item.potential_bytes, item.sample_distinct, item.name),
    )[:candidate_pool_size]
    targets = [
        item
        for item in profiles
        if item.eligible and item.sample_distinct > 1 and item.byte_size > 0
    ]
    # Run costs are evaluated repeatedly; cap very wide schemas at the columns
    # representing nearly all bytes so planning remains quick.
    targets.sort(key=lambda item: (-item.order_sensitive_bytes, item.name))
    total_bytes = sum(item.order_sensitive_bytes for item in targets)
    kept: list[ColumnProfile] = []
    covered = 0
    for item in targets:
        kept.append(item)
        covered += item.order_sensitive_bytes
        if len(kept) >= 32 or (total_bytes and covered / total_bytes >= 0.95):
            break
    targets = kept

    cache: dict[tuple[str, ...], float] = {}

    def cost(keys: Sequence[str]) -> float:
        cache_key = tuple(keys)
        if cache_key not in cache:
            cache[cache_key] = _weighted_run_cost(
                planning_sample,
                cache_key,
                targets,
                null_placement=null_placement,
            )
        return cache[cache_key]

    chosen: list[str] = []
    current = cost(prefix)
    baseline = current
    remaining = {item.name for item in pool}
    while remaining and len(chosen) < slots:
        trials: list[tuple[float, str]] = []
        for name in sorted(remaining):
            try:
                trials.append((cost([*prefix, *chosen, name]), name))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                continue
        if not trials:
            break
        best_cost, best_name = min(trials)
        improvement = (current - best_cost) / current if current else 0.0
        if improvement < min_run_improvement:
            break
        chosen.append(best_name)
        remaining.remove(best_name)
        current = best_cost
    return chosen, baseline, current


def _encoded_size(
    table: pa.Table,
    keys: Sequence[str],
    *,
    compression: str | Mapping[str, str] | None,
    compression_level: int | Mapping[str, int] | None,
    null_placement: str,
    keys_table: pa.Table | None = None,
    permutation: pa.Array | None = None,
) -> int:
    if permutation is None:
        permutation = (
            sort_key_indices(keys_table, keys, null_placement=null_placement)
            if keys and keys_table is not None
            else sort_indices(table, keys, null_placement=null_placement) if keys else None
        )
    ordered = table.take(permutation) if permutation is not None else table
    sink = pa.BufferOutputStream()
    effective_level = effective_compression_level(compression, compression_level)
    pq.write_table(
        ordered,
        sink,
        compression=dict(compression) if isinstance(compression, Mapping) else compression,
        compression_level=effective_level,
        use_dictionary=True,
        write_statistics=False,
        row_group_size=max(1, ordered.num_rows),
    )
    return sink.getvalue().size


class _TrialEncoder:
    """Measure candidate orders as Parquet bytes on one deterministic sample.

    Key columns are ranked lazily, once each, and sizes are cached per order,
    so searches sharing an encoder never encode the same order twice. Every
    attempted encode, including a failed one, counts as an evaluation.
    """

    def __init__(
        self,
        table: pa.Table,
        *,
        trial_sample_rows: int,
        compression: str | Mapping[str, str] | None,
        compression_level: int | Mapping[str, int] | None,
        null_placement: str,
        sort_backend: str,
    ) -> None:
        self.sample = sample_table(table, min(trial_sample_rows, table.num_rows))
        self.evaluations = 0
        self._options = dict(
            compression=compression,
            compression_level=compression_level,
            null_placement=null_placement,
        )
        self._sort_backend = sort_backend
        self._keys: dict[str, pa.ChunkedArray] = {}
        self._sizes: dict[tuple[tuple[str, ...], int | None], int] = {}

    def _keys_table(self, keys: Sequence[str]) -> pa.Table:
        missing = [name for name in keys if name not in self._keys]
        if missing:
            ranked = key_table(
                self.sample,
                missing,
                sort_backend=self._sort_backend,
                null_placement=self._options["null_placement"],
            )
            self._keys.update(zip(missing, ranked.columns))
        return pa.table([self._keys[name] for name in keys], names=list(keys))

    def size(self, keys: Sequence[str], *, frequency_from: int | None = None) -> int:
        """Encode natural order, or frequency-rank the keys after a natural prefix."""

        cache_key = (tuple(keys), frequency_from)
        if cache_key not in self._sizes:
            self.evaluations += 1
            if frequency_from is not None:
                from .strategies import frequency_sort_indices

                permutation = frequency_sort_indices(
                    self.sample,
                    keys[frequency_from:],
                    prefix_keys=keys[:frequency_from],
                    null_placement=self._options["null_placement"],
                )
                size = _encoded_size(
                    self.sample, keys, permutation=permutation, **self._options
                )
            else:
                size = _encoded_size(
                    self.sample,
                    keys,
                    keys_table=self._keys_table(keys) if keys else None,
                    **self._options,
                )
            self._sizes[cache_key] = size
        return self._sizes[cache_key]


def _codec_order(
    table: pa.Table,
    candidates: Sequence[ColumnProfile],
    *,
    prefix: Sequence[str],
    slots: int,
    trial_sample_rows: int,
    candidate_pool_size: int,
    min_trial_improvement: float,
    compression: str | Mapping[str, str] | None,
    compression_level: int | Mapping[str, int] | None,
    null_placement: str,
    fast_candidate_count: int | None = None,
    max_trial_evaluations: int | None = None,
    sort_backend: str = "auto",
    encoder: _TrialEncoder | None = None,
) -> tuple[list[str], float, float, int]:
    """Greedy key search scored by real Parquet bytes on a bounded sample.

    The budget counts this search's own encodes; orders already measured by
    a shared ``encoder`` are reused without counting again.
    """

    if encoder is None:
        encoder = _TrialEncoder(
            table,
            trial_sample_rows=trial_sample_rows,
            compression=compression,
            compression_level=compression_level,
            null_placement=null_placement,
            sort_backend=sort_backend,
        )
    started = encoder.evaluations
    if encoder.sample.num_rows <= 1 or slots <= 0 or not candidates:
        size = float(encoder.size(prefix))
        return [], size, size, encoder.evaluations - started

    pool = sorted(
        candidates,
        key=lambda item: (-item.potential_bytes, item.sample_distinct, item.name),
    )[:candidate_pool_size]
    candidate_names = [item.name for item in pool]

    chosen: list[str] = []
    current = encoder.size(prefix)
    baseline = current
    remaining = {item.name for item in pool}
    while remaining and len(chosen) < slots:
        trials: list[tuple[int, str]] = []
        shortlist = [name for name in candidate_names if name in remaining]
        if fast_candidate_count is not None:
            shortlist = shortlist[:fast_candidate_count]
        for name in shortlist:
            if (
                max_trial_evaluations is not None
                and encoder.evaluations - started >= max_trial_evaluations
            ):
                break
            try:
                trials.append((encoder.size([*prefix, *chosen, name]), name))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                continue
        if not trials:
            break
        best_size, best_name = min(trials)
        improvement = (current - best_size) / current if current else 0.0
        if improvement < min_trial_improvement:
            break
        chosen.append(best_name)
        remaining.remove(best_name)
        current = best_size
    return chosen, float(baseline), float(current), encoder.evaluations - started


def _portfolio_order(
    table: pa.Table,
    candidates: Sequence[ColumnProfile],
    *,
    prefix: Sequence[str],
    slots: int,
    trial_sample_rows: int,
    candidate_pool_size: int,
    min_trial_improvement: float,
    compression: str | Mapping[str, str] | None,
    compression_level: int | Mapping[str, int] | None,
    null_placement: str,
    max_trial_evaluations: int,
    sort_backend: str,
    encoder: _TrialEncoder | None = None,
) -> tuple[list[str], float, float, int, str | None]:
    """Choose among complete cheap-heuristic plans using Parquet bytes.

    Unlike ``_codec_order``, this performs no greedy prefix expansion.  It
    constructs one complete natural-order plan from each cheap heuristic and
    spends at most one trial encode on every distinct plan.  The input/prefix
    order is always the baseline and remains selected unless a candidate beats
    it by ``min_trial_improvement``.
    """

    if encoder is None:
        encoder = _TrialEncoder(
            table,
            trial_sample_rows=trial_sample_rows,
            compression=compression,
            compression_level=compression_level,
            null_placement=null_placement,
            sort_backend=sort_backend,
        )
    pool = sorted(
        candidates,
        key=lambda item: (-item.potential_bytes, item.sample_distinct, item.name),
    )[:candidate_pool_size]

    proposed = _heuristic_proposals(pool, slots)
    prefix_tuple = tuple(prefix)
    plans: list[tuple[str, list[str], tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = {prefix_tuple}
    for name, selected in proposed:
        keys = (*prefix_tuple, *selected)
        if keys in seen:
            continue
        seen.add(keys)
        plans.append((name, selected, keys))

    started = encoder.evaluations
    baseline = encoder.size(prefix_tuple)
    best_size = baseline
    best_selected: list[str] = []
    best_name: str | None = None
    for name, selected, keys in plans:
        if encoder.evaluations - started >= max_trial_evaluations:
            break
        try:
            candidate_size = encoder.size(keys)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            continue
        if candidate_size < best_size:
            best_size = candidate_size
            best_selected = selected
            best_name = name

    evaluations = encoder.evaluations - started
    improvement = (baseline - best_size) / baseline if baseline else 0.0
    if best_size >= baseline or improvement < min_trial_improvement:
        return [], float(baseline), float(baseline), evaluations, None
    return (
        best_selected,
        float(baseline),
        float(best_size),
        evaluations,
        best_name,
    )


def _heuristic_proposals(
    candidates: Sequence[ColumnProfile], slots: int
) -> tuple[tuple[str, list[str]], ...]:
    """Complete natural-order proposals of the profile-only heuristics."""

    from .strategies import entropy_order, payload_benefit_order

    return (
        ("cardinality", _cardinality_order(candidates, slots)),
        ("weighted", _weighted_order(candidates, slots)),
        ("entropy", entropy_order(candidates, slots)),
        ("payload", payload_benefit_order(candidates, slots)),
    )


def _all_order(
    table: pa.Table,
    candidates: Sequence[ColumnProfile],
    profiles: Sequence[ColumnProfile],
    *,
    prefix: Sequence[str],
    slots: int,
    run_sample_rows: int,
    trial_sample_rows: int,
    candidate_pool_size: int,
    fast_candidate_count: int,
    max_trial_evaluations: int,
    min_run_improvement: float,
    min_trial_improvement: float,
    compression: str | Mapping[str, str] | None,
    compression_level: int | Mapping[str, int] | None,
    null_placement: str,
    sort_backend: str,
) -> tuple[list[str], float, float, int, str, str]:
    """Run every planner, then encode each distinct proposal on one sample.

    Proposals are the profile heuristics (on all candidates and on the bounded
    portfolio pool), entropy keys with frequency-ranked values, and the runs,
    codec_fast, and codec searches. The input/prefix order is the baseline and
    is kept unless the smallest proposal beats it by ``min_trial_improvement``.
    """

    encoder = _TrialEncoder(
        table,
        trial_sample_rows=trial_sample_rows,
        compression=compression,
        compression_level=compression_level,
        null_placement=null_placement,
        sort_backend=sort_backend,
    )
    pool = sorted(
        candidates,
        key=lambda item: (-item.potential_bytes, item.sample_distinct, item.name),
    )[:candidate_pool_size]
    proposals: dict[tuple[tuple[str, ...], str], list[str]] = {}

    def propose(name: str, selected: Sequence[str], value_order: str = "natural") -> None:
        if selected:
            proposals.setdefault((tuple(selected), value_order), []).append(name)

    heuristics = _heuristic_proposals(candidates, slots)
    for name, selected in heuristics:
        propose(name, selected)
    for name, selected in _heuristic_proposals(pool, slots):
        propose(f"portfolio:{name}", selected)
    # The frequency planner ranks the values of the entropy-selected keys.
    propose("frequency", dict(heuristics)["entropy"], "frequency")
    selected, _, _ = _runs_order(
        table,
        candidates,
        profiles,
        prefix=prefix,
        slots=slots,
        run_sample_rows=run_sample_rows,
        candidate_pool_size=candidate_pool_size,
        min_run_improvement=min_run_improvement,
        null_placement=null_placement,
    )
    propose("runs", selected)
    for name, fast in (("codec_fast", True), ("codec", False)):
        selected, *_ = _codec_order(
            table,
            candidates,
            prefix=prefix,
            slots=slots,
            trial_sample_rows=trial_sample_rows,
            candidate_pool_size=candidate_pool_size,
            min_trial_improvement=min_trial_improvement,
            compression=compression,
            compression_level=compression_level,
            null_placement=null_placement,
            fast_candidate_count=fast_candidate_count if fast else None,
            max_trial_evaluations=max_trial_evaluations if fast else None,
            sort_backend=sort_backend,
            encoder=encoder,
        )
        propose(name, selected)

    prefix_tuple = tuple(prefix)
    baseline = encoder.size(prefix_tuple)
    best: tuple[int, tuple[str, ...], str, list[str]] = (baseline, (), "natural", [])
    for (selected, value_order), names in proposals.items():
        try:
            size = encoder.size(
                (*prefix_tuple, *selected),
                frequency_from=len(prefix_tuple) if value_order == "frequency" else None,
            )
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            continue
        if size < best[0]:
            best = (size, selected, value_order, names)

    best_size, best_selected, value_order, names = best
    improvement = (baseline - best_size) / baseline if baseline else 0.0
    if not best_selected or improvement < min_trial_improvement:
        note = f"all compared {len(proposals)} proposals; input/prefix order retained"
        return [], float(baseline), float(baseline), encoder.evaluations, "natural", note
    note = f"all compared {len(proposals)} proposals; selected {', '.join(names)}"
    return (
        list(best_selected),
        float(baseline),
        float(best_size),
        encoder.evaluations,
        value_order,
        note,
    )


def plan_sort(
    value: pa.Table | pa.RecordBatch,
    *,
    algorithm: str = "auto",
    prefix: Iterable[str] = (),
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    sample_rows: int | None = DEFAULT_SAMPLE_ROWS,
    run_sample_rows: int = 50_000,
    trial_sample_rows: int = 250_000,
    max_sort_columns: int = 8,
    candidate_pool_size: int = 12,
    min_run_improvement: float = 0.005,
    min_trial_improvement: float = 0.005,
    trial_compression: str | Mapping[str, str] | None = "zstd",
    trial_compression_level: int | Mapping[str, int] | None = None,
    detect_json: bool = True,
    null_placement: str = "at_end",
    fast_candidate_count: int = 4,
    max_trial_evaluations: int = 24,
    sort_backend: str = "auto",
    full: bool = False,
) -> SortPlan:
    """Choose a compression-friendly lexicographic sort order.

    Algorithms:
      * ``cardinality``: ClickHouse/Lemire ascending-cardinality heuristic.
      * ``weighted``: select high-byte, repeatable columns, then cardinality-sort.
      * ``runs``: greedily minimize byte-weighted runs on a small sample.
      * ``codec``: greedy real-Parquet scoring on a bounded sample.
      * ``codec_fast``: bounded shortlist search with reusable ranked keys.
      * ``portfolio``: score complete cardinality, weighted, entropy, and
        payload plans against the unsorted/prefix baseline.
      * ``entropy``: effective cardinality adjusted for skew.
      * ``payload``: favors byte-heavy columns with repetition potential.
      * ``frequency``: frequency-ranked values using entropy-selected keys.
      * ``all``: run every planner above and keep the proposal with the
        fewest Parquet bytes, or the input/prefix order.
      * ``auto``: always use guarded, bounded ``codec_fast`` selection.
      * ``none``: retain input order (an explicit prefix is still honored).

    ``full=True`` plans with every row: profiles, run proxies, and trial
    encodes use the whole input instead of bounded samples.

    A table containing a column Arrow cannot reorder (see
    ``unreorderable_columns``) is not searched or sorted; a required prefix
    is then an error.
    """

    table = as_table(value)
    validate_unique_names(table)
    requested = algorithm
    normalized = _normalize_algorithm(algorithm)
    if max_sort_columns < 0:
        raise ValueError("max_sort_columns must be non-negative")
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")
    if not 0 <= min_run_improvement < 1:
        raise ValueError("min_run_improvement must be in [0, 1)")
    if not 0 <= min_trial_improvement < 1:
        raise ValueError("min_trial_improvement must be in [0, 1)")
    if trial_sample_rows <= 0:
        raise ValueError("trial_sample_rows must be positive")
    if run_sample_rows <= 0:
        raise ValueError("run_sample_rows must be positive")
    if fast_candidate_count <= 0:
        raise ValueError("fast_candidate_count must be positive")
    if max_trial_evaluations <= 0:
        raise ValueError("max_trial_evaluations must be positive")
    if sort_backend not in SORT_BACKENDS:
        raise ValueError(f"sort_backend must be one of {SORT_BACKENDS}")
    if null_placement not in {"at_start", "at_end"}:
        raise ValueError("null_placement must be 'at_start' or 'at_end'")

    prefix_keys = tuple(prefix)
    if len(set(prefix_keys)) != len(prefix_keys):
        raise ValueError("prefix contains duplicate columns")
    unknown_prefix = set(prefix_keys).difference(table.column_names)
    if unknown_prefix:
        raise ValueError(f"prefix columns not in table: {sorted(unknown_prefix)}")
    excluded = set(exclude)
    unknown_exclude = excluded.difference(table.column_names)
    if unknown_exclude:
        raise ValueError(f"excluded columns not in table: {sorted(unknown_exclude)}")
    conflict = excluded.intersection(prefix_keys)
    if conflict:
        raise ValueError(f"prefix columns cannot be excluded: {sorted(conflict)}")
    included = set(include) if include is not None else None
    if included is not None:
        unknown_include = included.difference(table.column_names)
        if unknown_include:
            raise ValueError(f"included columns not in table: {sorted(unknown_include)}")
    if full:
        sample_rows = None
        run_sample_rows = trial_sample_rows = max(1, table.num_rows)

    blocked = unreorderable_columns(table.schema)
    if blocked and prefix_keys:
        raise ValueError(f"cannot sort by the required prefix: Arrow cannot reorder {blocked}")
    if blocked and normalized != "none":
        # Rows move as a unit, so one such column rules out any reordering.
        return SortPlan(
            requested_algorithm=requested,
            algorithm=normalized,
            sort_keys=(),
            prefix_keys=(),
            profiles=(),
            total_rows=table.num_rows,
            sampled_rows=0,
            null_placement=null_placement,
            note=f"input order kept: Arrow cannot reorder {blocked}",
        )

    profiles = profile_table(
        table.select(prefix_keys) if normalized == "none" else table,
        sample_rows=sample_rows,
        exclude=() if normalized == "none" else excluded,
        detect_json=detect_json,
    )
    by_name = {item.name: item for item in profiles}
    bad_prefix = [name for name in prefix_keys if not by_name[name].eligible]
    if bad_prefix:
        raise ValueError(f"prefix contains unsupported columns: {bad_prefix}")
    candidates = _candidate_profiles(
        profiles,
        include=included,
        prefix=set(prefix_keys),
    )
    slots = max(0, max_sort_columns - len(prefix_keys))

    resolved = normalized
    note: str | None = None
    if normalized == "auto":
        resolved = "codec_fast"
        note = f"auto selected {resolved} from {len(candidates)} candidates"

    selected: list[str] = []
    baseline_cost: float | None = None
    final_cost: float | None = None
    score_kind: str | None = None
    trial_evaluations = 0
    value_order = "frequency" if resolved == "frequency" else "natural"
    if resolved == "cardinality":
        selected = _cardinality_order(candidates, slots)
    elif resolved == "weighted":
        selected = _weighted_order(candidates, slots)
    elif resolved == "runs" and candidates and slots and table.num_rows > 1:
        # Without a selectable key there is nothing to score: skip the search.
        score_kind = "weighted runs"
        selected, baseline_cost, final_cost = _runs_order(
            table,
            candidates,
            profiles,
            prefix=prefix_keys,
            slots=slots,
            run_sample_rows=run_sample_rows,
            candidate_pool_size=max(candidate_pool_size, slots),
            min_run_improvement=min_run_improvement,
            null_placement=null_placement,
        )
    elif resolved in {"codec", "codec_fast"} and candidates and slots and table.num_rows > 1:
        score_kind = "Parquet bytes"
        selected, baseline_cost, final_cost, trial_evaluations = _codec_order(
            table,
            candidates,
            prefix=prefix_keys,
            slots=slots,
            trial_sample_rows=trial_sample_rows,
            candidate_pool_size=max(candidate_pool_size, slots),
            min_trial_improvement=min_trial_improvement,
            compression=trial_compression,
            compression_level=trial_compression_level,
            null_placement=null_placement,
            fast_candidate_count=fast_candidate_count if resolved == "codec_fast" else None,
            max_trial_evaluations=max_trial_evaluations if resolved == "codec_fast" else None,
            sort_backend=sort_backend,
        )
    elif resolved == "portfolio" and candidates and slots and table.num_rows > 1:
        score_kind = "Parquet bytes"
        (
            selected,
            baseline_cost,
            final_cost,
            trial_evaluations,
            winner,
        ) = _portfolio_order(
            table,
            candidates,
            prefix=prefix_keys,
            slots=slots,
            trial_sample_rows=trial_sample_rows,
            candidate_pool_size=max(candidate_pool_size, slots),
            min_trial_improvement=min_trial_improvement,
            compression=trial_compression,
            compression_level=trial_compression_level,
            null_placement=null_placement,
            max_trial_evaluations=max_trial_evaluations,
            sort_backend=sort_backend,
        )
        note = (
            f"portfolio selected {winner}"
            if winner is not None
            else "portfolio retained the input/prefix order"
        )
    elif resolved == "all" and candidates and slots and table.num_rows > 1:
        score_kind = "Parquet bytes"
        (
            selected,
            baseline_cost,
            final_cost,
            trial_evaluations,
            value_order,
            note,
        ) = _all_order(
            table,
            candidates,
            profiles,
            prefix=prefix_keys,
            slots=slots,
            run_sample_rows=run_sample_rows,
            trial_sample_rows=trial_sample_rows,
            candidate_pool_size=max(candidate_pool_size, slots),
            fast_candidate_count=fast_candidate_count,
            max_trial_evaluations=max_trial_evaluations,
            min_run_improvement=min_run_improvement,
            min_trial_improvement=min_trial_improvement,
            compression=trial_compression,
            compression_level=trial_compression_level,
            null_placement=null_placement,
            sort_backend=sort_backend,
        )
    elif resolved in {"entropy", "payload", "frequency"}:
        from .strategies import entropy_order, payload_benefit_order

        order = payload_benefit_order if resolved == "payload" else entropy_order
        selected = order(candidates, slots)
    elif resolved == "none":
        selected = []

    keys = (*prefix_keys, *selected)
    return SortPlan(
        requested_algorithm=requested,
        algorithm=resolved,
        sort_keys=keys,
        prefix_keys=prefix_keys,
        profiles=profiles,
        total_rows=table.num_rows,
        sampled_rows=profiles[0].sampled_rows if profiles else 0,
        baseline_score=baseline_cost,
        estimated_score=final_cost,
        score_kind=score_kind,
        note=note,
        value_order=value_order,
        null_placement=null_placement,
        trial_evaluations=trial_evaluations,
    )
