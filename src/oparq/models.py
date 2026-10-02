"""Public result models used by oparq."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
from pathlib import Path
import re
from typing import Any

import pyarrow as pa


def _legacy_type_pattern(dtype: pa.DataType) -> str:
    """Conservatively accept Parquet's cosmetic list child rename.

    Old plan artifacts only recorded Arrow's display strings. Construct the
    pattern from the actual type tree, rather than replacing text globally:
    struct field names can themselves contain text resembling a list type.
    New artifacts store a serialized schema and do not need this fallback.
    """

    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        kind = "fixed_size_list" if pa.types.is_fixed_size_list(dtype) else (
            "large_list" if pa.types.is_large_list(dtype) else "list"
        )
        field = dtype.value_field
        names = {field.name, "item", "element"}
        field_pattern = "(?:" + "|".join(re.escape(name) for name in sorted(names)) + ")"
        nullable = "" if field.nullable else " not null"
        size = re.escape(f"[{dtype.list_size}]") if pa.types.is_fixed_size_list(dtype) else ""
        return re.escape(kind + "<") + field_pattern + re.escape(": ") + _legacy_type_pattern(field.type) + re.escape(nullable + ">") + size
    if pa.types.is_struct(dtype):
        fields = [re.escape(field.name + ": ") + _legacy_type_pattern(field.type)
                  + re.escape("" if field.nullable else " not null") for field in dtype]
        return re.escape("struct<") + re.escape(", ").join(fields) + re.escape(">")
    if pa.types.is_dictionary(dtype):
        return re.escape("dictionary<values=") + _legacy_type_pattern(dtype.value_type) + re.escape(", indices=") + _legacy_type_pattern(dtype.index_type) + re.escape(f", ordered={int(dtype.ordered)}>")
    return re.escape(str(dtype))


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Compression-relevant measurements for one Arrow column.

    ``sample_distinct`` includes null as one value. ``sample_runs`` describes
    the current sample order. Both are exact for the planning sample and for
    the full table when ``sampled_rows == total_rows``. ``potential_bytes`` is
    a ranking proxy, not a promised byte saving.
    """

    name: str
    arrow_type: str
    byte_size: int
    order_sensitive_bytes: int
    null_count: int
    sampled_rows: int
    sample_distinct: int
    sample_runs: int
    dominant_fraction: float
    gini_impurity: float
    repeatability: float
    cluster_gain: float
    potential_bytes: float
    eligible: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SortPlan:
    """A deterministic decision about how a table should be sorted."""

    requested_algorithm: str
    algorithm: str
    sort_keys: tuple[str, ...]
    prefix_keys: tuple[str, ...]
    profiles: tuple[ColumnProfile, ...]
    total_rows: int
    sampled_rows: int
    value_order: str = "natural"
    null_placement: str = "at_end"
    trial_evaluations: int = 0
    baseline_score: float | None = None
    estimated_score: float | None = None
    score_kind: str | None = None
    note: str | None = None

    @property
    def sort_order(self) -> list[tuple[str, str]]:
        """Natural ordering that can be truthfully stored in Parquet metadata."""

        return [(name, "ascending") for name in self.sorting_keys]

    @property
    def sorting_keys(self) -> tuple[str, ...]:
        """Keys whose standard ordering is guaranteed by this plan.

        Frequency-ranked values are deliberately not natural-order sorted, so
        only an explicit natural-order prefix may be advertised to readers.
        """

        if self.value_order == "frequency":
            return self.prefix_keys
        return self.sort_keys

    def profile(self, name: str) -> ColumnProfile:
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise KeyError(name)

    def as_dict(self, *, include_profiles: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requested_algorithm": self.requested_algorithm,
            "algorithm": self.algorithm,
            "sort_keys": list(self.sort_keys),
            "prefix_keys": list(self.prefix_keys),
            "total_rows": self.total_rows,
            "sampled_rows": self.sampled_rows,
            "value_order": self.value_order,
            "null_placement": self.null_placement,
            "trial_evaluations": self.trial_evaluations,
            "baseline_score": self.baseline_score,
            "estimated_score": self.estimated_score,
            "score_kind": self.score_kind,
            "note": self.note,
        }
        if include_profiles:
            result["profiles"] = [profile.as_dict() for profile in self.profiles]
        return result

    def explain(self) -> str:
        keys = ", ".join(self.sort_keys) if self.sort_keys else "(keep input order)"
        lines = [
            f"algorithm: {self.algorithm}",
            f"sample: {self.sampled_rows:,} / {self.total_rows:,} rows",
            f"sort keys: {keys}",
        ]
        if self.value_order != "natural":
            lines.append(f"value order: {self.value_order}")
        if self.trial_evaluations:
            lines.append(f"trial encodes: {self.trial_evaluations:,}")
        if self.baseline_score is not None and self.estimated_score is not None:
            if self.baseline_score:
                change = self.estimated_score / self.baseline_score - 1.0
                label = self.score_kind or "planning score"
                lines.append(f"sample {label}: {change:+.1%}")
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RewritePlan:
    """Reusable ordering learned from samples, independent of row count.

    Codec and level are intentional output settings; unlike a Parquet footer,
    this artifact records the compression level as well as the codec.
    """

    algorithm: str
    sort_keys: tuple[str, ...]
    prefix_keys: tuple[str, ...]
    column_types: tuple[tuple[str, str], ...]
    value_order: str = "natural"
    null_placement: str = "at_end"
    compression: str | dict[str, str] | None = "zstd"
    compression_level: int | dict[str, int] | None = None
    sampled_rows: int = 0
    evaluations: tuple[dict[str, Any], ...] = ()
    format_version: int = 1
    schema_base64: str | None = None

    def _recorded_schema(self) -> pa.Schema | None:
        if self.schema_base64 is None:
            return None
        try:
            raw = base64.b64decode(self.schema_base64, validate=True)
            schema = pa.ipc.read_schema(pa.BufferReader(raw))
        except (ValueError, TypeError, pa.ArrowException) as error:
            raise ValueError("rewrite plan contains an invalid serialized schema") from error
        if len(set(schema.names)) != len(schema):
            raise ValueError("rewrite plan schema contains duplicate columns")
        if tuple((field.name, str(field.type)) for field in schema) != self.column_types:
            raise ValueError("rewrite plan schema disagrees with recorded column types")
        return schema

    def for_table(self, value: pa.Table | pa.RecordBatch) -> SortPlan:
        """Validate the learned schema and bind the fixed keys to these rows."""

        if len(set(value.schema.names)) != len(value.schema.names):
            raise ValueError("table contains duplicate column names")
        recorded = self._recorded_schema()
        expected = dict(self.column_types)
        for name, kind in expected.items():
            if name not in value.schema.names:
                raise ValueError(f"learned column {name!r} is missing")
            field = value.schema.field(name)
            actual = field.type
            compatible = (actual.equals(recorded.field(name).type)
                          if recorded is not None else
                          re.fullmatch(_legacy_type_pattern(actual), kind) is not None)
            if not compatible:
                raise ValueError(f"column {name!r} changed type: {kind} -> {actual}")
            if recorded is not None and field.nullable != recorded.field(name).nullable:
                raise ValueError(f"column {name!r} changed nullability")
        missing = set(self.sort_keys).difference(value.schema.names)
        if missing:
            raise ValueError(f"sort keys are missing: {sorted(missing)}")
        return SortPlan(
            requested_algorithm=self.algorithm,
            algorithm=self.algorithm,
            sort_keys=self.sort_keys,
            prefix_keys=self.prefix_keys,
            profiles=(),
            total_rows=value.num_rows,
            sampled_rows=self.sampled_rows,
            value_order=self.value_order,
            null_placement=self.null_placement,
            note="fixed ordering learned from earlier samples; planning skipped",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RewritePlan:
        if not isinstance(payload, dict):
            raise ValueError("rewrite plan must be a JSON object")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported rewrite plan format_version")
        missing = sorted({"algorithm", "sort_keys", "prefix_keys", "column_types"}.difference(payload))
        if missing:
            raise ValueError(f"rewrite plan is missing fields: {missing}")
        from .planning import ALGORITHMS

        if payload.get("algorithm") not in ALGORITHMS:
            raise ValueError("rewrite plan uses an unknown algorithm")
        if payload.get("value_order", "natural") not in {"natural", "frequency"}:
            raise ValueError("invalid rewrite plan value_order")
        if payload.get("null_placement", "at_end") not in {"at_start", "at_end"}:
            raise ValueError("invalid rewrite plan null_placement")
        keys = tuple(payload["sort_keys"])
        prefix = tuple(payload["prefix_keys"])
        if len(set(keys)) != len(keys) or keys[:len(prefix)] != prefix:
            raise ValueError("rewrite plan must have unique keys beginning with prefix")
        kinds = tuple(tuple(item) for item in payload["column_types"])
        if len({item[0] for item in kinds}) != len(kinds):
            raise ValueError("rewrite plan contains duplicate columns")
        if set(keys).difference(dict(kinds)):
            raise ValueError("rewrite plan keys must have recorded column types")
        result = cls(
            algorithm=payload["algorithm"], sort_keys=keys, prefix_keys=prefix,
            column_types=kinds, value_order=payload.get("value_order", "natural"),
            null_placement=payload.get("null_placement", "at_end"),
            compression=payload.get("compression"),
            compression_level=payload.get("compression_level"),
            sampled_rows=payload.get("sampled_rows", 0),
            evaluations=tuple(payload.get("evaluations", ())),
            schema_base64=payload.get("schema_base64"),
        )
        result._recorded_schema()
        return result

    def save(self, destination: str | Path, *, overwrite: bool = False) -> None:
        import json
        from .storage import open_output

        with open_output(destination, overwrite=overwrite) as stream:
            stream.write((json.dumps(self.as_dict(), indent=2) + "\n").encode())

    @classmethod
    def load(cls, source: str | Path) -> RewritePlan:
        import json
        from .storage import open_input

        with open_input(source) as stream:
            return cls.from_dict(json.loads(stream.read()))


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    table: pa.Table
    plan: SortPlan
    planning_seconds: float
    sort_seconds: float

    @property
    def planning_and_sort_seconds(self) -> float:
        """Total optimizer time retained for backwards-compatible reporting."""

        return self.planning_seconds + self.sort_seconds


@dataclass(frozen=True, slots=True)
class WriteResult:
    path: Path | str
    file_size: int
    plan: SortPlan
    planning_seconds: float
    sort_seconds: float
    write_seconds: float
    permutation_seconds: float | None = None
    gathering_seconds: float | None = None
    action: str = "rewritten"

    @property
    def planning_and_sort_seconds(self) -> float:
        return self.planning_seconds + self.sort_seconds


@dataclass(frozen=True, slots=True)
class DatasetRewriteResult:
    source: str
    destination: str
    files: tuple[WriteResult, ...]

    @property
    def rewritten_files(self) -> int:
        return sum(item.action == "rewritten" for item in self.files)

    @property
    def copied_files(self) -> int:
        return sum(item.action == "copied" for item in self.files)

    @property
    def skipped_files(self) -> int:
        return sum(item.action == "skipped" for item in self.files)

@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    algorithm: str
    resolved_algorithm: str
    sort_keys: tuple[str, ...]
    size_bytes: int
    planning_seconds: float
    sort_seconds: float
    write_seconds: float
    savings_fraction: float = 0.0
    permutation_seconds: float | None = None
    gathering_seconds: float | None = None

    @property
    def planning_and_sort_seconds(self) -> float:
        """Return the former combined timing field."""

        return self.planning_seconds + self.sort_seconds
