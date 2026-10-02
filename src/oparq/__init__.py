"""oparq: compression-oriented row ordering for Parquet."""

from .benchmark import benchmark
from .core import apply_plan, optimize
from .io import read_parquet, rewrite, write
from .dataset import rewrite_dataset
from .learning import fit, fit_dataset, sample_dataset
from .models import (
    BenchmarkResult,
    ColumnProfile,
    OptimizationResult,
    SortPlan,
    RewritePlan,
    DatasetRewriteResult,
    WriteResult,
)
from .planning import ALGORITHMS, plan_sort
from .profile import profile_table

__all__ = [
    "ALGORITHMS",
    "BenchmarkResult",
    "ColumnProfile",
    "OptimizationResult",
    "SortPlan",
    "RewritePlan",
    "DatasetRewriteResult",
    "WriteResult",
    "apply_plan",
    "benchmark",
    "fit",
    "fit_dataset",
    "optimize",
    "plan_sort",
    "profile_table",
    "read_parquet",
    "rewrite",
    "rewrite_dataset",
    "sample_dataset",
    "write",
]

__version__ = "0.3.0"
