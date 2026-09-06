"""Research-grade validation framework: walk-forward OOS evaluation, ablation, baselines, stress
tests, bootstrap / Monte Carlo, regimes, sanity and leakage tests, manifests, gates and reports."""

from .simulate import SimInputs, SimResult, simulate_strategy
from .metrics import StrategyMetrics, compute_strategy_metrics
from .windows import WalkForwardSchedule, Window, build_schedule
from .walkforward import OOSSeries, WalkForwardResult, WalkForwardRunner, WindowResult
from .runner import STAGES, ResearchRun, list_runs, load_summary, resolve_stages
from .gates import evaluate_gates
from .manifest import RunManifest, compare_runs

__all__ = ["SimInputs", "SimResult", "simulate_strategy", "StrategyMetrics", "compute_strategy_metrics",
           "WalkForwardSchedule", "Window", "build_schedule", "OOSSeries", "WalkForwardResult", "WalkForwardRunner",
           "WindowResult", "STAGES", "ResearchRun", "list_runs", "load_summary", "resolve_stages", "evaluate_gates",
           "RunManifest", "compare_runs"]
