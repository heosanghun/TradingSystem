"""Chapter VI Evaluation, Cost Model, Walk-Forward, Statistical Tests, SHAP, Latency & XAI."""

from ts.eval.backtester import BacktestResult, CausalPortfolioBacktester
from ts.eval.cost_model import (
    compute_phase1_cost_rates,
    compute_phase2_bar_base_cost_rates,
    compute_phase2_step_cost_rate,
    compute_wilder_atr,
)
from ts.eval.latency_benchmark import LatencyBenchmarkReport, benchmark_student_and_teacher_latency
from ts.eval.metrics import (
    CrisisLossMetrics,
    DistributionStats,
    PerformanceSummary,
    compute_crisis_window_loss,
    compute_distribution_and_stationarity_stats,
    compute_performance_summary,
)
from ts.eval.shap_analysis import ModalitySHAPResult, compute_modality_kernel_shap
from ts.eval.significance import SharpeSignificanceResult, ledoit_wolf_and_block_bootstrap_test
from ts.eval.walk_forward import WalkForwardFold, generate_phase1_kfold_splits, generate_phase2_walk_forward_folds
from ts.eval.xai_metrics import Table615XAIMetrics, compute_table_615_xai_metrics

__all__ = [
    "BacktestResult",
    "CausalPortfolioBacktester",
    "CrisisLossMetrics",
    "DistributionStats",
    "LatencyBenchmarkReport",
    "ModalitySHAPResult",
    "PerformanceSummary",
    "SharpeSignificanceResult",
    "Table615XAIMetrics",
    "WalkForwardFold",
    "benchmark_student_and_teacher_latency",
    "compute_crisis_window_loss",
    "compute_distribution_and_stationarity_stats",
    "compute_modality_kernel_shap",
    "compute_performance_summary",
    "compute_phase1_cost_rates",
    "compute_phase2_bar_base_cost_rates",
    "compute_phase2_step_cost_rate",
    "compute_table_615_xai_metrics",
    "compute_wilder_atr",
    "generate_phase1_kfold_splits",
    "generate_phase2_walk_forward_folds",
    "ledoit_wolf_and_block_bootstrap_test",
]
