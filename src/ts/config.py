"""
Typed configuration loader aligned with Appendix A (Tables A.1-A.4) and Chapter VI.

Uses JSON (`configs/appendix_a_config.json`) so no `.yaml`/`.yml`/`.config` files
are touched or modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "appendix_a_config.json"


@dataclass(frozen=True)
class DataConfig:
    primary_symbol: str = "BTCUSDT"
    warmup_start: str = "2020-08-01T00:00:00Z"
    first_train_start: str = "2020-10-01T00:00:00Z"
    eval_start: str = "2021-10-01T00:00:00Z"
    eval_end: str = "2023-12-31T23:00:00Z"
    bar_interval: str = "1h"
    control_symbols: Tuple[str, ...] = ("SPY", "GLD")
    lob_ticks: int = 100
    lob_features: int = 40
    lob_windows_per_bar: int = 10


@dataclass(frozen=True)
class Study1PPOConfig:
    action_dim: int = 5
    target_weights: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    decision_interval_hours: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    minibatch_size: int = 1024
    rollout_steps: int = 2048
    default_rollouts_per_fold: int = 64
    ppo_epochs: int = 4
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-6
    bear_sortino_window_bars: int = 24
    bear_dd_floor: float = 1e-4
    bear_cost_penalty: float = 1.0
    sideways_turnover_penalty: float = 0.002
    rolling_sharpe_window_bars: int = 720
    ensemble_temperature: float = 10.0
    experts_per_regime: int = 5
    max_experts_per_regime_for_sensitivity: int = 7
    hidden_dims: Tuple[int, ...] = (128, 64)


@dataclass(frozen=True)
class Study1EncoderConfig:
    candlestick_window_bars: int = 60
    image_size: Tuple[int, int] = (224, 224)
    cnn_backbone: str = "resnet18_cbam"
    cbam_reduction: int = 16
    cbam_spatial_kernel: int = 7
    d_visual: int = 128
    d_tech: int = 64
    tech_zscore_window: int = 120
    sentiment_model: str = "DeepSeek-R1-Distill-Qwen-32B"
    d_senti: int = 32
    senti_ewma_span_bars: int = 24
    senti_ewma_alpha: float = 0.08
    cross_attn_dim: int = 128
    sma_slope_window: int = 50
    rel_vol_window: int = 20
    trend_quantile: float = 0.60
    vol_quantile: float = 0.75
    trend_scan_min_horizon: int = 24
    trend_scan_max_horizon: int = 120
    trend_scan_t_threshold: float = 2.0
    regime_switch_threshold: float = 0.60
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05


@dataclass(frozen=True)
class Study2Config:
    num_macro_nodes_target: int = 60
    granger_lag_bars: int = 24
    granger_p_threshold: float = 0.05
    te_lag_bars: int = 24
    te_top_quantile: float = 0.90
    rolling_window_days: int = 60
    update_interval_hours: int = 24
    edge_weight_alpha: float = 0.5
    max_hyperedges: int = 20
    llm_model: str = "DeepSeek-R1-Distill-Qwen-32B-4bit"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 256
    max_debate_rounds: int = 3
    max_citation_retries: int = 3
    te_citation_tolerance: float = 0.05
    risk_beta: float = 8.0
    risk_gamma: float = 6.0


@dataclass(frozen=True)
class Study3Config:
    lob_seq_len: int = 100
    lob_input_dim: int = 40
    student_hidden_dim: int = 128
    projection_mid_dim: int = 256
    reasoning_dim: int = 768
    finbert_model: str = "ProsusAI/finbert"
    teacher_sigma_w: float = 0.25
    distill_temperature_tau: float = 2.0
    lambda_task: float = 0.4
    lambda_policy: float = 0.4
    lambda_reasoning: float = 0.2
    student_ppo_clip: float = 0.2
    student_value_coef: float = 0.5
    student_entropy_coef: float = 0.01
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class EvaluationConfig:
    initial_capital_usdt: float = 1_000_000.0
    risk_free_rate_annual: float = 0.0
    btc_bars_per_year: int = 8760
    equity_bars_per_year: int = 1638
    phase1_fixed_fee_rate: float = 0.0005
    phase1_kfold_splits: int = 5
    phase2_fee_rate: float = 0.0005
    phase2_atr_period: int = 14
    phase2_kappa: float = 0.1
    phase2_eta: float = 0.05
    walk_forward_train_months: int = 12
    walk_forward_test_months: int = 3
    walk_forward_step_months: int = 3
    walk_forward_num_folds: int = 9
    bootstrap_block_bars: int = 24
    bootstrap_replications: int = 5000
    latency_benchmark_runs: int = 10000
    sensitivity_lambda3_grid: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    sensitivity_temp_grid: Tuple[Union[float, str], ...] = (1.0, 5.0, 10.0, 20.0, "static_1_over_N")
    sensitivity_experts_grid: Tuple[int, ...] = (3, 5, 7)
    crisis_windows: Dict[str, Tuple[str, str]] = field(
        default_factory=lambda: {
            "luna_rate_hike": ("2022-05-01T00:00:00Z", "2022-06-30T23:00:00Z"),
            "ftx_collapse": ("2022-11-07T00:00:00Z", "2022-11-09T23:00:00Z"),
        }
    )


@dataclass(frozen=True)
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    study1_ppo: Study1PPOConfig = field(default_factory=Study1PPOConfig)
    study1_encoders: Study1EncoderConfig = field(default_factory=Study1EncoderConfig)
    study2: Study2Config = field(default_factory=Study2Config)
    study3: Study3Config = field(default_factory=Study3Config)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    config_sha256: str = ""

    @classmethod
    def from_json(cls, path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> "SystemConfig":
        p = Path(path)
        raw_bytes = p.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
        d: Dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))

        data_d = dict(d.get("data", {}))
        if "control_symbols" in data_d:
            data_d["control_symbols"] = tuple(data_d["control_symbols"])

        ppo_d = dict(d.get("study1_ppo", {}))
        if "target_weights" in ppo_d:
            ppo_d["target_weights"] = tuple(float(x) for x in ppo_d["target_weights"])
        if "hidden_dims" in ppo_d:
            ppo_d["hidden_dims"] = tuple(int(x) for x in ppo_d["hidden_dims"])

        enc_d = dict(d.get("study1_encoders", {}))
        if "image_size" in enc_d:
            enc_d["image_size"] = (int(enc_d["image_size"][0]), int(enc_d["image_size"][1]))

        eval_d = dict(d.get("evaluation", {}))
        if "sensitivity_lambda3_grid" in eval_d:
            eval_d["sensitivity_lambda3_grid"] = tuple(float(x) for x in eval_d["sensitivity_lambda3_grid"])
        if "sensitivity_temp_grid" in eval_d:
            eval_d["sensitivity_temp_grid"] = tuple(eval_d["sensitivity_temp_grid"])
        if "sensitivity_experts_grid" in eval_d:
            eval_d["sensitivity_experts_grid"] = tuple(int(x) for x in eval_d["sensitivity_experts_grid"])
        if "crisis_windows" in eval_d:
            eval_d["crisis_windows"] = {
                k: (str(v[0]), str(v[1])) for k, v in eval_d["crisis_windows"].items()
            }

        return cls(
            data=DataConfig(**data_d),
            study1_ppo=Study1PPOConfig(**ppo_d),
            study1_encoders=Study1EncoderConfig(**enc_d),
            study2=Study2Config(**d.get("study2_hypergraph_debate", {})),
            study3=Study3Config(**d.get("study3_distillation", {})),
            evaluation=EvaluationConfig(**eval_d),
            config_sha256=digest,
        )
