from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.make_tables import REQUIRED_27_BLANK_IDS, compile_markdown_tables
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_step_cost_rate
from ts.eval.latency_benchmark import benchmark_student_and_teacher_latency
from ts.eval.metrics import compute_crisis_window_loss, compute_distribution_and_stationarity_stats
from ts.eval.shap_analysis import compute_modality_kernel_shap
from ts.eval.significance import ledoit_wolf_and_block_bootstrap_test
from ts.eval.xai_metrics import compute_table_615_xai_metrics
from ts.ledger import NumberLedger, RunContext
from ts.study1.encoders import CrossModalStateFusion
from ts.study1.ensemble import DynamicRegimeEnsemble
from ts.study2.citation_verifier import CitationVerificationStats


def test_phase2_cost_calibration_calm_vs_crisis() -> None:
    # Thesis Ch. VI Sec 3.3 (Eq. VI.8):
    # Calm market (ATR/P = 0.5%) -> ~0.10% before impact; Crisis (ATR/P = 1.0%) -> >= 0.15%
    c_calm = compute_phase2_step_cost_rate(
        price_t=30000.0, atr14_t=150.0, bar_base_vol_t=5000.0, abs_delta_w=0.5, prev_equity=1_000_000.0
    )
    c_crisis = compute_phase2_step_cost_rate(
        price_t=30000.0, atr14_t=300.0, bar_base_vol_t=5000.0, abs_delta_w=1.0, prev_equity=1_000_000.0
    )
    assert 0.0010 <= c_calm < 0.0013
    assert c_crisis >= 0.0015


def test_backtester_ledoit_wolf_shap_latency_and_27_blank_compiler(tmp_path: Path) -> None:
    rng = np.random.default_rng(2026)
    n = 300
    idx = pd.date_range("2022-05-01", periods=n, freq="1h", tz="UTC")
    close = 38000.0 * np.exp(np.cumsum(rng.normal(-0.0005, 0.008, size=n)))
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": np.full(n, 1500.0),
        },
        index=idx,
    )

    dist = compute_distribution_and_stationarity_stats(df["close"], symbol="BTCUSDT")
    assert dist.n_bars == n - 1
    assert dist.pearson_kurtosis > 1.5

    bt = CausalPortfolioBacktester()
    bh_res = bt.run(df, np.ones(n), phase=2)
    short_res = bt.run(df, -np.ones(n), phase=2)
    crisis = compute_crisis_window_loss(bh_res.equity_curve, "2022-05-01T00:00:00Z", "2022-05-10T23:00:00Z", "luna")
    assert crisis.window_mdd_pct <= 0.0

    # Genuine Ledoit-Wolf + Circular Block Bootstrap
    r_good = rng.normal(0.0015, 0.005, size=500)
    r_bad = rng.normal(-0.0005, 0.006, size=500)
    sig = ledoit_wolf_and_block_bootstrap_test(r_good, r_bad, n_bootstrap=200, block_size=24)
    assert sig.delta_sr_ann > 0.0
    assert sig.hac_p_value < 0.01
    assert sig.bootstrap_p_value < 0.05

    # KernelSHAP Eq. VI.12 modality attribution sums to 100%
    fusion = CrossModalStateFusion()
    ens = DynamicRegimeEnsemble(state_dim=128, experts_per_regime=2, ppo_kwargs={"device": torch.device("cpu")})
    mods = rng.standard_normal((24, 224)).astype(np.float32)
    shap_res = compute_modality_kernel_shap(
        mods, fusion, ens, n_background=8, n_explain=6, nsamples=32, device=torch.device("cpu")
    )
    assert abs((shap_res.visual_pct + shap_res.tech_plus_sentiment_pct) - 100.0) < 1e-4

    # Latency benchmark
    lat = benchmark_student_and_teacher_latency(n_student_runs=20, warmup_runs=5)
    assert lat.student_online.mean_ms > 0.0

    # XAI Table 6.15 & Ledger 27-blank verification
    c_stats = CitationVerificationStats(
        total_agent_outputs=100,
        rejected_agent_outputs=4,
        total_consensus_citations=96,
        matched_consensus_citations=96,
    )
    xai = compute_table_615_xai_metrics(
        citation_stats=c_stats,
        shap_result=shap_res,
        teacher_discrete_actions=np.array([0, 1, 2, 3, 4]),
        student_discrete_actions=np.array([0, 1, 2, 3, 2]),
        grounded_debate_actions=np.array([0, 1, 4, 3, 0]),
        stripped_evidence_actions=np.array([2, 1, 2, 3, 2]),
    )
    assert xai.citation_match_rate_pct == 100.0
    assert xai.hallucination_rejection_rate_pct == 4.0
    assert xai.teacher_student_agreement_pct == 80.0
    assert xai.evidence_removal_flip_rate_pct == 60.0

    ledger_file = tmp_path / "numbers.csv"
    ledger = NumberLedger(ledger_file, run_context=RunContext(stage="wp0", enforce_git=False))
    for nid, tbl, desc in REQUIRED_27_BLANK_IDS:
        ledger.record(nid, tbl, desc, 0.85, "{:.2f}", "", "tests/test_eval_and_tables.py", "abc123", "runs/test.json")

    md, audit = compile_markdown_tables(ledger.read_all())
    assert audit["total_required_blank_cells"] == 27
    assert audit["populated_count"] == 27
    assert audit["pending_count"] == 0
    sec_611_to_613 = md.split("## 표 6.11")[1].split("## 표 6.14")[0]
    assert "측정 대기" not in sec_611_to_613


def test_ledger_routes_non_final_runs_to_provisional(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TS_SMOKE_RUN", raising=False)
    path = tmp_path / "numbers.csv"
    args = ("{:.2f}", "", "tests/x.py", "abc", "runs/x.json")

    # No context at all: never final.
    NumberLedger(path).record("A", "T", "m", 1.0, *args)

    # Simulated LLM teacher behind a Study 3 number.
    sim_flags = dict(
        llm_simulated=True,
        teacher_signals_present=True,
        lob_input_mode="subbar_1m_microstructure",
        finbert_offline_hash=False,
        quick_verify=False,
        reduced_training_budget=False,
    )
    NumberLedger(path, RunContext("study3", sim_flags, enforce_git=False)).record("B", "T", "m", 1.0, *args)

    # A stage that forgot to declare its flags.
    NumberLedger(path, RunContext("study3", {}, enforce_git=False)).record("C", "T", "m", 1.0, *args)

    # Fully declared real run.
    real_flags = dict(sim_flags, llm_simulated=False)
    final_ledger = NumberLedger(path, RunContext("study3", real_flags, enforce_git=False))
    final_ledger.record("D", "T", "m", 1.0, *args)

    # Smoke environment variable overrides everything.
    monkeypatch.setenv("TS_SMOKE_RUN", "1")
    NumberLedger(path, RunContext("study3", real_flags, enforce_git=False)).record("E", "T", "m", 1.0, *args)

    assert set(final_ledger.read_all()) == {"D"}
    prov = final_ledger.read_provisional()
    assert set(prov) == {"A", "B", "C", "E"}
    assert "llm_simulated=True" in prov["B"]["provisional_reasons"]
    assert "not declared" in prov["C"]["provisional_reasons"]
    assert "TS_SMOKE_RUN" in prov["E"]["provisional_reasons"]
