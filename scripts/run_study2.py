"""
WP3 — Study 2 Execution (Rolling 60-Day Granger+TE + Dynamic Hypergraph `H_t` + Grounded LLM Debate).

Measures & records:
- Hourly Teacher continuous target weight `W_final ∈ [-1, 1]` and consensus rationale `C_t`
- Table 6.14: Direct Teacher pipeline execution latency (`mean_ms`, speedup vs Student)
- Table 6.15 (3 blank cells):
  1. `인용 정합률 (Citation Match Rate)`
  2. `환각 기각률 (Hallucination Rejection Rate)`
  5. `근거 제거 시 결정 변화율 (Counterfactual Evidence Removal Decision Flip Rate)`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.macro_fetcher import MacroAssetFetcher
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.eval.latency_benchmark import benchmark_student_and_teacher_latency
from ts.ledger import NumberLedger
from ts.study1.regime_labeling import REGIME_NAMES
from ts.study2.causality import discover_candidate_edges
from ts.study2.citation_verifier import CitationVerifier
from ts.study2.debate import GroundedDebateEngine
from ts.study2.hypergraph import DynamicFinancialHypergraph, build_daily_hypergraph
from ts.study2.llm_client import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Study 2 Hypergraph & Multi-Agent LLM Debate.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--llm-backend", type=str, default="openai_compat", choices=["openai_compat", "ollama", "deterministic_econometric"])
    parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--news-file", type=str, default="")
    parser.add_argument("--counterfactual-stride", type=int, default=1, help="Run strip_evidence counterfactual every N bars (1 = all bars).")
    args = parser.parse_args()

    cfg = SystemConfig.from_json(args.config)
    cache_dir = PROJECT_ROOT / "data" / "cache"
    out_dir = PROJECT_ROOT / "runs" / "study2"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = "runs/study2/metrics.json"
    script_rel = "scripts/run_study2.py"

    binance = BinanceDataFetcher(cache_dir=cache_dir)
    df_all, _ = binance.fetch_ohlcv(
        symbol=cfg.data.primary_symbol,
        interval=cfg.data.bar_interval,
        start_utc=cfg.data.warmup_start,
        end_utc=cfg.data.eval_end,
        realign_grid=False,
    )

    macro_fetcher = MacroAssetFetcher(cache_dir=cache_dir)
    returns_60_df = macro_fetcher.build_aligned_60_asset_returns(pd.DatetimeIndex(df_all.index))
    news_store = (
        PointInTimeNewsStore.from_file(args.news_file)
        if args.news_file and Path(args.news_file).exists()
        else PointInTimeNewsStore()
    )

    # Load Study 1 OOS signals if present; otherwise evaluate across first_train_start..eval_end
    s1_file = PROJECT_ROOT / "runs" / "study1" / "study1_oos_signals.npz"
    if s1_file.exists():
        s1_npz = np.load(s1_file)
        target_bar_indices = s1_npz["oos_bar_indices"]
        oos_regimes = s1_npz["oos_regimes"]
        oos_probs = s1_npz["oos_ens_probs"]
    else:
        first_tr_ts = pd.Timestamp(cfg.data.first_train_start)
        target_bar_indices = np.where(df_all.index >= first_tr_ts)[0]
        oos_regimes = np.full(len(target_bar_indices), 2, dtype=np.int64)
        oos_probs = np.full((len(target_bar_indices), 5), 0.2, dtype=np.float32)

    llm = LLMClient(
        backend=args.llm_backend,
        model_name=args.llm_model,
        endpoint_url=args.llm_endpoint,
        temperature=cfg.study2.llm_temperature,
        max_tokens=cfg.study2.llm_max_tokens,
        cache_db_path=cache_dir / "llm_cache.sqlite",
    )
    verifier = CitationVerifier(te_tolerance=cfg.study2.te_citation_tolerance)
    debate_engine = GroundedDebateEngine(
        llm_client=llm,
        citation_verifier=verifier,
        max_rounds=cfg.study2.max_debate_rounds,
        max_citation_retries=cfg.study2.max_citation_retries,
        beta=cfg.study2.risk_beta,
        gamma=cfg.study2.risk_gamma,
    )

    rolling_bars = cfg.study2.rolling_window_days * 24  # 60d * 24 = 1,440 bars
    edge_week_cache: Dict[Tuple[str, str, str], Tuple[str, float, str]] = {}
    current_day_str = ""
    current_hg: DynamicFinancialHypergraph | None = None

    n_target = len(target_bar_indices)
    w_finals = np.zeros(n_target, dtype=np.float32)
    teacher_actions = np.zeros(n_target, dtype=np.int64)
    consensus_texts: List[str] = []

    cf_grounded_acts: List[int] = []
    cf_stripped_acts: List[int] = []

    for pos_i, bar_idx in enumerate(target_bar_indices):
        bar_ts = pd.Timestamp(df_all.index[bar_idx])
        day_str = bar_ts.strftime("%Y-%m-%d")

        # Update hypergraph H_t once per UTC day on strictly past 60-day window `[bar_idx - 1440 : bar_idx]`
        if current_hg is None or day_str != current_day_str:
            w_start = max(0, int(bar_idx) - rolling_bars)
            win_ret_df = returns_60_df.iloc[w_start:bar_idx]
            cands = discover_candidate_edges(
                win_ret_df,
                granger_lag=cfg.study2.granger_lag_bars,
                granger_p_threshold=cfg.study2.granger_p_threshold,
                te_lag=cfg.study2.te_lag_bars,
                te_top_quantile=cfg.study2.te_top_quantile,
            )
            news_ctx = news_store.format_headlines_for_llm(bar_ts, lookback_hours=24.0)
            iso_cal = bar_ts.isocalendar()
            iso_wk = f"{iso_cal.year}-W{iso_cal.week:02d}"
            current_hg = build_daily_hypergraph(
                nodes=list(returns_60_df.columns),
                candidates=cands,
                llm_client=llm,
                news_context=news_ctx,
                alpha=cfg.study2.edge_weight_alpha,
                max_hyperedges=cfg.study2.max_hyperedges,
                edge_week_cache=edge_week_cache,
                iso_week_key=iso_wk,
            )
            current_day_str = day_str

        assert current_hg is not None
        reg_name = REGIME_NAMES.get(int(oos_regimes[pos_i]), "Sideways")
        headlines_1h = news_store.format_headlines_for_llm(bar_ts + pd.Timedelta(hours=1), lookback_hours=6.0)

        outcome = debate_engine.run_bar_debate(
            hypergraph=current_hg,
            news_headlines_text=headlines_1h,
            study1_regime_name=reg_name,
            study1_probs=oos_probs[pos_i],
            target_asset=cfg.data.primary_symbol,
            strip_evidence=False,
        )
        w_finals[pos_i] = outcome.w_final
        teacher_actions[pos_i] = outcome.discrete_action
        consensus_texts.append(outcome.consensus_text)

        if pos_i % max(1, args.counterfactual_stride) == 0:
            cf_out = debate_engine.run_bar_debate(
                hypergraph=current_hg,
                news_headlines_text=headlines_1h,
                study1_regime_name=reg_name,
                study1_probs=oos_probs[pos_i],
                target_asset=cfg.data.primary_symbol,
                strip_evidence=True,
            )
            cf_grounded_acts.append(outcome.discrete_action)
            cf_stripped_acts.append(cf_out.discrete_action)

    # Benchmark direct Teacher inference latency (Table 6.14 Teacher row)
    sample_hg = current_hg or DynamicFinancialHypergraph(nodes=list(returns_60_df.columns), simple_edges=[])
    lat_report = benchmark_student_and_teacher_latency(
        student_model=None,
        teacher_step_fn=lambda: debate_engine.run_bar_debate(
            hypergraph=sample_hg,
            news_headlines_text="Benchmark headline check.",
            study1_regime_name="Bear",
            study1_probs=np.array([0.3, 0.3, 0.2, 0.1, 0.1], dtype=np.float32),
            strip_evidence=False,
        ),
        n_student_runs=1000,
        n_teacher_runs=20,
    )

    cf_g = np.asarray(cf_grounded_acts, dtype=np.int64)
    cf_s = np.asarray(cf_stripped_acts, dtype=np.int64)
    flip_rate_pct = float(np.mean(cf_g != cf_s)) * 100.0 if len(cf_g) > 0 else 0.0
    cite_match_pct = verifier.stats.citation_match_rate * 100.0
    halluc_rej_pct = verifier.stats.hallucination_rejection_rate * 100.0

    np.savez_compressed(
        out_dir / "teacher_signals.npz",
        bar_indices=target_bar_indices,
        w_finals=w_finals,
        teacher_actions=teacher_actions,
        cf_grounded_actions=cf_g,
        cf_stripped_actions=cf_s,
    )
    (out_dir / "consensus_texts.json").write_text(
        json.dumps(consensus_texts, ensure_ascii=False), encoding="utf-8"
    )

    payload = {
        "config_sha256": cfg.config_sha256,
        "is_simulated_llm": llm.is_simulated_llm,
        "llm_backend": args.llm_backend,
        "llm_model": args.llm_model,
        "n_bars_processed": int(n_target),
        "table_6_15_study2_rows": {
            "citation_match_rate_pct": cite_match_pct,
            "hallucination_rejection_rate_pct": halluc_rej_pct,
            "evidence_removal_flip_rate_pct": flip_rate_pct,
        },
        "table_6_14_latency": lat_report.to_dict(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if not llm.is_simulated_llm:
        ledger = NumberLedger()
        rec_kw = dict(script_path=script_rel, config_sha256=cfg.config_sha256, run_log_path=log_path)
        ledger.record("T6.14_TEACHER_MEAN_MS", "Table 6.14", "Teacher direct inference latency mean", lat_report.teacher_direct.mean_ms, "{:.1f}", "ms", **rec_kw)
        ledger.record("T6.15_CITATION_MATCH_PCT", "Table 6.15", "Citation Match Rate", cite_match_pct, "{:.2f}", "%", **rec_kw)
        ledger.record("T6.15_HALLUCINATION_REJECT_PCT", "Table 6.15", "Hallucination Rejection Rate", halluc_rej_pct, "{:.2f}", "%", **rec_kw)
        ledger.record("T6.15_EVIDENCE_FLIP_PCT", "Table 6.15", "Counterfactual Evidence Removal Flip Rate", flip_rate_pct, "{:.2f}", "%", **rec_kw)

    print(json.dumps(payload["table_6_15_study2_rows"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
