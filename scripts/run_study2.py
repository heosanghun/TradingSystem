"""Run the hourly hypergraph teacher with explicit inputs and durable checkpoints."""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SystemConfig
from ts.data.binance_fetcher import BinanceDataFetcher
from ts.data.macro_fetcher import MacroAssetFetcher
from ts.data.news_fetcher import PointInTimeNewsStore
from ts.eval.latency_benchmark import benchmark_student_and_teacher_latency
from ts.ledger import NumberLedger, RunContext
from ts.study1.regime_labeling import REGIME_NAMES
from ts.study2.causality import discover_candidate_edges_cached
from ts.study2.citation_verifier import CitationVerificationStats, CitationVerifier
from ts.study2.debate import GroundedDebateEngine
from ts.study2.hypergraph import DynamicFinancialHypergraph, VerifiedSimpleEdge, build_daily_hypergraph
from ts.study2.llm_client import LLMClient, LLMCallError
from ts.study2.persistent_state import SQLiteJSONMap, run_checkpointed_bar


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Study 2 Hypergraph & Multi-Agent LLM Debate.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--llm-backend", type=str, default="openai_compat", choices=["openai_compat", "ollama", "deterministic_econometric"])
    parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--news-file", type=str, default="")
    parser.add_argument("--counterfactual-stride", type=int, default=1, help="Run strip_evidence counterfactual every N bars (1 = all bars).")
    parser.add_argument("--debate-stride", type=int, default=1)
    parser.add_argument("--diagnostic-run", action="store_true")
    args = parser.parse_args()
    if args.debate_stride < 1 or args.counterfactual_stride < 1:
        parser.error("Strides must be positive")
    if args.debate_stride != 1 and not args.diagnostic_run:
        parser.error("Reduced debate coverage requires a diagnostic run until D-4 is resolved")

    cfg = SystemConfig.from_json(args.config)
    cache_dir = PROJECT_ROOT / "data" / "cache"
    out_dir = PROJECT_ROOT / "runs" / "study2"
    if args.diagnostic_run:
        out_dir = out_dir / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = str((out_dir / "metrics.json").relative_to(PROJECT_ROOT)).replace("\\", "/")
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
    try:
        returns_60_df = macro_fetcher.build_aligned_60_asset_returns(pd.DatetimeIndex(df_all.index))
    except RuntimeError:
        (out_dir / "metrics.json").write_text(json.dumps({"status": "data_incomplete",
            "universe": macro_fetcher.last_universe_report, "ledger_written": False}, indent=2), encoding="utf-8")
        raise
    if not args.news_file or not Path(args.news_file).is_file():
        raise FileNotFoundError("A real point-in-time news archive is required via --news-file")
    news_store = (
        PointInTimeNewsStore.from_file(args.news_file)
        if args.news_file and Path(args.news_file).exists()
        else PointInTimeNewsStore()
    )
    if news_store.df.empty:
        raise ValueError("The supplied news archive is empty")
    if not args.diagnostic_run:
        required_months = pd.period_range(pd.Timestamp(cfg.data.eval_start).tz_localize(None).to_period("M"),
                                         pd.Timestamp(cfg.data.eval_end).tz_localize(None).to_period("M"), freq="M")
        counts = news_store.df.published_at.dt.tz_localize(None).dt.to_period("M").value_counts()
        missing_months = [str(month) for month in required_months if counts.get(month, 0) == 0]
        if missing_months:
            raise ValueError(f"News archive has no evaluation headlines in: {missing_months}")

    # Load Study 1 full Teacher timeline signals (`2020-10`..`2023-12`) if present
    s1_file = PROJECT_ROOT / "runs" / "study1" / "study1_oos_signals.npz"
    eval_start_ts = pd.Timestamp(cfg.data.eval_start)
    if s1_file.exists():
        s1_npz = np.load(s1_file)
        if "teacher_bar_indices" in s1_npz.files:
            target_bar_indices = s1_npz["teacher_bar_indices"]
            bar_regimes = s1_npz["teacher_regimes"]
            bar_probs = s1_npz["teacher_ens_probs"]
            is_oos_mask = s1_npz["teacher_is_oos"].astype(bool)
        else:
            target_bar_indices = s1_npz["oos_bar_indices"]
            bar_regimes = s1_npz["oos_regimes"]
            bar_probs = s1_npz["oos_ens_probs"]
            is_oos_mask = np.ones(len(target_bar_indices), dtype=bool)
    else:
        raise FileNotFoundError("Study 1 teacher timeline is required; refusing uniform substitute signals")

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
    identity = hashlib.sha256()
    identity.update(cfg.config_sha256.encode())
    identity.update(json.dumps([args.llm_backend, args.llm_endpoint, args.llm_model,
                                args.debate_stride, args.counterfactual_stride,
                                args.diagnostic_run], sort_keys=True).encode())
    identity.update(s1_file.read_bytes())
    identity.update(pd.util.hash_pandas_object(returns_60_df, index=True).values.tobytes())
    identity.update(json.dumps(list(returns_60_df.columns)).encode())
    if args.news_file:
        identity.update(Path(args.news_file).read_bytes())
    for source in sorted((PROJECT_ROOT / "src" / "ts" / "study2").glob("*.py")):
        identity.update(source.read_bytes())
    identity.update(Path(__file__).read_bytes())
    run_id = identity.hexdigest()
    edge_week_cache = SQLiteJSONMap(cache_dir / "edge_week_cache.sqlite", run_id)
    bar_store = SQLiteJSONMap(out_dir / "checkpoint.sqlite", run_id)
    day_store = SQLiteJSONMap(out_dir / "checkpoint.sqlite", run_id + ":graphs")
    current_day_str = ""
    current_hg = None
    n_target = len(target_bar_indices)
    w_finals = np.full(n_target, np.nan, dtype=np.float32)
    teacher_actions = np.full(n_target, -1, dtype=np.int64)
    consensus_texts = [""] * n_target
    cf_grounded_acts = []
    cf_stripped_acts = []
    failed_bars = []
    totals = CitationVerificationStats()
    oos_counter = 0
    previous_record = None

    for pos_i, bar_idx in enumerate(target_bar_indices):
        is_oos_bar = bool(is_oos_mask[pos_i])
        bar_ts = pd.Timestamp(df_all.index[bar_idx])
        day_str = bar_ts.strftime("%Y-%m-%d")
        if pos_i in bar_store:
            record = bar_store[pos_i]
        elif args.debate_stride > 1 and pos_i % args.debate_stride and previous_record is not None:
            record = {"status": "carried", "outcome": previous_record["outcome"],
                      "counterfactual": None, "stats_delta": {}}
            bar_store[pos_i] = record
        else:
            try:
                if current_hg is None or day_str != current_day_str:
                    if day_str in day_store:
                        saved = day_store[day_str]
                        current_hg = DynamicFinancialHypergraph(
                            nodes=saved["nodes"],
                            simple_edges=[VerifiedSimpleEdge(**row) for row in saved["edges"]],
                            max_hyperedges=cfg.study2.max_hyperedges)
                    else:
                        win_ret_df = returns_60_df.iloc[max(0, int(bar_idx) - rolling_bars):bar_idx]
                        cands = discover_candidate_edges_cached(
                            win_ret_df, cache_path=cache_dir / "hypergraph" / f"{day_str}.parquet",
                            granger_lag=cfg.study2.granger_lag_bars,
                            granger_p_threshold=cfg.study2.granger_p_threshold,
                            te_lag=cfg.study2.te_lag_bars, te_top_quantile=cfg.study2.te_top_quantile)
                        iso_cal = bar_ts.isocalendar()
                        current_hg = build_daily_hypergraph(
                            nodes=list(returns_60_df.columns), candidates=cands, llm_client=llm,
                            news_context=news_store.format_headlines_for_llm(bar_ts, lookback_hours=24.),
                            alpha=cfg.study2.edge_weight_alpha, max_hyperedges=cfg.study2.max_hyperedges,
                            edge_week_cache=edge_week_cache,
                            iso_week_key=f"{iso_cal.year}-W{iso_cal.week:02d}")
                        day_store[day_str] = {"nodes": list(returns_60_df.columns),
                                              "edges": [asdict(e) for e in current_hg.simple_edges.values()]}
                    current_day_str = day_str
                record = run_checkpointed_bar(
                    bar_store, pos_i, debate_engine,
                    dict(hypergraph=current_hg,
                         news_headlines_text=news_store.format_headlines_for_llm(
                             bar_ts + pd.Timedelta(hours=1), lookback_hours=6.),
                         study1_regime_name=REGIME_NAMES.get(int(bar_regimes[pos_i]), "Sideways"),
                         study1_probs=bar_probs[pos_i], target_asset=cfg.data.primary_symbol),
                    counterfactual=is_oos_bar and oos_counter % args.counterfactual_stride == 0)
            except LLMCallError as exc:
                logging.getLogger(__name__).error("Teacher graph failed at %s: %s", bar_ts, exc)
                record = {"status": "failed", "stage": "hypergraph", "error": str(exc), "stats_delta": {}}
                bar_store[pos_i] = record
        if record["status"] == "failed":
            failed_bars.append({"bar_index": int(bar_idx), "timestamp": bar_ts.isoformat(),
                                "error": record["error"]})
            previous_record = None
        else:
            outcome = record["outcome"]
            previous_record = record
            w_finals[pos_i] = outcome["w_final"]
            teacher_actions[pos_i] = outcome["discrete_action"]
            consensus_texts[pos_i] = outcome["consensus_text"]
            if is_oos_bar and record.get("counterfactual"):
                cf_grounded_acts.append(outcome["discrete_action"])
                cf_stripped_acts.append(record["counterfactual"]["discrete_action"])
        if is_oos_bar:
            for name, delta in record.get("stats_delta", {}).items():
                setattr(totals, name, getattr(totals, name) + delta)
            totals.rejection_reasons.extend(record.get("rejection_reasons", []))
            oos_counter += 1
    verifier.stats = totals
    if failed_bars:
        payload = dict(config_sha256=cfg.config_sha256, run_id=run_id, status="incomplete",
                       n_bars_processed=n_target, n_failed_bars=len(failed_bars),
                       failed_bars=failed_bars, llm_telemetry=dict(llm.telemetry),
                       universe=macro_fetcher.last_universe_report, ledger_written=False)
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Failure checkpoints are durable. Do not export incomplete teacher arrays
        # for the student to silently treat as cash/neutral training examples.
        raise RuntimeError(f"{len(failed_bars)} teacher bars failed; inspect {out_dir / 'metrics.json'}")

    # Snapshot Table 6.15 OOS citation & hallucination telemetry BEFORE synthetic latency benchmark!
    cite_match_pct = verifier.stats.citation_match_rate * 100.0
    halluc_rej_pct = verifier.stats.hallucination_rejection_rate * 100.0
    cf_g = np.asarray(cf_grounded_acts, dtype=np.int64)
    cf_s = np.asarray(cf_stripped_acts, dtype=np.int64)
    flip_rate_pct = float(np.mean(cf_g != cf_s)) * 100.0 if len(cf_g) > 0 else 0.0

    # Benchmark direct Teacher inference latency (Table 6.14 Teacher row)
    last_day = pd.Timestamp(df_all.index[target_bar_indices[-1]]).strftime("%Y-%m-%d")
    if current_hg is None or current_day_str != last_day:
        saved = day_store[last_day]
        current_hg = DynamicFinancialHypergraph(nodes=saved["nodes"],
            simple_edges=[VerifiedSimpleEdge(**row) for row in saved["edges"]],
            max_hyperedges=cfg.study2.max_hyperedges)
    sample_hg = current_hg
    latency_failure = None
    previous_cache_mode = llm.use_cache
    llm.use_cache = False
    try:
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
    except LLMCallError as exc:
        logging.getLogger(__name__).error("Teacher latency benchmark failed: %s", exc)
        lat_report = None
        latency_failure = str(exc)
    finally:
        llm.use_cache = previous_cache_mode

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
        "run_id": run_id,
        "universe": macro_fetcher.last_universe_report,
        "n_failed_bars": 0,
        "debate_stride": args.debate_stride,
        "counterfactual_stride": args.counterfactual_stride,
        "llm_telemetry": dict(llm.telemetry),
        "llm_telemetry_scope": "current process only; persisted historical calls excluded",
        "diagnostic_run": args.diagnostic_run,
        "table_6_15_study2_rows": {
            "citation_match_rate_pct": cite_match_pct,
            "hallucination_rejection_rate_pct": halluc_rej_pct,
            "evidence_removal_flip_rate_pct": flip_rate_pct,
        },
        "table_6_14_latency": lat_report.to_dict() if lat_report else {"status": "failed", "error": latency_failure},
        "status": "complete" if lat_report else "latency_incomplete",
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.diagnostic_run and lat_report is not None:
        ledger = NumberLedger(run_context=RunContext(stage="study2", flags={
            "llm_simulated": llm.is_simulated_llm, "quick_verify": False}))
        rec_kw = dict(script_path=script_rel, config_sha256=cfg.config_sha256, run_log_path=log_path)
        ledger.record("T6.14_TEACHER_MEAN_MS", "Table 6.14", "Teacher direct inference latency mean", lat_report.teacher_direct.mean_ms, "{:.1f}", "ms", **rec_kw)
        ledger.record("T6.15_CITATION_MATCH_PCT", "Table 6.15", "Citation Match Rate", cite_match_pct, "{:.2f}", "%", **rec_kw)
        ledger.record("T6.15_HALLUCINATION_REJECT_PCT", "Table 6.15", "Hallucination Rejection Rate", halluc_rej_pct, "{:.2f}", "%", **rec_kw)
        ledger.record("T6.15_EVIDENCE_FLIP_PCT", "Table 6.15", "Counterfactual Evidence Removal Flip Rate", flip_rate_pct, "{:.2f}", "%", **rec_kw)

    print(json.dumps(payload["table_6_15_study2_rows"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
