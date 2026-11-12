from __future__ import annotations

import numpy as np
import pandas as pd

from ts.study2.causality import discover_candidate_edges, granger_var_ftest, normalized_transfer_entropy_kde
from ts.study2.citation_verifier import CitationVerifier
from ts.study2.debate import GroundedDebateEngine
from ts.study2.hypergraph import build_daily_hypergraph
from ts.study2.llm_client import LLMClient
import pytest


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cached_pairwise_search_matches_reference(seed, tmp_path) -> None:
    from ts.study2.causality import discover_candidate_edges_cached
    frame = pd.DataFrame(np.random.default_rng(seed).normal(size=(180, 5)), columns=list("ABCDE"))
    records = []
    for source in frame:
        for target in frame:
            if source == target:
                continue
            f, p = granger_var_ftest(frame[source].values, frame[target].values, lag=4)
            te = normalized_transfer_entropy_kde(frame[source].values, frame[target].values, lag=4)
            records.append((source, target, f, p, te))
    threshold = np.quantile([row[4] for row in records], .9)
    expected = [row for row in records if row[3] < .05 or (row[4] > threshold and row[4] > 0)]
    path = tmp_path / "day.parquet"
    result = discover_candidate_edges_cached(frame, path, granger_lag=4, te_lag=4)
    assert [(e.source, e.target) for e in result] == [(r[0], r[1]) for r in expected]
    for edge, row in zip(result, expected):
        np.testing.assert_allclose([edge.granger_f, edge.granger_p, edge.te_norm], row[2:], atol=1e-8, rtol=0)
    assert result == discover_candidate_edges_cached(frame, path, granger_lag=4, te_lag=4)
    first_key = pd.read_parquet(path).attrs["input_sha256"]
    changed = frame.copy()
    changed.iloc[-1, 0] += 1
    discover_candidate_edges_cached(changed, path, granger_lag=4, te_lag=4)
    assert pd.read_parquet(path).attrs["input_sha256"] != first_key


def test_pairwise_discovery_handles_duplicate_and_constant_series() -> None:
    rng = np.random.default_rng(123)
    values = rng.normal(size=240)
    frame = pd.DataFrame({"A": values, "duplicate": values, "B": rng.normal(size=240), "constant": 1.})
    expected = []
    for source in ["A", "duplicate", "B"]:
        for target in ["A", "duplicate", "B"]:
            if source != target:
                f, p = granger_var_ftest(frame[source].values, frame[target].values, lag=4)
                te = normalized_transfer_entropy_kde(frame[source].values, frame[target].values, lag=4)
                expected.append((source, target, f, p, te))
    cutoff = np.quantile([r[4] for r in expected], .9)
    selected = [r for r in expected if r[3] < .05 or (r[4] > cutoff and r[4] > 0)]
    result = discover_candidate_edges(frame, granger_lag=4, te_lag=4)
    assert [(r.source, r.target) for r in result] == [(r[0], r[1]) for r in selected]
    for edge, row in zip(result, selected):
        np.testing.assert_allclose([edge.granger_f, edge.granger_p, edge.te_norm], row[2:], atol=1e-8, rtol=0)


def test_independent_close_decision_accounting() -> None:
    from ts.baselines.independent_audit import run_audit
    rng = np.random.default_rng(11)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, .01, 1000)))
    bars = pd.DataFrame({"open": prices, "close": prices, "high": prices * 1.01,
                         "low": prices * .99, "volume": 10000.},
                        index=pd.date_range("2022-01-01", periods=1000, freq="h", tz="UTC"))
    result = run_audit(bars)
    assert all(case["passed"] for case in result["cases"].values())


def test_llm_retries_cache_bypass_and_permanent_failure(monkeypatch, tmp_path) -> None:
    import requests
    from ts.study2.llm_client import LLMCallError
    responses = [requests.Timeout("test timeout"), "broken", '{"decision":"ACCEPT"}',
                 '{"decision":"REJECT"}']
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                return {"choices": [{"message": {"content": value}}]}
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(cache_db_path=tmp_path / "llm.sqlite", retry_backoff_sec=0)
    assert client.generate_json("role", "system", "user")["decision"] == "ACCEPT"
    assert len(calls) == 3 and client.telemetry["retries"] == 2
    assert client.generate_json("role", "system", "user")["decision"] == "ACCEPT"
    assert len(calls) == 3
    assert client.generate_json("role", "system", "user", use_cache=False)["decision"] == "REJECT"
    assert client.generate_json("role", "system", "user")["decision"] == "ACCEPT"
    responses.extend([requests.Timeout("test")] * 3)
    with pytest.raises(LLMCallError):
        client.generate_json("other", "system", "user")
    assert client.telemetry["failed_calls"] == 1


def test_llm_role_schema_is_checked_before_caching(monkeypatch, tmp_path) -> None:
    import requests
    import json
    answers = [{"decision": "ACCEPT"},
               {"decision": "ACCEPT", "confidence_score": .8, "economic_rationale": "Test rationale"}]
    def post(*args, **kwargs):
        value = answers.pop(0)
        class Response:
            def raise_for_status(self):
                return None
            def json(self):
                return {"choices": [{"message": {"content": json.dumps(value)}}]}
        return Response()
    monkeypatch.setattr(requests, "post", post)
    client = LLMClient(cache_db_path=tmp_path / "cache.sqlite", retry_backoff_sec=0)
    assert client.generate_json("edge_verifier", "s", "u")["confidence_score"] == .8
    assert client.telemetry["failed_attempts"] == 1
    assert client.generate_json("edge_verifier", "s", "u")["confidence_score"] == .8
    assert client.telemetry["cache_hits"] == 1


def test_bar_checkpoint_restart_and_explicit_failures(tmp_path) -> None:
    from ts.study2.persistent_state import SQLiteJSONMap, run_checkpointed_bar
    from ts.study2.llm_client import LLMCallError
    from ts.study2.citation_verifier import CitationVerificationStats
    from ts.study2.debate import DebateOutcome
    class Engine:
        def __init__(self):
            from types import SimpleNamespace
            self.verifier = SimpleNamespace(stats=CitationVerificationStats())
            self.calls = 0
        def run_bar_debate(self, index, strip_evidence=False):
            self.calls += 1
            if index == 2:
                raise LLMCallError("injected failure")
            self.verifier.stats.total_agent_outputs += 1
            return DebateOutcome(.5, 0, .5, .25, 2, "test", 1, [], [], "AGREE")
    path = tmp_path / "state.sqlite"
    first = SQLiteJSONMap(path, "run")
    engine = Engine()
    saved = run_checkpointed_bar(first, 0, engine, {"index": 0}, True)
    assert engine.calls == 2
    restarted = SQLiteJSONMap(path, "run")
    assert run_checkpointed_bar(restarted, 0, engine, {"index": 0}, True) == saved
    assert engine.calls == 2
    for index in [1, 2, 3]:
        run_checkpointed_bar(restarted, index, engine, {"index": index})
    assert restarted[2]["status"] == "failed"
    assert restarted[3]["status"] == "ok"
    assert sum(row["status"] == "failed" for row in restarted.values()) == 1
    fresh = SQLiteJSONMap(tmp_path / "fresh.sqlite", "run")
    other = Engine()
    for index in range(4):
        run_checkpointed_bar(fresh, index, other, {"index": index}, index == 0)
    assert dict(fresh) == dict(restarted)
    weekly = SQLiteJSONMap(path, "weeks")
    weekly[("BTC", "ETH", "2022-W01")] = ["ACCEPT", .8, "test"]
    assert SQLiteJSONMap(path, "weeks")[("BTC", "ETH", "2022-W01")][0] == "ACCEPT"


def test_teacher_script_resumes_after_interruption(monkeypatch, tmp_path) -> None:
    import importlib.util
    import json
    import shutil
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from ts.config import SystemConfig
    spec = importlib.util.spec_from_file_location("cx_study2_runner", Path("scripts/run_study2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "LLMClient", EconometricTestClient)
    cfg = SystemConfig.from_json()
    monkeypatch.setattr(runner.SystemConfig, "from_json", lambda *_: cfg)
    index = pd.date_range("2022-01-01", periods=12, freq="h", tz="UTC")
    bars = pd.DataFrame({"close": np.arange(12.) + 100}, index=index)
    monkeypatch.setattr(runner.BinanceDataFetcher, "fetch_ohlcv", lambda *a, **k: (bars, {}))
    monkeypatch.setattr(runner.MacroAssetFetcher, "build_aligned_60_asset_returns",
                        lambda *a, **k: pd.DataFrame({"BTCUSDT": np.arange(12.)}, index=index))
    monkeypatch.setattr(runner, "discover_candidate_edges_cached", lambda *a, **k: [])
    stage = SimpleNamespace(mean_ms=1., median_ms=1., p99_ms=1.)
    latency = SimpleNamespace(total=SimpleNamespace(n_runs=2, mean_ms=1., median_ms=1., p99_ms=1.),
                              stages={"DEBATE": stage}, to_dict=lambda: {"test": True})
    monkeypatch.setattr(runner, "benchmark_staged_pipeline", lambda *a, **k: latency)
    first = tmp_path / "interrupted"
    folder = first / "runs" / "study1"
    folder.mkdir(parents=True)
    np.savez(folder / "study1_oos_signals.npz", teacher_bar_indices=np.arange(12),
             teacher_regimes=np.full(12, 2), teacher_ens_probs=np.full((12, 5), .2),
             teacher_is_oos=np.ones(12, dtype=bool))
    news = first / "news.csv"
    pd.DataFrame({"published_at": [index[0].isoformat()],
                  "headline": ["Test fixture headline"], "source": ["test"]}).to_csv(news, index=False)
    uninterrupted = tmp_path / "uninterrupted"
    shutil.copytree(first, uninterrupted)

    def invoke(root):
        monkeypatch.setattr(runner, "PROJECT_ROOT", root)
        monkeypatch.setattr(sys, "argv", ["runner", "--news-file", str(root / "news.csv"),
                            "--llm-backend", "openai_compat", "--diagnostic-run"])
        runner.main()

    original = runner.run_checkpointed_bar
    def interrupted(store, key, *args, **kwargs):
        result = original(store, key, *args, **kwargs)
        if key == 3:
            raise KeyboardInterrupt("test process interruption")
        return result
    monkeypatch.setattr(runner, "run_checkpointed_bar", interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(first)
    monkeypatch.setattr(runner, "run_checkpointed_bar", original)
    invoke(first)
    invoke(uninterrupted)
    a = first / "runs/study2/diagnostic"
    b = uninterrupted / "runs/study2/diagnostic"
    with np.load(a / "teacher_signals.npz") as x, np.load(b / "teacher_signals.npz") as y:
        for field in x.files:
            np.testing.assert_array_equal(x[field], y[field])
    assert json.loads((a / "consensus_texts.json").read_text()) == json.loads((b / "consensus_texts.json").read_text())
    assert json.loads((a / "metrics.json").read_text())["table_6_15_study2_rows"] == json.loads(
        (b / "metrics.json").read_text())["table_6_15_study2_rows"]


def test_hourly_macro_alignment_and_exclusion_reporting(tmp_path) -> None:
    from ts.data.macro_fetcher import MacroAssetFetcher
    observed = pd.date_range("2022-01-01 08:30", periods=3, freq="h", tz="UTC")
    fixture = pd.DataFrame({"close": [100., 110., 121.]}, index=observed)
    fixture.attrs.update(resolution="1h", timestamp_semantics="bar_close", source="synthetic test fixture")
    for symbol in ["a", "b"]:
        fixture.to_parquet(tmp_path / f"{symbol}_1h.parquet")
    daily = fixture.copy()
    daily.attrs["resolution"] = "1d"
    daily.to_parquet(tmp_path / "daily_1h.parquet")
    clock = pd.date_range("2022-01-01 09:00", periods=4, freq="h", tz="UTC")
    fetcher = MacroAssetFetcher(tmp_path)
    first = fetcher.build_aligned_60_asset_returns(clock, crypto_symbols=[],
                                                  macro_symbols=["A", "B", "DAILY", "MISSING"])
    np.testing.assert_allclose(first.A, [0., np.log(1.1), np.log(1.1), 0.])
    assert [x["symbol"] for x in fetcher.last_universe_report["included"]] == ["A", "B"]
    assert [x["symbol"] for x in fetcher.last_universe_report["excluded"]] == ["DAILY", "MISSING"]
    fixture.loc[observed[-1], "close"] = 999.
    fixture.to_parquet(tmp_path / "a_1h.parquet")
    second = fetcher.build_aligned_60_asset_returns(clock, crypto_symbols=[], macro_symbols=["A", "B"])
    np.testing.assert_array_equal(first.A.iloc[:2], second.A.iloc[:2])
    with pytest.raises(FileNotFoundError, match="fallback is disabled"):
        fetcher.fetch_us_control_asset("SPY")


def test_explicit_daily_us_controls_follow_dst(monkeypatch) -> None:
    from ts.data.macro_fetcher import MacroAssetFetcher
    import ts.data.macro_fetcher as module
    daily = pd.DataFrame({"Close": [100., 101.]}, index=pd.to_datetime(["2022-03-11", "2022-03-14"]))
    monkeypatch.setattr(module.yf, "download", lambda *a, **k: daily.copy())
    result = MacroAssetFetcher().fetch_us_control_asset("SPY", "2022-03-01", "2022-03-20", prefer_hourly=False)
    assert list(result.index.hour) == [21, 20]
    assert result.attrs["resolution"] == "1d"


def test_news_preparation_reports_coverage_and_rejects_ambiguous_times(tmp_path) -> None:
    import runpy
    import json
    prepare = runpy.run_path("data/collect_inputs.py")["prepare_news"]
    source = tmp_path / "news.csv"
    output = tmp_path / "news.parquet"
    pd.DataFrame({"published_at": ["2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"],
                  "headline": ["Test news item", "Test news item"], "source": ["fixture", "fixture"]}).to_csv(source, index=False)
    report = prepare(source, output, "test://fixture", "test data", "2022-01-01", "2022-01-31")
    assert report["deduplicated_rows"] == 1
    assert report["monthly_counts"] == {"2022-01": 1}
    missing_output = tmp_path / "missing.parquet"
    with pytest.raises(ValueError, match="empty months"):
        prepare(source, missing_output, "test://fixture", "test data", "2022-01-01", "2022-02-28")
    assert not missing_output.exists()
    assert json.loads(missing_output.with_suffix(".coverage.json").read_text())["empty_months"] == ["2022-02"]
    frame = pd.read_csv(source)
    frame["published_at"] = ["2022-01-01", "2022-01-02"]
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="timezone"):
        prepare(source, output, "test://fixture", "test data")


def test_snapshot_trade_adapter_uses_only_prior_receipts() -> None:
    from ts.data.tardis_adapter import snapshot_trade_features
    book = {"local_timestamp": [100, 200, 300]}
    for level in range(10):
        book[f"bids[{level}].price"] = [100-level] * 3
        book[f"asks[{level}].price"] = [101+level] * 3
        book[f"bids[{level}].amount"] = [10.] * 3
        book[f"asks[{level}].amount"] = [10.] * 3
    trades = pd.DataFrame({"local_timestamp": [50, 100, 150, 250, 350],
                           "amount": [1., 2., 3., 4., 5.],
                           "side": ["buy", "sell", "buy", "sell", "buy"]})
    _, features = snapshot_trade_features(pd.DataFrame(book), trades)
    # At 100, the trade timestamped exactly 100 is excluded.
    np.testing.assert_allclose(features[:, 30], np.log1p([1., 5., 4.]))
    np.testing.assert_allclose(features[:, 31], [1., .2, -1.])
    changed = trades.copy()
    changed.loc[changed.local_timestamp >= 250, "amount"] *= 100
    _, later = snapshot_trade_features(pd.DataFrame(book), changed)
    np.testing.assert_array_equal(features[:2], later[:2])


def test_granger_and_kde_transfer_entropy_detect_planted_lead_lag() -> None:
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0.0, 1.0, size=n)
    y = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        y[t] = 0.75 * x[t - 1] + 0.2 * rng.normal(0.0, 1.0)
    z = rng.normal(0.0, 1.0, size=n)

    f_xy, p_xy = granger_var_ftest(x, y, lag=6)
    _, p_zy = granger_var_ftest(z, y, lag=6)
    te_xy = normalized_transfer_entropy_kde(x, y, lag=6)
    te_zy = normalized_transfer_entropy_kde(z, y, lag=6)

    assert f_xy > 5.0 and p_xy < 1e-4
    assert p_zy > 0.01
    assert 0.0 < te_xy <= 1.0
    assert te_xy > te_zy


def test_hypergraph_stationary_distribution_and_citation_verification() -> None:
    rng = np.random.default_rng(99)
    n = 200
    vix = rng.normal(0, 1, size=n)
    btc = np.zeros(n)
    eth = np.zeros(n)
    spy = rng.normal(0, 1, size=n)
    for t in range(1, n):
        btc[t] = -0.65 * vix[t - 1] + 0.55 * spy[t - 1] + 0.3 * rng.normal()
        eth[t] = 0.70 * btc[t - 1] + 0.3 * rng.normal()

    df = pd.DataFrame({"BTCUSDT": btc, "ETHUSDT": eth, "^VIX": vix, "SPY": spy})
    cands = discover_candidate_edges(df, granger_lag=4, te_lag=4)
    assert len(cands) >= 2

    llm = EconometricTestClient()
    hg = build_daily_hypergraph(
        nodes=list(df.columns),
        candidates=cands,
        llm_client=llm,
        news_context="VIX spikes on macro tightening.",
    )

    # Box #2 verification: every hyperedge weight w(e) lies strictly in [0, 1]
    for he in hg.hyperedges:
        assert 0.0 <= he.weight <= 1.0

    # Proposition 4.1 verification: P is row-stochastic and π^T P == π^T
    assert np.allclose(hg.transition_P.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(hg.stationary_pi @ hg.transition_P, hg.stationary_pi, atol=1e-6)

    # CitationVerifier rejects fabricated IDs and mismatched TE scores
    verifier = CitationVerifier(te_tolerance=0.05)
    bad_res = verifier.verify_agent_rationale(
        "Bullish breakout supported by [Edge ID: #E999, TE: 0.8800].", hg
    )
    assert not bad_res.is_valid
    assert verifier.stats.rejected_agent_outputs == 1

    # GroundedDebateEngine + counterfactual strip_evidence run
    engine = GroundedDebateEngine(llm_client=llm, citation_verifier=verifier, max_rounds=2)
    out_grounded = engine.run_bar_debate(
        hypergraph=hg,
        news_headlines_text="[2022-05-09 10:00 UTC] Macro selloff deepens",
        study1_regime_name="Bear",
        study1_probs=np.array([0.1, 0.2, 0.2, 0.3, 0.2], dtype=np.float32),
        target_asset="BTCUSDT",
        strip_evidence=False,
    )
    assert -1.0 <= out_grounded.w_final <= 1.0
    assert 0 <= out_grounded.discrete_action <= 4
    assert verifier.stats.citation_match_rate == 1.0

    out_stripped = engine.run_bar_debate(
        hypergraph=hg,
        news_headlines_text="[2022-05-09 10:00 UTC] Macro selloff deepens",
        study1_regime_name="Bear",
        study1_probs=np.array([0.1, 0.2, 0.2, 0.3, 0.2], dtype=np.float32),
        target_asset="BTCUSDT",
        strip_evidence=True,
    )
    assert -1.0 <= out_stripped.w_final <= 1.0


class EconometricTestClient(LLMClient):
    """Deterministic teacher fixture; never used by production entry points."""
    def __init__(self, backend="deterministic_econometric", **kwargs):
        super().__init__(backend="openai_compat", **kwargs)
        self.backend = "deterministic_econometric"
        self.is_simulated_llm = True

    def _generate_json_once(self, role, system_prompt, user_prompt, retry_attempt=0,
                            context_hint=None, use_cache=True):
        return self._deterministic_econometric_response(role, context_hint or {})

    @staticmethod
    def _deterministic_econometric_response(role: str, ctx: dict) -> dict:
        """
        Deterministic, citation-faithful econometric rule generator used ONLY in automated
        tests (`backend='deterministic_econometric'`).
        """
        if role == "edge_verifier":
            te = float(ctx.get("te_score", 0.0))
            p_gc = float(ctx.get("granger_p", 1.0))
            accept = (te >= 0.02) or (p_gc < 0.05)
            conf = float(min(0.95, max(0.35, 0.5 * te + (0.4 if p_gc < 0.05 else 0.1))))
            return {
                "decision": "ACCEPT" if accept else "REJECT",
                "confidence_score": round(conf, 4),
                "economic_rationale": (
                    f"Cross-asset transmission from {ctx.get('source', 'X')} to {ctx.get('target', 'Y')} "
                    f"is supported by statistical information flow. Liquidity spillover channel is plausible."
                ),
            }

        if role == "alpha_analyst":
            pos_edges = ctx.get("pos_edges", [])
            ens_w = float(ctx.get("study1_target_weight", 0.0))
            asset_w = float(min(1.0, max(0.0, 0.5 + 0.5 * ens_w)))
            if pos_edges:
                e0 = pos_edges[0]
                cite = f"[Edge ID: #{e0['edge_id']}, TE: {e0['te_norm']:.4f}]"
            else:
                cite = "No active positive edge available."
            return {
                "Proposed Portfolio Weights": {"Asset": round(asset_w, 4), "Cash": round(1.0 - asset_w, 4)},
                "Grounded Rationale": f"{cite} indicates constructive momentum transmission aligned with ensemble baseline.",
            }

        if role == "risk_manager":
            neg_edges = ctx.get("neg_edges", [])
            ens_w = float(ctx.get("study1_target_weight", 0.0))
            if neg_edges:
                e0 = neg_edges[0]
                max_w = float(e0.get("weight", 0.0))
                cite = f"[Edge ID: #{e0['edge_id']}, TE: {e0['te_norm']:.4f}]"
                if max_w > 0.65:
                    dec = "STRONG REJECT"
                    short_w = min(1.0, round(max_w, 4))
                    asset_w = 0.0
                elif max_w > 0.35:
                    dec = "MODIFY"
                    short_w = round(0.5 * max_w, 4)
                    asset_w = 0.0
                else:
                    dec = "AGREE"
                    short_w = 0.0
                    asset_w = max(0.0, round(ens_w, 4))
            else:
                dec = "AGREE"
                short_w = 0.0
                asset_w = max(0.0, round(ens_w, 4))
                cite = "No active tail-risk edge triggered."
            cash_w = max(0.0, round(1.0 - asset_w - short_w, 4))
            return {
                "Rebuttal Decision": dec,
                "Revised Portfolio Weights": {"Asset": asset_w, "Cash": cash_w, "Short": short_w},
                "Risk Rationale": f"{cite} governs downside contagion assessment and drawdown protection.",
            }

        raise ValueError(f"Unknown deterministic role: {role}")


def test_citation_notations_and_rejection_categories() -> None:
    from ts.study2.hypergraph import DynamicFinancialHypergraph, VerifiedSimpleEdge

    edges = [VerifiedSimpleEdge(edge_id=f"E0{k}", source="ETHUSDT", target="BTCUSDT", te_norm=te,
                                llm_score=0.8, weight=0.5, sign_corr=1.0, rationale="r")
             for k, te in ((1, 0.30), (2, 0.4537))]
    hg = DynamicFinancialHypergraph(nodes=["BTCUSDT", "ETHUSDT"], simple_edges=edges)
    v = CitationVerifier(te_tolerance=0.05)

    # Bare identifiers with a following TE value are read like bracketed citations.
    assert v.extract_citations("E02 (ETHUSDT -> BTCUSDT, TE: 0.4537) and [Edge ID: #E01, TE: 0.3000]") == [
        ("E02", 0.4537), ("E01", 0.30)]
    # Ticker symbols and ordinary words are not identifiers.
    assert v.extract_citations("ETHUSDT leads; THE12 and E-commerce are not edge IDs") == []

    assert v.verify_agent_rationale("E02 (ETHUSDT -> BTCUSDT, TE: 0.4537) supports longs.", hg).is_valid
    assert v.verify_agent_rationale("Momentum is strong.", hg).category == "missing_citation"
    assert v.verify_agent_rationale("E77 (TE: 0.40) supports longs.", hg).category == "unknown_id"
    assert v.verify_agent_rationale("E02 (TE: 0.9000) supports longs.", hg).category == "value_mismatch"
    st = v.stats
    assert (st.total_agent_outputs, st.rejected_agent_outputs) == (4, 3)
    assert (st.rejected_missing_citation, st.rejected_unknown_id, st.rejected_value_mismatch) == (1, 1, 1)


def test_edge_verifier_top_k_and_counts() -> None:
    from ts.study2.causality import CandidateEdge
    import dataclasses

    fields = {f.name for f in dataclasses.fields(CandidateEdge)}
    def cand(i, te):
        kw = dict(edge_id=f"E{i:02d}", source=f"S{i}", target="BTCUSDT", te_norm=te, granger_p=0.01, sign_corr=1.0)
        kw.update({name: 0.0 for name in fields - set(kw)})
        return CandidateEdge(**kw)

    cands = [cand(1, 0.10), cand(2, 0.90), cand(3, 0.50), cand(4, 0.70)]
    llm = EconometricTestClient()
    hg = build_daily_hypergraph(nodes=["BTCUSDT", "S1", "S2", "S3", "S4"], candidates=cands,
                                llm_client=llm, news_context="n", max_candidates=2)
    assert hg.verification_counts == {"candidates": 4, "sent_to_verifier": 2, "accepted": 2}
    assert set(hg.simple_edges) == {"E02", "E04"}


def test_ollama_call_settings_and_truncation(monkeypatch) -> None:
    import ts.study2.llm_client as mod

    sent = []
    replies = [
        {"message": {"content": "{"}, "done_reason": "length", "eval_count": 256, "prompt_eval_count": 100},
        {"message": {"content": '{"decision": "ACCEPT", "confidence_score": 0.7, "economic_rationale": "x"}'},
         "done_reason": "stop", "eval_count": 40, "prompt_eval_count": 100},
    ]

    class Resp:
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body

    def fake_post(url, json=None, timeout=None):
        sent.append(json)
        return Resp(replies[len(sent) - 1])

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    client = LLMClient(backend="ollama", model_name="m", endpoint_url="http://localhost:11434/api/chat")
    out = client.generate_json("edge_verifier", "sys", "user")
    assert out["decision"] == "ACCEPT"
    assert len(sent) == 2 and client.telemetry["truncated_responses"] == 1
    assert sent[0]["options"]["num_ctx"] == 4096 and sent[0]["options"]["num_predict"] == 256
    assert sent[0]["messages"][-1] == {"role": "assistant", "content": mod.EMPTY_THINK_PREFILL}
