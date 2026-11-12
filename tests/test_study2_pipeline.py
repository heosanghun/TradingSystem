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

    llm = LLMClient(backend="deterministic_econometric")
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
