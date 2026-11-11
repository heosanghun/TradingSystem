from __future__ import annotations

import numpy as np
import pandas as pd

from ts.study2.causality import discover_candidate_edges, granger_var_ftest, normalized_transfer_entropy_kde
from ts.study2.citation_verifier import CitationVerifier
from ts.study2.debate import GroundedDebateEngine
from ts.study2.hypergraph import build_daily_hypergraph
from ts.study2.llm_client import LLMClient


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
