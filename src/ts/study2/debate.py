"""
Evidence-Grounded Multi-Agent Debate & Meta PM Synthesis (Ch. IV Sec 3–4, Eqs. IV.7–IV.8).

Implements:
- Up to 3 rounds of `Alpha Analyst` ⇄ `Risk Manager` debate grounded on `H_t`
- Mandatory programmatic citation check with automatic rejection & retry on hallucination
- Meta PM dynamic risk-aversion synthesis (Eqs. IV.7–IV.8):
    `α_risk = σ(β * max_{e ∈ E_neg} w(e) - γ * mean_{e ∈ E_pos} w(e))`,   `β = 8, γ = 6`
    `W_final = (1 - α_risk) * W_analyst + α_risk * W_risk ∈ [-1, 1]`
- Counterfactual Evidence-Stripped Re-runner (`run_debate(..., strip_evidence=True)`)
  for Table 6.15 Row 5 (`근거 제거 시 결정 변화율`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ts.study1.ppo_expert import ACTION_WEIGHTS
from ts.study2.citation_verifier import CitationVerifier
from ts.study2.hypergraph import DynamicFinancialHypergraph, VerifiedSimpleEdge
from ts.study2.llm_client import LLMClient
from ts.study2.prompts import (
    ALPHA_ANALYST_SYSTEM_PROMPT,
    RISK_MANAGER_SYSTEM_PROMPT,
    format_alpha_analyst_user_prompt,
    format_risk_manager_user_prompt,
)


@dataclass
class DebateOutcome:
    w_analyst: float
    w_risk: float
    alpha_risk: float
    w_final: float
    discrete_action: int
    consensus_text: str
    rounds_executed: int
    cited_pos_edges: List[str]
    cited_neg_edges: List[str]
    rebuttal_decision: str


def _sigmoid(x: float) -> float:
    x_clip = float(np.clip(x, -30.0, 30.0))
    return 1.0 / (1.0 + float(np.exp(-x_clip)))


def map_continuous_weight_to_discrete_action(w_final: float) -> int:
    """Nearest 5-action index in `[-1.0, -0.5, 0.0, +0.5, +1.0]`."""
    return int(np.argmin(np.abs(ACTION_WEIGHTS - float(w_final))))


class GroundedDebateEngine:
    """Executes the System 2 multi-agent debate and Meta PM synthesis per 1-hour bar."""

    def __init__(
        self,
        llm_client: LLMClient,
        citation_verifier: Optional[CitationVerifier] = None,
        max_rounds: int = 3,
        max_citation_retries: int = 3,
        beta: float = 8.0,
        gamma: float = 6.0,
    ) -> None:
        self.llm = llm_client
        self.verifier = citation_verifier or CitationVerifier()
        self.max_rounds = max_rounds
        self.max_citation_retries = max_citation_retries
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _serialize_edges(edges: Sequence[VerifiedSimpleEdge], limit: int = 8) -> str:
        items = [
            {
                "edge_id": e.edge_id,
                "source": e.source,
                "target": e.target,
                "te_norm": round(e.te_norm, 4),
                "weight": round(e.weight, 4),
                "rationale": e.rationale,
            }
            for e in edges[:limit]
        ]
        return json.dumps(items, ensure_ascii=False)

    def compute_meta_pm_consensus(
        self,
        w_analyst: float,
        w_risk: float,
        pos_edge_weights: Sequence[float],
        neg_edge_weights: Sequence[float],
    ) -> tuple[float, float]:
        """
        Eqs. (IV.7)–(IV.8):
          α_risk = σ(β * max_{e ∈ E_neg} w(e) - γ * (1/|E_pos|) Σ_{e ∈ E_pos} w(e))
          W_final = (1 - α_risk) * W_analyst + α_risk * W_risk
        """
        max_neg = float(np.max(neg_edge_weights)) if len(neg_edge_weights) > 0 else 0.0
        mean_pos = float(np.mean(pos_edge_weights)) if len(pos_edge_weights) > 0 else 0.0
        alpha_risk = _sigmoid(self.beta * max_neg - self.gamma * mean_pos)
        w_final = float(np.clip((1.0 - alpha_risk) * w_analyst + alpha_risk * w_risk, -1.0, 1.0))
        return alpha_risk, w_final

    def run_bar_debate(
        self,
        hypergraph: DynamicFinancialHypergraph,
        news_headlines_text: str,
        study1_regime_name: str,
        study1_probs: np.ndarray,
        target_asset: str = "BTCUSDT",
        strip_evidence: bool = False,
    ) -> DebateOutcome:
        """
        Run one hourly debate.
        If `strip_evidence=True` (used for Table 6.15 Row 5 counterfactual evaluation),
        all hypergraph edges are masked out (`[]`) so we measure how often removing cited
        hypergraph evidence alters the final discrete trading decision.
        """
        study1_act = int(np.argmax(study1_probs))
        study1_w = float(ACTION_WEIGHTS[study1_act])
        macro_summary = (
            f"Regime={study1_regime_name}, Study1_Ensemble_Probs={[round(float(p), 3) for p in study1_probs]}, "
            f"Study1_Baseline_Weight={study1_w:+.2f}. Headlines:\n{news_headlines_text}"
        )

        if strip_evidence:
            pos_edges: List[VerifiedSimpleEdge] = []
            neg_edges: List[VerifiedSimpleEdge] = []
            pos_json = "[]"
            neg_json = "[]"
        else:
            pos_edges, neg_edges = hypergraph.positive_and_danger_edges(target_asset=target_asset)
            pos_json = self._serialize_edges(pos_edges)
            neg_json = self._serialize_edges(neg_edges)

        w_analyst = max(0.0, study1_w)
        w_risk = min(0.0, study1_w)
        analyst_rationale = ""
        risk_rationale = ""
        rebuttal_decision = "MODIFY"
        cited_pos: List[str] = []
        cited_neg: List[str] = []
        prior_rebuttal = ""
        rounds_done = 0

        for _round in range(1, self.max_rounds + 1):
            rounds_done = _round

            # 1. Alpha Analyst turn with citation verification & retry
            a_prompt = format_alpha_analyst_user_prompt(pos_json, macro_summary, prior_rebuttal)
            a_valid = False
            for retry in range(self.max_citation_retries):
                a_resp = self.llm.generate_json(
                    role="alpha_analyst",
                    system_prompt=ALPHA_ANALYST_SYSTEM_PROMPT,
                    user_prompt=a_prompt,
                    retry_attempt=retry,
                    context_hint={
                        "pos_edges": [
                            {"edge_id": e.edge_id, "te_norm": e.te_norm, "weight": e.weight}
                            for e in pos_edges
                        ],
                        "study1_target_weight": study1_w,
                    },
                )
                weights_d = a_resp.get("Proposed Portfolio Weights", {})
                w_analyst = float(np.clip(float(weights_d.get("Asset", max(0.0, study1_w))), 0.0, 1.0))
                analyst_rationale = str(a_resp.get("Grounded Rationale", ""))

                if strip_evidence:
                    a_valid = True
                    cited_pos = []
                    break

                chk_a = self.verifier.verify_agent_rationale(
                    analyst_rationale,
                    hypergraph,
                    require_at_least_one_if_nonempty=(len(pos_edges) > 0),
                    record_in_stats=True,
                )
                if chk_a.is_valid:
                    a_valid = True
                    cited_pos = chk_a.matched_ids
                    break

            if not a_valid:
                # Safe baseline fallback if all retries hallucinated
                w_analyst = max(0.0, study1_w)
                analyst_rationale = "Fallback to Study 1 baseline after citation verification rejection."
                cited_pos = []

            # 2. Risk Manager turn with citation verification & retry
            analyst_json_str = json.dumps(
                {"w_analyst": w_analyst, "rationale": analyst_rationale}, ensure_ascii=False
            )
            r_prompt = format_risk_manager_user_prompt(analyst_json_str, neg_json)
            r_valid = False
            for retry in range(self.max_citation_retries):
                r_resp = self.llm.generate_json(
                    role="risk_manager",
                    system_prompt=RISK_MANAGER_SYSTEM_PROMPT,
                    user_prompt=r_prompt,
                    retry_attempt=retry,
                    context_hint={
                        "neg_edges": [
                            {"edge_id": e.edge_id, "te_norm": e.te_norm, "weight": e.weight}
                            for e in neg_edges
                        ],
                        "study1_target_weight": study1_w,
                    },
                )
                rebuttal_decision = str(r_resp.get("Rebuttal Decision", "MODIFY")).strip().upper()
                rw_d = r_resp.get("Revised Portfolio Weights", {})
                long_w = float(np.clip(float(rw_d.get("Asset", 0.0)), 0.0, 1.0))
                short_w = float(np.clip(float(rw_d.get("Short", 0.0)), 0.0, 1.0))
                w_risk = float(np.clip(long_w - short_w, -1.0, 1.0))
                risk_rationale = str(r_resp.get("Risk Rationale", ""))

                if strip_evidence:
                    r_valid = True
                    cited_neg = []
                    break

                chk_r = self.verifier.verify_agent_rationale(
                    risk_rationale,
                    hypergraph,
                    require_at_least_one_if_nonempty=(len(neg_edges) > 0),
                    record_in_stats=True,
                )
                if chk_r.is_valid:
                    r_valid = True
                    cited_neg = chk_r.matched_ids
                    break

            if not r_valid:
                rebuttal_decision = "MODIFY"
                w_risk = min(0.0, study1_w)
                risk_rationale = "Defensive cash/baseline fallback after citation verification rejection."
                cited_neg = []

            if rebuttal_decision == "AGREE":
                break
            prior_rebuttal = f"{rebuttal_decision}: {risk_rationale}"

        # 3. Meta PM Consensus (Eqs. IV.7–IV.8)
        pos_w_vals = [
            hypergraph.simple_edges[eid].weight
            for eid in cited_pos
            if eid in hypergraph.simple_edges
        ]
        neg_w_vals = [
            hypergraph.simple_edges[eid].weight
            for eid in cited_neg
            if eid in hypergraph.simple_edges
        ]
        alpha_risk, w_final = self.compute_meta_pm_consensus(
            w_analyst=w_analyst,
            w_risk=w_risk,
            pos_edge_weights=pos_w_vals,
            neg_edge_weights=neg_w_vals,
        )
        disc_act = map_continuous_weight_to_discrete_action(w_final)

        consensus_text = (
            f"[Meta PM Consensus] Regime={study1_regime_name} | "
            f"alpha_risk={alpha_risk:.3f} | W_final={w_final:+.3f} "
            f"(Analyst={w_analyst:+.2f}, Risk={w_risk:+.2f}, Decision={rebuttal_decision}). "
            f"Bullish Evidence: {analyst_rationale} "
            f"Risk Evidence: {risk_rationale}"
        )

        if not strip_evidence:
            self.verifier.audit_final_consensus(consensus_text, hypergraph)

        return DebateOutcome(
            w_analyst=w_analyst,
            w_risk=w_risk,
            alpha_risk=alpha_risk,
            w_final=w_final,
            discrete_action=disc_act,
            consensus_text=consensus_text,
            rounds_executed=rounds_done,
            cited_pos_edges=cited_pos,
            cited_neg_edges=cited_neg,
            rebuttal_decision=rebuttal_decision,
        )
