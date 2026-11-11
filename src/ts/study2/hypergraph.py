"""
Dynamic Financial Hypergraph `H_t` Builder & Zhou Random-Walk Operator (Ch. IV Sec 2.3–2.4).

Implements:
- LLM Edge Verification + Eq. (IV.5):
    `w(e) = 0.5 * TE_tilde + 0.5 * LLM_score ∈ [0, 1]`
- Spectral Clustering with Eigengap Heuristic (`<= 20` hyperedges) + 1 Global Market-Factor
  Hyperedge covering all nodes + Self-Lag Aperiodicity
- Hyperedge Weight `w(e) ∈ [0, 1]` defined as the **mean** of constituent simple-edge weights
  (resolving Revised Thesis p. 46 **저자 확인 Box #2** consistently in both code & Table 4.1!)
- Exact Hypergraph Random Walk Transition Matrix & Stationary Distribution (Eq. IV.6 & Prop. 4.1):
    `P = D_v^{-1} H W D_e^{-1} H^T`,   `π(v) = d(v) / Σ_u d(u)`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ts.study2.causality import CandidateEdge
from ts.study2.llm_client import LLMClient
from ts.study2.prompts import EDGE_VERIFIER_SYSTEM_PROMPT, format_edge_verifier_user_prompt


@dataclass(frozen=True)
class VerifiedSimpleEdge:
    edge_id: str
    source: str
    target: str
    te_norm: float
    llm_score: float
    weight: float       # w(e) = 0.5 * te_norm + 0.5 * llm_score ∈ [0, 1]
    sign_corr: float    # >= 0: positive propagation, < 0: danger/tail-risk contagion
    rationale: str


@dataclass(frozen=True)
class HyperEdgeRecord:
    hyperedge_id: str
    nodes: Tuple[str, ...]
    weight: float       # Mean of constituent simple-edge weights ∈ [0, 1]
    is_global_factor: bool = False


class DynamicFinancialHypergraph:
    """Represents daily snapshot `H_t = (V, E_hyper, W)` with simple-edge lookup for citation verification."""

    def __init__(
        self,
        nodes: Sequence[str],
        simple_edges: Sequence[VerifiedSimpleEdge],
        max_hyperedges: int = 20,
    ) -> None:
        self.nodes: List[str] = list(dict.fromkeys(nodes))
        self.node_to_idx: Dict[str, int] = {v: i for i, v in enumerate(self.nodes)}
        self.simple_edges: Dict[str, VerifiedSimpleEdge] = {e.edge_id: e for e in simple_edges}
        self.max_hyperedges = max_hyperedges

        self.hyperedges: List[HyperEdgeRecord] = self._build_spectral_hyperedges()
        self.hyperedge_map: Dict[str, HyperEdgeRecord] = {he.hyperedge_id: he for he in self.hyperedges}
        self.incidence_H, self.weights_W, self.transition_P, self.stationary_pi = (
            self._compute_random_walk_operators()
        )

    def _build_spectral_hyperedges(self) -> List[HyperEdgeRecord]:
        n_v = len(self.nodes)
        edges = list(self.simple_edges.values())
        hyperedges: List[HyperEdgeRecord] = []

        if n_v >= 3 and len(edges) >= 2:
            # Build symmetric affinity matrix A from verified directed edge weights
            adj = np.zeros((n_v, n_v), dtype=np.float64)
            for e in edges:
                u = self.node_to_idx.get(e.source)
                v = self.node_to_idx.get(e.target)
                if u is not None and v is not None and u != v:
                    adj[u, v] = max(adj[u, v], e.weight)
                    adj[v, u] = max(adj[v, u], e.weight)

            active_mask = np.sum(adj, axis=1) > 0.0
            active_indices = np.where(active_mask)[0]

            if len(active_indices) >= 3:
                sub_a = adj[np.ix_(active_indices, active_indices)]
                deg = np.sum(sub_a, axis=1)
                d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
                lap = np.eye(len(active_indices)) - d_inv_sqrt @ sub_a @ d_inv_sqrt
                lap = 0.5 * (lap + lap.T)

                eigvals, eigvecs = np.linalg.eigh(lap)
                max_k = min(self.max_hyperedges, len(active_indices) - 1)
                if max_k >= 2:
                    # Eigengap heuristic over k ∈ [2 .. max_k]
                    gaps = np.diff(eigvals[1 : max_k + 1])
                    k_opt = int(np.argmax(gaps) + 2) if len(gaps) > 0 else 2
                    k_opt = max(2, min(k_opt, self.max_hyperedges))
                else:
                    k_opt = 1

                emb = eigvecs[:, :k_opt]
                row_norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / np.maximum(row_norms, 1e-12)
                km = KMeans(n_clusters=k_opt, n_init=5, random_state=42)
                labels = km.fit_predict(emb)

                he_idx = 1
                for c_id in range(k_opt):
                    member_local = np.where(labels == c_id)[0]
                    if len(member_local) < 2:
                        continue
                    member_nodes = tuple(self.nodes[active_indices[m]] for m in member_local)
                    member_set = set(member_nodes)
                    # Mean of constituent simple-edge weights (Box #2: w(e) ∈ [0, 1])
                    constituent_w = [
                        e.weight
                        for e in edges
                        if e.source in member_set and e.target in member_set
                    ]
                    if not constituent_w:
                        constituent_w = [
                            e.weight
                            for e in edges
                            if e.source in member_set or e.target in member_set
                        ]
                    mean_w = float(np.clip(np.mean(constituent_w) if constituent_w else 0.5, 0.01, 1.0))
                    hyperedges.append(
                        HyperEdgeRecord(
                            hyperedge_id=f"H{he_idx:02d}",
                            nodes=member_nodes,
                            weight=mean_w,
                            is_global_factor=False,
                        )
                    )
                    he_idx += 1
                    if len(hyperedges) >= self.max_hyperedges:
                        break

        # Global market-wide common-factor hyperedge (guarantees connectedness for Prop. 4.1)
        all_w = [e.weight for e in edges]
        global_w = float(np.clip(np.mean(all_w) if all_w else 0.1, 0.05, 1.0))
        hyperedges.append(
            HyperEdgeRecord(
                hyperedge_id="H_GLOBAL",
                nodes=tuple(self.nodes),
                weight=global_w,
                is_global_factor=True,
            )
        )
        return hyperedges

    def _compute_random_walk_operators(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Implements Eq. (IV.6) & Proposition 4.1 (Zhou et al. [59]):
          P_rw = D_v^{-1} H W D_e^{-1} H^T
        With lazy self-lag `(1 - eps_self) P_rw + eps_self I` ensuring strict aperiodicity
        while preserving the exact unique stationary distribution `π(v) = d(v) / Σ_u d(u)`.
        """
        n_v = len(self.nodes)
        n_e = len(self.hyperedges)
        if n_v == 0:
            return np.zeros((0, 0)), np.zeros(0), np.zeros((0, 0)), np.zeros(0)

        H = np.zeros((n_v, n_e), dtype=np.float64)
        w_vec = np.zeros(n_e, dtype=np.float64)
        for j, he in enumerate(self.hyperedges):
            w_vec[j] = he.weight
            for node_name in he.nodes:
                H[self.node_to_idx[node_name], j] = 1.0

        d_v = H @ w_vec                         # d(v) = Σ_e w(e) h(v, e)
        d_e = np.sum(H, axis=0)                 # δ(e) = Σ_v h(v, e)

        inv_dv = np.diag(1.0 / np.maximum(d_v, 1e-12))
        inv_de = np.diag(1.0 / np.maximum(d_e, 1e-12))
        W = np.diag(w_vec)

        p_raw = inv_dv @ H @ W @ inv_de @ H.T
        # Self-lag term (5%) guarantees strict aperiodicity without altering stationary π(v)
        P = 0.95 * p_raw + 0.05 * np.eye(n_v)
        P = P / np.sum(P, axis=1, keepdims=True)

        pi = d_v / max(float(np.sum(d_v)), 1e-12)
        return H, w_vec, P, pi

    def positive_and_danger_edges(
        self, target_asset: str = "BTCUSDT"
    ) -> Tuple[List[VerifiedSimpleEdge], List[VerifiedSimpleEdge]]:
        """Partition verified edges into constructive (`E_pos`) and downside/volatility (`E_neg`)."""
        pos: List[VerifiedSimpleEdge] = []
        neg: List[VerifiedSimpleEdge] = []
        vol_tickers = {"^VIX", "VIXY", "UVXY", "^TNX", "^IRX", "UUP"}

        for e in sorted(self.simple_edges.values(), key=lambda x: x.weight, reverse=True):
            is_vol_shock = (e.source in vol_tickers) or (e.sign_corr < 0.0)
            if is_vol_shock:
                neg.append(e)
            else:
                pos.append(e)
        # Keep target_asset-related edges prioritized at front if present
        pos.sort(key=lambda e: (e.target == target_asset, e.weight), reverse=True)
        neg.sort(key=lambda e: (e.target == target_asset, e.weight), reverse=True)
        return pos, neg


def build_daily_hypergraph(
    nodes: Sequence[str],
    candidates: Sequence[CandidateEdge],
    llm_client: LLMClient,
    news_context: str,
    alpha: float = 0.5,
    max_hyperedges: int = 20,
    edge_week_cache: Optional[Dict[Tuple[str, str, str], Tuple[str, float, str]]] = None,
    iso_week_key: str = "",
) -> DynamicFinancialHypergraph:
    """
    Run LLM Edge Verifier over statistical candidate edges (with optional `(src, dst, iso_week)`
    cache from `구현계획_261111.md` Sec 6.3 WP3), compute Eq. (IV.5) hybrid edge weights,
    and construct `DynamicFinancialHypergraph`.
    """
    verified: List[VerifiedSimpleEdge] = []

    for cand in candidates:
        cache_tuple = (cand.source, cand.target, iso_week_key)
        if edge_week_cache is not None and iso_week_key and cache_tuple in edge_week_cache:
            decision, conf, rationale = edge_week_cache[cache_tuple]
        else:
            user_prompt = format_edge_verifier_user_prompt(
                source_node=cand.source,
                target_node=cand.target,
                te_score=cand.te_norm,
                news_context=news_context,
            )
            resp = llm_client.generate_json(
                role="edge_verifier",
                system_prompt=EDGE_VERIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                context_hint={
                    "source": cand.source,
                    "target": cand.target,
                    "te_score": cand.te_norm,
                    "granger_p": cand.granger_p,
                },
            )
            decision = str(resp.get("decision", "REJECT")).strip().upper()
            conf = float(np.clip(float(resp.get("confidence_score", 0.0)), 0.0, 1.0))
            rationale = str(resp.get("economic_rationale", "")).strip()
            if edge_week_cache is not None and iso_week_key:
                edge_week_cache[cache_tuple] = (decision, conf, rationale)

        if decision == "ACCEPT":
            # Eq. (IV.5): w(e) = alpha * TE_tilde + (1 - alpha) * LLM_score
            w_e = float(np.clip(alpha * cand.te_norm + (1.0 - alpha) * conf, 0.0, 1.0))
            verified.append(
                VerifiedSimpleEdge(
                    edge_id=cand.edge_id,
                    source=cand.source,
                    target=cand.target,
                    te_norm=cand.te_norm,
                    llm_score=conf,
                    weight=w_e,
                    sign_corr=cand.sign_corr,
                    rationale=rationale,
                )
            )

    return DynamicFinancialHypergraph(
        nodes=nodes,
        simple_edges=verified,
        max_hyperedges=max_hyperedges,
    )
