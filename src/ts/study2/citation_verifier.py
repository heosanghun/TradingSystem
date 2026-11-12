"""
Programmatic Citation Verifier & Hallucination Rejector (Ch. IV Sec 4.1 & Table 6.15).

Parses `[Edge ID: #EXX]` / `[Edge #EXX]` / `[Hyperedge ID: #HXX]` and optional
`TE: 0.XXXX` values from LLM JSON rationales, cross-checks them against `H_t`,
rejects fabricated IDs or mismatched TE scores, and accumulates telemetry
for Table 6.15:
  1. `인용 정합률 (Citation Match Rate)`
  2. `환각 기각률 (Hallucination Rejection Rate)`
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from ts.study2.hypergraph import DynamicFinancialHypergraph


# Matches patterns such as `[Edge ID: #E01, TE: 0.3412]`, `[Edge #E01]`, `[Edge ID: E23]`, `[Hyperedge #H02]`
CITATION_PATTERN = re.compile(
    r"\[\s*(?:Hyper)?Edge(?:\s*ID)?\s*:?\s*#?([A-Za-z0-9_]+)"
    r"(?:[^\]]*?TE\s*[=:]\s*([0-9]*\.?[0-9]+))?[^\]]*\]",
    flags=re.IGNORECASE,
)


@dataclass
class CitationCheckResult:
    is_valid: bool
    cited_ids: List[str]
    matched_ids: List[str]
    reason: str = ""


@dataclass
class CitationVerificationStats:
    total_agent_outputs: int = 0
    rejected_agent_outputs: int = 0
    total_consensus_citations: int = 0
    matched_consensus_citations: int = 0
    rejection_reasons: List[str] = field(default_factory=list)

    @property
    def hallucination_rejection_rate(self) -> float:
        """Table 6.15 Row 2: Rejected agent outputs / Total debate agent outputs."""
        if self.total_agent_outputs == 0:
            return 0.0
        return self.rejected_agent_outputs / self.total_agent_outputs

    @property
    def citation_match_rate(self) -> float:
        """Table 6.15 Row 1: Verified citations in final consensus `C_t` / Total citations in `C_t`."""
        if self.total_consensus_citations == 0:
            return 1.0
        return self.matched_consensus_citations / self.total_consensus_citations


class CitationVerifier:
    """Deterministic post-processor enforcing grounded hypergraph citations."""

    def __init__(self, te_tolerance: float = 0.05) -> None:
        self.te_tolerance = te_tolerance
        self.stats = CitationVerificationStats()

    @staticmethod
    def extract_citations(text: str) -> List[Tuple[str, float | None]]:
        out: List[Tuple[str, float | None]] = []
        for m in CITATION_PATTERN.finditer(text or ""):
            eid = m.group(1).strip()
            te_str = m.group(2)
            te_val = float(te_str) if te_str is not None else None
            out.append((eid, te_val))
        return out

    def verify_agent_rationale(
        self,
        rationale_text: str,
        hypergraph: DynamicFinancialHypergraph,
        require_at_least_one_if_nonempty: bool = True,
        record_in_stats: bool = True,
    ) -> CitationCheckResult:
        if record_in_stats:
            self.stats.total_agent_outputs += 1

        citations = self.extract_citations(rationale_text)
        has_simple_edges = len(hypergraph.simple_edges) > 0

        if not citations:
            if require_at_least_one_if_nonempty and has_simple_edges:
                reason = "Missing mandatory [Edge ID: #XX] citation despite non-empty hypergraph."
                if record_in_stats:
                    self.stats.rejected_agent_outputs += 1
                    self.stats.rejection_reasons.append(reason)
                return CitationCheckResult(False, [], [], reason)
            return CitationCheckResult(True, [], [], "No edges available to cite.")

        cited_ids: List[str] = []
        matched_ids: List[str] = []
        for raw_id, cited_te in citations:
            cited_ids.append(raw_id)
            if raw_id in hypergraph.simple_edges:
                edge = hypergraph.simple_edges[raw_id]
                if cited_te is not None and abs(cited_te - edge.te_norm) > self.te_tolerance:
                    reason = (
                        f"TE mismatch on {raw_id}: cited {cited_te:.4f} vs actual {edge.te_norm:.4f}"
                    )
                    if record_in_stats:
                        self.stats.rejected_agent_outputs += 1
                        self.stats.rejection_reasons.append(reason)
                    return CitationCheckResult(False, cited_ids, matched_ids, reason)
                matched_ids.append(raw_id)
            elif raw_id in hypergraph.hyperedge_map:
                he = hypergraph.hyperedge_map[raw_id]
                if cited_te is not None and abs(cited_te - he.weight) > self.te_tolerance:
                    reason = (
                        f"Hyperedge weight mismatch on {raw_id}: cited {cited_te:.4f} vs actual {he.weight:.4f}"
                    )
                    if record_in_stats:
                        self.stats.rejected_agent_outputs += 1
                        self.stats.rejection_reasons.append(reason)
                    return CitationCheckResult(False, cited_ids, matched_ids, reason)
                matched_ids.append(raw_id)
            else:
                reason = f"Hallucinated non-existent Edge/Hyperedge ID: {raw_id}"
                if record_in_stats:
                    self.stats.rejected_agent_outputs += 1
                    self.stats.rejection_reasons.append(reason)
                return CitationCheckResult(False, cited_ids, matched_ids, reason)

        return CitationCheckResult(True, cited_ids, matched_ids, "OK")

    def audit_final_consensus(
        self, consensus_text: str, hypergraph: DynamicFinancialHypergraph
    ) -> Tuple[int, int]:
        """Audit final `C_t` citations for Table 6.15 Row 1 (`인용 정합률`)."""
        citations = self.extract_citations(consensus_text)
        total = len(citations)
        matched = 0
        for raw_id, cited_te in citations:
            if raw_id in hypergraph.simple_edges:
                if cited_te is None or abs(cited_te - hypergraph.simple_edges[raw_id].te_norm) <= self.te_tolerance:
                    matched += 1
            elif raw_id in hypergraph.hyperedge_map:
                if cited_te is None or abs(cited_te - hypergraph.hyperedge_map[raw_id].weight) <= self.te_tolerance:
                    matched += 1
        self.stats.total_consensus_citations += total
        self.stats.matched_consensus_citations += matched
        return matched, total
