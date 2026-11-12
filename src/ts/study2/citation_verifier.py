"""
Programmatic Citation Verifier & Hallucination Rejector (Ch. IV Sec 4.1 & Table 6.15).

Reads edge citations from LLM JSON rationales, cross-checks them against `H_t`,
rejects fabricated IDs or mismatched TE scores, and accumulates telemetry
for Table 6.15.

Two notations are read as citations:
  - bracketed, as the prompt asks: `[Edge ID: #E01, TE: 0.3412]`, `[Edge #E01]`,
    `[Hyperedge ID: #H02]`
  - bare identifiers of the form the pipeline issues (`E` or `H` followed by digits,
    or `H_GLOBAL`), with a `TE: 0.XXXX` value if one follows before the next identifier:
    `E02 (ETHUSDT -> BTCUSDT, TE: 0.4537)`
The model was measured to use the second form in most answers (T-205). The check itself
is the same for both: the identifier must exist in `H_t` and a quoted TE must match.

A rejection is recorded under one of three reasons, so that a missing citation is not
counted as a fabricated one:
  - `missing`: no citation although edges were available
  - `unknown_id`: an identifier that is not in `H_t`
  - `value_mismatch`: a known identifier quoted with a wrong TE or weight

Table 6.15 rows:
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


# Bare identifiers as issued by `causality.py` (`E01`) and `hypergraph.py` (`H01`, `H_GLOBAL`).
BARE_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])#?(H_GLOBAL|[EH][0-9]{1,4})(?![A-Za-z0-9_])")
TE_VALUE_PATTERN = re.compile(r"TE\s*[=:]?\s*([0-9]*\.?[0-9]+)", flags=re.IGNORECASE)
# A TE value is attached to a bare identifier only if it follows within this many characters.
BARE_TE_LOOKAHEAD_CHARS = 80


@dataclass
class CitationCheckResult:
    is_valid: bool
    cited_ids: List[str]
    matched_ids: List[str]
    reason: str = ""
    category: str = ""


@dataclass
class CitationVerificationStats:
    total_agent_outputs: int = 0
    rejected_agent_outputs: int = 0
    rejected_missing_citation: int = 0
    rejected_unknown_id: int = 0
    rejected_value_mismatch: int = 0
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
        """Citations in order of appearance, each identifier position counted once."""
        text = text or ""
        found: List[Tuple[int, str, float | None]] = []
        covered: List[Tuple[int, int]] = []
        for m in CITATION_PATTERN.finditer(text):
            te_str = m.group(2)
            found.append((m.start(), m.group(1).strip(), float(te_str) if te_str is not None else None))
            covered.append((m.start(), m.end()))

        bare = [m for m in BARE_ID_PATTERN.finditer(text)
                if not any(a <= m.start() < b for a, b in covered)]
        for i, m in enumerate(bare):
            limit = min(len(text), m.end() + BARE_TE_LOOKAHEAD_CHARS)
            if i + 1 < len(bare):
                limit = min(limit, bare[i + 1].start())
            for a, _b in covered:
                if a > m.end():
                    limit = min(limit, a)
            te_m = TE_VALUE_PATTERN.search(text, m.end(), limit)
            found.append((m.start(), m.group(1), float(te_m.group(1)) if te_m else None))

        found.sort(key=lambda item: item[0])
        return [(eid, te) for _pos, eid, te in found]

    def _reject(self, category: str, reason: str, cited: List[str], matched: List[str],
                record_in_stats: bool) -> CitationCheckResult:
        if record_in_stats:
            self.stats.rejected_agent_outputs += 1
            counter = f"rejected_{category}"
            setattr(self.stats, counter, getattr(self.stats, counter) + 1)
            self.stats.rejection_reasons.append(reason)
        return CitationCheckResult(False, cited, matched, reason, category)

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
                return self._reject("missing_citation",
                                    "Missing mandatory edge citation despite non-empty hypergraph.",
                                    [], [], record_in_stats)
            return CitationCheckResult(True, [], [], "No edges available to cite.")

        cited_ids: List[str] = []
        matched_ids: List[str] = []
        for raw_id, cited_te in citations:
            cited_ids.append(raw_id)
            if raw_id in hypergraph.simple_edges:
                edge = hypergraph.simple_edges[raw_id]
                if cited_te is not None and abs(cited_te - edge.te_norm) > self.te_tolerance:
                    return self._reject(
                        "value_mismatch",
                        f"TE mismatch on {raw_id}: cited {cited_te:.4f} vs actual {edge.te_norm:.4f}",
                        cited_ids, matched_ids, record_in_stats)
                matched_ids.append(raw_id)
            elif raw_id in hypergraph.hyperedge_map:
                he = hypergraph.hyperedge_map[raw_id]
                if cited_te is not None and abs(cited_te - he.weight) > self.te_tolerance:
                    return self._reject(
                        "value_mismatch",
                        f"Hyperedge weight mismatch on {raw_id}: cited {cited_te:.4f} vs actual {he.weight:.4f}",
                        cited_ids, matched_ids, record_in_stats)
                matched_ids.append(raw_id)
            else:
                return self._reject("unknown_id",
                                    f"Hallucinated non-existent Edge/Hyperedge ID: {raw_id}",
                                    cited_ids, matched_ids, record_in_stats)

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
