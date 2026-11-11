"""Study 2 (Ch. IV): Hypergraph-Grounded Multi-Agent LLM Debate."""

from ts.study2.causality import (
    CandidateEdge,
    discover_candidate_edges,
    granger_var_ftest,
    normalized_transfer_entropy_kde,
)
from ts.study2.citation_verifier import CitationCheckResult, CitationVerificationStats, CitationVerifier
from ts.study2.debate import DebateOutcome, GroundedDebateEngine
from ts.study2.hypergraph import (
    DynamicFinancialHypergraph,
    HyperEdgeRecord,
    VerifiedSimpleEdge,
    build_daily_hypergraph,
)
from ts.study2.llm_client import LLMClient
from ts.study2.prompts import (
    ALPHA_ANALYST_SYSTEM_PROMPT,
    EDGE_VERIFIER_SYSTEM_PROMPT,
    RISK_MANAGER_SYSTEM_PROMPT,
)

__all__ = [
    "CandidateEdge",
    "discover_candidate_edges",
    "granger_var_ftest",
    "normalized_transfer_entropy_kde",
    "CitationCheckResult",
    "CitationVerificationStats",
    "CitationVerifier",
    "DebateOutcome",
    "GroundedDebateEngine",
    "DynamicFinancialHypergraph",
    "HyperEdgeRecord",
    "VerifiedSimpleEdge",
    "build_daily_hypergraph",
    "LLMClient",
    "ALPHA_ANALYST_SYSTEM_PROMPT",
    "EDGE_VERIFIER_SYSTEM_PROMPT",
    "RISK_MANAGER_SYSTEM_PROMPT",
]
