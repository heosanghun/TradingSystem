"""Teacher interfaces with lazy imports."""
from importlib import import_module

_EXPORTS = {
    'CandidateEdge': 'ts.study2.causality',
    'discover_candidate_edges': 'ts.study2.causality',
    'granger_var_ftest': 'ts.study2.causality',
    'normalized_transfer_entropy_kde': 'ts.study2.causality',
    'CitationCheckResult': 'ts.study2.citation_verifier',
    'CitationVerificationStats': 'ts.study2.citation_verifier',
    'CitationVerifier': 'ts.study2.citation_verifier',
    'DebateOutcome': 'ts.study2.debate',
    'GroundedDebateEngine': 'ts.study2.debate',
    'DynamicFinancialHypergraph': 'ts.study2.hypergraph',
    'HyperEdgeRecord': 'ts.study2.hypergraph',
    'VerifiedSimpleEdge': 'ts.study2.hypergraph',
    'build_daily_hypergraph': 'ts.study2.hypergraph',
    'LLMClient': 'ts.study2.llm_client',
    'ALPHA_ANALYST_SYSTEM_PROMPT': 'ts.study2.prompts',
    'EDGE_VERIFIER_SYSTEM_PROMPT': 'ts.study2.prompts',
    'RISK_MANAGER_SYSTEM_PROMPT': 'ts.study2.prompts',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
