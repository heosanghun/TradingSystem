"""
Explainability Evaluation Metrics Calculator (Chapter VI Sec 12, Table 6.15, p. 77).

Replaces the deleted t-SNE visualization from the 1st defense draft with 5 quantitative,
reproducible explainability metrics (including all 4 `"측정 후 기입"` cells in Table 6.15):
1. `인용 정합률 (Citation Match Rate)`: Fraction of `[Edge ID / TE]` citations in final
   consensus `C_t` matching active `H_t` logs across the test period.
2. `환각 기각률 (Hallucination Rejection Rate)`: Fraction of LLM debate responses rejected
   by `CitationVerifier` due to missing/fabricated Edge IDs or mismatched TE values.
3. `모달 기여도 (SHAP)`: Visual vs `(Tech + Sentiment)` percentage share on Variant D in Period II.
4. `교사-학생 행동 일치율 (Teacher-Student Action Agreement Rate)`: Fraction of out-of-sample
   test bars where `argmax_a π_student(a | X_t^micro) == a_teacher^*`.
5. `근거 제거 시 결정 변화율 (Counterfactual Evidence Removal Decision Flip Rate)`: Fraction of
   test bars where stripping cited hypergraph edges (`strip_evidence=True`) and re-running
   the debate changes the discrete target action `W_final`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:  # annotations only; importing these would load Study 1 and Study 2 code
    from ts.eval.shap_analysis import ModalitySHAPResult
    from ts.study2.citation_verifier import CitationVerificationStats


@dataclass
class Table615XAIMetrics:
    citation_match_rate_pct: float
    hallucination_rejection_rate_pct: float
    shap_visual_pct: float
    shap_tech_plus_senti_pct: float
    teacher_student_agreement_pct: float
    evidence_removal_flip_rate_pct: float
    n_test_bars: int
    n_counterfactual_bars: int

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def compute_table_615_xai_metrics(
    citation_stats: CitationVerificationStats,
    shap_result: ModalitySHAPResult,
    teacher_discrete_actions: np.ndarray,
    student_discrete_actions: np.ndarray,
    grounded_debate_actions: np.ndarray,
    stripped_evidence_actions: np.ndarray,
) -> Table615XAIMetrics:
    """Compute all 5 rows of Table 6.15 in percentage units `[0, 100]`."""
    t_act = np.asarray(teacher_discrete_actions, dtype=np.int64)
    s_act = np.asarray(student_discrete_actions, dtype=np.int64)
    if len(t_act) != len(s_act):
        raise ValueError(f"Teacher ({len(t_act)}) and Student ({len(s_act)}) action length mismatch")

    agreement = float(np.mean(t_act == s_act)) * 100.0 if len(t_act) > 0 else 0.0

    g_act = np.asarray(grounded_debate_actions, dtype=np.int64)
    cf_act = np.asarray(stripped_evidence_actions, dtype=np.int64)
    if len(g_act) != len(cf_act):
        raise ValueError(f"Grounded ({len(g_act)}) vs counterfactual ({len(cf_act)}) length mismatch")

    flip_rate = float(np.mean(g_act != cf_act)) * 100.0 if len(g_act) > 0 else 0.0

    return Table615XAIMetrics(
        citation_match_rate_pct=citation_stats.citation_match_rate * 100.0,
        hallucination_rejection_rate_pct=citation_stats.hallucination_rejection_rate * 100.0,
        shap_visual_pct=shap_result.visual_pct,
        shap_tech_plus_senti_pct=shap_result.tech_plus_sentiment_pct,
        teacher_student_agreement_pct=agreement,
        evidence_removal_flip_rate_pct=flip_rate,
        n_test_bars=int(len(t_act)),
        n_counterfactual_bars=int(len(g_act)),
    )
