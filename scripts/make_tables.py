"""
WP5 — Automatic Chapter VI Table Compiler & 27-Blank Audit Generator (`scripts/make_tables.py`).

Reads `ledger/numbers.csv` and emits:
1. `runs/tables/chapter6_tables.md` (GitHub-Flavored Markdown for every empirical table in Ch. VI)
2. `runs/tables/blank_27_audit.json` (Explicit status check of all 27 `"측정 후 기입"` cells in `2_개정본`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from ts.config import PROJECT_ROOT
from ts.ledger import DEFAULT_LEDGER_PATH, NumberLedger


# Exact 27 `"측정 후 기입"` cells in `[개정본] 박사논문_개정본_본보기_261110.pdf`
REQUIRED_27_BLANK_IDS: Tuple[Tuple[str, str, str], ...] = (
    # Table 6.8 (3 cells)
    ("T6.8_LUNA_STATIC_ENS_MDD", "Table 6.8", "루나·금리 인상 — Static Ensemble RL 손실률"),
    ("T6.8_LUNA_PROPOSED_MDD", "Table 6.8", "루나·금리 인상 — 제안 모델 손실률"),
    ("T6.8_FTX_PROPOSED_MDD", "Table 6.8", "FTX 파산 — 제안 모델 손실률"),
    # Table 6.11 (6 cells)
    ("T6.11_LAMBDA3_0_0_SHARPE", "Table 6.11", "λ_3 = 0.0 Phase 2 Sharpe"),
    ("T6.11_LAMBDA3_0_1_SHARPE", "Table 6.11", "λ_3 = 0.1 Phase 2 Sharpe"),
    ("T6.11_LAMBDA3_0_2_SHARPE", "Table 6.11", "λ_3 = 0.2 Phase 2 Sharpe"),
    ("T6.11_LAMBDA3_0_3_SHARPE", "Table 6.11", "λ_3 = 0.3 Phase 2 Sharpe"),
    ("T6.11_LAMBDA3_0_4_SHARPE", "Table 6.11", "λ_3 = 0.4 Phase 2 Sharpe"),
    ("T6.11_LAMBDA3_0_5_SHARPE", "Table 6.11", "λ_3 = 0.5 Phase 2 Sharpe"),
    # Table 6.12 (5 cells)
    ("T6.12_TEMP_1_0_SHARPE", "Table 6.12", "앙상블 온도 T = 1 Phase 2 Sharpe"),
    ("T6.12_TEMP_5_0_SHARPE", "Table 6.12", "앙상블 온도 T = 5 Phase 2 Sharpe"),
    ("T6.12_TEMP_10_0_SHARPE", "Table 6.12", "앙상블 온도 T = 10 Phase 2 Sharpe"),
    ("T6.12_TEMP_20_0_SHARPE", "Table 6.12", "앙상블 온도 T = 20 Phase 2 Sharpe"),
    ("T6.12_TEMP_static_1_over_N_SHARPE", "Table 6.12", "앙상블 온도 1/N(정적) Phase 2 Sharpe"),
    # Table 6.13 (9 cells)
    ("T6.13_EXP_3_SHARPE", "Table 6.13", "국면당 전문가 K = 3 Sharpe"),
    ("T6.13_EXP_3_SORTINO", "Table 6.13", "국면당 전문가 K = 3 Sortino"),
    ("T6.13_EXP_3_MDD", "Table 6.13", "국면당 전문가 K = 3 MDD"),
    ("T6.13_EXP_5_SHARPE", "Table 6.13", "국면당 전문가 K = 5 Sharpe"),
    ("T6.13_EXP_5_SORTINO", "Table 6.13", "국면당 전문가 K = 5 Sortino"),
    ("T6.13_EXP_5_MDD", "Table 6.13", "국면당 전문가 K = 5 MDD"),
    ("T6.13_EXP_7_SHARPE", "Table 6.13", "국면당 전문가 K = 7 Sharpe"),
    ("T6.13_EXP_7_SORTINO", "Table 6.13", "국면당 전문가 K = 7 Sortino"),
    ("T6.13_EXP_7_MDD", "Table 6.13", "국면당 전문가 K = 7 MDD"),
    # Table 6.15 (4 cells)
    ("T6.15_CITATION_MATCH_PCT", "Table 6.15", "인용 정합률 (%)"),
    ("T6.15_HALLUCINATION_REJECT_PCT", "Table 6.15", "환각 기각률 (%)"),
    ("T6.15_TEACHER_STUDENT_AGREE_PCT", "Table 6.15", "교사-학생 행동 일치율 (%)"),
    ("T6.15_EVIDENCE_FLIP_PCT", "Table 6.15", "근거 제거 시 결정 변화율 (%)"),
)


# Stages of one teacher decision, matching the description under Table 6.14.
TEACHER_LATENCY_STAGES: Tuple[Tuple[str, str], ...] = (
    ("뉴스 검색", "NEWS"),
    ("전이 엔트로피 재계산", "TE"),
    ("하이퍼그래프 갱신", "GRAPH"),
    ("근거 기반 토론", "DEBATE"),
)


def _fmt(rows: Dict[str, Dict[str, str]], nid: str, suffix: str = "") -> str:
    if nid not in rows:
        return "측정 대기"
    val = rows[nid].get("formatted_value", "")
    return f"{val}{suffix}"


def compile_markdown_tables(ledger_rows: Dict[str, Dict[str, str]]) -> Tuple[str, Dict[str, object]]:
    lines: List[str] = []
    lines.append("# 제VI장 실측 결과 표 자동 생성본 (`ledger/numbers.csv` 기반)\n")

    # Table 6.2
    lines.append("## 표 6.2. 로그 수익률 기초 통계량 (2021.10~2023.12)\n")
    lines.append("| 자산 | 표본 수 ($n$) | 평균 | 표준편차 | 최솟값 | 최댓값 | 첨도 (Pearson) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for label, k in [("BTC/USDT", "BTC"), ("S&P 500 (SPY)", "SPY"), ("Gold (GLD)", "GLD")]:
        lines.append(
            f"| {label} | {_fmt(ledger_rows, f'T6.2_{k}_N')} | {_fmt(ledger_rows, f'T6.2_{k}_MEAN')} | "
            f"{_fmt(ledger_rows, f'T6.2_{k}_STD')} | {_fmt(ledger_rows, f'T6.2_{k}_MIN')} | "
            f"{_fmt(ledger_rows, f'T6.2_{k}_MAX')} | {_fmt(ledger_rows, f'T6.2_{k}_KURT')} |"
        )
    lines.append("")
    if any(f"T6.2_{k}_N" not in ledger_rows for k in ("SPY", "GLD")):
        lines.append("주: 이 표는 1시간 봉 기준이다. SPY와 GLD는 1시간 봉을 확보한 경우에만 기록한다.")
        lines.append("")

    # Table 6.6 & Table 6.7
    for ph, title in [(1, "표 6.6. Phase 1 성과 비교 (고정 비용 0.05%)"), (2, "표 6.7. Phase 2 성과 비교 (ATR 동적 비용)")]:
        tno = 5 + ph
        lines.append(f"## {title}\n")
        lines.append("| 모델 | 누적 수익률 (%) | CAGR (%) | Sharpe | Sortino | MDD (%) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for m_label, m_key in [
            ("Buy & Hold", "BH"),
            ("Single PPO", "SINGLE_PPO"),
            ("Static Ensemble RL", "STATIC_ENS"),
            ("Meta-Learning DRL", "MAML"),
            ("제안 모델 (Student 1D-CNN)", "PROPOSED"),
        ]:
            lines.append(
                f"| {m_label} | {_fmt(ledger_rows, f'T6.{tno}_{m_key}_CUMRET', '%')} | "
                f"{_fmt(ledger_rows, f'T6.{tno}_{m_key}_CAGR', '%')} | "
                f"{_fmt(ledger_rows, f'T6.{tno}_{m_key}_SHARPE')} | "
                f"{_fmt(ledger_rows, f'T6.{tno}_{m_key}_SORTINO')} | "
                f"{_fmt(ledger_rows, f'T6.{tno}_{m_key}_MDD', '%')} |"
            )
        lines.append("")

    # Table 6.8
    lines.append("## 표 6.8. 위기 구간별 하락률과 손실 (빈칸 3개 포함)\n")
    lines.append("| 위기 구간 | Buy & Hold | Static Ensemble RL | 제안 모델 |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(
        f"| 루나·금리 인상 (2022.05~2022.06) | {_fmt(ledger_rows, 'T6.8_LUNA_BH_MDD', '%')} | "
        f"{_fmt(ledger_rows, 'T6.8_LUNA_STATIC_ENS_MDD', '%')} | {_fmt(ledger_rows, 'T6.8_LUNA_PROPOSED_MDD', '%')} |"
    )
    lines.append(
        f"| FTX 파산 (2022.11.07~11.09) | {_fmt(ledger_rows, 'T6.8_FTX_BH_MDD', '%')} | "
        f"{_fmt(ledger_rows, 'T6.8_FTX_STATIC_ENS_MDD', '%')} | {_fmt(ledger_rows, 'T6.8_FTX_PROPOSED_MDD', '%')} |"
    )
    lines.append("")

    # Table 6.9
    lines.append("## 표 6.9. 위기 구간의 모달리티별 SHAP 기여도 (Variant D, 구간 II)\n")
    lines.append("| 모달리티 | 상대 기여도 (%) |")
    lines.append("| :--- | :---: |")
    lines.append(f"| 시각 (차트 이미지, ResNet-18+CBAM) | {_fmt(ledger_rows, 'T6.9_SHAP_VISUAL_PCT', '%')} |")
    lines.append(f"| 기술 지표 | {_fmt(ledger_rows, 'T6.9_SHAP_TECH_PCT', '%')} |")
    lines.append(f"| 텍스트 (뉴스 감성) | {_fmt(ledger_rows, 'T6.9_SHAP_SENTIMENT_PCT', '%')} |")
    lines.append(f"| 텍스트 + 기술 지표 합계 | {_fmt(ledger_rows, 'T6.9_SHAP_TECH_SENTI_PCT', '%')} |")
    lines.append("")

    # Table 6.10
    lines.append("## 표 6.10. 구성 요소 소거 실험 (Phase 2 샤프 지수)\n")
    lines.append("| 구성 | Phase 2 Sharpe |")
    lines.append("| :--- | :---: |")
    lines.append(f"| 제안 모델 (전체) | {_fmt(ledger_rows, 'T6.7_PROPOSED_SHARPE')} |")
    lines.append(f"| Variant A: 다중 모달 정보 제거 | {_fmt(ledger_rows, 'T6.10_VAR_A_SHARPE')} |")
    lines.append(f"| Variant B: 동적 가중 제거 (1/N 고정) | {_fmt(ledger_rows, 'T6.10_VAR_B_SHARPE')} |")
    lines.append(f"| Variant C: 앙상블 제거 (단일 에이전트) | {_fmt(ledger_rows, 'T6.10_VAR_C_SHARPE')} |")
    lines.append(f"| Variant D: 교사 증류 제거 (연구 1 단독) | {_fmt(ledger_rows, 'T6.10_VAR_D_SHARPE')} |")
    lines.append("")

    # Table 6.11
    lines.append("## 표 6.11. 추론 증류 가중치 $\\lambda_3$ 민감도 (빈칸 6개)\n")
    lines.append("| $\\lambda_3$ | 0.0 | 0.1 | 0.2 (기본) | 0.3 | 0.4 | 0.5 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    lines.append(
        "| Phase 2 Sharpe | "
        + " | ".join(
            _fmt(ledger_rows, f"T6.11_LAMBDA3_0_{i}_SHARPE") for i in range(6)
        )
        + " |\n"
    )

    # Table 6.12
    lines.append("## 표 6.12. 앙상블 온도 $T$ 민감도 (빈칸 5개)\n")
    lines.append("| 온도 $T$ | 1 | 5 | 10 (기본) | 20 | 1/N (정적) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    lines.append(
        f"| Phase 2 Sharpe | {_fmt(ledger_rows, 'T6.12_TEMP_1_0_SHARPE')} | "
        f"{_fmt(ledger_rows, 'T6.12_TEMP_5_0_SHARPE')} | "
        f"{_fmt(ledger_rows, 'T6.12_TEMP_10_0_SHARPE')} | "
        f"{_fmt(ledger_rows, 'T6.12_TEMP_20_0_SHARPE')} | "
        f"{_fmt(ledger_rows, 'T6.12_TEMP_static_1_over_N_SHARPE')} |\n"
    )

    # Table 6.13
    lines.append("## 표 6.13. 국면당 전문가 수 민감도 (빈칸 9개)\n")
    lines.append("| 국면당 전문가 수 ($K$) | Sharpe | Sortino | MDD (%) |")
    lines.append("| :---: | :---: | :---: | :---: |")
    for k_val in (3, 5, 7):
        lines.append(
            f"| {k_val} | {_fmt(ledger_rows, f'T6.13_EXP_{k_val}_SHARPE')} | "
            f"{_fmt(ledger_rows, f'T6.13_EXP_{k_val}_SORTINO')} | "
            f"{_fmt(ledger_rows, f'T6.13_EXP_{k_val}_MDD', '%')} |"
        )
    lines.append("")

    # Table 6.14
    lines.append("## 표 6.14. 실행 지연 비교\n")
    lines.append("| 실행 경로 | 평균 지연 (ms) | 중앙값 (ms) | P99 (ms) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(
        f"| 교사 직접 추론 (System 2) | {_fmt(ledger_rows, 'T6.14_TEACHER_MEAN_MS', ' ms')} | "
        f"{_fmt(ledger_rows, 'T6.14_TEACHER_MEDIAN_MS', ' ms')} | {_fmt(ledger_rows, 'T6.14_TEACHER_P99_MS', ' ms')} |"
    )
    for stage_label, stage_key in TEACHER_LATENCY_STAGES:
        lines.append(
            f"| 교사 구간: {stage_label} | {_fmt(ledger_rows, f'T6.14_TEACHER_{stage_key}_MEAN_MS', ' ms')} | - | - |"
        )
    lines.append(
        f"| 학생 단독 실행 (System 1 `Student1DCNN`) | {_fmt(ledger_rows, 'T6.14_STUDENT_MEAN_MS', ' ms')} | "
        f"{_fmt(ledger_rows, 'T6.14_STUDENT_MEDIAN_MS', ' ms')} | {_fmt(ledger_rows, 'T6.14_STUDENT_P99_MS', ' ms')} |\n"
    )

    # Table 6.15
    lines.append("## 표 6.15. 설명 가능성 평가 지표 (빈칸 4개 포함)\n")
    lines.append("| 지표 | 측정값 |")
    lines.append("| :--- | :---: |")
    lines.append(f"| 인용 정합률 | {_fmt(ledger_rows, 'T6.15_CITATION_MATCH_PCT', '%')} |")
    lines.append(f"| 환각 기각률 | {_fmt(ledger_rows, 'T6.15_HALLUCINATION_REJECT_PCT', '%')} |")
    lines.append(
        f"| 모달 기여도 (SHAP) | 시각 {_fmt(ledger_rows, 'T6.9_SHAP_VISUAL_PCT', '%')}, "
        f"기술 지표 {_fmt(ledger_rows, 'T6.9_SHAP_TECH_PCT', '%')}, "
        f"텍스트 {_fmt(ledger_rows, 'T6.9_SHAP_SENTIMENT_PCT', '%')} |"
    )
    lines.append(f"| 교사-학생 행동 일치율 | {_fmt(ledger_rows, 'T6.15_TEACHER_STUDENT_AGREE_PCT', '%')} |")
    lines.append(f"| 근거 제거 시 결정 변화율 | {_fmt(ledger_rows, 'T6.15_EVIDENCE_FLIP_PCT', '%')} |\n")

    populated = []
    pending = []
    for nid, tbl, desc in REQUIRED_27_BLANK_IDS:
        if nid in ledger_rows:
            populated.append({"number_id": nid, "table": tbl, "description": desc, "value": ledger_rows[nid]["formatted_value"]})
        else:
            pending.append({"number_id": nid, "table": tbl, "description": desc})

    audit = {
        "total_required_blank_cells": len(REQUIRED_27_BLANK_IDS),
        "populated_count": len(populated),
        "pending_count": len(pending),
        "populated": populated,
        "pending": pending,
    }
    return "\n".join(lines), audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Chapter VI tables from ledger/numbers.csv.")
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "runs" / "tables"))
    args = parser.parse_args()

    ledger = NumberLedger(args.ledger)
    rows = ledger.read_all()
    md_text, audit = compile_markdown_tables(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chapter6_tables.md").write_text(md_text, encoding="utf-8")
    (out_dir / "blank_27_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_dir / 'chapter6_tables.md'} ({audit['populated_count']}/{audit['total_required_blank_cells']} blanks populated).")


if __name__ == "__main__":
    main()
