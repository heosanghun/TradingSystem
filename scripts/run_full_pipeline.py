"""
Master End-to-End Orchestrator for `fig1` Overall System Architecture.

Executes the complete empirical measurement pipeline in exact `fig1` topological order:
  1. [입력 ①·③·④ & 시스템 1 지연 실측] `scripts/run_wp0_data_and_stats.py`
  2. [오프라인 System 2 교사 — 연구 1] `scripts/run_study1.py`
     (캔들 차트 ResNet-18+CBAM + 기술지표 + 뉴스 감성 EWMA -> 크로스 어텐션 -> XGBoost 국면 게이트 -> 3국면 PPO+PER 볼츠만 앙상블)
  3. [오프라인 System 2 교사 — 연구 2] `scripts/run_study2.py`
     (60일 그레인저+전이 엔트로피 -> 동적 하이퍼그래프 H_t -> 근거 인용 검증 알파·리스크·메타 PM 토론 -> W_final, C_t)
  4. [교사 신호 변환 & 연구 3 추론 증류 -> 온라인 System 1 단독 추론] `scripts/run_study3.py`
     (P_teacher^tau + FinBERT E_reasoning + 10창 브로드캐스트 -> L_task+L_policy+L_reasoning -> Student1DCNN.act_online)
  5. [통제 비교군 & 유의성 검정] `scripts/run_baselines_and_eval.py`
  6. [제VI장 표 6.2~6.15 및 27개 빈칸 검증] `scripts/make_tables.py`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def _run_stage(title: str, cmd: List[str]) -> None:
    banner = "=" * 78
    print(f"\n{banner}\n[FIG-1 PIPELINE] {title}\n  CMD: {' '.join(cmd)}\n{banner}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{title}' failed with exit code {proc.returncode} after {elapsed:.1f}s.")
    print(f"[FIG-1 PIPELINE] Completed '{title}' in {elapsed:.1f}s.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the full Fig 1 Dual-Process Trading System pipeline end-to-end."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--news-file", type=str, default="", help="Optional CSV/Parquet point-in-time news archive.")
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="openai_compat",
        choices=["openai_compat", "ollama", "deterministic_econometric"],
    )
    parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--fetch-1m-subbars", action="store_true", help="Fetch 1m Binance sub-bars for LOB microstructure.")
    parser.add_argument(
        "--quick-verify",
        action="store_true",
        help="Fast smoke-verification preset (reduced epochs/rollouts/counterfactual stride + deterministic econometric fallback if requested).",
    )
    parser.add_argument("--skip-wp0", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    py_exe = sys.executable
    scripts_dir = PROJECT_ROOT / "scripts"

    pretrain_epochs = "1" if args.quick_verify else "4"
    ppo_rollouts = "2" if args.quick_verify else "8"
    cf_stride = "48" if args.quick_verify else "1"
    s3_epochs_args = ["--epochs", "2"] if args.quick_verify else []

    if not args.skip_wp0:
        _run_stage(
            "Stage 1/6 — WP0: 입력 ①·③·④ 실측 통계 & 온라인 학생 신경망(System 1) 추론 지연 벤치마크",
            [py_exe, str(scripts_dir / "run_wp0_data_and_stats.py"), "--config", args.config],
        )

    s1_cmd = [
        py_exe,
        str(scripts_dir / "run_study1.py"),
        "--config",
        args.config,
        "--pretrain-epochs",
        pretrain_epochs,
        "--ppo-rollouts",
        ppo_rollouts,
    ]
    if args.news_file:
        s1_cmd.extend(["--news-file", args.news_file])
    _run_stage("Stage 2/6 — 연구 1 (System 2 상위 모듈): 멀티모달 인코더 + XGBoost 국면 + 동적 PPO+PER 앙상블", s1_cmd)

    s2_cmd = [
        py_exe,
        str(scripts_dir / "run_study2.py"),
        "--config",
        args.config,
        "--llm-backend",
        args.llm_backend,
        "--llm-endpoint",
        args.llm_endpoint,
        "--llm-model",
        args.llm_model,
        "--counterfactual-stride",
        cf_stride,
    ]
    if args.news_file:
        s2_cmd.extend(["--news-file", args.news_file])
    _run_stage("Stage 3/6 — 연구 2 (System 2 하위 모듈): 동적 하이퍼그래프 H_t + 인용 검증 멀티에이전트 LLM 토론", s2_cmd)

    s3_cmd = [
        py_exe,
        str(scripts_dir / "run_study3.py"),
        "--config",
        args.config,
        *s3_epochs_args,
    ]
    if args.fetch_1m_subbars:
        s3_cmd.append("--fetch-1m-subbars")
    _run_stage(
        "Stage 4/6 — 연구 3: 오프라인 추론 증류(L_task + L_policy + L_reasoning) & 온라인 System 1 단독 실행",
        s3_cmd,
    )

    if not args.skip_baselines:
        _run_stage(
            "Stage 5/6 — 통제 비교 모델(Single PPO, Static Ensemble, MAML) & Ledoit-Wolf 부트스트랩 검정",
            [
                py_exe,
                str(scripts_dir / "run_baselines_and_eval.py"),
                "--config",
                args.config,
                "--ppo-rollouts",
                ppo_rollouts,
            ],
        )

    _run_stage(
        "Stage 6/6 — 제VI장 표 6.2~6.15 마크다운 렌더링 및 출처 대장(ledger/numbers.csv) 최종 검증",
        [py_exe, str(scripts_dir / "make_tables.py")],
    )


if __name__ == "__main__":
    main()
