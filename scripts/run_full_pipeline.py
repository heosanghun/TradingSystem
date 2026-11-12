"""
End-to-end orchestrator for the Dual-Process Trading System pipeline.

Executes the empirical measurement stages in topological order:
  1. [WP0] `scripts/run_wp0_data_and_stats.py`
  2. [Study 1] `scripts/run_study1.py`
  3. [Study 2] `scripts/run_study2.py`
  4. [Study 3] `scripts/run_study3.py`
  5. [Baselines & Significance] `scripts/run_baselines_and_eval.py`
  6. [Tables] `scripts/make_tables.py`
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from ts.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from ts.ledger import SMOKE_ENV_VAR


def _run_stage(title: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    banner = "=" * 78
    print(f"\n{banner}\n[PIPELINE] {title}\n  CMD: {' '.join(cmd)}\n{banner}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{title}' failed with exit code {proc.returncode} after {elapsed:.1f}s.")
    print(f"[PIPELINE] Completed '{title}' in {elapsed:.1f}s.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the Dual-Process Trading System pipeline end-to-end."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--news-file", type=str, default="", help="Point-in-time news archive (CSV/Parquet).")
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="openai_compat",
        choices=["openai_compat", "ollama"],
    )
    parser.add_argument("--llm-endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--llm-model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--skip-wp0", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument(
        "--smoke-test",
        "--quick-verify",
        dest="smoke_test",
        action="store_true",
        help="Smoke check mode with reduced epochs/rollouts; sets TS_SMOKE_RUN=1 so numbers go to provisional.csv.",
    )
    args = parser.parse_args()

    py_exe = sys.executable
    scripts_dir = PROJECT_ROOT / "scripts"

    child_env = dict(os.environ)
    if args.smoke_test:
        child_env[SMOKE_ENV_VAR] = "1"

    if not args.skip_wp0:
        _run_stage(
            "Stage 1/6 - WP0: Data statistics and online Student inference latency",
            [py_exe, str(scripts_dir / "run_wp0_data_and_stats.py"), "--config", args.config],
            env=child_env,
        )

    s1_cmd = [py_exe, str(scripts_dir / "run_study1.py"), "--config", args.config]
    if args.smoke_test:
        s1_cmd.extend(["--pretrain-epochs", "1", "--ppo-rollouts", "2"])
    if args.news_file:
        s1_cmd.extend(["--news-file", args.news_file])
    _run_stage("Stage 2/6 - Study 1: Multimodal encoders, XGBoost regime gate, and PPO+PER ensemble", s1_cmd, env=child_env)

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
    ]
    if args.smoke_test:
        s2_cmd.extend(["--counterfactual-stride", "48"])
    if args.news_file:
        s2_cmd.extend(["--news-file", args.news_file])
    _run_stage("Stage 3/6 - Study 2: Dynamic hypergraph and citation-verified LLM debate", s2_cmd, env=child_env)

    s3_cmd = [py_exe, str(scripts_dir / "run_study3.py"), "--config", args.config]
    if args.smoke_test:
        s3_cmd.extend(["--epochs", "2"])
    _run_stage("Stage 4/6 - Study 3: Reasoning distillation and online Student evaluation", s3_cmd, env=child_env)

    if not args.skip_baselines:
        b_cmd = [py_exe, str(scripts_dir / "run_baselines_and_eval.py"), "--config", args.config]
        if args.smoke_test:
            b_cmd.extend(["--ppo-rollouts", "2"])
        _run_stage("Stage 5/6 - Baselines and Ledoit-Wolf bootstrap significance", b_cmd, env=child_env)

    _run_stage(
        "Stage 6/6 - Chapter VI Markdown tables and ledger verification",
        [py_exe, str(scripts_dir / "make_tables.py")],
        env=child_env,
    )


if __name__ == "__main__":
    main()
