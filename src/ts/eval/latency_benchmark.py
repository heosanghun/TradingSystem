"""
10,000-Run Execution Latency Benchmark (Chapter VI Sec 11, Table 6.14).

Measures single-observation online inference latency (`mean`, `median`, `p95`, `p99`, `std` in ms)
with explicit `torch.cuda.synchronize()` barriers before and after each timed forward pass:
1. `Student1DCNN.act_online`: LOB `(1, 40, 100)` Z-score normalization + 1D-CNN forward + argmax action
   (verifying sub-12ms millisecond execution without Teacher/LLM overhead).
2. `Teacher Direct Pipeline`: Rolling Transfer Entropy + Hypergraph lookup + 3-round Debate step.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from ts.data.lob_processor import LOBWindowProcessor
from ts.study3.student_model import Student1DCNN


@dataclass
class LatencyStatsMs:
    n_runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    device_name: str

    def to_dict(self) -> Dict[str, float | int | str]:
        return asdict(self)


@dataclass
class LatencyBenchmarkReport:
    student_online: LatencyStatsMs
    teacher_direct: LatencyStatsMs
    speedup_ratio: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "student_online": self.student_online.to_dict(),
            "teacher_direct": self.teacher_direct.to_dict(),
            "speedup_ratio": self.speedup_ratio,
        }


def benchmark_student_and_teacher_latency(
    student_model: Optional[Student1DCNN] = None,
    teacher_step_fn: Optional[Callable[[], object]] = None,
    n_student_runs: int = 10_000,
    n_teacher_runs: int = 50,
    warmup_runs: int = 100,
    device: Optional[torch.device] = None,
) -> LatencyBenchmarkReport:
    """
    Executes the exact Table 6.14 latency benchmark from raw LOB feature normalization
    to discrete action output.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"

    model = (student_model or Student1DCNN()).to(dev)
    model.eval()

    rng = np.random.default_rng(42)
    raw_lob_100x40 = rng.standard_normal((100, 40)).astype(np.float32)

    def _run_single_student() -> int:
        # Includes per-window Z-score feature normalization + H2D transfer + 1D-CNN + argmax
        normed = LOBWindowProcessor.zscore_window(raw_lob_100x40)[None, :, :]  # (1, 40, 100)
        x_t = torch.as_tensor(normed, dtype=torch.float32, device=dev)
        act, _ = model.act_online(x_t)
        return int(act.item())

    for _ in range(warmup_runs):
        _run_single_student()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    s_times_ms = np.empty(n_student_runs, dtype=np.float64)
    for i in range(n_student_runs):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        _ = _run_single_student()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        s_times_ms[i] = (time.perf_counter() - t0) * 1000.0

    student_stats = LatencyStatsMs(
        n_runs=n_student_runs,
        mean_ms=float(np.mean(s_times_ms)),
        median_ms=float(np.median(s_times_ms)),
        p95_ms=float(np.percentile(s_times_ms, 95.0)),
        p99_ms=float(np.percentile(s_times_ms, 99.0)),
        std_ms=float(np.std(s_times_ms, ddof=1)) if n_student_runs > 1 else 0.0,
        device_name=dev_name,
    )

    if teacher_step_fn is not None:
        t_times_ms = np.empty(n_teacher_runs, dtype=np.float64)
        for j in range(n_teacher_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = teacher_step_fn()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_times_ms[j] = (time.perf_counter() - t0) * 1000.0
        teacher_stats = LatencyStatsMs(
            n_runs=n_teacher_runs,
            mean_ms=float(np.mean(t_times_ms)),
            median_ms=float(np.median(t_times_ms)),
            p95_ms=float(np.percentile(t_times_ms, 95.0)),
            p99_ms=float(np.percentile(t_times_ms, 99.0)),
            std_ms=float(np.std(t_times_ms, ddof=1)) if n_teacher_runs > 1 else 0.0,
            device_name=dev_name,
        )
    else:
        teacher_stats = LatencyStatsMs(
            n_runs=0,
            mean_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            std_ms=0.0,
            device_name=dev_name,
        )

    speedup = (
        teacher_stats.mean_ms / max(student_stats.mean_ms, 1e-6)
        if teacher_stats.mean_ms > 0
        else 0.0
    )

    return LatencyBenchmarkReport(
        student_online=student_stats,
        teacher_direct=teacher_stats,
        speedup_ratio=float(speedup),
    )
