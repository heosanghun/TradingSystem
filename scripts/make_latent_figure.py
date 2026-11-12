"""
t-SNE projection of the student's 128-dimensional hidden vectors, with crisis periods marked.

Input : runs/study3/student_oos_hidden.npz, written by the Study 3 run, with
          timestamps_ns  (N,)      bar open time, UTC nanoseconds
          hidden         (N, 128)  `Student1DCNN.encode_hidden` output for each test-fold bar
Output: runs/figures/student_latent_tsne.png (and .pdf), runs/figures/student_latent_tsne.json

Everything on the figure is computed from that file. The numbers in the annotation are the
measured values written to the JSON; nothing is typed in by hand.

How to read the numbers
- A t-SNE picture depends on its settings (perplexity, seed), so the picture alone is not
  evidence. Two statistics are therefore computed in the original 128-dimensional space,
  where no projection is involved:
    * `knn_crisis_share`: for each crisis bar, the share of its k nearest neighbours that are
      also crisis bars, averaged over crisis bars. Compare with `crisis_base_rate`, the share
      expected if crisis bars were spread at random.
    * `knn_permutation_p`: how often a random relabelling of the same number of bars reaches
      a share at least as large.
- Two statistics describe the dashed ellipse on the 2-D picture (the 2-sigma covariance
  ellipse of the crisis points):
    * `ellipse_crisis_inside`: share of crisis points inside the ellipse.
    * `ellipse_purity`: share of all points inside the ellipse that are crisis points.

Bars one hour apart have nearly identical inputs, so neighbours in time are neighbours in
the latent space whatever the model has learned. To keep that from counting as structure,
neighbours within `--exclude-hours` of a bar are ignored in the k-NN statistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ts.config import PROJECT_ROOT

# Crisis windows (UTC). The first and third follow Table 6.8 of the thesis in spirit; the
# FTX window is widened from the three-day price drop to the two weeks around it so that it
# contains enough bars to be visible. The second is the month of the 75 bp rate rise.
CRISIS_WINDOWS = (
    ("LUNA 디페깅·금리 인상 (2022.05~06)", "2022-05-01", "2022-07-01", "^", "#C62828"),
    ("연준 자이언트 스텝 (2022.09)", "2022-09-01", "2022-10-01", "s", "#EF7F1A"),
    ("FTX 파산 (2022.11)", "2022-11-01", "2022-11-21", "D", "#7B3FA0"),
)


def knn_crisis_share(hidden: np.ndarray, is_crisis: np.ndarray, hours: np.ndarray, k: int, exclude_hours: float) -> float:
    from sklearn.neighbors import NearestNeighbors

    # Ask for more neighbours than needed, then drop those close in time.
    extra = int(2 * exclude_hours) + k + 1
    nn = NearestNeighbors(n_neighbors=min(extra, len(hidden))).fit(hidden)
    idx_crisis = np.where(is_crisis)[0]
    _, nbrs = nn.kneighbors(hidden[idx_crisis])
    shares = []
    for row, i in zip(nbrs, idx_crisis):
        keep = row[np.abs(hours[row] - hours[i]) > exclude_hours][:k]
        if len(keep):
            shares.append(float(np.mean(is_crisis[keep])))
    return float(np.mean(shares)) if shares else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE figure of the student's hidden vectors.")
    parser.add_argument("--hidden-file", type=str, default=str(PROJECT_ROOT / "runs" / "study3" / "student_oos_hidden.npz"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "runs" / "figures"))
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--exclude-hours", type=float, default=24.0)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--max-crisis-markers", type=int, default=120, help="Markers drawn per crisis (all points are used in the statistics).")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.manifold import TSNE

    data = np.load(args.hidden_file)
    ts = pd.to_datetime(data["timestamps_ns"], utc=True)
    hidden = np.asarray(data["hidden"], dtype=np.float64)
    keep = (ts >= pd.Timestamp(args.start, tz="UTC")) & (ts < pd.Timestamp(args.end, tz="UTC"))
    ts, hidden = ts[keep], hidden[keep]
    if len(ts) < 100:
        raise SystemExit(f"only {len(ts)} bars in the requested period")
    hours = (ts - ts[0]).total_seconds().to_numpy() / 3600.0

    crisis_id = np.full(len(ts), -1, dtype=int)
    for j, (_label, a, b, _m, _c) in enumerate(CRISIS_WINDOWS):
        crisis_id[(ts >= pd.Timestamp(a, tz="UTC")) & (ts < pd.Timestamp(b, tz="UTC"))] = j
    is_crisis = crisis_id >= 0

    # Statistics in the original space.
    rng = np.random.default_rng(args.seed)
    observed = knn_crisis_share(hidden, is_crisis, hours, args.knn, args.exclude_hours)
    base_rate = float(is_crisis.mean())
    # Permutation reference: move the crisis windows to random positions in time, keeping
    # their lengths, so that the reference has the same clustering in time as the real labels.
    perm = []
    lengths = [int((crisis_id == j).sum()) for j in range(len(CRISIS_WINDOWS))]
    for _ in range(args.permutations):
        fake = np.zeros(len(ts), dtype=bool)
        for n in lengths:
            if 0 < n < len(ts):
                start = int(rng.integers(0, len(ts) - n))
                fake[start:start + n] = True
        perm.append(knn_crisis_share(hidden, fake, hours, args.knn, args.exclude_hours))
    perm = np.asarray(perm)
    p_value = float((np.sum(perm >= observed) + 1) / (len(perm) + 1)) if len(perm) else float("nan")

    xy = TSNE(n_components=2, perplexity=args.perplexity, init="pca", learning_rate="auto",
              random_state=args.seed).fit_transform(hidden)

    # 2-sigma covariance ellipse of the crisis points in the projection.
    cxy = xy[is_crisis]
    center = cxy.mean(axis=0)
    cov = np.cov(cxy.T)
    evals, evecs = np.linalg.eigh(cov)
    inv = np.linalg.inv(cov)
    d2 = np.einsum("ni,ij,nj->n", xy - center, inv, xy - center)
    inside = d2 <= 4.0
    crisis_inside = float(np.mean(inside[is_crisis]))
    purity = float(np.mean(is_crisis[inside])) if inside.any() else float("nan")

    stats = {
        "source": str(Path(args.hidden_file).name), "n_bars": int(len(ts)),
        "period": [str(ts.min()), str(ts.max())], "seed": args.seed, "perplexity": args.perplexity,
        "crisis_windows": [{"label": w[0], "start": w[1], "end_exclusive": w[2], "n_bars": n}
                           for w, n in zip(CRISIS_WINDOWS, lengths)],
        "crisis_base_rate": base_rate, "knn": args.knn, "knn_exclude_hours": args.exclude_hours,
        "knn_crisis_share": observed, "knn_permutation_mean": float(perm.mean()) if len(perm) else None,
        "knn_permutation_p": p_value, "n_permutations": int(len(perm)),
        "ellipse_crisis_inside": crisis_inside, "ellipse_purity": purity,
    }

    plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=200)
    tnum = (ts - ts[0]).total_seconds().to_numpy()
    calm = ~is_crisis
    sc = ax.scatter(xy[calm, 0], xy[calm, 1], c=tnum[calm], cmap="viridis", s=9, alpha=0.75, linewidths=0)
    for j, (label, _a, _b, marker, color) in enumerate(CRISIS_WINDOWS):
        idx = np.where(crisis_id == j)[0]
        if len(idx) > args.max_crisis_markers:
            idx = np.sort(rng.choice(idx, size=args.max_crisis_markers, replace=False))
        ax.scatter(xy[idx, 0], xy[idx, 1], marker=marker, s=46, color=color, edgecolors="white", linewidths=0.6, label=label, zorder=3)
    angle = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))
    ax.add_patch(Ellipse(center, 4 * np.sqrt(evals[1]), 4 * np.sqrt(evals[0]), angle=angle, fill=False, linestyle="--", linewidth=1.3, edgecolor="#333333", zorder=4))
    note = (f"위기 구간 벡터의 2σ 타원\n포함률 {crisis_inside * 100:.1f}%, 타원 내 위기 비율 {purity * 100:.1f}%\n"
            f"128차원 k-NN 위기 비율 {observed * 100:.1f}% (무작위 기준 {stats['knn_permutation_mean'] * 100:.1f}%, p={p_value:.3f})")
    # Arrow target: the point of the ellipse boundary that is highest on the page.
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    boundary = center + 2.0 * (np.outer(np.cos(theta), evecs[:, 1] * np.sqrt(evals[1]))
                               + np.outer(np.sin(theta), evecs[:, 0] * np.sqrt(evals[0])))
    target = boundary[int(np.argmax(boundary[:, 1]))]
    ax.annotate(note, xy=target, xytext=(0.99, 1.0), textcoords="axes fraction", fontsize=10,
                fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999", alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#333333", lw=1.0), zorder=6)
    ax.margins(y=0.12)
    ax.scatter([], [], s=14, color="#3b6f8f", label="그 밖의 구간 (색상 = 시점)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1], loc="lower left", fontsize=9.5, frameon=True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    quarters = pd.date_range(ts.min().normalize(), ts.max(), freq="QS")
    ticks = [(q - ts[0]).total_seconds() for q in quarters if q >= ts[0]]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{q.year} Q{q.quarter}" for q in quarters if q >= ts[0]])
    cbar.set_label("시점")
    ax.set_xlabel("t-SNE 차원 1")
    ax.set_ylabel("t-SNE 차원 2")
    ax.set_title(f"System 1 (1D-CNN) 128차원 은닉 잠재 공간의 t-SNE 투영 (시드 {args.seed})", fontsize=14, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "student_latent_tsne.png")
    fig.savefig(out_dir / "student_latent_tsne.pdf")
    (out_dir / "student_latent_tsne.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
