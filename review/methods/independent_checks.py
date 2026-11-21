"""Independent checks of P-61 (read-only; writes nothing in the project)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.baselines.trend_entry_student import entry_signals, day_volume
from ts.baselines.breakout_student import daily_inputs
from ts.baselines.early_breakout_audit import policy_inputs, historical_prices

A = PROJECT_ROOT / "runs/baselines/trend_entry"
P = A / "P61_seed42"
cache = PROJECT_ROOT / "data/cache"
hist = pd.read_parquet(cache / "btcusdt_1h_20200801_20231231.parquet")
later = pd.read_parquet(cache / "p01_sources/confirmed_ohlcv.parquet")
print("cache attrs:", hist.attrs, "| later attrs:", later.attrs)
folds = generate_phase2_walk_forward_folds(hist.index)
idx = np.concatenate([f.test_indices for f in folds])
paper = hist.iloc[idx]
bt = CausalPortfolioBacktester()


def mets(r):
    r = np.asarray(r, float)
    eq = np.cumprod(1 + r)
    s = r.mean() / r.std(ddof=1) * np.sqrt(8760)
    so = r.mean() * np.sqrt(8760) / np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return dict(sharpe=round(s, 3), sortino=round(so, 3), cum=round((eq[-1] - 1) * 100, 2),
                cagr=round((eq[-1] ** (8760 / len(r)) - 1) * 100, 2),
                mdd=round(float(np.min(eq / np.maximum.accumulate(eq) - 1)) * 100, 2))


# (a) same engine reproduces the ledger's proposed model on the same bars
z = np.load(PROJECT_ROOT / "runs/study3/student_oos_predictions.npz")
assert np.array_equal(z["oos_bar_indices"], idx)
res = bt.run_decisions(paper, z["student_weights"], phase=2)
print("(a) ledger proposed via default engine:", mets(res.bar_net_returns),
      "| saved:", mets(z["phase2_net_returns"]),
      "| identical returns:", np.allclose(res.bar_net_returns, z["phase2_net_returns"], atol=1e-12, rtol=0))

# (d) context on both windows with the same engine
dec = {k: pd.read_parquet(P / f"{k}_decisions.parquet") for k in ("paper", "reused")}
ret = {k: pd.read_parquet(P / f"{k}_returns.parquet") for k in ("paper", "reused")}
for k, bars in (("paper", paper), ("reused", later.loc["2024-01-01":"2026-10-31"])):
    d = dec[k].student
    print(f"(d) {k}: P61 student", mets(ret[k].student), "| rule", mets(ret[k].rule), "| untrained", mets(ret[k].untrained))
    bh = bt.run_decisions(bars, np.ones(len(bars)), phase=2)
    print(f"    buy&hold (enter first close)", mets(bh.bar_net_returns))
    w = d.to_numpy()
    print(f"    exposure>0 share {np.mean(w > 0):.3f}, mean weight {w.mean():.3f}, weight changes {int(np.count_nonzero(np.diff(w)))}, "
          f"entries {int(np.sum((w[1:] > 0) & (w[:-1] == 0)))}")
    r = ret[k].student
    by_q = r.groupby([r.index.year, r.index.quarter]).apply(lambda s: (1 + s).prod() - 1) * 100
    print("    quarterly %:", {f"{a}Q{b}": round(v, 1) for (a, b), v in by_q.items()})

# (b) look-ahead: signals and features at date t unchanged when data after t is removed
bars_all = pd.concat([hist[later.columns], later])
old = historical_prices()
full = pd.concat([old, bars_all])
close_all = full.close[full.index.hour == 23].asfreq("D")
sig_full = entry_signals(close_all, day_volume(full)).P61
feat_full = daily_inputs(bars_all)[0]
pol_full = policy_inputs(bars_all)
rng = np.random.default_rng(7)
cuts = sorted(rng.choice(pd.date_range("2021-10-05", "2026-10-25", freq="D"), 25, replace=False))
bad = 0
for c in cuts:
    c = pd.Timestamp(c).tz_localize("UTC") + pd.Timedelta(hours=23)
    f2 = full.loc[:c]
    s2 = entry_signals(f2.close[f2.index.hour == 23].asfreq("D"), day_volume(f2)).P61
    b2 = bars_all.loc[:c]
    common = s2.index
    bad += int((s2 != sig_full.loc[common]).sum())
    fa, fb = daily_inputs(b2)[0], feat_full.loc[:c]
    bad += int(((fa - fb).abs() > 1e-9).sum().sum())
    pa, pb = policy_inputs(b2), pol_full.loc[:c]
    bad += int(((pa - pb).abs() > 1e-9).sum().sum())
print(f"(b) truncation test over {len(cuts)} random cut dates: mismatching values = {bad}")

# (c) fold boundaries recorded by the run
info = json.loads((P / "metrics.json").read_text())
for f in info["folds"]:
    ok = f["last_label"] is None or pd.Timestamp(f["last_label"]) < pd.Timestamp(f["test_from"])
    print(f"(c) test_from {f['test_from'][:10]} n_train {f['n_train']:4d} active {f['active']:4d} last_label {str(f['last_label'])[:16]} ok={ok}")

# (e) candidate count and deflated Sharpe on the paper window
sh = {}
for p in sorted(A.glob("P*_seed42")):
    m = json.loads((p / "metrics.json").read_text())
    sh[p.name] = (m["paper"]["student"]["sharpe"], m["paper"]["student"]["sortino"],
                  m["reused"]["student"]["sharpe"], m["reused"]["student"]["sortino"])
df = pd.DataFrame(sh, index=["paper_sh", "paper_so", "reused_sh", "reused_so"]).T
print("(e) archived candidates with seed 42:", len(df), "| paper Sharpe sd", round(df.paper_sh.std(), 3),
      "| count with paper Sortino>2.08:", int((df.paper_so > 2.08).sum()), "| both periods > 2.08:",
      int(((df.paper_so > 2.08) & (df.reused_so > 2.08)).sum()))
r = ret["paper"].student.to_numpy()
T = len(r)
sr = r.mean() / r.std(ddof=1)
g3, g4 = skew(r), kurtosis(r, fisher=False)
se = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1))
gam = 0.5772156649
for N, var_ann in ((32, df.paper_sh.var()), (62, df.paper_sh.var()), (62, 0.67 ** 2), (100, 0.67 ** 2)):
    v = var_ann / 8760
    emax = np.sqrt(v) * ((1 - gam) * norm.ppf(1 - 1 / N) + gam * norm.ppf(1 - 1 / (N * np.e)))
    print(f"    N={N:3d} trial-SR sd(ann)={np.sqrt(var_ann):.2f}: E[max SR] ann {emax*np.sqrt(8760):.2f}, DSR {norm.cdf((sr-emax)/se):.3f}")
print(f"    P61 paper ann SR {sr*np.sqrt(8760):.3f}, ann SE {se*np.sqrt(8760):.3f}, PSR(0) {norm.cdf(sr/se):.3f}, "
      f"P(SR>0.92) {norm.cdf((sr-0.92/np.sqrt(8760))/se):.3f}")
