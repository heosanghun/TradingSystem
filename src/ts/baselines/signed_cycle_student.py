"""P-13 two conditional spot-price experts, evaluated with the thesis signed simulator."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.nn import functional as F

from ts.baselines.breakout_student import TARGET, daily_inputs, teacher_examples, windows
from ts.baselines.cycle_student import OUT as CYCLE, economic_labels, training_indices
from ts.baselines.early_breakout_audit import historical_prices, policy_inputs
from ts.baselines.risk_student import RiskStudent, fit as fit_buy
from ts.baselines.weekly_confirmation import paired_sharpe_interval
from ts.baselines.weekly_student import OUT as PRIOR, decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester
from ts.eval.cost_model import compute_phase2_bar_base_cost_rates
from ts.eval.walk_forward import generate_phase2_walk_forward_folds
from ts.study3.distillation import ReasoningDistillationLoss

OUT = PROJECT_ROOT / "runs/baselines/signed_cycle_student"


def sell_states(close, budget):
    past = close.shift(1)
    signals = {}
    for key, entry, exit_days in (("fast", 20, 10), ("slow", 55, 20)):
        state = pd.Series(np.nan, index=close.index)
        state.loc[close < past.rolling(entry).min()] = 1.
        state.loc[close > past.rolling(exit_days).max()] = 0.
        signals[key] = state.ffill().fillna(0.)
    return pd.DataFrame(signals).assign(budget=budget)


def sell_labels(close, states, volatility, fee):
    actions = np.linspace(-1., 0., 5)
    center = -(states.fast + states.slow) * states.budget / 2
    net_move = -np.log(close.shift(-7) / close) - 2 * fee
    advantage = (net_move / (volatility.clip(lower=1e-8) * np.sqrt(7/365))).clip(-3, 3)
    scores = -.5 * ((actions - center.to_numpy()[:, None]) / .25)**2
    scores += advantage.to_numpy()[:, None] * np.abs(actions)
    return pd.Series(softmax(scores, axis=1) @ actions, index=close.index)


class SellStudent(RiskStudent):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        torch.nn.init.constant_(self.position[-1].bias, -.25)

    def weight(self, hidden, today):
        return self.position(torch.cat([hidden, today], dim=1)).squeeze(-1).clamp(-1., 0.)


def fit_sell(x, current, target, logits, reasons):
    torch.manual_seed(42)
    model = SellStudent(x.shape[1])
    optim = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    distill = ReasoningDistillationLoss()
    zero = torch.zeros(())
    for _ in range(50):
        for rows in torch.randperm(len(x)).split(64):
            h = model.encode_hidden(x[rows])
            task = F.smooth_l1_loss(model.weight(h, current[rows]), target[rows], beta=.1)
            total, *_ = distill(model.actor(h), logits[rows], model.projection_head(h),
                                reasons[rows], task, zero, zero)
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optim.step()
    return model.eval()


def main():
    torch.set_num_threads(2)
    cache = PROJECT_ROOT / "data/cache"
    base = cache / "btcusdt_1h_20200801_20231231.parquet"
    follow = cache / "p01_sources/confirmed_ohlcv.parquet"
    hashes = json.loads((PRIOR / "metrics.json").read_text())["sha256"]
    if digest(base) != hashes[base.name] or digest(follow) != json.loads((PRIOR / "confirmation.json").read_text())["data_sha256"]:
        raise ValueError("Unexpected BTC archive")
    hist, later = pd.read_parquet(base), pd.read_parquet(follow)
    prices = pd.concat([hist[later.columns], later])
    long_close = pd.concat([historical_prices().close, prices.close])
    daily = long_close[long_close.index.hour == 23].asfreq("D")
    avg = daily.rolling(200).mean()
    f = daily_inputs(prices)[0]
    close = prices.close.reindex(f.index)
    fees = pd.Series(compute_phase2_bar_base_cost_rates(prices), index=prices.index).reindex(f.index)
    times, x = windows(f)
    buy = policy_inputs(prices)
    sell = sell_states(close, f.budget)
    trend = (buy.fast + buy.slow) * buy.budget / 2
    states = {name: torch.tensor(s.loc[times].to_numpy(), dtype=torch.float32)
              for name, s in (("buy", buy), ("sell", sell))}
    targets = {"buy": economic_labels(close, trend, f.volatility, fees).loc[times].to_numpy(),
               "sell": sell_labels(close, sell, f.volatility, fees).loc[times].to_numpy()}
    if avg.loc[times].isna().any():
        raise ValueError("Missing 200-day regime")
    eligible = {"buy": (daily > avg).loc[times].to_numpy(), "sell": (daily < avg).loc[times].to_numpy()}
    known = times <= hist.index[-1]
    _, logits, reasons = teacher_examples(times[known], hist, trend.loc[times[known]].to_numpy(),
                                          PROJECT_ROOT / "runs/study2", hashes)
    folds = generate_phase2_walk_forward_folds(hist.index)
    if len(folds) != 9:
        raise ValueError("Expected nine folds")
    spans = [(f.train_start, f.test_start, f.test_end) for f in folds]
    spans.append((later.index[0] - pd.DateOffset(years=1), later.index[0], later.index[-1]))
    models, frozen, records, predictions = {}, {}, [], []
    OUT.mkdir(exist_ok=True, parents=True)
    for i, (start, cutoff, end) in enumerate(spans):
        test = np.flatnonzero((times >= cutoff) & (times <= end))
        decisions = pd.Series(0., index=times[test])
        for side, trainer in (("buy", fit_buy), ("sell", fit_sell)):
            y = targets[side]
            rows = training_indices(times, start, cutoff, eligible[side] & known, y, True)
            active = test[eligible[side][test]]
            if len(rows):
                models[side] = trainer(x[rows], states[side][rows], torch.tensor(y[rows], dtype=torch.float32),
                                       logits[rows], reasons[rows])
            model = models.get(side)
            if len(active) and model is not None:
                with torch.inference_mode():
                    decisions.loc[times[active]] = model.weight(model.encode_hidden(x[active]), states[side][active]).numpy()
            records.append(dict(test_from=str(cutoff), side=side, n_train=len(rows), active_days=len(active),
                                last_label=str(times[rows[-1]] + pd.Timedelta(days=7)) if len(rows) else None))
            if i == len(folds) and model is not None:
                path = OUT / f"{side}.pt"
                torch.save(model.state_dict(), path)
                frozen[side] = digest(path)
        predictions.append(decisions)
    payload = {"folds": records, "frozen_sha256": frozen, "code_sha256": digest(Path(__file__))}
    paper = hist.iloc[np.concatenate([f.test_indices for f in folds])]
    for stage, bars, series in (("paper", paper, pd.concat(predictions[:-1])), ("reused", later, predictions[-1])):
        dates = bars.index[bars.index.hour == 23]
        ref = pd.read_parquet(CYCLE / f"{stage}_decisions.parquet").P12
        if not dates.equals(series.index) or not bars.index.equals(ref.index):
            raise ValueError("P-12 clock mismatch")
        buy_days = dates[daily.loc[dates] > avg.loc[dates]]
        np.testing.assert_allclose(series.loc[buy_days], ref.loc[buy_days], atol=1e-7, rtol=0)
        decision = decisions_at(bars.index, dates, series)
        sim = CausalPortfolioBacktester().run_decisions(bars, decision, phase=2)
        short_exposure = float(np.maximum(-sim.weights, 0.).mean())
        benchmark = pd.read_parquet(CYCLE / f"{stage}_returns.parquet").P12
        payload[stage] = {"results": {"P13": sim.summary.to_dict()}, "mean_short_exposure": short_exposure,
                          "annual_simple_borrow_drag_pct": {str(rate): rate*short_exposure for rate in (10, 25)},
                          "paired_sharpe_diff_ci_vs_p12": paired_sharpe_interval(sim.bar_net_returns, benchmark)}
        pd.DataFrame({"P13": decision}, index=bars.index).to_parquet(OUT / f"{stage}_decisions.parquet")
        pd.DataFrame({"P13": sim.bar_net_returns}, index=bars.index).to_parquet(OUT / f"{stage}_returns.parquet")
        print(stage, {k: round(sim.summary.to_dict()[k], 3) for k in TARGET}, "short exposure", short_exposure)
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
