"""Independent metric identities and decision-time falsification checks for P-20."""
import json

import numpy as np
import pandas as pd
import torch

from ts.baselines.breakout_student import daily_inputs, windows
from ts.baselines.cycle_student import OUT as CYCLE, allowed_dates
from ts.baselines.early_breakout_audit import historical_prices, policy_inputs
from ts.baselines.momentum_condition_student import OUT, momentum
from ts.baselines.risk_student import RiskStudent
from ts.baselines.weekly_student import decisions_at, digest
from ts.config import PROJECT_ROOT
from ts.eval.backtester import CausalPortfolioBacktester


def statistics(r):
    r = np.asarray(r, dtype=float)
    nav = np.r_[1., np.cumprod(1+r)]
    ann = len(r)/8760
    downside = np.sqrt(np.minimum(r, 0.) @ np.minimum(r, 0.) / len(r) * 8760)
    mean = r.sum()/ann
    return {"cum_return_pct":100*(nav[-1]-1), "cagr_pct":100*(nav[-1]**(1/ann)-1),
            "sharpe":mean/(r.std(ddof=1)*np.sqrt(8760)) if r.std()>0 else 0.,
            "sortino":mean/downside if downside else 0., "mdd_pct":100*np.min(nav/np.maximum.accumulate(nav)-1),
            "annual_arithmetic_pct":100*mean, "annual_downside_pct":100*downside}


def block_intervals(frame):
    """Paired 168-observation circular block bootstrap, identical draws for each policy."""
    r = frame.to_numpy()
    n, k = r.shape
    moments = np.column_stack((r,np.minimum(r,0.)**2))
    cum = np.vstack([np.zeros(2*k), np.cumsum(np.vstack([moments,moments[:168]]),axis=0)])
    i = np.arange(n)
    reps, tail = divmod(n,168)
    full, partial = cum[i+168]-cum[i], cum[i+tail]-cum[i]
    rng = np.random.default_rng(42)
    ratios=[]
    for _ in range(50):
        sample = (full[rng.integers(n,size=(100,reps))].sum(axis=1)+partial[rng.integers(n,size=100)])/n
        denominator = np.sqrt(np.maximum(sample[:,k:],0.))
        ratios.append(np.divide(sample[:,:k], denominator, out=np.zeros_like(denominator),
                                where=denominator>0)*np.sqrt(8760))
    draws=np.vstack(ratios)
    return {name: np.percentile(values,[2.5,97.5]).tolist() for name,values in
            [("student",draws[:,0]), *((f"student_minus_{key}",draws[:,0]-draws[:,j])
                                     for j,key in enumerate(frame.columns) if j)]}


def infer(prices, old, model, start):
    daily = pd.concat([old.close, prices.close])
    daily = daily[daily.index.hour==23].asfreq("D")
    open_state = allowed_dates(daily).eq(True) & momentum(daily,30)
    t,x=windows(daily_inputs(prices)[0])
    states=policy_inputs(prices).loc[t]
    dates=t[t>=start]
    active=(t>=start)&open_state.loc[t].to_numpy(dtype=bool)
    out=pd.Series(0.,index=dates)
    with torch.inference_mode():
        current=torch.tensor(states.loc[t[active]].to_numpy(),dtype=torch.float32)
        out.loc[t[active]]=model.weight(model.encode_hidden(x[active]),current).numpy()
    return out


def main():
    torch.set_num_threads(2)
    folder=OUT/"P20_30d_seed42"
    info=json.loads((folder/"metrics.json").read_text())
    cache=PROJECT_ROOT/"data/cache"
    hist=pd.read_parquet(cache/"btcusdt_1h_20200801_20231231.parquet")
    later=pd.read_parquet(cache/"p01_sources/confirmed_ohlcv.parquet")
    prices=pd.concat([hist[later.columns],later])
    old=historical_prices()
    source=folder/"frozen.pt"
    if digest(source)!=info["weights_sha256"]:
        raise ValueError("Frozen model identity differs")
    model=RiskStudent(input_dim=daily_inputs(prices)[0].shape[1])
    model.load_state_dict(torch.load(source,map_location="cpu",weights_only=True))
    model.eval()
    ref=infer(prices,old,model,later.index[0])
    same=pd.read_parquet(folder/"reused_decisions.parquet").student.loc[ref.index]
    np.testing.assert_allclose(ref,same,atol=1e-7,rtol=0.)
    shift=pd.Timedelta(days=13)
    moved=infer(prices.shift(freq=shift),old.shift(freq=shift),model,later.index[0]+shift)
    assert (moved.index==ref.index+shift).all()
    np.testing.assert_array_equal(ref.values,moved.values)
    boundary=later.index[len(later)//2]
    changed=prices.copy()
    mask=changed.index>=boundary
    factor=1.1+.2*np.cos(np.arange(mask.sum())/24)
    changed.loc[mask,["open","high","low","close"]]*=factor[:,None]
    altered=infer(changed,old,model,later.index[0])
    prefix=ref.index<boundary
    np.testing.assert_allclose(ref.loc[prefix],altered.loc[prefix],atol=1e-7,rtol=0.)
    if np.allclose(ref.loc[~prefix],altered.loc[~prefix]):
        raise AssertionError("Policy did not respond to changed market data")
    all_days=pd.concat([old.close,prices.close])
    all_days=all_days[all_days.index.hour==23].asfreq("D")
    gate=allowed_dates(all_days).eq(True)&momentum(all_days,30)
    report=dict(date_shift_max_abs=float(np.max(np.abs(moved.values-ref.values))),
                future_perturbation_prefix_max_abs=float(np.max(np.abs(ref.loc[prefix]-altered.loc[prefix]))),
                frozen_replay_max_abs=float(np.max(np.abs(ref-same))), future_changed_days=int((abs(altered-ref)>1e-6).sum()),
                folds_with_no_training=[v["test_from"] for v in info["folds"] if v["n_train"]==0])
    for stage, source_bars in (("paper",hist),("reused",later)):
        book=pd.read_parquet(folder/f"{stage}_decisions.parquet")
        saved=pd.read_parquet(folder/f"{stage}_returns.parquet")
        bars=source_bars.loc[book.index]
        baseline=pd.read_parquet(CYCLE/f"{stage}_returns.parquet").P12
        if not saved.index.equals(baseline.index):
            raise ValueError("Unmatched reference")
        dates=bars.index[bars.index.hour==23]
        flat_fraction=decisions_at(bars.index,dates,.25*gate.loc[dates].astype(float))
        fresh=CausalPortfolioBacktester().run_decisions(bars,book.student.to_numpy(),phase=2)
        np.testing.assert_allclose(saved.student,fresh.bar_net_returns,atol=1e-12,rtol=0.)
        metrics=statistics(saved.student)
        for key in ("sortino","sharpe","mdd_pct","cagr_pct","cum_return_pct"):
            np.testing.assert_allclose(info[stage]["student"][key],metrics[key],atol=1e-8,rtol=0.)
        twice=CausalPortfolioBacktester(phase2_fee_rate=.001,phase2_kappa=.2,phase2_eta=.1)
        stress=twice.run_decisions(bars,book.student.to_numpy(),phase=2)
        late=CausalPortfolioBacktester().run_decisions(bars,book.student.shift(1,fill_value=0).to_numpy(),phase=2)
        simple=CausalPortfolioBacktester().run_decisions(bars,flat_fraction,phase=2)
        quarters=saved.student.add(1).resample("QE").prod()-1
        best=quarters.idxmax()
        ex_best=saved.student[(saved.index.year!=best.year) | (saved.index.quarter!=best.quarter)]
        output=dict(student=metrics, baseline=statistics(baseline), gross=statistics(fresh.bar_gross_returns),
                    rule=statistics(saved.rule), untrained=statistics(saved.untrained),
                    fixed_quarter_exposure=statistics(simple.bar_net_returns), doubled_cost=statistics(stress.bar_net_returns),
                    one_extra_bar_delay=statistics(late.bar_net_returns), best_quarter=str(best.date()),
                    best_quarter_return_pct=100*quarters.max(), sortino_without_best_quarter=statistics(ex_best)["sortino"],
                    weekly_interval=block_intervals(saved[["student","rule","untrained"]].assign(P12=baseline)),
                    turnover=fresh.summary.avg_turnover, mean_weight=float(fresh.weights.mean()))
        # Attribute a realized bar to the market state known at the preceding close.
        monthly=momentum(all_days,30).reindex(prices.index).ffill().shift(1).reindex(bars.index)
        weak=monthly.eq(False)
        downside=baseline.clip(upper=0).pow(2)
        output["P12_nonpositive_monthly_momentum"]={
            "fraction_of_total_downside_squares": float(downside.loc[weak].sum()/downside.sum()),
            "sum_hourly_net_returns": float(baseline.loc[weak].sum())}
        if stage=="paper":
            # No-sample fold ablation, decided on the normal daily clock.
            test_dates=info["folds"][:-1]
            w=book.loc[dates,"student"].copy()
            for i,f in enumerate(test_dates):
                if not f["n_train"]:
                    start=pd.Timestamp(f["test_from"])
                    end=pd.Timestamp(info["folds"][i+1]["test_from"])
                    w.loc[(w.index>=start)&(w.index<end)]=0.
            sim=CausalPortfolioBacktester().run_decisions(bars,decisions_at(bars.index,dates,w),phase=2)
            output["cash_when_no_training"]=statistics(sim.bar_net_returns)
        report[stage]=output
    (folder/"audit.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
