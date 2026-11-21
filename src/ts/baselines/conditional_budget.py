"""Causal risk budgets used as training targets and contemporaneous model inputs."""
import numpy as np
import pandas as pd


NAMES = ("P51", "P52", "P53", "P54", "P55", "P56")


def risk_budgets(full):
    hourly = np.log(full.close.asfreq("h")).diff()
    clock = full.index[full.index.hour == 23]
    close = full.close.loc[clock].asfreq("D")
    sigma = np.log(close).diff().rolling(30).std()
    base = .25 / (sigma * np.sqrt(365)).clip(lower=1e-8)
    realized = {n: (hourly.rolling(n * 24, min_periods=(n - 1) * 24).std()
                    .reindex(close.index) * np.sqrt(8760)) for n in (7, 30, 90)}
    downside = np.sqrt(hourly.clip(upper=0).pow(2).rolling(168, min_periods=144).mean()
                       .reindex(close.index) * (2 * 8760))
    impulse = pd.DataFrame({n: np.log(close).diff(n) / (sigma.clip(lower=1e-8) * np.sqrt(n))
                            for n in (7, 14, 30)})
    return pd.DataFrame({
        "P51": .25 / realized[7].clip(lower=1e-8),
        "P52": .25 / np.maximum(realized[7], realized[30]).clip(lower=1e-8),
        "P53": .25 * realized[90] / realized[30].pow(2).clip(lower=1e-8),
        "P54": .25 / downside.clip(lower=1e-8),
        "P55": base.clip(upper=1.) * impulse[7].clip(0., 1.),
        "P56": base.clip(upper=1.) * impulse.mean(axis=1, skipna=False).clip(0., 1.),
    }).clip(0., 1.)
