"""Shared walk-forward scorecard for feature experiments.

Protocol: ideas are developed and accepted on 2019-2024 only (DEV_SEASONS). An idea is accepted when its weekly
Spearman gain over the baseline has a 95% bootstrap CI above zero. Accepted ideas get one final check on
2025-2026 (HOLDOUT_SEASONS); nothing is tuned after looking at the holdout.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEV_SEASONS = range(2019, 2025)
HOLDOUT_SEASONS = range(2025, 2027)


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def walk_forward(f, cols, target, seasons, make=ridge):
    """Predict each season from all earlier seasons. Returns predictions indexed like f."""
    pred = np.full(len(f), np.nan)
    for s in seasons:
        tr, te = (f.season < s).to_numpy(), (f.season == s).to_numpy()
        pred[te] = make().fit(f.loc[tr, cols], f.loc[tr, target]).predict(f.loc[te, cols])
    return pred


def weekly(f, pred, target, min_teams=6):
    """Per-week Spearman and top-5 gain over the week's average."""
    d = f[[target, 'season', 'week']].assign(p=pred).dropna(subset=['p'])
    sp, gain = [], []
    for _, w in d.groupby(['season', 'week']):
        if len(w) < min_teams:
            continue
        sp.append(w.p.corr(w[target], method='spearman'))
        gain.append(w.nlargest(5, 'p')[target].mean() - w[target].mean())
    return np.array(sp), np.array(gain)


def scorecard(f, target, candidates, baseline, seasons, seed=0):
    """candidates: {name: pred array}. Compares each with candidates[baseline] week by week."""
    rng = np.random.default_rng(seed)
    base_sp, _ = weekly(f, candidates[baseline], target)
    rows = []
    for name, pred in candidates.items():
        sp, gain = weekly(f, pred, target)
        diff = sp - base_sp
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(model=name, weekly_spearman=sp.mean(), top5_gain=gain.mean(), vs_base=diff.mean(),
                         ci_lo=lo, ci_hi=hi, weeks_better=f'{(diff > 0).sum()}/{len(diff)}',
                         accept='YES' if lo > 0 else ''))
    import pandas as pd
    return pd.DataFrame(rows)
