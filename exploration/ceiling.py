"""Rough ceiling on weekly accuracy: how much of the within-week spread in D/ST and kicker points is noise that no
pre-game model can remove, even one that knows every team's true expected stats.

Irreducible pieces (treated as noise around a perfectly known mean):
  D/ST: opponent points vs the Vegas implied total (market error), pushed through the PA tier table;
        sacks, takeaways, blocks/safeties and return TDs as Poisson counts at their true rates.
  K:    team points vs Vegas implied total -> PAT count; FG attempts as Poisson; makes as Bernoulli.
Yards allowed is left out of the D/ST noise, so the D/ST ceiling is an upper bound.
Usage: python -I ceiling.py <data_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
rng = np.random.default_rng(0)
N = 400  # simulations per team-game


def pa_score(pa):
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 17, pa <= 27, pa <= 34, pa <= 45], [5, 4, 3, 1, 0, -1, -3], -5)


def ya_score(ya):
    return np.select([ya < 100, ya < 200, ya < 300, ya < 350, ya < 400, ya < 450, ya < 500, ya < 550],
                     [5, 3, 2, 0, -1, -3, -5, -6], -7)


def within_week_var(df, col):
    return (df[col] - df.groupby(['season', 'week'])[col].transform('mean')).var()


def ceiling(total_var, noise_var, ours_r, label):
    r2 = 1 - noise_var / total_var
    print(f'{label}: within-week variance {total_var:.1f}, irreducible noise >= {noise_var:.1f} '
          f'({noise_var / total_var:.0%}) -> max R^2 <= {r2:.2f}, max correlation <= {np.sqrt(r2):.2f}; '
          f'ours {ours_r:.2f} ({ours_r / np.sqrt(r2):.0%} of the ceiling)')


def ours_pearson(wf, pred, target):
    return np.mean([g[pred].corr(g[target]) for _, g in wf.groupby(['season', 'week']) if len(g) > 5])


# ---------------- D/ST ----------------
f = pd.read_parquet(DATA.parent / 'dst_v2_features.parquet')
f = f[(f.season >= 2019) & f.F_opp_implied.notna()].copy()
market_sd = (f.pa - f.F_opp_implied).std()
lam_sack, lam_take = f.sacks.mean(), f.takeaways.mean()
lam_td = f.dst_tds.mean()
other = f.dst_pts - f.sacks - 2 * f.takeaways - 6 * f.dst_tds
# PA tier noise given a perfectly known implied total
pa_draws = np.clip(np.round(f.F_opp_implied.to_numpy()[:, None] + market_sd * rng.standard_normal((len(f), N))), 0, None)
pa_noise = pa_score(pa_draws).var(axis=1).mean()
# count noise at league-average true rates (Poisson variance = mean)
count_noise = lam_sack + 4 * lam_take + 36 * lam_td
noise = pa_noise + count_noise
wf = pd.read_parquet(DATA.parent / 'dst_v2_walkforward.parquet')
print(f'D/ST pieces: PA tier {pa_noise:.1f} (market sd {market_sd:.1f} pts), sacks {lam_sack:.1f}, '
      f'takeaways {4 * lam_take:.1f}, return TDs {36 * lam_td:.1f}')
ceiling(within_week_var(f, 'dst_pts'), noise, ours_pearson(wf, 'ridge, all + weather', 'dst_pts'), 'D/ST (loose: yards allowed fully predictable)')
# tighter: yards-allowed residual from a ridge on all features also counted as noise (includes some model error,
# so this end of the range is conservative)
from sklearn.linear_model import RidgeCV
feats = [c for c in f.columns if c.startswith('F_')]
ff = f.dropna(subset=feats)
ya_sd = (ff.ya - RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(ff[feats], ff.ya).predict(ff[feats])).std()
ya_draws = ff.ya.mean() + ya_sd * rng.standard_normal((2000, N))
ya_noise = ya_score(ya_draws).var(axis=1).mean()
print(f'  yards-allowed tier noise {ya_noise:.1f} (residual sd {ya_sd:.0f} yds)')
ceiling(within_week_var(f, 'dst_pts'), noise + ya_noise, ours_pearson(wf, 'ridge, all + weather', 'dst_pts'), 'D/ST (tight)')

# ---------------- K ----------------
k = pd.read_parquet(DATA.parent / 'k_model_features.parquet')
k = k[(k.season >= 2019) & k.F_team_implied.notna()].copy()
ex = pd.read_parquet(DATA.parent / 'k_extras.parquet')
k = k.merge(ex[['game_id', 'team', 'fga', 'fgm', 'fga50']], on=['game_id', 'team'])
fg_pct = k.fgm.sum() / k.fga.sum()
fga_mean = k.fga.mean()
# make/miss given the attempt: a make is 3/4/5 (mean ~3.7, E[V^2] ~14.3), a miss -1
ex1 = fg_pct * 3.7 - (1 - fg_pct)
ex2 = fg_pct * 14.3 + (1 - fg_pct)
make_noise = fga_mean * (ex2 - ex1 ** 2)
# team scoring vs the Vegas line, passed through how kicker points move with team points
games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'home_score', 'away_score'])
k = k.merge(games, on='game_id')
k['pts'] = np.where(k.team == k.home_team, k.home_score, k.away_score)
slope = np.polyfit(k.pts - k.F_team_implied, k.k_pts, 1)[0]
market_noise = (slope * market_sd) ** 2
k_noise = make_noise + market_noise
wfk = pd.read_parquet(DATA.parent / 'k_model_walkforward.parquet')
print(f'\nK pieces: make/miss {make_noise:.1f} (FG% {fg_pct:.0%}, {fga_mean:.2f} att/game), '
      f'team scoring vs line {market_noise:.1f} ({slope:.2f} K pts per team pt)')
ceiling(within_week_var(k, 'k_pts'), k_noise, ours_pearson(wfk, 'ridge, all', 'k_pts'), 'K')
