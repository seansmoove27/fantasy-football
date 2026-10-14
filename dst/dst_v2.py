"""D/ST model v2 for the Limers league scoring: build features, walk-forward compare models.

Usage:
  python -I dst_v2.py build <data_dir>     # cache team-game table + features to dst_v2_features.parquet
  python -I dst_v2.py eval  <data_dir>     # walk-forward comparison of candidate models
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODE, DATA = sys.argv[1], Path(sys.argv[2])
OUT = DATA.parent / 'dst_v2_features.parquet'
YEARS = range(2015, 2027)
RELOCATE = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
HALFLIFE = 6  # games; recent form carries across seasons and decays


def pa_score(pa):
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 17, pa <= 27, pa <= 34, pa <= 45], [5, 4, 3, 1, 0, -1, -3], -5)


def ya_score(ya):
    return np.select([ya < 100, ya < 200, ya < 300, ya < 350, ya < 400, ya < 450, ya < 500, ya < 550],
                     [5, 3, 2, 0, -1, -3, -5, -6], -7)


def expected_bracket(mu, sd, scorer, grid):
    """E[scorer(X)] for X ~ Normal(mu, sd), evaluated on an integer grid."""
    w = np.exp(-0.5 * ((grid[None, :] - mu[:, None]) / sd) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    return (w * scorer(grid)[None, :]).sum(axis=1)


def build_games():
    cols = ['game_id', 'season', 'season_type', 'week', 'home_team', 'away_team', 'posteam', 'defteam',
            'play_type', 'two_point_attempt', 'sack', 'qb_hit', 'qb_dropback', 'interception', 'fumble_lost',
            'fumbled_1_team', 'safety', 'punt_blocked', 'field_goal_result', 'extra_point_result', 'touchdown',
            'td_team', 'yards_gained', 'epa', 'total_home_score', 'total_away_score']
    rows = []
    for y in YEARS:
        pbp = pd.read_parquet(DATA / f'pbp_{y}.parquet', columns=cols)
        pbp = pbp[pbp.season_type == 'REG']
        for c in ('home_team', 'away_team', 'posteam', 'defteam', 'fumbled_1_team', 'td_team'):
            pbp[c] = pbp[c].replace(RELOCATE)
        scrim = (pbp.play_type.isin(['pass', 'run'])) & (pbp.two_point_attempt != 1)
        for gid, g in pbp.groupby('game_id'):
            home, away = g.home_team.iloc[0], g.away_team.iloc[0]
            final = g.iloc[-1]
            score = {home: final.total_home_score, away: final.total_away_score}
            s = g[scrim.loc[g.index]]
            lost = g[g.fumble_lost == 1]
            for d, o in ((home, away), (away, home)):
                on_d, all_d = s[s.defteam == d], g[g.defteam == d]
                fum_rec = (lost.fumbled_1_team == o).sum()  # credit the team that did not fumble
                blocks = (all_d.punt_blocked.fillna(0).sum() + (all_d.field_goal_result == 'blocked').sum()
                          + (all_d.extra_point_result == 'blocked').sum())
                tds = g[(g.touchdown == 1) & (g.td_team == d)]
                dst_tds = ((tds.play_type == 'kickoff') | (tds.posteam != d)).sum()
                ints, sacks, safeties = all_d.interception.sum(), on_d.sack.sum(), all_d.safety.sum()
                ya, pa = on_d.yards_gained.sum(), score[o]
                pts = (sacks + 2 * (ints + fum_rec + safeties + blocks) + 6 * dst_tds
                       + int(pa_score(np.array([pa]))[0]) + int(ya_score(np.array([ya]))[0]))
                rows.append(dict(game_id=gid, season=y, week=g.week.iloc[0], team=d, opp=o, dst_pts=pts,
                                 sacks=sacks, hits=on_d.qb_hit.sum(), dropbacks=on_d.qb_dropback.sum(),
                                 takeaways=ints + fum_rec, plays=len(on_d), ya=ya, pa=pa, epa=on_d.epa.sum(),
                                 dst_tds=dst_tds))
        print('built', y, file=sys.stderr)
    return pd.DataFrame(rows)


def add_form(tg):
    """Exponentially weighted pre-game form for each team as a defense and as an offense."""
    tg = tg.sort_values(['season', 'week']).reset_index(drop=True)
    tg['order'] = tg.season * 100 + tg.week
    stats = ['dst_pts', 'sacks', 'hits', 'dropbacks', 'takeaways', 'plays', 'ya', 'pa', 'epa']

    def ew(df, key, cols, prefix):
        df = df.sort_values([key, 'order'])
        g = df.groupby(key)
        out = pd.DataFrame(index=df.index)
        for c in cols:
            out[prefix + c] = g[c].transform(lambda s: s.shift().ewm(halflife=HALFLIFE).mean())
        out['n_' + prefix] = g.cumcount()
        return out

    d_form = ew(tg, 'team', stats, 'd_')  # this defense's own recent games
    o_form = ew(tg, 'opp', stats, 'o_')   # the opponent offense's recent games (what it gave up to D/STs)
    tg = tg.join(d_form).join(o_form)
    return tg


def add_features(tg):
    games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'home_team', 'spread_line', 'total_line',
                                                     'roof', 'temp', 'wind', 'div_game', 'home_rest', 'away_rest'])
    games['home_team'] = games.home_team.replace(RELOCATE)
    f = tg.merge(games, on='game_id', how='left')
    home = f.team == f.home_team
    f['F_home'] = home.astype(float)
    f['F_opp_implied'] = np.where(home, (f.total_line - f.spread_line) / 2, (f.total_line + f.spread_line) / 2)
    f['F_spread'] = np.where(home, f.spread_line, -f.spread_line)
    f['F_total'] = f.total_line
    indoor = f.roof.isin(['dome', 'closed'])
    f['F_dome'] = indoor.astype(float)
    f['F_wind'] = np.where(indoor, 0, f.wind.fillna(f.wind.median()))
    f['F_temp'] = np.where(indoor, 70, f.temp.fillna(f.temp.median()))
    f['F_div'] = f.div_game.astype(float)
    f['F_rest_diff'] = np.where(home, f.home_rest - f.away_rest, f.away_rest - f.home_rest)
    # opponent offense form (what it allows to D/STs)
    f['F_o_dst_allowed'] = f.o_dst_pts
    f['F_o_sack_rate'] = f.o_sacks / f.o_dropbacks
    f['F_o_hit_rate'] = f.o_hits / f.o_dropbacks
    f['F_o_giveaway_rate'] = f.o_takeaways / f.o_plays
    f['F_o_yards'] = f.o_ya
    f['F_o_points'] = f.o_pa
    f['F_o_epa'] = f.o_epa / f.o_plays
    # this defense's form
    f['F_d_dst_pts'] = f.d_dst_pts
    f['F_d_sack_rate'] = f.d_sacks / f.d_dropbacks
    f['F_d_hit_rate'] = f.d_hits / f.d_dropbacks
    f['F_d_takeaway_rate'] = f.d_takeaways / f.d_plays
    f['F_d_yards_allowed'] = f.d_ya
    f['F_d_epa_allowed'] = f.d_epa / f.d_plays
    # structural: expected bracket points under the league's step functions
    f['F_exp_pa_pts'] = expected_bracket(f.F_opp_implied.to_numpy(), 9.5, pa_score, np.arange(0, 80))
    mu_yards = 0.5 * f.o_ya + 0.5 * f.d_ya
    f['F_exp_ya_pts'] = expected_bracket(mu_yards.fillna(330).to_numpy(), 80.0, ya_score, np.arange(0, 800, 5))
    return f


if MODE == 'build':
    tg = build_games()
    tg = add_form(tg)
    f = add_features(tg)
    f.to_parquet(OUT)
    print(f'wrote {OUT}: {len(f)} rows; mean pts {f.dst_pts.mean():.2f}')
    sys.exit()

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if MODE == 'predict':
    # usage: dst_v2.py predict <data_dir> <season> <week> <forecast_wind.csv: team,wind>
    season, week, wind_csv = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    hist = pd.read_parquet(OUT)
    raw = ['game_id', 'season', 'week', 'team', 'opp', 'dst_pts', 'sacks', 'hits', 'dropbacks',
           'takeaways', 'plays', 'ya', 'pa', 'epa', 'dst_tds']
    tg = hist[raw]
    sched = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'season', 'game_type', 'week', 'home_team', 'away_team'])
    sched = sched[(sched.season == season) & (sched.week == week) & (sched.game_type == 'REG')]
    pseudo = pd.concat([sched.assign(team=sched.home_team, opp=sched.away_team),
                        sched.assign(team=sched.away_team, opp=sched.home_team)])[['game_id', 'season', 'week', 'team', 'opp']]
    pseudo[['team', 'opp']] = pseudo[['team', 'opp']].replace(RELOCATE)
    fut = add_features(add_form(pd.concat([tg, pseudo], ignore_index=True)))
    fut = fut[(fut.season == season) & (fut.week == week)].copy()
    team_to_wind = pd.read_csv(wind_csv).set_index('team').wind
    outdoor = fut.F_dome == 0
    home_team_wind = fut.home_team.map(team_to_wind)
    fut.loc[outdoor & home_team_wind.notna(), 'F_wind'] = home_team_wind[outdoor & home_team_wind.notna()]
    feats = [c for c in hist.columns if c.startswith('F_')]
    train = hist[(hist.n_d_ >= 3) & (hist.n_o_ >= 3)].dropna(subset=feats + ['dst_pts'])
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(train[feats], train.dst_pts)
    struct_cols = ['F_exp_pa_pts', 'F_exp_ya_pts', 'F_spread']
    struct = make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-6])).fit(train[struct_cols], train.dst_pts)
    fut['ridge_v2'] = ridge.predict(fut[feats])
    fut['structural'] = struct.predict(fut[struct_cols])
    fut['v1'] = 18.15 - 0.553 * fut.F_opp_implied + 0.035 * fut.F_spread
    print(fut[['team', 'opp', 'F_opp_implied', 'F_spread', 'F_wind', 'v1', 'structural', 'ridge_v2']]
          .sort_values('ridge_v2', ascending=False).round(2).to_string(index=False))
    sys.exit()

# ---------------- eval ----------------

f = pd.read_parquet(OUT)
FEATS = [c for c in f.columns if c.startswith('F_')]
f = f[(f.n_d_ >= 3) & (f.n_o_ >= 3)].dropna(subset=FEATS + ['dst_pts']).copy()  # needs a little history
NO_WEATHER = [c for c in FEATS if c not in ('F_wind', 'F_temp')]

MODELS = {
    'v1: implied + spread (current)': ('ols', ['F_opp_implied', 'F_spread']),
    'structural: exp PA + exp YA pts': ('ols', ['F_exp_pa_pts', 'F_exp_ya_pts', 'F_spread']),
    'ridge, all (no weather)': ('ridge', NO_WEATHER),
    'ridge, all + weather': ('ridge', FEATS),
    'boosting, all + weather': ('gbm', FEATS),
    'ridge+boost blend + weather': ('blend', FEATS),
}


def make(kind):
    if kind == 'ols':
        return make_pipeline(StandardScaler(), RidgeCV(alphas=[1e-6]))
    if kind == 'ridge':
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                         min_samples_leaf=80, l2_regularization=1.0, random_state=0)


preds = {m: pd.Series(np.nan, index=f.index) for m in MODELS}
for test_season in range(2019, 2027):
    tr, te = f[f.season < test_season], f[f.season == test_season]
    for name, (kind, cols) in MODELS.items():
        if kind == 'blend':
            p = 0.5 * make('ridge').fit(tr[cols], tr.dst_pts).predict(te[cols]) \
                + 0.5 * make('gbm').fit(tr[cols], tr.dst_pts).predict(te[cols])
        else:
            p = make(kind).fit(tr[cols], tr.dst_pts).predict(te[cols])
        preds[name].loc[te.index] = p
    print('fold', test_season, file=sys.stderr)

test = f[f.season >= 2019].copy()
weeks = list(test.groupby(['season', 'week']).groups.items())


def week_metrics(col):
    sp, gain = [], []
    for _, idx in weeks:
        g = test.loc[idx]
        if len(g) < 6:
            continue
        sp.append(g[col].corr(g.dst_pts, method='spearman'))
        gain.append(g.nlargest(5, col).dst_pts.mean() - g.dst_pts.mean())
    return np.array(sp), np.array(gain)


rows, base_sp = [], None
rng = np.random.default_rng(0)
for name in MODELS:
    test['p'] = preds[name].loc[test.index]
    sp, gain = week_metrics('p')
    if base_sp is None:
        base_sp, base_gain = sp, gain
    diff = sp - base_sp
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    rows.append(dict(model=name, weekly_spearman=sp.mean(), top5_gain=gain.mean(),
                     mae=(test.p - test.dst_pts).abs().mean(),
                     vs_v1=diff.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5)))
    # how many of the 7 test seasons this model beats v1 in
    test['b'] = preds['v1: implied + spread (current)'].loc[test.index]
    season_sp = lambda col, t: np.mean([g[col].corr(g.dst_pts, method='spearman') for _, g in t.groupby('week') if len(g) > 5])
    wins = sum(season_sp('p', test[test.season == s]) > season_sp('b', test[test.season == s]) for s in range(2019, 2027))
    rows[-1]['seasons_beating_v1'] = f'{wins}/8'
print(f'\nWALK-FORWARD 2019-2025 (each season predicted from earlier seasons only), {len(test)} team-games')
print(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))

# feature importance proxy: ridge standardized coefficients on all data
r = make('ridge').fit(f[FEATS], f.dst_pts)
coef = pd.Series(r[-1].coef_, index=[c[2:] for c in FEATS]).sort_values(key=abs, ascending=False)
print('\nridge standardized coefficients (all seasons):')
print(coef.round(3).to_string())

# save walk-forward predictions for comparison with FantasyPros ECR
saved = test[['game_id', 'season', 'week', 'team', 'opp', 'dst_pts']].copy()
for name in MODELS:
    saved[name] = preds[name].loc[test.index]
saved.to_parquet(DATA.parent / 'dst_v2_walkforward.parquet')
