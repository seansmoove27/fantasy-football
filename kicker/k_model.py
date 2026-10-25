"""Kicker model for the Limers league scoring: build team-game table, walk-forward compare models, weekly board.

Scoring: PAT 1, FG 0-39 3, 40-49 4, 50-59 5, 60+ 6, missed FG -1.
Usage:
  python -I k_model.py build   <data_dir>                                    # cache team-game kicker points + features
  python -I k_model.py eval    <data_dir>                                    # walk-forward comparison of candidate models
  python -I k_model.py predict <data_dir> <season> <week> [wind.csv: team,wind]  # weekly board
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODE, DATA = sys.argv[1], Path(sys.argv[2])
OUT = DATA.parent / 'k_model_features.parquet'
YEARS = range(2015, 2027)
RELOCATE = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
HALFLIFE = 6  # games; recent form carries across seasons and decays


def fg_points(dist):
    if dist >= 60: return 6
    if dist >= 50: return 5
    if dist >= 40: return 4
    return 3


def build_games():
    rows = []
    for y in YEARS:
        pbp = pd.read_parquet(DATA / f'pbp_{y}.parquet', columns=[
            'game_id', 'season', 'season_type', 'week', 'home_team', 'away_team', 'posteam',
            'field_goal_attempt', 'field_goal_result', 'kick_distance', 'extra_point_result'])
        pbp = pbp[pbp.season_type == 'REG']
        for game_id, g in pbp.groupby('game_id'):
            home, away = g.home_team.iloc[0], g.away_team.iloc[0]
            for team, opp in ((home, away), (away, home)):
                t = g[g.posteam == team]
                fga = t[t.field_goal_attempt == 1]
                made = fga[fga.field_goal_result == 'made']
                pts = sum(fg_points(d) for d in made.kick_distance) - (len(fga) - len(made)) \
                    + (t.extra_point_result == 'good').sum()
                rows.append(dict(game_id=game_id, season=g.season.iloc[0], week=g.week.iloc[0],
                                 team=team, opp=opp, k_pts=pts))
        print('built', y, file=sys.stderr)
    return pd.DataFrame(rows)


def add_features(tg):
    games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'home_team', 'spread_line', 'total_line', 'roof', 'wind'])
    # future games can be missing roof; use the stadium's most recent known roof
    last_roof = games.dropna(subset=['roof']).groupby('home_team').roof.last()
    games['roof'] = games.roof.fillna(games.home_team.map(last_roof))
    f = tg.merge(games, on='game_id')
    home = f.team == f.home_team
    f['F_team_implied'] = np.where(home, (f.total_line + f.spread_line) / 2, (f.total_line - f.spread_line) / 2)
    f['F_spread'] = np.where(home, f.spread_line, -f.spread_line)
    f['F_home'] = home.astype(float)
    indoor = f.roof.isin(['dome', 'closed'])
    f['F_dome'] = indoor.astype(float)
    f['F_wind'] = np.where(indoor, 0, f.wind.fillna(f.wind.median()))
    # recent kicker scoring by the team and allowed by the opponent, shifted so a game never sees itself
    f = f.sort_values(['season', 'week'])
    ew = lambda s: s.shift().ewm(halflife=HALFLIFE, min_periods=3).mean()
    f['F_team_k_form'] = f.groupby('team').k_pts.transform(ew)
    f['F_opp_k_allowed'] = f.groupby('opp').k_pts.transform(ew)
    return f


if MODE == 'build':
    tg = build_games()
    tg[['team', 'opp']] = tg[['team', 'opp']].replace(RELOCATE)
    f = add_features(tg)
    f.to_parquet(OUT)
    print(f'wrote {OUT}: {len(f)} rows; mean pts {f.k_pts.mean():.2f}')
    sys.exit()

hist = pd.read_parquet(OUT)
FEATS = [c for c in hist.columns if c.startswith('F_')]
MODELS = {
    'implied only': ['F_team_implied'],
    'implied + dome': ['F_team_implied', 'F_dome'],
    'implied + dome + wind': ['F_team_implied', 'F_dome', 'F_wind'],
    'ridge, all': FEATS,
}
BEST = 'ridge, all'


def make(cols):
    alphas = np.logspace(-2, 4, 25) if len(cols) > 3 else [1e-6]
    return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))


train_all = hist.dropna(subset=FEATS + ['k_pts'])

# ESPN proTeamId -> nflverse team abbreviation
ESPN_TEAMS = {1: 'ATL', 2: 'BUF', 3: 'CHI', 4: 'CIN', 5: 'CLE', 6: 'DAL', 7: 'DEN', 8: 'DET', 9: 'GB', 10: 'TEN',
              11: 'IND', 12: 'KC', 13: 'LV', 14: 'LA', 15: 'MIA', 16: 'MIN', 17: 'NE', 18: 'NO', 19: 'NYG', 20: 'NYJ',
              21: 'PHI', 22: 'ARI', 23: 'PIT', 24: 'LAC', 25: 'SF', 26: 'SEA', 27: 'TB', 28: 'WAS', 29: 'CAR',
              30: 'JAX', 33: 'BAL', 34: 'HOU'}


def espn_kickers(season, week):
    """Current NFL kickers per team from ESPN's public player list, most-rostered first."""
    import json
    import urllib.request
    filt = {'filterActive': {'value': True}, 'filterSlotIds': {'value': [17]}, 'limit': 300,
            'sortPercOwned': {'sortPriority': 1, 'sortAsc': False}}
    req = urllib.request.Request(
        f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/players'
        f'?scoringPeriodId={week}&view=players_wl', headers={'x-fantasy-filter': json.dumps(filt)})
    with urllib.request.urlopen(req, timeout=20) as r:
        players = json.load(r)
    by_team = {}
    for p in players:
        team = ESPN_TEAMS.get(p['proTeamId'])
        if team:
            by_team.setdefault(team, []).append(p['fullName'])
    return {t: ', '.join(names) for t, names in by_team.items()}

if MODE == 'predict':
    season, week = int(sys.argv[3]), int(sys.argv[4])
    sched = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'season', 'game_type', 'week', 'home_team', 'away_team'])
    sched = sched[(sched.season == season) & (sched.week == week) & (sched.game_type == 'REG')]
    pseudo = pd.concat([sched.assign(team=sched.home_team, opp=sched.away_team),
                        sched.assign(team=sched.away_team, opp=sched.home_team)])[['game_id', 'season', 'week', 'team', 'opp']]
    pseudo[['team', 'opp']] = pseudo[['team', 'opp']].replace(RELOCATE)
    base = hist[['game_id', 'season', 'week', 'team', 'opp', 'k_pts']]
    fut = add_features(pd.concat([base, pseudo], ignore_index=True))
    fut = fut[(fut.season == season) & (fut.week == week)].copy()
    if len(sys.argv) > 5:
        team_to_wind = pd.read_csv(sys.argv[5]).set_index('team').wind
        w = fut.home_team.map(team_to_wind)
        fut.loc[(fut.F_dome == 0) & w.notna(), 'F_wind'] = w
    for name, cols in MODELS.items():
        fut[name] = make(cols).fit(train_all[cols], train_all.k_pts).predict(fut[cols])
    # predictions are per team; who kicked last can differ from who is on the roster after a mid-week signing
    pbp = pd.read_parquet(DATA / f'pbp_{season}.parquet',
                          columns=['posteam', 'kicker_player_name', 'field_goal_attempt', 'extra_point_attempt'])
    kicks = pbp[(pbp.field_goal_attempt == 1) | (pbp.extra_point_attempt == 1)]
    fut['last_kicker'] = fut.team.map(kicks.groupby('posteam').kicker_player_name.last())
    cols = ['team', 'last_kicker', 'opp', 'F_team_implied', 'F_dome', 'F_wind'] + list(MODELS)
    try:
        fut['espn_kicker'] = fut.team.map(espn_kickers(season, week)).fillna('')
        last_name = fut.last_kicker.fillna('').str.split('.').str[-1]
        changed = [ln not in names for ln, names in zip(last_name, fut.espn_kicker)]
        fut['change'] = np.where(changed, 'CHANGED', '')
        cols[2:2] = ['espn_kicker', 'change']
    except OSError as e:
        print(f'ESPN kicker check failed ({e}); confirm kickers by hand', file=sys.stderr)
    print(fut[cols].sort_values(BEST, ascending=False).round(2).to_string(index=False))
    if 'change' in fut and (fut.change != '').any():
        print('\nCHANGED: the team kicker on ESPN differs from who kicked last; the projection follows the team.')
    sys.exit()

# ---------------- eval ----------------

f = train_all.copy()
preds = {m: pd.Series(np.nan, index=f.index) for m in MODELS}
for test_season in range(2019, 2027):
    tr, te = f[f.season < test_season], f[f.season == test_season]
    for name, cols in MODELS.items():
        preds[name].loc[te.index] = make(cols).fit(tr[cols], tr.k_pts).predict(te[cols])

test = f[f.season >= 2019].copy()
weeks = [idx for _, idx in test.groupby(['season', 'week']).groups.items() if len(idx) >= 6]
rng = np.random.default_rng(0)
rows, base_sp = [], None
for name in MODELS:
    test['p'] = preds[name].loc[test.index]
    sp = np.array([test.loc[i, 'p'].corr(test.loc[i, 'k_pts'], method='spearman') for i in weeks])
    gain = np.array([test.loc[i].nlargest(5, 'p').k_pts.mean() - test.loc[i, 'k_pts'].mean() for i in weeks])
    if base_sp is None:
        base_sp = sp
    diff = sp - base_sp
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    rows.append(dict(model=name, weekly_spearman=sp.mean(), top5_gain=gain.mean(),
                     mae=(test.p - test.k_pts).abs().mean(), vs_implied=diff.mean(),
                     ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5)))
print(f'\nWALK-FORWARD 2019-2026 (each season predicted from earlier seasons only), {len(test)} team-games, {len(weeks)} weeks')
print(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))
coef = make(MODELS[BEST]).fit(f[MODELS[BEST]], f.k_pts)[-1].coef_
print(f'\n{BEST} standardized coefficients:', dict(zip(MODELS[BEST], coef.round(3))))

# save walk-forward predictions for comparison with FantasyPros ECR
saved = test[['game_id', 'season', 'week', 'team', 'opp', 'k_pts']].copy()
for name in MODELS:
    saved[name] = preds[name].loc[test.index]
saved.to_parquet(DATA.parent / 'k_model_walkforward.parquet')
