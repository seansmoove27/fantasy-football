"""Kicker feature experiments on top of kicker/k_model.py's ridge (see harness.py for the protocol).

Ideas:
  coach   - how often the head coach kicks on 4th down in long-FG range (opp 20-45) instead of going or punting
  kicker  - the kicker's own recent points, FG% and 50+ attempt rate (follows the kicker, not the team)
  temp    - game temperature (indoor = 70)

Usage: python -I k_experiments.py <data_dir> [dev|holdout]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from harness import DEV_SEASONS, HOLDOUT_SEASONS, scorecard, walk_forward  # noqa: E402

DATA = Path(sys.argv[1])
STAGE = sys.argv[2] if len(sys.argv) > 2 else 'dev'
EXTRAS = DATA.parent / 'k_extras.parquet'
YEARS = range(2015, 2027)
RELOCATE = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}


def build_extras():
    rows = []
    for y in YEARS:
        pbp = pd.read_parquet(DATA / f'pbp_{y}.parquet', columns=[
            'game_id', 'season_type', 'home_team', 'away_team', 'posteam', 'home_coach', 'away_coach', 'down',
            'yardline_100', 'play_type', 'field_goal_attempt', 'field_goal_result', 'kick_distance',
            'extra_point_attempt', 'kicker_player_id'])
        pbp = pbp[pbp.season_type == 'REG']
        for c in ('home_team', 'away_team', 'posteam'):
            pbp[c] = pbp[c].replace(RELOCATE)
        fourth = pbp[(pbp.down == 4) & pbp.yardline_100.between(20, 45)
                     & pbp.play_type.isin(['field_goal', 'pass', 'run', 'punt'])]
        kicks = pbp[(pbp.field_goal_attempt == 1) | (pbp.extra_point_attempt == 1)]
        for gid, g in pbp.groupby('game_id'):
            home = g.home_team.iloc[0]
            coach = {home: g.home_coach.iloc[0], g.away_team.iloc[0]: g.away_coach.iloc[0]}
            for team in coach:
                d = fourth[(fourth.game_id == gid) & (fourth.posteam == team)]
                k = kicks[(kicks.game_id == gid) & (kicks.posteam == team)]
                fga = k[k.field_goal_attempt == 1]
                rows.append(dict(game_id=gid, team=team, coach=coach[team],
                                 dec_n=len(d), dec_kick=(d.play_type == 'field_goal').sum(),
                                 kicker_id=k.kicker_player_id.mode().iloc[0] if len(k) else None,
                                 fga=len(fga), fgm=(fga.field_goal_result == 'made').sum(),
                                 fga50=(fga.kick_distance >= 50).sum()))
        print('extras', y, file=sys.stderr)
    return pd.DataFrame(rows)


if not EXTRAS.exists():
    build_extras().to_parquet(EXTRAS)
f = pd.read_parquet(DATA.parent / 'k_model_features.parquet').merge(pd.read_parquet(EXTRAS), on=['game_id', 'team'])
f = f.sort_values(['season', 'week']).reset_index(drop=True)


def shrunk_rate(df, key, num, den, window, prior, prior_n):
    """sum(num)/sum(den) over the key's previous `window` games, shrunk toward prior with prior_n pseudo-counts."""
    g = df.groupby(key)
    n = g[num].transform(lambda s: s.shift().rolling(window, min_periods=1).sum()).fillna(0)
    d = g[den].transform(lambda s: s.shift().rolling(window, min_periods=1).sum()).fillna(0)
    return (n + prior * prior_n) / (d + prior_n)


ew = lambda s, hl: s.shift().ewm(halflife=hl, min_periods=1).mean()
f['F_coach_kick_rate'] = shrunk_rate(f, 'coach', 'dec_kick', 'dec_n', 48, f.dec_kick.sum() / f.dec_n.sum(), 10)
f['kicker_id'] = f.kicker_id.fillna(f.team + '_unknown')
f['F_kicker_form'] = f.groupby('kicker_id').k_pts.transform(lambda s: ew(s, 6)).fillna(f.k_pts.mean())
f['F_kicker_fg_pct'] = shrunk_rate(f, 'kicker_id', 'fgm', 'fga', 32, f.fgm.sum() / f.fga.sum(), 15)
f['F_kicker_long_rate'] = f.groupby('kicker_id').fga50.transform(lambda s: ew(s, 8)).fillna(f.fga50.mean())
games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'temp'])
f = f.merge(games, on='game_id', how='left')
f['F_temp'] = np.where(f.F_dome == 1, 70, f.temp.fillna(f.temp.median()))

BASE = ['F_team_implied', 'F_spread', 'F_home', 'F_dome', 'F_wind', 'F_team_k_form', 'F_opp_k_allowed']
IDEAS = {
    'coach': ['F_coach_kick_rate'],
    'kicker': ['F_kicker_form', 'F_kicker_fg_pct', 'F_kicker_long_rate'],
    'temp': ['F_temp'],
}
f = f.dropna(subset=BASE + sum(IDEAS.values(), []) + ['k_pts'])
seasons = DEV_SEASONS if STAGE == 'dev' else HOLDOUT_SEASONS
f = f[f.season <= max(seasons)].reset_index(drop=True)

sets = {'baseline (k_model ridge)': BASE, **{f'+ {k}': BASE + v for k, v in IDEAS.items()},
        '+ all ideas': BASE + sum(IDEAS.values(), [])}
if len(sys.argv) > 3:  # explicit accepted set for the holdout check, e.g. coach,kicker
    sets['+ accepted'] = BASE + sum((IDEAS[k] for k in sys.argv[3].split(',')), [])
preds = {name: walk_forward(f, cols, 'k_pts', seasons) for name, cols in sets.items()}
test = f[f.season.isin(seasons)]
preds = {k: v[test.index] for k, v in preds.items()}
print(f'\nKICKER {STAGE} {min(seasons)}-{max(seasons)}: {len(test)} team-games')
print(scorecard(test, 'k_pts', preds, 'baseline (k_model ridge)', seasons)
      .to_string(index=False, float_format='%.3f'))
print('\nsingle-feature weekly Spearman (dev rows):')
for c in sum(IDEAS.values(), []):
    w = [g[c].corr(g.k_pts, method='spearman') for _, g in test.groupby(['season', 'week']) if len(g) > 5]
    print(f'  {c}: {np.mean(w):.3f}')
