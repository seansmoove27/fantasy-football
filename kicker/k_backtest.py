"""Backtest which pre-game indicators best predict kicker points under the Limers league scoring.

Scoring: PAT 1, FG 0-39 3, 40-49 4, 50-59 5, 60+ 6, missed FG -1.
Usage: python -I k_backtest.py <data_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
YEARS = range(2018, 2026)


def fg_points(dist):
    if dist >= 60: return 6
    if dist >= 50: return 5
    if dist >= 40: return 4
    return 3


rows = []
for y in YEARS:
    pbp = pd.read_parquet(DATA / f'pbp_{y}.parquet', columns=[
        'game_id', 'season', 'season_type', 'week', 'home_team', 'away_team', 'posteam', 'defteam', 'roof',
        'field_goal_attempt', 'field_goal_result', 'kick_distance', 'extra_point_result', 'drive', 'fixed_drive_result',
        'yardline_100', 'total_home_score', 'total_away_score'])
    pbp = pbp[pbp.season_type == 'REG']
    for game_id, g in pbp.groupby('game_id'):
        home, away = g.home_team.iloc[0], g.away_team.iloc[0]
        final = g.iloc[-1]
        score = {home: final.total_home_score, away: final.total_away_score}
        dome = g.roof.iloc[0] in ('dome', 'closed')
        for team, opp in ((home, away), (away, home)):
            t = g[g.posteam == team]
            fga = t[t.field_goal_attempt == 1]
            made = fga[fga.field_goal_result == 'made']
            pts = sum(fg_points(d) for d in made.kick_distance) - (len(fga) - len(made)) + (t.extra_point_result == 'good').sum()
            # red-zone trips: drives that reached the opponent 20; TD share of those trips
            rz = t[t.yardline_100 <= 20].drop_duplicates('drive')
            rz_trips, rz_tds = len(rz), (rz.fixed_drive_result == 'Touchdown').sum()
            rows.append(dict(game_id=game_id, season=g.season.iloc[0], week=g.week.iloc[0], team=team, opp=opp,
                             k_pts=pts, fga=len(fga), pts_scored=score[team], rz_trips=rz_trips, rz_tds=rz_tds, dome=dome))
    print('built', y, file=sys.stderr)
kg = pd.DataFrame(rows).sort_values(['season', 'team', 'week'])

# what each defense gave up to opposing kickers
allowed = kg[['game_id', 'season', 'week', 'opp', 'k_pts', 'fga', 'pts_scored', 'rz_trips', 'rz_tds']].rename(
    columns={'opp': 'team', 'k_pts': 'a_k_pts', 'fga': 'a_fga', 'pts_scored': 'a_pts', 'rz_trips': 'a_rz_trips', 'rz_tds': 'a_rz_tds'})


def season_to_date(df, cols):
    df = df.sort_values(['season', 'team', 'week']).copy()
    g = df.groupby(['season', 'team'])
    for c in cols:
        df['std_' + c] = g[c].transform(lambda s: s.shift().expanding().sum())
    df['std_n'] = g.cumcount()
    return df


k = season_to_date(kg, ['k_pts', 'fga', 'pts_scored', 'rz_trips', 'rz_tds'])
a = season_to_date(allowed, ['a_k_pts', 'a_fga', 'a_pts', 'a_rz_trips', 'a_rz_tds'])
a = a[['game_id', 'team'] + [c for c in a.columns if c.startswith('std_')]].rename(
    columns=lambda c: c if c == 'game_id' else 'opp_' + c).rename(columns={'opp_team': 'opp'})
f = k.merge(a, on=['game_id', 'opp'])

n, on = f.std_n, f.opp_std_n
f['F_team_k_ppg'] = f.std_k_pts / n
f['F_team_fga_pg'] = f.std_fga / n
f['F_team_pts_pg'] = f.std_pts_scored / n
f['F_team_rz_stall'] = 1 - f.std_rz_tds / f.std_rz_trips
f['F_opp_k_allowed_pg'] = f.opp_std_a_k_pts / on
f['F_opp_fga_allowed_pg'] = f.opp_std_a_fga / on
f['F_opp_pts_allowed_pg'] = f.opp_std_a_pts / on
f['F_opp_rz_stall'] = 1 - f.opp_std_a_rz_tds / f.opp_std_a_rz_trips
f['F_dome'] = f.dome.astype(float)

games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'home_team', 'spread_line', 'total_line'])
f = f.merge(games, on='game_id')
is_home = f.team == f.home_team
f['F_team_implied_total'] = np.where(is_home, (f.total_line + f.spread_line) / 2, (f.total_line - f.spread_line) / 2)
f['F_spread'] = np.where(is_home, f.spread_line, -f.spread_line)
f['F_game_total'] = f.total_line

FEATURES = [c for c in f.columns if c.startswith('F_')]
s = f[(f.week >= 4) & (f.std_n >= 3) & (f.opp_std_n >= 3)].dropna(subset=FEATURES + ['k_pts'])
print(f'\nsample: {len(s)} team-games, weeks 4+; kicker pts mean {s.k_pts.mean():.2f}, sd {s.k_pts.std():.2f}')


def weekly_spearman(df, col):
    return np.mean([g[col].corr(g.k_pts, method='spearman') for _, g in df.groupby(['season', 'week']) if len(g) > 5])


def top_gain(df, col, k=5):
    return np.mean([g.nlargest(k, col).k_pts.mean() - g.k_pts.mean() for _, g in df.groupby(['season', 'week'])])


print('\nSINGLE INDICATORS')
print(pd.DataFrame([dict(indicator=c[2:], weekly_spearman=weekly_spearman(s, c), pearson_r=s[c].corr(s.k_pts),
                         top5_gain=top_gain(s, c)) for c in FEATURES])
      .sort_values('weekly_spearman', ascending=False).to_string(index=False, float_format='%.3f'))

train, test = s[s.season <= 2022], s[s.season >= 2023].copy()


def fit(cols, df=train):
    X = np.column_stack([np.ones(len(df))] + [df[c] for c in cols])
    return np.linalg.lstsq(X, df.k_pts, rcond=None)[0]


MODELS = {
    'implied total only': ['F_team_implied_total'],
    'implied + dome': ['F_team_implied_total', 'F_dome'],
    'implied + dome + opp rz stall': ['F_team_implied_total', 'F_dome', 'F_opp_rz_stall'],
    'implied + dome + team k ppg + opp k allowed': ['F_team_implied_total', 'F_dome', 'F_team_k_ppg', 'F_opp_k_allowed_pg'],
    'season stats only (no Vegas)': [c for c in FEATURES if c not in ('F_team_implied_total', 'F_spread', 'F_game_total')],
    'everything': FEATURES,
}
print('\nOUT-OF-SAMPLE (train 2018-22, test 2023-25)')
out = []
for name, cols in MODELS.items():
    b = fit(cols)
    test['pred'] = np.column_stack([np.ones(len(test))] + [test[c] for c in cols]) @ b
    r = test.pred.corr(test.k_pts)
    out.append(dict(model=name, weekly_spearman=weekly_spearman(test, 'pred'), r=r, r2=r * r, top5_gain=top_gain(test, 'pred')))
print(pd.DataFrame(out).to_string(index=False, float_format='%.3f'))

b = fit(['F_team_implied_total', 'F_dome'], s)
print('\nfull-sample model: pts = %.2f + %.3f*team_implied + %.2f*dome' % tuple(b))
print(s.groupby(pd.cut(s.F_team_implied_total, [0, 17, 19, 21, 23, 25, 27, 29, 40])).k_pts.agg(['mean', 'count']).round(2).to_string())
