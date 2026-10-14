"""Backtest which pre-game indicators best predict D/ST points under the Limers league scoring.

Usage: python -I dst_backtest.py <data_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
YEARS = range(2018, 2026)


def points_allowed_score(pa):
    if pa == 0: return 5
    if pa <= 6: return 4
    if pa <= 13: return 3
    if pa <= 17: return 1
    if pa <= 27: return 0
    if pa <= 34: return -1
    if pa <= 45: return -3
    return -5


def yards_allowed_score(ya):
    if ya < 100: return 5
    if ya < 200: return 3
    if ya < 300: return 2
    if ya < 350: return 0
    if ya < 400: return -1
    if ya < 450: return -3
    if ya < 500: return -5
    if ya < 550: return -6
    return -7


def team_games(pbp):
    """One row per (game, defense) with that defense's fantasy points and raw stats."""
    pbp = pbp[pbp.season_type == 'REG']
    scrimmage = pbp[pbp.play_type.isin(['pass', 'run']) & (pbp.two_point_attempt != 1)]
    rows = []
    for game_id, g in pbp.groupby('game_id'):
        home, away = g.home_team.iloc[0], g.away_team.iloc[0]
        final = g.iloc[-1]
        score = {home: final.total_home_score, away: final.total_away_score}
        sg = scrimmage[scrimmage.game_id == game_id]
        for d, opp in ((home, away), (away, home)):
            on_d = sg[sg.defteam == d]
            all_d = g[g.defteam == d]
            sacks = on_d.sack.sum()
            ints = all_d.interception.sum()
            fum_rec = all_d.fumble_lost.sum()
            safeties = all_d.safety.sum()
            blocks = (all_d.punt_blocked.fillna(0).sum()
                      + (all_d.field_goal_result == 'blocked').sum()
                      + (all_d.extra_point_result == 'blocked').sum())
            tds = g[(g.touchdown == 1) & (g.td_team == d)]
            # D/ST TDs: kickoff returns, plus any TD scored by the team without the ball
            dst_tds = ((tds.play_type == 'kickoff') | (tds.posteam != d)).sum()
            yards_allowed = on_d.yards_gained.sum()
            pa = score[opp]
            pts = (sacks + 2 * (ints + fum_rec + safeties + blocks) + 6 * dst_tds
                   + points_allowed_score(pa) + yards_allowed_score(yards_allowed))
            rows.append(dict(
                game_id=game_id, season=g.season.iloc[0], week=g.week.iloc[0], team=d, opp=opp,
                dst_pts=pts, sacks=sacks, takeaways=ints + fum_rec, pa=pa, ya=yards_allowed,
                d_dropbacks=on_d.qb_dropback.sum(), d_hits=on_d.qb_hit.sum(),
                d_plays=len(on_d), d_epa=on_d.epa.sum(),
            ))
    return pd.DataFrame(rows)


frames = []
for y in YEARS:
    pbp = pd.read_parquet(DATA / f'pbp_{y}.parquet', columns=[
        'game_id', 'season', 'season_type', 'week', 'home_team', 'away_team', 'posteam', 'defteam',
        'play_type', 'two_point_attempt', 'sack', 'qb_hit', 'qb_dropback', 'interception', 'fumble_lost',
        'safety', 'punt_blocked', 'field_goal_result', 'extra_point_result', 'touchdown', 'td_team',
        'yards_gained', 'epa', 'total_home_score', 'total_away_score'])
    frames.append(team_games(pbp))
    print('built', y, file=sys.stderr)
tg = pd.concat(frames, ignore_index=True).sort_values(['season', 'team', 'week'])

# Offense-side view of the same games: what each offense gives up to opposing D/STs
off = tg[['game_id', 'season', 'week', 'opp', 'dst_pts', 'sacks', 'takeaways', 'pa', 'ya', 'd_dropbacks', 'd_plays', 'd_epa']]
off = off.rename(columns={'opp': 'team', 'dst_pts': 'o_dst_allowed', 'sacks': 'o_sacks', 'takeaways': 'o_giveaways',
                          'pa': 'o_pts', 'ya': 'o_yds', 'd_dropbacks': 'o_dropbacks', 'd_plays': 'o_plays', 'd_epa': 'o_epa'})
# 'team' here is the offense; columns describe what it gave up to opposing D/STs


def season_to_date(df, team_col, cols):
    """Mean of each column over the team's prior games this season (excludes the current game)."""
    df = df.sort_values(['season', team_col, 'week']).copy()
    g = df.groupby(['season', team_col])
    for c in cols:
        df['std_' + c] = g[c].transform(lambda s: s.shift().expanding().sum())
    df['std_n'] = g.cumcount()
    return df


tg['d_sack_rate'] = tg.sacks / tg.d_dropbacks.replace(0, np.nan)
d_std = season_to_date(tg, 'team', ['dst_pts', 'sacks', 'takeaways', 'pa', 'ya', 'd_dropbacks', 'd_hits', 'd_plays', 'd_epa'])
o_std = season_to_date(off, 'team', ['o_dst_allowed', 'o_sacks', 'o_giveaways', 'o_pts', 'o_yds', 'o_dropbacks', 'o_plays', 'o_epa'])

feat = d_std.merge(
    o_std[['game_id', 'team'] + [c for c in o_std.columns if c.startswith('std_')]].rename(
        columns=lambda c: c if c in ('game_id', 'team') else 'opp_' + c).rename(columns={'team': 'opp'}),
    on=['game_id', 'opp'])

n = feat.std_n
on = feat.opp_std_n
feat['F_def_dst_ppg'] = feat.std_dst_pts / n
feat['F_def_sack_rate'] = feat.std_sacks / feat.std_d_dropbacks
feat['F_def_hit_rate'] = feat.std_d_hits / feat.std_d_dropbacks
feat['F_def_takeaways_pg'] = feat.std_takeaways / n
feat['F_def_pa_pg'] = -feat.std_pa / n
feat['F_def_ya_pg'] = -feat.std_ya / n
feat['F_def_epa_play'] = -feat.std_d_epa / feat.std_d_plays
feat['F_opp_dst_allowed_pg'] = feat.opp_std_o_dst_allowed / on
feat['F_opp_sack_rate'] = feat.opp_std_o_sacks / feat.opp_std_o_dropbacks
feat['F_opp_giveaways_pg'] = feat.opp_std_o_giveaways / on
feat['F_opp_pts_pg'] = -feat.opp_std_o_pts / on
feat['F_opp_yds_pg'] = -feat.opp_std_o_yds / on
feat['F_opp_epa_play'] = -feat.opp_std_o_epa / feat.opp_std_o_plays

# Betting lines: nflverse spread_line is the home team's expected margin
games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'home_team', 'away_team', 'spread_line', 'total_line'])
feat = feat.merge(games, on='game_id')
is_home = feat.team == feat.home_team
home_implied = (feat.total_line + feat.spread_line) / 2
away_implied = (feat.total_line - feat.spread_line) / 2
feat['F_opp_implied_total'] = -np.where(is_home, away_implied, home_implied)  # lower opp total = better
feat['F_spread'] = np.where(is_home, feat.spread_line, -feat.spread_line)      # D's team expected margin
feat['F_game_total'] = -feat.total_line

FEATURES = [c for c in feat.columns if c.startswith('F_')]
sample = feat[(feat.week >= 4) & (feat.std_n >= 3) & (feat.opp_std_n >= 3)].dropna(subset=FEATURES + ['dst_pts'])
print(f'\nsample: {len(sample)} team-games, {sample.season.min()}-{sample.season.max()}, weeks 4+')
print(f'D/ST points: mean {sample.dst_pts.mean():.2f}, sd {sample.dst_pts.std():.2f}')


def weekly_spearman(df, col):
    """Average within-week rank correlation: how well the indicator orders that week's D/STs."""
    vals = [g[col].corr(g.dst_pts, method='spearman') for _, g in df.groupby(['season', 'week']) if len(g) > 5]
    return np.mean(vals)


def top_pick_gain(df, col, k=5):
    """Avg actual pts of each week's top-k by the indicator, minus that week's average."""
    gains = []
    for _, g in df.groupby(['season', 'week']):
        top = g.nlargest(k, col)
        gains.append(top.dst_pts.mean() - g.dst_pts.mean())
    return np.mean(gains)


print('\nSINGLE INDICATORS (all seasons)')
res = pd.DataFrame([dict(indicator=c[2:], weekly_spearman=weekly_spearman(sample, c),
                         pearson_r=sample[c].corr(sample.dst_pts), top5_gain=top_pick_gain(sample, c))
                    for c in FEATURES]).sort_values('weekly_spearman', ascending=False)
print(res.to_string(index=False, float_format='%.3f'))

# Out-of-sample models: train 2018-2022, test 2023-2025
train, test = sample[sample.season <= 2022], sample[sample.season >= 2023].copy()


def fit_predict(cols):
    X = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
    beta, *_ = np.linalg.lstsq(X, train.dst_pts, rcond=None)
    Xt = np.column_stack([np.ones(len(test))] + [test[c] for c in cols])
    return Xt @ beta, beta


MODELS = {
    'implied total only': ['F_opp_implied_total'],
    'implied + spread': ['F_opp_implied_total', 'F_spread'],
    'implied + D dst ppg + opp dst allowed': ['F_opp_implied_total', 'F_def_dst_ppg', 'F_opp_dst_allowed_pg'],
    'implied + pressure (sack/hit) + giveaways': ['F_opp_implied_total', 'F_def_sack_rate', 'F_def_hit_rate', 'F_opp_sack_rate', 'F_opp_giveaways_pg'],
    'season stats only (no Vegas)': [c for c in FEATURES if c not in ('F_opp_implied_total', 'F_spread', 'F_game_total')],
    'everything': FEATURES,
}
print('\nOUT-OF-SAMPLE MODELS (train 2018-22, test 2023-25)')
out = []
for name, cols in MODELS.items():
    test['pred'], beta = fit_predict(cols)
    r = test.pred.corr(test.dst_pts)
    out.append(dict(model=name, weekly_spearman=weekly_spearman(test, 'pred'), r=r, r2=r ** 2,
                    top5_gain=top_pick_gain(test, 'pred'),
                    mae=(test.pred - test.dst_pts).abs().mean()))
print(pd.DataFrame(out).to_string(index=False, float_format='%.3f'))

test['pred'], beta = fit_predict(MODELS['everything'])
print('\nstandardized weights in "everything" model:')
sd = train[FEATURES].std()
print((pd.Series(beta[1:], index=[f[2:] for f in FEATURES]) * sd.values).sort_values(key=abs, ascending=False).to_string(float_format='%.3f'))

sample.to_parquet(DATA.parent / 'dst_sample.parquet')
