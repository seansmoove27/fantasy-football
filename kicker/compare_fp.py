"""Compare our walk-forward kicker predictions with FantasyPros weekly kicker ECR on identical team-weeks.

Usage: python -I compare_fp.py <db_fpecr.parquet> <games.csv> <k_model_walkforward.parquet>
"""
import sys

import numpy as np
import pandas as pd

ecr_path, games_path, wf_path = sys.argv[1:4]
RELOCATE = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}

d = pd.read_parquet(ecr_path, columns=['pos', 'ecr_type', 'team', 'ecr', 'scrape_date'])
ecr = d[(d.pos == 'K') & (d.ecr_type == 'wp')].copy()
ecr['scrape_date'] = pd.to_datetime(ecr.scrape_date)
ecr['team'] = ecr.team.replace(RELOCATE).replace({'JAC': 'JAX', 'LAR': 'LA', 'WSH': 'WAS'})

# Map each snapshot to the regular-season week whose Sunday games come 1-4 days later
g = pd.read_csv(games_path, usecols=['season', 'game_type', 'week', 'gameday'])
g = g[g.game_type == 'REG']
g['gameday'] = pd.to_datetime(g.gameday)
sundays = g[g.gameday.dt.dayofweek == 6].groupby(['season', 'week']).gameday.min().reset_index()
snap_to_week = {}
for snap in ecr.scrape_date.unique():
    hit = sundays[(sundays.gameday > snap) & (sundays.gameday <= snap + pd.Timedelta(days=4))]
    if len(hit):
        snap_to_week[snap] = (int(hit.season.iloc[0]), int(hit.week.iloc[0]))
ecr['season'] = ecr.scrape_date.map(lambda s: snap_to_week.get(s, (None, None))[0])
ecr['week'] = ecr.scrape_date.map(lambda s: snap_to_week.get(s, (None, None))[1])
ecr = ecr.dropna(subset=['season']).astype({'season': int, 'week': int})
# one kicker per team: the best-ranked one (the expected starter)
ecr = ecr.sort_values('scrape_date').drop_duplicates(['season', 'week', 'team', 'ecr'], keep='last')
ecr = ecr.sort_values('ecr').drop_duplicates(['season', 'week', 'team'], keep='first')

wf = pd.read_parquet(wf_path)
models = [c for c in wf.columns if c not in ('game_id', 'season', 'week', 'team', 'opp', 'k_pts')]
m = wf.merge(ecr[['season', 'week', 'team', 'ecr']], on=['season', 'week', 'team'], how='inner')
m['fp_ecr'] = -m.ecr  # lower ECR rank = better, flip so higher = better
print(f'matched {len(m)} team-weeks across {m.groupby(["season", "week"]).ngroups} weeks, seasons {sorted(m.season.unique())}')

cols = ['fp_ecr'] + models
weeks = [grp for _, grp in m.groupby(['season', 'week']) if len(grp) >= 10]


def per_week(col):
    sp = np.array([w[col].corr(w.k_pts, method='spearman') for w in weeks])
    gain = np.array([w.nlargest(5, col).k_pts.mean() - w.k_pts.mean() for w in weeks])
    return sp, gain


base_sp, _ = per_week('fp_ecr')
rng = np.random.default_rng(0)
rows = []
for c in cols:
    sp, gain = per_week(c)
    diff = sp - base_sp
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
    rows.append(dict(source=c, weekly_spearman=sp.mean(), top5_gain=gain.mean(), vs_fp=diff.mean(),
                     ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                     weeks_better=f'{(diff > 0).sum()}/{len(diff)}'))
print(f'\n{len(weeks)} weeks with >=10 matched kickers')
print(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))

print('\nper-season weekly Spearman:')
season_rows = []
for s, grp in m.groupby('season'):
    ws = [w for _, w in grp.groupby('week') if len(w) >= 10]
    season_rows.append({'season': s, 'weeks': len(ws), **{c: np.mean([w[c].corr(w.k_pts, method='spearman') for w in ws])
                                                          for c in ['fp_ecr', 'implied only', 'ridge, all']}})
print(pd.DataFrame(season_rows).to_string(index=False, float_format='%.3f'))

m['rank_fp'] = m.groupby(['season', 'week']).fp_ecr.rank(pct=True)
m['rank_ours'] = m.groupby(['season', 'week'])['ridge, all'].rank(pct=True)
for wfp in (0.25, 0.5, 0.75):
    m['blend'] = wfp * m.rank_fp + (1 - wfp) * m.rank_ours
    weeks = [grp for _, grp in m.groupby(['season', 'week']) if len(grp) >= 10]
    sp, gain = per_week('blend')
    print(f'blend {int(wfp*100)}% FP / {int((1-wfp)*100)}% ours: weekly_spearman {sp.mean():.3f}, top5_gain {gain.mean():.3f}')
