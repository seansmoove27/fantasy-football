"""Grade 2026 D/ST rankings on the defenses the user could actually start in the Limers league.

Sources: our model (walk-forward for weeks 1-5, the saved pre-game board for week 6), FantasyPros "Latest ECR",
FantasyPros all-expert ECR from the Friday archive snapshot, FantasyPros top-10-accuracy consensus (2026 and 2025
in-season), and draftsharks.com weekly DEF rankings. The pool each week is every D/ST not on another fantasy
roster going into that week (the previous week's rosters, i.e. before waivers), plus the user's own D/ST; pass
"post" as the last argument to use that week's rosters instead (what was left after waivers). Unranked teams
count as tied for last.
Usage: python -I dst_available_2026.py <dst_sources.txt> <draftsharks_dst_w1-5.csv> <dst_v2_walkforward.parquet> \
           <week6_board.txt> <db_fpecr.parquet> <games.csv> [pre|post]
"""
import sys

import numpy as np
import pandas as pd

src_path, ds_path, wf_path, board_path, ecr_path, games_path = sys.argv[1:7]
POST = len(sys.argv) > 7 and sys.argv[7] == 'post'
FIX = {'JAC': 'JAX', 'LVR': 'LV', 'LAR': 'LA', 'WSH': 'WAS', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
ME = 7

ranks, rosters = [], []
for line in open(src_path):
    if line.startswith('#') or not line.strip():
        continue
    src, wk, items = line.strip().split('|')
    for it in items.split(','):
        a, b = it.split(':')
        if src == 'espn':
            rosters.append(dict(week=int(wk), fteam=int(a), team=FIX.get(b, b)))
        else:
            ranks.append(dict(source=src, week=int(wk), team=FIX.get(a, a), rank=int(b)))
ranks = pd.DataFrame(ranks)
ds = pd.read_csv(ds_path).rename(columns={'ds_rank': 'rank'}).assign(source='draftsharks')
ranks = pd.concat([ranks, ds[['source', 'week', 'team', 'rank']]])
ranks['team'] = ranks.team.replace(FIX)
rosters = pd.DataFrame(rosters)

# all-expert ECR as of the last archive snapshot in the 4 days before each week's first Sunday
g = pd.read_csv(games_path, usecols=['season', 'game_type', 'week', 'gameday'])
g = g[(g.game_type == 'REG') & (g.season == 2026)]
g['gameday'] = pd.to_datetime(g.gameday)
sundays = g[g.gameday.dt.dayofweek == 6].groupby('week').gameday.min()
e = pd.read_parquet(ecr_path, columns=['pos', 'ecr_type', 'team', 'ecr', 'scrape_date'])
e = e[(e.ecr_type == 'wp') & (e.pos == 'DST')].copy()
e['scrape_date'] = pd.to_datetime(e.scrape_date)
e['team'] = e.team.replace(FIX)
for wk, sun in sundays.items():
    snap = e[(e.scrape_date < sun) & (e.scrape_date >= sun - pd.Timedelta(days=4))]
    snap = snap[snap.scrape_date == snap.scrape_date.max()].sort_values('ecr').drop_duplicates('team')
    ranks = pd.concat([ranks, pd.DataFrame(dict(source='fp_friday', week=wk, team=snap.team, rank=snap.ecr.rank()))])

wf = pd.read_parquet(wf_path)
wf = wf[wf.season == 2026][['week', 'team', 'dst_pts', 'ridge, all + weather']]
lines = [l.split() for l in open(board_path)][1:]  # team first, ridge_v2 last
board = pd.DataFrame({'team': [l[0] for l in lines if l], 'ridge, all + weather': [float(l[-1]) for l in lines if l]}).assign(week=6)
ours = pd.concat([wf[wf.week < 6][['week', 'team', 'ridge, all + weather']], board])
ours['rank'] = ours.groupby('week')['ridge, all + weather'].rank(ascending=False)
ranks = pd.concat([ranks, ours.assign(source='ours')[['source', 'week', 'team', 'rank']]])
pts = wf[['week', 'team', 'dst_pts']]

SOURCES = ['ours', 'fp_latest', 'fp_friday', 'fp_top10_todate', 'fp_top10_2026', 'fp_top10_2025', 'draftsharks']
rows = []
for wk in sorted(pts.week.unique()):
    prev = rosters[rosters.week == (wk if POST else max(wk - 1, 1))]
    taken = set(prev[prev.fteam != ME].team)
    mine = rosters[(rosters.week == wk) & (rosters.fteam == ME)].team.iloc[0]
    pool = pts[(pts.week == wk) & (~pts.team.isin(taken) | (pts.team == mine))].copy()
    best = pool.dst_pts.max()
    started = pool.loc[pool.team == mine, 'dst_pts'].iloc[0] if mine in set(pool.team) else np.nan
    for s in SOURCES:
        r = ranks[(ranks.source == s) & (ranks.week == wk)].set_index('team')['rank']
        if r.empty:
            continue
        p = pool.assign(rank=pool.team.map(r).fillna(999))
        top = p[p['rank'] == p['rank'].min()]
        rows.append(dict(week=wk, source=s, pool=len(pool), pick=','.join(top.team), pick_pts=top.dst_pts.mean(),
                         top3_avg=p.nsmallest(3, 'rank').dst_pts.mean(),
                         spearman=(-p['rank']).corr(p.dst_pts, method='spearman'),
                         best_avail=best, pool_avg=pool.dst_pts.mean(), started=mine, started_pts=started))
res = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(f'pool = {"that week" if POST else "previous week"}\'s rosters')
print(res.to_string(index=False, float_format='%.2f'))
print('\nSummary over weeks (pick_pts = points from each source\'s #1 available D/ST):')
summ = res.groupby('source').agg(weeks=('week', 'nunique'), pick_pts=('pick_pts', 'sum'), avg_pick=('pick_pts', 'mean'),
                                 top3_avg=('top3_avg', 'mean'), spearman=('spearman', 'mean'))
print(summ.loc[[s for s in SOURCES if s in summ.index]].to_string(float_format='%.2f'))
