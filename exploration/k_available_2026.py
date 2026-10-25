"""Grade 2026 kicker rankings on the kickers the user could actually start in the Limers league.

Sources: our model (walk-forward for weeks 1-5, the saved pre-game board for week 6), FantasyPros "Latest ECR",
FantasyPros top-10-accuracy consensus (2026 and 2025 in-season), and draftsharks.com weekly K rankings.
The pool each week is every team's kicker not on another fantasy roster going into that week (the previous
week's rosters, i.e. before waivers), plus the user's own kicker. Unranked kickers count as tied for last.
Usage: python -I k_available_2026.py <k_sources.txt> <draftsharks_k_w1-5.csv> <k_model_walkforward.parquet> <week6_board.txt>
"""
import sys

import numpy as np
import pandas as pd

src_path, ds_path, wf_path, board_path = sys.argv[1:5]
FIX = {'JAC': 'JAX', 'LVR': 'LV', 'LAR': 'LA', 'WSH': 'WAS'}
ME = 7

ranks, rosters = [], []
for line in open(src_path):
    if line.startswith('#') or not line.strip():
        continue
    src, wk, items = line.strip().split('|')
    for it in items.split(','):
        parts = it.split(':')
        if src == 'espn':
            rosters.append(dict(week=int(wk), fteam=int(parts[0]), team=FIX.get(parts[1], parts[1]), slot=int(parts[3])))
        else:
            ranks.append(dict(source=src, week=int(wk), team=FIX.get(parts[0], parts[0]), rank=int(parts[1])))
ranks = pd.DataFrame(ranks)
ds = pd.read_csv(ds_path).rename(columns={'ds_rank': 'rank'}).assign(source='draftsharks')
ranks = pd.concat([ranks, ds[['source', 'week', 'team', 'rank']]])
ranks['team'] = ranks.team.replace(FIX)
rosters = pd.DataFrame(rosters)

wf = pd.read_parquet(wf_path)
wf = wf[wf.season == 2026][['week', 'team', 'k_pts', 'ridge, all']]
lines = [l.split() for l in open(board_path)][1:]  # header names contain spaces; team first, ridge last
board = pd.DataFrame({'team': [l[0] for l in lines if l], 'ridge, all': [float(l[-1]) for l in lines if l]}).assign(week=6)
ours = pd.concat([wf[wf.week < 6][['week', 'team', 'ridge, all']], board])
ours['rank'] = ours.groupby('week')['ridge, all'].rank(ascending=False)
ranks = pd.concat([ranks, ours.assign(source='ours')[['source', 'week', 'team', 'rank']]])
pts = wf[['week', 'team', 'k_pts']]

SOURCES = ['ours', 'fp_latest', 'fp_top10_rolling', 'fp_top10_2026', 'fp_top10_2025', 'draftsharks']
rows, picks = [], []
for wk in sorted(pts.week.unique()):
    prev = rosters[rosters.week == max(wk - 1, 1)]
    taken = set(prev[prev.fteam != ME].team)
    mine = rosters[(rosters.week == wk) & (rosters.fteam == ME) & (rosters.slot == 17)].team.iloc[0]
    pool = pts[(pts.week == wk) & (~pts.team.isin(taken) | (pts.team == mine))].copy()
    best = pool.k_pts.max()
    started = pool.loc[pool.team == mine, 'k_pts'].iloc[0] if mine in set(pool.team) else np.nan
    for s in SOURCES:
        r = ranks[(ranks.source == s) & (ranks.week == wk)].set_index('team')['rank']
        if r.empty:
            continue
        p = pool.assign(rank=pool.team.map(r).fillna(999))
        p = p.sort_values(['rank', 'k_pts'], ascending=[True, False])  # unranked ties broken pessimistically below
        top = p[p['rank'] == p['rank'].min()]
        top1_pts = top.k_pts.mean()  # if several share the top rank, use their average
        rows.append(dict(week=wk, source=s, pool=len(pool), pick=','.join(top.team), pick_pts=top1_pts,
                         top3_avg=p.nsmallest(3, 'rank').k_pts.mean(),
                         spearman=(-p['rank']).corr(p.k_pts, method='spearman'),
                         best_avail=best, started=mine, started_pts=started))
res = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(res.to_string(index=False, float_format='%.2f'))
print('\nSummary over weeks (pick_pts = points from each source\'s #1 available kicker):')
summ = res.groupby('source').agg(weeks=('week', 'nunique'), pick_pts=('pick_pts', 'sum'), avg_pick=('pick_pts', 'mean'),
                                 top3_avg=('top3_avg', 'mean'), spearman=('spearman', 'mean'))
print(summ.loc[[s for s in SOURCES if s in summ.index]].to_string(float_format='%.2f'))
