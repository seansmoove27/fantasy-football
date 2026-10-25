"""Compare our walk-forward D/ST and kicker predictions with ESPN projections, FantasyPros all-expert ECR,
and FantasyPros' top-10-expert ECR (2025 in-season accuracy) on identical 2026 team-weeks.

ESPN projections come from the league's kona_player_info view; top-10 ECR from the FantasyPros rankings page
filtered to "2025 In-Season Accuracy: Top 10 K / Top 10 DST".
Usage: python -I compare_sources_2026.py <espn_proj.csv> <fp_top10.csv> <db_fpecr.parquet> <games.csv> \
           <dst_v2_walkforward.parquet> <k_model_walkforward.parquet>
"""
import sys

import numpy as np
import pandas as pd

espn_path, top10_path, ecr_path, games_path, dst_path, k_path = sys.argv[1:7]
FIX = {'LAR': 'LA', 'WSH': 'WAS', 'JAC': 'JAX', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}

espn = pd.read_csv(espn_path).assign(season=2026)
espn['team'] = espn.team.replace(FIX)
espn = espn[espn.espn_proj > 0]  # zero = bye or no starter on ESPN's current roster (LAR kicker change)
top10 = pd.read_csv(top10_path).assign(season=2026)
top10['fp_top10'] = -top10.top10_rank

# all-expert ECR: Friday-ish snapshot before each week's Sunday games
g = pd.read_csv(games_path, usecols=['season', 'game_type', 'week', 'gameday'])
g = g[(g.game_type == 'REG') & (g.season == 2026)]
g['gameday'] = pd.to_datetime(g.gameday)
sundays = g[g.gameday.dt.dayofweek == 6].groupby('week').gameday.min()
d = pd.read_parquet(ecr_path, columns=['pos', 'ecr_type', 'team', 'ecr', 'scrape_date'])
d = d[(d.ecr_type == 'wp') & d.pos.isin(['K', 'DST'])].copy()
d['scrape_date'] = pd.to_datetime(d.scrape_date)
d['team'] = d.team.replace(FIX)
rows = []
for wk, sun in sundays.items():
    snap = d[(d.scrape_date < sun) & (d.scrape_date >= sun - pd.Timedelta(days=4))]
    snap = snap[snap.scrape_date == snap.scrape_date.max()]
    rows.append(snap.sort_values('ecr').drop_duplicates(['pos', 'team']).assign(week=wk))
fp_all = pd.concat(rows).assign(season=2026, fp_all=lambda x: -x.ecr)[['pos', 'season', 'week', 'team', 'fp_all']]

boards = {
    'DST': (pd.read_parquet(dst_path), 'dst_pts', 'ridge, all + weather'),
    'K': (pd.read_parquet(k_path), 'k_pts', 'ridge, all'),
}
key = ['season', 'week', 'team']
for pos, (wf, target, ours) in boards.items():
    m = wf[wf.season == 2026][key + [target, ours]].rename(columns={ours: 'ours'})
    for src, col in ((espn, 'espn_proj'), (top10, 'fp_top10'), (fp_all, 'fp_all')):
        m = m.merge(src[src.pos == pos][key + [col]], on=key)  # inner: only team-weeks every source ranked
    # 50/50 blend of our ranks with the top-10 experts' ranks within each week
    r = m.groupby('week')
    m['blend_ours_top10'] = 0.5 * r.ours.rank(pct=True) + 0.5 * r.fp_top10.rank(pct=True)
    weeks = [w for _, w in m.groupby('week')]
    print(f'\n{pos}: {len(m)} team-weeks ranked by every source, weeks {sorted(m.week.unique())}')
    base = np.array([w.fp_top10.corr(w[target], method='spearman') for w in weeks])
    out = []
    for c in ['fp_top10', 'fp_all', 'espn_proj', 'ours', 'blend_ours_top10']:
        sp = np.array([w[c].corr(w[target], method='spearman') for w in weeks])
        top5 = np.array([w.nlargest(5, c)[target].mean() - w[target].mean() for w in weeks])
        top10_pts = np.array([w.nlargest(10, c)[target].mean() for w in weeks])
        out.append(dict(source=c, weekly_spearman=sp.mean(), top5_gain=top5.mean(), top10_avg_pts=top10_pts.mean(),
                        vs_top10_experts=(sp - base).mean(), weeks_better=f'{(sp > base).sum()}/{len(sp)}',
                        by_week=' '.join(f'{v:.2f}' for v in sp)))
    print(pd.DataFrame(out).to_string(index=False, float_format='%.3f'))
