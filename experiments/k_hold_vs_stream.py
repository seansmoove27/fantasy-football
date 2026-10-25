"""Hold one waiver kicker for the rest of the season vs stream a new one every week.

From start week W to the season's last fantasy week, the "pool" each week is every team's kicker except the 14
that other fantasy teams are assumed to roster. Two proxies for who is rostered:
  own: the 14 kickers with the highest ESPN ownership in that week's FantasyPros snapshot (2020-2024)
  ros: the 14 best by FantasyPros rest-of-season ECR that week (all seasons; harsher, the best kickers never free)
Strategies:
  hold_ros     claim the best rest-of-season ECR kicker in the pool at week W, keep him (bye week: our model's #1)
  hold_ros7    hold whoever is rest-of-season ECR #7 at week W (a McPherson-like kicker), regardless of the pool
  stream_ours  our model's #1 in the pool every week (walk-forward predictions, closing lines)
  stream_ecr   FantasyPros weekly ECR #1 in the pool every week
Usage: python -I k_hold_vs_stream.py <db_fpecr.parquet> <games.csv> <k_model_walkforward.parquet>
"""
import sys

import numpy as np
import pandas as pd

ecr_path, games_path, wf_path = sys.argv[1:4]
FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'JAC': 'JAX', 'LAR': 'LA', 'WSH': 'WAS'}
OTHERS = 14

g = pd.read_csv(games_path, usecols=['season', 'game_type', 'week', 'gameday'])
g = g[g.game_type == 'REG']
g['gameday'] = pd.to_datetime(g.gameday)
sundays = g[g.gameday.dt.dayofweek == 6].groupby(['season', 'week']).gameday.min().reset_index()

d = pd.read_parquet(ecr_path, columns=['pos', 'ecr_type', 'team', 'ecr', 'player_owned_espn', 'scrape_date'])
d = d[(d.pos == 'K') & d.ecr_type.isin(['rp', 'wp'])].copy()
d['scrape_date'] = pd.to_datetime(d.scrape_date)
d['team'] = d.team.replace(FIX)
rows = []
for _, s in sundays.iterrows():
    snap = d[(d.scrape_date < s.gameday) & (d.scrape_date >= s.gameday - pd.Timedelta(days=4))]
    for t, x in snap.groupby('ecr_type'):
        x = x[x.scrape_date == x.scrape_date.max()].sort_values('ecr').drop_duplicates('team')
        rows.append(x.assign(season=s.season, week=s.week))
snaps = pd.concat(rows)
ros = snaps[snaps.ecr_type == 'rp'].set_index(['season', 'week', 'team'])
wk_ecr = snaps[snaps.ecr_type == 'wp'].set_index(['season', 'week', 'team']).ecr

wf = pd.read_parquet(wf_path)
wf = wf[wf.season < 2026]
pts = wf.set_index(['season', 'week', 'team']).k_pts
ours = wf.set_index(['season', 'week', 'team'])['ridge, all']


def rostered(season, week, proxy):
    try:
        x = ros.loc[(season, week)]
    except KeyError:
        return None
    if proxy == 'own':
        if x.player_owned_espn.isna().all():
            return None
        return set(x.player_owned_espn.nlargest(OTHERS).index)
    return set(x.ecr.nsmallest(OTHERS).index)


def best(cands, scores):
    s = scores.reindex(cands).dropna()
    return None if s.empty else s.idxmax()


results = []
for proxy in ('own', 'ros'):
    for season in sorted(wf.season.unique()):
        last = wf[wf.season == season].week.max() - 1  # fantasy season ends a week before the NFL's
        for start in range(4, 11):
            taken0 = rostered(season, start, proxy)
            if taken0 is None:
                continue
            playing0 = set(pts.loc[(season, start)].index)
            avail0 = [t for t in ros.loc[(season, start)].index if t not in taken0]
            held = best(avail0, -ros.loc[(season, start)].ecr)
            held7 = ros.loc[(season, start)].ecr.nsmallest(7).index[-1]  # a McPherson-like ROS #7 kicker
            tot = {'hold_ros': 0.0, 'hold_ros7': 0.0, 'stream_ours': 0.0, 'stream_ecr': 0.0}
            n = 0
            ok = True
            for w in range(start, last + 1):
                taken = rostered(season, w, proxy)
                if taken is None or (season, w) not in pts.index.droplevel(2):
                    ok = False
                    break
                playing = list(pts.loc[(season, w)].index)
                pool = [t for t in playing if t not in taken]  # streaming only sees free agents
                o = ours.loc[(season, w)]
                pick_ours = best(pool, o)
                e = -wk_ecr.loc[(season, w)] if (season, w) in wk_ecr.index.droplevel(2) else o
                pick_ecr = best(pool, e) or pick_ours
                hold_pick = held if held in playing else pick_ours  # bye: stream that week
                week_pts = pts.loc[(season, w)]
                tot['hold_ros'] += week_pts[hold_pick]
                tot['hold_ros7'] += week_pts[held7 if held7 in playing else pick_ours]
                tot['stream_ours'] += week_pts[pick_ours]
                tot['stream_ecr'] += week_pts[pick_ecr]
                n += 1
            if ok and n:
                results.append(dict(proxy=proxy, season=season, start=start, weeks=n, held=held, **tot))

r = pd.DataFrame(results)
r['ours_minus_hold'] = r.stream_ours - r.hold_ros
r['ecr_minus_hold'] = r.stream_ecr - r.hold_ros
r['ours_minus_hold7'] = r.stream_ours - r.hold_ros7
rng = np.random.default_rng(0)
for proxy, x in r.groupby('proxy'):
    print(f'\nproxy={proxy}: {len(x)} season x start-week runs, seasons {sorted(x.season.unique())}')
    per_wk = x[['hold_ros', 'hold_ros7', 'stream_ours', 'stream_ecr']].sum() / x.weeks.sum()
    print('points per week:', per_wk.round(2).to_dict())
    for c in ('ours_minus_hold', 'ecr_minus_hold', 'ours_minus_hold7'):
        v = (x[c] / x.weeks).to_numpy()
        # resample whole seasons (start weeks within a season overlap, so they are not independent)
        seasons = x.season.unique()
        boot = []
        for _ in range(4000):
            pick = rng.choice(seasons, len(seasons))
            boot.append(np.concatenate([v[x.season.to_numpy() == s] for s in pick]).mean())
        print(f'  {c}: {v.mean():+.2f} pts/week (95% CI {np.percentile(boot, 2.5):+.2f} to {np.percentile(boot, 97.5):+.2f}), '
              f'stream ahead in {(v > 0).sum()}/{len(v)} runs')
    print(x[x.start == 7][['season', 'held', 'weeks', 'hold_ros', 'stream_ours', 'stream_ecr']].to_string(index=False))
