"""Drop model: rest-of-season PPR points per game for every RB/WR/TE, so a roster can be compared with the waiver pool.

Same features as waiver_model.py (usage, expected points, recent scoring, draft capital, FantasyPros ROS rank, upcoming
implied totals). Targets: rest-of-season PPR / team games (missed games count as 0) and the next 4 weeks.
Walk-forward: each test season is fit on earlier seasons only; compared with the FantasyPros ROS rank on its own.
Usage:
  python -I drop_model.py eval    <data_dir>
  python -I drop_model.py predict <data_dir> <season> <week> <my_roster.csv> <rostered.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import waiver_model as wm  # noqa: E402

wm.DATA = DATA = Path(sys.argv[2])
# recency-weighted form replaces the last-2-game averages; plus position rank, vacated opportunity, depth-chart competition
L2 = {'ppg_last2', 'ep_last2', 'snap_last2', 'tgt_last2', 'carry_last2'}
COLS = [c for c in wm.FEATURES['all'] if c not in L2] + ['ppg_ewm', 'ep_ewm', 'snap_ewm', 'tgt_ewm', 'carry_ewm',
                                                         'comp_snap', 'comp_tgt']
FIT_RANK = 40   # fit only on players outside the FP ROS top 40 (the tier where drop decisions happen); score everyone
BENCH_TIER = (100, 300)   # FantasyPros ROS overall ranks where drop decisions happen in a 14-team league


def history(seasons):
    ids = wm.load_ids()
    pw, played = wm.player_weeks(ids)
    imp = wm.implied_totals()
    df = wm.add_ranks(wm.build(ids, pw, played, imp, range(2, 14), seasons), imp)
    fresh = wm.ecr_snapshots(imp, fresh=True).rename(columns={'rank': 'rank_fresh'})
    df = df.merge(fresh[['season', 'week', 'fantasypros_id', 'rank_fresh']], on=['season', 'week', 'fantasypros_id'], how='left')
    df['rank_fresh'] = df.rank_fresh.fillna(500)
    df = wm.add_labels(df, pw, played)                 # next_ppg over the next 4 weeks
    # rest of season
    out = []
    for (s, w), d in df.groupby(['season', 'week']):
        fut = pw[(pw.season == s) & (pw.week > w)]
        tg = played[(played.season == s) & (played.week > w)].groupby('team').size()
        d = d.copy()
        d['ros_ppg'] = d.gsis_id.map(fut.groupby('gsis_id').fantasy_points_ppr.sum()).fillna(0) / d.team.map(tg)
        out.append(d)
    return pd.concat(out), (ids, pw, played, imp)


def model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def evaluate():
    df, _ = history(range(2020, 2026))
    test = range(2021, 2026)
    preds = {}
    for target in ('ros_ppg', 'next_ppg'):
        p = pd.Series(index=df.index, dtype=float)
        for s in test:
            tr, te = df[(df.season < s) & (df['rank'] >= FIT_RANK)], df[df.season == s]
            p[te.index] = model().fit(wm.design(tr, COLS), tr[target]).predict(wm.design(te, COLS))
        preds[target] = p
    # FP baselines: overall rank mixes positional scarcity into ppg, so fit log(rank) x position walk-forward too,
    # on the stale snapshot (what a Tuesday claim sees) and the fresh one (Friday, knows week w)
    def fp_design(d, col):
        pos = pd.get_dummies(d.position, dtype=float).reindex(columns=wm.POS, fill_value=0)
        return pd.concat([pos, pos.mul(np.log(d[col]), axis=0).add_suffix('_x')], axis=1)
    for target in ('ros_ppg', 'next_ppg'):
        for col in ('rank', 'rank_fresh'):
            p = pd.Series(index=df.index, dtype=float)
            for s in test:
                tr, te = df[df.season < s], df[df.season == s]
                p[te.index] = model().fit(fp_design(tr, col), tr[target]).predict(fp_design(te, col))
            preds[(target, col)] = p
    t = df[df.season.isin(test)].copy()
    rng = np.random.default_rng(0)
    for target in ('ros_ppg', 'next_ppg'):
        t['model'], t['ppg'] = preds[target][t.index], t.ppg_season
        t['fp_stale'], t['fp_fresh'] = preds[(target, 'rank')][t.index], preds[(target, 'rank_fresh')][t.index]
        for label, sub in [('all RB/WR/TE', t), (f'bench tier (FP rank {BENCH_TIER[0]}-{BENCH_TIER[1]})',
                                                  t[t['rank'].between(*BENCH_TIER)])]:
            keys = ('model', 'fp_stale', 'fp_fresh', 'ppg')
            wk = sub.groupby(['season', 'week']).apply(
                lambda g: pd.Series({k: spearmanr(g[k], g[target])[0] for k in keys}), include_groups=False)
            print(f'{target:8s} {label:32s} weekly Spearman  model {wk.model.mean():.3f}  FP stale {wk.fp_stale.mean():.3f}  '
                  f'FP fresh {wk.fp_fresh.mean():.3f}  season ppg {wk.ppg.mean():.3f}')
            for base in ('fp_stale', 'fp_fresh'):
                d = wk.model - wk[base]
                by = {s: d.loc[s] for s in test}   # season-block bootstrap
                boot = [pd.concat([by[s] for s in rng.choice(list(by), len(by))]).mean() for _ in range(5000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                print(f'    model - {base}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  by season '
                      f'{[round(by[s].mean(), 3) for s in test]}')
    # drop test: bench-tier pairs where model and fresh position-aware FP disagree on who is better, who was right?
    b = t[t['rank'].between(*BENCH_TIER)].copy()
    b['model'], b['fp'] = preds['ros_ppg'][b.index], preds[('ros_ppg', 'rank_fresh')][b.index]
    right, n = 0, 0
    for _, g in b.groupby(['season', 'week']):
        g = g.sample(min(len(g), 120), random_state=0)
        m, f, y = g.model.values, g.fp.values, g.ros_ppg.values
        i, j = np.triu_indices(len(g), 1)
        dis = np.sign(m[i] - m[j]) != np.sign(f[i] - f[j])
        right += (np.sign(m[i] - m[j]) == np.sign(y[i] - y[j]))[dis].sum()
        n += dis.sum()
    print(f'\nbench-tier pairs where model and fresh FP disagree: model right {right / n:.1%} of {n} pairs')
    fit = model().fit(wm.design(df, COLS), df.ros_ppg)
    print('\nridge coefficients (ROS ppg, standardized):')
    print(pd.Series(fit[-1].coef_, index=wm.design(df, COLS).columns).round(2).sort_values().to_string())


def predict(season, week, my_csv, rostered_csv):
    hist, (ids, pw, played, imp) = history(range(2020, season))
    hist = hist[hist['rank'] >= FIT_RANK]
    ros = model().fit(wm.design(hist, COLS), hist.ros_ppg)
    nxt = model().fit(wm.design(hist, COLS), hist.next_ppg)
    cur, unscored = wm.current(ids, pw, played, imp, wm.ecr_snapshots(imp), season, week)
    cur['pred_ros'] = ros.predict(wm.design(cur, COLS))
    cur['pred_next4'] = nxt.predict(wm.design(cur, COLS))
    mine = pd.read_csv(my_csv)
    rostered = pd.read_csv(rostered_csv)
    fa = cur[cur.espn_id.notna() & ~cur.espn_id.isin(rostered.espn_id)]
    repl = fa.sort_values('pred_ros', ascending=False).groupby('position').head(3)
    best = repl.groupby('position').pred_ros.max()
    me = mine.merge(cur, on='espn_id', how='left', suffixes=('', '_nfl'))
    me['vs_best_FA'] = me.pred_ros - me.position.map(best)
    show = ['player', 'slot', 'position', 'team', 'rank', 'ppg_season', 'ppg_last2', 'snap_last2', 'tgt_last2', 'carry_last2',
            'pred_next4', 'pred_ros', 'vs_best_FA']
    print(f'data through week {week}\n')
    print('your roster (lowest predicted rest-of-season first):')
    print(me.sort_values('pred_ros')[show].round(2).to_string(index=False))
    missing = me[me.pred_ros.isna()]
    if len(missing):
        print('\nno snaps in the last 3 weeks (not scored):', ', '.join(missing.player))
    fa_out = unscored[~unscored.espn_id.isin(rostered.espn_id)]
    if len(fa_out):
        print('free agents FP ranks but the model can\'t score (injured / no recent snaps):',
              ', '.join(f'{r.name} ({int(r.rank)})' for r in fa_out.itertuples()))
    print('\nbest free agents by predicted rest-of-season:')
    print(repl.sort_values(['position', 'pred_ros'], ascending=[True, False])
          [['player', 'position', 'team', 'rank', 'ppg_season', 'ppg_last2', 'snap_last2', 'pred_next4', 'pred_ros']]
          .round(2).to_string(index=False))


if __name__ == '__main__' and sys.argv[1] == 'eval':
    evaluate()
elif __name__ == '__main__' and sys.argv[1] == 'predict':
    predict(int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
