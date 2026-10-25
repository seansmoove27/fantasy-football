"""Rookie breakout model: which Week 1-5 signals predict a rookie RB/WR/TE's rest-of-season PPR points per game.

Backtest on 2018-2025 rookies (leave-one-season-out), then score the current season's rookies.
Breakout = rest-of-season PPR >= 12 per game (WR3 / RB2 range) over at least 4 games.
Usage:
  python -I rookie_breakout.py eval    <data_dir>                  # compare feature sets
  python -I rookie_breakout.py predict <data_dir> <season> [week]  # board for this season's rookies through <week>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODE, DATA = sys.argv[1], Path(sys.argv[2])
YEARS = range(2018, 2027)
CUT = 5            # weeks 1..CUT are the observation window
BREAKOUT = 12.0    # PPR points per game, rest of season
LATE_PPG = 10.0    # "not yet broken out" through the window
POS = ['RB', 'WR', 'TE']
# snap_counts position labels: CIN backs are 'HB' (2020, 2025-26); 2020 has slash labels
POS_FIX = {'HB': 'RB', 'FB': 'RB', 'RB/W': 'RB', 'RB/F': 'RB', 'FB/R': 'RB', 'WR/R': 'WR'}

FEATURE_SETS = {
    'draft only': ['log_pick', 'undrafted'],
    'early ppg only': ['ppg'],
    'usage': ['snap_pct', 'snap_last2', 'tgt_share', 'air_share', 'carry_share'],
    'draft + usage': ['log_pick', 'undrafted', 'snap_pct', 'snap_last2', 'snap_trend', 'tgt_share', 'air_share', 'carry_share'],
    'all': ['log_pick', 'undrafted', 'ppg', 'snap_pct', 'snap_last2', 'snap_trend', 'tgt_share', 'air_share', 'carry_share'],
}


def load_ids():
    ids = pd.read_csv(DATA / 'db_playerids.csv', usecols=['gsis_id', 'pfr_id', 'name', 'draft_year', 'draft_ovr'])
    return ids.dropna(subset=['gsis_id']).drop_duplicates('gsis_id')


def player_weeks(ids):
    """One row per player-game: snaps (any offensive snap counts as a game) joined to box-score stats."""
    out = []
    for y in YEARS:
        snaps = pd.read_parquet(DATA / f'snap_counts_{y}.parquet')
        snaps['position'] = snaps.position.replace(POS_FIX)
        snaps = snaps[(snaps.game_type == 'REG') & (snaps.offense_snaps > 0) & snaps.position.isin(POS)]
        pfr = ids.dropna(subset=['pfr_id']).drop_duplicates('pfr_id')
        snaps = snaps.merge(pfr[['pfr_id', 'gsis_id']], left_on='pfr_player_id', right_on='pfr_id', how='left')
        by_name = ids[~ids.name.duplicated(keep=False)].set_index('name').gsis_id
        miss = snaps.gsis_id.isna()
        snaps.loc[miss, 'gsis_id'] = snaps.loc[miss, 'player'].map(by_name)
        st = pd.read_parquet(DATA / f'stats_player_week_{y}.parquet')
        st = st[st.season_type == 'REG']
        team_carries = st.groupby(['team', 'week']).carries.sum().rename('team_carries').reset_index()
        st = st.merge(team_carries, on=['team', 'week'])
        cols = ['player_id', 'week', 'fantasy_points_ppr', 'targets', 'target_share', 'air_yards_share', 'carries', 'team_carries']
        g = snaps.merge(st[cols], left_on=['gsis_id', 'week'], right_on=['player_id', 'week'], how='left')
        g = g.fillna({'fantasy_points_ppr': 0, 'targets': 0, 'target_share': 0, 'air_yards_share': 0, 'carries': 0})
        g['carry_share'] = g.carries / g.team_carries.replace(0, np.nan)
        g['carry_share'] = g.carry_share.fillna(0)
        out.append(g[['season', 'week', 'gsis_id', 'pfr_player_id', 'player', 'position', 'team', 'offense_pct',
                      'fantasy_points_ppr', 'target_share', 'air_yards_share', 'carry_share']])
    return pd.concat(out, ignore_index=True)


def rookies(pw, ids):
    """Rookie = drafted that season, or undrafted with no earlier snaps (first season with snaps since 2016)."""
    first = pw.groupby('pfr_player_id').season.min().rename('first_season')
    pw = pw.merge(first, on='pfr_player_id').merge(ids[['gsis_id', 'draft_year', 'draft_ovr']], on='gsis_id', how='left')
    drafted = pw.draft_year == pw.season
    udfa = pw.draft_ovr.isna() & (pw.first_season == pw.season) & (pw.season > YEARS[0])
    return pw[drafted | udfa]


def features(rk, last_week):
    early = rk[rk.week <= last_week].sort_values('week')
    agg = early.groupby(['season', 'pfr_player_id']).agg(
        player=('player', 'last'), position=('position', 'last'), team=('team', 'last'), gsis_id=('gsis_id', 'last'),
        draft_ovr=('draft_ovr', 'last'), games=('week', 'size'), ppg=('fantasy_points_ppr', 'mean'),
        snap_pct=('offense_pct', 'mean'), tgt_share=('target_share', 'mean'), air_share=('air_yards_share', 'mean'),
        carry_share=('carry_share', 'mean')).reset_index()
    last2 = early[early.week > last_week - 2].groupby(['season', 'pfr_player_id']).offense_pct.mean().rename('snap_last2')
    first2 = early[early.week <= 2].groupby(['season', 'pfr_player_id']).offense_pct.mean().rename('snap_first2')
    agg = agg.merge(last2, on=['season', 'pfr_player_id'], how='left').merge(first2, on=['season', 'pfr_player_id'], how='left')
    agg['snap_last2'] = agg.snap_last2.fillna(0)   # no snaps in the last two weeks
    agg['snap_trend'] = agg.snap_last2 - agg.snap_first2.fillna(0)
    agg['undrafted'] = agg.draft_ovr.isna().astype(int)
    agg['log_pick'] = np.log(agg.draft_ovr.fillna(300))
    return agg


def labels(rk, last_week):
    late = rk[rk.week > last_week].groupby(['season', 'pfr_player_id']).agg(
        ros_games=('week', 'size'), ros_ppg=('fantasy_points_ppr', 'mean')).reset_index()
    return late


def design(df, cols):
    return pd.concat([df[cols], pd.get_dummies(df.position, dtype=float)[POS]], axis=1)


def evaluate():
    ids = load_ids()
    rk = rookies(player_weeks(ids), ids)
    df = features(rk, CUT).merge(labels(rk, CUT), on=['season', 'pfr_player_id'], how='left')
    df = df[(df.season < 2026)]
    df['ros_ppg'] = df.ros_ppg.fillna(0)  # never played again counts as zero
    df['breakout'] = ((df.ros_ppg >= BREAKOUT) & (df.ros_games >= 4)).astype(int)
    print(f'{len(df)} rookie RB/WR/TE seasons with a Wk1-{CUT} snap, {df.breakout.sum()} breakouts '
          f'(ROS >= {BREAKOUT} PPR/g, 4+ games), seasons {df.season.min()}-{df.season.max()}')
    print(df[df.breakout == 1].sort_values('ros_ppg', ascending=False)
          [['season', 'player', 'position', 'draft_ovr', 'ppg', 'snap_pct', 'tgt_share', 'ros_ppg']].head(25).to_string(index=False))
    print()
    rows = []
    for name, cols in FEATURE_SETS.items():
        pred_ppg, pred_p = pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)
        for s in sorted(df.season.unique()):
            tr, te = df[df.season != s], df[df.season == s]
            reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(design(tr, cols), tr.ros_ppg)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(design(tr, cols), tr.breakout)
            pred_ppg[te.index] = reg.predict(design(te, cols))
            pred_p[te.index] = clf.predict_proba(design(te, cols))[:, 1]
        top = df.assign(p=pred_p).sort_values('p', ascending=False).groupby('season').head(10)
        # late breakouts: under LATE_PPG through the window, i.e. the ones still on waivers
        late = df.ppg < LATE_PPG
        top_late = df[late].assign(p=pred_p[late]).sort_values('p', ascending=False).groupby('season').head(10)
        rows.append({'features': name, 'ROS ppg rank corr': spearmanr(pred_ppg, df.ros_ppg)[0],
                     'breakout AUC': roc_auc_score(df.breakout, pred_p),
                     'breakouts in top 10/yr': top.breakout.sum() / df.season.nunique(),
                     f'late AUC (<{LATE_PPG:g} ppg)': roc_auc_score(df.breakout[late], pred_p[late]),
                     'late in top 10/yr': top_late.breakout.sum() / df.season.nunique(),
                     'late breakouts/yr': df.breakout[late].sum() / df.season.nunique()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    late = df[(df.ppg < LATE_PPG) & (df.breakout == 1)].sort_values('ros_ppg', ascending=False)
    print(f'\nlate breakouts ({len(late)}):')
    print(late[['season', 'player', 'position', 'draft_ovr', 'ppg', 'snap_pct', 'snap_last2', 'snap_trend', 'tgt_share',
                'carry_share', 'ros_ppg']].round(2).to_string(index=False))
    print('\nmedians, late non-breakouts with draft_ovr <= 100:')
    nb = df[(df.ppg < LATE_PPG) & (df.breakout == 0) & (df.draft_ovr <= 100)]
    print(nb[['ppg', 'snap_pct', 'snap_last2', 'snap_trend', 'tgt_share', 'carry_share']].median().round(2).to_dict())


def predict(season, week):
    ids = load_ids()
    rk = rookies(player_weeks(ids), ids)
    hist = features(rk[rk.season < season], CUT).merge(labels(rk[rk.season < season], CUT), on=['season', 'pfr_player_id'], how='left')
    hist['ros_ppg'] = hist.ros_ppg.fillna(0)
    hist['breakout'] = ((hist.ros_ppg >= BREAKOUT) & (hist.ros_games >= 4)).astype(int)
    cols = FEATURE_SETS['all']
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(design(hist, cols), hist.ros_ppg)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(design(hist, cols), hist.breakout)
    cur = features(rk[rk.season == season], week)
    cur['p_breakout'] = clf.predict_proba(design(cur, cols))[:, 1]
    cur['pred_ros_ppg'] = reg.predict(design(cur, cols))
    coef = pd.Series(clf[-1].coef_[0], index=design(hist, cols).columns).round(2)
    print('logistic coefficients (standardized):', coef.to_dict(), '\n')
    out = cur.merge(ids[['gsis_id']].assign(), on='gsis_id', how='left')
    out = out.sort_values('p_breakout', ascending=False)
    show = ['player', 'position', 'team', 'draft_ovr', 'games', 'ppg', 'snap_pct', 'snap_last2', 'tgt_share', 'carry_share',
            'p_breakout', 'pred_ros_ppg']
    print(out[show].head(30).round(2).to_string(index=False))
    out[show + ['gsis_id']].to_csv(DATA.parent / f'rookie_board_{season}_wk{week}.csv', index=False)


if MODE == 'eval':
    evaluate()
elif MODE == 'predict':
    predict(int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else CUT)
