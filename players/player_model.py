"""Can a model add to FantasyPros weekly ECR for QB/RB/WR/TE in full-PPR (Limers scoring)?

  python -I player_model.py build <data_dir>   # player-week table: ECR + pre-game features + actual PPR points
  python -I player_model.py eval <data_dir>    # walk-forward 2022+ : ECR vs model vs ECR+model blend

The model never replaces ECR: it predicts points with ECR as an input, and the blend weight
toward it is picked each season from earlier seasons only. Scored only on players who played
(a Sunday decision-maker knows who is inactive; Friday ECR and our features both don't).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODE, DATA = sys.argv[1], Path(sys.argv[2])
OUT = DATA.parent / 'player_features.parquet'
SEASONS = range(2018, 2027)
RELOCATE = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
POS = ['QB', 'RB', 'WR', 'TE']
TOP_N = {'QB': 24, 'RB': 48, 'WR': 60, 'TE': 24}  # roughly the startable pool in a 14-team league


def load_ecr():
    d = pd.read_parquet(DATA / 'db_fpecr.parquet',
                        columns=['id', 'pos', 'ecr_type', 'ecr', 'sd', 'best', 'worst', 'scrape_date', 'fp_page'])
    d = d[(d.ecr_type == 'wp') & d.pos.isin(POS)
          & d.fp_page.str.contains('qb|ppr-rb|ppr-wr|ppr-te')].copy()
    d['scrape_date'] = pd.to_datetime(d.scrape_date)
    # snapshot -> the regular-season week whose Sunday games are 1-4 days later
    g = pd.read_csv(DATA / 'games.csv', usecols=['season', 'game_type', 'week', 'gameday'])
    g = g[g.game_type == 'REG']
    g['gameday'] = pd.to_datetime(g.gameday)
    sundays = g[g.gameday.dt.dayofweek == 6].groupby(['season', 'week']).gameday.min().reset_index()
    snap_week = {}
    for snap in d.scrape_date.unique():
        hit = sundays[(sundays.gameday > snap) & (sundays.gameday <= snap + pd.Timedelta(days=4))]
        if len(hit):
            snap_week[snap] = (int(hit.season.iloc[0]), int(hit.week.iloc[0]))
    d = d[d.scrape_date.isin(snap_week)]
    d['season'] = d.scrape_date.map(lambda s: snap_week[s][0])
    d['week'] = d.scrape_date.map(lambda s: snap_week[s][1])
    d = d.sort_values('scrape_date').drop_duplicates(['season', 'week', 'id'], keep='last')
    ids = pd.read_csv(DATA / 'db_playerids.csv', usecols=['fantasypros_id', 'gsis_id', 'pfr_id']).dropna(subset=['fantasypros_id', 'gsis_id'])
    ids['fantasypros_id'] = ids.fantasypros_id.astype(int).astype(str)
    d = d.assign(id=d.id.astype(str)).merge(ids.drop_duplicates('fantasypros_id'), left_on='id', right_on='fantasypros_id')
    return d.rename(columns={'gsis_id': 'player_id'})[
        ['season', 'week', 'player_id', 'pfr_id', 'pos', 'ecr', 'sd', 'best', 'worst']]


def load_history():
    """One row per player-game with the usage stats we roll forward."""
    st = pd.concat([pd.read_parquet(DATA / f'stats_player_week_{y}.parquet') for y in SEASONS])
    st = st[st.season_type == 'REG']
    st = st[['player_id', 'season', 'week', 'team', 'opponent_team', 'position', 'fantasy_points_ppr',
             'targets', 'carries', 'attempts', 'target_share', 'air_yards_share', 'wopr']]
    ep = pd.concat([pd.read_parquet(DATA / f'ep_weekly_{y}.parquet') for y in SEASONS])
    ep = ep[['player_id', 'season', 'week', 'total_fantasy_points_exp', 'rush_attempt', 'rush_attempt_team',
             'total_fantasy_points_exp_team']].astype({'week': int, 'season': int})
    ep = ep.groupby(['player_id', 'season', 'week'], as_index=False).sum()
    h = st.merge(ep, on=['player_id', 'season', 'week'], how='left')
    h['carry_share'] = h.rush_attempt / h.rush_attempt_team
    h['xfp_share'] = h.total_fantasy_points_exp / h.total_fantasy_points_exp_team
    sn = pd.concat([pd.read_parquet(DATA / f'snap_counts_{y}.parquet') for y in SEASONS])
    sn = sn[sn.game_type == 'REG'][['pfr_player_id', 'season', 'week', 'offense_pct']]
    ids = pd.read_csv(DATA / 'db_playerids.csv', usecols=['gsis_id', 'pfr_id']).dropna().drop_duplicates('pfr_id')
    sn = sn.merge(ids, left_on='pfr_player_id', right_on='pfr_id').rename(columns={'gsis_id': 'player_id'})
    h = h.merge(sn[['player_id', 'season', 'week', 'offense_pct']].drop_duplicates(['player_id', 'season', 'week']),
                on=['player_id', 'season', 'week'], how='left')
    h = h.rename(columns={'fantasy_points_ppr': 'pts', 'total_fantasy_points_exp': 'xfp'})
    h['fpoe'] = h.pts - h.xfp
    h[['team', 'opponent_team']] = h[['team', 'opponent_team']].replace(RELOCATE)
    h[['season', 'week']] = h[['season', 'week']].astype('int64')
    h['t'] = h.season * 100 + h.week
    return h.sort_values(['player_id', 't']).reset_index(drop=True)


ROLL = ['pts', 'xfp', 'fpoe', 'targets', 'carries', 'target_share', 'carry_share', 'xfp_share',
        'air_yards_share', 'wopr', 'offense_pct']


def player_form(h):
    """Form as of each game INCLUDING it; joined later to the next week strictly after."""
    g = h.groupby('player_id')
    f = h[['player_id', 'season', 't']].copy()
    for c in ROLL:
        f[f'l3_{c}'] = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
        f[f'ew_{c}'] = g[c].transform(lambda s: s.ewm(halflife=4, min_periods=1).mean())
    f['n_games'] = g.cumcount() + 1
    return f


def opp_allowed(h):
    """Points each defense allowed to a position, exponentially weighted, as of each week (inclusive)."""
    a = h.groupby(['opponent_team', 'position', 'season', 'week', 't'], as_index=False)[['pts', 'xfp']].sum()
    a = a.sort_values('t')
    g = a.groupby(['opponent_team', 'position'])
    a['opp_pts_allowed'] = g.pts.transform(lambda s: s.ewm(halflife=6).mean())
    a['opp_xfp_allowed'] = g.xfp.transform(lambda s: s.ewm(halflife=6).mean())
    return a[['opponent_team', 'position', 't', 'opp_pts_allowed', 'opp_xfp_allowed']]


def game_context():
    g = pd.read_csv(DATA / 'games.csv', usecols=['season', 'game_type', 'week', 'home_team', 'away_team',
                                                  'spread_line', 'total_line', 'roof', 'wind'])
    g = g[g.game_type == 'REG']
    g[['home_team', 'away_team']] = g[['home_team', 'away_team']].replace(RELOCATE)
    home = g.assign(team=g.home_team, opp=g.away_team, implied=g.total_line / 2 + g.spread_line / 2,
                    spread=g.spread_line, home=1)
    away = g.assign(team=g.away_team, opp=g.home_team, implied=g.total_line / 2 - g.spread_line / 2,
                    spread=-g.spread_line, home=0)
    c = pd.concat([home, away])
    c['dome'] = c.roof.isin(['dome', 'closed']).astype(int)
    c['wind'] = np.where(c.dome == 1, 0, c.wind)
    return c[['season', 'week', 'team', 'opp', 'implied', 'spread', 'total_line', 'home', 'dome', 'wind']]


if MODE == 'build':
    ecr = load_ecr()
    h = load_history()
    form = player_form(h)
    played = h[['player_id', 'season', 'week', 'team', 'pts']]
    # the player's team that week: from the game itself if he played, else his most recent team
    rows = ecr.merge(played, on=['player_id', 'season', 'week'], how='left')
    rows['t'] = (rows.season * 100 + rows.week).astype('int64')
    rows = rows.sort_values('t')
    last_team = h[['player_id', 't', 'team']].rename(columns={'team': 'last_team'}).sort_values('t')
    rows = pd.merge_asof(rows, last_team, on='t', by='player_id', allow_exact_matches=False)
    rows['team'] = rows.team.fillna(rows.last_team)
    rows = pd.merge_asof(rows.sort_values('t'), form.drop(columns='season').sort_values('t'),
                         on='t', by='player_id', allow_exact_matches=False)
    # season-to-date games this season (early-season rows lean on last season's form)
    prev = h.groupby(['player_id', 'season'], as_index=False)[['pts', 'xfp', 'target_share', 'carry_share']].mean()
    prev = prev.assign(season=prev.season + 1).rename(columns=lambda c: c if c in ('player_id', 'season') else f'prev_{c}')
    rows = rows.merge(prev, on=['player_id', 'season'], how='left')
    ctx = game_context()
    rows = rows.merge(ctx, on=['season', 'week', 'team'], how='left')
    opp = opp_allowed(h).rename(columns={'opponent_team': 'opp', 'position': 'pos'}).sort_values('t')
    rows = pd.merge_asof(rows.sort_values('t'), opp, on='t', by=['opp', 'pos'], allow_exact_matches=False)
    rows['played'] = rows.pts.notna()
    rows['log_ecr'] = np.log(rows.ecr)
    rows = rows.drop(columns=['last_team', 'pfr_id'])
    rows.to_parquet(OUT)
    print(f'{len(rows)} player-weeks, {rows.played.mean():.1%} played, seasons {rows.season.min()}-{rows.season.max()}')
    print(rows.groupby(['season']).agg(weeks=('week', 'nunique'), rows=('pts', 'size')).T.to_string())
    print('feature coverage (played rows):')
    print(rows[rows.played].notna().mean().round(2).to_string())
    sys.exit()

# ---------------- eval ----------------

f = pd.read_parquet(OUT)
f = f[f.played & f.implied.notna()].copy()
f['top'] = f.groupby(['season', 'week', 'pos']).ecr.rank(method='first') <= f.pos.map(TOP_N)
f = f[f.top]
FORM = [c for c in f.columns if c.startswith(('l3_', 'ew_', 'prev_'))] + ['n_games']
VEGAS = ['implied', 'spread', 'total_line', 'home', 'dome', 'wind']
OPP = ['opp_pts_allowed', 'opp_xfp_allowed']
ECRF = ['ecr', 'log_ecr', 'sd', 'best', 'worst']
FEATSETS = {
    'model: ECR + form + vegas + opp': ECRF + FORM + VEGAS + OPP,
    'model: ECR + form (no vegas)': ECRF + FORM + OPP,
    'model: no ECR (stats + vegas only)': FORM + VEGAS + OPP,
}


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                         min_samples_leaf=100, l2_regularization=1.0, random_state=0)


TEST = range(2021, 2027)
for name in ['ecr_pts', *FEATSETS]:
    f[name] = np.nan
for s in TEST:
    for pos in POS:
        tr = f[(f.season < s) & (f.pos == pos)]
        te_idx = f[(f.season == s) & (f.pos == pos)].index
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip').fit(tr.ecr, tr.pts)
        f.loc[te_idx, 'ecr_pts'] = iso.predict(f.loc[te_idx, 'ecr'])
        for name, cols in FEATSETS.items():
            f.loc[te_idx, name] = gbm().fit(tr[cols], tr.pts).predict(f.loc[te_idx, cols])

# ---- residual model: a few signals ECR misses, selected per season from earlier seasons only ----
f['xfp_minus_ecr'] = f.ew_xfp - f.ecr_pts  # recent role says more/less than ECR implies
f['pts_minus_ecr'] = f.ew_pts - f.ecr_pts
f['wind'] = f.wind.fillna(8.0)
CANDIDATES = ['xfp_minus_ecr', 'pts_minus_ecr', 'ew_fpoe', 'l3_fpoe', 'ew_xfp_share', 'ew_target_share',
              'ew_carry_share', 'ew_offense_pct', 'implied', 'spread', 'opp_pts_allowed', 'opp_xfp_allowed',
              'sd', 'wind', 'home', 'n_games']
f['resid_model'] = np.nan
picked = {}
for pos in POS:
    for s in range(2023, 2027):
        tr = f[(f.pos == pos) & (f.season.between(2021, s - 1))].copy()
        tr['resid'] = tr.pts - tr.ecr_pts
        keep = []
        for c in CANDIDATES:  # same-sign correlation with ECR's error in every earlier season
            by = tr.groupby('season').apply(lambda d: d[c].corr(d.resid, method='spearman'))
            if (np.sign(by) == np.sign(by.mean())).all() and abs(by.mean()) >= 0.02:
                keep.append(c)
        picked[(pos, s)] = keep
        idx = f[(f.pos == pos) & (f.season == s)].index
        if not keep:
            f.loc[idx, 'resid_model'] = f.loc[idx, 'ecr_pts']
            continue
        med = tr[keep].median()
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 20))).fit(tr[keep].fillna(med), tr.resid)
        f.loc[idx, 'resid_model'] = f.loc[idx, 'ecr_pts'] + m.predict(f.loc[idx, keep].fillna(med))

# ---- GBM blend: ECR points + k * (GBM - ECR points), k per position picked on earlier seasons only ----
MAIN = 'model: ECR + form + vegas + opp'
K = np.linspace(0, 1, 11)


def pair_acc(df, col, max_gap=10):
    """Share of close start/sit pairs (ECR ranks within max_gap) where `col` picks the higher scorer."""
    r, p, e = df[col].to_numpy(), df.pts.to_numpy(), df.ecr.to_numpy()
    i, j = np.triu_indices(len(df), 1)
    keep = (np.abs(e[i] - e[j]) <= max_gap) & (p[i] != p[j])
    return ((r[i] > r[j]) == (p[i] > p[j]))[keep].mean() if keep.any() else np.nan


f['ecr_rank_score'] = -f.ecr
# break ties in every score by ECR so isotonic plateaus never get a free pass
for c in ['ecr_pts', 'resid_model', *FEATSETS]:
    f[c] = f[c] - 1e-6 * f.ecr
for pos in POS:
    for s in TEST:
        past = f[(f.pos == pos) & (f.season < s) & f[MAIN].notna()]
        if past.season.nunique() == 0:
            k = 0.0
        else:
            score = []
            for k_ in K:
                b = past.ecr_pts + k_ * (past[MAIN] - past.ecr_pts)
                score.append(np.nanmean([pair_acc(w.assign(_b=b.loc[w.index]), '_b') for _, w in past.groupby(['season', 'week'])]))
            k = K[int(np.nanargmax(score))]
        idx = f[(f.pos == pos) & (f.season == s)].index
        f.loc[idx, 'blend'] = f.loc[idx, 'ecr_pts'] + k * (f.loc[idx, MAIN] - f.loc[idx, 'ecr_pts'])
        f.loc[idx, 'blend_k'] = k

test = f[f.season >= 2023]  # 2021-22 only train the blend weight / residual signal selection
SOURCES = ['ecr_rank_score', *FEATSETS, 'blend', 'resid_model']
rng = np.random.default_rng(0)
print(f'\nWALK-FORWARD {test.season.min()}-{test.season.max()}, top-N played players per position-week')
for pos in POS:
    t = test[test.pos == pos]
    weeks = [w for _, w in t.groupby(['season', 'week']) if len(w) >= 10]
    res, base = [], None
    for src in SOURCES:
        sp = np.array([w[src].corr(w.pts, method='spearman') for w in weeks])
        pa = np.array([pair_acc(w, src) for w in weeks])
        if base is None:
            base = (sp, pa)
        dsp, dpa = sp - base[0], pa - base[1]
        bs = [rng.choice(dpa, len(dpa)).mean() for _ in range(2000)]
        res.append(dict(source=src, spearman=sp.mean(), pair_acc=np.nanmean(pa), d_pair_acc=np.nanmean(dpa),
                        ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5),
                        mae=(t[src] - t.pts).abs().mean() if src != 'ecr_rank_score' else (t.ecr_pts - t.pts).abs().mean()))
    print(f'\n{pos}: {len(t)} player-weeks, {len(weeks)} weeks, blend k by season {t.groupby("season").blend_k.first().to_dict()}')
    print(pd.DataFrame(res).to_string(index=False, float_format='%.3f'))
    print('  residual signals picked:', {s: picked[(pos, s)] for s in range(2023, 2027)})

f.to_parquet(DATA.parent / 'player_walkforward.parquet')
