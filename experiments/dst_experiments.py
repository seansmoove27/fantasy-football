"""D/ST feature experiments on top of dst/dst_v2.py's ridge (see harness.py for the protocol).

Ideas:
  qb         - the opposing starting QB's own career-to-date sack rate, INT rate, experience, and whether he is
               a change from his team's previous starter (follows the QB across teams)
  components - predict points allowed and yards allowed, push each through the league's tier tables with the
               residual spread, add predicted sacks and takeaways (instead of one ridge on total points)

Usage: python -I dst_experiments.py <data_dir> [dev|holdout] [accepted ideas, e.g. qb]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from harness import DEV_SEASONS, HOLDOUT_SEASONS, ridge, scorecard, walk_forward  # noqa: E402

DATA = Path(sys.argv[1])
STAGE = sys.argv[2] if len(sys.argv) > 2 else 'dev'
EXTRAS = DATA.parent / 'dst_qb_extras.parquet'
YEARS = range(2015, 2027)
RELOCATE = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}


def pa_score(pa):
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 17, pa <= 27, pa <= 34, pa <= 45], [5, 4, 3, 1, 0, -1, -3], -5)


def ya_score(ya):
    return np.select([ya < 100, ya < 200, ya < 300, ya < 350, ya < 400, ya < 450, ya < 500, ya < 550],
                     [5, 3, 2, 0, -1, -3, -5, -6], -7)


def expected_bracket(mu, sd, scorer, grid):
    w = np.exp(-0.5 * ((grid[None, :] - mu[:, None]) / sd) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    return (w * scorer(grid)[None, :]).sum(axis=1)


def build_qb_games():
    """One row per QB-game: dropbacks, sacks, interceptions."""
    rows = []
    for y in YEARS:
        pbp = pd.read_parquet(DATA / f'pbp_{y}.parquet', columns=[
            'game_id', 'season_type', 'week', 'posteam', 'qb_dropback', 'sack', 'interception', 'passer_player_id'])
        pbp = pbp[(pbp.season_type == 'REG') & (pbp.qb_dropback == 1)].dropna(subset=['passer_player_id'])
        q = pbp.groupby(['game_id', 'passer_player_id']).agg(
            dropbacks=('qb_dropback', 'sum'), sacks=('sack', 'sum'), ints=('interception', 'sum')).reset_index()
        q['season'], q['week'] = y, q.game_id.map(pbp.groupby('game_id').week.first())
        rows.append(q)
        print('qb', y, file=sys.stderr)
    return pd.concat(rows).rename(columns={'passer_player_id': 'qb_id'})


if not EXTRAS.exists():
    build_qb_games().to_parquet(EXTRAS)
qg = pd.read_parquet(EXTRAS).sort_values(['season', 'week']).reset_index(drop=True)
WINDOW = 16  # games
g = qg.groupby('qb_id')
for c in ('dropbacks', 'sacks', 'ints'):
    qg['prev_' + c] = g[c].transform(lambda s: s.shift().rolling(WINDOW, min_periods=1).sum()).fillna(0)
qg['career_db'] = g.dropbacks.transform(lambda s: s.shift().cumsum()).fillna(0)
lg_sack, lg_int = qg.sacks.sum() / qg.dropbacks.sum(), qg.ints.sum() / qg.dropbacks.sum()
qg['qb_sack_rate'] = (qg.prev_sacks + lg_sack * 150) / (qg.prev_dropbacks + 150)
qg['qb_int_rate'] = (qg.prev_ints + lg_int * 300) / (qg.prev_dropbacks + 300)
qg['qb_exp'] = np.log1p(qg.career_db)

games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'season', 'week', 'home_team', 'away_team',
                                                 'home_qb_id', 'away_qb_id'])
for c in ('home_team', 'away_team'):
    games[c] = games[c].replace(RELOCATE)
starters = pd.concat([games.rename(columns={'home_team': 'qb_team', 'home_qb_id': 'qb_id'}),
                      games.rename(columns={'away_team': 'qb_team', 'away_qb_id': 'qb_id'})])[
    ['game_id', 'season', 'week', 'qb_team', 'qb_id']].sort_values(['season', 'week'])
starters['qb_new'] = (starters.groupby('qb_team').qb_id.shift() != starters.qb_id).astype(float)
starters = starters.merge(qg[['game_id', 'qb_id', 'qb_sack_rate', 'qb_int_rate', 'qb_exp']],
                          on=['game_id', 'qb_id'], how='left')
# a starter with no dropbacks that game (rare) gets league-average rates and zero recorded experience
starters = starters.fillna({'qb_sack_rate': lg_sack, 'qb_int_rate': lg_int, 'qb_exp': 0})

f = pd.read_parquet(DATA.parent / 'dst_v2_features.parquet')
f = f.merge(starters.rename(columns={'qb_team': 'opp'})[['game_id', 'opp', 'qb_sack_rate', 'qb_int_rate', 'qb_exp', 'qb_new']],
            on=['game_id', 'opp'], how='left')
f = f.rename(columns={'qb_sack_rate': 'F_oqb_sack_rate', 'qb_int_rate': 'F_oqb_int_rate', 'qb_exp': 'F_oqb_exp',
                      'qb_new': 'F_oqb_new'})
QB = ['F_oqb_sack_rate', 'F_oqb_int_rate', 'F_oqb_exp', 'F_oqb_new']
BASE = [c for c in f.columns if c.startswith('F_') and c not in QB]
f = f[(f.n_d_ >= 3) & (f.n_o_ >= 3)].dropna(subset=BASE + QB + ['dst_pts'])
seasons = DEV_SEASONS if STAGE == 'dev' else HOLDOUT_SEASONS
f = f[f.season <= max(seasons)].sort_values(['season', 'week']).reset_index(drop=True)


def components(cols):
    """Ridge for PA, YA, sacks and the rest; PA/YA go through the tier tables with their residual spread."""
    out = np.full(len(f), np.nan)
    rest = f.dst_pts - pa_score(f.pa.to_numpy()) - ya_score(f.ya.to_numpy())
    for s in seasons:
        tr, te = (f.season < s).to_numpy(), (f.season == s).to_numpy()
        X, Xt = f.loc[tr, cols], f.loc[te, cols]
        parts = []
        for col, scorer, grid in (('pa', pa_score, np.arange(0, 80)), ('ya', ya_score, np.arange(0, 800, 5))):
            m = ridge().fit(X, f.loc[tr, col])
            sd = np.std(f.loc[tr, col] - m.predict(X))
            parts.append(expected_bracket(m.predict(Xt), sd, scorer, grid))
        parts.append(ridge().fit(X, rest[tr]).predict(Xt))
        out[te] = sum(parts)
    return out


preds = {
    'baseline (dst_v2 ridge)': walk_forward(f, BASE, 'dst_pts', seasons),
    '+ qb': walk_forward(f, BASE + QB, 'dst_pts', seasons),
    'components': components(BASE),
    'components + qb': components(BASE + QB),
}
if len(sys.argv) > 3 and sys.argv[3] == 'qb':
    preds = {k: preds[k] for k in ('baseline (dst_v2 ridge)', '+ qb')}
test = f[f.season.isin(seasons)]
preds = {k: v[test.index] for k, v in preds.items()}
print(f'\nD/ST {STAGE} {min(seasons)}-{max(seasons)}: {len(test)} team-games')
print(scorecard(test, 'dst_pts', preds, 'baseline (dst_v2 ridge)', seasons).to_string(index=False, float_format='%.3f'))
print('\nsingle-feature weekly Spearman (test rows):')
for c in QB:
    w = [g[c].corr(g.dst_pts, method='spearman') for _, g in test.groupby(['season', 'week']) if len(g) > 5]
    print(f'  {c}: {np.mean(w):.3f}')
