"""Do sportsbook prop lines (ESPN public odds API, 2024+) improve the D/ST and kicker models?

Props only exist from 2024, so each prop model is stacked on the existing model: inputs are our walk-forward
prediction plus prop features, refit every week on all earlier weeks that have props (expanding window).
Baseline is our walk-forward prediction alone on the same team-games. Test weeks: 2025 week 4 on (the first
weeks of 2025 and all of 2024 are training only) through 2026. Same acceptance rule: 95% CI above zero.

Prop expectations: each market's main line (no-vig over probability closest to 50%) is turned into a mean,
mu = line + sd * z(p_over), with a fixed per-market spread. Yes/no "to throw an interception" uses -ln(1 - p).
Usage: python -I props_experiments.py <data_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from harness import ridge, scorecard  # noqa: E402

DATA = Path(sys.argv[1])
FIX = {'LAR': 'LA', 'WSH': 'WAS', 'JAC': 'JAX', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
START = 202408  # first test week (season*100 + week); earlier prop weeks are training only
SD = {'Total Kicking Points': 3.5, 'Total Field Goals Made': 0.9, 'Team Total Points': 9.0,
      'Total Passing Interceptions': 0.8, 'Total Defensive Interceptions': 0.3, 'Total Sacks': 0.6}


def no_vig_over(over, under):
    p = lambda a: np.where(a > 0, 100 / (a + 100), -a / (-a + 100))
    po, pu = p(over), p(under)
    return po / (po + pu)


props = pd.read_parquet(DATA / 'espn_props.parquet')
props['base_market'] = props.market.str.replace(r' \(incl\. overtime\)', '', regex=True)
# over and under arrive as separate entries for the same line: pair them
props = (props.groupby(['game_id', 'base_market', 'espn_athlete', 'espn_team', 'line'], dropna=False)
         [['over', 'under']].first().reset_index().dropna(subset=['over', 'under']))
props['p_over'] = no_vig_over(props.over.to_numpy(), props.under.to_numpy())
key = ['game_id', 'base_market', 'espn_athlete', 'espn_team']
props['dist'] = (props.p_over - 0.5).abs()
main = props.sort_values('dist').drop_duplicates([c for c in key], keep='first').copy()
yesno = main.base_market == 'To throw an interception'
main['mu'] = np.where(yesno, -np.log(1 - main.p_over.clip(0.01, 0.99)),
                      main.line + main.base_market.map(SD).fillna(1) * norm.ppf(main.p_over.clip(0.01, 0.99)))
main['base_market'] = main.base_market.replace({'To throw an interception': 'Total Passing Interceptions'})
main = main.sort_values('dist').drop_duplicates(key)  # yes/no and over/under INT markets can both exist

# ---- map athletes to teams for that game: weekly rosters (espn_id), falling back to the ID crosswalk
games = pd.read_csv(DATA / 'games.csv', usecols=['game_id', 'season', 'week', 'home_team', 'away_team', 'espn'])
ros = pd.concat(pd.read_parquet(DATA / f'roster_weekly_{y}.parquet', columns=['season', 'week', 'team', 'espn_id', 'gsis_id'])
                for y in (2024, 2025, 2026))
ros['team'] = ros.team.replace(FIX)
ids = pd.read_csv(DATA / 'db_playerids.csv', usecols=['espn_id', 'gsis_id'], low_memory=False).dropna()
ids['espn_id'] = ids.espn_id.astype('int64').astype(str)
ros_e = ros.dropna(subset=['espn_id']).assign(espn_id=lambda r: r.espn_id.astype(str).str.replace(r'\.0$', '', regex=True))
ros_g = ros.dropna(subset=['gsis_id']).merge(ids, on='gsis_id', suffixes=('_r', ''))[['season', 'week', 'team', 'espn_id']]
roster = pd.concat([ros_e[['season', 'week', 'team', 'espn_id']], ros_g]).drop_duplicates(['season', 'week', 'espn_id'])
pl = main[main.espn_athlete.notna()].merge(games[['game_id', 'season', 'week', 'home_team', 'away_team']], on='game_id')
pl = pl.merge(roster.rename(columns={'espn_id': 'espn_athlete'}), on=['season', 'week', 'espn_athlete'], how='left')
pl = pl[(pl.team == pl.home_team) | (pl.team == pl.away_team)]
print(f'player props mapped to a team in that game: {len(pl)} of {main.espn_athlete.notna().sum()}')

# team-level props: ESPN team ids -> abbreviations
tt = main[(main.base_market == 'Team Total Points') & main.espn_team.notna()].merge(games, on='game_id')
espn_team_abbr = {'1': 'ATL', '2': 'BUF', '3': 'CHI', '4': 'CIN', '5': 'CLE', '6': 'DAL', '7': 'DEN', '8': 'DET', '9': 'GB',
                  '10': 'TEN', '11': 'IND', '12': 'KC', '13': 'LV', '14': 'LA', '15': 'MIA', '16': 'MIN', '17': 'NE',
                  '18': 'NO', '19': 'NYG', '20': 'NYJ', '21': 'PHI', '22': 'ARI', '23': 'PIT', '24': 'LAC', '25': 'SF',
                  '26': 'SEA', '27': 'TB', '28': 'WAS', '29': 'CAR', '30': 'JAX', '33': 'BAL', '34': 'HOU'}
tt['team'] = tt.espn_team.map(espn_team_abbr)
team_total = tt[['game_id', 'team', 'mu']].rename(columns={'mu': 'tt_mu'})

agg = lambda m: pl[pl.base_market == m].groupby(['game_id', 'team']).mu.sum()
kick = pl[pl.base_market == 'Total Kicking Points'].groupby(['game_id', 'team']).mu.max().rename('kp_mu')
fgm = pl[pl.base_market == 'Total Field Goals Made'].groupby(['game_id', 'team']).mu.max().rename('fgm_mu')
qb_int = pl[pl.base_market == 'Total Passing Interceptions'].groupby(['game_id', 'team']).mu.max().rename('qbint_mu')
sacks = agg('Total Sacks').rename('sack_mu')
dints = agg('Total Defensive Interceptions').rename('dint_mu')
feat = pd.concat([kick, fgm, qb_int, sacks, dints], axis=1).reset_index()
print('team-games with each prop:', feat.notna().sum().to_dict(), '| team totals:', len(team_total))


def stacked(f, base_col, cols, target):
    """Expanding-window stack: each test week refit on all earlier rows (that have the features)."""
    f = f.sort_values(['season', 'week']).reset_index(drop=True)
    order = f.season * 100 + f.week
    test_weeks = sorted(order[(order >= START)].unique())
    pred = np.full(len(f), np.nan)
    X = f[[base_col] + cols]
    for w in test_weeks:
        tr = (order < w) & X.notna().all(axis=1)
        te = (order == w)
        if tr.sum() < 200:
            continue
        Xte = X[te].fillna(X[tr].mean())
        pred[te.to_numpy()] = ridge().fit(X[tr], f.loc[tr, target]).predict(Xte)
    return f, pred, order >= START


def report(f, base_col, ideas, target, label):
    """Each idea is scored against the baseline on the rows where that idea has a prediction."""
    f = f.sort_values(['season', 'week']).reset_index(drop=True)
    in_test = (f.season * 100 + f.week >= START).to_numpy()
    print(f'\n{label} (test from {START // 100} wk{START % 100})')
    for name, cols in ideas.items():
        _, p, _ = stacked(f, base_col, cols, target)
        preds = {'baseline (current model)': f[base_col].to_numpy(), name: p}
        if len(cols) == 1:
            preds[f'{cols[0]} alone'] = f[cols[0]].to_numpy()
        m = in_test & ~np.isnan(np.column_stack(list(preds.values()))).any(axis=1)
        test = f[m]
        card = scorecard(test, target, {k: v[m] for k, v in preds.items()}, 'baseline (current model)', None)
        print(f'  {name}: {len(test)} team-games, {test.groupby(["season", "week"]).ngroups} weeks')
        print('   ' + card.to_string(index=False, float_format='%.3f').replace('\n', '\n   '))


# ================= KICKER =================
k = pd.read_parquet(DATA.parent / 'k_model_walkforward.parquet')
k = k[k.season >= 2024].merge(feat, on=['game_id', 'team'], how='left')
k = k.dropna(subset=['kp_mu'])
report(k, 'ridge, all', {'+ kicking points prop': ['kp_mu'], '+ kicking pts + FGs made props': ['kp_mu', 'fgm_mu']},
       'k_pts', 'KICKER')

# ================= D/ST =================
d = pd.read_parquet(DATA.parent / 'dst_v2_walkforward.parquet')
d = d[d.season >= 2024]
d = d.merge(team_total.rename(columns={'team': 'opp', 'tt_mu': 'opp_tt_mu'}), on=['game_id', 'opp'], how='left')
d = d.merge(feat[['game_id', 'team', 'qbint_mu']].rename(columns={'team': 'opp', 'qbint_mu': 'opp_qbint_mu'}),
            on=['game_id', 'opp'], how='left')
d = d.merge(feat[['game_id', 'team', 'sack_mu', 'dint_mu']], on=['game_id', 'team'], how='left')
d['neg_opp_tt_mu'] = -d.opp_tt_mu
for c in ('opp_qbint_mu', 'sack_mu', 'dint_mu'):
    print(f'D/ST rows with {c}: {d[c].notna().mean():.0%}', end='; ')
print(f'opp team total: {d.opp_tt_mu.notna().mean():.0%}')
report(d, 'ridge, all + weather',
       {'+ opp QB interceptions prop': ['opp_qbint_mu'], '+ own defenders sacks props': ['sack_mu'],
        '+ QB INT + sacks props': ['opp_qbint_mu', 'sack_mu'], '+ opp team total (2025+)': ['neg_opp_tt_mu']},
       'dst_pts', 'D/ST')
