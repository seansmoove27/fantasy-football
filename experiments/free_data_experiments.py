"""Experiments with free nflverse data not used yet (see harness.py for the protocol).

  pressure - PFR weekly advanced stats (2018+): each offense's pressure rate allowed and bad-throw rate, each
             defense's pressure rate, as exponentially weighted pre-game form
  injuries - final pre-game injury report: Out/Doubtful counts by position group (offensive line, skill players,
             defense) for both teams

PFR starts in 2018, so dev seasons here are 2020-2024 (every fold trains on at least two PFR seasons).
Usage: python -I free_data_experiments.py <data_dir> [dev|holdout] [accepted ideas for the holdout, e.g. pressure]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from harness import HOLDOUT_SEASONS, scorecard, walk_forward  # noqa: E402

DATA = Path(sys.argv[1])
STAGE = sys.argv[2] if len(sys.argv) > 2 else 'dev'
SEASONS = range(2020, 2025) if STAGE == 'dev' else HOLDOUT_SEASONS
FIX = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX', 'WSH': 'WAS'}
HALFLIFE = 6
OL = {'T', 'G', 'C', 'OT', 'OG', 'OL'}
SKILL = {'WR', 'TE', 'RB', 'FB'}
DEF = {'DE', 'DT', 'DL', 'NT', 'LB', 'OLB', 'ILB', 'MLB', 'CB', 'S', 'SS', 'FS', 'DB'}

# ---- per team-game offensive pressure stats from PFR (QB rows summed by team)
pfr = pd.concat(pd.read_parquet(DATA / f'advstats_week_pass_{y}.parquet') for y in range(2018, 2027))
pfr = pfr[pfr.game_type == 'REG']
pfr['team'] = pfr.team.replace(FIX)
off = pfr.groupby(['game_id', 'team'], as_index=False)[['times_pressured', 'passing_bad_throws']].sum()

# ---- injury counts per team-week (Out or Doubtful on the final report)
inj = pd.concat(pd.read_parquet(DATA / f'injuries_{y}.parquet') for y in range(2015, 2027))
inj = inj[(inj.game_type == 'REG') & inj.report_status.isin(['Out', 'Doubtful'])]
inj['team'] = inj.team.replace(FIX)
grp = np.select([inj.position.isin(OL), inj.position.isin(SKILL), inj.position.isin(DEF)], ['ol', 'skill', 'def'], 'other')
inj_counts = inj.assign(g=grp).pivot_table(index=['season', 'week', 'team'], columns='g', values='gsis_id',
                                           aggfunc='count', fill_value=0).reset_index()


def ew_form(df, key, col):
    return df.groupby(key)[col].transform(lambda s: s.shift().ewm(halflife=HALFLIFE).mean())


def add_injuries(df, team_col, prefix):
    m = df[['season', 'week', team_col]].merge(inj_counts, left_on=['season', 'week', team_col],
                                               right_on=['season', 'week', 'team'], how='left')
    for c in ('ol', 'skill', 'def'):
        df[f'F_{prefix}_{c}_out'] = m[c].fillna(0).to_numpy() if c in m else 0.0
    return df


# ================= D/ST =================
d = pd.read_parquet(DATA.parent / 'dst_v2_features.parquet')
d = d[(d.n_d_ >= 3) & (d.n_o_ >= 3)].sort_values(['season', 'week']).reset_index(drop=True)
BASE_D = [c for c in d.columns if c.startswith('F_')]
d = d.merge(off.rename(columns={'team': 'opp'}), on=['game_id', 'opp'], how='left')  # opp offense vs this defense
print(f'D/ST rows with PFR pressure data (2018+): {d[d.season >= 2018].times_pressured.notna().mean():.0%}')
d['press_rate'] = d.times_pressured / d.dropbacks
d['bad_rate'] = d.passing_bad_throws / d.dropbacks
d['F_o_pressure_allowed'] = ew_form(d, 'opp', 'press_rate')
d['F_o_bad_throw_rate'] = ew_form(d, 'opp', 'bad_rate')
d['F_d_pressure_rate'] = ew_form(d, 'team', 'press_rate')
d = add_injuries(d, 'opp', 'o')   # opponent offense
d = add_injuries(d, 'team', 'd')  # this defense
IDEAS_D = {
    'pressure': ['F_o_pressure_allowed', 'F_o_bad_throw_rate', 'F_d_pressure_rate'],
    'injuries': ['F_o_ol_out', 'F_o_skill_out', 'F_d_def_out'],
}

# ================= K =================
k = pd.read_parquet(DATA.parent / 'k_model_features.parquet').sort_values(['season', 'week']).reset_index(drop=True)
BASE_K = [c for c in k.columns if c.startswith('F_')]
k = k.merge(off, on=['game_id', 'team'], how='left')
k['F_off_pressure_count'] = ew_form(k, 'team', 'times_pressured')
k = add_injuries(k, 'team', 'k_team')  # kicker's own offense
k = add_injuries(k, 'opp', 'k_opp')    # opposing defense
IDEAS_K = {
    'pressure': ['F_off_pressure_count'],
    'injuries': ['F_k_team_ol_out', 'F_k_team_skill_out', 'F_k_opp_def_out'],
}


def run(f, base, ideas, target, label):
    f = f[f.season >= 2018].dropna(subset=base + sum(ideas.values(), []) + [target])
    f = f[f.season <= max(SEASONS)].reset_index(drop=True)
    sets = {'baseline': base, **{f'+ {k}': base + v for k, v in ideas.items()}, '+ both': base + sum(ideas.values(), [])}
    if len(sys.argv) > 3:
        sets = {'baseline': base, '+ accepted': base + sum((ideas[i] for i in sys.argv[3].split(',')), [])}
    test = f[f.season.isin(SEASONS)]
    preds = {name: walk_forward(f, cols, target, SEASONS)[test.index] for name, cols in sets.items()}
    print(f'\n{label} {STAGE} {min(SEASONS)}-{max(SEASONS)}: {len(test)} team-games (train from 2018)')
    print(scorecard(test, target, preds, 'baseline', SEASONS).to_string(index=False, float_format='%.3f'))
    for c in sum(ideas.values(), []):
        w = [g[c].corr(g[target], method='spearman') for _, g in test.groupby(['season', 'week']) if len(g) > 5]
        print(f'  alone {c}: {np.nanmean(w):.3f}')


run(d, BASE_D, IDEAS_D, 'dst_pts', 'D/ST')
run(k, BASE_K, IDEAS_K, 'k_pts', 'KICKER')
