import sys, pandas as pd
df = pd.read_parquet(sys.argv[1])
df = df[df.season_type == 'REG']
print('weeks:', sorted(df.week.unique()), 'plays:', len(df))
plays = df[df.play_type.isin(['pass', 'run'])]
dropbacks = df[(df.qb_dropback == 1)]

def per_team(group_col):
    g_db = dropbacks.groupby(group_col)
    g_pl = plays.groupby(group_col)
    out = pd.DataFrame({
        'games': df.groupby(group_col).game_id.nunique(),
        'dropbacks': g_db.size(),
        'sacks': g_db.sack.sum(),
        'qb_hits': g_db.qb_hit.sum(),
        'ints': df.groupby(group_col).interception.sum(),
        'fum_lost': df.groupby(group_col).fumble_lost.sum(),
        'plays': g_pl.size(),
        'explosive': g_pl.apply(lambda p: ((p.play_type == 'pass') & (p.yards_gained >= 20) | (p.play_type == 'run') & (p.yards_gained >= 10)).sum()),
        'yards': g_pl.yards_gained.sum(),
        'epa': g_pl.epa.mean(),
    })
    out['sack_rate'] = out.sacks / out.dropbacks * 100
    out['hit_rate'] = (out.qb_hits) / out.dropbacks * 100
    out['to_per_g'] = (out.ints + out.fum_lost) / out.games
    out['to_rate'] = (out.ints + out.fum_lost) / out.plays * 100
    out['expl_rate'] = out.explosive / out.plays * 100
    out['ypp'] = out.yards / out.plays
    return out

defense = per_team('defteam')
offense = per_team('posteam')
cols = ['games', 'sack_rate', 'hit_rate', 'to_per_g', 'to_rate', 'expl_rate', 'ypp', 'epa']

def show(tbl, teams, title, best_low):
    ranked = tbl[cols].copy()
    for c in cols[1:]:
        asc = c in best_low
        ranked[c + '_rk'] = tbl[c].rank(ascending=asc, method='min').astype(int)
    print('\n' + title)
    for t in teams:
        r = ranked.loc[t]
        print(t, ' '.join(f"{c}={r[c]:.2f}(#{int(r[c+'_rk'])})" if c != 'games' else f"g={int(r[c])}" for c in cols))

# defense: high sack/hit/TO good; low explosive/ypp/epa good
show(defense, ['NYG', 'IND', 'NO', 'SF'], 'DEFENSE (rank 1 = best for a fantasy D)', best_low={'expl_rate', 'ypp', 'epa'})
# opposing offense: rank 1 = most generous to a fantasy D (high sacks/TO allowed, low explosive/ypp/epa)
show(offense, ['NO', 'TEN', 'NYG', 'WAS'], 'OPPOSING OFFENSE (rank 1 = best matchup for the D)', best_low={'expl_rate', 'ypp', 'epa'})
