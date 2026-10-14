import sys
from pathlib import Path
import pandas as pd
D = Path(sys.argv[1])
g = pd.read_csv(D/'games.csv')
for y in range(2018, 2026):
    p = pd.read_parquet(D/f'pbp_{y}.parquet', columns=['game_id','home_team','away_team','season_type'])
    p = p[p.season_type=='REG'].groupby('game_id').first().reset_index()
    m = p.merge(g[['game_id','home_team','away_team']], on='game_id', suffixes=('_p','_g'))
    bad = m[(m.home_team_p!=m.home_team_g)|(m.away_team_p!=m.away_team_g)]
    if len(bad): print(y, len(bad)); print(bad.to_string())
