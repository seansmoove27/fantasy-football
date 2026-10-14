import sys
from pathlib import Path
import numpy as np, pandas as pd
D = Path(sys.argv[1])
g = pd.read_csv(D/'games.csv')
g = g[(g.season>=2018)&(g.season<=2025)&(g.game_type=='REG')]
print('corr spread_line vs result(home margin):', g.spread_line.corr(g.result))
print('mean result when spread>0:', g[g.spread_line>0].result.mean(), ' spread<0:', g[g.spread_line<0].result.mean())
# team abbreviations consistency
for y in [2018, 2020, 2025]:
    p = pd.read_parquet(D/f'pbp_{y}.parquet', columns=['game_id','home_team','away_team','season_type','total_home_score','total_away_score'])
    p = p[p.season_type=='REG']
    gp = p.groupby('game_id').agg(h=('home_team','first'), a=('away_team','first'), hs=('total_home_score','last'), as_=('total_away_score','last')).reset_index()
    m = gp.merge(g, on='game_id', how='left')
    print(y, 'games', len(gp), 'unmatched', m.season.isna().sum(), 'home mismatch', (m.h!=m.home_team).sum(),
          'score mismatch', ((m.hs!=m.home_score)|(m.as_!=m.away_score)).sum())
    print('  missing lines', m.spread_line.isna().sum(), m.total_line.isna().sum())
print(sorted(g.roof.dropna().unique()))
