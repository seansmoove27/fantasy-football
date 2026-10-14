import sys
from pathlib import Path
import pandas as pd
D = Path(sys.argv[1])
cols=['game_id','season_type','week','posteam','defteam','play_type','two_point_attempt','sack','interception','yards_gained','fumble_lost','field_goal_attempt','field_goal_result','kick_distance','yardline_100','roof','drive','fixed_drive_result','extra_point_result','home_team']
for y in [2019, 2023]:
    p = pd.read_parquet(D/f'pbp_{y}.parquet', columns=cols); p=p[p.season_type=='REG']
    sc = p[p.play_type.isin(['pass','run'])&(p.two_point_attempt!=1)]
    print(y,'sacks top', sc.groupby('defteam').sack.sum().sort_values().tail(3).to_dict())
    print(y,'ints top', p.groupby('defteam').interception.sum().sort_values().tail(3).to_dict(), 'int on 2pt', p[(p.two_point_attempt==1)].interception.sum())
    print(y,'sacks with play_type != pass', p[(p.sack==1)].play_type.value_counts().to_dict())
    fg = p[p.field_goal_attempt==1]
    print(y,'fg results', fg.field_goal_result.value_counts().to_dict(), 'dist-(yl+17) diffs', (fg.kick_distance-(fg.yardline_100+17)).value_counts().head(5).to_dict(), 'nan dist', fg.kick_distance.isna().sum())
    print(y,'fga play_type', fg.play_type.value_counts().to_dict())
    print(y,'roof', p.groupby('game_id').roof.first().value_counts().to_dict())
    print(y,'nan drive among posteam yl<=20', p[(p.yardline_100<=20)&p.posteam.notna()].drive.isna().sum())
    print(y,'XP results', p.extra_point_result.value_counts().to_dict())
    print(y,'kneel/spike yards', p[p.play_type.isin(['qb_kneel','qb_spike'])].yards_gained.sum())
p = pd.read_parquet(D/'pbp_2023.parquet', columns=cols+['total_home_score']); 
g = p[p.game_id=='2023_01_DET_KC']
sc = g[g.play_type.isin(['pass','run'])&(g.two_point_attempt!=1)]
print('DET@KC 2023 wk1 yards by offense', sc.groupby('posteam').yards_gained.sum().to_dict(), 'sacks by def', sc.groupby('defteam').sack.sum().to_dict())
