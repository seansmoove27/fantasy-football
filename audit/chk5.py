import sys
from pathlib import Path
import pandas as pd
D = Path(sys.argv[1])
p = pd.read_parquet(D/'pbp_2023.parquet', columns=['field_goal_attempt','kick_distance','yardline_100','desc','play_type','posteam','drive','extra_point_result','two_point_attempt'])
fg = p[p.field_goal_attempt==1].copy()
fg['desc_d'] = fg.desc.str.extract(r'(\d+) yard field goal').astype(float)
print('desc==kick_distance', (fg.desc_d==fg.kick_distance).mean(), fg.desc_d.isna().sum())
print(fg[fg.kick_distance!=fg.yardline_100+18][['kick_distance','yardline_100','desc']].head(5).to_string())
nd = p[(p.yardline_100<=20)&p.posteam.notna()&p.drive.isna()]
print(nd.play_type.value_counts(dropna=False).to_dict())
