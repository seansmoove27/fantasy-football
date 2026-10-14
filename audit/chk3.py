import sys
from pathlib import Path
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
D = Path(sys.argv[1])
p = pd.read_parquet(D/'pbp_2023.parquet')
p = p[p.season_type=='REG']
k = p[p.play_type=='kickoff']
print('kickoff: posteam==home?', 'sample:'); print(k[['posteam','defteam','desc']].head(3).to_string())
# fumble_lost: does recovering team == defteam?
f = p[p.fumble_lost==1]
f = f.assign(rec_is_def = f.fumble_recovery_1_team==f.defteam)
print(f.groupby('play_type').rec_is_def.agg(['mean','sum','count']))
print(f[~f.rec_is_def][['play_type','posteam','defteam','fumble_recovery_1_team','desc']].head(8).to_string())
# total fumble recoveries by non-fumbling team vs fumble_lost-based count
# safeties
s = p[p.safety==1]; print('safeties', len(s)); print(s[['play_type','posteam','defteam','desc']].to_string()[:2500])
# TDs scored by team not posteam or on kickoffs/punts
t = p[p.touchdown==1]
odd = t[(t.td_team!=t.posteam)|(t.play_type.isin(['kickoff','punt']))]
print(odd.groupby(['play_type', odd.td_team==odd.posteam]).size())
print(t[(t.play_type=='kickoff')][['posteam','defteam','td_team','desc']].to_string()[:3000])
