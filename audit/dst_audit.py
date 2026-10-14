import sys
from pathlib import Path
import numpy as np, pandas as pd
D = Path(sys.argv[1]); OUT = Path(sys.argv[2])
def pas(pa):
    for lim, s in ((0,5),(6,4),(13,3),(17,1),(27,0),(34,-1),(45,-3)):
        if pa <= lim: return s
    return -5
def yas(ya):
    for lim, s in ((100,5),(200,3),(300,2),(350,0),(400,-1),(450,-3),(500,-5),(550,-6)):
        if ya < lim: return s
    return -7
rows=[]
for y in range(2018,2026):
    p = pd.read_parquet(D/f'pbp_{y}.parquet', columns=['game_id','season','season_type','week','home_team','away_team','posteam','defteam','play_type','two_point_attempt','sack','qb_hit','qb_dropback','interception','fumble_lost','fumbled_1_team','safety','punt_blocked','field_goal_result','extra_point_result','touchdown','td_team','yards_gained','epa','total_home_score','total_away_score'])
    p = p[p.season_type=='REG']
    p = p.replace({'home_team':{'OAK':'LV'}})
    for gid, g in p.groupby('game_id'):
        home, away = g.home_team.iloc[0], g.away_team.iloc[0]
        fin = g.iloc[-1]; score={home:fin.total_home_score, away:fin.total_away_score}
        sc = g[g.play_type.isin(['pass','run'])&(g.two_point_attempt!=1)]
        fl = g[g.fumble_lost==1]
        tdall = g[(g.touchdown==1)]
        nonoff = {t: ((tdall.td_team==t)&((tdall.play_type=='kickoff')|(tdall.posteam!=t))).sum() for t in (home,away)}
        for d, opp in ((home,away),(away,home)):
            on_d = sc[sc.defteam==d]; all_d = g[g.defteam==d]
            sacks=on_d.sack.sum(); ints=all_d.interception.sum()
            fr_orig = all_d.fumble_lost.sum()
            fr_fix = (fl.fumbled_1_team==opp).sum()
            saf=all_d.safety.sum()
            blk=all_d.punt_blocked.fillna(0).sum()+(all_d.field_goal_result=='blocked').sum()+(all_d.extra_point_result=='blocked').sum()
            ya=on_d.yards_gained.sum(); pa=score[opp]
            base = sacks+2*(ints+saf+blk)+6*nonoff[d]+yas(ya)
            rows.append(dict(game_id=gid,season=g.season.iloc[0],week=g.week.iloc[0],team=d,opp=opp,
                pts_orig=base+2*fr_orig+pas(pa), pts_fix=base+2*fr_fix+pas(pa),
                pts_fix_pa=base+2*fr_fix+pas(max(pa-7*nonoff[opp],0)), fr_orig=fr_orig, fr_fix=fr_fix, opp_nonoff=nonoff[opp]))
    print('built',y,file=sys.stderr)
pd.DataFrame(rows).to_parquet(OUT)
