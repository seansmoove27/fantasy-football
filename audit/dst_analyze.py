import sys
from pathlib import Path
import numpy as np, pandas as pd
S = pd.read_parquet(sys.argv[1]); P = pd.read_parquet(sys.argv[2]); G = pd.read_csv(sys.argv[3], usecols=['game_id','home_team','spread_line','total_line','result'])
G['home_team'] = G.home_team.replace({'OAK':'LV'})
print('pts_orig matches script dst_pts:', (S.merge(P,on=['game_id','team']).eval('pts_orig==dst_pts')).mean())
print('fumble recs orig vs fix per season-total:', P.fr_orig.sum(), P.fr_fix.sum(), 'rows differing', (P.fr_orig!=P.fr_fix).sum(), 'of', len(P))
print('mean pts orig/fix/fix_pa', P.pts_orig.mean(), P.pts_fix.mean(), P.pts_fix_pa.mean(), 'rows w/ opp nonoff TD', (P.opp_nonoff>0).mean())
print('corr orig vs fix_pa', P.pts_orig.corr(P.pts_fix_pa))
# features for all weeks
P = P.merge(G, on='game_id')
ih = P.team==P.home_team
P['F_opp_implied_total'] = -np.where(ih,(P.total_line-P.spread_line)/2,(P.total_line+P.spread_line)/2)
P['F_spread'] = np.where(ih,P.spread_line,-P.spread_line)
P = P.sort_values(['season','week'])
# opponent offense dst allowed to date (pts_orig of D/STs facing opp)
off = P[['season','week','game_id','opp','pts_orig']].rename(columns={'opp':'off'}).sort_values(['season','off','week'])
gg = off.groupby(['season','off'])
off['n'] = gg.cumcount(); off['s'] = gg.pts_orig.transform(lambda s: s.shift().expanding().sum())
off['F_opp_dst_allowed_pg'] = off.s/off.n
P = P.merge(off[['game_id','off','F_opp_dst_allowed_pg','n']].rename(columns={'off':'opp','n':'opp_n'}), on=['game_id','opp'])
chk = S.merge(P[['game_id','team','F_opp_dst_allowed_pg']], on=['game_id','team'], suffixes=('','_me'))
print('opp_dst_allowed recompute match:', np.allclose(chk.F_opp_dst_allowed_pg, chk.F_opp_dst_allowed_pg_me))
# leakage test: correlate with *including* current game version should be higher; ours must not equal
def ws(df, col, y):
    return np.array([g[col].corr(g[y], method='spearman') for _, g in df.groupby(['season','week']) if len(g)>5])
# LV impact in script sample
S2 = S.merge(P[['game_id','team','F_opp_implied_total','F_spread','pts_fix','pts_fix_pa']], on=['game_id','team'], suffixes=('','_fix'))
print('rows w/ changed implied', (S2.F_opp_implied_total!=S2.F_opp_implied_total_fix).sum())
for col in ['F_opp_implied_total','F_opp_implied_total_fix','F_spread','F_spread_fix','F_opp_dst_allowed_pg','F_def_sack_rate','F_def_takeaways_pg']:
    print(f'{col:28s}', ' '.join(f'{y}={ws(S2,col,y).mean():.3f}' for y in ['dst_pts','pts_fix','pts_fix_pa']))
# per-season, bootstrap
rng = np.random.default_rng(0)
print('\nper season weekly spearman (fixed pts, fixed lines): implied | spread | opp_dst_allowed | def_sack_rate')
for s, d in S2.groupby('season'):
    print(s, ' '.join(f'{ws(d,c,"pts_fix").mean():.3f}' for c in ['F_opp_implied_total_fix','F_spread_fix','F_opp_dst_allowed_pg','F_def_sack_rate']))
for c in ['F_opp_implied_total_fix','F_spread_fix','F_opp_dst_allowed_pg','F_def_sack_rate','F_def_takeaways_pg']:
    v = ws(S2,c,'pts_fix'); b=[rng.choice(v,len(v)).mean() for _ in range(2000)]
    print(f'{c:26s} mean {v.mean():.3f} 95%CI [{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}] nweeks {len(v)}')
v = ws(S2,'F_opp_implied_total_fix','pts_fix')-ws(S2,'F_opp_dst_allowed_pg','pts_fix'); b=[rng.choice(v,len(v)).mean() for _ in range(2000)]
print('implied minus opp_dst_allowed diff', v.mean(), np.percentile(b,[2.5,97.5]))
print('excl 2020:', {c: round(ws(S2[S2.season!=2020],c,'pts_fix').mean(),3) for c in ['F_opp_implied_total_fix','F_spread_fix','F_opp_dst_allowed_pg','F_def_sack_rate']})
W2 = P[(P.week>=2)&(P.opp_n>=1)].dropna(subset=['F_opp_dst_allowed_pg'])
print('weeks 2+ n=',len(W2), {c: round(ws(W2,c,'pts_fix').mean(),3) for c in ['F_opp_implied_total','F_spread','F_opp_dst_allowed_pg']})
W = P.copy(); print('all weeks implied (no std filter) n=',len(W), round(ws(W,'F_opp_implied_total','pts_fix').mean(),3))
# pooled vs weekly
print('pooled spearman implied', S2.F_opp_implied_total_fix.corr(S2.pts_fix, method='spearman'))
# OOS with fixes
feats = [c for c in S.columns if c.startswith('F_')]
S2['F_opp_implied_total']=S2.F_opp_implied_total_fix; S2['F_spread']=S2.F_spread_fix
for excl in [False, True]:
    d = S2[S2.season!=2020] if excl else S2
    tr, te = d[d.season<=2022], d[d.season>=2023].copy()
    for name, cols in [('implied',['F_opp_implied_total']),('everything',feats),('no vegas',[c for c in feats if c not in ('F_opp_implied_total','F_spread','F_game_total')])]:
        X=np.column_stack([np.ones(len(tr))]+[tr[c] for c in cols]); b=np.linalg.lstsq(X,tr.pts_fix,rcond=None)[0]
        te['pred']=np.column_stack([np.ones(len(te))]+[te[c] for c in cols])@b
        print('excl2020' if excl else 'all', name, round(ws(te,'pred','pts_fix').mean(),3), 'coef' if name=='implied' else '', b.round(3) if name=='implied' else '')
# multicollinearity
X = S2[feats]; C = np.corrcoef(X.values.T); vif = np.diag(np.linalg.inv(C))
print('VIF', dict(zip([f[2:] for f in feats], vif.round(1))))
print('corr implied/spread/total', S2[['F_opp_implied_total','F_spread','F_game_total']].corr().round(3).values.tolist())
