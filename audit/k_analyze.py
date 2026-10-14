import sys
import numpy as np, pandas as pd
S = pd.read_parquet(sys.argv[1]); rng=np.random.default_rng(0)
def ws(df,c): return np.array([g[c].corr(g.k_pts,method='spearman') for _,g in df.groupby(['season','week']) if len(g)>5])
for s,d in S.groupby('season'): print(s, ' '.join(f'{c[2:]}={ws(d,c).mean():.3f}' for c in ['F_team_implied_total','F_dome','F_team_k_ppg']))
for c in ['F_team_implied_total','F_dome','F_team_k_ppg','F_team_rz_stall','F_opp_rz_stall']:
    v=ws(S,c); b=[rng.choice(v,len(v)).mean() for _ in range(2000)]; print(c, round(v.mean(),3), np.percentile(b,[2.5,97.5]).round(3))
print('excl2020', {c: round(ws(S[S.season!=2020],c).mean(),3) for c in ['F_team_implied_total','F_dome']})
print('dome pct', S.F_dome.mean(), S.groupby('F_dome').k_pts.mean().to_dict())
feats=[c for c in S.columns if c.startswith('F_')]
def oos(cols, d=S):
    tr,te=d[d.season<=2022],d[d.season>=2023].copy()
    b=np.linalg.lstsq(np.column_stack([np.ones(len(tr))]+[tr[c] for c in cols]),tr.k_pts,rcond=None)[0]
    te['p']=np.column_stack([np.ones(len(te))]+[te[c] for c in cols])@b; v=ws(te,'p'); r=te.p.corr(te.k_pts); return round(v.mean(),3), round(r*r,3)
print('implied+dome', oos(['F_team_implied_total','F_dome']), 'everything', oos(feats))
for c in feats:
    if c not in ('F_team_implied_total','F_dome'): print(' +',c[2:], oos(['F_team_implied_total','F_dome',c]))
v1=ws(S[S.season>=2023].assign(),'F_team_implied_total')
