import sys, numpy as np, pandas as pd
sys.path.insert(0,'/mnt/user-data/uploads'); sys.path.insert(0,'/home/claude')
import permanova as P
d=pd.read_csv('/mnt/user-data/uploads/merged_dataset.tsv',sep='\t',keep_default_na=False,na_values=[''])
d=d.merge(pd.read_csv('/mnt/user-data/uploads/virus_annotations.tsv',sep='\t'),on='virus_group')
d=d[d.Baltimore_group!='VII'].reset_index(drop=True); d['Baltimore_group']=d.Baltimore_group.fillna('NA').astype(str)
mut=P.find_mutation_columns(d.columns)
Xc=P.clr(d[mut].to_numpy(float),None); g=Xc@Xc.T; sq=np.diag(g); G=P.gower(np.maximum(sq[:,None]+sq[None,:]-2*g,0))
rng=np.random.default_rng(3)
for t in ['Baltimore_group','host','enveloped']:
    blk=[P.dummies(d[t])]; ss,k=P.hat_trace(blk,G); r2=ss/np.trace(G); N=len(d)
    F=(ss/(k-1))/((np.trace(G)-ss)/(N-k)); ge=1
    for _ in range(9999):
        p=rng.permutation(N); s2,_=P.hat_trace(blk,G[np.ix_(p,p)])
        if (s2/(k-1))/((np.trace(G)-s2)/(N-k))>=F: ge+=1
    print(f"{t}: marginal R2={r2:.3f} df={k-1} F={F:.1f} p={ge/10000:.3g}")
