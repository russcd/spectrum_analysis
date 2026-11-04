import sys, numpy as np, pandas as pd
sys.path.insert(0,'/mnt/user-data/uploads'); sys.path.insert(0,'/home/claude')
import permanova as P
from family_map import family
rng=np.random.default_rng(2)
d=pd.read_csv('/mnt/user-data/uploads/merged_dataset.tsv',sep='\t',keep_default_na=False,na_values=[''])
d=d.merge(pd.read_csv('/mnt/user-data/uploads/virus_annotations.tsv',sep='\t'),on='virus_group')
d['family']=d.virus_group.map(family); d['Baltimore_group']=d.Baltimore_group.fillna('NA').astype(str)
d=d[d.Baltimore_group!='VII'].reset_index(drop=True)
comp={'A':'T','C':'G','G':'C','T':'A'}
mut=P.find_mutation_columns(d.columns)
pairs={}
for m in mut:
    rc=comp[m[0]]+comp[m[1]]; key=tuple(sorted([m,rc])); pairs[key]=1
S=pd.DataFrame({'/'.join(k):d[k[0]]+d[k[1]] for k in pairs})
S=S.div(S.sum(1),axis=0)
def G_of(X):
    Xc=P.clr(X,None); g=Xc@Xc.T; sq=np.diag(g)
    return P.gower(np.maximum(sq[:,None]+sq[None,:]-2*g,0)),Xc
def one_way(G,lab):
    codes,u=pd.factorize(lab);k=len(u);N=len(lab);Z=np.eye(k)[codes];n=Z.sum(0)
    ssb=np.sum(np.einsum('ig,ij,jg->g',Z,G,Z)/n);sst=np.trace(G)
    return ssb/sst,(ssb/(k-1))/((sst-ssb)/(N-k))
def pval(G,lab,n):
    r,F=one_way(G,lab);ge=1
    for _ in range(n):
        if one_way(G,rng.permutation(lab))[1]>=F: ge+=1
    return r,F,ge/(n+1)
for name,X in [('12-type (reference strand)',d[mut].to_numpy(float)),('6-type (strand-symmetric)',S.to_numpy())]:
    G,Xc=G_of(X); r,F,p=pval(G,d.Baltimore_group.values,9999)
    print(f"{name}: R2={r:.3f} F={F:.1f} p={p:.4g}")
    # one per family
    R=[]
    for b in range(500):
        s=d.groupby('family').sample(1,random_state=int(rng.integers(1e9)))
        vc=s.Baltimore_group.value_counts(); s=s[s.Baltimore_group.isin(vc[vc>=2].index)]
        Xs=(X[s.index] if True else None)
        Gs,_=G_of(Xs); R.append(one_way(Gs,s.Baltimore_group.values)[0])
    print(f"   one-per-family median R2={np.median(R):.3f} ({np.percentile(R,2.5):.3f}-{np.percentile(R,97.5):.3f})")
# which pooled types differ among classes
print(S.groupby(d.Baltimore_group).median().round(3).to_string())
