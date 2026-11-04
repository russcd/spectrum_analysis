#!/usr/bin/env python3
"""Phylogenetic non-independence checks for the Baltimore-class PERMANOVA.
(1) one virus per family, resampled; (2) family-level (block) permutation of Baltimore labels.
Uses the same CLR/Aitchison distance and pseudocount rule as permanova.py."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/mnt/user-data/uploads')
import permanova as P
from family_map import family

rng = np.random.default_rng(1)
d = pd.read_csv('/mnt/user-data/uploads/merged_dataset.tsv', sep='\t', keep_default_na=False, na_values=[''])
ann = pd.read_csv('/mnt/user-data/uploads/virus_annotations.tsv', sep='\t')
d = d.merge(ann, on='virus_group'); d['family'] = d.virus_group.map(family)
d['Baltimore_group'] = d.Baltimore_group.fillna('NA').astype(str)
mut = P.find_mutation_columns(d.columns)
d = d[d.Baltimore_group != 'VII'].reset_index(drop=True)          # as in main analysis

def gower_from(df):
    Xc = P.clr(df[mut].to_numpy(float), None)
    G_ = Xc @ Xc.T; sq = np.diag(G_)
    return P.gower(np.maximum(sq[:,None]+sq[None,:]-2*G_, 0))

def one_way(G, lab):                      # R2, pseudo-F for a single factor
    codes, uniq = pd.factorize(lab); k = len(uniq); N = len(lab)
    Z = np.eye(k)[codes]; n = Z.sum(0)
    ssb = np.sum(np.einsum('ig,ij,jg->g', Z, G, Z) / n)
    sst = np.trace(G); ssw = sst-ssb
    return ssb/sst, (ssb/(k-1))/(ssw/(N-k))

def perm_p(G, lab, nperm, labels_fn=None):
    r2o, Fo = one_way(G, lab); ge = 1
    for _ in range(nperm):
        l = labels_fn() if labels_fn else rng.permutation(lab)
        if one_way(G, l)[1] >= Fo: ge += 1
    return r2o, Fo, ge/(nperm+1)

def seq_r2(G, df, terms):                 # sequential R2 per term, no permutations
    cum, prev, out = [], 0.0, {}
    for t in terms:
        cum.append(P.dummies(df[t])); ss,_ = P.hat_trace(cum, G)
        out[t] = (ss-prev)/np.trace(G); prev = ss
    return out

# sanity check against permanova.py
G = gower_from(d)
print("FULL DATA  N=%d  Baltimore R2=%.4f F=%.2f" % ((len(d),)+one_way(G, d.Baltimore_group.values)))

# ---- (2) family-level permutation of Baltimore labels (whole families carry one label)
fam2bal = d.groupby('family').Baltimore_group.first()
fam_idx = d.family.values
def fam_perm():
    shuf = pd.Series(rng.permutation(fam2bal.values), index=fam2bal.index)
    return shuf.loc[fam_idx].values
r2o, Fo, p = perm_p(G, d.Baltimore_group.values, 9999, fam_perm)
null = []
for _ in range(2000): null.append(one_way(G, fam_perm())[0])
print("\n(2) FAMILY-LEVEL PERMUTATION  (%d families)" % len(fam2bal))
print("  observed R2=%.3f ; null R2 median=%.3f, 95th=%.3f, 99th=%.3f, max=%.3f ; p=%.4g (9999 perms)"
      % (r2o, np.median(null), np.percentile(null,95), np.percentile(null,99), max(null), p))

# ---- (1) one virus per family, resampled
def one_per_family(df, groups_keep):
    s = df.groupby('family', group_keys=False).sample(1, random_state=int(rng.integers(1e9)))
    return s[s.Baltimore_group.isin(groups_keep)].reset_index(drop=True)

for label, keepers in [("all groups except VII (as main analysis; groups with <2 families dropped)", None),
                       ("classes I-V only (each >=3 families)", ['I','II','III','IV','V'])]:
    res = []
    for b in range(1000):
        s = one_per_family(d, keepers or d.Baltimore_group.unique())
        vc = s.Baltimore_group.value_counts(); s = s[s.Baltimore_group.isin(vc[vc>=2].index)].reset_index(drop=True)
        Gs = gower_from(s)
        r2, F, p = perm_p(Gs, s.Baltimore_group.values, 499)
        sq = seq_r2(Gs, s, ['Baltimore_group','enveloped','host'])
        res.append((len(s), s.Baltimore_group.nunique(), r2, F, p, sq['enveloped'], sq['host']))
    r = pd.DataFrame(res, columns=['n','k','R2','F','p','env','host'])
    print("\n(1) ONE PER FAMILY, 1000 draws: %s" % label)
    print("  n viruses per draw: %d-%d ; groups: %s" % (r.n.min(), r.n.max(), sorted(set(r.k))))
    print("  Baltimore R2: median %.3f [2.5-97.5%%: %.3f-%.3f]" % (r.R2.median(), *np.percentile(r.R2,[2.5,97.5])))
    print("  pseudo-F median %.2f ; p median %.4g ; frac draws p<0.05: %.3f ; p<0.001: %.3f" % (r.F.median(), r.p.median(), (r.p<.05).mean(), (r.p<.001).mean()))
    print("  envelope R2 median %.3f ; host R2 median %.3f (sequential, after Baltimore)" % (r.env.median(), r.host.median()))
print("\nfamilies per class:\n", d.groupby('Baltimore_group').family.nunique().to_string())
