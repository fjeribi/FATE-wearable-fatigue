"""Run one MEFAR GCA job of run_published.py (same model, folds and cache file) so that ladder levels can run in
parallel processes. Usage: python run_gca_one.py <rung>
rung: random_sample | random_60s_chunk | held_out_session | held_out_participant | proxy"""
import sys, os, time, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from common import *
from published import NoohGCA

rung = sys.argv[1]
dm, Fm = load_mefar(); Xm = dm[Fm].values; ym = dm.y.values
G = NoohGCA(max_train=22000)
grp = {'random_sample': None, 'random_60s_chunk': 'chunk', 'held_out_session': 'unit', 'held_out_participant': 'participant'}
if rung == 'proxy':
    tag, r, y = 'proxy_mefar', 'held_out_participant', (dm.session == 'evening').astype(int).values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xm, y, dm['participant']))
else:
    tag, r, y = 'mefar', rung, ym
    folds = (list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xm, y)) if grp[rung] is None
             else list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xm, y, dm[grp[rung]])))
f = f'cache/{tag}/{r}__Nooh_GCA.csv.gz'
if os.path.exists(f): print('exists', f); sys.exit()
os.makedirs(os.path.dirname(f), exist_ok=True); t0 = time.time(); parts = []
MAXF = int(os.environ.get('MAX_FOLDS', '99')); newf = 0
ONLY = [int(k) for k in os.environ.get('ONLY_FOLDS', '').split(',') if k]   # optional subset, for parallel calls   # folds to compute in this call (keeps each call short)
for k, (tr, te) in enumerate(folds):
    pf = f'cache/{tag}/{r}__Nooh_GCA.fold{k}.csv.gz'          # per-fold checkpoint (resumable)
    if os.path.exists(pf): parts.append(pd.read_csv(pf)); continue
    if ONLY and k not in ONLY: continue
    if newf >= MAXF: print(tag, r, 'paused after', newf, 'new folds', flush=True); sys.exit()
    newf += 1
    s = clone(G).fit(Xm[tr], y[tr]).predict_proba(Xm[te])[:, 1]
    p = pd.DataFrame({'idx': te, 'fold': k, 'score': s}); p.to_csv(pf, index=False); parts.append(p)
    print(tag, r, 'fold', k, f'{time.time()-t0:.0f}s', flush=True)
if len(parts) < len(folds): print(tag, r, 'partial:', len(parts), 'of', len(folds), 'folds', flush=True); sys.exit()
pd.concat(parts).to_csv(f, index=False)
for k in range(len(folds)): os.remove(f'cache/{tag}/{r}__Nooh_GCA.fold{k}.csv.gz')
print(tag, r, 'Nooh GCA', f'{time.time()-t0:.0f}s', flush=True)
