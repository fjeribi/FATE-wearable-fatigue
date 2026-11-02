"""Evaluate re-implemented published pipelines under the FATE partition ladder, and run the
proxy (confound) tasks for every model. OOF scores cached exactly as in ladder.py."""
import sys, os, time, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
from common import *
from published import jampala_models, NoohGCA

def folds_for(rung, X, y, d, g):
    if rung == 'random_sample': return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    if rung == 'loso': return list(LeaveOneGroupOut().split(X, y, d[g]))
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, d[g]))

def run(tag, rung, name, m, X, y, folds):
    f = f'cache/{tag}/{rung}__{name.replace(" ", "_")}.csv.gz'
    if os.path.exists(f): return
    os.makedirs(os.path.dirname(f), exist_ok=True); t0 = time.time(); parts = []
    for k, (tr, te) in enumerate(folds):
        s = clone(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        parts.append(pd.DataFrame({'idx': te, 'fold': k, 'score': s}))
    pd.concat(parts).to_csv(f, index=False); print(tag, rung, name, f'{time.time()-t0:.0f}s', flush=True)

dm, Fm = load_mefar(); Xm = dm[Fm].values; ym = dm.y.values
df, Ff = load_fs(); Xf = df[Ff].values; yf = df.y.values
MR = [('random_sample', None), ('random_60s_chunk', 'chunk'), ('held_out_session', 'unit'), ('held_out_participant', 'participant')]
FR = [('random_sample', None), ('held_out_block', 'chunk'), ('held_out_session', 'session'), ('held_out_participant', 'participant'), ('loso', 'participant')]
J = jampala_models(scale=True); G = NoohGCA(max_train=22000); Gf = NoohGCA()
# proxy tasks: MEFAR session (evening vs morning); FatigueSet phase (last vs first block)
ms = (dm.session == 'evening').astype(int).values
fe = df[df.block.isin([0, 2])].reset_index(drop=True); fb = (fe.block == 2).astype(int).values; Xfe = fe[Ff].values
mine_m = models(False); mine_f = models(True)
jobs = []
for r, g in FR:                                   # FatigueSet: small, all models
    for n, m in {**J, 'Nooh GCA': Gf}.items(): jobs.append(('fs', r, n, m, Xf, yf, folds_for(r, Xf, yf, df, g)))
pf = list(LeaveOneGroupOut().split(Xfe, fb, fe.participant))
for n, m in {**mine_f, **J, 'Nooh GCA': Gf}.items(): jobs.append(('proxy_fs', 'loso', n, m, Xfe, fb, pf))
for r, g in MR:
    for n, m in J.items(): jobs.append(('mefar', r, n, m, Xm, ym, folds_for(r, Xm, ym, dm, g)))
pm = folds_for('held_out_participant', Xm, ms, dm, 'participant')
for n, m in {**mine_m, **J}.items(): jobs.append(('proxy_mefar', 'held_out_participant', n, m, Xm, ms, pm))
for r, g in [MR[0], MR[3], MR[2], MR[1]]:        # GCA on MEFAR last (slow)
    jobs.append(('mefar', r, 'Nooh GCA', G, Xm, ym, folds_for(r, Xm, ym, dm, g)))
    if r == 'held_out_participant': jobs.append(('proxy_mefar', 'held_out_participant', 'Nooh GCA', G, Xm, ms, pm))
for j in jobs: run(*j)
print('DONE', flush=True)
