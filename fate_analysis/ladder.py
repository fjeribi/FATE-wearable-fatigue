"""Leakage ladder: same data, same models, progressively stricter partitioning."""
import sys, os, time, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut
from common import *
ds=sys.argv[1]; only=sys.argv[2:] or None
if ds=='mefar': d,F=load_mefar(); RUNGS=[('random_sample',None),('random_60s_chunk','chunk'),('held_out_session','unit'),('held_out_participant','participant')]
elif ds=='fs': d,F=load_fs(); RUNGS=[('random_sample',None),('held_out_block','chunk'),('held_out_session','session'),('held_out_participant','participant'),('loso','participant')]
elif ds=='fsnorm': d,F=load_fs(norm=True); RUNGS=[('held_out_participant','participant'),('loso','participant')]
X=d[F].values; y=d.y.values
os.makedirs(f'cache/{ds}',exist_ok=True)
M=models(small=ds!='mefar')
for rung,g in RUNGS:
    if rung=='random_sample': folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    elif rung=='loso': folds=list(LeaveOneGroupOut().split(X,y,d[g]))
    else: folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,d[g]))
    if rung in('held_out_participant','loso'):
        for tr,te in folds: assert not set(d.participant.iloc[tr])&set(d.participant.iloc[te])
    for name,m in M.items():
        if only and name not in only: continue
        f=f'cache/{ds}/{rung}__{name.replace(" ","_")}.csv.gz'
        if os.path.exists(f): continue
        t0=time.time(); parts=[]
        for k,(tr,te) in enumerate(folds):
            mm=clone(m).fit(X[tr],y[tr]); s=mm.predict_proba(X[te])[:,1]
            parts.append(pd.DataFrame({'idx':te,'fold':k,'score':s}))
        o=pd.concat(parts); o.to_csv(f,index=False)
        print(ds,rung,name,f'{time.time()-t0:.0f}s',flush=True)
