"""Nested hyperparameter tuning under participant-grouped CV (outer 5-fold, inner 3-fold, both grouped by participant)."""
import sys, os, numpy as np, pandas as pd, itertools
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from common import *
ds=sys.argv[1]
d,F=load_mefar() if ds=='mefar' else load_fs()
X=d[F].values; y=d.y.values; g=d.participant.values
GRID={'Logistic Regression':('m__C',[0.01,0.1,1,10]),
      'Random Forest':('m__min_samples_leaf',[5,50,500] if ds=='mefar' else [1,5,20]),
      'XGBoost':('max_depth',[2,4,6]),
      'KNN':('m__n_neighbors',[5,51,501] if ds=='mefar' else [5,25,75])}
M=models(small=ds!='mefar'); M['Random Forest'].set_params(m__n_estimators=200)
outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,g))
os.makedirs('cache/nested',exist_ok=True)
rows=[]
for name,(p,vals) in GRID.items():
    f=f'cache/nested/{ds}__{name.replace(" ","_")}.csv.gz'
    if os.path.exists(f): continue
    parts=[]; chosen=[]
    for k,(tr,te) in enumerate(outer):
        sub=tr[::5] if ds=='mefar' else tr      # inner search on every 5th second (autocorrelated 1-Hz data)
        inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(X[sub],y[sub],g[sub]))
        best,bv=None,-1
        for v in vals:
            sc_=[]
            for itr,ite in inner:
                mm=clone(M[name]).set_params(**{p:v}).fit(X[sub][itr],y[sub][itr])
                sc_.append(roc_auc_score(y[sub][ite],mm.predict_proba(X[sub][ite])[:,1]))
            if np.mean(sc_)>bv: best,bv=v,np.mean(sc_)
        mm=clone(M[name]).set_params(**{p:best}).fit(X[tr],y[tr])
        parts.append(pd.DataFrame({'idx':te,'fold':k,'score':mm.predict_proba(X[te])[:,1]})); chosen.append(best)
        print(ds,name,'fold',k,'chosen',best,f'inner AUC {bv:.3f}',flush=True)
    pd.concat(parts).to_csv(f,index=False)
    pd.DataFrame({'fold':range(5),'chosen':chosen}).to_csv(f.replace('.csv.gz','_chosen.csv'),index=False)
