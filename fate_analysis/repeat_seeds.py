import sys, os, time, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from common import *
ds=sys.argv[1]; seeds=[1,2,3,4,5,6,7,8,9] if ds=='fs' else [1,2,3,4,5]
d,F=load_mefar() if ds=='mefar' else load_fs()
X=d[F].values; y=d.y.values
rungs=[('held_out_session','unit' if ds=='mefar' else 'session'),('held_out_participant','participant')]
if ds=='fs': rungs=[('held_out_block','chunk')]+rungs
M=models(small=ds!='mefar'); M['Random Forest'].set_params(m__n_estimators=100 if ds=='mefar' else 300)
os.makedirs(f'cache/seeds_{ds}',exist_ok=True); rows=[]
for sd in seeds:
    for rung,g in rungs:
        folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=sd).split(X,y,d[g]))
        for name,m in M.items():
            f=f'cache/seeds_{ds}/{rung}__{name.replace(" ","_")}__s{sd}.csv'
            if os.path.exists(f): continue
            parts=[]
            for k,(tr,te) in enumerate(folds):
                s=clone(m).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]; parts.append(pd.DataFrame({'idx':te,'fold':k,'score':s}))
            o=pd.concat(parts).sort_values('idx'); dd=d.iloc[o.idx.values].copy()
            auc,r=pooled_auc(dd.y.values,o.score.values,o.fold.values); dd['rank']=r
            pd.DataFrame([{'seed':sd,'rung':rung,'model':name,'pooled_auc':auc,'unit_auc':unit_auc(dd)}]).to_csv(f,index=False)
            print(ds,sd,rung,name,round(auc,3),flush=True)
