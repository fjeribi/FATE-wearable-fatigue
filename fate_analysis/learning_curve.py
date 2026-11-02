import sys, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from common import *
ds=sys.argv[1]; d,F=load_mefar() if ds=='mefar' else load_fs()
ps=np.array(sorted(d.participant.unique())); rng=np.random.default_rng(SEED)
n_test=5 if ds=='mefar' else 3; sizes=[4,8,12,16,18] if ds=='mefar' else [3,5,7,9]
M=models(small=ds!='mefar'); rows=[]
for rep in range(20):
    perm=rng.permutation(ps); test=perm[:n_test]; pool=perm[n_test:]
    te=d.participant.isin(test).values
    if d.y[te].nunique()<2: continue
    for n in sizes:
        tr=d.participant.isin(pool[:n]).values
        if d.y[tr].nunique()<2: continue
        for name in ['Logistic Regression','XGBoost']:
            s=clone(M[name]).fit(d.loc[tr,F].values,d.y[tr].values).predict_proba(d.loc[te,F].values)[:,1]
            dd=d.loc[te,['y','unit']].assign(s=s); u=dd.groupby('unit').agg(y=('y','first'),s=('s','mean'))
            rows.append({'rep':rep,'n_train':n,'model':name,'auc':roc_auc_score(dd.y,s),
                         'unit_auc':roc_auc_score(u.y,u.s) if u.y.nunique()==2 else np.nan})
    print(ds,'rep',rep,flush=True)
r=pd.DataFrame(rows); r.to_csv(f'learning_curve_{ds}.csv',index=False)
print(r.groupby(['model','n_train'])[['auc','unit_auc']].agg(['mean','std']).round(3))
