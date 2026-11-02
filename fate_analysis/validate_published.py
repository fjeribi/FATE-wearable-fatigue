"""Validate re-implementations on the public MEFAR_DOWN release with the published protocol (stratified random 80:20)."""
import numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from published import *
from common import MEFAR_DOWN as P
d=pd.read_csv(P); d.columns=d.columns.str.strip()
d=d.replace([np.inf,-np.inf],np.nan).dropna().drop_duplicates().reset_index(drop=True)
F=[c for c in d.columns if c!='class']; X=d[F].values.astype(float); y=d['class'].astype(int).values
print(d.shape, F, y.mean())
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=SEED)
rows=[]
M=jampala_models(scale=False); M["Nooh GCA (approx.)"]=NoohGCA()
for n,m in M.items():
    t=time.time(); m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    rows.append(dict(model=n,accuracy=accuracy_score(yte,p>=.5),f1=f1_score(yte,p>=.5),auc=roc_auc_score(yte,p),sec=round(time.time()-t),n_features=int(m.mask_.sum()) if hasattr(m,'mask_') else len(F)))
    print(rows[-1],flush=True)
    if n.startswith('Nooh'): print('features kept',np.array(F)[m.mask_])
pd.DataFrame(rows).to_csv('published_validation.csv',index=False)
