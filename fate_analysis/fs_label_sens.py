"""FatigueSet label-definition sensitivity."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut, KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from common import *
d0,F=load_fs()
blk=d0.groupby('block_id').agg(mf=('mental_fatigue','first'),p=('participant','first'))
t1,t2=blk.mf.quantile([1/3,2/3])
pm=blk.groupby('p').mf.mean()
defs={}
defs['median']=d0.assign(y=(d0.mental_fatigue>=blk.mf.median()).astype(int))
e=d0[(d0.mental_fatigue<=t1)|(d0.mental_fatigue>=t2)].copy(); e['y']=(e.mental_fatigue>=t2).astype(int); defs['tertile_extremes']=e.reset_index(drop=True)
w=d0.copy(); w['y']=(w.mental_fatigue>w.participant.map(pm)).astype(int)
w=w[~w.participant.isin(pm[pm==0].index)].reset_index(drop=True); defs['within_person']=w     # P07 reported 0 throughout
M=models(small=True); rows=[]
def run(d,rung):
    X=d[F].values; y=d.y.values
    if rung=='random': folds=StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y)
    elif rung=='block': folds=StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,d.block_id)
    elif rung=='session': folds=StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,d.session)
    else: folds=LeaveOneGroupOut().split(X,y,d.participant)
    return list(folds)
for lab,d in defs.items():
    for rung in ['random','block','session','loso']:
        folds=run(d,rung)
        for name in ['Logistic Regression','Random Forest','XGBoost']:
            s=np.zeros(len(d)); fo=np.zeros(len(d))
            for k,(tr,te) in enumerate(folds):
                s[te]=clone(M[name]).fit(d[F].values[tr],d.y.values[tr]).predict_proba(d[F].values[te])[:,1]; fo[te]=k
            auc,r=pooled_auc(d.y.values,s,fo); dd=d.assign(rank=r)
            row={'label':lab,'rung':rung,'model':name,'pooled_auc':auc,'unit_auc':unit_auc(dd),'n_units':dd.unit.nunique(),'pos_rate':d.y.mean()}
            if rung=='loso':
                row['perm_p']=unit_perm_p(dd,n=2000)
                u=dd.groupby('unit').agg(s=('rank','mean'),mf=('mental_fatigue','first'),block=('block','first'),session=('session','first'),participant=('participant','first'))
                u['mf_z']=(u.mf-u.mf.mean())/u.mf.std()
                m=smf.ols('s ~ C(block) + mf_z + C(session)',u).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(u.participant)[0]})
                row['beta_mental_ctrl_block']=m.params['mf_z']; row['p_beta']=m.pvalues['mf_z']
            rows.append(row); print(lab,rung,name,round(auc,3),flush=True)
# continuous target (regression), block-level Spearman
d=d0; X=d[F].values; yv=d.mental_fatigue.values
R={'Ridge':sc(Ridge(alpha=1.0)),'RF regressor':imp(RandomForestRegressor(300,min_samples_leaf=5,n_jobs=2,random_state=SEED)),
   'XGB regressor':XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,n_jobs=2,random_state=SEED)}
for rung in ['random','block','session','loso']:
    if rung=='random': folds=KFold(5,shuffle=True,random_state=SEED).split(X)
    elif rung=='block': folds=GroupKFold(5).split(X,yv,d.block_id)
    elif rung=='session': folds=GroupKFold(5).split(X,yv,d.session)
    else: folds=LeaveOneGroupOut().split(X,yv,d.participant)
    folds=list(folds)
    for name,m in R.items():
        p=np.zeros(len(d))
        for tr,te in folds: p[te]=clone(m).fit(X[tr],yv[tr]).predict(X[te])
        u=d.assign(p=p).groupby('unit').agg(p=('p','mean'),mf=('mental_fatigue','first'),block=('block','first'),session=('session','first'),participant=('participant','first'))
        rho,pv=spearmanr(u.p,u.mf); row={'label':'continuous','rung':rung,'model':name,'spearman_block_level':rho,'p_spearman':pv}
        if rung=='loso':
            u['mf_z']=(u.mf-u.mf.mean())/u.mf.std(); u['pz']=(u.p-u.p.mean())/u.p.std()
            m=smf.ols('pz ~ C(block) + mf_z + C(session)',u).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(u.participant)[0]})
            row['beta_mental_ctrl_block']=m.params['mf_z']; row['p_beta']=m.pvalues['mf_z']
        rows.append(row); print('continuous',rung,name,round(rho,3),flush=True)
res=pd.DataFrame(rows); res.to_csv('fs_label_sensitivity.csv',index=False)
pd.set_option('display.width',250); print(res.round(3).to_string())
