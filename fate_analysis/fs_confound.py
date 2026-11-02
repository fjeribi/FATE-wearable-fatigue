"""FatigueSet confound tests on participant-held-out (LOSO) predictions."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from common import *
d,F=load_fs()
if os.path.exists(os.path.join(FATIGUESET_ROOT,'metadata.csv')):   # activity intensity of each session (descriptive only)
    meta=pd.read_csv(os.path.join(FATIGUESET_ROOT,'metadata.csv'),dtype=str); inten={}
    for _,r in meta.iterrows():
        for lvl in ['low','medium','high']: inten[f"P{r.participant_id}_S{r[lvl+'_session']}"]=lvl
    d['intensity']=d.session.map(inten)
out=[]
for name in ['Random Forest','XGBoost','Logistic Regression']:
    o=pd.read_csv(f'cache/fs/loso__{name.replace(" ","_")}.csv.gz').sort_values('idx')
    dd=d.iloc[o.idx.values].copy(); dd['fold']=o.fold.values
    dd['rank']=pd.Series(o.score.values).groupby(o.fold.values).rank(pct=True).values
    r={'model':name,'pooled':roc_auc_score(dd.y,dd['rank'])}
    for b in range(3):
        e=dd[dd.block==b]; r[f'auc_block{b}']=roc_auc_score(e.y,e['rank']) if e.y.nunique()==2 else np.nan
        r[f'n_high_block{b}']=int(e.groupby('unit').y.first().sum())
    # block-level table
    u=dd.groupby('unit').agg(s=('rank','mean'),mf=('mental_fatigue','first'),pf=('physical_fatigue','first'),
        block=('block','first'),session=('session','first'),participant=('participant','first')).reset_index()
    u['mf_z']=(u.mf-u.mf.mean())/u.mf.std(); u['pf_z']=(u.pf-u.pf.mean())/u.pf.std()
    m=smf.ols('s ~ C(block) + mf_z + C(session)',u).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(u.participant)[0]})
    r['beta_mental_within_session']=m.params['mf_z']; r['p_mental']=m.pvalues['mf_z']
    m0=smf.ols('s ~ mf_z + C(session)',u).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(u.participant)[0]})
    r['beta_mental_no_block_ctrl']=m0.params['mf_z']; r['p_mental_no_block']=m0.pvalues['mf_z']
    mb=smf.ols('s ~ C(block) + C(session)',u).fit()
    r['block2_minus_block0']=mb.params.get('C(block)[T.2]',np.nan); r['p_block2']=mb.pvalues.get('C(block)[T.2]',np.nan)
    r['block1_minus_block0']=mb.params.get('C(block)[T.1]',np.nan); r['p_block1']=mb.pvalues.get('C(block)[T.1]',np.nan)
    rho,p=spearmanr(u.s,u.pf); r['rho_physical']=rho; r['p_rho_physical']=p
    rho,p=spearmanr(u.s,u.mf); r['rho_mental']=rho; r['p_rho_mental']=p
    out.append(r)
res=pd.DataFrame(out); print(res.T.round(3).to_string()); res.to_csv('fs_confound.csv',index=False)
# Block proxy: predict late (block 2) vs early (block 0) from the same features, LOSO
e=d[d.block.isin([0,2])].reset_index(drop=True); yb=(e.block==2).astype(int).values; X=e[F].values
rows=[]
for name in ['Random Forest','XGBoost','Logistic Regression']:
    s=np.zeros(len(e)); fo=np.zeros(len(e))
    for k,(tr,te) in enumerate(LeaveOneGroupOut().split(X,yb,e.participant)):
        s[te]=clone(models(True)[name]).fit(X[tr],yb[tr]).predict_proba(X[te])[:,1]; fo[te]=k
    r_=pd.Series(s).groupby(fo).rank(pct=True).values
    rows.append({'model':name,'block_proxy_auc':roc_auc_score(yb,r_)})
bp=pd.DataFrame(rows); print(bp.round(3)); bp.to_csv('fs_block_proxy.csv',index=False)
