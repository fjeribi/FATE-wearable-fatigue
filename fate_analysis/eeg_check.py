import os, glob, numpy as np, pandas as pd
from common import MEFAR_CSV, FATIGUESET_ROOT
m=pd.read_csv(MEFAR_CSV,usecols=['Delta','Theta','Alpha1','Alpha2','Beta1','Beta2','Gamma1','Gamma2'])
cm=m.corr(method='spearman'); off=cm.values[np.triu_indices(8,1)]
print('MEFAR mean inter-band rho',off.mean().round(3),'range',off.min().round(3),off.max().round(3))
print('MEFAR P(Delta>Gamma2)',(m.Delta>m.Gamma2).mean().round(3))
cm.round(3).to_csv('eeg_corr_mefar.csv')
bands=['delta','theta','alpha','beta','gamma']; rows=[]
for s in sorted(glob.glob(os.path.join(FATIGUESET_ROOT,'*','*'))):
    parts=[]
    for b in bands:
        fs=[f for f in [f'{s}/forehead_eeg_{b}_abs.csv',f'{s}/forehead_eeg_{b}_abs2.csv'] if glob.os.path.exists(f)]
        x=pd.concat([pd.read_csv(f).loc[:,lambda z:[c for c in z.columns if not c.startswith('Unnamed')]] for f in fs]).dropna(subset=['timestamp'])
        x=x.assign(v=x[['TP9','AF7','AF8','TP10']].replace([np.inf,-np.inf],np.nan).mean(axis=1))
        x['sec']=(x.timestamp//1000).astype('int64'); parts.append(x.groupby('sec').v.mean().rename(b))
    rows.append(pd.concat(parts,axis=1).dropna())
fsd=pd.concat(rows)
cf=fsd.corr(method='spearman'); off=cf.values[np.triu_indices(5,1)]
print('FatigueSet mean inter-band rho',off.mean().round(3),'range',off.min().round(3),off.max().round(3), 'n seconds',len(fsd))
print('FatigueSet P(delta>gamma)',(fsd.delta>fsd.gamma).mean().round(3))
print('FatigueSet median log power by band',fsd.median().round(3).to_dict())
print('MEFAR median power by band',m.median().round(0).to_dict())
cf.round(3).to_csv('eeg_corr_fatigueset.csv')
