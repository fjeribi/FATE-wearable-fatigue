"""MEFAR time-of-day tests for all models: conventional (ladder, participant-held-out, fixed settings) + saved deep OOF."""
import glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, mannwhitneyu
from common import *
R=MEFAR_RESULTS+os.sep
d,F=load_mefar(); ch=pd.read_csv(R+'chalder_scores.csv'); ch['unit']=ch.subject_id+'_'+ch.session
frames=[]
for f in glob.glob('cache/mefar/held_out_participant__*.csv.gz'):
    name=os.path.basename(f).split('__')[1].replace('.csv.gz','').replace('_',' ')
    o=pd.read_csv(f).sort_values('idx'); dd=d.iloc[o.idx.values][['participant','session','unit','y']].copy()
    dd['fold']=o.fold.values; dd['score']=o.score.values; dd['model']=name; frames.append(dd)
deep=pd.read_csv(R+'p2/p2_deep_oof_scores.csv.gz').rename(columns={'subject_id':'participant','session_id':'unit','y_true':'y'})
deep['session']=deep.unit.str.split('_').str[-1]; deep['model']=deep.model.replace({'ANN':'ANN/MLP','1D_CNN':'1D-CNN'})+' (paper run)'
frames.append(deep[['participant','session','unit','y','fold','score','model']])
a=pd.concat(frames,ignore_index=True); a['y']=a.y.astype(int)
a['rank']=a.groupby(['model','fold']).score.rank(pct=True)
rows=[]; rng=np.random.default_rng(SEED)
for m,dd in a.groupby('model'):
    r={'model':m,'pooled':roc_auc_score(dd.y,dd['rank']),'session_level':unit_auc(dd)}
    for s in ['morning','evening']:
        e=dd[dd.session==s]; u=e.groupby('unit').agg(y=('y','first'),s=('rank','mean'))
        r[f'{s}_sec']=roc_auc_score(e.y,e['rank']); r[f'{s}_sess']=roc_auc_score(u.y,u.s)
        perm=[roc_auc_score(rng.permutation(u.y.values),u.s) for _ in range(5000)]
        r[f'{s}_sess_p']=(1+np.sum(np.array(perm)>=r[f'{s}_sess']))/5001
    u=dd.groupby(['participant','session']).agg(s=('rank','mean'),y=('y','first')).reset_index()
    w=u.pivot(index='participant',columns='session',values='s'); wy=u.pivot(index='participant',columns='session',values='y')
    c=ch.pivot(index='subject_id',columns='session',values='chalder_score').loc[w.index]
    dsc=w.evening-w.morning; dch=c.evening-c.morning; up=wy.evening>wy.morning; stable=wy.evening==wy.morning
    r['rise_up']=dsc[up].mean(); r['rise_stable']=dsc[stable].mean()
    r['p_up_vs_stable']=mannwhitneyu(dsc[up],dsc[stable],alternative='greater').pvalue
    r['rho_dscore_dchalder'],r['p_rho']=spearmanr(dsc,dch); r['share_rising']=(dsc>0).mean()
    rows.append(r)
res=pd.DataFrame(rows).sort_values('pooled',ascending=False); res.to_csv('mefar_confound_all.csv',index=False)
pd.set_option('display.width',250); print(res.round(3).to_string(index=False))
