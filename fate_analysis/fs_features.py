"""FatigueSet: 30-s window features inside the three cognitive test blocks of each session.

Block b (b = 0, 1, 2) runs from the b-th 'start_crt' marker to the next of
start_activity / start_fatigue / end_session, i.e. the CRT + n-back tests that
follow each fatigue survey. Every window in block b is labelled with the
mental-fatigue score reported in survey b (0-100 VAS). Same seated task in all
blocks; only fatigue and time-on-session differ.
"""
import os, glob, warnings
import numpy as np, pandas as pd
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')
from common import FATIGUESET_ROOT as ROOT, FS_WINDOWS
WIN, STEP = 30, 15
BANDS=['delta','theta','alpha','beta','gamma']

def load(folder, name):
    parts=[]
    for f in [f'{name}.csv', f'{name}2.csv']:
        p=os.path.join(folder,f)
        if os.path.exists(p):
            d=pd.read_csv(p)
            d=d.loc[:,[c for c in d.columns if not c.startswith('Unnamed')]]
            if len(d): parts.append(d)
    if not parts: return None
    d=pd.concat(parts).dropna(subset=['timestamp']).sort_values('timestamp')
    d['t']=d.timestamp/1000.0
    return d

def slope(t,v):
    ok=np.isfinite(v)
    return np.polyfit(t[ok],v[ok],1)[0] if ok.sum()>3 and np.ptp(t[ok])>0 else np.nan

def win_feats(S,t0,t1):
    f={}
    def seg(k):
        d=S[k]
        return None if d is None else d[(d.t>=t0)&(d.t<t1)]
    h=seg('wrist_hr')
    if h is not None and len(h)>=10:
        f.update(hr_mean=h.hr.mean(),hr_sd=h.hr.std(),hr_slope=slope(h.t.values,h.hr.values))
    i=seg('wrist_ibi')
    if i is not None and len(i)>=8:
        x=i.duration.values; x=x[(x>333)&(x<1500)]
        if len(x)>=8: f.update(ibi_mean=x.mean(),sdnn=x.std(ddof=1),rmssd=np.sqrt(np.mean(np.diff(x)**2)))
    e=seg('wrist_eda')
    if e is not None and len(e)>=40:
        v=e.eda.values; pk,_=find_peaks(v,prominence=0.01,distance=4)
        f.update(eda_mean=v.mean(),eda_sd=v.std(),eda_slope=slope(e.t.values,v),scr_per_min=len(pk)/(WIN/60))
    tp=seg('wrist_skin_temperature')
    if tp is not None and len(tp)>=40:
        f.update(temp_mean=tp.temp.mean(),temp_slope=slope(tp.t.values,tp.temp.values))
    a=seg('wrist_acc')
    if a is not None and len(a)>=300:
        m=np.sqrt(a.ax**2+a.ay**2+a.az**2); f.update(acc_mean=m.mean(),acc_sd=m.std())
    bp={}
    for b in BANDS:
        d=seg(f'forehead_eeg_{b}_abs')
        if d is not None and len(d)>=50:
            v=d[['TP9','AF7','AF8','TP10']].replace([np.inf,-np.inf],np.nan).mean(axis=1)
            bp[b]=v.mean()
    if len(bp)==5:
        for b in BANDS: f[f'eeg_{b}']=bp[b]
        f['eeg_theta_alpha']=bp['theta']-bp['alpha']        # log-power difference
        f['eeg_ta_beta']=np.log10((10**bp['theta']+10**bp['alpha'])/10**bp['beta'])
    return f

rows=[]
for sess in sorted(glob.glob(f'{ROOT}/[0-9][0-9]/[0-9][0-9]')):
    pid,sid=sess.split('/')[-2:]
    mk=pd.read_csv(f'{sess}/exp_markers.csv'); fat=pd.read_csv(f'{sess}/exp_fatigue.csv')
    mk['t']=mk.utcTime/1000
    starts=mk[mk.eventMarker=='start_crt'].t.values
    ends=[mk[mk.eventMarker==e].t.values[0] for e in ['start_activity','start_fatigue','end_session']]
    S={k:load(sess,k) for k in ['wrist_hr','wrist_ibi','wrist_eda','wrist_skin_temperature','wrist_acc']+[f'forehead_eeg_{b}_abs' for b in BANDS]}
    for b in range(3):
        t0,t1=starts[b],ends[b]
        for w0 in np.arange(t0,t1-WIN+1e-9,STEP):
            f=win_feats(S,w0,w0+WIN)
            f.update(participant=f'P{pid}',session=f'P{pid}_S{sid}',block=b,block_id=f'P{pid}_S{sid}_B{b}',
                     t_rel=w0-t0,mental_fatigue=fat.mentalFatigueScore[b],physical_fatigue=fat.physicalFatigueScore[b])
            rows.append(f)
    print(pid,sid,[round(ends[b]-starts[b]) for b in range(3)],flush=True)
df=pd.DataFrame(rows)
df.to_csv(FS_WINDOWS,index=False)
print(df.shape); print(df.isna().mean().round(3).to_string())
