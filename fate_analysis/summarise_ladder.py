import sys, glob, os, numpy as np, pandas as pd
from common import *
ds=sys.argv[1]
d,F=load_mefar() if ds=='mefar' else load_fs(norm=(ds=='fsnorm'))
rows=[]
for f in sorted(glob.glob(f'cache/{ds}/*.csv.gz')):
    rung,name=os.path.basename(f).replace('.csv.gz','').split('__'); name=name.replace('_',' ')
    o=pd.read_csv(f).sort_values('idx'); dd=d.iloc[o.idx.values].copy(); dd['fold']=o.fold.values; dd['score']=o.score.values
    auc,r=pooled_auc(dd.y.values,dd.score.values,dd.fold.values); dd['rank']=r
    row={'dataset':ds,'rung':rung,'model':name,'pooled_auc':auc,'unit_auc':unit_auc(dd)}
    if rung in('held_out_participant','loso'):
        lo,hi=boot_ci(dd,n=500); row.update(ci_low=lo,ci_high=hi,unit_perm_p=unit_perm_p(dd,n=2000))
    rows.append(row)
r=pd.DataFrame(rows); r.to_csv(f'ladder_{ds}.csv',index=False)
print(r.round(3).to_string())
