"""FATE indices for every model on both datasets.
LII  = pooled ROC-AUC (level 1, random samples) - pooled ROC-AUC (level 4, held-out participants)
CDI  = proxy ROC-AUC - fatigue ROC-AUC, both participant-independent
       (MEFAR: evening vs morning session, five-fold held-out participants; FatigueSet: last vs first block, LOSO)
TS-AUC = mean of unit-level ROC-AUCs computed within strata of the proxy
       (MEFAR: morning and evening sessions; FatigueSet: block positions 0, 1, 2), participant-independent.
Also reports participant-bootstrap CI and unit-level permutation p for level 4 / LOSO."""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from common import *
NAMES = ['Logistic Regression','Gaussian Naive Bayes','KNN','Random Forest','XGBoost','MLP','Jampala RF','Jampala LR','Jampala KNN','Nooh GCA']
def load(tag, rung, name, d):
    f = f'cache/{tag}/{rung}__{name.replace(" ","_")}.csv.gz'
    if not os.path.exists(f): return None
    o = pd.read_csv(f).sort_values('idx'); dd = d.iloc[o.idx.values].copy(); dd['fold'] = o.fold.values
    dd['score'] = o.score.values; dd['rank'] = pd.Series(o.score.values).groupby(o.fold.values).rank(pct=True).values; return dd
def ts_auc(dd, strat, n=2000, seed=SEED):
    """Mean within-stratum unit-level AUC and a permutation p that shuffles labels within strata."""
    U = [e.groupby('unit').agg(y=('y','first'), s=('rank','mean')) for _, e in dd.groupby(strat)]
    U = [u for u in U if u.y.nunique() == 2]
    a = [roc_auc_score(u.y, u.s) for u in U]; obs = np.mean(a); rng = np.random.default_rng(seed)
    perm = [np.mean([roc_auc_score(rng.permutation(u.y.values), u.s) for u in U]) for _ in range(n)]
    return obs, a, (1 + np.sum(np.array(perm) >= obs)) / (1 + n)
rows = []
dm, _ = load_mefar(); dm['proxy'] = (dm.session == 'evening').astype(int)
df, _ = load_fs(); fe = df[df.block.isin([0, 2])].reset_index(drop=True); fe['y'] = (fe.block == 2).astype(int)
for ds, d, l1, l4, main, strat, ptag, prung, pd_ in [
        ('MEFAR', dm, 'random_sample', 'held_out_participant', 'held_out_participant', 'session', 'proxy_mefar', 'held_out_participant', dm.assign(y=dm.proxy)),
        ('FatigueSet', df, 'random_sample', 'held_out_participant', 'loso', 'block', 'proxy_fs', 'loso', fe)]:
    tag = 'mefar' if ds == 'MEFAR' else 'fs'
    for n in NAMES:
        a1, a4, am, pr = [load(t, r, n, x) for t, r, x in [(tag, l1, d), (tag, l4, d), (tag, main, d), (ptag, prung, pd_)]]
        r = {'dataset': ds, 'model': n}
        if a1 is not None: r['auc_l1'] = roc_auc_score(a1.y, a1['rank']); r['unit_l1'] = unit_auc(a1); r['acc_l1'] = np.mean((a1.score >= .5) == a1.y)
        if a4 is not None: r['auc_l4'] = roc_auc_score(a4.y, a4['rank']); r['acc_l4'] = np.mean((a4.score >= .5) == a4.y); r['bacc_l4'] = balanced_accuracy_score(a4.y, a4.score >= .5)
        if am is not None:
            r['auc_main'] = roc_auc_score(am.y, am['rank']); r['unit_main'] = unit_auc(am)
            r['ci_lo'], r['ci_hi'] = boot_ci(am, n=500); r['perm_p'] = unit_perm_p(am, n=2000)
            r['ts_auc'], parts, r['ts_p'] = ts_auc(am, strat); r['ts_parts'] = ' / '.join(f'{x:.3f}' for x in parts)
        if pr is not None: r['proxy_auc'] = roc_auc_score(pr.y, pr['rank'])
        if 'auc_l1' in r and 'auc_l4' in r: r['LII'] = r['auc_l1'] - r['auc_l4']
        if 'proxy_auc' in r and 'auc_main' in r: r['CDI'] = r['proxy_auc'] - r['auc_main']
        rows.append(r)
res = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
for ds in res.dataset.unique():
    k = res.dataset == ds
    res.loc[k, 'perm_p_holm'] = multipletests(res.loc[k, 'perm_p'], method='holm')[1]
    res.loc[k, 'ts_p_holm'] = multipletests(res.loc[k, 'ts_p'], method='holm')[1]
def tier(r):
    if r.perm_p_holm >= 0.05: return 1                       # no participant-independent evidence
    if r.ts_p_holm >= 0.05 or r.CDI > 0: return 2             # generalises, but attributable to the proxy
    return 3                                                  # fatigue-attributable evidence
res['tier'] = res.apply(tier, axis=1)
res.to_csv('fate_indices.csv', index=False)
pd.set_option('display.width', 250); print(res.drop(columns='ts_parts').round(3).to_string(index=False))
