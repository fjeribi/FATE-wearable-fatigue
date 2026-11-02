"""EEG integrity by data source (revision, Reviewer 1 comment 7).
Compares the eight MEFAR band powers in (i) the table rebuilt by us from the raw NeuroSky files,
(ii) the public MEFAR_DOWN release, and FatigueSet Muse band powers, on three properties that
recorded EEG band powers share: positive correlation between bands, delta power exceeding high-gamma
power, and temporal persistence (lag-1 autocorrelation within a session; wristband EDA as reference).
Writes eeg_integrity_sources.csv."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import MEFAR_CSV, MEFAR_DOWN, FS_WINDOWS
B8 = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
FSB = ['eeg_delta', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma']
rb = pd.read_csv(MEFAR_CSV).dropna(subset=B8)
rl = pd.read_csv(MEFAR_DOWN); rl.columns = rl.columns.str.strip(); rl = rl.dropna(subset=B8)
fs = pd.read_csv(FS_WINDOWS).dropna(subset=FSB)

def corr_stats(X):
    r = spearmanr(X).correlation; iu = np.triu_indices(X.shape[1], 1); return np.mean(r[iu]), np.min(r[iu]), np.max(r[iu])
def lag1(df, cols, by):
    out = []
    for _, g in df.groupby(by):
        for c in cols:
            x = g[c].values
            if len(x) > 3 and x.std() > 0: out.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    return np.nanmean(out)
rows = []
for name, X, lo, hi, ac_eeg, ac_ref in [
        ('MEFAR rebuilt from raw NeuroSky files', rb[B8].values, 0, 7, lag1(rb, B8, 'session_id'), lag1(rb, ['EDA'], 'session_id')),
        ('MEFAR_DOWN release', rl[B8].values, 0, 7, lag1(rl.assign(blk=np.arange(len(rl)) // 60), B8, 'blk'),
         lag1(rl.assign(blk=np.arange(len(rl)) // 60), ['EDA'], 'blk')),
        ('FatigueSet Muse (30-s windows)', fs[FSB].values, 0, 4, lag1(fs, FSB, 'block_id'), lag1(fs, ['eda_mean'], 'block_id'))]:
    m, a, b = corr_stats(X)
    rows.append({'source': name, 'mean_rho': m, 'min_rho': a, 'max_rho': b, 'delta_gt_gamma': np.mean(X[:, lo] > X[:, hi]),
                 'lag1_autocorr_eeg': ac_eeg, 'lag1_autocorr_eda': ac_ref})
E = pd.DataFrame(rows); E.to_csv('eeg_integrity_sources.csv', index=False); print(E.round(3).to_string(index=False))
