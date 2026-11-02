"""Semi-synthetic spike-in on the real designs: power, minimum detectable AUC and tier recovery
(revision, Reviewer 1 comments 1 and 3).

A known signal of size delta (in pooled standard deviations) is added to four randomly chosen
wristband features of the real MEFAR (every 5th second) or FatigueSet window table, either
  fatigue : to every sample of each fatigued labelled unit (unit-specific gain 1 + 0.3 N(0, 1)), or
  clock   : to every sample of each evening session (MEFAR) / in proportion to block position
            (FatigueSet), i.e. a pure time-of-day / task-phase shortcut.
Logistic regression is then evaluated under the FATE L4 partition (MEFAR: five-fold held-out
participants; FatigueSet: LOSO) and graded. delta = 0 reproduces the real data. Reports, per delta,
the achieved unit-level ROC-AUC, the power of the restricted L4 test and of TS-AUC, and the share of
replications assigned each tier at alpha = 0.05 and at alpha = 0.005 (the first Holm step for ten
models). Usage: python spikein.py [n_reps]   -> spikein_reps.csv, spikein_summary.csv
"""
import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fate
from common import load_mefar, load_fs, sc
from sklearn.linear_model import LogisticRegression

DELTAS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
WRIST_M = ['BVP', 'EDA', 'TEMP', 'AccX', 'AccY', 'AccZ', 'HR']

def data(ds):
    if ds == 'mefar':
        d, F = load_mefar(); d = d[d.second % 5 == 0].reset_index(drop=True)
        d['proxy'] = (d.session == 'evening').astype(int); d['stratum'] = d.proxy; d['phase'] = d.proxy.astype(float)
        wrist = WRIST_M
    else:
        d, F = load_fs(); d['proxy'] = (d.block == 2).astype(int); d['stratum'] = d.block; d['phase'] = d.block / 2.0
        wrist = [c for c in F if not c.startswith('eeg')]
    return d, F, wrist

def run(ds, scen, delta, rep, cache):
    d, F, wrist = cache[ds]; rng = np.random.default_rng(rep * 1009 + int(delta * 1000) + (0 if scen == 'fatigue' else 7))
    X = d[F].values.astype(float).copy(); sd = np.nanstd(X, 0)
    cols = rng.choice([F.index(c) for c in wrist], 4, replace=False)
    units = d.unit.unique(); gain = dict(zip(units, 1 + 0.3 * rng.normal(size=len(units))))
    g = d.unit.map(gain).values
    if scen == 'fatigue': amp = d.y.values * g
    else:                 amp = d.phase.values * g
    sign = rng.choice([-1, 1], 4)
    for j, c in enumerate(cols): X[:, c] += sign[j] * delta * sd[c] * amp
    y = d.y.values; P = d.participant.values; U = d.unit.values
    folds = fate.ladder_folds('L4_heldout_participants', y, P, U, None, 5, rep, loso=(ds == 'fs'))
    model = sc(LogisticRegression(max_iter=3000))
    s, fo = fate.oof_scores(model, X, y, folds); r = fate.rank_within_fold(s, fo)
    u = fate.unit_table(y, r, U, P)
    p_l4 = fate.unit_perm_p(y, r, U, P, 'within', 1000, rep)
    ts, ts_p = fate.ts_auc(y, r, U, d.stratum.values, P, 1000, rep)
    out = {'dataset': ds, 'scenario': scen, 'delta': delta, 'rep': rep, 'unit_auc': fate.auc_fast(u.y.values, u.s.values),
           'pooled_auc': fate.auc_fast(y, r), 'p_l4': p_l4, 'ts_auc': ts, 'ts_p': ts_p}
    for a, tag in [(0.05, ''), (0.005, '_holm10')]:
        out['tier' + tag] = fate.tier(min(1, p_l4 * 0.05 / a), min(1, ts_p * 0.05 / a))
    return out

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    OUT = os.getcwd(); f = os.path.join(OUT, 'spikein_reps.csv')
    done = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=['dataset', 'scenario', 'delta', 'rep'])
    key = set(zip(done.dataset, done.scenario, done.delta.astype(float).round(3), done.rep.astype(int)))
    cache = {ds: data(ds) for ds in ['mefar', 'fs']}; rows = []; t0 = time.time()
    for rep in range(n):
        for ds in ['mefar', 'fs']:
            for scen in ['fatigue', 'clock']:
                for dl in DELTAS:
                    if scen == 'clock' and dl == 0: continue
                    if (ds, scen, round(dl, 3), rep) in key: continue
                    rows.append(run(ds, scen, dl, rep, cache))
        if rows and (rep % 5 == 4 or rep == n - 1):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); rows = []
            done.to_csv(f, index=False); print(f'rep {rep+1}/{n} {time.time()-t0:.0f}s', flush=True)
    d = pd.read_csv(f)
    d['tier'] = [fate.tier(a, b) for a, b in zip(d.p_l4, d.ts_p)]
    d['tier_holm10'] = [fate.tier(min(1, a * 10), min(1, b * 10)) for a, b in zip(d.p_l4, d.ts_p)]
    d.to_csv(f, index=False); g = d.groupby(['dataset', 'scenario', 'delta'])
    S = g.agg(n=('rep', 'size'), unit_auc=('unit_auc', 'median'), unit_auc_q25=('unit_auc', lambda x: x.quantile(.25)),
              unit_auc_q75=('unit_auc', lambda x: x.quantile(.75)), ts_auc=('ts_auc', 'median')).reset_index()
    S['power_l4'] = g.apply(lambda e: np.mean(e.p_l4 < .05)).values
    S['power_ts'] = g.apply(lambda e: np.mean(e.ts_p < .05)).values
    for t in [1, 2, 3]:
        S[f'tier{t}'] = g.apply(lambda e: np.mean(e.tier == t)).values
        S[f'tier{t}_holm10'] = g.apply(lambda e: np.mean(e.tier_holm10 == t)).values
    S.to_csv(os.path.join(OUT, 'spikein_summary.csv'), index=False)
    pd.set_option('display.width', 250); print(S.round(3).to_string(index=False))
