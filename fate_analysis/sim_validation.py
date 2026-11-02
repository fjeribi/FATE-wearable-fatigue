"""Simulation validation of FATE (revision, Reviewer 1 comment 1).

Synthetic recordings with known ground truth are generated for two designs that mirror the
real datasets:
  mefar : 23 participants x 2 sessions (morning/evening = proxy), session-level label
  fs    : 12 participants x 3 sessions x 3 test blocks (block position = proxy), block-level label
Every sample has d = 8 features:
  x = participant offset + labelled-unit offset (fingerprint, sd = sigma_u) + AR(1) noise
      + beta_f * F_unit * w_f + beta_z * z_unit * w_z
where F is the latent fatigue of the unit (F = gamma*z + participant trait + noise) and the label
is F dichotomised at its median, optionally after adding rating noise.
Scenarios (expected tier):
  S0 null                 no signal, weak fingerprints                         Tier 1
  S1 leakage              no signal, strong unit fingerprints                   Tier 1 (large LII)
  S2 shortcut             features encode the proxy only                         Tier 2
  S3 fatigue              features encode latent fatigue only                    Tier 3
  S4 fatigue + shortcut   features encode fatigue and, separately, the proxy     Tier 3
  S5 fatigue, noisy label features encode fatigue; fatigue strongly tied to time; rating noise  Tier 3
Comparators evaluated on the same replications: random split (L1), participant-grouped CV
(L4 with an unrestricted permutation test), cross-validated confound regression (proxy regressed
out of every feature within the training fold, then L4), and the version-1 tier rule.
Usage: python sim_validation.py [n_reps]   (writes sim_validation_reps.csv, sim_validation_summary.csv)
"""
import sys, os, time, zlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

OUT = os.environ.get('FATE_OUT', 'results_rev'); os.makedirs(OUT, exist_ok=True)
D = 8
SCEN = {  # sigma_u, beta_f, beta_z, gamma (fatigue-time coupling), rating noise sd, expected tier
    'S0_null':            dict(su=0.3, bf=0.0, bz=0.0, g=1.2, rn=0.0, exp=1),
    'S1_leakage':         dict(su=1.0, bf=0.0, bz=0.0, g=1.2, rn=0.0, exp=1),
    'S2_shortcut':        dict(su=1.0, bf=0.0, bz=0.8, g=1.2, rn=0.0, exp=2),
    'S3_fatigue':         dict(su=1.0, bf=1.2, bz=0.0, g=1.2, rn=0.0, exp=3),
    'S4_fatigue_shortcut':dict(su=1.0, bf=1.2, bz=0.8, g=1.2, rn=0.0, exp=3),
    'S5_fatigue_noisylab':dict(su=1.0, bf=1.2, bz=0.0, g=2.0, rn=0.7, exp=3),
}

def generate(design, sc, rng):
    if design == 'mefar': P, S, B, T = 23, 2, 1, 60
    else:                 P, S, B, T = 12, 3, 3, 11
    rows = []
    trait = rng.normal(0, 1, P)
    for p in range(P):
        for s in range(S):
            for b in range(B):
                z = s if design == 'mefar' else b                         # proxy value (stratum)
                zc = (z - (0.5 if design == 'mefar' else 1.0)) / (0.5 if design == 'mefar' else 0.8165)
                F = sc['g'] * zc + 0.8 * trait[p] + rng.normal(0, 1)
                rows.append((p, f'{p}_{s}_{b}', s, z, zc, F, F + rng.normal(0, sc['rn']) if sc['rn'] else F))
    U = pd.DataFrame(rows, columns=['p', 'u', 'sess', 'z', 'zc', 'F', 'Fobs'])
    U['y'] = (U.Fobs >= U.Fobs.median()).astype(int)
    U['Fz'] = (U.F - U.F.mean()) / U.F.std()
    wf = rng.normal(0, 1, D); wf /= np.linalg.norm(wf) / np.sqrt(D / 2)
    wz = rng.normal(0, 1, D); wz /= np.linalg.norm(wz) / np.sqrt(D / 2)
    po = rng.normal(0, 1, (P, D)); X, meta = [], []
    for _, r in U.iterrows():
        e = np.zeros((T, D)); e[0] = rng.normal(0, 1, D)
        for t in range(1, T): e[t] = 0.9 * e[t - 1] + rng.normal(0, np.sqrt(1 - 0.81), D)
        x = po[r.p] + rng.normal(0, sc['su'], D) + e + sc['bf'] * r.Fz * wf + sc['bz'] * r.zc * wz
        X.append(x); meta += [(r.p, r.u, r.y, r.z, f'{r.p}_{r.sess}', t // max(1, T // 4)) for t in range(T)]
    m = pd.DataFrame(meta, columns=['participant', 'unit', 'y', 'z', 'session', 'chunk'])
    m['block'] = m.unit + '_' + m.chunk.astype(str)
    return np.vstack(X), m

def cvcr_oof(model, X, y, c, folds):
    s = np.full(len(y), np.nan); f = np.full(len(y), -1)
    for i, (tr, te) in enumerate(folds):
        lr = LinearRegression().fit(c[tr, None], X[tr])
        Xtr = X[tr] - lr.predict(c[tr, None]); Xte = X[te] - lr.predict(c[te, None])
        s[te] = clone(model).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]; f[te] = i
    return s, f

def one(design, scen, model_name, rep):
    sc = SCEN[scen]; rng = np.random.default_rng(10_000 * rep + zlib.crc32(f'{design}{scen}'.encode()) % 9973)
    X, m = generate(design, sc, rng); y = m.y.values; z = m.z.values
    if design == 'fs':  # binary proxy for CDI: last vs first block; strata = block position
        keep_px = np.isin(z, [0, 2]); proxy = (z == 2).astype(int)
    else:
        keep_px = np.ones(len(z), bool); proxy = z.astype(int)
    model = (Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=2000))]) if model_name == 'LR'
             else RandomForestClassifier(50, min_samples_leaf=5, n_jobs=1, random_state=rep))
    loso = design == 'fs'; P = m.participant.values; Uu = m.unit.values
    out = {'design': design, 'scenario': scen, 'model': model_name, 'rep': rep, 'expected': sc['exp']}
    lev = {}
    for L in fate.LEVELS:
        g3, g2 = (Uu, m.block.values) if design == 'mefar' else (m.session.values, Uu)
        f = fate.ladder_folds(L, y, P, g3, g2, 5, rep, loso and L == fate.LEVELS[3])
        s, fo = fate.oof_scores(model, X, y, f); r = fate.rank_within_fold(s, fo)
        lev[L] = fate.auc_fast(y, r)
        if L == fate.LEVELS[0]:
            out['L1_auc'] = lev[L]; out['L1_perm_p_free'] = fate.unit_perm_p(y, r, np.arange(len(y)), None, 'free', 500, rep)
        if L == fate.LEVELS[3]:
            out['L4_auc'] = lev[L]
            out['p_within'] = fate.unit_perm_p(y, r, Uu, P, 'within', 1000, rep)
            out['p_free'] = fate.unit_perm_p(y, r, Uu, P, 'free', 1000, rep)
            out['ts_auc'], out['ts_p'] = fate.ts_auc(y, r, Uu, z, P, 1000, rep)
            out['ls_auc'], out['ls_p'] = fate.ls_auc(y[keep_px], r[keep_px], Uu[keep_px], proxy[keep_px], P[keep_px], 1000, rep)
            # confound regression comparator
            s2, f2 = cvcr_oof(model, X, y, z.astype(float), f); r2 = fate.rank_within_fold(s2, f2)
            out['cvcr_auc'] = fate.auc_fast(y, r2); out['cvcr_p'] = fate.unit_perm_p(y, r2, Uu, P, 'within', 1000, rep)
    unit_lvl = fate.LEVELS[2] if design == 'mefar' else fate.LEVELS[1]
    out.update(fate.lii_parts(lev[fate.LEVELS[0]], lev[unit_lvl], lev[fate.LEVELS[3]]))
    f = fate.ladder_folds(fate.LEVELS[3], proxy[keep_px], P[keep_px], Uu[keep_px], None, 5, rep, loso)
    s, fo = fate.oof_scores(model, X[keep_px], proxy[keep_px], f)
    out['proxy_auc'] = fate.pooled_auc(proxy[keep_px], s, fo); out['CDI'] = out['proxy_auc'] - out['L4_auc']
    out['tier_v2'] = fate.tier(out['p_within'], out['ts_p']); out['tier_v1'] = fate.tier_v1(out['p_within'], out['ts_p'], out['CDI'])
    out['tier_v1_free'] = fate.tier_v1(out['p_free'], out['ts_p'], out['CDI'])
    return out

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    models_ = sys.argv[2].split(',') if len(sys.argv) > 2 else ['LR', 'RF']
    f = os.path.join(OUT, 'sim_validation_reps.csv')
    done = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=['design', 'scenario', 'model', 'rep'])
    key = set(zip(done.design, done.scenario, done.model, done.rep))
    rows = []; t0 = time.time()
    for rep in range(n):
        for design in ['mefar', 'fs']:
            for scen in SCEN:
                for mn in models_:
                    if mn == 'RF' and rep >= 30: continue           # RF on the first 30 replications
                    if (design, scen, mn, rep) in key: continue
                    rows.append(one(design, scen, mn, rep))
        if rows and (rep % 5 == 4 or rep == n - 1):           # resumable checkpoint
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); rows = []
            done.to_csv(f, index=False); print(f'rep {rep+1}/{n}  {time.time()-t0:.0f}s', flush=True)
    d = pd.read_csv(f)
    g = d.groupby(['design', 'scenario', 'model'])
    summ = g.agg(n=('rep', 'size'), expected=('expected', 'first'), L1=('L1_auc', 'mean'), L4=('L4_auc', 'mean'),
                 LII=('LII', 'mean'), share_unit=('share_unit', 'median'), CDI=('CDI', 'mean'), TS=('ts_auc', 'mean'),
                 LS=('ls_auc', 'mean')).reset_index()
    for c in ['tier_v2', 'tier_v1', 'tier_v1_free']:
        summ[f'correct_{c}'] = g.apply(lambda e: np.mean(e[c] == e.expected)).values
    for t in [1, 2, 3]: summ[f'v2_tier{t}'] = g.apply(lambda e: np.mean(e.tier_v2 == t)).values
    summ['L1_sig'] = g.apply(lambda e: np.mean(e.L1_perm_p_free < .05)).values
    summ['grouped_sig_free'] = g.apply(lambda e: np.mean(e.p_free < .05)).values
    summ['grouped_sig_within'] = g.apply(lambda e: np.mean(e.p_within < .05)).values
    summ['cvcr_sig'] = g.apply(lambda e: np.mean(e.cvcr_p < .05)).values
    summ['ts_sig'] = g.apply(lambda e: np.mean(e.ts_p < .05)).values
    summ['ls_sig'] = g.apply(lambda e: np.mean(e.ls_p < .05)).values
    summ.to_csv(os.path.join(OUT, 'sim_validation_summary.csv'), index=False)
    pd.set_option('display.width', 250); print(summ.round(3).to_string(index=False))
