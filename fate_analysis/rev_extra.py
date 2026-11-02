"""Further revision analyses on the real data.
 1. Deep networks of the earlier MEFAR analysis (scripts/04): same unit-level inference as rev_stats.
 2. Confound-control comparators on the main participant-independent partition:
    (a) cross-validated confound regression (CVCR): the proxy is regressed out of every feature on
        the training participants only, then the model is refitted;
    (b) the partial and full confounder tests of Spisak (2022, mlconfound) on unit-level predictions
        (H0 partial: predictions independent of the proxy given the label; H0 full: predictions
        independent of the label given the proxy).
 3. FatigueSet block-level regression of predictions on reported mental fatigue (session fixed
    effects, block adjustment) with wild cluster bootstrap-t p-values (Webb six-point weights,
    null imposed, 9,999 replications; 12 participant clusters).
 4. Fold-assignment stability of the version-2 tiers: L4 repeated with seeds 1-5 (MEFAR) and
    1-9 (FatigueSet, five-fold held-out participants) for the six reference classifiers.
Writes rev_deep.csv, rev_comparators.csv, rev_fs_wild.csv, rev_seed_tiers.csv.
Usage: python rev_extra.py [deep|comp|wild|seeds] ...
"""
import os, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fate
from common import load_mefar, load_fs, models, SEED, MEFAR_RESULTS
from rev_stats import load, NAMES, REF, NP
from published import jampala_models

def unit_inference(dd, strat_col, proxy_col, keep=None):
    y, rk, U, P = dd.y.values, dd['rank'].values, dd.unit.values, dd.participant.values
    ut = fate.unit_table(y, rk, U, P); r = {'pooled_auc': fate.auc_fast(y, rk), 'unit_auc': fate.auc_fast(ut.y, ut.s)}
    for sch in ['free', 'within', 'block']: r[f'p_{sch}'] = fate.unit_perm_p(y, rk, U, P, sch, NP)
    r['ts_auc'], r['ts_p'] = fate.ts_auc(y, rk, U, dd[strat_col].values, P, NP)
    k = np.ones(len(dd), bool) if keep is None else keep
    r['ls_auc'], r['ls_p'] = fate.ls_auc(y[k], rk[k], U[k], dd[proxy_col].values[k], P[k], NP)
    return r

def deep():
    d, _ = load_mefar()
    o = pd.read_csv(os.path.join(MEFAR_RESULTS, 'p2', 'p2_deep_oof_scores.csv.gz'))
    rows = []
    for m, g in o.groupby('model', sort=False):
        g = g.copy(); g['y'] = g.y_true.astype(int); g['participant'] = g.subject_id; g['unit'] = g.session_id
        g['session'] = g.session_id.str.split('_').str[-1]; g['proxy'] = (g.session == 'evening').astype(int)
        g['rank'] = fate.rank_within_fold(g.score.values, g.fold.values)
        r = {'model': m, **unit_inference(g, 'session', 'proxy')}
        a, se = fate.obuchowski_se(*[fate.unit_table(g.y.values, g['rank'].values, g.unit.values, g.participant.values)[c].values for c in ['y', 's', 'p']])
        r['unit_obu_lo'], r['unit_obu_hi'] = a - 1.96 * se, a + 1.96 * se
        rows.append(r); print(r, flush=True)
    R = pd.DataFrame(rows); R['p_within_holm4'] = fate.holm(R.p_within); R['ts_p_holm4'] = fate.holm(R.ts_p)
    R['tier'] = [fate.tier(a, b) for a, b in zip(R.p_within_holm4, R.ts_p_holm4)]
    R.to_csv('rev_deep.csv', index=False); print(R.round(3).to_string())

def cvcr_oof(model, X, y, c, folds):
    s = np.full(len(y), np.nan); f = np.full(len(y), -1)
    for i, (tr, te) in enumerate(folds):
        med = np.nanmedian(X[tr], 0); Xf = np.where(np.isnan(X), med, X)      # training-fold median imputation
        lr = LinearRegression().fit(c[tr, None], Xf[tr]); X_ = X; X = Xf
        s[te] = clone(model).fit(X[tr] - lr.predict(c[tr, None]), y[tr]).predict_proba(X[te] - lr.predict(c[te, None]))[:, 1]; f[te] = i; X = X_
    return s, f

def comparators():
    from mlconfound.stats import partial_confound_test, full_confound_test
    rows = []; fo_ = 'rev_comparators.csv'
    prev = pd.read_csv(fo_) if os.path.exists(fo_) else pd.DataFrame(columns=['dataset'])
    dss = os.environ.get('REV_DS', 'MEFAR,FatigueSet').split(','); prev = prev[~prev.dataset.isin(dss)]
    for ds in dss:
        if ds == 'MEFAR':
            d, F = load_mefar(); d['proxy'] = (d.session == 'evening').astype(int); d['stratum'] = d.session; tag, main = 'mefar', 'held_out_participant'
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(d[F], d.y, d.participant)); M = {**models(False), **jampala_models(True)}
        else:
            d, F = load_fs(); d['proxy'] = (d.block == 2).astype(int); d['stratum'] = d.block; tag, main = 'fs', 'loso'
            folds = list(LeaveOneGroupOut().split(d[F], d.y, d.participant)); M = {**models(True), **jampala_models(True)}
        X = d[F].values.astype(float); y = d.y.values; c = (d.session == 'evening').astype(float).values if ds == 'MEFAR' else d.block.values.astype(float)
        keep = None if ds == 'MEFAR' else d.block.isin([0, 2]).values
        for n in NAMES:
            r = {'dataset': ds, 'model': n}
            am = load(tag, main, n, d)
            if am is None: continue
            ut = am.groupby('unit').agg(y=('y', 'first'), s=('rank', 'mean'), c=('stratum', 'first'))
            cc = ut.c.map({'morning': 0, 'evening': 1}).values if ds == 'MEFAR' else ut.c.values
            pt = partial_confound_test(ut.y.values, ut.s.values, cc, num_perms=2000, cat_y=True, cat_c=True, random_state=SEED, progress=False, n_jobs=1)
            ft = full_confound_test(ut.y.values, ut.s.values, cc, num_perms=2000, cat_y=True, cat_c=True, random_state=SEED, progress=False, n_jobs=1)
            r['partial_p'] = pt.p; r['full_p'] = ft.p; r['r2_y_yhat'] = pt.r2_y_yhat; r['r2_yhat_c'] = pt.r2_yhat_c; r['r2_y_c'] = pt.r2_y_c
            if n in M:                                                         # CVCR (GCA excluded: run time)
                s, fo = cvcr_oof(M[n], X, y, c, folds); dd = d.copy(); dd['score'] = s; dd['fold'] = fo
                dd['rank'] = fate.rank_within_fold(s, fo); u = unit_inference(dd, 'stratum', 'proxy', keep)
                r.update({f'cvcr_{k}': v for k, v in u.items()})
            rows.append(r); print(r, flush=True)
    R = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
    for ds, g in R.groupby('dataset'):
        for c_ in ['partial_p', 'full_p', 'cvcr_p_within', 'cvcr_ts_p']:
            k = g[c_].notna(); R.loc[g.index[k], c_ + '_holm'] = fate.holm(g[c_][k])
    R.to_csv('rev_comparators.csv', index=False); print(R.round(3).to_string())

def wild_cluster_p(u, formula, term, cluster, B=9999, seed=SEED):
    """Wild cluster bootstrap-t (restricted, Webb six-point weights) p-value for one coefficient."""
    rng = np.random.default_rng(seed)
    full = smf.ols(formula, u).fit(cov_type='cluster', cov_kwds={'groups': cluster}); t0 = full.tvalues[term]
    rest = smf.ols(formula.replace(f' + {term}', ''), u).fit(); yhat0 = rest.fittedvalues.values; e0 = rest.resid.values
    W6 = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)]); cl = np.unique(cluster); tb = []
    for _ in range(B):
        w = dict(zip(cl, rng.choice(W6, len(cl)))); ys = yhat0 + e0 * np.array([w[c] for c in cluster])
        fb = smf.ols(formula, u.assign(s=ys)).fit(cov_type='cluster', cov_kwds={'groups': cluster}); tb.append(fb.tvalues[term])
    tb = np.array(tb); return full.params[term], full.pvalues[term], np.mean(np.abs(tb) >= abs(t0))

def wild():
    d, _ = load_fs(); rows = []
    for n in ['Logistic Regression', 'Random Forest', 'XGBoost']:
        am = load('fs', 'loso', n, d)
        u = am.groupby('unit').agg(s=('rank', 'mean'), mf=('mental_fatigue', 'first'), block=('block', 'first'),
                                   session=('session', 'first'), participant=('participant', 'first')).reset_index()
        u['mf_z'] = (u.mf - u.mf.mean()) / u.mf.std(); cl = pd.factorize(u.participant)[0]
        for adj, f in [('unadjusted', 's ~ C(session) + mf_z'), ('block-adjusted', 's ~ C(block) + C(session) + mf_z')]:
            b, p_cr, p_wb = wild_cluster_p(u, f, 'mf_z', cl, B=int(os.environ.get('REV_NWILD', 9999)))
            rows.append({'model': n, 'model_spec': adj, 'beta': b, 'p_cluster_robust': p_cr, 'p_wild_cluster': p_wb}); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('rev_fs_wild.csv', index=False)

def seeds():
    rows = []
    for ds in os.environ.get('REV_SEED_DS', 'MEFAR,FatigueSet').split(','):
        if ds == 'MEFAR': d, F = load_mefar(); d['stratum'] = d.session; d['proxy'] = (d.session == 'evening').astype(int); sl = [1, 2, 3, 4, 5]; M = models(False); keep = None
        else: d, F = load_fs(); d['stratum'] = d.block; d['proxy'] = (d.block == 2).astype(int); sl = list(range(1, 10)); M = models(True); keep = d.block.isin([0, 2]).values
        X = d[F].values; y = d.y.values
        f_ = f'rev_seed_tiers_{ds}.csv'; done = pd.read_csv(f_) if os.path.exists(f_) else pd.DataFrame(columns=['seed', 'model'])
        for sd in sl:
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(X, y, d.participant))
            for n in REF:
                if ((done.seed == sd) & (done.model == n)).any(): continue
                s, fo = fate.oof_scores(M[n], X, y, folds); dd = d.assign(score=s, fold=fo, rank=fate.rank_within_fold(s, fo))
                r = {'dataset': ds, 'seed': sd, 'model': n, **unit_inference(dd, 'stratum', 'proxy', keep)}
                done = pd.concat([done, pd.DataFrame([r])], ignore_index=True); done.to_csv(f_, index=False); print(r, flush=True)
        rows.append(done)
    R = pd.concat(rows, ignore_index=True)
    for (ds, sd), g in R.groupby(['dataset', 'seed']):
        R.loc[g.index, 'tier'] = [fate.tier(a, b) for a, b in zip(fate.holm(g.p_within), fate.holm(g.ts_p))]
    R.to_csv('rev_seed_tiers.csv', index=False); print(R.groupby(['dataset', 'model']).tier.value_counts().unstack(fill_value=0))

if __name__ == '__main__':
    for job in (sys.argv[1:] or ['deep', 'comp', 'wild', 'seeds']):
        {'deep': deep, 'comp': comparators, 'wild': wild, 'seeds': seeds}[job]()
