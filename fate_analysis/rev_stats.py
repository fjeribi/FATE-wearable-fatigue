"""Revised inference for the FATE indices on the real data (revision; Reviewer 1 comments 3-6,
Reviewer 2 comments 2-5). Works from the cached out-of-fold scores written by ladder.py and
run_published.py (identical models, folds and seeds to the original analysis).

For every model and dataset:
  * pooled ROC-AUC at every ladder level, from rank-normalised and from raw scores;
  * unit-level ROC-AUC (rank-normalised and raw) with participant-bootstrap and clustered
    (Obuchowski) 95% CIs;
  * unit-level permutation p under three exchangeability schemes: free (version 1), within
    participants (restricted; primary) and whole-participant blocks;
  * LII with its labelled-unit and participant parts, and CDI, each with a paired
    participant-bootstrap 95% CI (participants resampled jointly for all levels and the proxy task);
  * TS-AUC (labels permuted within proxy strata, and within participants where a participant has
    several units in a stratum) and LS-AUC (label-stratified proxy AUC of the fatigue scores);
  * tiers under the version-2 rule for every combination of permutation scheme, score scale and
    multiplicity family, and under the version-1 rule;
  * pairwise rank correlation of model predictions (duplicate detection).
Writes rev_indices.csv, rev_tier_stability.csv, rev_duplicates.csv.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fate
from common import load_mefar, load_fs, SEED

NAMES = ['Logistic Regression', 'Gaussian Naive Bayes', 'KNN', 'Random Forest', 'XGBoost', 'MLP',
         'Jampala RF', 'Jampala LR', 'Jampala KNN', 'Nooh GCA']
REF = NAMES[:6]; PUB = NAMES[6:]
NB = int(os.environ.get('REV_NBOOT', 1000)); NP = int(os.environ.get('REV_NPERM', 2000))

def load(tag, rung, name, d):
    f = f'cache/{tag}/{rung}__{name.replace(" ", "_")}.csv.gz'
    if not os.path.exists(f): return None
    o = pd.read_csv(f).sort_values('idx'); dd = d.iloc[o.idx.values].copy(); dd['fold'] = o.fold.values
    dd['score'] = o.score.values; dd['rank'] = fate.rank_within_fold(o.score.values, o.fold.values); return dd

def setup():
    dm, _ = load_mefar(); dm['proxy'] = (dm.session == 'evening').astype(int); dm['stratum'] = dm.session
    df, _ = load_fs(); df['proxy'] = (df.block == 2).astype(int); df['stratum'] = df.block
    fe = df[df.block.isin([0, 2])].reset_index(drop=True); fe['y'] = fe.proxy
    return [dict(ds='MEFAR', tag='mefar', d=dm, L=['random_sample', 'random_60s_chunk', 'held_out_session', 'held_out_participant'],
                 unit_level='held_out_session', main='held_out_participant', ptag='proxy_mefar', prung='held_out_participant',
                 pd=dm.assign(y=dm.proxy)),
            dict(ds='FatigueSet', tag='fs', d=df, L=['random_sample', 'held_out_block', 'held_out_session', 'held_out_participant'],
                 unit_level='held_out_block', main='loso', ptag='proxy_fs', prung='loso', pd=fe)]

def boot_indices(levels, prox, main, nb=NB, seed=SEED):
    """Paired participant bootstrap of LII, its parts and CDI."""
    rng = np.random.default_rng(seed); parts = np.unique(main.participant.values)
    def prep(dd): return dd.y.values, dd['rank'].values, {p: np.flatnonzero(dd.participant.values == p) for p in parts}
    P = {k: prep(v) for k, v in levels.items()}; PX = prep(prox); PM = prep(main)
    def auc(t, b):
        y, r, ix = t; i = np.concatenate([ix[p] for p in b if len(ix[p])])
        return fate.auc_fast(y[i], r[i]) if 0 < y[i].sum() < len(i) else np.nan
    out = []
    for _ in range(nb):
        b = rng.choice(parts, len(parts))
        a = {k: auc(v, b) for k, v in P.items()}; am = auc(PM, b); ap = auc(PX, b)
        out.append({'LII': a['L1'] - a['L4'], 'LII_unit': a['L1'] - a['LU'], 'LII_participant': a['LU'] - a['L4'], 'CDI': ap - am})
    B = pd.DataFrame(out)
    return {f'{c}_lo': np.nanpercentile(B[c], 2.5) for c in B} | {f'{c}_hi': np.nanpercentile(B[c], 97.5) for c in B}

def main():
    rows, dup = [], []
    for S in [q for q in setup() if q['ds'] in os.environ.get('REV_DS', 'MEFAR,FatigueSet').split(',')]:
        d = S['d']
        for n in NAMES:
            lv = {r: load(S['tag'], r, n, d) for r in S['L']}
            am = load(S['tag'], S['main'], n, d); pr = load(S['ptag'], S['prung'], n, S['pd'])
            if am is None or any(v is None for v in lv.values()) or pr is None:
                print('missing', S['ds'], n); continue
            r = {'dataset': S['ds'], 'model': n, 'family': 'reference' if n in REF else 'published'}
            for k, v in lv.items():
                r[f'auc_{k}'] = fate.auc_fast(v.y, v['rank']); r[f'auc_raw_{k}'] = fate.auc_fast(v.y, v.score)
            y, rk, sc_, U, P = am.y.values, am['rank'].values, am.score.values, am.unit.values, am.participant.values
            r['auc_main'] = fate.auc_fast(y, rk); r['auc_main_raw'] = fate.auc_fast(y, sc_)
            ut = fate.unit_table(y, rk, U, P); utr = fate.unit_table(y, sc_, U, P)
            r['unit_auc'] = fate.auc_fast(ut.y, ut.s); r['unit_auc_raw'] = fate.auc_fast(utr.y, utr.s)
            a_, se = fate.obuchowski_se(ut.y.values, ut.s.values, ut.p.values)
            r['unit_obu_lo'], r['unit_obu_hi'] = a_ - 1.96 * se, a_ + 1.96 * se
            r['boot_lo'], r['boot_hi'] = fate.participant_bootstrap_ci(y, rk, P, NB)
            for sch in ['free', 'within', 'block']:
                r[f'p_{sch}'] = fate.unit_perm_p(y, rk, U, P, sch, NP)
                r[f'p_{sch}_raw'] = fate.unit_perm_p(y, sc_, U, P, sch, NP)
            strata = am.stratum.values
            r['ts_auc'], r['ts_p'] = fate.ts_auc(y, rk, U, strata, P, NP)
            r['ts_auc_raw'], r['ts_p_raw'] = fate.ts_auc(y, sc_, U, strata, P, NP)
            r['ts_p_v1'] = fate.ts_auc(y, rk, U, strata, None, NP)[1]
            keep = am.stratum.isin([0, 2]).values if S['ds'] == 'FatigueSet' else np.ones(len(am), bool)
            r['ls_auc'], r['ls_p'] = fate.ls_auc(y[keep], rk[keep], U[keep], am.proxy.values[keep], P[keep], NP)
            r['proxy_auc'] = fate.auc_fast(pr.y, pr['rank'])
            r.update(fate.lii_parts(r[f'auc_{S["L"][0]}'], r[f'auc_{S["unit_level"]}'], r[f'auc_{S["L"][3]}']))
            r['CDI'] = r['proxy_auc'] - r['auc_main']
            r.update(boot_indices({'L1': lv[S['L'][0]], 'LU': lv[S['unit_level']], 'L4': lv[S['L'][3]]}, pr, am))
            rows.append(r); print(S['ds'], n, {k: round(v, 3) for k, v in r.items() if k in ['unit_auc', 'p_free', 'p_within', 'p_block', 'ts_auc', 'ts_p', 'ls_auc', 'ls_p', 'CDI', 'CDI_lo', 'CDI_hi']}, flush=True)
        # duplicates: rank correlation of main-partition predictions
        M = {n: load(S['tag'], S['main'], n, d) for n in NAMES}
        for i, a in enumerate(NAMES):
            for b in NAMES[i + 1:]:
                if M[a] is None or M[b] is None: continue
                dup.append({'dataset': S['ds'], 'a': a, 'b': b, 'spearman': spearmanr(M[a].score, M[b].score).correlation,
                            'unit_spearman': spearmanr(fate.unit_table(M[a].y.values, M[a]['rank'].values, M[a].unit.values).s,
                                                       fate.unit_table(M[b].y.values, M[b]['rank'].values, M[b].unit.values).s).correlation})
    R = pd.DataFrame(rows); D = pd.DataFrame(dup); D.to_csv('rev_duplicates.csv', index=False)
    dupl = set(D[D.unit_spearman > 0.99].apply(lambda e: (e.dataset, e.b), axis=1))
    R['duplicate'] = [(a, b) in dupl for a, b in zip(R.dataset, R.model)]
    # tiers under every inference choice
    st = []
    for ds, g in R.groupby('dataset'):
        for sch in ['within', 'free', 'block']:
            for scale in ['rank', 'raw']:
                pc = f'p_{sch}' + ('_raw' if scale == 'raw' else ''); tc = 'ts_p' + ('_raw' if scale == 'raw' else '')
                for fam in ['holm_all', 'holm_dedup', 'holm_family', 'bh_all', 'none']:
                    pa = pd.Series(np.nan, g.index); ta = pd.Series(np.nan, g.index)
                    if fam == 'holm_all': pa[:] = fate.holm(g[pc]); ta[:] = fate.holm(g[tc])
                    elif fam == 'bh_all': pa[:] = fate.bh(g[pc]); ta[:] = fate.bh(g[tc])
                    elif fam == 'none': pa[:] = g[pc]; ta[:] = g[tc]
                    elif fam == 'holm_dedup':
                        k = ~g.duplicate; pa[k] = fate.holm(g[pc][k]); ta[k] = fate.holm(g[tc][k])
                    else:
                        for f_, e in g.groupby('family'): pa[e.index] = fate.holm(e[pc]); ta[e.index] = fate.holm(e[tc])
                    for i in g.index:
                        if np.isnan(pa[i]): continue
                        st.append({'dataset': ds, 'model': g.model[i], 'scheme': sch, 'scale': scale, 'family': fam,
                                   'p_adj': pa[i], 'ts_p_adj': ta[i], 'tier': fate.tier(pa[i], ta[i]),
                                   'tier_v1': fate.tier_v1(pa[i], ta[i], g.CDI[i])})
    T = pd.DataFrame(st); T.to_csv('rev_tier_stability.csv', index=False)
    prim = T[(T.scheme == 'within') & (T.scale == 'rank') & (T.family == 'holm_dedup')].set_index(['dataset', 'model'])
    R = R.join(prim[['p_adj', 'ts_p_adj', 'tier', 'tier_v1']], on=['dataset', 'model'])
    for ds, g in R.groupby('dataset'):      # exact version-1 grading (free permutation, unrestricted TS-AUC)
        pa = fate.holm(g.p_free); ta = fate.holm(g.ts_p_v1)
        R.loc[g.index, 'tier_v1_original'] = [fate.tier_v1(a, b, c) for a, b, c in zip(pa, ta, g.CDI)]
    R.to_csv('rev_indices.csv', index=False)
    pd.set_option('display.width', 300)
    print(R[['dataset', 'model', 'auc_main', 'auc_main_raw', 'unit_auc', 'unit_auc_raw', 'p_free', 'p_within', 'p_block', 'ts_auc', 'ts_p',
             'ls_auc', 'ls_p', 'LII', 'LII_lo', 'LII_hi', 'share_unit', 'CDI', 'CDI_lo', 'CDI_hi', 'tier', 'tier_v1']].round(3).to_string(index=False))
    print(T.groupby(['dataset', 'scheme', 'scale', 'family']).tier.apply(lambda x: ''.join(map(str, x))).to_string())

if __name__ == '__main__':
    main()
