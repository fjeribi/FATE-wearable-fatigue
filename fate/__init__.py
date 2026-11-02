"""FATE: leakage- and confound-aware evaluation of wearable fatigue classifiers (version 2).

Minimal usage
-------------
    import fate
    res = fate.evaluate(model, X, y, participant=pid, unit=session_id,
                        block=chunk_id, proxy=is_evening)
    print(res['indices'])          # LII (with unit/participant parts), CDI, TS-AUC, LS-AUC

`unit` is the labelled unit (every sample in a unit shares one label); `proxy` is a binary
variable that co-varies with the label by design (time of day, task phase); `strata` defines
the strata within which TS-AUC is computed (the proxy itself unless given).

Changes in version 2 (revision)
-------------------------------
* Unit-level permutation tests respect the nesting of units in participants: labels are
  exchanged only among the units of the same participant ('within'), or whole participant
  label vectors are exchanged between participants with the same number of units ('block').
  The unrestricted scheme of version 1 ('free') is kept for comparison.
* Pooled ROC-AUC is reported both from raw out-of-fold scores and after rank-normalisation
  within test folds; the unit-level AUC can be computed from either.
* Clustered (Obuchowski) standard errors and participant-bootstrap CIs for every index,
  including LII and CDI.
* LII is decomposed into a labelled-unit part (L1 -> level where the unit is first held out,
  participant distribution unchanged) and a participant part (that level -> L4).
* LS-AUC: label-stratified proxy AUC of the *fatigue* model's own scores, a direct test of
  whether the fatigue predictions carry proxy information beyond the label.
* Tier rule: Tier 3 requires a significant L4 test and a significant TS-AUC. CDI and LS-AUC are
  reported with CIs as descriptors of proxy encoding and no longer gate the tier (tier_v1 keeps
  the version-1 rule for comparison).
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut

SEED = 42
LEVELS = ('L1_random_samples', 'L2_random_blocks', 'L3_heldout_units', 'L4_heldout_participants')

# ---------------- C1: partition ladder ----------------
def ladder_folds(level, y, participant=None, unit=None, block=None, k=5, seed=SEED, loso=False):
    idx = np.zeros(len(y))
    if level == 'L1_random_samples':
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y))
    g = {'L2_random_blocks': block, 'L3_heldout_units': unit, 'L4_heldout_participants': participant}[level]
    if level == 'L4_heldout_participants' and loso:
        folds = list(LeaveOneGroupOut().split(idx, y, g))
    else:
        folds = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(idx, y, g))
    g = np.asarray(g)
    for tr, te in folds:                                   # leakage guard
        assert not set(g[tr]) & set(g[te]), f'group overlap at {level}'
    return folds

def oof_scores(model, X, y, folds):
    s = np.full(len(y), np.nan); f = np.full(len(y), -1)
    for i, (tr, te) in enumerate(folds):
        s[te] = clone(model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; f[te] = i
    return s, f

# ---------------- fast AUC helpers ----------------
def auc_fast(y, s):
    """Mann-Whitney ROC-AUC (ties count 1/2)."""
    y = np.asarray(y).astype(bool); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def _auc_matrix(Y, s):
    """AUC of fixed scores s against each row of a 0/1 label matrix Y (same n1 per row)."""
    r = rankdata(s); n1 = Y[0].sum(); n0 = Y.shape[1] - n1
    return (Y @ r - n1 * (n1 + 1) / 2) / (n1 * n0)

# ---------------- C2: unit-level metrics and uncertainty ----------------
def rank_within_fold(s, fold):
    return pd.Series(s).groupby(np.asarray(fold)).rank(pct=True).values

def pooled_auc(y, s, fold=None, ranked=True):
    """Pooled out-of-fold ROC-AUC; ranked=False uses the raw scores."""
    return roc_auc_score(y, rank_within_fold(s, fold) if ranked else s)

def unit_table(y, r, unit, participant=None, extra=None):
    d = pd.DataFrame({'y': y, 's': r, 'u': unit})
    agg = dict(y=('y', 'first'), s=('s', 'mean'))
    if participant is not None: d['p'] = participant; agg['p'] = ('p', 'first')
    if extra is not None:
        for k, v in extra.items(): d[k] = v; agg[k] = (k, 'first')
    return d.groupby('u').agg(**agg)

def unit_auc(y, r, unit):
    u = unit_table(y, r, unit); return roc_auc_score(u.y, u.s)

def permute_labels(yu, pu, n, scheme='within', seed=SEED):
    """n permutations of unit labels yu (participants pu).
    free  : labels shuffled across all units (version 1);
    within: labels shuffled among the units of the same participant (restricted permutation);
    block : whole participant label vectors exchanged between participants with equal numbers
            of units, unit order within participant preserved."""
    rng = np.random.default_rng(seed); yu = np.asarray(yu); pu = np.asarray(pu); Y = np.empty((n, len(yu)), int)
    if scheme == 'free':
        for b in range(n): Y[b] = rng.permutation(yu)
    elif scheme == 'within':
        groups = [np.flatnonzero(pu == p) for p in pd.unique(pu)]
        for b in range(n):
            yb = yu.copy()
            for g in groups: yb[g] = yu[rng.permutation(g)]
            Y[b] = yb
    elif scheme == 'block':
        groups = [np.flatnonzero(pu == p) for p in pd.unique(pu)]
        sizes = np.array([len(g) for g in groups])
        for b in range(n):
            yb = yu.copy()
            for m in np.unique(sizes):
                gi = np.flatnonzero(sizes == m); perm = rng.permutation(gi)
                for a, c in zip(gi, perm): yb[groups[a]] = yu[groups[c]]
            Y[b] = yb
    else:
        raise ValueError(scheme)
    return Y

def unit_perm_p(y, r, unit, participant=None, scheme='within', n=2000, seed=SEED):
    """Unit-level permutation p-value (one-sided, AUC > null)."""
    u = unit_table(y, r, unit, participant)
    obs = auc_fast(u.y.values, u.s.values)
    if scheme != 'free' and participant is None: raise ValueError('participant needed')
    Y = permute_labels(u.y.values, u.p.values if participant is not None else np.zeros(len(u)), n, scheme, seed)
    keep = Y.sum(1) == u.y.sum()
    null = _auc_matrix(Y[keep], u.s.values)
    return (1 + np.sum(null >= obs - 1e-12)) / (1 + keep.sum())

def obuchowski_se(y, s, cluster):
    """Clustered ROC-AUC and its standard error (Obuchowski 1997), clusters = participants."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, float); c = np.asarray(cluster)
    pos, neg = s[y], s[~y]; cp, cn = c[y], c[~y]
    psi = lambda a, b: (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])
    P = psi(pos, neg); A = P.mean(); N1, N0 = len(pos), len(neg)
    cl = np.unique(c); I = len(cl)
    V10 = np.array([P[cp == k].sum() for k in cl]) / N0      # per-cluster sums over positives
    V01 = np.array([P[:, cn == k].sum() for k in cl]) / N1   # per-cluster sums over negatives
    m = np.array([(cp == k).sum() for k in cl]); n_ = np.array([(cn == k).sum() for k in cl])
    S10 = I / ((I - 1) * N1) * np.sum((V10 - m * A) ** 2)
    S01 = I / ((I - 1) * N0) * np.sum((V01 - n_ * A) ** 2)
    S11 = I / (I - 1) * np.sum((V10 - m * A) * (V01 - n_ * A))
    var = S10 / N1 + S01 / N0 + 2 * S11 / (N1 * N0)
    return A, np.sqrt(max(var, 0))

def participant_bootstrap_ci(y, r, participant, n=1000, seed=SEED):
    rng = np.random.default_rng(seed); p = np.asarray(participant); ps = np.unique(p)
    ix = {q: np.flatnonzero(p == q) for q in ps}; out = []
    for _ in range(n):
        b = np.concatenate([ix[q] for q in rng.choice(ps, len(ps))])
        if len(np.unique(y[b])) == 2: out.append(roc_auc_score(y[b], r[b]))
    return tuple(np.percentile(out, [2.5, 97.5]))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj

def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; adj[o] = np.minimum(q, 1)
    return adj

# ---------------- C3: confound attribution ----------------
def lii(auc_l1, auc_l4):
    """Leakage Inflation Index."""
    return auc_l1 - auc_l4

def lii_parts(auc_l1, auc_unit_level, auc_l4):
    """LII = unit part (L1 -> first level holding out the labelled unit, same participants in
    training) + participant part (that level -> L4)."""
    return {'LII': auc_l1 - auc_l4, 'LII_unit': auc_l1 - auc_unit_level, 'LII_participant': auc_unit_level - auc_l4,
            'share_unit': (auc_l1 - auc_unit_level) / (auc_l1 - auc_l4) if auc_l1 != auc_l4 else np.nan}

def cdi(auc_proxy, auc_label):
    """Confound Dominance Index (both participant-independent). Descriptor of proxy encoding."""
    return auc_proxy - auc_label

def ts_auc(y, r, unit, strata, participant=None, n=2000, seed=SEED):
    """Stratified AUC: mean unit-level AUC within proxy strata. The permutation shuffles labels
    within strata, and additionally within participants when a participant contributes several
    units to a stratum (participant given)."""
    d = pd.DataFrame({'y': y, 's': r, 'u': unit, 'g': strata})
    if participant is not None: d['p'] = participant
    U = []
    for _, e in d.groupby('g'):
        u = e.groupby('u').agg(y=('y', 'first'), s=('s', 'mean'), **({'p': ('p', 'first')} if participant is not None else {}))
        if u.y.nunique() == 2: U.append(u)
    obs = np.mean([auc_fast(u.y.values, u.s.values) for u in U])
    nulls = []
    for i, u in enumerate(U):
        many = participant is not None and u.p.duplicated().any()
        Y = permute_labels(u.y.values, u.p.values if many else np.zeros(len(u)), n, 'within' if many else 'free', seed + i)
        nulls.append(_auc_matrix(Y, u.s.values))
    null = np.mean(nulls, 0)
    return obs, (1 + np.sum(null >= obs - 1e-12)) / (1 + n)

def ls_auc(y, r, unit, proxy, participant=None, n=2000, seed=SEED):
    """Label-stratified proxy AUC: within units that share a label, do the fatigue model's scores
    separate proxy = 1 from proxy = 0? Mean over label strata, with a within-stratum permutation of
    the proxy. A direct test of proxy information in the fatigue predictions beyond the label."""
    return ts_auc(proxy, r, unit, y, participant, n, seed)

# ---------------- C4: grading ----------------
def tier(p_l4_holm, ts_p_holm, cdi_value=None, alpha=0.05):
    """Version 2: 1 = no participant-independent evidence; 2 = generalises, but no evidence
    beyond the proxy (TS-AUC not significant); 3 = fatigue-attributable (TS-AUC significant)."""
    if p_l4_holm >= alpha: return 1
    if ts_p_holm >= alpha: return 2
    return 3

def tier_v1(p_l4_holm, ts_p_holm, cdi_value, alpha=0.05):
    """Version-1 rule (CDI > 0 also gave Tier 2), kept for comparison."""
    if p_l4_holm >= alpha: return 1
    if ts_p_holm >= alpha or cdi_value > 0: return 2
    return 3

def evaluate(model, X, y, participant, unit, block, proxy, strata=None, k=5, seed=SEED, loso=False,
             first_unit_level=None, scheme='within', n_perm=2000):
    """Run C1-C3 for one model. first_unit_level: the level at which the labelled unit is first
    held out ('L2_random_blocks' or 'L3_heldout_units'; default L3). Holm correction and tiers are
    applied across models by the caller."""
    y = np.asarray(y); proxy = np.asarray(proxy); participant = np.asarray(participant)
    strata = proxy if strata is None else np.asarray(strata); fu = first_unit_level or LEVELS[2]
    out = {}
    for L in LEVELS:
        f = ladder_folds(L, y, participant, unit, block, k, seed, loso and L == LEVELS[3])
        s, fo = oof_scores(model, X, y, f); r = rank_within_fold(s, fo)
        out[L] = {'pooled_auc': roc_auc_score(y, r), 'pooled_auc_raw': roc_auc_score(y, s), 'unit_auc': unit_auc(y, r, unit)}
        if L == LEVELS[3]:
            u = unit_table(y, r, unit, participant)
            out[L].update(ci=participant_bootstrap_ci(y, r, participant),
                          obuchowski=obuchowski_se(u.y.values, u.s.values, u.p.values),
                          perm_p=unit_perm_p(y, r, unit, participant, scheme, n_perm),
                          perm_p_free=unit_perm_p(y, r, unit, None, 'free', n_perm))
            out['ts_auc'], out['ts_p'] = ts_auc(y, r, unit, strata, participant, n_perm)
            out['ls_auc'], out['ls_p'] = ls_auc(y, r, unit, proxy, participant, n_perm)
    f = ladder_folds(LEVELS[3], proxy, participant, unit, block, k, seed, loso)
    s, fo = oof_scores(model, X, proxy, f); out['proxy_auc'] = pooled_auc(proxy, s, fo)
    a1, au, a4 = out[LEVELS[0]]['pooled_auc'], out[fu]['pooled_auc'], out[LEVELS[3]]['pooled_auc']
    out['indices'] = {**lii_parts(a1, au, a4), 'CDI': cdi(out['proxy_auc'], a4),
                      'TS_AUC': out['ts_auc'], 'LS_AUC': out['ls_auc']}
    return out
