"""Validation of the rebuilt participant-level MEFAR table against the MEFAR_DOWN release
(revision, Reviewer 1 comment 7).

The release is stored in recording order (lag-1 autocorrelation of TEMP 0.999) but carries no
participant or session identifiers and is assembled from 60-row segments. Exact row matching is not
possible because the release and the rebuilt table resample the wristband streams differently, so
each consecutive 60-row block of the release is located in the rebuilt table by trajectory matching:
the mean Pearson correlation of five wristband channels (EDA, HR, three-axis acceleration) between
the block and every 60-s window of every rebuilt session (scale-free, so the unknown min-max scaling
of the release does not matter). A time-reversed copy of each block gives the null distribution of
the best correlation. For accepted matches the recovered participant/session and the class label are
compared with the release. Consecutive release blocks that map to the same rebuilt session 180 s
apart (the release keeps one minute in every three) confirm the match independently of the threshold.
Writes rebuilt_match_blocks.csv and rebuilt_match_summary.csv."""
import numpy as np, pandas as pd
from common import MEFAR_CSV, MEFAR_DOWN
C = ['EDA', 'HR', 'AccX', 'AccY', 'AccZ']; W = 60
rb = pd.read_csv(MEFAR_CSV); rl = pd.read_csv(MEFAR_DOWN); rl.columns = rl.columns.str.strip()
rl = rl.dropna(subset=C).reset_index(drop=True)
S = [(k, g[C].values.astype(float), g['class'].values, g.subject_id.iloc[0], g.session.iloc[0])
     for k, g in rb.groupby('session_id', sort=False)]

def sliding_corr(x, y):
    """Correlation of a length-W block x with every length-W window of y."""
    n = len(y) - W + 1
    cs = np.cumsum(np.r_[0, y]); cs2 = np.cumsum(np.r_[0, y ** 2])
    m = (cs[W:] - cs[:-W]) / W; v = (cs2[W:] - cs2[:-W]) / W - m ** 2
    xz = (x - x.mean()); sx = np.sqrt((xz ** 2).mean())
    dot = np.correlate(y, xz, 'valid') / W
    with np.errstate(invalid='ignore', divide='ignore'):
        r = dot / (sx * np.sqrt(np.maximum(v, 0)))
    r[~np.isfinite(r)] = 0
    return r[:n]

def best(block):
    out = (-9.0, None, None)
    if (block.std(0) == 0).all(): return out
    for k, Y, cl, sub, ses in S:
        if len(Y) < W: continue
        r = np.mean([sliding_corr(block[:, j], Y[:, j]) if block[:, j].std() > 0 else np.zeros(len(Y) - W + 1)
                     for j in range(len(C))], 0)
        i = int(np.argmax(r))
        if r[i] > out[0]: out = (float(r[i]), k, i)
    return out

rows = []
for b0 in range(0, len(rl) - W + 1, W):
    blk = rl[C].values[b0:b0 + W].astype(float)
    r, k, o = best(blk); rn, _, _ = best(blk[::-1])
    cls_rel = rl['class'].values[b0:b0 + W].mean()
    rec = next(s for s in S if s[0] == k) if k else None
    rows.append({'block_start': b0, 'r_best': r, 'r_reversed': rn, 'session_id': k, 'offset': o,
                 'subject': rec[3] if rec else None, 'session': rec[4] if rec else None,
                 'class_release': cls_rel, 'class_rebuilt': rec[2][o:o + W].mean() if rec else np.nan})
    if len(rows) % 50 == 0: print(len(rows), flush=True)
B = pd.DataFrame(rows); B.to_csv('rebuilt_match_blocks.csv', index=False)
thr = np.quantile(B.r_reversed, 0.99)                       # 99th percentile of the null
acc = B[B.r_best > thr]
ok = B.r_best > thr; cons = [(B.session_id[i] == B.session_id[i + 1]) and abs(B.offset[i + 1] - B.offset[i] - 3 * W) <= 2
        for i in range(len(B) - 1) if ok[i] and ok[i + 1]]
summ = pd.DataFrame([{'consecutive_pairs_consistent': np.mean(cons), 'n_consecutive_pairs': len(cons),'blocks': len(B), 'null_q99': thr, 'median_r_best': B.r_best.median(),
                      'median_r_reversed': B.r_reversed.median(), 'accepted_share': len(acc) / len(B),
                      'class_agreement_accepted': np.mean(np.round(acc.class_release) == np.round(acc.class_rebuilt)),
                      'participants_recovered': acc.subject.nunique(), 'sessions_recovered': acc.session_id.nunique()}])
summ.to_csv('rebuilt_match_summary.csv', index=False); print(summ.T)
