# Literature audit — independent verification summary

All 27 rows in `literature_audit.csv` were checked against publisher pages, DOI
resolvers, PubMed/PMC, arXiv, Semantic Scholar/CrossRef search results, and
(where those were blocked) mirror sources (Frontiers text mirrors, Europe PMC,
open-access PDFs, ResearchGate abstract pages). Every row's paper was located
and confirmed to exist with a matching title/first-author/year (after
corrections below); no row was dropped.

## Verification outcome

- **Verified as-is: 23 rows**
- **Corrected: 4 rows**
- **Dropped: 0 rows**

### Corrections made

1. **Row 11 (Kalanadhabhatta et al. 2022, FatigueSet):** `n_participants` was
   listed as "not confirmed in accessible text." The dataset paper's
   open-access camera-ready PDF states the study recruited **12 participants**
   (9 male, 3 female, ages 21–40). Corrected `n_participants` to `12`.

2. **Row 20 (Siddhad et al. 2026, Modified TSception):** `n_participants` was
   listed as "not stated in accessible text" and `confidence` as "from
   abstract only." The arXiv full text (2512.21747) states SEED-VIG has 23
   participants and STEW has 48 subjects, and describes the split ("40,710
   samples split into training (70%)/validation (15%)/test (15%)," then
   "stratified five-fold cross-validation") without confirming subject-wise
   grouping — so `protocol_class` correctly stays "unclear from paper," but
   `n_participants` and `confidence` were corrected since full text was in
   fact accessible.

3. **Row 23 (Cui, Wu, Ying and Li 2025, arXiv 2507.16859):** `headline_metric`
   claimed "accuracy 92.86% (imputed EEG+ECG, Transformer classifier)." The
   paper's Table 4 evaluates only **MLP, LSTM, and CNN1D** classifiers (no
   Transformer), and the value 92.86% does not appear anywhere in the text.
   The correct highest result for the VPFD w/ EEG+ECG (imputed) configuration
   is **LSTM, 92.12%**. Corrected `headline_metric` accordingly. (This value
   feeds the sample/window-level median below; the corrected 92.12 does not
   change the median.)

4. **Row 27 (originally "Chen et al. (EEG-Fest)"):** Both `author` and
   `title` were wrong. The paper at arXiv 2211.03878 (published as *Biomedical
   Physics & Engineering Express*, DOI 10.1088/2057-1976/ad0f3f) is authored
   by **Ning Ding, Ce Zhang, Azim Eskandarian**, not "Chen et al.," and its
   real title is **"EEG-Fest: Few-shot based Attention Network for Driver's
   Drowsiness Estimation with EEG Signals"** (not "A Generalized Few-Shot
   Model for Driver Drowsiness Detection"). The reported 5-shot/1-shot
   drowsy-class accuracies (94%/92%) and the subject-independent framing were
   confirmed correct. Corrected `author` to "Ding et al. (EEG-Fest)" and
   `title` to the real title.

No row's DOI/URL failed to resolve to a real, matching paper, so nothing was
dropped for non-existence. All other fields (dataset, n_participants,
signals, protocol_class, protocol_evidence_quote, headline_metric) checked
out against publisher/full-text sources for the remaining 23 rows.

## Counts by protocol class (n = 27, unchanged from the original audit)

| protocol_class | n |
|---|---|
| unclear from paper | 8 |
| subject-independent (LOSO / grouped by participant / held-out participants) | 7 |
| sample/window-level random split or k-fold (participants in both train and test) | 6 |
| within-subject only (separate model per person) | 4 |
| both reported | 2 |

("both reported" rows — Zhou et al. 2026 and Ammar et al. 2026 — each
contribute a number to both pools below, per the task instructions.)

## Median headline accuracy, sample-level vs. subject-independent

Only rows reporting a plain percentage accuracy are included (fraction-format
figures such as Jiang et al.'s "0.7454 ± 0.2147" and macro-F1 values such as
Ammar et al.'s are excluded, as in the original audit, since they are not on
the same scale).

- **Sample/window-level split or k-fold (participants in both train and test), n = 7:**
  Fan et al. 95.29%, Zhou et al. (subject-mixed) 97.47%, Butkevičiūtė et al. 94.5%,
  Cos et al. 96.0%, Sia et al. 75.7%, Zendehbad et al. 90.2%,
  Cui/Wu/Ying/Li 92.12% (corrected from the erroneous 92.86%)
  **Median = 94.5%** (unchanged by the correction)

- **Subject-independent (LOSO / grouped-by-participant / held-out participants), n = 7:**
  Zhou et al. (LOSO) 85.43%, Cui et al. 78.35%, Feng et al. 77.16%,
  Al Imran et al. 72.0%, Jaiswal et al. 84.1%, Guo et al. 88.6%,
  Ding et al. (EEG-Fest, corrected author) 94.0%
  **Median = 84.1%**

This reproduces the original audit's central finding: a **10.4-percentage-point
drop** in median headline accuracy from sample/window-level splitting to
genuine subject-independent evaluation. The four corrections made during this
verification pass (two n_participants/confidence fixes, one wrong headline
number, and one wrong author/title) do not change either median or the
protocol-class counts, but they do matter for citation accuracy — the
row-27 correction in particular (wrong author and wrong title) would have
been a citable misattribution had it gone uncorrected.
