# FATE: Leakage- and Confound-Aware Evaluation Framework for Wearable Mental-Fatigue Detection

Code accompanying the FATE paper (Frontiers in Public Health, revised version).

**Maintainer and corresponding author:** Dr Fathe Jeribi, Jazan University (fjeribi@jazanu.edu.sa)

FATE evaluates a wearable fatigue classifier in four steps:

| Component | What it does | Output |
|---|---|---|
| **C1 Partition ladder** | Same model at L1 random samples, L2 random time blocks, L3 held-out sessions, L4 held-out participants (plus LOSO) | Pooled out-of-fold ROC-AUC per level |
| **C2 Unit-level metrics** | Pooled AUC (rank-normalised and raw), unit-level AUC with clustered (Obuchowski) CI, participant-restricted permutation p, Holm correction over distinct models | L4 estimate with uncertainty |
| **C3 Confound attribution** | **LII** = AUC(L1) − AUC(L4), split into a labelled-unit and a participant part; **CDI** = AUC(proxy) − AUC(label); **TS-AUC** = unit AUC within proxy strata; **LS-AUC** = proxy AUC of the fatigue predictions within label strata; participant-bootstrap CIs | Indices with CIs |
| **C4 Evidence grade** | Tier 0 (leaky only), 1 (no significant L4 evidence), 2 (generalises but not separable from the proxy), 3 (significant L4 test and significant TS-AUC) | Tier + checklist |

Revision 2 of the toolkit: permutations are restricted to the units of each participant by default
(`scheme='within'`; `'free'` and `'block'` are available), CDI is a descriptor and no longer changes
the tier (the version-1 rule is kept as `fate.tier_v1`), and LS-AUC is added.

## Getting the code

```bash
git clone https://github.com/fjeribi/FATE-wearable-fatigue.git
cd FATE-wearable-fatigue
```

## Repository layout

```
fate/              FATE toolkit (a single importable module)
fate_analysis/     All analyses of the FATE paper (MEFAR and FatigueSet)
scripts/, src/     Reconstruction of the participant-level MEFAR table and the
                   earlier deep-network analysis (MLP, 1D-CNN, LSTM, MF-AFEN)
results/           Chalder scores and saved outputs of the earlier deep-network analysis
results_fate/      Result tables of the FATE paper (CSV)
figures/           Figures 1–6 of the paper (TIFF, 300 dpi)
run_all.sh         Step 1: rebuild the MEFAR table and run the earlier analysis
run_fate.sh        Step 2: run every FATE analysis
run_revision.sh    Step 3: analyses added in the revision (simulation, spike-in, robustness)
results_rev/       Result tables of the revised paper (CSV)
```

## Using the toolkit on your own data

```python
import fate
res = fate.evaluate(model, X, y,
                    participant=participant_id,   # one id per sample
                    unit=session_id,              # labelled unit: every sample shares its label
                    block=chunk_id,               # contiguous time block (L2)
                    proxy=is_evening)             # binary variable that co-varies with the label
print(res['indices'])                             # {'LII': ..., 'CDI': ..., 'TS_AUC': ...}
```
`model` is any scikit-learn-compatible classifier. Apply `fate.holm` across the models you compare
and `fate.tier` to assign the evidence grade.

## Data

| Dataset | Source | Put it at (or set the variable) |
|---|---|---|
| MEFAR raw files and `MEFAR_DOWN.csv` | https://doi.org/10.17632/z3g26tphnv.5 | `data/MEFAR/`, `data/MEFAR_DOWN.csv` (`MEFAR_DOWN`) |
| FatigueSet | https://www.esense.io/datasets/fatigueset | `data/fatigueset/` (`FATIGUESET_ROOT`) |

Other variables: `MEFAR_RESULTS` (default `results/`, output of step 1) and `FATE_OUT`
(default `results_fate/`, where FATE writes tables, caches and figures).

## Installation and reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh data/MEFAR data/MEFAR_DOWN.csv results     # step 1 (see scripts/ for details)
bash run_fate.sh                                            # step 2 (first-submission results, results_fate/)
FATE_OUT=results_rev bash run_fate.sh                       # step 2 again for the revised paper (results_rev/)
FATE_OUT=results_rev bash run_revision.sh                   # step 3: analyses added in the revision
```

The graph autoencoder on the rebuilt MEFAR table is the slowest step (about 30 CPU-minutes per fold). It can
also be run one ladder level at a time, in parallel processes, with per-fold checkpoints that make it resumable:

```bash
FATE_OUT=results_rev python fate_analysis/run_gca_one.py held_out_participant   # or random_sample,
# random_60s_chunk, held_out_session, proxy; MAX_FOLDS=2 limits the folds per call, ONLY_FOLDS=3,4 picks folds
```

| Script (`fate_analysis/`) | Paper output |
|---|---|
| `fs_features.py` | FatigueSet 30-s window features |
| `ladder.py`, `summarise_ladder.py` | Table 6, Figure 2 (reference models) |
| `validate_published.py` | Table 5, reproduction of published pipelines on `MEFAR_DOWN` |
| `published.py`, `run_published.py` | Re-implemented published pipelines under FATE; proxy tasks for all models |
| `fate_metrics.py` | Tables 7–8: LII, CDI, TS-AUC, Holm-adjusted p, tiers |
| `mefar_confound_all.py` | Table 9, Figure 3a |
| `fs_confound.py`, `fs_label_sens.py` | Table 10 |
| `repeat_seeds.py`, `nested.py`, `learning_curve.py` | Robustness analyses (Table 7) |
| `mfafen_jax.py` | Table 11 (FatigueSet) |
| `eeg_check.py` | Figure 4 |
| `fig_framework.py`, `figures.py` | Figures 1–4 |
| `sim_validation.py` | Simulation validation, Table 14, Figure 5, Supplementary Table S3 |
| `spikein.py` | Spike-in power and minimum detectable AUC, Figure 6, Supplementary Table S4 |
| `rev_stats.py` | Restricted permutations, raw vs rank AUC, LII/CDI CIs, LS-AUC, tier stability (Tables 8–9, S5) |
| `rev_extra.py` | Deep networks, confound-control comparators (Table 15, S7), wild cluster bootstrap (Table 11), fold seeds (S6) |
| `match_release.py`, `validate_rebuilt.py` | Validation of the rebuilt MEFAR table and EEG integrity by source (Supplementary Methods 2) |
| `run_gca_one.py` | One MEFAR graph-autoencoder ladder level per process, with per-fold checkpoints |
| `figures_rev.py` | Figures 5–6 |

All runs are CPU-only with seed 42. The revision results were regenerated with the versions listed in
`requirements.txt` (XGBoost 2.1.4; results with XGBoost 3.x differ slightly), so `results_rev/`
supersedes `results_fate/`. The graph autoencoder on the rebuilt MEFAR table takes about
40 CPU-minutes per ladder level on two cores; everything else runs in minutes.

## Re-implemented published pipelines

- **Jampala et al. (2025):** random forest, logistic regression and k-nearest neighbours with
  library defaults.
- **Nooh et al. (2025), approximation:** min-max scaling, binary genetic-algorithm wrapper feature
  selection (in place of the binary olympiad optimiser), an inductive GraphSAGE-style graph
  convolutional autoencoder, and fixed settings in place of the hippopotamus hyperparameter search.
  The original code was not released; see the docstring of `fate_analysis/published.py`.

On `MEFAR_DOWN` with the published 80:20 protocol these reach 97.2% (RF), 63.8% (LR), 86.9% (KNN)
and 95.1% (graph autoencoder) accuracy (`results_fate/published_validation.csv`).

## Citation

If you use FATE, please cite the paper (see `CITATION.cff`). Questions about the code: Dr Fathe Jeribi,
fjeribi@jazanu.edu.sa.
