#!/usr/bin/env bash
# Reproduce the FATE analyses (run after run_all.sh has produced results/MEFAR_subjectwise_1Hz_FINAL.csv).
# Usage: MEFAR_DOWN=data/MEFAR_DOWN.csv FATIGUESET_ROOT=data/fatigueset bash run_fate.sh
set -euo pipefail
S=fate_analysis
python $S/fs_features.py                       # FatigueSet 30-s window features
python $S/ladder.py mefar; python $S/ladder.py fs; python $S/ladder.py fsnorm   # C1 ladder, reference models
python $S/validate_published.py                # published pipelines on the public release (80:20)
python $S/run_published.py                     # published pipelines under the ladder + proxy tasks (slow: GCA)
for d in mefar fs fsnorm; do python $S/summarise_ladder.py $d; done
python $S/fate_metrics.py                      # LII, CDI, TS-AUC, Holm, tiers
python $S/mefar_confound_all.py; python $S/fs_confound.py; python $S/fs_label_sens.py
python $S/repeat_seeds.py mefar; python $S/repeat_seeds.py fs
python $S/nested.py mefar; python $S/nested.py fs
python $S/learning_curve.py mefar; python $S/learning_curve.py fs
python $S/mfafen_jax.py; python $S/eeg_check.py
python $S/fig_framework.py; python $S/figures.py
echo "FATE analyses complete. Results in ${FATE_OUT:-results_fate}/"
