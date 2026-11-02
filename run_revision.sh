#!/usr/bin/env bash
# Revision analyses (run after run_fate.sh; resumable: each script skips finished work).
# Usage: MEFAR_DOWN=data/MEFAR_DOWN.csv FATIGUESET_ROOT=data/fatigueset bash run_revision.sh
set -euo pipefail
S=fate_analysis
python $S/sim_validation.py 100          # synthetic validation of FATE (scenarios S0-S5), comparators
python $S/spikein.py 100                 # spike-in on the real designs: power, minimum detectable AUC
python $S/rev_stats.py                   # restricted permutations, raw vs rank AUC, CIs for LII/CDI, LS-AUC, tier stability
python $S/rev_extra.py deep comp wild seeds   # deep networks, confound-control comparators, wild cluster bootstrap, fold seeds
python $S/match_release.py               # rebuilt MEFAR table located in the public release
python $S/validate_rebuilt.py            # EEG integrity by data source
python $S/figures_rev.py                 # revision figures
echo "Revision analyses complete. Results in ${FATE_OUT:-results_fate}/"
