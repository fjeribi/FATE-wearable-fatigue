"""Descriptive checks on the reconstructed table (Table 4, Section 4.6.2) and
exact parameter counts of the deep-learning models (Table 15).

Also reports whether E4 header rows leaked into the second-0 rows
(see scripts/02_reconstruct_subjectwise.py, --legacy-e4-parsing).

Usage:
    python scripts/09_descriptives.py --out-dir results [--release MEFAR_DOWN.csv]
"""
import argparse
import os

import _common  # noqa: F401
import pandas as pd

from src.config import SUBJECTWISE_CSV, TARGET
from src.data import load_subjectwise
from src.utils import banner, ensure_dir


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))

    banner("Label structure (Table 4)")
    print(f"Rows {len(df)} | participants {df.subject_id.nunique()} | sessions {df.session_id.nunique()}")
    sess = df.groupby(["subject_id", "session"])[TARGET].first().reset_index()
    t_sessions = pd.crosstab(sess.session, sess[TARGET], margins=True)
    t_rows = pd.crosstab(df.session, df[TARGET], margins=True)
    print("Sessions by label:\n", t_sessions, "\n\nObservations by label:\n", t_rows)
    wide = sess.pivot(index="subject_id", columns="session", values=TARGET)
    changed = wide[wide.morning != wide.evening]
    print(f"\nParticipants whose label changes between sessions: {len(changed)} of {len(wide)}")
    print(f"Session agrees with label (evening=fatigued) for "
          f"{((sess.session == 'evening').astype(int) == sess[TARGET]).mean():.1%} of sessions")
    t_sessions.to_csv(os.path.join(out, "label_structure_sessions.csv"))
    t_rows.to_csv(os.path.join(out, "label_structure_rows.csv"))
    wide.to_csv(os.path.join(out, "labels_per_participant.csv"))

    banner("Class proportions")
    print("Reconstructed:", df[TARGET].value_counts(normalize=True).round(4).to_dict())
    if args.release:
        rel = pd.read_csv(args.release); rel.columns = rel.columns.str.strip()
        print("Release:      ", rel[TARGET].value_counts(normalize=True).round(4).to_dict())

    banner("E4 header-row check (second-0 rows)")
    cols = ["BVP", "EDA", "TEMP", "HR"]
    at0 = df.loc[df.second == 0, cols]
    print(at0.describe().T[["min", "max"]])
    if (at0.abs() > 1e8).any().any():
        print("WARNING: second-0 values look like Unix timestamps; the table was built with legacy parsing.")
    else:
        print("OK: no timestamp values at second 0.")

    if not args.skip_params:
        banner("Parameter counts (Table 15)")
        from src.models_dl import (ABLATION_CONFIGS, build_ablation_config, build_ann, build_cnn,
                                   build_lstm, build_mf_afen)
        counts = {"ANN / MLP": build_ann(17).count_params(), "1D-CNN": build_cnn(17).count_params(),
                  "LSTM": build_lstm(17).count_params(), "MF-AFEN": build_mf_afen().count_params()}
        counts.update({f"ablation {c}": build_ablation_config(c)[0].count_params() for c in ABLATION_CONFIGS})
        for k, v in counts.items():
            print(f"{k:40s} {v:>8,}")
        pd.Series(counts, name="n_params").to_csv(os.path.join(out, "parameter_counts.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None)
    ap.add_argument("--release", default=None)
    ap.add_argument("--skip-params", action="store_true")
    main(ap.parse_args())
