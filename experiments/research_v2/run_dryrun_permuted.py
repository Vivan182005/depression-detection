"""
Script: experiments/research_v2/run_dryrun_permuted.py
Step E1: Dry-run on PERMUTED labels only (R=2) for all 19 variants.
Every variant must run end-to-end with wall-clock timings; fixes any crashes.
Strictly label-blind: clinical labels are permuted within domain x gender strata.
Zero real labels observed.
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids
from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine
from data_loader import load_cohort_info, load_variant_features
from pipeline_engine import run_pipeline

VARIANTS = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19'
]

TIMINGS_FILE = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'dryrun_permuted_timings.json')

def main():
    print("======================================================================")
    print("Phase 6A - Step E1: Permuted Dry-Run (R=2 Replications across 19 Variants)")
    print("Execution mode: PERMUTED LABELS ONLY (Zero real clinical labels evaluated)")
    print("======================================================================")

    # 1. Load development cohort info
    cohort_df, _, groups, strata = load_cohort_info(permute=False)
    assert len(cohort_df) == 239, f"Expected 239 dev participants, got {len(cohort_df)}"
    assert_no_sealed_ids(cohort_df['participant_id'].values)

    print(f"Cohort loaded: {len(cohort_df)} participants ({sum(cohort_df['domain']=='woz')} WoZ, {sum(cohort_df['domain']=='avatar')} Avatar)")

    # 2. Check feature completeness for all 239
    print("\nVerifying feature readiness for 19 variants...")
    unready = []
    for vid in VARIANTS:
        try:
            X = load_variant_features(vid, cohort_df)
            if isinstance(X, np.ndarray):
                print(f"  {vid}: Matrix shape {X.shape}, dtype {X.dtype}")
            elif isinstance(X, list):
                print(f"  {vid}: Segment list of length {len(X)} (Sample shape {X[0].shape})")
            elif isinstance(X, dict):
                print(f"  {vid}: Stacking dict with {len(X)} constituent representations")
        except Exception as e:
            print(f"  {vid}: NOT READY ({e})")
            unready.append(vid)

    if unready:
        print(f"\n[WAIT] Variants not yet fully extracted: {unready}")
        print("Feature extraction is currently running in background. Please retry once complete.")
        sys.exit(2)

    print("\nAll 19 variants verified ready! Beginning R=2 permuted dry-run...")
    results = {}
    total_start = time.time()

    R = 2
    for vid in VARIANTS:
        print(f"\n--------------------------------------------------")
        print(f"Running Variant {vid} (R={R} permuted replications)...")
        print(f"--------------------------------------------------")

        X = load_variant_features(vid, cohort_df)
        variant_timings = []
        variant_aucs = []

        for r in range(1, R + 1):
            perm_seed = 8000 + r * 100 + int(vid.replace('V', ''))
            # Generate strictly permuted labels within domain x gender strata
            _, y_perm, _, _ = load_cohort_info(permute=True, permute_seed=perm_seed)

            t0 = time.time()
            res = run_pipeline(
                X=X,
                y=y_perm,
                groups=groups,
                seeds=(0, 1, 2, 3, 4),
                variant_id=vid,
                inner_cv_splits=4,
                outer_cv_splits=5,
                strata=None,  # prereg v1.1 sec 5.1: stratify outer folds on the binary label
                device='cpu'
            )
            elapsed = time.time() - t0

            variant_timings.append(elapsed)
            variant_aucs.append(res['auc'])
            print(f"  Rep {r}/{R}: Wall-clock = {elapsed:.2f}s | Permuted AUC = {res['auc']:.4f}")

        mean_time = float(np.mean(variant_timings))
        results[vid] = {
            'variant_id': vid,
            'replications': R,
            'timings_sec': variant_timings,
            'mean_timing_sec': mean_time,
            'permuted_aucs': variant_aucs,
            'mean_permuted_auc': float(np.mean(variant_aucs)),
            'status': 'SUCCESS'
        }
        print(f"  --> {vid} SUCCESS: Mean timing = {mean_time:.2f}s across {R} runs.")

    total_time = time.time() - total_start
    print("\n======================================================================")
    print(f"STEP E1 DRY-RUN COMPLETE: All 19 variants succeeded in {total_time/60:.2f} minutes.")
    print("======================================================================")

    # Save timing results
    summary = {
        'step': 'E1_DRY_RUN',
        'replications': R,
        'cohort_n': 239,
        'label_mode': 'PERMUTED_ONLY',
        'total_wall_clock_sec': total_time,
        'variants': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(TIMINGS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved execution timings to {TIMINGS_FILE}.")

if __name__ == '__main__':
    main()
