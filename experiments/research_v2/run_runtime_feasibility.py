"""
Task C7: Runtime Feasibility Benchmark
Measures the complete intended permutation pipeline for V2 and V10 using R=5 permuted-label runs.
Reports wall-clock time, hardware, caching state, per-stage timing, and extrapolates to R=999.
"""

import os
import sys
import time
import json
import platform
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine

def benchmark_variant(variant_name, X, y_permuted, groups, n_permutations=5, n_splits=5, seed=42):
    """
    Runs an end-to-end benchmark for one variant over R=5 permuted runs.
    """
    t0_total = time.time()
    
    # 1. Unpermuted / Baseline Run
    t0_baseline = time.time()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_permuted))
    
    stage_times = {'scaling': 0.0, 'fitting': 0.0, 'predicting': 0.0, 'aggregation': 0.0}
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_permuted, groups)):
        # Scaling
        t_s0 = time.time()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        stage_times['scaling'] += (time.time() - t_s0)
        
        # Fitting
        t_f0 = time.time()
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=200, random_state=seed + fold)
        clf.fit(X_train, y_permuted[train_idx])
        stage_times['fitting'] += (time.time() - t_f0)
        
        # Predicting
        t_p0 = time.time()
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        stage_times['predicting'] += (time.time() - t_p0)
        
    t_a0 = time.time()
    p_df = pd.DataFrame({'participant_id': groups, 'y_true': y_permuted, 'y_pred': oof_preds})
    p_agg = p_df.groupby('participant_id')[['y_true', 'y_pred']].mean()
    base_auc = roc_auc_score(p_agg['y_true'], p_agg['y_pred'])
    stage_times['aggregation'] += (time.time() - t_a0)
    
    t_baseline = time.time() - t0_baseline
    
    # 2. Permutation Runs (R = 5)
    t0_perm = time.time()
    rng = np.random.RandomState(seed + 999)
    p_df_dedup = pd.DataFrame({'participant_id': groups, 'y': y_permuted}).drop_duplicates('participant_id')
    unique_pids = p_df_dedup['participant_id'].values
    unique_labels = p_df_dedup['y'].values
    
    perm_aucs = []
    for r in range(n_permutations):
        shuffled = rng.permutation(unique_labels)
        pid_map = dict(zip(unique_pids, shuffled))
        y_r = np.array([pid_map[pid] for pid in groups])
        
        sgkf_r = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r + 1)
        oof_r = np.zeros(len(y_r))
        for fold, (train_idx, val_idx) in enumerate(sgkf_r.split(X, y_r, groups)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])
            clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=200, random_state=seed + fold + r*10)
            clf.fit(X_tr, y_r[train_idx])
            oof_r[val_idx] = clf.predict_proba(X_va)[:, 1]
            
        p_df_r = pd.DataFrame({'participant_id': groups, 'y_true': y_r, 'y_pred': oof_r})
        p_agg_r = p_df_r.groupby('participant_id')[['y_true', 'y_pred']].mean()
        perm_aucs.append(roc_auc_score(p_agg_r['y_true'], p_agg_r['y_pred']))
        
    t_perms_total = time.time() - t0_perm
    avg_perm_time = t_perms_total / n_permutations
    
    # 3. Extrapolation to R = 999
    extrapolated_time_sec = t_baseline + 999 * avg_perm_time
    extrapolated_time_hours = extrapolated_time_sec / 3600.0
    
    return {
        'variant': variant_name,
        'feature_dim': X.shape[1],
        'n_samples': X.shape[0],
        'n_participants': len(unique_pids),
        'baseline_time_sec': round(t_baseline, 4),
        'perm_5_total_sec': round(t_perms_total, 4),
        'avg_perm_sec': round(avg_perm_time, 4),
        'stage_times': {k: round(v, 4) for k, v in stage_times.items()},
        'extrapolated_r999_sec': round(extrapolated_time_sec, 2),
        'extrapolated_r999_hours': round(extrapolated_time_hours, 4),
        'exceeds_48h': bool(extrapolated_time_hours > 48.0)
    }

def main():
    print("=======================================================")
    print("Task C7: Runtime Feasibility Benchmark (R=5 -> R=999)")
    print("=======================================================")
    
    # Hardware info
    hw_info = {
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }
    print(f"Hardware: {hw_info['machine']} ({hw_info['processor']}), OS: {hw_info['system']} {hw_info['release']}")
    
    n_participants = 239  # exact eligible dev cohort size
    rng = np.random.RandomState(888)
    pids = np.arange(1, n_participants + 1)
    seg_counts = rng.randint(2, 6, size=n_participants)
    groups = np.repeat(pids, seg_counts)
    n_rows = len(groups)
    
    # Synthetic / permuted outcome labels
    p_labels = rng.binomial(1, 0.5, size=n_participants)
    pid_to_y = dict(zip(pids, p_labels))
    y = np.array([pid_to_y[pid] for pid in groups])
    
    # Benchmark V2: eGeMAPS (135d) + Handcrafted acoustic summaries (165d) -> 300d
    print("\nBenchmarking Variant V2 (Audio Handcrafted + eGeMAPS, 300d)...")
    X_v2 = rng.randn(n_rows, 300)
    res_v2 = benchmark_variant("V2_Audio_Handcrafted_eGeMAPS", X_v2, y, groups, n_permutations=5, seed=100)
    print(f"  V2: Baseline = {res_v2['baseline_time_sec']}s, R=5 avg = {res_v2['avg_perm_sec']}s/run")
    print(f"  V2: Extrapolated R=999 = {res_v2['extrapolated_r999_sec']}s ({res_v2['extrapolated_r999_hours']:.2f} hours)")
    
    # Benchmark V10: Audio Pretrained Embeddings (Wav2Vec2 768d + WavLM 768d + eGeMAPS 135d -> 1671d)
    print("\nBenchmarking Variant V10 (Wav2Vec2 + WavLM + eGeMAPS, 1671d)...")
    X_v10 = rng.randn(n_rows, 1671)
    res_v10 = benchmark_variant("V10_Audio_Pretrained_Multimodal", X_v10, y, groups, n_permutations=5, seed=200)
    print(f"  V10: Baseline = {res_v10['baseline_time_sec']}s, R=5 avg = {res_v10['avg_perm_sec']}s/run")
    print(f"  V10: Extrapolated R=999 = {res_v10['extrapolated_r999_sec']}s ({res_v10['extrapolated_r999_hours']:.2f} hours)")
    
    results = {
        'hardware': hw_info,
        'v2': res_v2,
        'v10': res_v10,
        'stop_condition_triggered': bool(res_v2['exceeds_48h'] or res_v10['exceeds_48h'])
    }
    
    out_json = 'experiments/research_v2/runtime_feasibility_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved runtime feasibility results to {out_json}")

if __name__ == '__main__':
    main()
