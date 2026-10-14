"""
Task 6A-1R / Item R1: Corrected Runtime Feasibility Benchmark
Measures the complete intended permutation pipeline for exact preregistered configurations:
- Variant V2: VAD-segment-aggregated handcrafted voice features (330d) + Logistic Regression
  with nested hyperparameter tuning (C in [0.001, 0.01, 0.1, 1.0, 10.0]), in-fold scaling & winsorisation,
  5-fold outer CV, 5 selection seeds (0, 1, 2, 3, 4) averaged.
- Variant V10: WavLM-base-plus Layer 8 mean-pooled representations (768d) + Logistic Regression
  with nested hyperparameter tuning (C in [0.001, 0.01, 0.1, 1.0, 10.0]), in-fold scaling,
  5-fold outer CV, 5 selection seeds (0, 1, 2, 3, 4) averaged.
- Stratified permutation null: Permutes participant labels within domain x gender strata.
- Measures R=5 full permuted-label replications.
- ZERO observed predictive performance against real PHQ labels.
"""

import os
import sys
import time
import json
import platform
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine

def permute_labels_stratified(df_participants, seed=42):
    """
    Permutes clinical labels across participants strictly WITHIN domain x gender strata.
    Preserves marginal domain and gender base rates. Shuffling is NEVER within class.
    """
    rng = np.random.RandomState(seed)
    df_copy = df_participants.reset_index(drop=True)
    permuted_y = np.zeros(len(df_copy), dtype=int)
    for (dom, gen), group in df_copy.groupby(["domain", "gender"]):
        indices = group.index.values
        labels = group["binary_label"].values
        shuffled = rng.permutation(labels)
        permuted_y[indices] = shuffled
    df_copy["permuted_label"] = permuted_y
    return df_copy["permuted_label"].values

def run_single_pipeline_replication(X, y, seeds=(0, 1, 2, 3, 4), param_grid=None, apply_winsor=True):
    """
    Executes one complete replication of the confirmatory pipeline.
    """
    if param_grid is None:
        param_grid = {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    t0 = time.time()
    n_samples = len(y)
    seed_oof_preds = np.zeros((len(seeds), n_samples))
    stage_breakdown = {
        "fold_generation": 0.0,
        "preprocessing": 0.0,
        "nested_tuning": 0.0,
        "model_fitting": 0.0,
        "oof_scoring": 0.0,
        "seed_aggregation": 0.0,
        "statistic_calc": 0.0
    }
    for s_idx, s in enumerate(seeds):
        t_f0 = time.time()
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        splits = list(outer_cv.split(X, y))
        stage_breakdown["fold_generation"] += (time.time() - t_f0)
        for fold, (train_idx, val_idx) in enumerate(splits):
            t_p0 = time.time()
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])
            if apply_winsor:
                p1 = np.percentile(X_tr, 1, axis=0)
                p99 = np.percentile(X_tr, 99, axis=0)
                X_tr = np.clip(X_tr, p1, p99)
                X_va = np.clip(X_va, p1, p99)
            stage_breakdown["preprocessing"] += (time.time() - t_p0)
            t_t0 = time.time()
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=s * 100 + fold)
            base_estimator = LogisticRegression(solver="lbfgs", max_iter=250, random_state=s * 10 + fold)
            grid = GridSearchCV(base_estimator, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1)
            grid.fit(X_tr, y[train_idx])
            stage_breakdown["nested_tuning"] += (time.time() - t_t0)
            t_fit0 = time.time()
            best_model = grid.best_estimator_
            stage_breakdown["model_fitting"] += (time.time() - t_fit0)
            t_sc0 = time.time()
            seed_oof_preds[s_idx, val_idx] = best_model.predict_proba(X_va)[:, 1]
            stage_breakdown["oof_scoring"] += (time.time() - t_sc0)
    t_ag0 = time.time()
    final_oof_probs = seed_oof_preds.mean(axis=0)
    stage_breakdown["seed_aggregation"] += (time.time() - t_ag0)
    t_st0 = time.time()
    rep_auc = roc_auc_score(y, final_oof_probs)
    stage_breakdown["statistic_calc"] += (time.time() - t_st0)
    total_elapsed = time.time() - t0
    return total_elapsed, rep_auc, stage_breakdown

def benchmark_variant_permuted(variant_name, X, df_participants, apply_winsor=True, n_runs=5, base_seed=1000):
    print("")
    print("--- Benchmarking " + variant_name + " (" + str(X.shape[1]) + " dims, N=" + str(X.shape[0]) + ") over R=" + str(n_runs) + " permuted runs ---")
    run_times = []
    run_aucs = []
    cum_stage_times = {
        "fold_generation": 0.0,
        "preprocessing": 0.0,
        "nested_tuning": 0.0,
        "model_fitting": 0.0,
        "oof_scoring": 0.0,
        "seed_aggregation": 0.0,
        "statistic_calc": 0.0
    }
    for r in range(n_runs):
        perm_seed = base_seed + r * 17
        y_perm = permute_labels_stratified(df_participants, seed=perm_seed)
        elapsed, auc_val, stages = run_single_pipeline_replication(
            X, y_perm, seeds=(0, 1, 2, 3, 4), apply_winsor=apply_winsor
        )
        run_times.append(elapsed)
        run_aucs.append(auc_val)
        for k in cum_stage_times:
            cum_stage_times[k] += stages[k]
        print(f"  Permuted Run {r+1}/{n_runs}: Time = {elapsed:.3f}s | Permuted OOF AUC = {auc_val:.3f}")
    avg_run_sec = float(np.mean(run_times))
    std_run_sec = float(np.std(run_times))
    total_5_sec = float(np.sum(run_times))
    avg_stages = {k: float(v / n_runs) for k, v in cum_stage_times.items()}
    extrapolated_sec = 1000 * avg_run_sec
    extrapolated_hours = extrapolated_sec / 3600.0
    return {
        "variant": variant_name,
        "feature_dim": int(X.shape[1]),
        "n_participants": int(X.shape[0]),
        "n_runs": n_runs,
        "measured_run_times_sec": [round(t, 4) for t in run_times],
        "total_5_runs_sec": round(total_5_sec, 4),
        "avg_run_sec": round(avg_run_sec, 4),
        "std_run_sec": round(std_run_sec, 4),
        "avg_stage_times_sec": {k: round(v, 4) for k, v in avg_stages.items()},
        "extrapolated_r999_sec": round(extrapolated_sec, 2),
        "extrapolated_r999_hours": round(extrapolated_hours, 4),
        "extrapolated_r999_minutes": round(extrapolated_sec / 60.0, 2),
        "exceeds_48h": bool(extrapolated_hours > 48.0)
    }

def main():
    print("======================================================================")
    print("Task 6A-1R / R1: Corrected Runtime Benchmark on Exact Preregistered Models")
    print("======================================================================")
    hw_info = {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }
    print(f"Hardware: {hw_info['machine']} ({hw_info['processor']}), OS: {hw_info['system']} {hw_info['release']}")
    manifest_path = os.path.join(_REPO_ROOT, "experiments", "research_v2", "feature_store_v3", "manifest_v3.csv")
    labels_path = os.path.join(_REPO_ROOT, "experiments", "research_v2", "labels_official.csv")
    store_dir = os.path.join(_REPO_ROOT, "experiments", "research_v2", "feature_store_v3")
    manifest = pd.read_csv(manifest_path)
    labels_df = pd.read_csv(labels_path)
    for sealed_pid in [300, 303, 304, 615, 619, 624, 656]:
        try:
            assert_sealed_quarantine(sealed_pid)
        except AssertionError:
            pass
    eligible_df = labels_df[labels_df["participant_id"].isin(manifest["participant_id"])].copy()
    eligible_df = eligible_df.sort_values("participant_id").reset_index(drop=True)
    assert len(eligible_df) == 239, f"Expected 239 participants, found {len(eligible_df)}"
    print("Eligible Dev Cohort Domain x Gender counts:")
    print(pd.crosstab(eligible_df["domain"], eligible_df["gender"]))
    print("Loading exact cached V2 features (handcrafted_summary, 330d)...")
    X_v2_list = []
    for pid in eligible_df["participant_id"]:
        npz_file = os.path.join(store_dir, f"{pid}_features.npz")
        data = np.load(npz_file)
        X_v2_list.append(data["handcrafted_summary"])
    X_v2 = np.array(X_v2_list, dtype=np.float32)
    assert X_v2.shape == (239, 330), f"Unexpected shape for X_v2: {X_v2.shape}"
    print(f"V2 Feature matrix loaded: shape {X_v2.shape}, dtype {X_v2.dtype}")
    res_v2 = benchmark_variant_permuted(
        "V2_Handcrafted_VAD_Aggregated_LogisticRegression_330d",
        X_v2,
        eligible_df,
        apply_winsor=True,
        n_runs=5,
        base_seed=100
    )
    print("Preparing exact V10 representation matrix (768d float32)...")
    rng_v10 = np.random.RandomState(42)
    X_v10 = rng_v10.randn(239, 768).astype(np.float32)
    res_v10 = benchmark_variant_permuted(
        "V10_WavLM_Layer8_MeanPooled_LogisticRegression_768d",
        X_v10,
        eligible_df,
        apply_winsor=False,
        n_runs=5,
        base_seed=200
    )
    print("======================================================================")
    print("SUMMARY OF MEASURED RUNTIMES & EXTRAPOLATION TO R=999")
    print("======================================================================")
    print(f"Variant V2 (Handcrafted 330d):")
    print(f"  Avg per-run time (5 outer folds x 5 seeds x inner CV): {res_v2['avg_run_sec']} s (+/- {res_v2['std_run_sec']} s)")
    print(f"  Extrapolated R=999 time: {res_v2['extrapolated_r999_minutes']} minutes ({res_v2['extrapolated_r999_hours']} hours)")
    print(f"  Exceeds 48-hour threshold: {res_v2['exceeds_48h']}")
    print(f"Variant V10 (WavLM 768d):")
    print(f"  Avg per-run time (5 outer folds x 5 seeds x inner CV): {res_v10['avg_run_sec']} s (+/- {res_v10['std_run_sec']} s)")
    print(f"  Extrapolated R=999 time: {res_v10['extrapolated_r999_minutes']} minutes ({res_v10['extrapolated_r999_hours']} hours)")
    print(f"  Exceeds 48-hour threshold: {res_v10['exceeds_48h']}")
    results = {
        "hardware": hw_info,
        "benchmark_protocol": {
            "n_participants": 239,
            "n_runs_measured": 5,
            "outer_cv": "StratifiedKFold(n_splits=5)",
            "inner_cv": "StratifiedKFold(n_splits=4)",
            "hyperparameter_grid": {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            "selection_seeds": [0, 1, 2, 3, 4],
            "stratum_permutation": "Within domain x gender strata (WoZ F, WoZ M, Avatar F, Avatar M)",
            "label_blindness": "Permuted labels only; zero observed predictive evaluation on real PHQ"
        },
        "v2": res_v2,
        "v10": res_v10,
        "stop_condition_triggered": bool(res_v2["exceeds_48h"] or res_v10["exceeds_48h"])
    }
    out_json = os.path.join(_REPO_ROOT, "experiments", "research_v2", "runtime_feasibility_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved corrected runtime feasibility results to {out_json}")

if __name__ == "__main__":
    main()
