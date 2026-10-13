"""
Experiment Suite 1 & 2: Modality Ablation & Classical ML Benchmarking Suite.
Runs strictly on development data (N=146, 144 unique participants) using
StratifiedGroupKFold(5 splits, random_state=42) grouped on participant_id.

Evaluates:
- Modality Ablations: Acoustic (154d), Wav2Vec (768d), BERT (768d), and combinations.
- Classical Model Families:
  - Logistic Regression (L2, tuned C)
  - Linear SVM & RBF Kernel SVM
  - Random Forest & Extra Trees
  - HistGradientBoostingClassifier (LightGBM equivalent)
Saves comparison tables, out-of-fold metrics, and model benchmarks.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_accuracy_v2")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from experiments.capstone_improvement_v1.train_improvements import load_dataset_features


def compute_metrics(y_true, y_pred, y_prob=None):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    y_bin_true = (y_true >= 1).astype(int)
    y_bin_pred = (y_pred >= 1).astype(int)
    bin_acc = float(accuracy_score(y_bin_true, y_bin_pred) * 100.0)
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'weighted_f1': weighted_f1,
        'recall_0': float(r_per[0]),
        'recall_1': float(r_per[1]),
        'recall_2': float(r_per[2]),
        'precision_0': float(p_per[0]),
        'precision_1': float(p_per[1]),
        'precision_2': float(p_per[2]),
        'f1_0': float(f_per[0]),
        'f1_1': float(f_per[1]),
        'f1_2': float(f_per[2]),
        'binary_accuracy': bin_acc,
        'binary_f1': bin_f1,
        'confusion_matrix': cm
    }


def evaluate_model_cv(model_fn, X_dict, y, pids, modality_keys=['ac', 'w2v', 'bert'], n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = [int(p) for p in pids]

    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), 3), dtype=np.float32)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dict['ac'], y, groups=groups)):
        # Construct modality feature matrices strictly fit on train_idx
        X_tr_parts = []
        X_va_parts = []
        for k in modality_keys:
            feat_tr = X_dict[k][train_idx]
            feat_va = X_dict[k][val_idx]
            scaler = StandardScaler().fit(feat_tr)
            X_tr_parts.append(scaler.transform(feat_tr))
            X_va_parts.append(scaler.transform(feat_va))

        X_tr = np.hstack(X_tr_parts)
        X_va = np.hstack(X_va_parts)
        y_tr = y[train_idx]
        y_va = y[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        preds_va = model.predict(X_va)
        oof_preds[val_idx] = preds_va

        if hasattr(model, "predict_proba"):
            probs_va = model.predict_proba(X_va)
            oof_probs[val_idx] = probs_va

        m = compute_metrics(y_va, preds_va)
        fold_metrics.append(m)

    # Compute overall OOF metrics and mean/std across folds
    oof_m = compute_metrics(y, oof_preds)

    summary = {
        'oof_accuracy': oof_m['accuracy'],
        'oof_macro_f1': oof_m['macro_f1'],
        'oof_balanced_accuracy': oof_m['balanced_accuracy'],
        'oof_recall_0': oof_m['recall_0'],
        'oof_recall_1': oof_m['recall_1'],
        'oof_recall_2': oof_m['recall_2'],
        'oof_binary_f1': oof_m['binary_f1'],
        'oof_confusion_matrix': oof_m['confusion_matrix'],
        'cv_acc_mean': float(np.mean([f['accuracy'] for f in fold_metrics])),
        'cv_acc_std': float(np.std([f['accuracy'] for f in fold_metrics])),
        'cv_macro_f1_mean': float(np.mean([f['macro_f1'] for f in fold_metrics])),
        'cv_macro_f1_std': float(np.std([f['macro_f1'] for f in fold_metrics])),
        'cv_bal_acc_mean': float(np.mean([f['balanced_accuracy'] for f in fold_metrics])),
        'cv_bal_acc_std': float(np.std([f['balanced_accuracy'] for f in fold_metrics])),
        'cv_rec0_mean': float(np.mean([f['recall_0'] for f in fold_metrics])),
        'cv_rec1_mean': float(np.mean([f['recall_1'] for f in fold_metrics])),
        'cv_rec2_mean': float(np.mean([f['recall_2'] for f in fold_metrics])),
        'cv_bin_f1_mean': float(np.mean([f['binary_f1'] for f in fold_metrics]))
    }
    return summary, oof_preds, oof_probs


def main():
    print("==========================================================================")
    print("  EXPERIMENT SUITE 1 & 2: MODALITY ABLATIONS & CLASSICAL MODEL BENCHMARK  ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, _ = split_joint_stratified(master_df, verbose=False)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    d_dev = load_dataset_features(dev_df, fs, clean_labels=True)
    X_dict = {
        'ac': d_dev['ac'],
        'w2v': d_dev['w2v'],
        'bert': d_dev['bert']
    }
    y = d_dev['y_class']
    pids = d_dev['pids']
    sids = d_dev['sids']

    print(f"Eligible Development Data: N={len(y)} sessions, {len(set(pids))} unique participants")
    print(f"Class counts: {np.bincount(y, minlength=3)}")

    # -------------------------------------------------------------
    # 1. Modality Ablation Experiments (using tuned Logistic Regression)
    # -------------------------------------------------------------
    print("\n--- Part 1: Modality Ablations (Logistic Regression C=0.1) ---")
    modality_configs = [
        ('Acoustic Only (154d)', ['ac']),
        ('Wav2Vec2 Only (768d)', ['w2v']),
        ('BERT Text Only (768d)', ['bert']),
        ('Acoustic + Wav2Vec2 (922d)', ['ac', 'w2v']),
        ('Acoustic + BERT (922d)', ['ac', 'bert']),
        ('Wav2Vec2 + BERT (1536d)', ['w2v', 'bert']),
        ('Tri-Modal Full (1690d)', ['ac', 'w2v', 'bert'])
    ]

    modality_results = []
    for name, mods in modality_configs:
        m_fn = lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        summary, _, _ = evaluate_model_cv(m_fn, X_dict, y, pids, modality_keys=mods)
        summary['config'] = name
        summary['modalities'] = "+".join(mods)
        modality_results.append(summary)
        print(f"  {name:28s} | Acc: {summary['cv_acc_mean']:.2f} ± {summary['cv_acc_std']:.2f}% | Macro-F1: {summary['cv_macro_f1_mean']:.2f} ± {summary['cv_macro_f1_std']:.2f}% | Bal-Acc: {summary['cv_bal_acc_mean']:.2f}%")

    df_mod = pd.DataFrame(modality_results)
    df_mod.to_csv(os.path.join(EXP_DIR, "modality_ablation_benchmarks.csv"), index=False)

    # -------------------------------------------------------------
    # 2. Comprehensive Model Family Benchmark (on Tri-Modal features)
    # -------------------------------------------------------------
    print("\n--- Part 2: Model Family Benchmarks on Tri-Modal Features (1690d) ---")
    models_to_test = [
        # Linear & Regularized Models
        ('LogReg (C=1.0, unweighted)', lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ('LogReg (C=0.1, unweighted)', lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ('LogReg (C=0.01, unweighted)', lambda: LogisticRegression(C=0.01, max_iter=1000, random_state=42)),
        ('LogReg (C=0.1, balanced)', lambda: LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)),
        ('RidgeClassifier (alpha=10.0)', lambda: RidgeClassifier(alpha=10.0, random_state=42)),
        ('RidgeClassifier (alpha=100.0)', lambda: RidgeClassifier(alpha=100.0, random_state=42)),

        # Support Vector Machines
        ('LinearSVC (C=0.01)', lambda: LinearSVC(C=0.01, max_iter=2000, random_state=42)),
        ('LinearSVC (C=0.1)', lambda: LinearSVC(C=0.1, max_iter=2000, random_state=42)),
        ('SVC RBF (C=1.0, gamma=scale)', lambda: SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
        ('SVC RBF (C=2.0, gamma=scale)', lambda: SVC(C=2.0, kernel='rbf', probability=True, random_state=42)),
        ('SVC RBF (C=1.0, balanced)', lambda: SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),

        # Tree Ensembles
        ('RandomForest (n=200, depth=6)', lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42)),
        ('RandomForest (n=200, depth=10)', lambda: RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)),
        ('RandomForest (n=200, balanced)', lambda: RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)),
        ('ExtraTrees (n=200, depth=6)', lambda: ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42)),
        ('ExtraTrees (n=200, balanced)', lambda: ExtraTreesClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)),

        # Gradient Boosted Trees (HistGradientBoosting / LightGBM equiv)
        ('HistGradientBoosting (lr=0.05, d=3)', lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=42)),
        ('HistGradientBoosting (lr=0.05, d=5)', lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_depth=5, l2_regularization=2.0, random_state=42)),
        ('HistGradientBoosting (balanced)', lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, class_weight='balanced', random_state=42))
    ]

    model_benchmarks = []
    best_acc = 0.0
    best_model_name = ""

    for name, m_fn in models_to_test:
        summary, oof_preds, oof_probs = evaluate_model_cv(m_fn, X_dict, y, pids, modality_keys=['ac', 'w2v', 'bert'])
        summary['model_name'] = name
        model_benchmarks.append(summary)

        print(f"  {name:38s} | Acc: {summary['cv_acc_mean']:.2f} ± {summary['cv_acc_std']:.2f}% | Macro-F1: {summary['cv_macro_f1_mean']:.2f} ± {summary['cv_macro_f1_std']:.2f}% | Bal-Acc: {summary['cv_bal_acc_mean']:.2f}% | Rec1: {summary['cv_rec1_mean']:.1f}% | Rec2: {summary['cv_rec2_mean']:.1f}%")

        if summary['cv_acc_mean'] > best_acc and summary['cv_bal_acc_mean'] > 40.0:
            best_acc = summary['cv_acc_mean']
            best_model_name = name

    df_models = pd.DataFrame(model_benchmarks)
    df_models.to_csv(os.path.join(EXP_DIR, "classical_models_cv_benchmarks.csv"), index=False)

    print(f"\nTop Classical Model by Accuracy (with Bal-Acc > 40%): {best_model_name} (Acc: {best_acc:.2f}%)")

    # Save summary json
    with open(os.path.join(EXP_DIR, "modality_and_classical_summary.json"), "w") as f:
        json.dump({
            'modality_ablations': modality_results,
            'classical_models': model_benchmarks
        }, f, indent=2)

    print("Saved all benchmark CSVs and JSON summary to experiments/capstone_accuracy_v2/ successfully!")


if __name__ == "__main__":
    main()
