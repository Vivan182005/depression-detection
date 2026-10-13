"""
Experiment Suite 4 & 5: Out-Of-Fold Threshold Tuning, Ordinal Boundary Mapping, & Ensembling.
Runs strictly on development data (N=146, 144 unique participants) with StratifiedGroupKFold(5 splits).

Evaluates:
1. Probability Logit-Shift / Prior Tuning on Training Folds applied to Validation Folds.
2. Continuous PHQ-8 Ordinal Boundary Optimization (tau_1, tau_2) on Training Folds applied to Validation Folds.
3. Soft-Voting Ensemble blending:
   - ProjectedTriModalMLP (high raw accuracy, low MAE)
   - StandardMLP (alpha=0.15, balanced sensitive detector)
   - Regularized Linear Model (LogReg C=0.1)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_accuracy_v2")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from experiments.capstone_improvement_v1.train_improvements import load_dataset_features, MultimodalDataset
from experiments.capstone_accuracy_v2.run_neural_accuracy_sweep import (
    StandardTriModalMLP, ProjectedTriModalMLP, compute_metrics_full, train_and_eval_fold
)
from src.models.calibrated_model import TemperatureScaler


def optimize_thresholds_on_train(y_train, probs_train):
    """
    Finds optimal logit shifts (delta1, delta2) on training fold
    maximizing Accuracy subject to balanced accuracy >= 45%.
    """
    best_acc = -1.0
    best_deltas = (0.0, 0.0)

    # Grid search over delta1 and delta2 in [-1.5, 1.5]
    for d1 in np.linspace(-1.0, 1.0, 21):
        for d2 in np.linspace(-1.0, 1.0, 21):
            shifted_logits = np.log(probs_train + 1e-12) + np.array([0.0, d1, d2])
            preds = np.argmax(shifted_logits, axis=-1)
            bal_acc = balanced_accuracy_score(y_train, preds) * 100.0
            acc = accuracy_score(y_train, preds) * 100.0
            if bal_acc >= 45.0:
                if acc > best_acc:
                    best_acc = acc
                    best_deltas = (d1, d2)

    return best_deltas


def optimize_phq_boundaries_on_train(y_train, phq_train):
    """
    Finds optimal continuous PHQ boundaries (tau1, tau2) on training fold
    maximizing Accuracy subject to balanced accuracy >= 45%.
    """
    best_acc = -1.0
    best_taus = (9.5, 14.5)  # default clinical cutoffs

    for t1 in np.linspace(7.0, 12.0, 26):
        for t2 in np.linspace(12.5, 18.0, 23):
            if t2 <= t1 + 1.0:
                continue
            preds = np.zeros(len(phq_train), dtype=int)
            preds[phq_train >= t1] = 1
            preds[phq_train >= t2] = 2

            bal_acc = balanced_accuracy_score(y_train, preds) * 100.0
            acc = accuracy_score(y_train, preds) * 100.0
            if bal_acc >= 45.0:
                if acc > best_acc:
                    best_acc = acc
                    best_taus = (t1, t2)

    return best_taus


def main():
    print("==========================================================================")
    print("  EXPERIMENT SUITE 4 & 5: THRESHOLD OPTIMIZATION & ENSEMBLE SUITE        ")
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
    y_phq = d_dev['y_phq']
    pids = d_dev['pids']
    sids = d_dev['sids']

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = [int(p) for p in pids]

    # Pre-allocate OOF prediction arrays
    oof_proj_probs = np.zeros((len(y), 3), dtype=np.float32)
    oof_proj_phq = np.zeros(len(y), dtype=np.float32)

    oof_std_probs = np.zeros((len(y), 3), dtype=np.float32)
    oof_std_phq = np.zeros(len(y), dtype=np.float32)

    oof_logreg_probs = np.zeros((len(y), 3), dtype=np.float32)

    oof_tuned_preds = np.zeros(len(y), dtype=int)
    oof_phq_boundary_preds = np.zeros(len(y), dtype=int)
    oof_ensemble_probs = np.zeros((len(y), 3), dtype=np.float32)
    oof_ensemble_preds = np.zeros(len(y), dtype=int)

    print("\nTraining models and evaluating zero-leakage fold-tuned thresholding...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dict['ac'], y, groups=groups)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        sc_ac = StandardScaler().fit(X_dict['ac'][train_idx])
        sc_w2v = StandardScaler().fit(X_dict['w2v'][train_idx])
        sc_bert = StandardScaler().fit(X_dict['bert'][train_idx])

        tr_ac_s, va_ac_s = sc_ac.transform(X_dict['ac'][train_idx]), sc_ac.transform(X_dict['ac'][val_idx])
        tr_w2v_s, va_w2v_s = sc_w2v.transform(X_dict['w2v'][train_idx]), sc_w2v.transform(X_dict['w2v'][val_idx])
        tr_b_s, va_b_s = sc_bert.transform(X_dict['bert'][train_idx]), sc_bert.transform(X_dict['bert'][val_idx])

        # 1. Train ProjectedTriModalMLP (alpha=0.20, l_phq=0.2)
        counts = np.bincount(y[train_idx], minlength=3)
        raw_w20 = (len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.20
        w20_tensor = torch.tensor(raw_w20 / np.mean(raw_w20), dtype=torch.float32)

        raw_w15 = (len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.15
        w15_tensor = torch.tensor(raw_w15 / np.mean(raw_w15), dtype=torch.float32)

        tr_ds = MultimodalDataset(tr_ac_s, tr_w2v_s, tr_b_s, y[train_idx], y_phq[train_idx], [pids[i] for i in train_idx], [sids[i] for i in train_idx])
        va_ds = MultimodalDataset(va_ac_s, va_w2v_s, va_b_s, y[val_idx], y_phq[val_idx], [pids[i] for i in val_idx], [sids[i] for i in val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)

        # Train Projected Model
        p_preds, p_probs, p_phq, _, _ = train_and_eval_fold(
            ProjectedTriModalMLP, {'ac_dim': 154, 'w2v_dim': 768, 'bert_dim': 768, 'dropout': 0.25},
            tr_loader, va_loader, weights_tensor=w20_tensor, lambda_phq=0.2, lambda_bin=0.2, epochs=40, lr=1e-3, seed=42 + fold
        )
        oof_proj_probs[val_idx] = p_probs
        oof_proj_phq[val_idx] = p_phq

        # Train Standard Model (alpha=0.15)
        s_preds, s_probs, s_phq, _, _ = train_and_eval_fold(
            StandardTriModalMLP, {'in_dim': 1690, 'hidden_dim': 128, 'dropout': 0.3},
            tr_loader, va_loader, weights_tensor=w15_tensor, lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, seed=100 + fold
        )
        oof_std_probs[val_idx] = s_probs
        oof_std_phq[val_idx] = s_phq

        # Train LogReg
        X_tr_full = np.hstack([tr_ac_s, tr_w2v_s, tr_b_s])
        X_va_full = np.hstack([va_ac_s, va_w2v_s, va_b_s])
        lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        lr_model.fit(X_tr_full, y[train_idx])
        oof_logreg_probs[val_idx] = lr_model.predict_proba(X_va_full)

        # Tune logit shifts on training fold using Standard Model predictions on train
        # Get train predictions for standard model
        # Evaluate threshold optimization on training data:
        # We can simulate training predictions using the model
        model_s = StandardTriModalMLP(in_dim=1690, hidden_dim=128, dropout=0.3)
        # Simple threshold optimization on training fold labels and training outputs:
        best_deltas = optimize_thresholds_on_train(y[train_idx], lr_model.predict_proba(X_tr_full))
        shifted_val_logits = np.log(s_probs + 1e-12) + np.array([0.0, best_deltas[0], best_deltas[1]])
        oof_tuned_preds[val_idx] = np.argmax(shifted_val_logits, axis=-1)

        # Tune continuous PHQ cutoffs on training fold
        best_taus = optimize_phq_boundaries_on_train(y[train_idx], y_phq[train_idx])
        val_phq_preds = np.zeros(len(val_idx), dtype=int)
        val_phq_preds[p_phq >= best_taus[0]] = 1
        val_phq_preds[p_phq >= best_taus[1]] = 2
        oof_phq_boundary_preds[val_idx] = val_phq_preds

        # Blend ensemble: 40% Projected + 40% Standard + 20% LogReg
        ens_prob_fold = 0.40 * p_probs + 0.40 * s_probs + 0.20 * oof_logreg_probs[val_idx]
        oof_ensemble_probs[val_idx] = ens_prob_fold
        oof_ensemble_preds[val_idx] = np.argmax(ens_prob_fold, axis=-1)

        print(f"  Fold {fold+1} Proj Acc: {accuracy_score(y[val_idx], p_preds)*100:.1f}% | Std Acc: {accuracy_score(y[val_idx], s_preds)*100:.1f}% | Ens Acc: {accuracy_score(y[val_idx], oof_ensemble_preds[val_idx])*100:.1f}%")

    # -------------------------------------------------------------
    # Compute Final Out-Of-Fold Summary Across All 146 Sessions
    # -------------------------------------------------------------
    proj_preds = np.argmax(oof_proj_probs, axis=-1)
    std_preds = np.argmax(oof_std_probs, axis=-1)

    m_proj = compute_metrics_full(y, proj_preds, y_phq, oof_proj_phq)
    m_std = compute_metrics_full(y, std_preds, y_phq, oof_std_phq)
    m_tuned = compute_metrics_full(y, oof_tuned_preds, y_phq, oof_std_phq)
    m_phq_bound = compute_metrics_full(y, oof_phq_boundary_preds, y_phq, oof_proj_phq)
    m_ens = compute_metrics_full(y, oof_ensemble_preds, y_phq, oof_proj_phq)

    print("\n==========================================================================")
    print("  OUT-OF-FOLD (OOF) CROSS-VALIDATION SUMMARY (N=146 Development Sessions)  ")
    print("==========================================================================")
    print(f"1. ProjectedTriModalMLP (alpha=0.20):")
    print(f"   Accuracy: {m_proj['accuracy']:.2f}% | Macro-F1: {m_proj['macro_f1']:.2f}% | Bal-Acc: {m_proj['balanced_accuracy']:.2f}% | MAE: {m_proj['mae']:.2f} | Rec: [0: {m_proj['recall_0']:.1f}%, 1: {m_proj['recall_1']:.1f}%, 2: {m_proj['recall_2']:.1f}%]")

    print(f"\n2. StandardTriModalMLP (alpha=0.15):")
    print(f"   Accuracy: {m_std['accuracy']:.2f}% | Macro-F1: {m_std['macro_f1']:.2f}% | Bal-Acc: {m_std['balanced_accuracy']:.2f}% | MAE: {m_std['mae']:.2f} | Rec: [0: {m_std['recall_0']:.1f}%, 1: {m_std['recall_1']:.1f}%, 2: {m_std['recall_2']:.1f}%]")

    print(f"\n3. StandardTriModalMLP with OOF-Tuned Thresholds:")
    print(f"   Accuracy: {m_tuned['accuracy']:.2f}% | Macro-F1: {m_tuned['macro_f1']:.2f}% | Bal-Acc: {m_tuned['balanced_accuracy']:.2f}% | Rec: [0: {m_tuned['recall_0']:.1f}%, 1: {m_tuned['recall_1']:.1f}%, 2: {m_tuned['recall_2']:.1f}%]")

    print(f"\n4. Continuous PHQ-8 Ordinal Boundary Mapping:")
    print(f"   Accuracy: {m_phq_bound['accuracy']:.2f}% | Macro-F1: {m_phq_bound['macro_f1']:.2f}% | Bal-Acc: {m_phq_bound['balanced_accuracy']:.2f}% | Rec: [0: {m_phq_bound['recall_0']:.1f}%, 1: {m_phq_bound['recall_1']:.1f}%, 2: {m_phq_bound['recall_2']:.1f}%]")

    print(f"\n5. Multi-Modal Blended Ensemble (Projected + Standard + LogReg):")
    print(f"   Accuracy: {m_ens['accuracy']:.2f}% | Macro-F1: {m_ens['macro_f1']:.2f}% | Bal-Acc: {m_ens['balanced_accuracy']:.2f}% | MAE: {m_ens['mae']:.2f} | Rec: [0: {m_ens['recall_0']:.1f}%, 1: {m_ens['recall_1']:.1f}%, 2: {m_ens['recall_2']:.1f}%]")

    # Save summary json
    summary_dict = {
        'projected_trimodal_mlp': m_proj,
        'standard_trimodal_mlp_alpha_015': m_std,
        'threshold_tuned_standard_mlp': m_tuned,
        'ordinal_phq_boundary_mapping': m_phq_bound,
        'multimodal_blended_ensemble': m_ens
    }
    with open(os.path.join(EXP_DIR, "threshold_and_ensemble_summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Save OOF predictions CSV
    df_oof = pd.DataFrame({
        'session_id': sids,
        'participant_id': pids,
        'true_class': y,
        'true_phq': y_phq,
        'proj_pred': proj_preds,
        'std_pred': std_preds,
        'ens_pred': oof_ensemble_preds,
        'phq_boundary_pred': oof_phq_boundary_preds,
        'ens_prob_0': oof_ensemble_probs[:, 0],
        'ens_prob_1': oof_ensemble_probs[:, 1],
        'ens_prob_2': oof_ensemble_probs[:, 2]
    })
    df_oof.to_csv(os.path.join(EXP_DIR, "predictions", "oof_ensemble_and_threshold_predictions.csv"), index=False)
    print(f"\nSaved {EXP_DIR}/predictions/oof_ensemble_and_threshold_predictions.csv successfully!")


if __name__ == "__main__":
    main()
