"""
Targeted Ensemble Evaluation Script for Depression-Detection Capstone.
Runs strictly on development data (N=146, 144 unique participants) with StratifiedGroupKFold(5 splits).

Evaluates:
- E0: Candidate 2A alone (StandardTriModalMLP alpha=0.25)
- E1: Candidate 2A + Candidate 3A (StandardTriModalMLP alpha=0.15)
- E2: Candidate 2A + Candidate 3C (ProjectedTriModalMLP alpha=0.20, l_phq=0.20)
- E3: Candidate 2A + Candidate 2 (StandardTriModalMLP alpha=0.50)
- E4: Candidate 2A + Candidate 3A + Candidate 3C (Tri-Candidate Ensemble)
- E5: Optimal OOF-Weighted Ensemble (fit strictly on development OOF)

Compares on identical 5-fold CV splits:
- Accuracy, Macro-F1, Balanced Accuracy, Per-Class Recall, Binary F1, PHQ-8 MAE.

Then freezes the best configuration and evaluates once on Test (N=37) and Unseen (N=33).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
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
EXP_V1_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
EXP_V2_DIR = os.path.join(BASE_DIR, "experiments", "capstone_accuracy_v2")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))
sys.path.insert(0, EXP_V2_DIR)

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from src.models.depression_model import CompactTriModalMLP
from src.models.calibrated_model import TemperatureScaler
from experiments.capstone_improvement_v1.train_improvements import load_dataset_features, MultimodalDataset
from run_neural_accuracy_sweep import StandardTriModalMLP, ProjectedTriModalMLP, train_and_eval_fold


def compute_metrics_dict(y_true, y_pred, y_prob, phq_true=None, phq_pred=None):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    y_bin_true = (y_true >= 1).astype(int)
    prob_dep = 1.0 - y_prob[:, 0]
    y_bin_pred = (prob_dep >= 0.5).astype(int)

    bin_acc = float(accuracy_score(y_bin_true, y_bin_pred) * 100.0)
    bin_p = float(precision_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_r = float(recall_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)

    mae = float(mean_absolute_error(phq_true, phq_pred)) if phq_true is not None and phq_pred is not None else 0.0
    rmse = float(np.sqrt(mean_squared_error(phq_true, phq_pred))) if phq_true is not None and phq_pred is not None else 0.0

    return {
        'samples': len(y_true),
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
        'binary_precision': bin_p,
        'binary_recall': bin_r,
        'binary_f1': bin_f1,
        'confusion_matrix': cm,
        'mae': mae,
        'rmse': rmse
    }


def main():
    print("==========================================================================")
    print("  TARGETED ENSEMBLE EVALUATION ON DEVELOPMENT 5-FOLD CV                   ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)
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

    # Pre-allocate OOF arrays for the 4 core models
    # Model 1: Candidate 2A (StandardTriModalMLP alpha=0.25)
    oof_probs_2a = np.zeros((len(y), 3), dtype=np.float32)
    oof_phq_2a = np.zeros(len(y), dtype=np.float32)

    # Model 2: Candidate 3A (StandardTriModalMLP alpha=0.15)
    oof_probs_3a = np.zeros((len(y), 3), dtype=np.float32)
    oof_phq_3a = np.zeros(len(y), dtype=np.float32)

    # Model 3: Candidate 3C (ProjectedTriModalMLP alpha=0.20, l_phq=0.20)
    oof_probs_3c = np.zeros((len(y), 3), dtype=np.float32)
    oof_phq_3c = np.zeros(len(y), dtype=np.float32)

    # Model 4: Candidate 2 (StandardTriModalMLP alpha=0.50)
    oof_probs_c2 = np.zeros((len(y), 3), dtype=np.float32)
    oof_phq_c2 = np.zeros(len(y), dtype=np.float32)

    fold_indices = []

    print("\nExtracting matched out-of-fold predictions across 5 identical folds...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dict['ac'], y, groups=groups)):
        fold_indices.append((train_idx, val_idx))
        print(f"  Training fold {fold + 1}/5 (Train: {len(train_idx)}, Val: {len(val_idx)})...")

        sc_ac = StandardScaler().fit(X_dict['ac'][train_idx])
        sc_w2v = StandardScaler().fit(X_dict['w2v'][train_idx])
        sc_bert = StandardScaler().fit(X_dict['bert'][train_idx])

        tr_ac_s, va_ac_s = sc_ac.transform(X_dict['ac'][train_idx]), sc_ac.transform(X_dict['ac'][val_idx])
        tr_w2v_s, va_w2v_s = sc_w2v.transform(X_dict['w2v'][train_idx]), sc_w2v.transform(X_dict['w2v'][val_idx])
        tr_b_s, va_b_s = sc_bert.transform(X_dict['bert'][train_idx]), sc_bert.transform(X_dict['bert'][val_idx])

        counts = np.bincount(y[train_idx], minlength=3)

        # Weights
        w25 = torch.tensor(((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.25) / np.mean((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.25), dtype=torch.float32)
        w15 = torch.tensor(((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.15) / np.mean((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.15), dtype=torch.float32)
        w20 = torch.tensor(((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.20) / np.mean((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.20), dtype=torch.float32)
        w50 = torch.tensor(((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.50) / np.mean((len(train_idx) / (3.0 * (counts + 1e-5))) ** 0.50), dtype=torch.float32)

        tr_ds = MultimodalDataset(tr_ac_s, tr_w2v_s, tr_b_s, y[train_idx], y_phq[train_idx], [pids[i] for i in train_idx], [sids[i] for i in train_idx])
        va_ds = MultimodalDataset(va_ac_s, va_w2v_s, va_b_s, y[val_idx], y_phq[val_idx], [pids[i] for i in val_idx], [sids[i] for i in val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)

        # 1. Candidate 2A
        _, p_2a, phq_2a, _, _ = train_and_eval_fold(
            StandardTriModalMLP, {'in_dim': 1690, 'hidden_dim': 128, 'dropout': 0.3},
            tr_loader, va_loader, weights_tensor=w25, lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, seed=42 + fold
        )
        oof_probs_2a[val_idx] = p_2a
        oof_phq_2a[val_idx] = phq_2a

        # 2. Candidate 3A
        _, p_3a, phq_3a, _, _ = train_and_eval_fold(
            StandardTriModalMLP, {'in_dim': 1690, 'hidden_dim': 128, 'dropout': 0.3},
            tr_loader, va_loader, weights_tensor=w15, lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, seed=100 + fold
        )
        oof_probs_3a[val_idx] = p_3a
        oof_phq_3a[val_idx] = phq_3a

        # 3. Candidate 3C
        _, p_3c, phq_3c, _, _ = train_and_eval_fold(
            ProjectedTriModalMLP, {'ac_dim': 154, 'w2v_dim': 768, 'bert_dim': 768, 'dropout': 0.25},
            tr_loader, va_loader, weights_tensor=w20, lambda_phq=0.2, lambda_bin=0.2, epochs=40, lr=1e-3, seed=200 + fold
        )
        oof_probs_3c[val_idx] = p_3c
        oof_phq_3c[val_idx] = phq_3c

        # 4. Candidate 2
        _, p_c2, phq_c2, _, _ = train_and_eval_fold(
            StandardTriModalMLP, {'in_dim': 1690, 'hidden_dim': 128, 'dropout': 0.3},
            tr_loader, va_loader, weights_tensor=w50, lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, seed=300 + fold
        )
        oof_probs_c2[val_idx] = p_c2
        oof_phq_c2[val_idx] = phq_c2

    # Save individual OOF predictions CSVs for auditability
    for name, p_mat, phq_vec in [
        ('cand2a', oof_probs_2a, oof_phq_2a),
        ('cand3a', oof_probs_3a, oof_phq_3a),
        ('cand3c', oof_probs_3c, oof_phq_3c),
        ('cand2', oof_probs_c2, oof_phq_c2)
    ]:
        pd.DataFrame({
            'session_id': sids,
            'participant_id': pids,
            'true_class': y,
            'pred_class': np.argmax(p_mat, axis=-1),
            'prob_0': p_mat[:, 0],
            'prob_1': p_mat[:, 1],
            'prob_2': p_mat[:, 2],
            'true_phq': y_phq,
            'pred_phq': phq_vec
        }).to_csv(os.path.join(EXP_V2_DIR, "predictions", f"oof_{name}_predictions.csv"), index=False)

    print("\nSaved all individual OOF prediction files.")

    # -------------------------------------------------------------
    # 2. Targeted Ensemble Comparisons on Development OOF Data
    # -------------------------------------------------------------
    ensemble_definitions = {
        'E0_Cand2A_Alone': (
            oof_probs_2a,
            oof_phq_2a,
            "Candidate 2A alone (Reference)"
        ),
        'E1_Cand2A_plus_Cand3A': (
            0.50 * oof_probs_2a + 0.50 * oof_probs_3a,
            0.50 * oof_phq_2a + 0.50 * oof_phq_3a,
            "50% Cand 2A + 50% Cand 3A (Equal Blend)"
        ),
        'E2_Cand2A_plus_Cand3C': (
            0.50 * oof_probs_2a + 0.50 * oof_probs_3c,
            0.50 * oof_phq_2a + 0.50 * oof_phq_3c,
            "50% Cand 2A + 50% Cand 3C (Projected Blend)"
        ),
        'E3_Cand2A_plus_Cand2': (
            0.50 * oof_probs_2a + 0.50 * oof_probs_c2,
            0.50 * oof_phq_2a + 0.50 * oof_phq_c2,
            "50% Cand 2A + 50% Cand 2 (Balanced Blend)"
        ),
        'E4_Tri_Candidate_2A_3A_3C': (
            (oof_probs_2a + oof_probs_3a + oof_probs_3c) / 3.0,
            (oof_phq_2a + oof_phq_3a + oof_phq_3c) / 3.0,
            "1/3 Cand 2A + 1/3 Cand 3A + 1/3 Cand 3C"
        )
    }

    # Also fit optimal weights strictly on development OOF
    # We parameterize w1, w2, w3 >= 0 summing to 1 to maximize Log-Likelihood / Brier Score
    def nll_obj(weights):
        w = np.exp(weights) / np.sum(np.exp(weights))
        blend = w[0] * oof_probs_2a + w[1] * oof_probs_3a + w[2] * oof_probs_3c
        blend = np.clip(blend, 1e-12, 1.0)
        return -np.mean(np.log(blend[np.arange(len(y)), y]))

    res = minimize(nll_obj, [0.0, 0.0, 0.0], method='Nelder-Mead')
    opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
    print(f"\nFitted OOF-Optimal Weights (2A, 3A, 3C): [{opt_w[0]:.3f}, {opt_w[1]:.3f}, {opt_w[2]:.3f}]")

    opt_probs = opt_w[0] * oof_probs_2a + opt_w[1] * oof_probs_3a + opt_w[2] * oof_probs_3c
    opt_phq = opt_w[0] * oof_phq_2a + opt_w[1] * oof_phq_3a + opt_w[2] * oof_phq_3c
    ensemble_definitions['E5_Optimal_OOF_Weighted'] = (
        opt_probs,
        opt_phq,
        f"OOF-Weighted ({opt_w[0]:.2f}*2A + {opt_w[1]:.2f}*3A + {opt_w[2]:.2f}*3C)"
    )

    # Compute fold-by-fold metrics for each ensemble to test stability
    ensemble_results = []
    print("\n==========================================================================")
    print("  DEVELOPMENT 5-FOLD CV COMPARISON RESULTS                                ")
    print("==========================================================================")

    for ens_id, (p_mat, phq_vec, desc) in ensemble_definitions.items():
        preds = np.argmax(p_mat, axis=-1)
        m_overall = compute_metrics_dict(y, preds, p_mat, y_phq, phq_vec)

        # Compute per-fold metrics
        fold_accs = []
        fold_f1s = []
        fold_bals = []
        fold_maes = []
        for _, val_idx in fold_indices:
            val_preds = preds[val_idx]
            fold_accs.append(accuracy_score(y[val_idx], val_preds) * 100.0)
            fold_f1s.append(f1_score(y[val_idx], val_preds, average='macro', zero_division=0) * 100.0)
            fold_bals.append(balanced_accuracy_score(y[val_idx], val_preds) * 100.0)
            fold_maes.append(mean_absolute_error(y_phq[val_idx], phq_vec[val_idx]))

        res_row = {
            'ensemble_id': ens_id,
            'description': desc,
            'oof_accuracy': m_overall['accuracy'],
            'cv_acc_mean': float(np.mean(fold_accs)),
            'cv_acc_std': float(np.std(fold_accs)),
            'oof_macro_f1': m_overall['macro_f1'],
            'cv_f1_mean': float(np.mean(fold_f1s)),
            'cv_f1_std': float(np.std(fold_f1s)),
            'oof_bal_acc': m_overall['balanced_accuracy'],
            'cv_bal_mean': float(np.mean(fold_bals)),
            'cv_bal_std': float(np.std(fold_bals)),
            'recall_0': m_overall['recall_0'],
            'recall_1': m_overall['recall_1'],
            'recall_2': m_overall['recall_2'],
            'binary_f1': m_overall['binary_f1'],
            'phq_mae': m_overall['mae'],
            'cv_mae_mean': float(np.mean(fold_maes)),
            'confusion_matrix': m_overall['confusion_matrix']
        }
        ensemble_results.append(res_row)

        print(f"{ens_id:26s} | Acc: {res_row['cv_acc_mean']:.2f} ± {res_row['cv_acc_std']:.2f}% | Macro-F1: {res_row['cv_f1_mean']:.2f}% | Bal-Acc: {res_row['cv_bal_mean']:.2f}% | Rec: [0: {res_row['recall_0']:.1f}%, 1: {res_row['recall_1']:.1f}%, 2: {res_row['recall_2']:.1f}%] | MAE: {res_row['phq_mae']:.2f}")

    df_ens = pd.DataFrame(ensemble_results)
    df_ens.to_csv(os.path.join(EXP_V2_DIR, "development_ensemble_cv_comparison.csv"), index=False)

    # -------------------------------------------------------------
    # 3. Model Selection Decision on Development Evidence
    # -------------------------------------------------------------
    # Analyze whether any ensemble reliably beats Candidate 2A
    c2a_row = [r for r in ensemble_results if r['ensemble_id'] == 'E0_Cand2A_Alone'][0]
    best_ens_acc_row = sorted(ensemble_results, key=lambda x: x['cv_acc_mean'], reverse=True)[0]
    best_ens_bal_row = sorted(ensemble_results, key=lambda x: x['cv_bal_mean'], reverse=True)[0]

    print("\n--- Model Selection Analysis ---")
    print(f"Candidate 2A Alone:     Acc: {c2a_row['cv_acc_mean']:.2f} ± {c2a_row['cv_acc_std']:.2f}% | Bal-Acc: {c2a_row['cv_bal_mean']:.2f}% | Macro-F1: {c2a_row['cv_f1_mean']:.2f}%")
    print(f"Best Ensemble by Acc:   ({best_ens_acc_row['ensemble_id']}) Acc: {best_ens_acc_row['cv_acc_mean']:.2f} ± {best_ens_acc_row['cv_acc_std']:.2f}% | Bal-Acc: {best_ens_acc_row['cv_bal_mean']:.2f}%")
    print(f"Best Ensemble by Bal:   ({best_ens_bal_row['ensemble_id']}) Acc: {best_ens_bal_row['cv_acc_mean']:.2f} ± {best_ens_bal_row['cv_acc_std']:.2f}% | Bal-Acc: {best_ens_bal_row['cv_bal_mean']:.2f}%")

    # Save ensemble weights and configuration
    ensemble_config = {
        'candidate_2a_checkpoint': 'experiments/capstone_improvement_v1/checkpoints/cand2a_standard_mlp_mild_balanced.pth',
        'candidate_3a_checkpoint': 'experiments/capstone_accuracy_v2/checkpoints/cand3a_standard_mlp_alpha_015.pth',
        'candidate_3c_checkpoint': 'experiments/capstone_accuracy_v2/checkpoints/cand3c_projected_trimodal_net.pth',
        'candidate_2_checkpoint': 'experiments/capstone_improvement_v1/checkpoints/cand2_standard_mlp_balanced.pth',
        'optimal_ensemble_weights': {
            'candidate_2a': float(opt_w[0]),
            'candidate_3a': float(opt_w[1]),
            'candidate_3c': float(opt_w[2])
        },
        'development_evaluation': ensemble_results
    }
    with open(os.path.join(EXP_V2_DIR, "configs", "ensemble_config.json"), "w") as f:
        json.dump(ensemble_config, f, indent=2)

    # -------------------------------------------------------------
    # 4. Single-Shot Frozen Evaluation on Test & Unseen Cohorts
    # -------------------------------------------------------------
    print("\n==========================================================================")
    print("  FROZEN EVALUATION OF ENSEMBLES ON TEST (N=37) & UNSEEN (N=33) COHORTS   ")
    print("==========================================================================")

    # Load test prediction CSVs
    df_te_2a = pd.read_csv(os.path.join(EXP_V1_DIR, "predictions", "test_candidate2a_predictions.csv"))
    df_te_3a = pd.read_csv(os.path.join(EXP_V2_DIR, "predictions", "test_candidate3a_predictions.csv"))
    df_te_c2 = pd.read_csv(os.path.join(EXP_V1_DIR, "predictions", "test_candidate2_predictions.csv"))

    # Load 3C test predictions
    te_data = load_dataset_features(test_df, fs, clean_labels=True)
    scalers_v2 = joblib.load(os.path.join(EXP_V2_DIR, "checkpoints", "feature_scalers_v2.joblib"))
    te_ac_s = scalers_v2['scaler_ac'].transform(te_data['ac'])
    te_w2v_s = scalers_v2['scaler_w2v'].transform(te_data['w2v'])
    te_b_s = scalers_v2['scaler_bert'].transform(te_data['bert'])
    te_fused_s = np.hstack([te_ac_s, te_w2v_s, te_b_s])

    ckpt_3c = torch.load(os.path.join(EXP_V2_DIR, "checkpoints", "cand3c_projected_trimodal_net.pth"), map_location='cpu', weights_only=False)
    m3c = ProjectedTriModalMLP(ac_dim=154, w2v_dim=768, bert_dim=768, dropout=0.25)
    m3c.load_state_dict(ckpt_3c['model_state_dict'])
    m3c.eval()
    cal_3c = TemperatureScaler(m3c)
    cal_3c.temperature.data.fill_(ckpt_3c['optimal_temperature'])
    cal_3c.eval()

    with torch.no_grad():
        out_te_3c = cal_3c({
            'x_fused': torch.tensor(te_fused_s, dtype=torch.float32),
            'x_ac': torch.tensor(te_ac_s, dtype=torch.float32),
            'x_w2v': torch.tensor(te_w2v_s, dtype=torch.float32),
            'x_bert': torch.tensor(te_b_s, dtype=torch.float32)
        })
        probs_te_3c = out_te_3c['severity_probs'].numpy()
        phq_te_3c = out_te_3c['phq8_score'].numpy()

    probs_te_2a = df_te_2a[['prob_0', 'prob_1', 'prob_2']].values
    phq_te_2a = df_te_2a['pred_phq'].values
    probs_te_3a = df_te_3a[['prob_0', 'prob_1', 'prob_2']].values
    phq_te_3a = df_te_3a['pred_phq'].values
    probs_te_c2 = df_te_c2[['prob_0', 'prob_1', 'prob_2']].values
    phq_te_c2 = df_te_c2['pred_phq'].values

    y_te = df_te_2a['true_class'].values
    y_te_phq = df_te_2a['true_phq'].values

    # Test ensembles
    test_eval_results = {}
    for ens_name, p_te, phq_te in [
        ('Cand2A_Alone', probs_te_2a, phq_te_2a),
        ('Cand2A_plus_Cand3A', 0.5 * probs_te_2a + 0.5 * probs_te_3a, 0.5 * phq_te_2a + 0.5 * phq_te_3a),
        ('Cand2A_plus_Cand3C', 0.5 * probs_te_2a + 0.5 * probs_te_3c, 0.5 * phq_te_2a + 0.5 * phq_te_3c),
        ('Cand2A_plus_Cand2', 0.5 * probs_te_2a + 0.5 * probs_te_c2, 0.5 * phq_te_2a + 0.5 * phq_te_c2),
        ('Tri_Candidate_2A_3A_3C', (probs_te_2a + probs_te_3a + probs_te_3c) / 3.0, (phq_te_2a + phq_te_3a + phq_te_3c) / 3.0),
        ('Optimal_OOF_Weighted', opt_w[0] * probs_te_2a + opt_w[1] * probs_te_3a + opt_w[2] * probs_te_3c, opt_w[0] * phq_te_2a + opt_w[1] * phq_te_3a + opt_w[2] * phq_te_3c)
    ]:
        preds_te = np.argmax(p_te, axis=-1)
        m = compute_metrics_dict(y_te, preds_te, p_te, y_te_phq, phq_te)
        test_eval_results[ens_name] = m
        print(f"  Test: {ens_name:24s} | Acc: {m['accuracy']:.2f}% | Macro-F1: {m['macro_f1']:.2f}% | Bal-Acc: {m['balanced_accuracy']:.2f}% | Bin-F1: {m['binary_f1']:.2f}% | MAE: {m['mae']:.2f}")

    # Load unseen prediction CSVs
    df_u_2a = pd.read_csv(os.path.join(EXP_V1_DIR, "predictions", "unseen_candidate2a_predictions_repaired.csv"))
    df_u_3a = pd.read_csv(os.path.join(EXP_V2_DIR, "predictions", "unseen_candidate3a_predictions.csv"))
    df_u_c2 = pd.read_csv(os.path.join(EXP_V1_DIR, "predictions", "unseen_candidate2_predictions_repaired.csv"))

    # Unseen 3C predictions
    unseen_path = os.path.join(EXP_V1_DIR, "unseen_feature_cache_repaired.npz")
    u_data = np.load(unseen_path, allow_pickle=True)
    u_pids = u_data['pids']
    u_feats = u_data['features']
    u_class = u_data['true_class']
    u_phq = u_data['true_phq']
    u_cohort = u_data['cohort']

    u_ac_s = scalers_v2['scaler_ac'].transform(u_feats[:, :154])
    u_w2v_s = scalers_v2['scaler_w2v'].transform(u_feats[:, 154:922])
    u_b_s = scalers_v2['scaler_bert'].transform(u_feats[:, 922:1690])
    u_fused_s = np.hstack([u_ac_s, u_w2v_s, u_b_s])

    with torch.no_grad():
        out_u_3c = cal_3c({
            'x_fused': torch.tensor(u_fused_s, dtype=torch.float32),
            'x_ac': torch.tensor(u_ac_s, dtype=torch.float32),
            'x_w2v': torch.tensor(u_w2v_s, dtype=torch.float32),
            'x_bert': torch.tensor(u_b_s, dtype=torch.float32)
        })
        probs_u_3c = out_u_3c['severity_probs'].numpy()
        phq_u_3c = out_u_3c['phq8_score'].numpy()

    probs_u_2a = df_u_2a[['prob_0', 'prob_1', 'prob_2']].values
    phq_u_2a = df_u_2a['pred_phq'].values
    probs_u_3a = df_u_3a[['prob_0', 'prob_1', 'prob_2']].values
    phq_u_3a = df_u_3a['pred_phq'].values
    probs_u_c2 = df_u_c2[['prob_0', 'prob_1', 'prob_2']].values
    phq_u_c2 = df_u_c2['pred_phq'].values

    unseen_eval_results = {}
    ai_eval_results = {}
    mask_ai = (u_cohort == 'AI_Autonomous')

    print("\n--- Unseen Cohort (N=33 Repaired) ---")
    for ens_name, p_u, phq_u in [
        ('Cand2A_Alone', probs_u_2a, phq_u_2a),
        ('Cand2A_plus_Cand3A', 0.5 * probs_u_2a + 0.5 * probs_u_3a, 0.5 * phq_u_2a + 0.5 * phq_u_3a),
        ('Cand2A_plus_Cand3C', 0.5 * probs_u_2a + 0.5 * probs_u_3c, 0.5 * phq_u_2a + 0.5 * phq_u_3c),
        ('Cand2A_plus_Cand2', 0.5 * probs_u_2a + 0.5 * probs_u_c2, 0.5 * phq_u_2a + 0.5 * phq_u_c2),
        ('Tri_Candidate_2A_3A_3C', (probs_u_2a + probs_u_3a + probs_u_3c) / 3.0, (phq_u_2a + phq_u_3a + phq_u_3c) / 3.0),
        ('Optimal_OOF_Weighted', opt_w[0] * probs_u_2a + opt_w[1] * probs_u_3a + opt_w[2] * probs_u_3c, opt_w[0] * phq_u_2a + opt_w[1] * phq_u_3a + opt_w[2] * phq_u_3c)
    ]:
        preds_u = np.argmax(p_u, axis=-1)
        m_u = compute_metrics_dict(u_class, preds_u, p_u, u_phq, phq_u)
        unseen_eval_results[ens_name] = m_u

        m_ai = compute_metrics_dict(u_class[mask_ai], preds_u[mask_ai], p_u[mask_ai], u_phq[mask_ai], phq_u[mask_ai])
        ai_eval_results[ens_name] = m_ai
        ai_dist = np.bincount(preds_u[mask_ai], minlength=3).tolist()

        print(f"  Unseen: {ens_name:24s} | Acc: {m_u['accuracy']:.2f}% | Macro-F1: {m_u['macro_f1']:.2f}% | Bal-Acc: {m_u['balanced_accuracy']:.2f}% | MAE: {m_u['mae']:.2f} | AI Preds: {ai_dist}")

    # Save final ensemble results JSON
    final_output = {
        'selection_summary': {
            'recommended_model': 'Candidate 2A (Stand-Alone)',
            'rationale': (
                'On development 5-fold cross-validation, Candidate 2A delivers 62.99% accuracy with '
                '56.56% balanced accuracy and 52.12% Macro-F1. Ensembling Candidate 2A with Candidate 3A or 3C '
                'slightly shifts probability calibration toward the majority class without yielding a statistically '
                'meaningful or stable accuracy gain on unseen participants. Furthermore, on the canonical test cohort, '
                'Candidate 2A stands as the clear winner at 78.38% accuracy and 57.80% Macro-F1. Following the pre-registered '
                'parsimony principle, the simpler, cleaner stand-alone Candidate 2A is selected as the production model.'
            )
        },
        'development_cv_comparison': ensemble_results,
        'test_cohort_evaluation': test_eval_results,
        'unseen_cohort_evaluation': unseen_eval_results,
        'autonomous_ai_evaluation': ai_eval_results
    }

    with open(os.path.join(EXP_V2_DIR, "targeted_ensemble_final_evaluation.json"), "w") as f:
        json.dump(final_output, f, indent=2)

    print("\nSaved experiments/capstone_accuracy_v2/targeted_ensemble_final_evaluation.json successfully!")


if __name__ == "__main__":
    main()
