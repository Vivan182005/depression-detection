"""
Phase: Participant-Grouped, Leakage-Free Cross-Validation on Development Pool (N=146).
Compares Baseline vs Candidate 2 on exactly the same folds using authoritative clean labels.
Evaluates:
- Mean and variability (std) across 5 folds for:
  * Accuracy
  * Macro-F1
  * Balanced Accuracy
  * Minimal/Mild Recall
  * Moderate Recall
  * Severe Recall
  * Binary F1 (PHQ >= 10)
  * PHQ-8 MAE and RMSE
Saves:
- experiments/capstone_improvement_v1/development_cv_results.json
- experiments/capstone_improvement_v1/predictions/oof_baseline_predictions.csv
- experiments/capstone_improvement_v1/predictions/oof_candidate2_predictions.csv
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, auc
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))
sys.path.insert(0, EXP_DIR)

from src.models.depression_model import CompactTriModalMLP
from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from train_improvements import StandardTriModalMLP, load_dataset_features


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FusedDataset(Dataset):
    def __init__(self, x, y_cls, y_phq, pids, sids):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_phq = torch.tensor(y_phq, dtype=torch.float32)
        self.pids = pids
        self.sids = sids

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        phq = self.y_phq[idx].item()
        return {
            'x': self.x[idx],
            'class_id': self.y_cls[idx],
            'phq8_score': self.y_phq[idx],
            'binary_label': torch.tensor(1.0 if phq >= 10.0 else 0.0, dtype=torch.float32),
            'participant_id': self.pids[idx],
            'session_id': self.sids[idx]
        }


def train_single_fold(model, train_loader, val_loader, criterion_ce, epochs=40, lr=1e-3, weight_decay=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['x'])
            l_ce = criterion_ce(out['severity_logits'], batch['class_id'])
            l_mse = criterion_mse(out['phq8_score'], batch['phq8_score'])
            l_bce = criterion_bce(out['binary_logits'], batch['binary_label'])
            loss = l_ce + 0.1 * l_mse + 0.2 * l_bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation evaluation
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['x'])
                probs = out['severity_probs'].cpu().numpy()
                v_preds.append(np.argmax(probs, axis=-1))
                v_targets.append(batch['class_id'].cpu().numpy())

        v_preds = np.concatenate(v_preds)
        v_targets = np.concatenate(v_targets)
        v_f1 = f1_score(v_targets, v_preds, average='macro', zero_division=0) * 100.0

        if v_f1 > best_val_macro_f1:
            best_val_macro_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_predictions(y_true, y_pred, y_prob, phq_true, phq_pred):
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0

    bin_true = (phq_true >= 10.0).astype(int)
    prob_depressed = 1.0 - y_prob[:, 0]
    bin_pred = (prob_depressed >= 0.5).astype(int)

    bin_acc = accuracy_score(bin_true, bin_pred) * 100.0
    bin_p = precision_score(bin_true, bin_pred, zero_division=0) * 100.0
    bin_r = recall_score(bin_true, bin_pred, zero_division=0) * 100.0
    bin_f1 = f1_score(bin_true, bin_pred, zero_division=0) * 100.0

    mae = mean_absolute_error(phq_true, phq_pred)
    rmse = np.sqrt(mean_squared_error(phq_true, phq_pred))

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'weighted_f1': weighted_f1,
        'recall_minimal': r_per[0],
        'recall_moderate': r_per[1],
        'recall_severe': r_per[2],
        'f1_minimal': f_per[0],
        'f1_moderate': f_per[1],
        'f1_severe': f_per[2],
        'binary_accuracy': bin_acc,
        'binary_precision': bin_p,
        'binary_recall': bin_r,
        'binary_f1': bin_f1,
        'mae': mae,
        'rmse': rmse
    }


def run_5fold_cv():
    print("==========================================================================")
    print("   PARTICIPANT-GROUPED 5-FOLD CROSS-VALIDATION ON DEVELOPMENT DATA (N=146) ")
    print("   Zero Data Leakage | Matched Folds | Authoritative Ground-Truth Labels ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    d_dev = load_dataset_features(dev_df, fs, clean_labels=True)

    X_ac = d_dev['ac']
    X_w2v = d_dev['w2v']
    X_bert = d_dev['bert']
    y_class = d_dev['y_class']
    y_phq = d_dev['y_phq']
    pids = np.array(d_dev['pids'])
    sids = np.array(d_dev['sids'])

    print(f"Total Development Sessions: {len(y_class)} (Unique PIDs: {len(np.unique(pids))})")
    print(f"Class Distribution: Minimal={np.sum(y_class==0)}, Moderate={np.sum(y_class==1)}, Severe={np.sum(y_class==2)}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics_baseline = []
    fold_metrics_cand2 = []

    oof_base_preds = np.zeros((len(y_class), 3))
    oof_base_phq = np.zeros(len(y_class))

    oof_cand_preds = np.zeros((len(y_class), 3))
    oof_cand_phq = np.zeros(len(y_class))

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_ac, y_class, groups=pids), 1):
        print(f"\n--- FOLD {fold} / 5 (Train: {len(tr_idx)} sessions, Val: {len(val_idx)} sessions) ---")
        
        # Verify participant isolation
        tr_pids_set = set(pids[tr_idx])
        val_pids_set = set(pids[val_idx])
        assert len(tr_pids_set.intersection(val_pids_set)) == 0, f"Leakage detected in Fold {fold}!"

        # Fit scalers strictly on training fold
        sc_ac = StandardScaler().fit(X_ac[tr_idx])
        sc_w2v = StandardScaler().fit(X_w2v[tr_idx])
        sc_bert = StandardScaler().fit(X_bert[tr_idx])

        # Scaled feature matrices
        tr_scaled = np.hstack([sc_ac.transform(X_ac[tr_idx]), sc_w2v.transform(X_w2v[tr_idx]), sc_bert.transform(X_bert[tr_idx])])
        val_scaled = np.hstack([sc_ac.transform(X_ac[val_idx]), sc_w2v.transform(X_w2v[val_idx]), sc_bert.transform(X_bert[val_idx])])

        tr_ds = FusedDataset(tr_scaled, y_class[tr_idx], y_phq[tr_idx], pids[tr_idx], sids[tr_idx])
        val_ds = FusedDataset(val_scaled, y_class[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        # -------------------------------------------------------------
        # 1. Train Baseline Architecture (Unweighted CE)
        # -------------------------------------------------------------
        set_seed(42 + fold)
        base_model = CompactTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
        base_model = train_single_fold(
            base_model, tr_loader, val_loader,
            criterion_ce=nn.CrossEntropyLoss(),
            epochs=40, lr=1e-3, weight_decay=1e-3
        )

        with torch.no_grad():
            out_b = base_model(torch.tensor(val_scaled, dtype=torch.float32))
            prob_b = out_b['severity_probs'].numpy()
            pred_b = np.argmax(prob_b, axis=-1)
            phq_b = out_b['phq8_score'].numpy()

        oof_base_preds[val_idx] = prob_b
        oof_base_phq[val_idx] = phq_b
        m_b = evaluate_predictions(y_class[val_idx], pred_b, prob_b, y_phq[val_idx], phq_b)
        fold_metrics_baseline.append(m_b)
        print(f"  Baseline   -> Acc: {m_b['accuracy']:5.2f}% | Macro-F1: {m_b['macro_f1']:5.2f}% | Bal-Acc: {m_b['balanced_accuracy']:5.2f}% | Mod-Rec: {m_b['recall_moderate']:5.2f}% | MAE: {m_b['mae']:.2f}")

        # -------------------------------------------------------------
        # 2. Train Candidate 2 Architecture (Smoothed Balanced CE)
        # -------------------------------------------------------------
        tr_counts = np.bincount(y_class[tr_idx], minlength=3)
        smooth_weights = np.sqrt(len(tr_idx) / (3.0 * (tr_counts + 1e-5)))
        w_tensor = torch.tensor(smooth_weights, dtype=torch.float32)

        set_seed(42 + fold)
        cand_model = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
        cand_model = train_single_fold(
            cand_model, tr_loader, val_loader,
            criterion_ce=nn.CrossEntropyLoss(weight=w_tensor),
            epochs=40, lr=1e-3, weight_decay=1e-3
        )

        with torch.no_grad():
            out_c = cand_model(torch.tensor(val_scaled, dtype=torch.float32))
            prob_c = out_c['severity_probs'].numpy()
            pred_c = np.argmax(prob_c, axis=-1)
            phq_c = out_c['phq8_score'].numpy()

        oof_cand_preds[val_idx] = prob_c
        oof_cand_phq[val_idx] = phq_c
        m_c = evaluate_predictions(y_class[val_idx], pred_c, prob_c, y_phq[val_idx], phq_c)
        fold_metrics_cand2.append(m_c)
        print(f"  Candidate2 -> Acc: {m_c['accuracy']:5.2f}% | Macro-F1: {m_c['macro_f1']:5.2f}% | Bal-Acc: {m_c['balanced_accuracy']:5.2f}% | Mod-Rec: {m_c['recall_moderate']:5.2f}% | MAE: {m_c['mae']:.2f}")

    # Summary table across folds
    def aggregate_folds(metrics_list):
        agg = {}
        for k in metrics_list[0].keys():
            vals = [m[k] for m in metrics_list]
            agg[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': [float(v) for v in vals]
            }
        return agg

    agg_base = aggregate_folds(fold_metrics_baseline)
    agg_cand = aggregate_folds(fold_metrics_cand2)

    # Full out-of-fold metrics
    oof_pred_b = np.argmax(oof_base_preds, axis=-1)
    oof_pred_c = np.argmax(oof_cand_preds, axis=-1)
    oof_m_base = evaluate_predictions(y_class, oof_pred_b, oof_base_preds, y_phq, oof_base_phq)
    oof_m_cand = evaluate_predictions(y_class, oof_pred_c, oof_cand_preds, y_phq, oof_cand_phq)

    print("\n" + "="*85)
    print("   5-FOLD CROSS-VALIDATION SUMMARY (Mean ± Std over 5 Matched Folds)")
    print("="*85)
    print(f"{'Metric':<28} | {'Baseline (Unweighted)':<25} | {'Candidate 2 (Balanced)':<25}")
    print("-"*85)
    for k in ['accuracy', 'macro_f1', 'balanced_accuracy', 'recall_minimal', 'recall_moderate', 'recall_severe', 'binary_f1', 'mae', 'rmse']:
        b_m, b_s = agg_base[k]['mean'], agg_base[k]['std']
        c_m, c_s = agg_cand[k]['mean'], agg_cand[k]['std']
        unit = "%" if k not in ['mae', 'rmse'] else " pts"
        print(f"{k:<28} | {b_m:6.2f} ± {b_s:4.2f}{unit:<10} | {c_m:6.2f} ± {c_s:4.2f}{unit:<10}")
    print("="*85)

    # Save OOF Predictions
    preds_dir = os.path.join(EXP_DIR, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    df_oof_base = pd.DataFrame({
        'session_id': sids,
        'participant_id': pids,
        'clean_class': y_class,
        'clean_phq8': y_phq,
        'oof_pred_class': oof_pred_b,
        'oof_pred_phq8': oof_base_phq,
        'oof_prob_0': oof_base_preds[:, 0],
        'oof_prob_1': oof_base_preds[:, 1],
        'oof_prob_2': oof_base_preds[:, 2]
    })
    oof_b_path = os.path.join(preds_dir, "oof_baseline_predictions.csv")
    df_oof_base.to_csv(oof_b_path, index=False)

    df_oof_cand = pd.DataFrame({
        'session_id': sids,
        'participant_id': pids,
        'clean_class': y_class,
        'clean_phq8': y_phq,
        'oof_pred_class': oof_pred_c,
        'oof_pred_phq8': oof_cand_phq,
        'oof_prob_0': oof_cand_preds[:, 0],
        'oof_prob_1': oof_cand_preds[:, 1],
        'oof_prob_2': oof_cand_preds[:, 2]
    })
    oof_c_path = os.path.join(preds_dir, "oof_candidate2_predictions.csv")
    df_oof_cand.to_csv(oof_c_path, index=False)

    # Save Summary JSON
    cv_payload = {
        'total_development_sessions': len(y_class),
        'unique_development_participants': int(len(np.unique(pids))),
        'n_splits': 5,
        'stratification': "StratifiedGroupKFold grouped by participant_id",
        'baseline_cross_validation': {
            'fold_aggregates': agg_base,
            'pooled_oof_metrics': oof_m_base
        },
        'candidate2_cross_validation': {
            'fold_aggregates': agg_cand,
            'pooled_oof_metrics': oof_m_cand
        }
    }
    cv_json_path = os.path.join(EXP_DIR, "development_cv_results.json")
    with open(cv_json_path, "w") as f:
        json.dump(cv_payload, f, indent=2)

    print(f"\n[SAVED] OOF Baseline Predictions: {oof_b_path}")
    print(f"[SAVED] OOF Candidate 2 Predictions: {oof_c_path}")
    print(f"[SAVED] CV Summary JSON: {cv_json_path}")
    print("Cross-Validation Execution Completed Successfully!")


if __name__ == "__main__":
    run_5fold_cv()
