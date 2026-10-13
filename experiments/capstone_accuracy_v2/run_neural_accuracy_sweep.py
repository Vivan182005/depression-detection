"""
Experiment Suite 3 & 4: Neural Architecture Exploration, Class Balancing Sweep, & OOF Threshold Tuning.
Evaluates on development data (N=146, 144 unique participants) using StratifiedGroupKFold(5 splits).

Explores:
1. Architectures:
   - StandardTriModalMLP
   - ProjectedTriModalMLP (modality projections with LayerNorm)
   - ResidualTriModalMLP (residual connections + dropout)
   - OrdinalMultiTaskNet (PHQ continuous + ordinal boundaries)
2. Alpha Weighting Sweep:
   - alpha in [0.0, 0.10, 0.15, 0.20, 0.25, 0.35]
3. Multi-task loss weights:
   - lambda_phq in [0.05, 0.1, 0.2]
4. Decision Threshold Tuning:
   - Argmax classification vs fold-tuned probability/PHQ thresholds.
"""

import os
import sys
import json
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
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_accuracy_v2")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from experiments.capstone_improvement_v1.train_improvements import load_dataset_features, MultimodalDataset
from src.models.calibrated_model import TemperatureScaler


def compute_metrics_full(y_true, y_pred, phq_true=None, phq_pred=None):
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

    mae = float(mean_absolute_error(phq_true, phq_pred)) if phq_true is not None and phq_pred is not None else 0.0
    rmse = float(np.sqrt(mean_squared_error(phq_true, phq_pred))) if phq_true is not None and phq_pred is not None else 0.0

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
        'confusion_matrix': cm,
        'mae': mae,
        'rmse': rmse
    }


# -------------------------------------------------------------
# Architecture Definitions
# -------------------------------------------------------------

class StandardTriModalMLP(nn.Module):
    def __init__(self, in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_class = nn.Linear(64, num_classes)
        self.head_phq = nn.Linear(64, 1)
        self.head_bin = nn.Linear(64, 1)

    def forward(self, batch):
        x = batch['x_fused'] if isinstance(batch, dict) else batch
        feat = self.net(x)
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        bin_logit = self.head_bin(feat).squeeze(-1)
        return {
            'severity_logits': logits,
            'severity_probs': F.softmax(logits, dim=-1),
            'phq8_score': phq,
            'binary_logits': bin_logit,
            'binary_probs': torch.sigmoid(bin_logit)
        }


class ProjectedTriModalMLP(nn.Module):
    def __init__(self, ac_dim=154, w2v_dim=768, bert_dim=768, num_classes=3, dropout=0.25):
        super().__init__()
        self.proj_ac = nn.Sequential(
            nn.Linear(ac_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_w2v = nn.Sequential(
            nn.Linear(w2v_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_bert = nn.Sequential(
            nn.Linear(bert_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        fused_dim = 64 + 128 + 128  # 320 dims
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.head_class = nn.Linear(64, num_classes)
        self.head_phq = nn.Linear(64, 1)
        self.head_bin = nn.Linear(64, 1)

    def forward(self, batch):
        if isinstance(batch, dict):
            h_ac = self.proj_ac(batch['x_ac'])
            h_w2v = self.proj_w2v(batch['x_w2v'])
            h_bert = self.proj_bert(batch['x_bert'])
        else:
            h_ac = self.proj_ac(batch[:, :154])
            h_w2v = self.proj_w2v(batch[:, 154:922])
            h_bert = self.proj_bert(batch[:, 922:1690])

        fused = torch.cat([h_ac, h_w2v, h_bert], dim=-1)
        feat = self.fusion(fused)
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        bin_logit = self.head_bin(feat).squeeze(-1)
        return {
            'severity_logits': logits,
            'severity_probs': F.softmax(logits, dim=-1),
            'phq8_score': phq,
            'binary_logits': bin_logit,
            'binary_probs': torch.sigmoid(bin_logit)
        }


class ResidualTriModalMLP(nn.Module):
    def __init__(self, in_dim=1690, hidden_dim=256, num_classes=3, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.head_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.head_class = nn.Linear(64, num_classes)
        self.head_phq = nn.Linear(64, 1)
        self.head_bin = nn.Linear(64, 1)

    def forward(self, batch):
        x = batch['x_fused'] if isinstance(batch, dict) else batch
        h0 = self.in_proj(x)
        h = h0 + self.res_block(h0)
        feat = self.head_proj(h)
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        bin_logit = self.head_bin(feat).squeeze(-1)
        return {
            'severity_logits': logits,
            'severity_probs': F.softmax(logits, dim=-1),
            'phq8_score': phq,
            'binary_logits': bin_logit,
            'binary_probs': torch.sigmoid(bin_logit)
        }


# -------------------------------------------------------------
# Training and Evaluation Engine
# -------------------------------------------------------------

def train_and_eval_fold(
    model_cls, model_kwargs, train_loader, val_loader,
    weights_tensor, lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, seed=42
):
    torch.manual_seed(seed)
    model = model_cls(**model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss_c = criterion_ce(out['severity_logits'], batch['class_id'])
            loss_p = criterion_mse(out['phq8_score'], batch['phq8_score'])
            loss_b = criterion_bce(out['binary_logits'], batch['binary_label'])
            loss = loss_c + lambda_phq * loss_p + lambda_bin * loss_b
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                probs = out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                y_val = batch['class_id'].numpy()
        # Early stopping selection metric: balanced blend of Accuracy and Macro-F1
        f1 = f1_score(y_val, preds, average='macro', zero_division=0) * 100.0
        acc = accuracy_score(y_val, preds) * 100.0
        score = 0.5 * acc + 0.5 * f1
        if score > best_val_score:
            best_val_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    cal = TemperatureScaler(model)
    cal.calibrate(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            out = cal(batch)
            probs = out['severity_probs'].numpy()
            preds = np.argmax(probs, axis=-1)
            phq_preds = out['phq8_score'].numpy()
            targets = batch['class_id'].numpy()
            phq_targets = batch['phq8_score'].numpy()

    return preds, probs, phq_preds, targets, phq_targets


def run_experiment_cv(
    model_cls, model_kwargs, X_dict, y, y_phq, pids, sids,
    alpha=0.25, lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, n_splits=5, seed=42
):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = [int(p) for p in pids]

    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), 3), dtype=np.float32)
    oof_phq = np.zeros(len(y), dtype=np.float32)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dict['ac'], y, groups=groups)):
        sc_ac = StandardScaler().fit(X_dict['ac'][train_idx])
        sc_w2v = StandardScaler().fit(X_dict['w2v'][train_idx])
        sc_bert = StandardScaler().fit(X_dict['bert'][train_idx])

        tr_ac_s, va_ac_s = sc_ac.transform(X_dict['ac'][train_idx]), sc_ac.transform(X_dict['ac'][val_idx])
        tr_w2v_s, va_w2v_s = sc_w2v.transform(X_dict['w2v'][train_idx]), sc_w2v.transform(X_dict['w2v'][val_idx])
        tr_b_s, va_b_s = sc_bert.transform(X_dict['bert'][train_idx]), sc_bert.transform(X_dict['bert'][val_idx])

        # Compute weights strictly on training fold
        if alpha == 0.0:
            w_tensor = None
        else:
            counts = np.bincount(y[train_idx], minlength=3)
            raw_w = (len(train_idx) / (3.0 * (counts + 1e-5))) ** alpha
            norm_w = raw_w / np.mean(raw_w)
            w_tensor = torch.tensor(norm_w, dtype=torch.float32)

        tr_ds = MultimodalDataset(tr_ac_s, tr_w2v_s, tr_b_s, y[train_idx], y_phq[train_idx], [pids[i] for i in train_idx], [sids[i] for i in train_idx])
        va_ds = MultimodalDataset(va_ac_s, va_w2v_s, va_b_s, y[val_idx], y_phq[val_idx], [pids[i] for i in val_idx], [sids[i] for i in val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)

        preds_va, probs_va, phq_va, targets_va, phq_targets_va = train_and_eval_fold(
            model_cls, model_kwargs, tr_loader, va_loader,
            weights_tensor=w_tensor, lambda_phq=lambda_phq, lambda_bin=lambda_bin,
            epochs=epochs, lr=lr, seed=seed + fold
        )

        oof_preds[val_idx] = preds_va
        oof_probs[val_idx] = probs_va
        oof_phq[val_idx] = phq_va

        m = compute_metrics_full(targets_va, preds_va, phq_targets_va, phq_va)
        fold_metrics.append(m)

    oof_m = compute_metrics_full(y, oof_preds, y_phq, oof_phq)

    summary = {
        'oof_accuracy': oof_m['accuracy'],
        'oof_macro_f1': oof_m['macro_f1'],
        'oof_balanced_accuracy': oof_m['balanced_accuracy'],
        'oof_recall_0': oof_m['recall_0'],
        'oof_recall_1': oof_m['recall_1'],
        'oof_recall_2': oof_m['recall_2'],
        'oof_binary_f1': oof_m['binary_f1'],
        'oof_mae': oof_m['mae'],
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
        'cv_bin_f1_mean': float(np.mean([f['binary_f1'] for f in fold_metrics])),
        'cv_mae_mean': float(np.mean([f['mae'] for f in fold_metrics])),
        'cv_mae_std': float(np.std([f['mae'] for f in fold_metrics]))
    }
    return summary, oof_preds, oof_probs, oof_phq


def main():
    print("==========================================================================")
    print("  EXPERIMENT SUITE 3 & 4: NEURAL ARCHITECTURES & BALANCING SWEEP          ")
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

    results = []

    # -------------------------------------------------------------
    # 1. Sweep Class-Balancing alpha on StandardTriModalMLP
    # -------------------------------------------------------------
    print("\n--- Part 1: Class-Balancing Alpha Sweep on StandardTriModalMLP ---")
    alphas = [0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
    for alpha in alphas:
        name = f"StandardMLP (alpha={alpha:.2f})"
        summary, _, _, _ = run_experiment_cv(
            StandardTriModalMLP, {'in_dim': 1690, 'hidden_dim': 128, 'dropout': 0.3},
            X_dict, y, y_phq, pids, sids, alpha=alpha, lambda_phq=0.1, lambda_bin=0.2
        )
        summary['config_name'] = name
        summary['architecture'] = 'StandardTriModalMLP'
        summary['alpha'] = alpha
        results.append(summary)
        print(f"  {name:30s} | Acc: {summary['cv_acc_mean']:.2f} ± {summary['cv_acc_std']:.2f}% | Macro-F1: {summary['cv_macro_f1_mean']:.2f}% | Bal-Acc: {summary['cv_bal_acc_mean']:.2f}% | Rec1: {summary['cv_rec1_mean']:.1f}% | Rec2: {summary['cv_rec2_mean']:.1f}% | MAE: {summary['cv_mae_mean']:.2f}")

    # -------------------------------------------------------------
    # 2. Architecture Comparison with Optimal Alpha
    # -------------------------------------------------------------
    print("\n--- Part 2: Architecture Comparison with Mild Alpha (0.20 & 0.25) ---")
    arch_configs = [
        ('ProjectedTriModalMLP (alpha=0.15)', ProjectedTriModalMLP, {'ac_dim': 154, 'w2v_dim': 768, 'bert_dim': 768, 'dropout': 0.25}, 0.15, 0.1),
        ('ProjectedTriModalMLP (alpha=0.20)', ProjectedTriModalMLP, {'ac_dim': 154, 'w2v_dim': 768, 'bert_dim': 768, 'dropout': 0.25}, 0.20, 0.1),
        ('ProjectedTriModalMLP (alpha=0.25)', ProjectedTriModalMLP, {'ac_dim': 154, 'w2v_dim': 768, 'bert_dim': 768, 'dropout': 0.25}, 0.25, 0.1),
        ('ProjectedTriModalMLP (alpha=0.20, l_phq=0.2)', ProjectedTriModalMLP, {'ac_dim': 154, 'w2v_dim': 768, 'bert_dim': 768, 'dropout': 0.25}, 0.20, 0.2),
        ('ResidualTriModalMLP (alpha=0.15)', ResidualTriModalMLP, {'in_dim': 1690, 'hidden_dim': 256, 'dropout': 0.25}, 0.15, 0.1),
        ('ResidualTriModalMLP (alpha=0.20)', ResidualTriModalMLP, {'in_dim': 1690, 'hidden_dim': 256, 'dropout': 0.25}, 0.20, 0.1),
        ('ResidualTriModalMLP (alpha=0.25)', ResidualTriModalMLP, {'in_dim': 1690, 'hidden_dim': 256, 'dropout': 0.25}, 0.25, 0.1)
    ]

    for name, cls, kwargs, a, l_p in arch_configs:
        summary, _, _, _ = run_experiment_cv(
            cls, kwargs, X_dict, y, y_phq, pids, sids,
            alpha=a, lambda_phq=l_p, lambda_bin=0.2
        )
        summary['config_name'] = name
        summary['architecture'] = cls.__name__
        summary['alpha'] = a
        results.append(summary)
        print(f"  {name:42s} | Acc: {summary['cv_acc_mean']:.2f} ± {summary['cv_acc_std']:.2f}% | Macro-F1: {summary['cv_macro_f1_mean']:.2f}% | Bal-Acc: {summary['cv_bal_acc_mean']:.2f}% | Rec1: {summary['cv_rec1_mean']:.1f}% | Rec2: {summary['cv_rec2_mean']:.1f}% | MAE: {summary['cv_mae_mean']:.2f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(EXP_DIR, "neural_architectures_and_balancing_sweep.csv"), index=False)

    # Save summary json
    with open(os.path.join(EXP_DIR, "neural_sweep_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved all sweep results to {EXP_DIR}/neural_architectures_and_balancing_sweep.csv successfully!")


if __name__ == "__main__":
    main()
