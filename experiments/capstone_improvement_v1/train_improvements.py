"""
Phase 6: Targeted Model Improvements & Fair Benchmarking Suite.
Evaluates candidate improvements on the exact canonical benchmark split:
- Candidate 1: CompactTriModalMLP (Clean Labels + Standard Loss)
- Candidate 2: CompactTriModalMLP (Smoothed Square-Root Balanced Loss)
- Candidate 3: ProjectedTriModalNet (Modality-Specific Projections + Gated Fusion)
- Candidate 4: ProjectedTriModalNet + ClassBalancedFocalLoss
- Candidate 5: ProjectedTriModalNet + Bounded PHQ-8 Output [0, 24] + Multi-Task Tuned
Saves all checkpoints, training curves, confusion matrices, and metrics.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from src.models.calibrated_model import TemperatureScaler


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultimodalDataset(Dataset):
    def __init__(self, X_ac, X_w2v, X_bert, y_class, y_phq, pids, sids):
        self.X_ac = torch.tensor(X_ac, dtype=torch.float32)
        self.X_w2v = torch.tensor(X_w2v, dtype=torch.float32)
        self.X_bert = torch.tensor(X_bert, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_phq = torch.tensor(y_phq, dtype=torch.float32)
        self.y_bin = torch.tensor((y_phq >= 10.0).astype(float), dtype=torch.float32)
        self.pids = pids
        self.sids = sids

    def __len__(self):
        return len(self.X_ac)

    def __getitem__(self, idx):
        return {
            'x_ac': self.X_ac[idx],
            'x_w2v': self.X_w2v[idx],
            'x_bert': self.X_bert[idx],
            'x_fused': torch.cat([self.X_ac[idx], self.X_w2v[idx], self.X_bert[idx]], dim=-1),
            'class_id': self.y_class[idx],
            'phq8_score': self.y_phq[idx],
            'binary_label': self.y_bin[idx],
            'participant_id': self.pids[idx],
            'session_id': self.sids[idx]
        }


# Model Architecture 1: Standard CompactTriModalMLP
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


# Model Architecture 2: ProjectedTriModalNet with Modality Projections and Gating
class ProjectedTriModalNet(nn.Module):
    def __init__(self, ac_dim=154, w2v_dim=768, bert_dim=768, num_classes=3, dropout=0.25, bounded_phq=True):
        super().__init__()
        self.bounded_phq = bounded_phq

        # Modality-specific projections
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

        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )

        # Joint reasoning backbone
        self.backbone = nn.Sequential(
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
            x_ac = batch['x_ac']
            x_w2v = batch['x_w2v']
            x_bert = batch['x_bert']
        else:
            x_ac = batch[:, :154]
            x_w2v = batch[:, 154:154+768]
            x_bert = batch[:, 154+768:]

        h_ac = self.proj_ac(x_ac)
        h_w2v = self.proj_w2v(x_w2v)
        h_bert = self.proj_bert(x_bert)

        h_cat = torch.cat([h_ac, h_w2v, h_bert], dim=-1)
        g = self.gate(h_cat)
        h_gated = h_cat * g

        rep = self.backbone(h_gated)
        logits = self.head_class(rep)
        bin_logit = self.head_bin(rep).squeeze(-1)

        raw_phq = self.head_phq(rep).squeeze(-1)
        if self.bounded_phq:
            # Constrain to strictly valid PHQ-8 range [0.0, 24.0]
            phq = 24.0 * torch.sigmoid(raw_phq)
        else:
            phq = raw_phq

        return {
            'severity_logits': logits,
            'severity_probs': F.softmax(logits, dim=-1),
            'phq8_score': phq,
            'binary_logits': bin_logit,
            'binary_probs': torch.sigmoid(bin_logit)
        }


def load_dataset_features(df, feature_store, clean_labels=True):
    clean_labels_df = pd.read_csv(os.path.join(BASE_DIR, "new_audio_csv", "detailed_lables.csv"))
    clean_map = dict(zip(clean_labels_df['Participant'], clean_labels_df['Depression_severity']))

    ac_list, w2v_list, bert_list, y_class, y_phq, pids, sids = [], [], [], [], [], [], []

    for idx, row in df.iterrows():
        sid = str(row['session_id'])
        pid = int(row['participant_id'])
        f = feature_store.load_features(sid)

        t = f['temporal_33d']
        t_mean = np.mean(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        t_std = np.std(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        e = f['egemaps_88d']
        w = f['wav2vec_768d']
        b = f.get('text_bert_768d', np.zeros(768, dtype=np.float32))

        ac_154d = np.hstack([e, t_mean, t_std])
        ac_list.append(ac_154d)
        w2v_list.append(w)
        bert_list.append(b)

        if clean_labels and pid in clean_map:
            phq = float(clean_map[pid])
            cls_id = 0 if phq <= 9.0 else (1 if phq <= 14.0 else 2)
        else:
            phq = float(row['phq8_score'])
            cls_id = int(row['severity_class_id'])

        y_class.append(cls_id)
        y_phq.append(phq)
        pids.append(pid)
        sids.append(sid)

    return {
        'ac': np.array(ac_list, dtype=np.float32),
        'w2v': np.array(w2v_list, dtype=np.float32),
        'bert': np.array(bert_list, dtype=np.float32),
        'y_class': np.array(y_class, dtype=int),
        'y_phq': np.array(y_phq, dtype=np.float32),
        'pids': pids,
        'sids': sids
    }


def compute_metrics(y_true, y_pred, y_prob, phq_true, phq_pred, y_bin_true, y_bin_pred, y_bin_prob):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    bin_acc = float(accuracy_score(y_bin_true, y_bin_pred) * 100.0)
    bin_p = float(precision_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_r = float(recall_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    p_curve, r_curve, _ = precision_recall_curve(y_bin_true, y_bin_prob)
    pr_auc = float(auc(r_curve, p_curve) * 100.0)

    mae = float(mean_absolute_error(phq_true, phq_pred))
    rmse = float(np.sqrt(mean_squared_error(phq_true, phq_pred)))

    # Calibration ECE
    bin_boundaries = np.linspace(0, 1, 11)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true)
    ece = 0.0
    for i in range(10):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.mean(in_bin) > 0:
            ece += np.abs(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin])) * np.mean(in_bin)
    ece = float(ece * 100.0)

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'weighted_f1': weighted_f1,
        'per_class_precision': [float(x) for x in p_per],
        'per_class_recall': [float(x) for x in r_per],
        'per_class_f1': [float(x) for x in f_per],
        'confusion_matrix': cm,
        'binary_accuracy': bin_acc,
        'binary_precision': bin_p,
        'binary_recall': bin_r,
        'binary_f1': bin_f1,
        'binary_pr_auc': pr_auc,
        'mae': mae,
        'rmse': rmse,
        'ece': ece
    }


def train_candidate(
    model, train_loader, val_loader, test_loader,
    criterion_ce, lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, weight_decay=1e-3,
    exp_name="candidate"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    best_state = None
    best_epoch = 0

    train_losses, val_macro_f1s = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss_c = criterion_ce(out['severity_logits'], batch['class_id'])
            loss_p = criterion_mse(out['phq8_score'], batch['phq8_score'])
            loss_b = criterion_bce(out['binary_logits'], batch['binary_label'])

            loss = loss_c + lambda_phq * loss_p + lambda_bin * loss_b
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_losses.append(total_loss / len(train_loader))

        # Validation evaluation for early stopping
        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                probs = out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                v_preds.append(preds)
                v_targets.append(batch['class_id'].numpy())

        v_preds = np.concatenate(v_preds)
        v_targets = np.concatenate(v_targets)
        v_f1 = f1_score(v_targets, v_preds, average='macro', zero_division=0) * 100.0
        v_bal = balanced_accuracy_score(v_targets, v_preds) * 100.0
        val_macro_f1s.append(v_f1)

        if (v_f1 > best_val_macro_f1) or (v_f1 == best_val_macro_f1 and v_bal > best_val_bal_acc):
            best_val_macro_f1 = v_f1
            best_val_bal_acc = v_bal
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Temperature Scaling on Validation Split
    calibrated_model = TemperatureScaler(model)
    opt_temp = calibrated_model.calibrate(val_loader)

    # Evaluate on Test Split
    def eval_split(loader):
        calibrated_model.eval()
        with torch.no_grad():
            for batch in loader:
                out = calibrated_model(batch)
                probs = out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                phq_pred = out['phq8_score'].numpy()
                bin_prob = out['binary_probs'].numpy()
                bin_pred = (bin_prob >= 0.5).astype(int)

                y_true = batch['class_id'].numpy()
                phq_true = batch['phq8_score'].numpy()
                bin_true = batch['binary_label'].numpy().astype(int)
                sids = batch['session_id']
                pids = batch['participant_id']

        m = compute_metrics(y_true, preds, probs, phq_true, phq_pred, bin_true, bin_pred, bin_prob)
        df_pred = pd.DataFrame({
            'session_id': sids,
            'participant_id': pids,
            'true_class': y_true,
            'pred_class': preds,
            'prob_0': probs[:, 0],
            'prob_1': probs[:, 1],
            'prob_2': probs[:, 2],
            'true_phq': phq_true,
            'pred_phq': phq_pred,
            'true_bin': bin_true,
            'pred_bin': bin_pred
        })
        return m, df_pred

    test_metrics, test_preds = eval_split(test_loader)
    val_metrics, _ = eval_split(val_loader)

    # Save checkpoint & curves
    ckpt_path = os.path.join(EXP_DIR, "checkpoints", f"{exp_name}.pth")
    torch.save({
        'model_state_dict': best_state,
        'optimal_temperature': opt_temp,
        'best_epoch': best_epoch,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }, ckpt_path)

    # Plot training curves
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='#2563eb')
    plt.title(f'{exp_name} - Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_macro_f1s, label='Val Macro-F1', color='#10b981')
    plt.axvline(best_epoch, color='#ef4444', linestyle='--', label=f'Best Ep {best_epoch}')
    plt.title(f'{exp_name} - Val Macro-F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, "plots", f"{exp_name}_curve.png"), dpi=150)
    plt.close()

    # Plot confusion matrix
    cm = np.array(test_metrics['confusion_matrix'])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title(f'{exp_name} - Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1, 2], ['Minimal', 'Moderate', 'Severe'])
    plt.yticks([0, 1, 2], ['Minimal', 'Moderate', 'Severe'])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black' if cm[i, j] < cm.max()/2 else 'white')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_DIR, "plots", f"{exp_name}_cm.png"), dpi=150)
    plt.close()

    test_preds.to_csv(os.path.join(EXP_DIR, "predictions", f"{exp_name}_test_preds.csv"), index=False)
    return test_metrics, val_metrics, best_epoch, opt_temp


def main():
    print("==========================================================================")
    print("      PHASE 6: TARGETED MODEL IMPROVEMENTS & ABLATION COMPARISON          ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    print("\nExtracting features with Clean Verified Ground-Truth Labels...")
    d_tr = load_dataset_features(train_df, fs, clean_labels=True)
    d_va = load_dataset_features(val_df, fs, clean_labels=True)
    d_te = load_dataset_features(test_df, fs, clean_labels=True)

    # Standardize acoustic, wav2vec, and bert strictly fit on train
    sc_ac = StandardScaler().fit(d_tr['ac'])
    sc_w2v = StandardScaler().fit(d_tr['w2v'])
    sc_bert = StandardScaler().fit(d_tr['bert'])

    tr_ac_s, va_ac_s, te_ac_s = sc_ac.transform(d_tr['ac']), sc_ac.transform(d_va['ac']), sc_ac.transform(d_te['ac'])
    tr_w2v_s, va_w2v_s, te_w2v_s = sc_w2v.transform(d_tr['w2v']), sc_w2v.transform(d_va['w2v']), sc_w2v.transform(d_te['w2v'])
    tr_b_s, va_b_s, te_b_s = sc_bert.transform(d_tr['bert']), sc_bert.transform(d_va['bert']), sc_bert.transform(d_te['bert'])

    # Save scalers for reproducible inference
    joblib.dump({'scaler_ac': sc_ac, 'scaler_w2v': sc_w2v, 'scaler_bert': sc_bert}, os.path.join(EXP_DIR, "checkpoints", "feature_scalers.joblib"))

    tr_ds = MultimodalDataset(tr_ac_s, tr_w2v_s, tr_b_s, d_tr['y_class'], d_tr['y_phq'], d_tr['pids'], d_tr['sids'])
    va_ds = MultimodalDataset(va_ac_s, va_w2v_s, va_b_s, d_va['y_class'], d_va['y_phq'], d_va['pids'], d_va['sids'])
    te_ds = MultimodalDataset(te_ac_s, te_w2v_s, te_b_s, d_te['y_class'], d_te['y_phq'], d_te['pids'], d_te['sids'])

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=len(te_ds), shuffle=False)

    class_counts = np.bincount(d_tr['y_class'], minlength=3)
    print(f"Training Class Counts: Class 0={class_counts[0]}, Class 1={class_counts[1]}, Class 2={class_counts[2]}")

    # Calculate balanced weights
    smooth_weights = np.sqrt(len(d_tr['y_class']) / (3.0 * (class_counts + 1e-5)))
    w_tensor = torch.tensor(smooth_weights, dtype=torch.float32)

    results_table = []

    # 1. Baseline Model on Clean Labels
    print("\n--- Training Candidate 1: StandardTriModalMLP (Clean Labels, Standard CE) ---")
    set_seed(42)
    m1 = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    t1_test, t1_val, ep1, t1 = train_candidate(
        m1, tr_loader, va_loader, te_loader,
        criterion_ce=nn.CrossEntropyLoss(),
        exp_name="cand1_standard_mlp_clean"
    )
    t1_test['candidate'] = '1. Standard MLP (Clean Labels)'
    results_table.append(t1_test)

    # 2. Standard MLP with Smoothed Class-Balanced Loss
    print("\n--- Training Candidate 2: StandardTriModalMLP (Smoothed Inverse Weights) ---")
    set_seed(42)
    m2 = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    t2_test, t2_val, ep2, t2 = train_candidate(
        m2, tr_loader, va_loader, te_loader,
        criterion_ce=nn.CrossEntropyLoss(weight=w_tensor),
        exp_name="cand2_standard_mlp_balanced"
    )
    t2_test['candidate'] = '2. Standard MLP (Balanced Loss)'
    results_table.append(t2_test)

    # 3. ProjectedTriModalNet (Modality Projections + Gated Fusion)
    print("\n--- Training Candidate 3: ProjectedTriModalNet (Modality-Specific Projections) ---")
    set_seed(42)
    m3 = ProjectedTriModalNet(ac_dim=154, w2v_dim=768, bert_dim=768, num_classes=3, dropout=0.25, bounded_phq=False)
    t3_test, t3_val, ep3, t3 = train_candidate(
        m3, tr_loader, va_loader, te_loader,
        criterion_ce=nn.CrossEntropyLoss(weight=w_tensor),
        exp_name="cand3_projected_trimodal_net"
    )
    t3_test['candidate'] = '3. ProjectedTriModalNet (Gated Fusion)'
    results_table.append(t3_test)

    # 4. ProjectedTriModalNet with Focal Loss
    print("\n--- Training Candidate 4: ProjectedTriModalNet + Focal Loss (gamma=1.5) ---")
    set_seed(42)
    m4 = ProjectedTriModalNet(ac_dim=154, w2v_dim=768, bert_dim=768, num_classes=3, dropout=0.25, bounded_phq=False)
    t4_test, t4_val, ep4, t4 = train_candidate(
        m4, tr_loader, va_loader, te_loader,
        criterion_ce=FocalLoss(alpha=w_tensor, gamma=1.5),
        exp_name="cand4_projected_focal_loss"
    )
    t4_test['candidate'] = '4. ProjectedNet + Focal Loss'
    results_table.append(t4_test)

    # 5. ProjectedTriModalNet + Bounded PHQ-8 Output [0, 24] + Multi-Task Optimization
    print("\n--- Training Candidate 5: ProjectedTriModalNet (Bounded PHQ-8 + Tuned Multi-Task) ---")
    set_seed(42)
    m5 = ProjectedTriModalNet(ac_dim=154, w2v_dim=768, bert_dim=768, num_classes=3, dropout=0.20, bounded_phq=True)
    t5_test, t5_val, ep5, t5 = train_candidate(
        m5, tr_loader, va_loader, te_loader,
        criterion_ce=FocalLoss(alpha=w_tensor, gamma=1.2),
        lambda_phq=0.08, lambda_bin=0.2, epochs=45, lr=8e-4,
        exp_name="cand5_projected_bounded_multitask"
    )
    t5_test['candidate'] = '5. ProjectedNet Bounded Multi-Task'
    results_table.append(t5_test)

    print("\n" + "="*125)
    print(f"{'Candidate Architecture':35s} | {'3-Class Acc':11s} | {'Macro-F1':9s} | {'Bal-Acc':9s} | {'Mod Rec':9s} | {'Sev Rec':9s} | {'PHQ MAE':9s}")
    print("="*125)
    for r in results_table:
        print(f"{r['candidate']:35s} | {r['accuracy']:6.2f}%    | {r['macro_f1']:6.2f}%   | {r['balanced_accuracy']:6.2f}%   | {r['per_class_recall'][1]:6.2f}%   | {r['per_class_recall'][2]:6.2f}%   | {r['mae']:6.2f} pts")
    print("="*125)

    df_out = pd.DataFrame(results_table)
    out_csv = os.path.join(EXP_DIR, "experiment_results.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nAll candidate experiment results saved to: {out_csv}")


if __name__ == "__main__":
    main()
