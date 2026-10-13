"""
Phase 5 & 6: Systematic Development Cross-Validation & Architecture Comparison.
Tests 4 candidate configurations across the exact same 5 stratified grouped folds on development data (N=146):
1. Cand 2A: Mild Balanced Weights (alpha=0.25)
2. Cand 2B: Threshold-Optimized Unweighted Model (Decision Boundary Tuning on OOF)
3. Cand 6: Modality-Specific Projected Network (154->64, 768->64, 768->64 -> 192d Latent)
4. Cand 7: Modality Projections + Mild Balanced Weights + Multi-Task PHQ Loss
Saves:
- experiments/capstone_improvement_v1/model_comparison_cv_results.csv
- experiments/capstone_improvement_v1/model_comparison_cv_summary.json
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
    confusion_matrix, mean_absolute_error, mean_squared_error
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))
sys.path.insert(0, EXP_DIR)

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from train_improvements import StandardTriModalMLP, load_dataset_features


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FusedDataset(Dataset):
    def __init__(self, x_ac, x_w2v, x_bert, y_cls, y_phq, pids, sids):
        self.x_ac = torch.tensor(x_ac, dtype=torch.float32)
        self.x_w2v = torch.tensor(x_w2v, dtype=torch.float32)
        self.x_bert = torch.tensor(x_bert, dtype=torch.float32)
        self.x_cat = torch.cat([self.x_ac, self.x_w2v, self.x_bert], dim=-1)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_phq = torch.tensor(y_phq, dtype=torch.float32)
        self.pids = pids
        self.sids = sids

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        phq = self.y_phq[idx].item()
        return {
            'x_ac': self.x_ac[idx],
            'x_w2v': self.x_w2v[idx],
            'x_bert': self.x_bert[idx],
            'x_cat': self.x_cat[idx],
            'class_id': self.y_cls[idx],
            'phq8_score': self.y_phq[idx],
            'binary_label': torch.tensor(1.0 if phq >= 10.0 else 0.0, dtype=torch.float32),
            'participant_id': self.pids[idx],
            'session_id': self.sids[idx]
        }


class ProjectedTriModalNet(nn.Module):
    """
    Modality-Specific Projection Architecture:
    Projects each modality to equal 64-dimensional latent subspace:
    - Acoustic (154d) -> 64d
    - Wav2Vec2 (768d) -> 64d
    - BERT (768d) -> 64d
    Total fused dimension: 192d. Prevents high-dim embeddings from overwhelming acoustics.
    """
    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        self.proj_ac = nn.Sequential(
            nn.Linear(154, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_w2v = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_bert = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        fused_dim = hidden_dim * 3 # 192
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_class = nn.Linear(64, num_classes)
        self.head_phq = nn.Linear(64, 1)
        self.head_bin = nn.Linear(64, 1)

    def forward(self, batch):
        if isinstance(batch, dict):
            h_ac = self.proj_ac(batch['x_ac'])
            h_w2v = self.proj_w2v(batch['x_w2v'])
            h_b = self.proj_bert(batch['x_bert'])
        else:
            h_ac = self.proj_ac(batch[:, :154])
            h_w2v = self.proj_w2v(batch[:, 154:154+768])
            h_b = self.proj_bert(batch[:, 154+768:])

        h_cat = torch.cat([h_ac, h_w2v, h_b], dim=-1)
        feat = self.fusion(h_cat)

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


def train_single_fold_general(model, train_loader, val_loader, criterion_ce, lambda_phq=0.2, epochs=40, lr=1e-3, weight_decay=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            l_ce = criterion_ce(out['severity_logits'], batch['class_id'])
            l_mse = criterion_mse(out['phq8_score'], batch['phq8_score'])
            l_bce = criterion_bce(out['binary_logits'], batch['binary_label'])
            loss = l_ce + lambda_phq * l_mse + 0.2 * l_bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                probs = out['severity_probs'].cpu().numpy()
                v_preds.append(np.argmax(probs, axis=-1))
                v_targets.append(batch['class_id'].cpu().numpy())

        v_preds = np.concatenate(v_preds)
        v_targets = np.concatenate(v_targets)
        v_f1 = f1_score(v_targets, v_preds, average='macro', zero_division=0) * 100.0
        v_acc = accuracy_score(v_targets, v_preds) * 100.0
        v_bal = balanced_accuracy_score(v_targets, v_preds) * 100.0
        
        # Selection criterion: Balance Macro-F1 + Balanced Accuracy + Accuracy
        v_score = 0.5 * v_f1 + 0.3 * v_bal + 0.2 * v_acc

        if v_score > best_val_score:
            best_val_score = v_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_metrics(y_true, y_pred, y_prob, phq_true, phq_pred):
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
        'f1_moderate': f_per[1],
        'f1_severe': f_per[2],
        'binary_accuracy': bin_acc,
        'binary_f1': bin_f1,
        'mae': mae,
        'rmse': rmse
    }


def run_candidate_benchmarks():
    print("==========================================================================")
    print("      5-FOLD CV BENCHMARKING OF CANDIDATE ARCHITECTURES (N=146 DEV)       ")
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

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        'Cand 2A (Mild Balanced Weights)': {'arch': 'standard', 'alpha': 0.25, 'lambda_phq': 0.2},
        'Cand 6 (Projected TriModalNet)': {'arch': 'projected', 'alpha': 0.0, 'lambda_phq': 0.2},
        'Cand 7 (ProjectedNet + Mild Balanced)': {'arch': 'projected', 'alpha': 0.25, 'lambda_phq': 0.2},
        'Cand 8 (ProjectedNet + MultiTask PHQ)': {'arch': 'projected', 'alpha': 0.35, 'lambda_phq': 0.4}
    }

    results_by_candidate = {k: [] for k in candidates.keys()}
    oof_predictions_by_candidate = {
        k: {'probs': np.zeros((len(y_class), 3)), 'phq': np.zeros(len(y_class))} for k in candidates.keys()
    }

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_ac, y_class, groups=pids), 1):
        print(f"\n--- FOLD {fold} / 5 (Train: {len(tr_idx)} | Val: {len(val_idx)}) ---")
        
        # Preprocessing strictly on train
        sc_ac = StandardScaler().fit(X_ac[tr_idx])
        sc_w2v = StandardScaler().fit(X_w2v[tr_idx])
        sc_bert = StandardScaler().fit(X_bert[tr_idx])

        tr_ac_s, val_ac_s = sc_ac.transform(X_ac[tr_idx]), sc_ac.transform(X_ac[val_idx])
        tr_w2v_s, val_w2v_s = sc_w2v.transform(X_w2v[tr_idx]), sc_w2v.transform(X_w2v[val_idx])
        tr_b_s, val_b_s = sc_bert.transform(X_bert[tr_idx]), sc_bert.transform(X_bert[val_idx])

        tr_ds = FusedDataset(tr_ac_s, tr_w2v_s, tr_b_s, y_class[tr_idx], y_phq[tr_idx], pids[tr_idx], sids[tr_idx])
        val_ds = FusedDataset(val_ac_s, val_w2v_s, val_b_s, y_class[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        tr_counts = np.bincount(y_class[tr_idx], minlength=3)

        for c_name, c_cfg in candidates.items():
            set_seed(42 + fold)
            
            # Loss weights
            if c_cfg['alpha'] > 0:
                raw_w = (len(tr_idx) / (3.0 * (tr_counts + 1e-5))) ** c_cfg['alpha']
                ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(raw_w, dtype=torch.float32))
            else:
                ce_loss = nn.CrossEntropyLoss()

            # Architecture
            if c_cfg['arch'] == 'standard':
                # Wraps forward for dictionary
                class StandardWrap(StandardTriModalMLP):
                    def forward(self, batch):
                        if isinstance(batch, dict):
                            return super().forward(batch['x_cat'])
                        return super().forward(batch)
                m = StandardWrap(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
            else:
                m = ProjectedTriModalNet(hidden_dim=64, num_classes=3, dropout=0.3)

            m = train_single_fold_general(
                m, tr_loader, val_loader,
                criterion_ce=ce_loss,
                lambda_phq=c_cfg['lambda_phq'],
                epochs=40, lr=1e-3, weight_decay=1e-3
            )

            # Evaluate on validation fold
            val_batch = {
                'x_ac': torch.tensor(val_ac_s, dtype=torch.float32),
                'x_w2v': torch.tensor(val_w2v_s, dtype=torch.float32),
                'x_bert': torch.tensor(val_b_s, dtype=torch.float32),
                'x_cat': torch.tensor(np.hstack([val_ac_s, val_w2v_s, val_b_s]), dtype=torch.float32)
            }
            with torch.no_grad():
                out = m(val_batch)
                probs = out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                phq_pred = out['phq8_score'].numpy()

            oof_predictions_by_candidate[c_name]['probs'][val_idx] = probs
            oof_predictions_by_candidate[c_name]['phq'][val_idx] = phq_pred

            res = evaluate_metrics(y_class[val_idx], preds, probs, y_phq[val_idx], phq_pred)
            results_by_candidate[c_name].append(res)
            print(f"  {c_name:36s} -> Acc: {res['accuracy']:5.2f}% | Macro-F1: {res['macro_f1']:5.2f}% | Bal-Acc: {res['balanced_accuracy']:5.2f}% | MAE: {res['mae']:.2f}")

    # Summary Table across folds
    rows = []
    print("\n" + "="*95)
    print("               DEVELOPMENT 5-FOLD CV COMPARISON SUMMARY TABLE")
    print("="*95)
    print(f"{'Candidate Name':<38} | {'Accuracy':<14} | {'Macro-F1':<14} | {'Bal-Acc':<14} | {'PHQ MAE':<14}")
    print("-"*95)

    for c_name, res_list in results_by_candidate.items():
        acc_m = np.mean([r['accuracy'] for r in res_list])
        acc_s = np.std([r['accuracy'] for r in res_list])
        f1_m = np.mean([r['macro_f1'] for r in res_list])
        f1_s = np.std([r['macro_f1'] for r in res_list])
        bal_m = np.mean([r['balanced_accuracy'] for r in res_list])
        bal_s = np.std([r['balanced_accuracy'] for r in res_list])
        mae_m = np.mean([r['mae'] for r in res_list])
        mae_s = np.std([r['mae'] for r in res_list])
        
        row = {
            'candidate': c_name,
            'accuracy_mean': acc_m, 'accuracy_std': acc_s,
            'macro_f1_mean': f1_m, 'macro_f1_std': f1_s,
            'balanced_acc_mean': bal_m, 'balanced_acc_std': bal_s,
            'mae_mean': mae_m, 'mae_std': mae_s
        }
        rows.append(row)
        print(f"{c_name:<38} | {acc_m:5.2f} ± {acc_s:4.2f}%  | {f1_m:5.2f} ± {f1_s:4.2f}%  | {bal_m:5.2f} ± {bal_s:4.2f}%  | {mae_m:5.2f} ± {mae_s:4.2f} pts")
    print("="*95)

    # Threshold Optimization on Out-Of-Fold Probabilities
    print("\n--- Testing Post-Hoc Threshold Optimization on OOF Probabilities ---")
    for c_name in candidates.keys():
        oof_p = oof_predictions_by_candidate[c_name]['probs']
        oof_phq = oof_predictions_by_candidate[c_name]['phq']
        raw_preds = np.argmax(oof_p, axis=-1)
        raw_m = evaluate_metrics(y_class, raw_preds, oof_p, y_phq, oof_phq)

        # Optimize class threshold multiplier: w0*p0, w1*p1, w2*p2
        best_w1, best_w2 = 1.0, 1.0
        best_score = raw_m['macro_f1']
        for w1 in [1.0, 1.2, 1.5, 1.8, 2.0]:
            for w2 in [1.0, 1.2, 1.5, 1.8, 2.0]:
                adj_p = oof_p.copy()
                adj_p[:, 1] *= w1
                adj_p[:, 2] *= w2
                t_preds = np.argmax(adj_p, axis=-1)
                t_f1 = f1_score(y_class, t_preds, average='macro', zero_division=0) * 100.0
                t_bal = balanced_accuracy_score(y_class, t_preds) * 100.0
                t_acc = accuracy_score(y_class, t_preds) * 100.0
                if t_f1 > best_score and t_acc >= 65.0:
                    best_score = t_f1
                    best_w1, best_w2 = w1, w2

        adj_p = oof_p.copy()
        adj_p[:, 1] *= best_w1
        adj_p[:, 2] *= best_w2
        adj_preds = np.argmax(adj_p, axis=-1)
        adj_m = evaluate_metrics(y_class, adj_preds, oof_p, y_phq, oof_phq)
        print(f"Candidate: {c_name}")
        print(f"  Raw Argmax  -> Acc: {raw_m['accuracy']:5.2f}% | Macro-F1: {raw_m['macro_f1']:5.2f}% | Bal-Acc: {raw_m['balanced_accuracy']:5.2f}%")
        print(f"  Tuned (w1={best_w1}, w2={best_w2}) -> Acc: {adj_m['accuracy']:5.2f}% | Macro-F1: {adj_m['macro_f1']:5.2f}% | Bal-Acc: {adj_m['balanced_accuracy']:5.2f}%")

    # Save summary files
    df_rows = pd.DataFrame(rows)
    csv_path = os.path.join(EXP_DIR, "model_comparison_cv_results.csv")
    df_rows.to_csv(csv_path, index=False)

    json_path = os.path.join(EXP_DIR, "model_comparison_cv_summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n[SAVED] Candidate Comparison CSV: {csv_path}")
    print(f"[SAVED] Candidate Comparison JSON: {json_path}")


if __name__ == "__main__":
    run_candidate_benchmarks()
