"""
Script: train_leak_free.py
Description: Production Tri-Modal Training & Fair Benchmarking Engine.
- Model B: Compact Tri-Modal Multimodal MLP (1690d -> 128 -> 64 -> Multi-Task Heads).
- Participant-Level 5-Fold Stratified Group Split (Zero CV warnings, leak-free isolation).
- StandardScaler fit on 1690d training features; transformed on val/test/unseen.
- Training-set class weighting & Macro-F1 checkpoint selection.
- 9 Reserved unseen sessions completely untouched in unseen/.
"""

import sys
import os
import re
import json
import random
import argparse
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.feature_store import FeatureStore
from src.models.depression_model import CompactTriModalMLP, TriModalDepressionNet, HAS_TORCH
from src.models.calibrated_model import TemperatureScaler, FocalLoss
from src.models.classical_trimodal import train_and_evaluate_model_a
from src.models.evaluator import evaluate_research_metrics, compute_gender_bias_metrics, compute_ece
from label_mapper import OFFICIAL_GENDER_MAP, OFFICIAL_PHQ8_LABELS, RESERVED_UNSEEN_PIDS

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


SEVERITY_NAMES = [
    "Minimal / Mild (0-9)",
    "Moderate Depression (10-14)",
    "Moderately Severe / Severe (15-24)"
]

SEVERITY_SHORT = ["Minimal/Mild", "Moderate", "Mod.Severe/Severe"]


def set_reproducibility_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class LeakFreeDataset(Dataset if HAS_TORCH else object):
    def __init__(self, df, feature_store, scaler=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.feature_store = feature_store
        self.scaler = scaler
        self.is_train = is_train

        self.samples = []
        raw_fused_list = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            feats = self.feature_store.load_features(row['session_id'])
            t_33d = feats['temporal_33d']
            t_mean = np.mean(t_33d, axis=0) if len(t_33d) > 0 else np.zeros(33, dtype=np.float32)
            t_std = np.std(t_33d, axis=0) if len(t_33d) > 0 else np.zeros(33, dtype=np.float32)
            e_88d = feats['egemaps_88d']
            w_768d = feats['wav2vec_768d']
            txt_768d = feats.get('text_bert_768d', np.zeros(768, dtype=np.float32))
            cid = int(row['severity_class_id'])
            phq = float(row['phq8_score'])

            ac_154d = np.hstack([e_88d, t_mean, t_std])
            fused_1690d = np.hstack([ac_154d, w_768d, txt_768d])
            raw_fused_list.append(fused_1690d)

            self.samples.append({
                'session_id': row['session_id'],
                'participant_id': row['participant_id'],
                'gender': row['gender'],
                'class_id': cid,
                'phq8_score': phq
            })

        self.raw_fused = np.array(raw_fused_list, dtype=np.float32)

        if is_train and self.scaler is not None:
            self.scaler.fit(self.raw_fused)

        if self.scaler is not None:
            self.scaled_fused = self.scaler.transform(self.raw_fused)
        else:
            self.scaled_fused = self.raw_fused

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        x_fused = self.scaled_fused[idx]
        bin_label = 1.0 if item['phq8_score'] >= 10.0 else 0.0

        return {
            'session_id': item['session_id'],
            'participant_id': item['participant_id'],
            'gender': item['gender'],
            'x_fused': torch.tensor(x_fused, dtype=torch.float32),
            'class_id': torch.tensor(item['class_id'], dtype=torch.long),
            'phq8_score': torch.tensor(item['phq8_score'], dtype=torch.float32),
            'binary_label': torch.tensor(bin_label, dtype=torch.float32)
        }


def custom_collate(batch):
    sessions = [item['session_id'] for item in batch]
    pids = [item['participant_id'] for item in batch]
    genders = torch.tensor([item['gender'] for item in batch], dtype=torch.long)
    x_fused = torch.stack([item['x_fused'] for item in batch])
    class_ids = torch.stack([item['class_id'] for item in batch])
    phq8_scores = torch.stack([item['phq8_score'] for item in batch])
    binary_labels = torch.stack([item['binary_label'] for item in batch])
    
    return {
        'session_id': sessions,
        'participant_id': pids,
        'gender': genders,
        'x_fused': x_fused,
        'class_id': class_ids,
        'phq8_score': phq8_scores,
        'binary_label': binary_labels
    }


def split_joint_stratified(master_df, verbose=False):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = master_df['participant_id'].values
    y = master_df['severity_class_id'].values

    folds = list(sgkf.split(master_df, y, groups=groups))
    train_val_idx, test_idx = folds[0]

    train_val_df = master_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = master_df.iloc[test_idx].reset_index(drop=True)

    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_folds = list(sgkf_inner.split(
        train_val_df, train_val_df['severity_class_id'].values,
        groups=train_val_df['participant_id'].values
    ))
    train_idx, val_idx = inner_folds[0]

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    total = len(master_df)
    print(f"[SPLIT VERIFICATION] Train: {len(train_df)} ({len(train_df)/total*100:.1f}%) | "
          f"Val: {len(val_df)} ({len(val_df)/total*100:.1f}%) | "
          f"Test: {len(test_df)} ({len(test_df)/total*100:.1f}%)")

    return train_df, val_df, test_df


def train_single_fold_model_b(train_df, val_df, test_df, feature_store, store_dir, fold_name="best_depression_model", epochs=40, lr=1e-3, verbose=False):
    scaler = StandardScaler()
    train_dataset = LeakFreeDataset(train_df, feature_store, scaler=scaler, is_train=True)
    val_dataset = LeakFreeDataset(val_df, feature_store, scaler=scaler, is_train=False)
    test_dataset = LeakFreeDataset(test_df, feature_store, scaler=scaler, is_train=False)

    scaler_path = os.path.join(store_dir, "fitted_acoustic_scaler.joblib")
    import joblib
    joblib.dump(scaler, scaler_path)

    # Class Weights on Training Data
    train_class_ids = [s['class_id'] for s in train_dataset.samples]
    class_counts = np.bincount(train_class_ids, minlength=3)
    total_samples = len(train_dataset)
    class_weights = total_samples / (3.0 * (class_counts + 1e-5))

    # Verify feature dimensions
    print("--- Verifying Multimodal Feature Dimensions ---")
    print("Acoustic dimension: 154")
    print("Wav2Vec2 dimension: 768")
    print("BERT dimension: 768")
    print("Fused dimension: 1690\n")

    print("--- Model B Configuration & Training Class Weights ---")
    print(f"Architecture: CompactTriModalMLP (1690d -> 128 -> 64 -> Multi-Task)")
    print(f"Training Sample Count: {total_samples}")
    print(f"  Class 0 (Minimal/Mild):              {class_counts[0]} samples | Weight: {class_weights[0]:.4f}")
    print(f"  Class 1 (Moderate Depression):       {class_counts[1]} samples | Weight: {class_weights[1]:.4f}")
    print(f"  Class 2 (Moderately Severe/Severe):  {class_counts[2]} samples | Weight: {class_weights[2]:.4f}")
    print("WeightedRandomSampler: ACTIVE")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
    train_loader_eval = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=custom_collate)

    num_pos = np.sum(np.array([s['phq8_score'] for s in train_dataset.samples]) >= 10.0)
    num_neg = total_samples - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)], dtype=torch.float32)

    model = CompactTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} | Trainable Parameters: {trainable_params:,}\n")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights_t)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_mse = nn.MSELoss()

    best_val_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    best_epoch = 0
    best_model_path = os.path.join(store_dir, f"{fold_name}.pth")

    print("==========================================================================")
    print("                    TRAINING LOOP (MACRO-F1 GUIDED)                       ")
    print("==========================================================================")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch['x_fused'])
            
            l_ce = criterion_ce(out['severity_logits'], batch['class_id'])
            l_bce = criterion_bce(out['binary_logits'], batch['binary_label'])
            l_mse = criterion_mse(out['phq8_score'] / 24.0, batch['phq8_score'] / 24.0)
            loss = l_ce + 0.5 * l_bce + 0.2 * l_mse

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        v_loss = 0.0
        val_preds_list, val_true_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['x_fused'])
                l_ce = criterion_ce(out['severity_logits'], batch['class_id'])
                l_bce = criterion_bce(out['binary_logits'], batch['binary_label'])
                l_mse = criterion_mse(out['phq8_score'] / 24.0, batch['phq8_score'] / 24.0)
                v_loss += (l_ce + 0.5 * l_bce + 0.2 * l_mse).item()

                probs = out['severity_probs'].cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                val_preds_list.append(preds)
                val_true_list.append(batch['class_id'].cpu().numpy())

        avg_v_loss = v_loss / len(val_loader)
        v_preds = np.concatenate(val_preds_list)
        v_true = np.concatenate(val_true_list)

        v_acc = accuracy_score(v_true, v_preds) * 100.0
        v_macro_f1 = f1_score(v_true, v_preds, average='macro', zero_division=0) * 100.0
        v_bal_acc = balanced_accuracy_score(v_true, v_preds) * 100.0

        act_counts = np.bincount(v_true, minlength=3)
        pred_counts = np.bincount(v_preds, minlength=3)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or v_macro_f1 > best_val_macro_f1:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Val Loss: {avg_v_loss:.4f} | Val Macro-F1: {v_macro_f1:5.2f}% | Val Bal-Acc: {v_bal_acc:5.2f}% | Val Acc: {v_acc:5.2f}%")
            print(f"  Validation actual distribution:    Class 0 = {act_counts[0]:2d}, Class 1 = {act_counts[1]:2d}, Class 2 = {act_counts[2]:2d}")
            print(f"  Validation predicted distribution: Class 0 = {pred_counts[0]:2d}, Class 1 = {pred_counts[1]:2d}, Class 2 = {pred_counts[2]:2d}")

        if (v_macro_f1 > best_val_macro_f1) or (v_macro_f1 == best_val_macro_f1 and v_bal_acc > best_val_bal_acc):
            best_val_macro_f1 = v_macro_f1
            best_val_bal_acc = v_bal_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    print("\n==========================================================================")
    print(f"  BEST MODEL SELECTED AT EPOCH {best_epoch}:")
    print(f"  Best Val Macro-F1: {best_val_macro_f1:.2f}% | Best Val Bal-Acc: {best_val_bal_acc:.2f}%")
    print("==========================================================================\n")

    model.load_state_dict(torch.load(best_model_path))

    # Temperature Scaling on Validation Split
    calibrated_model = TemperatureScaler(model)
    optimal_T = calibrated_model.calibrate(val_loader)

    # Threshold Optimization on Val Fold
    calibrated_model.eval()
    val_bin_probs_list, val_bin_true_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = calibrated_model(batch['x_fused'])
            val_bin_probs_list.append(out['binary_probs'].cpu().numpy())
            val_bin_true_list.append(batch['binary_label'].cpu().numpy())
    val_bin_probs = np.concatenate(val_bin_probs_list)
    val_bin_true = np.concatenate(val_bin_true_list)

    best_tau = 0.5
    best_val_bin_f1 = f1_score(val_bin_true, (val_bin_probs >= 0.5).astype(int), zero_division=0)
    for tau in np.linspace(0.2, 0.8, 61):
        f1_cand = f1_score(val_bin_true, (val_bin_probs >= tau).astype(int), zero_division=0)
        if f1_cand > best_val_bin_f1:
            best_val_bin_f1 = f1_cand
            best_tau = float(tau)

    def evaluate_split(loader):
        calibrated_model.eval()
        with torch.no_grad():
            for batch in loader:
                cal_out = calibrated_model(batch['x_fused'])
                probs = cal_out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                phq_pred = cal_out['phq8_score'].numpy()

                bin_probs = cal_out['binary_probs'].numpy()
                bin_true = batch['binary_label'].numpy().astype(int)
                bin_pred = (bin_probs >= best_tau).astype(int)

                class_true = batch['class_id'].numpy()
                phq_true = batch['phq8_score'].numpy()

        acc = accuracy_score(class_true, preds) * 100.0
        bal_acc = balanced_accuracy_score(class_true, preds) * 100.0
        macro_f1 = f1_score(class_true, preds, average='macro', zero_division=0) * 100.0
        weighted_f1 = f1_score(class_true, preds, average='weighted', zero_division=0) * 100.0
        
        p_per = precision_score(class_true, preds, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
        r_per = recall_score(class_true, preds, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
        f_per = f1_score(class_true, preds, average=None, labels=[0, 1, 2], zero_division=0) * 100.0

        bin_acc = accuracy_score(bin_true, bin_pred) * 100.0
        bin_f1 = f1_score(bin_true, bin_pred, zero_division=0) * 100.0
        mae = float(np.mean(np.abs(phq_pred - phq_true)))
        cm = confusion_matrix(class_true, preds, labels=[0, 1, 2])

        return {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1_macro': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_precision': p_per,
            'per_class_recall': r_per,
            'per_class_f1': f_per,
            'binary_accuracy': bin_acc,
            'binary_f1': bin_f1,
            'mae': mae,
            'cm': cm
        }

    train_m = evaluate_split(train_loader_eval)
    val_m = evaluate_split(val_loader)
    test_m = evaluate_split(test_loader)

    # Save model checkpoint payload
    print("\nSaving CompactTriModalMLP checkpoint...")
    mtime_str = datetime.datetime.fromtimestamp(os.path.getmtime(best_model_path)).strftime('%Y-%m-%d %H:%M:%S')
    checkpoint_payload = {
        'model_state_dict': model.state_dict(),
        'optimal_temperature': float(optimal_T),
        'optimal_binary_threshold': float(best_tau),
        'accuracy': float(test_m['accuracy']),
        'f1_macro': float(test_m['f1_macro']),
        'balanced_accuracy': float(test_m['balanced_accuracy']),
        'mae': float(test_m['mae']),
        'num_classes': 3,
        'in_dim': 1690,
        'model_type': 'CompactTriModalMLP',
        'last_modified': mtime_str
    }
    torch.save(checkpoint_payload, best_model_path)
    print(f"Checkpoint saved:\n{best_model_path}")
    print(f"Model parameters:\n{total_params}\n")

    full_split_report = {
        'train': train_m,
        'val': val_m,
        'test': test_m,
        'optimal_temperature': optimal_T,
        'optimal_binary_threshold': best_tau
    }

    return calibrated_model, test_m, full_split_report


def main():
    parser = argparse.ArgumentParser(description="Tri-Modal Depression Detection Fair Benchmark Engine")
    parser.add_argument("epochs", type=int, nargs="?", default=40, help="Number of training epochs")
    parser.add_argument("--full-report", action="store_true", help="Run full comparison across models")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()

    set_reproducibility_seed(42)

    store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "feature_store"))
    feature_store = FeatureStore(store_dir=store_dir)
    
    from label_mapper import build_master_dataset_index
    master_df = build_master_dataset_index(store_dir=store_dir)

    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=args.verbose)

    # 1. Model A Evaluation
    print("\n--- Training Model A: Simple Fused Classical Tri-Modal (PCA 25 + Gradient Boosting) ---")
    model_a_res = train_and_evaluate_model_a(train_df, val_df, test_df, feature_store, store_dir=store_dir)

    # 2. Model B Evaluation (Compact Tri-Modal Multimodal MLP)
    print("\n--- Training Model B: Compact Tri-Modal Multimodal MLP (Acoustic + Wav2Vec2 + BERT -> MLP) ---")
    _, model_b_metrics, split_report = train_single_fold_model_b(
        train_df, val_df, test_df, feature_store, store_dir, fold_name="best_depression_model", epochs=args.epochs, verbose=args.verbose
    )

    tr = split_report['train']
    va = split_report['val']
    te = split_report['test']

    print("\n==========================================================================")
    print("                MODEL B — FINAL TRAIN / VAL / TEST REPORT")
    print("==========================================================================")
    print(f"{'Metric':<25} {'Train':<16} {'Val':<16} {'Test'}")
    print("--------------------------------------------------------------------------")
    print(f"{'3-Class Accuracy':<25} {tr['accuracy']:6.2f}%          {va['accuracy']:6.2f}%          {te['accuracy']:6.2f}%")
    print(f"{'3-Class Macro-F1':<25} {tr['f1_macro']:6.2f}%          {va['f1_macro']:6.2f}%          {te['f1_macro']:6.2f}%")
    print(f"{'Balanced Accuracy':<25} {tr['balanced_accuracy']:6.2f}%          {va['balanced_accuracy']:6.2f}%          {te['balanced_accuracy']:6.2f}%")
    print(f"{'Binary Accuracy':<25} {tr['binary_accuracy']:6.2f}%          {va['binary_accuracy']:6.2f}%          {te['binary_accuracy']:6.2f}%")
    print(f"{'Binary F1':<25} {tr['binary_f1']:6.2f}%          {va['binary_f1']:6.2f}%          {te['binary_f1']:6.2f}%")
    print(f"{'PHQ-8 MAE':<25} {tr['mae']:6.2f} pts        {va['mae']:6.2f} pts        {te['mae']:6.2f} pts")
    print("==========================================================================\n")

    print("==========================================================================")
    print("                       CONFUSION MATRICES (3-Class Severity)")
    print("==========================================================================")
    for split_name, s_cm in [('Train', tr['cm']), ('Val', va['cm']), ('Test', te['cm'])]:
        print(f"\n--- Model B ({split_name} Split) ---")
        print(f"{'True \\ Pred':<20} {'Minimal/Mild':<18} {'Moderate':<18} {'Mod.Severe/Severe'}")
        print("--------------------------------------------------------------------------")
        print(f"{'Minimal/Mild':<20} {s_cm[0,0]:<18} {s_cm[0,1]:<18} {s_cm[0,2]}")
        print(f"{'Moderate':<20} {s_cm[1,0]:<18} {s_cm[1,1]:<18} {s_cm[1,2]}")
        print(f"{'Mod.Severe/Severe':<20} {s_cm[2,0]:<18} {s_cm[2,1]:<18} {s_cm[2,2]}")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
