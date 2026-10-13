"""
Final Training & Evaluation Script for High-Accuracy Depression Detection Candidates.
Trains candidates strictly on development train data (N=116),
validates and early-stops strictly on development validation data (N=30),
and calibrates temperature scaling on development validation data.

Evaluates candidates on:
1. Canonical Held-Out Test Set (N=37)
2. Repaired Reserved Unseen Cohort (N=33: 9 WoZ + 24 Autonomous AI)

Candidates evaluated:
- Baseline (app/feature_store/best_depression_model.pth)
- Candidate 2 (cand2_standard_mlp_balanced.pth, alpha=0.50)
- Candidate 2A (cand2a_standard_mlp_mild_balanced.pth, alpha=0.25)
- Candidate 3A (cand3a_standard_mlp_alpha_015.pth, alpha=0.15)
- Candidate 3B (cand3b_standard_mlp_alpha_018.pth, alpha=0.18)
- Candidate 3C (cand3c_projected_trimodal_net.pth, ProjectedNet alpha=0.20)
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
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_V1_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
EXP_V2_DIR = os.path.join(BASE_DIR, "experiments", "capstone_accuracy_v2")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from src.models.depression_model import CompactTriModalMLP
from src.models.calibrated_model import TemperatureScaler
from experiments.capstone_improvement_v1.train_improvements import load_dataset_features, MultimodalDataset
from experiments.capstone_accuracy_v2.run_neural_accuracy_sweep import (
    StandardTriModalMLP, ProjectedTriModalMLP, compute_metrics_full
)


def compute_comprehensive_metrics(y_true, y_pred, y_prob, phq_true, phq_pred):
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

    mae = float(mean_absolute_error(phq_true, phq_pred))
    rmse = float(np.sqrt(mean_squared_error(phq_true, phq_pred)))

    return {
        'samples': len(y_true),
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
        'mae': mae,
        'rmse': rmse
    }


def train_single_model(
    model, train_loader, val_loader, weights_tensor,
    lambda_phq=0.1, lambda_bin=0.2, epochs=40, lr=1e-3, seed=42
):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_ce = nn.CrossEntropyLoss(weight=weights_tensor)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_score = -1.0
    best_state = None
    best_epoch = 0

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

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                probs = out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                targets = batch['class_id'].numpy()

        f1 = f1_score(targets, preds, average='macro', zero_division=0) * 100.0
        acc = accuracy_score(targets, preds) * 100.0
        score = 0.5 * acc + 0.5 * f1
        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    cal = TemperatureScaler(model)
    opt_temp = cal.calibrate(val_loader)
    return cal, opt_temp, best_epoch


def main():
    print("==========================================================================")
    print("  FINAL TRAINING & EVALUATION OF HIGH-ACCURACY DEPRESSION DETECTORS       ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    d_tr = load_dataset_features(train_df, fs, clean_labels=True)
    d_va = load_dataset_features(val_df, fs, clean_labels=True)
    d_te = load_dataset_features(test_df, fs, clean_labels=True)

    # Standardize strictly on train
    sc_ac = StandardScaler().fit(d_tr['ac'])
    sc_w2v = StandardScaler().fit(d_tr['w2v'])
    sc_bert = StandardScaler().fit(d_tr['bert'])

    joblib.dump(
        {'scaler_ac': sc_ac, 'scaler_w2v': sc_w2v, 'scaler_bert': sc_bert},
        os.path.join(EXP_V2_DIR, "checkpoints", "feature_scalers_v2.joblib")
    )

    tr_ac_s, va_ac_s, te_ac_s = sc_ac.transform(d_tr['ac']), sc_ac.transform(d_va['ac']), sc_ac.transform(d_te['ac'])
    tr_w2v_s, va_w2v_s, te_w2v_s = sc_w2v.transform(d_tr['w2v']), sc_w2v.transform(d_va['w2v']), sc_w2v.transform(d_te['w2v'])
    tr_b_s, va_b_s, te_b_s = sc_bert.transform(d_tr['bert']), sc_bert.transform(d_va['bert']), sc_bert.transform(d_te['bert'])

    tr_ds = MultimodalDataset(tr_ac_s, tr_w2v_s, tr_b_s, d_tr['y_class'], d_tr['y_phq'], d_tr['pids'], d_tr['sids'])
    va_ds = MultimodalDataset(va_ac_s, va_w2v_s, va_b_s, d_va['y_class'], d_va['y_phq'], d_va['pids'], d_va['sids'])
    te_ds = MultimodalDataset(te_ac_s, te_w2v_s, te_b_s, d_te['y_class'], d_te['y_phq'], d_te['pids'], d_te['sids'])

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=len(te_ds), shuffle=False)

    counts = np.bincount(d_tr['y_class'], minlength=3)
    print(f"Train counts: {counts} (N={len(d_tr['y_class'])})")

    # Load Unseen Data
    unseen_path = os.path.join(EXP_V1_DIR, "unseen_feature_cache_repaired.npz")
    u_data = np.load(unseen_path, allow_pickle=True)
    u_pids = u_data['pids']
    u_feats = u_data['features']
    u_class = u_data['true_class']
    u_phq = u_data['true_phq']
    u_cohort = u_data['cohort']

    u_ac_s = sc_ac.transform(u_feats[:, :154])
    u_w2v_s = sc_w2v.transform(u_feats[:, 154:922])
    u_b_s = sc_bert.transform(u_feats[:, 922:1690])
    u_fused_s = np.hstack([u_ac_s, u_w2v_s, u_b_s])

    te_fused_s = np.hstack([te_ac_s, te_w2v_s, te_b_s])

    # -------------------------------------------------------------
    # Candidate 3A: StandardTriModalMLP (alpha=0.15)
    # -------------------------------------------------------------
    raw_w15 = (len(d_tr['y_class']) / (3.0 * (counts + 1e-5))) ** 0.15
    w15_t = torch.tensor(raw_w15 / np.mean(raw_w15), dtype=torch.float32)

    print("\n--- Training Candidate 3A: StandardTriModalMLP (alpha=0.15) ---")
    m3a = StandardTriModalMLP(in_dim=1690, hidden_dim=128, dropout=0.3)
    cal_3a, temp_3a, ep_3a = train_single_model(m3a, tr_loader, va_loader, w15_t, lambda_phq=0.1, lambda_bin=0.2, seed=42)
    print(f"Candidate 3A Best Epoch: {ep_3a}, Calibrated Temp: {temp_3a:.4f}")

    torch.save({
        'model_state_dict': cal_3a.model.state_dict(),
        'optimal_temperature': temp_3a,
        'best_epoch': ep_3a,
        'alpha': 0.15
    }, os.path.join(EXP_V2_DIR, "checkpoints", "cand3a_standard_mlp_alpha_015.pth"))

    # -------------------------------------------------------------
    # Candidate 3B: StandardTriModalMLP (alpha=0.18)
    # -------------------------------------------------------------
    raw_w18 = (len(d_tr['y_class']) / (3.0 * (counts + 1e-5))) ** 0.18
    w18_t = torch.tensor(raw_w18 / np.mean(raw_w18), dtype=torch.float32)

    print("\n--- Training Candidate 3B: StandardTriModalMLP (alpha=0.18) ---")
    m3b = StandardTriModalMLP(in_dim=1690, hidden_dim=128, dropout=0.3)
    cal_3b, temp_3b, ep_3b = train_single_model(m3b, tr_loader, va_loader, w18_t, lambda_phq=0.1, lambda_bin=0.2, seed=42)
    print(f"Candidate 3B Best Epoch: {ep_3b}, Calibrated Temp: {temp_3b:.4f}")

    torch.save({
        'model_state_dict': cal_3b.model.state_dict(),
        'optimal_temperature': temp_3b,
        'best_epoch': ep_3b,
        'alpha': 0.18
    }, os.path.join(EXP_V2_DIR, "checkpoints", "cand3b_standard_mlp_alpha_018.pth"))

    # -------------------------------------------------------------
    # Candidate 3C: ProjectedTriModalMLP (alpha=0.20, l_phq=0.2)
    # -------------------------------------------------------------
    raw_w20 = (len(d_tr['y_class']) / (3.0 * (counts + 1e-5))) ** 0.20
    w20_t = torch.tensor(raw_w20 / np.mean(raw_w20), dtype=torch.float32)

    print("\n--- Training Candidate 3C: ProjectedTriModalMLP (alpha=0.20, l_phq=0.2) ---")
    m3c = ProjectedTriModalMLP(ac_dim=154, w2v_dim=768, bert_dim=768, dropout=0.25)
    cal_3c, temp_3c, ep_3c = train_single_model(m3c, tr_loader, va_loader, w20_t, lambda_phq=0.2, lambda_bin=0.2, seed=42)
    print(f"Candidate 3C Best Epoch: {ep_3c}, Calibrated Temp: {temp_3c:.4f}")

    torch.save({
        'model_state_dict': cal_3c.model.state_dict(),
        'optimal_temperature': temp_3c,
        'best_epoch': ep_3c,
        'alpha': 0.20
    }, os.path.join(EXP_V2_DIR, "checkpoints", "cand3c_projected_trimodal_net.pth"))

    # -------------------------------------------------------------
    # Load Existing Baseline, Candidate 2, and Candidate 2A
    # -------------------------------------------------------------
    base_ckpt = torch.load(os.path.join(APP_DIR, "feature_store", "best_depression_model.pth"), map_location='cpu', weights_only=False)
    m_base = CompactTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    m_base.load_state_dict(base_ckpt['model_state_dict'])
    m_base.eval()
    cal_base = TemperatureScaler(m_base)
    cal_base.temperature.data.fill_(base_ckpt.get('optimal_temperature', 1.0))
    cal_base.eval()

    cand2_scalers = joblib.load(os.path.join(EXP_V1_DIR, "checkpoints", "feature_scalers.joblib"))
    cand2_ckpt = torch.load(os.path.join(EXP_V1_DIR, "checkpoints", "cand2_standard_mlp_balanced.pth"), map_location='cpu', weights_only=False)
    m_c2 = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    m_c2.load_state_dict(cand2_ckpt['model_state_dict'])
    m_c2.eval()
    cal_c2 = TemperatureScaler(m_c2)
    cal_c2.temperature.data.fill_(cand2_ckpt['optimal_temperature'])
    cal_c2.eval()

    cand2a_ckpt = torch.load(os.path.join(EXP_V1_DIR, "checkpoints", "cand2a_standard_mlp_mild_balanced.pth"), map_location='cpu', weights_only=False)
    m_c2a = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    m_c2a.load_state_dict(cand2a_ckpt['model_state_dict'])
    m_c2a.eval()
    cal_c2a = TemperatureScaler(m_c2a)
    cal_c2a.temperature.data.fill_(cand2a_ckpt['optimal_temperature'])
    cal_c2a.eval()

    # -------------------------------------------------------------
    # Evaluation Functions
    # -------------------------------------------------------------
    def run_eval(model_obj, is_dict, X_fused, X_ac, X_w2v, X_bert, y_true, phq_true):
        model_obj.eval()
        with torch.no_grad():
            if is_dict:
                inp = {'x_fused': torch.tensor(X_fused, dtype=torch.float32)}
                if X_ac is not None:
                    inp['x_ac'] = torch.tensor(X_ac, dtype=torch.float32)
                    inp['x_w2v'] = torch.tensor(X_w2v, dtype=torch.float32)
                    inp['x_bert'] = torch.tensor(X_bert, dtype=torch.float32)
            else:
                inp = torch.tensor(X_fused, dtype=torch.float32)
            out = model_obj(inp)
            probs = out['severity_probs'].numpy()
            preds = np.argmax(probs, axis=-1)
            phq_pred = out['phq8_score'].numpy()
        return compute_comprehensive_metrics(y_true, preds, probs, phq_true, phq_pred), preds, probs, phq_pred

    # -------------------------------------------------------------
    # Comprehensive Held-Out Test Evaluation (N=37)
    # -------------------------------------------------------------
    print("\n==========================================")
    print("  HELD-OUT CANONICAL TEST EVALUATION (N=37)")
    print("==========================================")

    te_raw = np.hstack([d_te['ac'], d_te['w2v'], d_te['bert']])
    te_c2_fused = np.hstack([
        cand2_scalers['scaler_ac'].transform(d_te['ac']),
        cand2_scalers['scaler_w2v'].transform(d_te['w2v']),
        cand2_scalers['scaler_bert'].transform(d_te['bert'])
    ])

    m_te_base, p_te_base, pr_te_base, phq_te_base = run_eval(cal_base, False, te_raw, None, None, None, d_te['y_class'], d_te['y_phq'])
    m_te_c2, p_te_c2, pr_te_c2, phq_te_c2 = run_eval(cal_c2, True, te_c2_fused, None, None, None, d_te['y_class'], d_te['y_phq'])
    m_te_c2a, p_te_c2a, pr_te_c2a, phq_te_c2a = run_eval(cal_c2a, True, te_fused_s, None, None, None, d_te['y_class'], d_te['y_phq'])
    m_te_c3a, p_te_c3a, pr_te_c3a, phq_te_c3a = run_eval(cal_3a, True, te_fused_s, None, None, None, d_te['y_class'], d_te['y_phq'])
    m_te_c3b, p_te_c3b, pr_te_c3b, phq_te_c3b = run_eval(cal_3b, True, te_fused_s, None, None, None, d_te['y_class'], d_te['y_phq'])
    m_te_c3c, p_te_c3c, pr_te_c3c, phq_te_c3c = run_eval(cal_3c, True, te_fused_s, te_ac_s, te_w2v_s, te_b_s, d_te['y_class'], d_te['y_phq'])

    all_test_metrics = {
        'Baseline': m_te_base,
        'Candidate 2': m_te_c2,
        'Candidate 2A': m_te_c2a,
        'Candidate 3A (alpha=0.15)': m_te_c3a,
        'Candidate 3B (alpha=0.18)': m_te_c3b,
        'Candidate 3C (ProjectedNet)': m_te_c3c
    }

    for name, m in all_test_metrics.items():
        print(f"  {name:28s} | Acc: {m['accuracy']:.2f}% | Macro-F1: {m['macro_f1']:.2f}% | Bal-Acc: {m['balanced_accuracy']:.2f}% | Bin-F1: {m['binary_f1']:.2f}% | MAE: {m['mae']:.2f}")

    # -------------------------------------------------------------
    # Comprehensive Unseen Cohort Evaluation (N=33 Repaired)
    # -------------------------------------------------------------
    print("\n==========================================")
    print("  REPAIRED UNSEEN COHORT EVALUATION (N=33)")
    print("==========================================")

    u_c2_fused = np.hstack([
        cand2_scalers['scaler_ac'].transform(u_feats[:, :154]),
        cand2_scalers['scaler_w2v'].transform(u_feats[:, 154:922]),
        cand2_scalers['scaler_bert'].transform(u_feats[:, 922:1690])
    ])

    m_u_base, p_u_base, pr_u_base, phq_u_base = run_eval(cal_base, False, u_feats, None, None, None, u_class, u_phq)
    m_u_c2, p_u_c2, pr_u_c2, phq_u_c2 = run_eval(cal_c2, True, u_c2_fused, None, None, None, u_class, u_phq)
    m_u_c2a, p_u_c2a, pr_u_c2a, phq_u_c2a = run_eval(cal_c2a, True, u_fused_s, None, None, None, u_class, u_phq)
    m_u_c3a, p_u_c3a, pr_u_c3a, phq_u_c3a = run_eval(cal_3a, True, u_fused_s, None, None, None, u_class, u_phq)
    m_u_c3b, p_u_c3b, pr_u_c3b, phq_u_c3b = run_eval(cal_3b, True, u_fused_s, None, None, None, u_class, u_phq)
    m_u_c3c, p_u_c3c, pr_u_c3c, phq_u_c3c = run_eval(cal_3c, True, u_fused_s, u_ac_s, u_w2v_s, u_b_s, u_class, u_phq)

    all_unseen_metrics = {
        'Baseline': m_u_base,
        'Candidate 2': m_u_c2,
        'Candidate 2A': m_u_c2a,
        'Candidate 3A (alpha=0.15)': m_u_c3a,
        'Candidate 3B (alpha=0.18)': m_u_c3b,
        'Candidate 3C (ProjectedNet)': m_u_c3c
    }

    for name, m in all_unseen_metrics.items():
        print(f"  {name:28s} | Acc: {m['accuracy']:.2f}% | Macro-F1: {m['macro_f1']:.2f}% | Bal-Acc: {m['balanced_accuracy']:.2f}% | Bin-F1: {m['binary_f1']:.2f}% | MAE: {m['mae']:.2f}")

    # -------------------------------------------------------------
    # Sub-Cohort Breakdown: Autonomous AI (N=24)
    # -------------------------------------------------------------
    print("\n--- Autonomous AI Sub-Cohort (N=24) ---")
    mask_ai = (u_cohort == 'AI_Autonomous')
    ai_metrics = {}
    for name, (preds, probs, phqs) in [
        ('Baseline', (p_u_base, pr_u_base, phq_u_base)),
        ('Candidate 2', (p_u_c2, pr_u_c2, phq_u_c2)),
        ('Candidate 2A', (p_u_c2a, pr_u_c2a, phq_u_c2a)),
        ('Candidate 3A (alpha=0.15)', (p_u_c3a, pr_u_c3a, phq_u_c3a)),
        ('Candidate 3B (alpha=0.18)', (p_u_c3b, pr_u_c3b, phq_u_c3b)),
        ('Candidate 3C (ProjectedNet)', (p_u_c3c, pr_u_c3c, phq_u_c3c))
    ]:
        m = compute_comprehensive_metrics(u_class[mask_ai], preds[mask_ai], probs[mask_ai], u_phq[mask_ai], phqs[mask_ai])
        ai_metrics[name] = m
        class_dist = np.bincount(preds[mask_ai], minlength=3).tolist()
        print(f"  {name:28s} | Acc: {m['accuracy']:.2f}% | Macro-F1: {m['macro_f1']:.2f}% | Bal-Acc: {m['balanced_accuracy']:.2f}% | Bin-F1: {m['binary_f1']:.2f}% | MAE: {m['mae']:.2f} | Preds: {class_dist}")

    # -------------------------------------------------------------
    # Sub-Cohort Breakdown: WoZ Human (N=9)
    # -------------------------------------------------------------
    print("\n--- Wizard-of-Oz Human Sub-Cohort (N=9) ---")
    mask_woz = (u_cohort == 'WoZ_Human')
    woz_metrics = {}
    for name, (preds, probs, phqs) in [
        ('Baseline', (p_u_base, pr_u_base, phq_u_base)),
        ('Candidate 2', (p_u_c2, pr_u_c2, phq_u_c2)),
        ('Candidate 2A', (p_u_c2a, pr_u_c2a, phq_u_c2a)),
        ('Candidate 3A (alpha=0.15)', (p_u_c3a, pr_u_c3a, phq_u_c3a)),
        ('Candidate 3B (alpha=0.18)', (p_u_c3b, pr_u_c3b, phq_u_c3b)),
        ('Candidate 3C (ProjectedNet)', (p_u_c3c, pr_u_c3c, phq_u_c3c))
    ]:
        m = compute_comprehensive_metrics(u_class[mask_woz], preds[mask_woz], probs[mask_woz], u_phq[mask_woz], phqs[mask_woz])
        woz_metrics[name] = m
        class_dist = np.bincount(preds[mask_woz], minlength=3).tolist()
        print(f"  {name:28s} | Acc: {m['accuracy']:.2f}% | Macro-F1: {m['macro_f1']:.2f}% | Bal-Acc: {m['balanced_accuracy']:.2f}% | Bin-F1: {m['binary_f1']:.2f}% | MAE: {m['mae']:.2f} | Preds: {class_dist}")

    # -------------------------------------------------------------
    # Save Prediction CSVs for Candidate 3A & 3B
    # -------------------------------------------------------------
    pd.DataFrame({
        'session_id': d_te['sids'],
        'participant_id': d_te['pids'],
        'true_class': d_te['y_class'],
        'pred_class': p_te_c3a,
        'prob_0': pr_te_c3a[:, 0],
        'prob_1': pr_te_c3a[:, 1],
        'prob_2': pr_te_c3a[:, 2],
        'true_phq': d_te['y_phq'],
        'pred_phq': phq_te_c3a
    }).to_csv(os.path.join(EXP_V2_DIR, "predictions", "test_candidate3a_predictions.csv"), index=False)

    pd.DataFrame({
        'participant_id': u_pids,
        'cohort': u_cohort,
        'true_class': u_class,
        'pred_class': p_u_c3a,
        'prob_0': pr_u_c3a[:, 0],
        'prob_1': pr_u_c3a[:, 1],
        'prob_2': pr_u_c3a[:, 2],
        'true_phq': u_phq,
        'pred_phq': phq_u_c3a
    }).to_csv(os.path.join(EXP_V2_DIR, "predictions", "unseen_candidate3a_predictions.csv"), index=False)

    pd.DataFrame({
        'session_id': d_te['sids'],
        'participant_id': d_te['pids'],
        'true_class': d_te['y_class'],
        'pred_class': p_te_c3b,
        'prob_0': pr_te_c3b[:, 0],
        'prob_1': pr_te_c3b[:, 1],
        'prob_2': pr_te_c3b[:, 2],
        'true_phq': d_te['y_phq'],
        'pred_phq': phq_te_c3b
    }).to_csv(os.path.join(EXP_V2_DIR, "predictions", "test_candidate3b_predictions.csv"), index=False)

    pd.DataFrame({
        'participant_id': u_pids,
        'cohort': u_cohort,
        'true_class': u_class,
        'pred_class': p_u_c3b,
        'prob_0': pr_u_c3b[:, 0],
        'prob_1': pr_u_c3b[:, 1],
        'prob_2': pr_u_c3b[:, 2],
        'true_phq': u_phq,
        'pred_phq': phq_u_c3b
    }).to_csv(os.path.join(EXP_V2_DIR, "predictions", "unseen_candidate3b_predictions.csv"), index=False)

    # Save complete evaluation JSON
    eval_summary = {
        'test_metrics': all_test_metrics,
        'unseen_metrics': all_unseen_metrics,
        'ai_subcohort_metrics': ai_metrics,
        'woz_subcohort_metrics': woz_metrics
    }
    with open(os.path.join(EXP_V2_DIR, "final_comparison_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\nSaved all predictions, checkpoints, and summary JSON to experiments/capstone_accuracy_v2/ successfully!")


if __name__ == "__main__":
    main()
