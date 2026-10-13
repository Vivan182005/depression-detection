"""
Final Training & Zero-Leakage Evaluation Script for Depression-Detection Capstone.

Trains Candidate 2A (Mild Balanced Class Weights, alpha=0.25) strictly on development train data (N=116),
validates and early-stops on development validation data (N=30),
and calibrates temperature scaling strictly on development validation data.

Then performs a one-shot, frozen, apples-to-apples evaluation across:
1. Baseline Model (app/feature_store/best_depression_model.pth)
2. Candidate 2 Model (cand2_standard_mlp_balanced.pth)
3. Candidate 2A Model (cand2a_standard_mlp_mild_balanced.pth)

Evaluated on:
- Canonical Test Set (N=37)
- Unseen Cohort (N=33) with old (zero-BERT) features
- Unseen Cohort (N=33) with repaired (real-BERT) features
- Breakdown by WoZ Human (N=9) and Autonomous AI (N=24)
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
sys.path.insert(0, EXP_DIR)

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from src.models.depression_model import CompactTriModalMLP
from src.models.calibrated_model import TemperatureScaler
from train_improvements import StandardTriModalMLP, MultimodalDataset, load_dataset_features


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_comprehensive_metrics(y_true, y_pred, y_prob, phq_true, phq_pred, y_bin_true=None, y_bin_pred=None, y_bin_prob=None):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    if y_bin_true is None:
        y_bin_true = (y_true >= 1).astype(int)
    if y_bin_prob is None:
        y_bin_prob = 1.0 - y_prob[:, 0]
    if y_bin_pred is None:
        y_bin_pred = (y_bin_prob >= 0.5).astype(int)

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


def main():
    set_seed(42)
    os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "configs"), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "predictions"), exist_ok=True)

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    # Load and clean development and test features
    d_tr = load_dataset_features(train_df, fs, clean_labels=True)
    d_va = load_dataset_features(val_df, fs, clean_labels=True)
    d_te = load_dataset_features(test_df, fs, clean_labels=True)

    # Fit scalers strictly on train
    sc_ac = StandardScaler().fit(d_tr['ac'])
    sc_w2v = StandardScaler().fit(d_tr['w2v'])
    sc_bert = StandardScaler().fit(d_tr['bert'])

    joblib.dump(
        {'scaler_ac': sc_ac, 'scaler_w2v': sc_w2v, 'scaler_bert': sc_bert},
        os.path.join(EXP_DIR, "checkpoints", "cand2a_feature_scalers.joblib")
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

    # Class weighting for Candidate 2A: mild balanced alpha=0.25
    counts = np.bincount(d_tr['y_class'], minlength=3)
    raw_w = (len(d_tr['y_class']) / (3.0 * (counts + 1e-5))) ** 0.25
    weights_2a = raw_w / np.mean(raw_w)
    w_tensor = torch.tensor(weights_2a, dtype=torch.float32)
    print(f"Train counts: {counts}, Candidate 2A weights (alpha=0.25): {weights_2a.round(4)}")

    # Train Candidate 2A
    print("\n--- Training Candidate 2A (Mild Balanced alpha=0.25) ---")
    model2a = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    optimizer = torch.optim.AdamW(model2a.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion_ce = nn.CrossEntropyLoss(weight=w_tensor)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, 41):
        model2a.train()
        for batch in tr_loader:
            optimizer.zero_grad()
            out = model2a(batch)
            loss_c = criterion_ce(out['severity_logits'], batch['class_id'])
            loss_p = criterion_mse(out['phq8_score'], batch['phq8_score'])
            loss_b = criterion_bce(out['binary_logits'], batch['binary_label'])
            loss = loss_c + 0.1 * loss_p + 0.2 * loss_b
            loss.backward()
            nn.utils.clip_grad_norm_(model2a.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validation early stopping check
        model2a.eval()
        with torch.no_grad():
            for batch in va_loader:
                out = model2a(batch)
                probs = out['severity_probs'].numpy()
                preds = np.argmax(probs, axis=-1)
                targets = batch['class_id'].numpy()
        v_f1 = f1_score(targets, preds, average='macro', zero_division=0) * 100.0
        v_bal = balanced_accuracy_score(targets, preds) * 100.0
        if (v_f1 > best_val_macro_f1) or (v_f1 == best_val_macro_f1 and v_bal > best_val_bal_acc):
            best_val_macro_f1 = v_f1
            best_val_bal_acc = v_bal
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model2a.state_dict().items()}

    print(f"Candidate 2A Best Epoch: {best_epoch}, Val Macro-F1: {best_val_macro_f1:.2f}%, Val Bal-Acc: {best_val_bal_acc:.2f}%")
    model2a.load_state_dict(best_state)
    model2a.eval()

    # Calibrate Temperature strictly on Validation split
    cal_2a = TemperatureScaler(model2a)
    opt_temp_2a = cal_2a.calibrate(va_loader)
    print(f"Optimal Temperature for Candidate 2A: {opt_temp_2a:.4f}")

    # Save Candidate 2A checkpoint and config
    cand2a_ckpt_path = os.path.join(EXP_DIR, "checkpoints", "cand2a_standard_mlp_mild_balanced.pth")
    torch.save({
        'model_state_dict': best_state,
        'optimal_temperature': opt_temp_2a,
        'best_epoch': best_epoch,
        'class_weights': weights_2a.tolist(),
        'val_macro_f1': best_val_macro_f1,
        'val_bal_acc': best_val_bal_acc
    }, cand2a_ckpt_path)

    config_2a = {
        'candidate_name': 'Candidate 2A: Standard MLP (Mild Balanced Weights alpha=0.25)',
        'architecture': 'StandardTriModalMLP',
        'in_dim': 1690,
        'hidden_dim': 128,
        'dropout': 0.3,
        'alpha_weighting': 0.25,
        'class_weights': [float(w) for w in weights_2a],
        'lambda_phq': 0.1,
        'lambda_bin': 0.2,
        'optimizer': 'AdamW',
        'learning_rate': 0.001,
        'weight_decay': 0.001,
        'scheduler': 'CosineAnnealingLR',
        'epochs': 40,
        'best_epoch': best_epoch,
        'optimal_temperature': float(opt_temp_2a)
    }
    with open(os.path.join(EXP_DIR, "configs", "candidate2a_config.json"), "w") as f:
        json.dump(config_2a, f, indent=2)

    # -------------------------------------------------------------
    # 2. Frozen Evaluation: Load Baseline and Candidate 2
    # -------------------------------------------------------------
    base_ckpt = torch.load(os.path.join(APP_DIR, "feature_store", "best_depression_model.pth"), map_location='cpu', weights_only=False)
    model_base = CompactTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    model_base.load_state_dict(base_ckpt['model_state_dict'])
    model_base.eval()
    cal_base = TemperatureScaler(model_base)
    cal_base.temperature.data.fill_(base_ckpt.get('optimal_temperature', 1.0))
    cal_base.eval()

    cand2_scalers = joblib.load(os.path.join(EXP_DIR, "checkpoints", "feature_scalers.joblib"))
    cand2_ckpt = torch.load(os.path.join(EXP_DIR, "checkpoints", "cand2_standard_mlp_balanced.pth"), map_location='cpu', weights_only=False)
    model2 = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3)
    model2.load_state_dict(cand2_ckpt['model_state_dict'])
    model2.eval()
    cal_2 = TemperatureScaler(model2)
    cal_2.temperature.data.fill_(cand2_ckpt['optimal_temperature'])
    cal_2.eval()

    # -------------------------------------------------------------
    # 3. Canonical Test Set Evaluation (N=37)
    # -------------------------------------------------------------
    print("\n==========================================")
    print("Evaluating Models on Canonical Held-Out Test Set (N=37)")
    print("==========================================")

    # Baseline on Test (Raw features)
    te_raw = np.hstack([d_te['ac'], d_te['w2v'], d_te['bert']])
    with torch.no_grad():
        out_base = cal_base(torch.tensor(te_raw, dtype=torch.float32))
        probs_base = out_base['severity_probs'].numpy()
        preds_base = np.argmax(probs_base, axis=-1)
        phq_base = out_base['phq8_score'].numpy()
        bin_prob_base = out_base['binary_probs'].numpy()
        bin_pred_base = (bin_prob_base >= 0.5).astype(int)

    m_test_base = compute_comprehensive_metrics(
        d_te['y_class'], preds_base, probs_base, d_te['y_phq'], phq_base,
        (d_te['y_class'] >= 1).astype(int), bin_pred_base, bin_prob_base
    )

    pd.DataFrame({
        'session_id': d_te['sids'],
        'participant_id': d_te['pids'],
        'true_class': d_te['y_class'],
        'pred_class': preds_base,
        'prob_0': probs_base[:, 0],
        'prob_1': probs_base[:, 1],
        'prob_2': probs_base[:, 2],
        'true_phq': d_te['y_phq'],
        'pred_phq': phq_base,
        'true_bin': (d_te['y_class'] >= 1).astype(int),
        'pred_bin': bin_pred_base
    }).to_csv(os.path.join(EXP_DIR, "predictions", "test_baseline_predictions.csv"), index=False)

    # Candidate 2 on Test
    te_cand2_fused = np.hstack([
        cand2_scalers['scaler_ac'].transform(d_te['ac']),
        cand2_scalers['scaler_w2v'].transform(d_te['w2v']),
        cand2_scalers['scaler_bert'].transform(d_te['bert'])
    ])
    with torch.no_grad():
        out_c2 = cal_2({'x_fused': torch.tensor(te_cand2_fused, dtype=torch.float32)})
        probs_c2 = out_c2['severity_probs'].numpy()
        preds_c2 = np.argmax(probs_c2, axis=-1)
        phq_c2 = out_c2['phq8_score'].numpy()
        bin_prob_c2 = out_c2['binary_probs'].numpy()
        bin_pred_c2 = (bin_prob_c2 >= 0.5).astype(int)

    m_test_c2 = compute_comprehensive_metrics(
        d_te['y_class'], preds_c2, probs_c2, d_te['y_phq'], phq_c2,
        (d_te['y_class'] >= 1).astype(int), bin_pred_c2, bin_prob_c2
    )

    pd.DataFrame({
        'session_id': d_te['sids'],
        'participant_id': d_te['pids'],
        'true_class': d_te['y_class'],
        'pred_class': preds_c2,
        'prob_0': probs_c2[:, 0],
        'prob_1': probs_c2[:, 1],
        'prob_2': probs_c2[:, 2],
        'true_phq': d_te['y_phq'],
        'pred_phq': phq_c2,
        'true_bin': (d_te['y_class'] >= 1).astype(int),
        'pred_bin': bin_pred_c2
    }).to_csv(os.path.join(EXP_DIR, "predictions", "test_candidate2_predictions.csv"), index=False)

    # Candidate 2A on Test
    te_cand2a_fused = np.hstack([te_ac_s, te_w2v_s, te_b_s])
    with torch.no_grad():
        out_c2a = cal_2a({'x_fused': torch.tensor(te_cand2a_fused, dtype=torch.float32)})
        probs_c2a = out_c2a['severity_probs'].numpy()
        preds_c2a = np.argmax(probs_c2a, axis=-1)
        phq_c2a = out_c2a['phq8_score'].numpy()
        bin_prob_c2a = out_c2a['binary_probs'].numpy()
        bin_pred_c2a = (bin_prob_c2a >= 0.5).astype(int)

    m_test_c2a = compute_comprehensive_metrics(
        d_te['y_class'], preds_c2a, probs_c2a, d_te['y_phq'], phq_c2a,
        (d_te['y_class'] >= 1).astype(int), bin_pred_c2a, bin_prob_c2a
    )

    pd.DataFrame({
        'session_id': d_te['sids'],
        'participant_id': d_te['pids'],
        'true_class': d_te['y_class'],
        'pred_class': preds_c2a,
        'prob_0': probs_c2a[:, 0],
        'prob_1': probs_c2a[:, 1],
        'prob_2': probs_c2a[:, 2],
        'true_phq': d_te['y_phq'],
        'pred_phq': phq_c2a,
        'true_bin': (d_te['y_class'] >= 1).astype(int),
        'pred_bin': bin_pred_c2a
    }).to_csv(os.path.join(EXP_DIR, "predictions", "test_candidate2a_predictions.csv"), index=False)

    print(f"Test Set (N=37) Results:")
    print(f"  Baseline:     Acc={m_test_base['accuracy']:.2f}%, Macro-F1={m_test_base['macro_f1']:.2f}%, Bal-Acc={m_test_base['balanced_accuracy']:.2f}%, Bin-F1={m_test_base['binary_f1']:.2f}%, PHQ-MAE={m_test_base['mae']:.2f}")
    print(f"  Candidate 2:  Acc={m_test_c2['accuracy']:.2f}%, Macro-F1={m_test_c2['macro_f1']:.2f}%, Bal-Acc={m_test_c2['balanced_accuracy']:.2f}%, Bin-F1={m_test_c2['binary_f1']:.2f}%, PHQ-MAE={m_test_c2['mae']:.2f}")
    print(f"  Candidate 2A: Acc={m_test_c2a['accuracy']:.2f}%, Macro-F1={m_test_c2a['macro_f1']:.2f}%, Bal-Acc={m_test_c2a['balanced_accuracy']:.2f}%, Bin-F1={m_test_c2a['binary_f1']:.2f}%, PHQ-MAE={m_test_c2a['mae']:.2f}")

    # -------------------------------------------------------------
    # 4. Unseen Cohort Evaluation (N=33) - Repaired Features
    # -------------------------------------------------------------
    print("\n==========================================")
    print("Evaluating Models on REPAIRED Unseen Cohort (N=33)")
    print("==========================================")

    unseen_rep = np.load(os.path.join(EXP_DIR, "unseen_feature_cache_repaired.npz"), allow_pickle=True)
    u_pids = unseen_rep['pids']
    u_feats = unseen_rep['features']
    u_class = unseen_rep['true_class']
    u_phq = unseen_rep['true_phq']
    u_cohort = unseen_rep['cohort']

    # Baseline on Unseen Repaired (raw features)
    with torch.no_grad():
        out_u_base = cal_base(torch.tensor(u_feats, dtype=torch.float32))
        probs_u_base = out_u_base['severity_probs'].numpy()
        preds_u_base = np.argmax(probs_u_base, axis=-1)
        phq_u_base = out_u_base['phq8_score'].numpy()
        bin_prob_u_base = out_u_base['binary_probs'].numpy()
        bin_pred_u_base = (bin_prob_u_base >= 0.5).astype(int)

    m_unseen_base = compute_comprehensive_metrics(
        u_class, preds_u_base, probs_u_base, u_phq, phq_u_base,
        (u_class >= 1).astype(int), bin_pred_u_base, bin_prob_u_base
    )

    pd.DataFrame({
        'participant_id': u_pids,
        'cohort': u_cohort,
        'true_class': u_class,
        'pred_class': preds_u_base,
        'prob_0': probs_u_base[:, 0],
        'prob_1': probs_u_base[:, 1],
        'prob_2': probs_u_base[:, 2],
        'true_phq': u_phq,
        'pred_phq': phq_u_base,
        'true_bin': (u_class >= 1).astype(int),
        'pred_bin': bin_pred_u_base
    }).to_csv(os.path.join(EXP_DIR, "predictions", "unseen_baseline_predictions_repaired.csv"), index=False)

    # Candidate 2 on Unseen Repaired
    u_cand2_fused = np.hstack([
        cand2_scalers['scaler_ac'].transform(u_feats[:, :154]),
        cand2_scalers['scaler_w2v'].transform(u_feats[:, 154:922]),
        cand2_scalers['scaler_bert'].transform(u_feats[:, 922:1690])
    ])
    with torch.no_grad():
        out_u_c2 = cal_2({'x_fused': torch.tensor(u_cand2_fused, dtype=torch.float32)})
        probs_u_c2 = out_u_c2['severity_probs'].numpy()
        preds_u_c2 = np.argmax(probs_u_c2, axis=-1)
        phq_u_c2 = out_u_c2['phq8_score'].numpy()
        bin_prob_u_c2 = out_u_c2['binary_probs'].numpy()
        bin_pred_u_c2 = (bin_prob_u_c2 >= 0.5).astype(int)

    m_unseen_c2 = compute_comprehensive_metrics(
        u_class, preds_u_c2, probs_u_c2, u_phq, phq_u_c2,
        (u_class >= 1).astype(int), bin_pred_u_c2, bin_prob_u_c2
    )

    pd.DataFrame({
        'participant_id': u_pids,
        'cohort': u_cohort,
        'true_class': u_class,
        'pred_class': preds_u_c2,
        'prob_0': probs_u_c2[:, 0],
        'prob_1': probs_u_c2[:, 1],
        'prob_2': probs_u_c2[:, 2],
        'true_phq': u_phq,
        'pred_phq': phq_u_c2,
        'true_bin': (u_class >= 1).astype(int),
        'pred_bin': bin_pred_u_c2
    }).to_csv(os.path.join(EXP_DIR, "predictions", "unseen_candidate2_predictions_repaired.csv"), index=False)

    # Candidate 2A on Unseen Repaired
    u_cand2a_fused = np.hstack([
        sc_ac.transform(u_feats[:, :154]),
        sc_w2v.transform(u_feats[:, 154:922]),
        sc_bert.transform(u_feats[:, 922:1690])
    ])
    with torch.no_grad():
        out_u_c2a = cal_2a({'x_fused': torch.tensor(u_cand2a_fused, dtype=torch.float32)})
        probs_u_c2a = out_u_c2a['severity_probs'].numpy()
        preds_u_c2a = np.argmax(probs_u_c2a, axis=-1)
        phq_u_c2a = out_u_c2a['phq8_score'].numpy()
        bin_prob_u_c2a = out_u_c2a['binary_probs'].numpy()
        bin_pred_u_c2a = (bin_prob_u_c2a >= 0.5).astype(int)

    m_unseen_c2a = compute_comprehensive_metrics(
        u_class, preds_u_c2a, probs_u_c2a, u_phq, phq_u_c2a,
        (u_class >= 1).astype(int), bin_pred_u_c2a, bin_prob_u_c2a
    )

    pd.DataFrame({
        'participant_id': u_pids,
        'cohort': u_cohort,
        'true_class': u_class,
        'pred_class': preds_u_c2a,
        'prob_0': probs_u_c2a[:, 0],
        'prob_1': probs_u_c2a[:, 1],
        'prob_2': probs_u_c2a[:, 2],
        'true_phq': u_phq,
        'pred_phq': phq_u_c2a,
        'true_bin': (u_class >= 1).astype(int),
        'pred_bin': bin_pred_u_c2a
    }).to_csv(os.path.join(EXP_DIR, "predictions", "unseen_candidate2a_predictions_repaired.csv"), index=False)

    print(f"Unseen Cohort (N=33) Results:")
    print(f"  Baseline:     Acc={m_unseen_base['accuracy']:.2f}%, Macro-F1={m_unseen_base['macro_f1']:.2f}%, Bal-Acc={m_unseen_base['balanced_accuracy']:.2f}%, Bin-F1={m_unseen_base['binary_f1']:.2f}%, PHQ-MAE={m_unseen_base['mae']:.2f}")
    print(f"  Candidate 2:  Acc={m_unseen_c2['accuracy']:.2f}%, Macro-F1={m_unseen_c2['macro_f1']:.2f}%, Bal-Acc={m_unseen_c2['balanced_accuracy']:.2f}%, Bin-F1={m_unseen_c2['binary_f1']:.2f}%, PHQ-MAE={m_unseen_c2['mae']:.2f}")
    print(f"  Candidate 2A: Acc={m_unseen_c2a['accuracy']:.2f}%, Macro-F1={m_unseen_c2a['macro_f1']:.2f}%, Bal-Acc={m_unseen_c2a['balanced_accuracy']:.2f}%, Bin-F1={m_unseen_c2a['binary_f1']:.2f}%, PHQ-MAE={m_unseen_c2a['mae']:.2f}")

    # -------------------------------------------------------------
    # 5. Cohort Breakdown: WoZ Human (N=9) vs Autonomous AI (N=24)
    # -------------------------------------------------------------
    mask_woz = (u_cohort == 'WoZ_Human')
    mask_ai = (u_cohort == 'AI_Autonomous')

    # WoZ Sub-Cohort
    woz_m_base = compute_comprehensive_metrics(u_class[mask_woz], preds_u_base[mask_woz], probs_u_base[mask_woz], u_phq[mask_woz], phq_u_base[mask_woz])
    woz_m_c2 = compute_comprehensive_metrics(u_class[mask_woz], preds_u_c2[mask_woz], probs_u_c2[mask_woz], u_phq[mask_woz], phq_u_c2[mask_woz])
    woz_m_c2a = compute_comprehensive_metrics(u_class[mask_woz], preds_u_c2a[mask_woz], probs_u_c2a[mask_woz], u_phq[mask_woz], phq_u_c2a[mask_woz])

    # AI Sub-Cohort
    ai_m_base = compute_comprehensive_metrics(u_class[mask_ai], preds_u_base[mask_ai], probs_u_base[mask_ai], u_phq[mask_ai], phq_u_base[mask_ai])
    ai_m_c2 = compute_comprehensive_metrics(u_class[mask_ai], preds_u_c2[mask_ai], probs_u_c2[mask_ai], u_phq[mask_ai], phq_u_c2[mask_ai])
    ai_m_c2a = compute_comprehensive_metrics(u_class[mask_ai], preds_u_c2a[mask_ai], probs_u_c2a[mask_ai], u_phq[mask_ai], phq_u_c2a[mask_ai])

    print("\n--- Autonomous AI Sub-Cohort (N=24) ---")
    print(f"  Baseline:     Acc={ai_m_base['accuracy']:.2f}%, Macro-F1={ai_m_base['macro_f1']:.2f}%, Bal-Acc={ai_m_base['balanced_accuracy']:.2f}%, Pred Class Counts: {np.bincount(preds_u_base[mask_ai], minlength=3).tolist()}")
    print(f"  Candidate 2:  Acc={ai_m_c2['accuracy']:.2f}%, Macro-F1={ai_m_c2['macro_f1']:.2f}%, Bal-Acc={ai_m_c2['balanced_accuracy']:.2f}%, Pred Class Counts: {np.bincount(preds_u_c2[mask_ai], minlength=3).tolist()}")
    print(f"  Candidate 2A: Acc={ai_m_c2a['accuracy']:.2f}%, Macro-F1={ai_m_c2a['macro_f1']:.2f}%, Bal-Acc={ai_m_c2a['balanced_accuracy']:.2f}%, Pred Class Counts: {np.bincount(preds_u_c2a[mask_ai], minlength=3).tolist()}")

    # -------------------------------------------------------------
    # 6. Save Complete Final JSON Summary
    # -------------------------------------------------------------
    final_summary = {
        'candidate_name': 'Candidate 2A: Standard MLP (Mild Balanced Weights alpha=0.25)',
        'test_evaluation': {
            'baseline': m_test_base,
            'candidate_2': m_test_c2,
            'candidate_2a': m_test_c2a
        },
        'unseen_repaired_evaluation': {
            'overall_unseen': {
                'baseline': m_unseen_base,
                'candidate_2': m_unseen_c2,
                'candidate_2a': m_unseen_c2a
            },
            'woz_human_subcohort': {
                'baseline': woz_m_base,
                'candidate_2': woz_m_c2,
                'candidate_2a': woz_m_c2a
            },
            'ai_autonomous_subcohort': {
                'baseline': ai_m_base,
                'candidate_2': ai_m_c2,
                'candidate_2a': ai_m_c2a
            }
        }
    }

    with open(os.path.join(EXP_DIR, "final_evaluation_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\nSaved experiments/capstone_improvement_v1/final_evaluation_summary.json successfully!")


if __name__ == "__main__":
    main()
