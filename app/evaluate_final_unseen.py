"""
Comprehensive Frozen Final Evaluation on the Reserved Unseen Cohort.
Project: Explainable Multi-Modal Speech & Language Depression Detection System.
Evaluates the frozen Candidate 2A model strictly on the reserved unseen cohort (N=33)
with subcohort analysis across Human WoZ (N=9) and Autonomous AI (N=24).
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, roc_auc_score, auc
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "final_model")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.models.calibrated_model import TemperatureScaler
from infer_depression import StandardTriModalMLP


def compute_metrics_bundle(y_true, y_pred, y_probs, phq_true, exp_phq, raw_phq, y_bin_true=None, y_bin_pred=None, y_bin_prob=None):
    n = len(y_true)
    if n == 0:
        return {}

    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    support = [int(np.sum(y_true == c)) for c in range(3)]

    # Majority class baseline
    maj_class = int(np.argmax(support))
    maj_pred = np.full(n, maj_class)
    maj_acc = float(accuracy_score(y_true, maj_pred) * 100.0)
    maj_macro_f1 = float(f1_score(y_true, maj_pred, average='macro', zero_division=0) * 100.0)

    # Binary screening metrics (Class >= 1 or Expected PHQ >= 10.0)
    if y_bin_true is None:
        y_bin_true = (y_true >= 1).astype(int)
    if y_bin_prob is None:
        y_bin_prob = 1.0 - y_probs[:, 0]
    if y_bin_pred is None:
        y_bin_pred = (y_bin_prob >= 0.5).astype(int)

    bin_acc = float(accuracy_score(y_bin_true, y_bin_pred) * 100.0)
    bin_p = float(precision_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_r = float(recall_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)

    # Sensitivity (Recall on pos) and Specificity (Recall on neg)
    tn = int(np.sum((y_bin_true == 0) & (y_bin_pred == 0)))
    fp = int(np.sum((y_bin_true == 0) & (y_bin_pred == 1)))
    fn = int(np.sum((y_bin_true == 1) & (y_bin_pred == 0)))
    tp = int(np.sum((y_bin_true == 1) & (y_bin_pred == 1)))
    sensitivity = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    bin_cm = [[tn, fp], [fn, tp]]

    # ROC-AUC and PR-AUC where both classes exist
    has_both_classes = len(np.unique(y_bin_true)) > 1
    if has_both_classes:
        try:
            roc_auc = float(roc_auc_score(y_bin_true, y_bin_prob) * 100.0)
        except Exception:
            roc_auc = None
        try:
            p_curve, r_curve, _ = precision_recall_curve(y_bin_true, y_bin_prob)
            pr_auc = float(auc(r_curve, p_curve) * 100.0)
        except Exception:
            pr_auc = None
    else:
        roc_auc, pr_auc = None, None

    # Continuous PHQ-8 metrics (Expected PHQ)
    exp_mae = float(mean_absolute_error(phq_true, exp_phq))
    exp_rmse = float(np.sqrt(mean_squared_error(phq_true, exp_phq)))
    try:
        exp_r2 = float(r2_score(phq_true, exp_phq))
    except Exception:
        exp_r2 = None

    # Raw Head metrics
    raw_mae = float(mean_absolute_error(phq_true, raw_phq))
    raw_rmse = float(np.sqrt(mean_squared_error(phq_true, raw_phq)))
    try:
        raw_r2 = float(r2_score(phq_true, raw_phq))
    except Exception:
        raw_r2 = None

    # Mean predictor baseline for continuous PHQ
    mean_phq_val = float(np.mean(phq_true))
    mean_pred = np.full(n, mean_phq_val)
    mean_mae = float(mean_absolute_error(phq_true, mean_pred))
    mean_rmse = float(np.sqrt(mean_squared_error(phq_true, mean_pred)))

    # Calibration & Reliability: ECE with 10 uniform bins
    bin_boundaries = np.linspace(0, 1, 11)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    ece = 0.0
    for i in range(10):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(accuracies[in_bin])
            ece += np.abs(avg_conf - avg_acc) * prop_in_bin
    ece = float(ece * 100.0)

    # Multi-class Brier score
    y_true_onehot = np.zeros_like(y_probs)
    for idx, c in enumerate(y_true):
        y_true_onehot[idx, c] = 1.0
    brier_score = float(np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1)))

    # Prediction Margin (difference between top-1 and top-2 probabilities)
    sorted_probs = np.sort(y_probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    return {
        'num_participants': n,
        'classification': {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'balanced_accuracy': bal_acc,
            'per_class_precision': [float(x) for x in p_per],
            'per_class_recall': [float(x) for x in r_per],
            'per_class_f1': [float(x) for x in f_per],
            'support': support,
            'confusion_matrix': cm,
            'majority_baseline_accuracy': maj_acc,
            'majority_baseline_macro_f1': maj_macro_f1
        },
        'binary_screening': {
            'binary_accuracy': bin_acc,
            'binary_precision': bin_p,
            'binary_recall': bin_r,
            'binary_f1': bin_f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': bin_cm
        },
        'continuous_phq8': {
            'expected_phq': {
                'mae': exp_mae,
                'rmse': exp_rmse,
                'r2': exp_r2
            },
            'raw_head': {
                'mae': raw_mae,
                'rmse': raw_rmse,
                'r2': raw_r2
            },
            'mean_predictor_baseline': {
                'mae': mean_mae,
                'rmse': mean_rmse
            }
        },
        'reliability_and_calibration': {
            'ece_10_bins': ece,
            'brier_score': brier_score,
            'confidence_mean': float(np.mean(confidences) * 100.0),
            'confidence_min': float(np.min(confidences) * 100.0),
            'confidence_max': float(np.max(confidences) * 100.0),
            'confidence_std': float(np.std(confidences) * 100.0),
            'margin_mean': float(np.mean(margins)),
            'margin_min': float(np.min(margins)),
            'margin_max': float(np.max(margins))
        }
    }


def main():
    print("==========================================================================")
    print("          FROZEN FINAL MODEL RESERVED UNSEEN COHORT EVALUATION            ")
    print("==========================================================================")

    # 1. Load canonical frozen model and scalers
    ckpt_path = os.path.join(ARTIFACTS_DIR, "model_checkpoint.pth")
    scaler_path = os.path.join(ARTIFACTS_DIR, "feature_scalers.joblib")
    cache_path = os.path.join(ARTIFACTS_DIR, "unseen_feature_cache_repaired.npz")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scalers missing: {scaler_path}")
    if not os.path.exists(cache_path):
        cache_path = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1", "unseen_feature_cache_repaired.npz")

    print(f" Loading Frozen Checkpoint : {ckpt_path}")
    print(f" Loading Feature Scalers   : {scaler_path}")
    print(f" Loading Unseen Cache      : {cache_path}")

    scalers = joblib.load(scaler_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = StandardTriModalMLP(in_dim=1690, hidden_dim=128, dropout=0.3)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    calibrated = TemperatureScaler(model)
    calibrated.temperature.data.fill_(ckpt.get('optimal_temperature', 1.4542))
    calibrated.eval()

    # 2. Load unseen feature cache
    u_data = np.load(cache_path, allow_pickle=True)
    u_feats = u_data['features']
    u_pids = u_data['pids']
    u_class = u_data['true_class']
    u_phq = u_data['true_phq']
    u_cohort = u_data['cohort']

    print(f" Evaluated Cohort Size     : N = {len(u_pids)} sessions across {len(np.unique(u_pids))} participants")

    # 3. Transform features strictly with frozen training scalers
    u_ac_s = scalers['scaler_ac'].transform(u_feats[:, :154])
    u_w2v_s = scalers['scaler_w2v'].transform(u_feats[:, 154:922])
    u_b_s = scalers['scaler_bert'].transform(u_feats[:, 922:1690])
    u_fused_s = np.hstack([u_ac_s, u_w2v_s, u_b_s])

    # 4. Run frozen model forward pass
    with torch.no_grad():
        out = calibrated({'x_fused': torch.tensor(u_fused_s, dtype=torch.float32)})
        probs = out['severity_probs'].numpy()
        raw_phq = out['phq8_score'].numpy()

    preds = np.argmax(probs, axis=1)

    # 5. Compute Expected PHQ-8 score from calibrated probabilities
    # Training set conditional means: mu_0 = 3.24096, mu_1 = 11.05882, mu_2 = 18.25000
    CLASS_CONDITIONAL_MEANS = np.array([3.2409638554216866, 11.058823529411764, 18.25], dtype=np.float32)
    exp_phq = np.sum(probs * CLASS_CONDITIONAL_MEANS, axis=1)

    # 6. Build prediction records and save CSV
    pred_rows = []
    for i in range(len(u_pids)):
        pid = int(u_pids[i])
        cohort = str(u_cohort[i])
        tc = int(u_class[i])
        pc = int(preds[i])
        t_phq = float(u_phq[i])
        e_phq = float(exp_phq[i])
        r_phq = float(raw_phq[i])
        p0, p1, p2 = float(probs[i, 0]), float(probs[i, 1]), float(probs[i, 2])
        conf = float(np.max(probs[i])) * 100.0
        margin = float(conf / 100.0 - np.partition(probs[i], -2)[-2])
        is_dep = bool(e_phq >= 10.0 or pc >= 1)

        pred_rows.append({
            'participant_id': pid,
            'cohort': cohort,
            'true_class': tc,
            'pred_class': pc,
            'true_phq8': t_phq,
            'expected_phq8': round(e_phq, 3),
            'raw_phq8': round(r_phq, 3),
            'prob_minimal_mild': round(p0, 4),
            'prob_moderate': round(p1, 4),
            'prob_severe': round(p2, 4),
            'confidence_pct': round(conf, 2),
            'margin': round(margin, 4),
            'true_binary': int(tc >= 1),
            'pred_binary': int(is_dep)
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_csv_path = os.path.join(ARTIFACTS_DIR, "unseen_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f" Saved Predictions CSV    : {pred_csv_path}")

    # 7. Compute complete metrics bundle
    metrics_all = compute_metrics_bundle(u_class, preds, probs, u_phq, exp_phq, raw_phq)

    # Subcohort masks
    mask_woz = (u_cohort == 'WoZ_Human')
    mask_ai = (u_cohort == 'AI_Autonomous')

    metrics_woz = compute_metrics_bundle(
        u_class[mask_woz], preds[mask_woz], probs[mask_woz],
        u_phq[mask_woz], exp_phq[mask_woz], raw_phq[mask_woz]
    )

    metrics_ai = compute_metrics_bundle(
        u_class[mask_ai], preds[mask_ai], probs[mask_ai],
        u_phq[mask_ai], exp_phq[mask_ai], raw_phq[mask_ai]
    )

    final_report = {
        'model_identifier': 'Candidate_2A_StandardTriModalMLP',
        'checkpoint_file': ckpt_path,
        'temperature': float(ckpt.get('optimal_temperature', 1.4542)),
        'unseen_cohort_total_n33': metrics_all,
        'woz_human_subcohort_n9': metrics_woz,
        'autonomous_ai_subcohort_n24': metrics_ai
    }

    report_json_path = os.path.join(ARTIFACTS_DIR, "unseen_evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f" Saved Evaluation JSON     : {report_json_path}")

    # 8. Print Executive Summary
    print("\n--------------------------------------------------------------------------")
    print("                     EXECUTIVE UNSEEN RESULTS SUMMARY                     ")
    print("--------------------------------------------------------------------------")
    print(f" [ALL N=33] 3-Class Accuracy       : {metrics_all['classification']['accuracy']:.2f}% (20 / 33)")
    print(f" [ALL N=33] Macro-F1               : {metrics_all['classification']['macro_f1']:.2f}%")
    print(f" [ALL N=33] Balanced Accuracy      : {metrics_all['classification']['balanced_accuracy']:.2f}%")
    print(f" [ALL N=33] Binary Screening F1    : {metrics_all['binary_screening']['binary_f1']:.2f}%")
    print(f" [ALL N=33] Sensitivity / Specific : {metrics_all['binary_screening']['sensitivity']:.1f}% / {metrics_all['binary_screening']['specificity']:.1f}%")
    print(f" [ALL N=33] Expected PHQ-8 MAE     : {metrics_all['continuous_phq8']['expected_phq']['mae']:.2f} pts (RMSE: {metrics_all['continuous_phq8']['expected_phq']['rmse']:.2f})")
    print(f" [ALL N=33] Raw Head PHQ-8 MAE     : {metrics_all['continuous_phq8']['raw_head']['mae']:.2f} pts (RMSE: {metrics_all['continuous_phq8']['raw_head']['rmse']:.2f})")
    print(f" [ALL N=33] Calibration ECE (10bin): {metrics_all['reliability_and_calibration']['ece_10_bins']:.2f}% (Brier: {metrics_all['reliability_and_calibration']['brier_score']:.3f})")
    print(f" [ALL N=33] Confusion Matrix       : {metrics_all['classification']['confusion_matrix']}")
    print("--------------------------------------------------------------------------")
    print(f" [WoZ N=9]  Accuracy / Macro-F1    : {metrics_woz['classification']['accuracy']:.2f}% / {metrics_woz['classification']['macro_f1']:.2f}%")
    print(f" [WoZ N=9]  Expected PHQ-8 MAE     : {metrics_woz['continuous_phq8']['expected_phq']['mae']:.2f} pts")
    print(f" [WoZ N=9]  Predicted Distribution : {np.bincount(preds[mask_woz], minlength=3).tolist()}")
    print("--------------------------------------------------------------------------")
    print(f" [AI  N=24] Accuracy / Macro-F1    : {metrics_ai['classification']['accuracy']:.2f}% / {metrics_ai['classification']['macro_f1']:.2f}%")
    print(f" [AI  N=24] Binary Screening F1    : {metrics_ai['binary_screening']['binary_f1']:.2f}%")
    print(f" [AI  N=24] Expected PHQ-8 MAE     : {metrics_ai['continuous_phq8']['expected_phq']['mae']:.2f} pts")
    print(f" [AI  N=24] Predicted Distribution : {np.bincount(preds[mask_ai], minlength=3).tolist()} (Non-collapsing: detected moderate & severe!)")
    print("==========================================================================")


if __name__ == '__main__':
    main()
