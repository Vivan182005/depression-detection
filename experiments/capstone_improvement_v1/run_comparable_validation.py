"""
Phase: Rigorous Apples-to-Apples Validation of Baseline vs. Candidate 2.
Evaluates both models on:
1. Exact same test participants (N=37).
2. Exact same authoritative ground-truth labels and class definitions.
3. Model-specific preprocessing and calibrated temperatures.
4. Exact same reserved unseen sessions (N=33 total: 9 WoZ human, 24 AI avatar).
Saves:
- experiments/capstone_improvement_v1/predictions/comparable_baseline_test_predictions.csv
- experiments/capstone_improvement_v1/predictions/comparable_candidate2_test_predictions.csv
- experiments/capstone_improvement_v1/predictions/comparable_unseen_predictions.csv
- experiments/capstone_improvement_v1/comparable_metrics.json
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
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
from src.models.calibrated_model import TemperatureScaler
from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified
from train_improvements import StandardTriModalMLP, load_dataset_features


def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece * 100.0)


def calculate_metrics_dict(y_true, y_pred, y_prob, phq_true, phq_pred, bin_true, bin_pred, bin_prob):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    support = np.bincount(y_true, minlength=3).tolist()

    bin_acc = float(accuracy_score(bin_true, bin_pred) * 100.0)
    bin_p = float(precision_score(bin_true, bin_pred, zero_division=0) * 100.0)
    bin_r = float(recall_score(bin_true, bin_pred, zero_division=0) * 100.0)
    bin_f1 = float(f1_score(bin_true, bin_pred, zero_division=0) * 100.0)
    
    p_cur, r_cur, _ = precision_recall_curve(bin_true, bin_prob)
    pr_auc = float(auc(r_cur, p_cur) * 100.0)

    mae = float(mean_absolute_error(phq_true, phq_pred))
    rmse = float(np.sqrt(mean_squared_error(phq_true, phq_pred)))
    ece = compute_ece(y_prob, y_true)

    return {
        'num_samples': len(y_true),
        'support': support,
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


def main():
    print("==========================================================================")
    print("       RUNNING APPLES-TO-APPLES VALIDATION OF BASELINE VS CANDIDATE 2     ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    print(f"Data Partitions: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # 1. Authoritative ground truth labels
    d_te = load_dataset_features(test_df, fs, clean_labels=True)
    pids = d_te['pids']
    sids = d_te['sids']
    y_clean = d_te['y_class']
    phq_clean = d_te['y_phq']
    bin_clean = (phq_clean >= 10.0).astype(int)

    # Legacy labels (for auditing the historical report)
    leg_labels_dict = master_df.set_index('session_id')['severity_class_id'].to_dict()
    leg_phq_dict = master_df.set_index('session_id')['phq8_score'].to_dict()
    y_leg = np.array([leg_labels_dict[s] for s in sids])
    phq_leg = np.array([leg_phq_dict[s] for s in sids])
    bin_leg = (phq_leg >= 10.0).astype(int)

    # Raw features for test
    X_raw = np.hstack([d_te['ac'], d_te['w2v'], d_te['bert']]).astype(np.float32)

    # ---------------------------------------------------------
    # Baseline Model Evaluation
    # ---------------------------------------------------------
    b_scaler_path = os.path.join(store_dir, "fitted_acoustic_scaler.joblib")
    b_scaler = joblib.load(b_scaler_path)
    X_base_scaled = b_scaler.transform(X_raw)

    b_ckpt_path = os.path.join(store_dir, "best_depression_model.pth")
    b_ckpt = torch.load(b_ckpt_path, map_location="cpu")
    b_model = CompactTriModalMLP(in_dim=1690, num_classes=3)
    b_model.load_state_dict(b_ckpt["model_state_dict"])
    b_model.eval()

    b_cal = TemperatureScaler(b_model)
    b_cal.temperature = torch.nn.Parameter(torch.tensor([float(b_ckpt.get("optimal_temperature", 1.2439))], dtype=torch.float32))
    b_cal.eval()

    with torch.no_grad():
        b_out = b_cal(torch.tensor(X_base_scaled, dtype=torch.float32))
        b_prob = b_out["severity_probs"].numpy()
        b_pred = np.argmax(b_prob, axis=-1)
        b_phq = b_out["phq8_score"].numpy()
        b_bin_prob = b_out["binary_probs"].numpy()
        b_bin_pred = (b_bin_prob >= float(b_ckpt.get("optimal_binary_threshold", 0.21))).astype(int)

    # ---------------------------------------------------------
    # Candidate 2 Model Evaluation
    # ---------------------------------------------------------
    c_scalers = joblib.load(os.path.join(EXP_DIR, "checkpoints", "feature_scalers.joblib"))
    ac_s = c_scalers["scaler_ac"].transform(d_te['ac'])
    w2v_s = c_scalers["scaler_w2v"].transform(d_te['w2v'])
    b_s = c_scalers["scaler_bert"].transform(d_te['bert'])
    X_cand_scaled = np.hstack([ac_s, w2v_s, b_s]).astype(np.float32)

    c_ckpt_path = os.path.join(EXP_DIR, "checkpoints", "cand2_standard_mlp_balanced.pth")
    c_ckpt = torch.load(c_ckpt_path, map_location="cpu")
    c_model = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3)
    c_model.load_state_dict(c_ckpt["model_state_dict"])
    c_model.eval()

    c_cal = TemperatureScaler(c_model)
    c_cal.temperature = torch.nn.Parameter(torch.tensor([float(c_ckpt["optimal_temperature"])], dtype=torch.float32))
    c_cal.eval()

    with torch.no_grad():
        c_out = c_cal(torch.tensor(X_cand_scaled, dtype=torch.float32))
        c_prob = c_out["severity_probs"].numpy()
        c_pred = np.argmax(c_prob, axis=-1)
        c_phq = c_out["phq8_score"].numpy()
        c_bin_prob = c_out["binary_probs"].numpy()
        c_bin_pred = (c_bin_prob >= 0.5).astype(int)

    # Calculate metrics on clean ground truth
    m_base_clean = calculate_metrics_dict(y_clean, b_pred, b_prob, phq_clean, b_phq, bin_clean, b_bin_pred, b_bin_prob)
    m_cand_clean = calculate_metrics_dict(y_clean, c_pred, c_prob, phq_clean, c_phq, bin_clean, c_bin_pred, c_bin_prob)

    # Calculate metrics on legacy labels (historical baseline)
    m_base_leg = calculate_metrics_dict(y_leg, b_pred, b_prob, phq_leg, b_phq, bin_leg, b_bin_pred, b_bin_prob)
    m_cand_leg = calculate_metrics_dict(y_leg, c_pred, c_prob, phq_leg, c_phq, bin_leg, c_bin_pred, c_bin_prob)

    # ---------------------------------------------------------
    # Save Test Prediction CSVs
    # ---------------------------------------------------------
    preds_dir = os.path.join(EXP_DIR, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    df_base_test = pd.DataFrame({
        'session_id': sids,
        'participant_id': pids,
        'clean_ground_truth_class': y_clean,
        'clean_ground_truth_phq8': phq_clean,
        'clean_ground_truth_binary': bin_clean,
        'legacy_phq8': phq_leg,
        'legacy_class': y_leg,
        'predicted_class': b_pred,
        'predicted_phq8': b_phq,
        'prob_minimal': b_prob[:, 0],
        'prob_moderate': b_prob[:, 1],
        'prob_severe': b_prob[:, 2],
        'predicted_binary': b_bin_pred,
        'prob_binary': b_bin_prob
    })
    base_csv_path = os.path.join(preds_dir, "comparable_baseline_test_predictions.csv")
    df_base_test.to_csv(base_csv_path, index=False)

    df_cand_test = pd.DataFrame({
        'session_id': sids,
        'participant_id': pids,
        'clean_ground_truth_class': y_clean,
        'clean_ground_truth_phq8': phq_clean,
        'clean_ground_truth_binary': bin_clean,
        'legacy_phq8': phq_leg,
        'legacy_class': y_leg,
        'predicted_class': c_pred,
        'predicted_phq8': c_phq,
        'prob_minimal': c_prob[:, 0],
        'prob_moderate': c_prob[:, 1],
        'prob_severe': c_prob[:, 2],
        'predicted_binary': c_bin_pred,
        'prob_binary': c_bin_prob
    })
    cand_csv_path = os.path.join(preds_dir, "comparable_candidate2_test_predictions.csv")
    df_cand_test.to_csv(cand_csv_path, index=False)

    # ---------------------------------------------------------
    # Unseen Cohort Evaluation
    # ---------------------------------------------------------
    unseen_cache_path = os.path.join(EXP_DIR, "unseen_feature_cache.npz")
    assert os.path.exists(unseen_cache_path), f"Unseen cache not found: {unseen_cache_path}"
    u_data = np.load(unseen_cache_path, allow_pickle=True)
    X_u_raw = u_data['features']
    pids_u = u_data['pids']
    phq_u = u_data['true_phq']
    y_u = u_data['true_class']
    cohort_u = u_data['cohort']
    bin_u = (phq_u >= 10.0).astype(int)

    # Baseline Unseen
    X_u_base_scaled = b_scaler.transform(X_u_raw)
    with torch.no_grad():
        b_u_out = b_cal(torch.tensor(X_u_base_scaled, dtype=torch.float32))
        b_u_prob = b_u_out["severity_probs"].numpy()
        b_u_pred = np.argmax(b_u_prob, axis=-1)
        b_u_phq = b_u_out["phq8_score"].numpy()
        b_u_bin_prob = b_u_out["binary_probs"].numpy()
        b_u_bin_pred = (b_u_bin_prob >= float(b_ckpt.get("optimal_binary_threshold", 0.21))).astype(int)

    # Candidate 2 Unseen
    u_ac_s = c_scalers["scaler_ac"].transform(X_u_raw[:, :154])
    u_w2v_s = c_scalers["scaler_w2v"].transform(X_u_raw[:, 154:154+768])
    u_b_s = c_scalers["scaler_bert"].transform(X_u_raw[:, 154+768:])
    X_u_cand_scaled = np.hstack([u_ac_s, u_w2v_s, u_b_s]).astype(np.float32)

    with torch.no_grad():
        c_u_out = c_cal(torch.tensor(X_u_cand_scaled, dtype=torch.float32))
        c_u_prob = c_u_out["severity_probs"].numpy()
        c_u_pred = np.argmax(c_u_prob, axis=-1)
        c_u_phq = c_u_out["phq8_score"].numpy()
        c_u_bin_prob = c_u_out["binary_probs"].numpy()
        c_u_bin_pred = (c_u_bin_prob >= 0.5).astype(int)

    df_unseen = pd.DataFrame({
        'participant_id': pids_u,
        'cohort': cohort_u,
        'clean_ground_truth_class': y_u,
        'clean_ground_truth_phq8': phq_u,
        'clean_ground_truth_binary': bin_u,
        'baseline_pred_class': b_u_pred,
        'baseline_pred_phq8': b_u_phq,
        'candidate2_pred_class': c_u_pred,
        'candidate2_pred_phq8': c_u_phq,
        'cand_prob_minimal': c_u_prob[:, 0],
        'cand_prob_moderate': c_u_prob[:, 1],
        'cand_prob_severe': c_u_prob[:, 2]
    })
    unseen_csv_path = os.path.join(preds_dir, "comparable_unseen_predictions.csv")
    df_unseen.to_csv(unseen_csv_path, index=False)

    # Subgroups
    is_woz = (cohort_u == "WoZ_Human")
    is_ai = (cohort_u == "AI_Autonomous")

    m_u_base_all = calculate_metrics_dict(y_u, b_u_pred, b_u_prob, phq_u, b_u_phq, bin_u, b_u_bin_pred, b_u_bin_prob)
    m_u_cand_all = calculate_metrics_dict(y_u, c_u_pred, c_u_prob, phq_u, c_u_phq, bin_u, c_u_bin_pred, c_u_bin_prob)

    m_u_base_woz = calculate_metrics_dict(y_u[is_woz], b_u_pred[is_woz], b_u_prob[is_woz], phq_u[is_woz], b_u_phq[is_woz], bin_u[is_woz], b_u_bin_pred[is_woz], b_u_bin_prob[is_woz])
    m_u_cand_woz = calculate_metrics_dict(y_u[is_woz], c_u_pred[is_woz], c_u_prob[is_woz], phq_u[is_woz], c_u_phq[is_woz], bin_u[is_woz], c_u_bin_pred[is_woz], c_u_bin_prob[is_woz])

    m_u_base_ai = calculate_metrics_dict(y_u[is_ai], b_u_pred[is_ai], b_u_prob[is_ai], phq_u[is_ai], b_u_phq[is_ai], bin_u[is_ai], b_u_bin_pred[is_ai], b_u_bin_prob[is_ai])
    m_u_cand_ai = calculate_metrics_dict(y_u[is_ai], c_u_pred[is_ai], c_u_prob[is_ai], phq_u[is_ai], c_u_phq[is_ai], bin_u[is_ai], c_u_bin_pred[is_ai], c_u_bin_prob[is_ai])

    # ---------------------------------------------------------
    # Assemble and Save Metrics JSON
    # ---------------------------------------------------------
    output_payload = {
        'evaluation_sample_count_test': 37,
        'evaluation_unique_participants_test': 35,
        'class_support_test_clean': [28, 5, 4],
        'class_support_test_legacy': [31, 4, 2],
        'held_out_test_evaluation': {
            'baseline_on_clean_ground_truth': m_base_clean,
            'candidate2_on_clean_ground_truth': m_cand_clean,
            'absolute_differences_clean': {
                'accuracy': round(m_cand_clean['accuracy'] - m_base_clean['accuracy'], 2),
                'macro_f1': round(m_cand_clean['macro_f1'] - m_base_clean['macro_f1'], 2),
                'balanced_accuracy': round(m_cand_clean['balanced_accuracy'] - m_base_clean['balanced_accuracy'], 2),
                'moderate_recall_diff': round(m_cand_clean['per_class_recall'][1] - m_base_clean['per_class_recall'][1], 2),
                'severe_recall_diff': round(m_cand_clean['per_class_recall'][2] - m_base_clean['per_class_recall'][2], 2),
                'binary_f1_diff': round(m_cand_clean['binary_f1'] - m_base_clean['binary_f1'], 2),
                'mae_diff': round(m_cand_clean['mae'] - m_base_clean['mae'], 4),
                'rmse_diff': round(m_cand_clean['rmse'] - m_base_clean['rmse'], 4)
            },
            'baseline_on_legacy_corrupted_labels': m_base_leg,
            'candidate2_on_legacy_corrupted_labels': m_cand_leg
        },
        'unseen_cohort_evaluation': {
            'overall_all_33': {
                'baseline': m_u_base_all,
                'candidate2': m_u_cand_all
            },
            'human_woz_group_9': {
                'baseline': m_u_base_woz,
                'candidate2': m_u_cand_woz
            },
            'autonomous_ai_group_24': {
                'baseline': m_u_base_ai,
                'candidate2': m_u_cand_ai
            }
        }
    }

    metrics_json_path = os.path.join(EXP_DIR, "comparable_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[SAVED] Comparable Test Predictions (Baseline): {base_csv_path}")
    print(f"[SAVED] Comparable Test Predictions (Candidate 2): {cand_csv_path}")
    print(f"[SAVED] Comparable Unseen Predictions: {unseen_csv_path}")
    print(f"[SAVED] Machine-Readable Comparable Metrics: {metrics_json_path}")
    print("Done!")


if __name__ == "__main__":
    main()
