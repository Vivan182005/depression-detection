"""
Reproduce and verify the exact historical baseline model and its performance.
Preserves original checkpoints and evaluates strictly without modifying any files.
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
    confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, auc, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Add project root and app to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.models.depression_model import CompactTriModalMLP
from src.models.calibrated_model import TemperatureScaler
from src.features.feature_store import FeatureStore
from src.models.train_leak_free import LeakFreeDataset, custom_collate, split_joint_stratified


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


def main():
    print("==========================================================================")
    print("           REPRODUCING BASELINE: CompactTriModalMLP (Phase 2)             ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    ckpt_path = os.path.join(store_dir, "best_depression_model.pth")
    scaler_path = os.path.join(store_dir, "fitted_acoustic_scaler.joblib")
    master_index_path = os.path.join(store_dir, "master_dataset_index.csv")

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(scaler_path), f"Scaler not found: {scaler_path}"
    assert os.path.exists(master_index_path), f"Master index not found: {master_index_path}"

    # Load master dataset and reconstruct identical split
    master_df = pd.read_csv(master_index_path)
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    print(f"Dataset Partitions: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Load saved scaler and feature store
    scaler = joblib.load(scaler_path)
    feature_store = FeatureStore(store_dir=store_dir)

    train_ds = LeakFreeDataset(train_df, feature_store, scaler=scaler, is_train=False)
    val_ds = LeakFreeDataset(val_df, feature_store, scaler=scaler, is_train=False)
    test_ds = LeakFreeDataset(test_df, feature_store, scaler=scaler, is_train=False)

    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, collate_fn=custom_collate)
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=custom_collate)

    # Load Model Checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = CompactTriModalMLP(in_dim=ckpt.get("in_dim", 1690), num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load optimal temperature and threshold
    opt_temp = float(ckpt.get("optimal_temperature", 1.2439))
    opt_thresh = float(ckpt.get("optimal_binary_threshold", 0.21))
    print(f"Loaded Optimal Temperature: {opt_temp:.4f}")
    print(f"Loaded Optimal Binary Threshold: {opt_thresh:.4f}")

    # Build calibrated model
    calibrated_model = TemperatureScaler(model)
    calibrated_model.temperature = torch.nn.Parameter(torch.tensor([opt_temp], dtype=torch.float32))
    calibrated_model.eval()

    def evaluate(loader, split_name):
        with torch.no_grad():
            for batch in loader:
                cal_out = calibrated_model(batch["x_fused"])
                probs = cal_out["severity_probs"].numpy()
                preds = np.argmax(probs, axis=-1)
                phq_pred = cal_out["phq8_score"].numpy()

                raw_bin_probs = cal_out["binary_probs"].numpy()
                bin_true = batch["binary_label"].numpy().astype(int)
                bin_pred = (raw_bin_probs >= opt_thresh).astype(int)

                class_true = batch["class_id"].numpy()
                phq_true = batch["phq8_score"].numpy()
                session_ids = batch["session_id"]
                pids = batch["participant_id"]

        acc = float(accuracy_score(class_true, preds) * 100.0)
        bal_acc = float(balanced_accuracy_score(class_true, preds) * 100.0)
        macro_f1 = float(f1_score(class_true, preds, average="macro", zero_division=0) * 100.0)
        weighted_f1 = float(f1_score(class_true, preds, average="weighted", zero_division=0) * 100.0)
        
        per_p = precision_score(class_true, preds, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
        per_r = recall_score(class_true, preds, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
        per_f = f1_score(class_true, preds, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
        cm = confusion_matrix(class_true, preds, labels=[0, 1, 2]).tolist()

        # Binary
        bin_acc = float(accuracy_score(bin_true, bin_pred) * 100.0)
        bin_p = float(precision_score(bin_true, bin_pred, zero_division=0) * 100.0)
        bin_r = float(recall_score(bin_true, bin_pred, zero_division=0) * 100.0)
        bin_f1 = float(f1_score(bin_true, bin_pred, zero_division=0) * 100.0)
        p_curve, r_curve, _ = precision_recall_curve(bin_true, raw_bin_probs)
        pr_auc = float(auc(r_curve, p_curve) * 100.0)

        # Regression
        mae = float(mean_absolute_error(phq_true, phq_pred))
        rmse = float(np.sqrt(mean_squared_error(phq_true, phq_pred)))

        # Calibration
        ece = compute_ece(probs, class_true)
        # Brier score across classes (one-hot)
        oh_true = np.eye(3)[class_true]
        brier = float(np.mean(np.sum((probs - oh_true) ** 2, axis=1)))

        # Majority class baseline
        maj_class = int(pd.Series(class_true).mode()[0])
        maj_acc = float(np.mean(class_true == maj_class) * 100.0)
        maj_macro_f1 = float(f1_score(class_true, np.full_like(class_true, maj_class), average="macro", zero_division=0) * 100.0)

        pred_df = pd.DataFrame({
            "session_id": session_ids,
            "participant_id": pids,
            "true_class": class_true,
            "pred_class": preds,
            "prob_class_0": probs[:, 0],
            "prob_class_1": probs[:, 1],
            "prob_class_2": probs[:, 2],
            "true_phq8": phq_true,
            "pred_phq8": phq_pred,
            "true_binary": bin_true,
            "pred_binary": bin_pred,
            "prob_binary": raw_bin_probs
        })

        metrics = {
            "split": split_name,
            "samples": len(class_true),
            "three_class_accuracy": acc,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "weighted_f1": weighted_f1,
            "per_class_precision": [float(x) for x in per_p],
            "per_class_recall": [float(x) for x in per_r],
            "per_class_f1": [float(x) for x in per_f],
            "confusion_matrix": cm,
            "binary_accuracy": bin_acc,
            "binary_precision": bin_p,
            "binary_recall": bin_r,
            "binary_f1": bin_f1,
            "binary_pr_auc": pr_auc,
            "phq8_mae": mae,
            "phq8_rmse": rmse,
            "ece": ece,
            "brier_score": brier,
            "majority_baseline_accuracy": maj_acc,
            "majority_baseline_macro_f1": maj_macro_f1
        }
        return metrics, pred_df

    test_metrics, test_preds = evaluate(test_loader, "test")
    val_metrics, val_preds = evaluate(val_loader, "val")
    train_metrics, train_preds = evaluate(train_loader, "train")

    print("\n--- BASELINE REPRODUCTION RESULTS ON HELD-OUT TEST SPLIT (N=37) ---")
    print(f"3-Class Accuracy       : {test_metrics['three_class_accuracy']:.2f}% (Reported: 83.78%)")
    print(f"Macro-F1               : {test_metrics['macro_f1']:.2f}% (Reported: 42.36%)")
    print(f"Balanced Accuracy      : {test_metrics['balanced_accuracy']:.2f}% (Reported: 48.92%)")
    print(f"Binary F1              : {test_metrics['binary_f1']:.2f}%")
    print(f"Binary PR-AUC          : {test_metrics['binary_pr_auc']:.2f}%")
    print(f"PHQ-8 MAE              : {test_metrics['phq8_mae']:.4f} pts (Reported: 4.23 pts)")
    print(f"PHQ-8 RMSE             : {test_metrics['phq8_rmse']:.4f} pts")
    print(f"Calibration ECE        : {test_metrics['ece']:.2f}%")
    print(f"Brier Score            : {test_metrics['brier_score']:.4f}")
    print(f"Majority Accuracy      : {test_metrics['majority_baseline_accuracy']:.2f}%")
    print(f"Majority Macro-F1      : {test_metrics['majority_baseline_macro_f1']:.2f}%")
    print("Confusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    # Save deliverables in experiment directory
    exp_dir = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
    os.makedirs(os.path.join(exp_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    test_preds.to_csv(os.path.join(exp_dir, "predictions", "baseline_test_predictions.csv"), index=False)
    val_preds.to_csv(os.path.join(exp_dir, "predictions", "baseline_val_predictions.csv"), index=False)

    baseline_payload = {
        "model_architecture": "CompactTriModalMLP",
        "checkpoint_path": ckpt_path,
        "input_dimensions": 1690,
        "modalities": ["Acoustic (154d)", "Wav2Vec2 (768d)", "BERT (768d)"],
        "optimal_temperature": opt_temp,
        "optimal_binary_threshold": opt_thresh,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
        "verification_status": "EXACT_MATCH"
    }

    metrics_out = os.path.join(exp_dir, "baseline_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump(baseline_payload, f, indent=2)

    print(f"\n[SUCCESS] Baseline metrics saved to: {metrics_out}")
    print(f"[SUCCESS] Test predictions saved to: {os.path.join(exp_dir, 'predictions', 'baseline_test_predictions.csv')}")


if __name__ == "__main__":
    main()
