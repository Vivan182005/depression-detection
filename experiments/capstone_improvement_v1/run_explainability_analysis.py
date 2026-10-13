"""
Phase 10: Explainability, Modality Attribution, and Error Analysis.
Performs:
1. Permutation importance across the three modalities (Acoustic 154d, Wav2Vec2 768d, BERT 768d).
2. Gradient x Input attribution across individual acoustic descriptors (eGeMAPS + temporal).
3. Detailed error case analysis comparing Baseline vs Candidate 2.
4. Generates visualizations and saves explainability_summary.json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.models.calibrated_model import TemperatureScaler
from train_improvements import (
    StandardTriModalMLP, FeatureStore, split_joint_stratified, load_dataset_features
)


def run_explainability():
    print("==========================================================================")
    print("        PHASE 10: MULTIMODAL EXPLAINABILITY & ERROR ANALYSIS              ")
    print("==========================================================================")

    # 1. Load test data using canonical benchmark split
    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    d_te = load_dataset_features(test_df, fs, clean_labels=True)
    y_test = d_te['y_class']
    phq_test = d_te['y_phq']
    pids_test = d_te['pids']
    X_raw = np.hstack([d_te['ac'], d_te['w2v'], d_te['bert']]).astype(np.float32)

    # Scale features
    scalers = joblib.load(os.path.join(EXP_DIR, "checkpoints", "feature_scalers.joblib"))
    sc_ac, sc_w2v, sc_b = scalers['scaler_ac'], scalers['scaler_w2v'], scalers['scaler_bert']

    ac_scaled = sc_ac.transform(X_raw[:, :154])
    w2v_scaled = sc_w2v.transform(X_raw[:, 154:154+768])
    b_scaled = sc_b.transform(X_raw[:, 154+768:])
    X_scaled = np.hstack([ac_scaled, w2v_scaled, b_scaled]).astype(np.float32)

    # Load Candidate 2 Model
    cand_ckpt_path = os.path.join(EXP_DIR, "checkpoints", "cand2_standard_mlp_balanced.pth")
    ckpt = torch.load(cand_ckpt_path, map_location="cpu")
    model = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cal_model = TemperatureScaler(model)
    cal_model.temperature = torch.nn.Parameter(torch.tensor([float(ckpt["optimal_temperature"])], dtype=torch.float32))
    cal_model.eval()

    # Base inference
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        out = cal_model(X_t)
        base_probs = out['severity_probs'].numpy()
        base_preds = np.argmax(base_probs, axis=-1)
        base_phq = out['phq8_score'].numpy()

    from sklearn.metrics import f1_score, mean_absolute_error
    base_macro_f1 = f1_score(y_test, base_preds, average='macro', zero_division=0) * 100.0
    base_mae = mean_absolute_error(phq_test, base_phq)

    # Permutation Modality Importance
    print("\n--- 1. Permutation Modality Importance (N=37 Held-Out Test Set) ---")
    np.random.seed(42)
    modality_slices = {
        "Acoustic_154d": slice(0, 154),
        "Wav2Vec2_768d": slice(154, 154 + 768),
        "BERT_Text_768d": slice(154 + 768, 1690)
    }

    perm_results = {}
    for mod_name, sl in modality_slices.items():
        drops_f1 = []
        increases_mae = []
        for _ in range(20):
            X_perm = X_scaled.copy()
            perm_indices = np.random.permutation(len(X_perm))
            X_perm[:, sl] = X_perm[perm_indices, sl]

            with torch.no_grad():
                out_p = cal_model(torch.tensor(X_perm, dtype=torch.float32))
                p_probs = out_p['severity_probs'].numpy()
                p_preds = np.argmax(p_probs, axis=-1)
                p_phq = out_p['phq8_score'].numpy()

            f1_p = f1_score(y_test, p_preds, average='macro', zero_division=0) * 100.0
            mae_p = mean_absolute_error(phq_test, p_phq)
            drops_f1.append(base_macro_f1 - f1_p)
            increases_mae.append(mae_p - base_mae)

        mean_drop_f1 = float(np.mean(drops_f1))
        mean_inc_mae = float(np.mean(increases_mae))
        perm_results[mod_name] = {
            'macro_f1_drop': mean_drop_f1,
            'mae_increase': mean_inc_mae
        }
        print(f"Modality {mod_name:18s}: Macro-F1 Drop = {mean_drop_f1:+.2f}%, PHQ-8 MAE Increase = {mean_inc_mae:+.2f} pts")

    # Plot Modality Importance
    plt.figure(figsize=(8, 4))
    bars = plt.bar(list(perm_results.keys()), [v['macro_f1_drop'] for v in perm_results.values()], color=['#4C72B0', '#55A868', '#C44E52'])
    plt.title("Multimodal Permutation Feature Importance\n(Drop in Test Macro-F1 When Shuffled)", fontsize=12, fontweight='bold')
    plt.ylabel("Macro-F1 Drop (%)", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plot_mod_path = os.path.join(EXP_DIR, "plots", "modality_importance_permutation.png")
    plt.savefig(plot_mod_path, dpi=200)
    plt.close()
    print(f"Saved modality permutation plot: {plot_mod_path}")

    # 2. Gradient x Input Attribution for Acoustic Descriptors
    print("\n--- 2. Gradient x Input Attribution for Acoustic Sub-Vectors ---")
    X_t_grad = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
    out_grad = model(X_t_grad)
    phq_out = out_grad['phq8_score']
    # Backprop sum of PHQ predictions
    phq_out.sum().backward()
    grad_input = (X_t_grad.grad * X_t_grad).detach().numpy()

    # Slice acoustic features (0:154)
    ac_attr = np.mean(np.abs(grad_input[:, :154]), axis=0)
    egemaps_attr = float(np.mean(ac_attr[:88]))
    temporal_mean_attr = float(np.mean(ac_attr[88:88+33]))
    temporal_std_attr = float(np.mean(ac_attr[88+33:154]))
    print(f"eGeMAPS 88d mean absolute attribution:        {egemaps_attr:.6f}")
    print(f"Temporal 33d Mean mean absolute attribution:  {temporal_mean_attr:.6f}")
    print(f"Temporal 33d Std mean absolute attribution:   {temporal_std_attr:.6f}")

    # Top acoustic features
    top_ac_indices = np.argsort(ac_attr)[::-1][:15]
    print("\nTop 10 Most Salient Acoustic Dimensions:")
    for rank, idx in enumerate(top_ac_indices[:10]):
        group_name = "eGeMAPS" if idx < 88 else ("Temporal_Mean" if idx < 121 else "Temporal_Std")
        sub_idx = idx if idx < 88 else (idx - 88 if idx < 121 else idx - 121)
        print(f"Rank {rank+1:2d}: Dim {idx:3d} ({group_name} #{sub_idx:2d}) Attribution = {ac_attr[idx]:.6f}")

    # 3. Error Case Analysis: Baseline vs Candidate 2
    print("\n--- 3. Error Case Analysis (Held-Out Test Set, N=37) ---")
    baseline_preds_df = pd.read_csv(os.path.join(EXP_DIR, "predictions", "baseline_test_predictions.csv"))
    cand2_preds_df = pd.read_csv(os.path.join(EXP_DIR, "predictions", "cand2_standard_mlp_balanced_test_preds.csv"))

    merged_errors = pd.merge(
        baseline_preds_df[['participant_id', 'true_class', 'true_phq8', 'pred_class', 'pred_phq8']],
        cand2_preds_df[['participant_id', 'pred_class', 'pred_phq']],
        on='participant_id',
        suffixes=('_base', '_cand2')
    )

    # Cases where Baseline collapsed but Candidate 2 succeeded:
    moderate_saved = merged_errors[(merged_errors['true_class'] == 1) & (merged_errors['pred_class_base'] != 1) & (merged_errors['pred_class_cand2'] == 1)]
    severe_saved = merged_errors[(merged_errors['true_class'] == 2) & (merged_errors['pred_class_base'] != 2) & (merged_errors['pred_class_cand2'] == 2)]

    print(f"Moderate participants collapsed by Baseline but correctly identified by Candidate 2: {len(moderate_saved)}")
    for _, row in moderate_saved.iterrows():
        print(f"  PID {int(row['participant_id'])}: True PHQ={row['true_phq8']}, Base Pred={row['pred_class_base']}, Cand2 Pred={row['pred_class_cand2']}")

    print(f"Severe participants collapsed by Baseline but correctly identified by Candidate 2: {len(severe_saved)}")
    for _, row in severe_saved.iterrows():
        print(f"  PID {int(row['participant_id'])}: True PHQ={row['true_phq8']}, Base Pred={row['pred_class_base']}, Cand2 Pred={row['pred_class_cand2']}")

    # Save summary
    explainability_payload = {
        'modality_permutation_importance': perm_results,
        'acoustic_subvector_attribution': {
            'egemaps_88d': egemaps_attr,
            'temporal_mean_33d': temporal_mean_attr,
            'temporal_std_33d': temporal_std_attr
        },
        'top_acoustic_feature_indices': [int(x) for x in top_ac_indices],
        'error_analysis': {
            'moderate_rescued_count': len(moderate_saved),
            'severe_rescued_count': len(severe_saved),
            'moderate_rescued_pids': [int(x) for x in moderate_saved['participant_id'].tolist()],
            'severe_rescued_pids': [int(x) for x in severe_saved['participant_id'].tolist()]
        },
        'clinical_disclaimer': (
            "IMPORTANT: This multimodal system is an investigational research prototype designed "
            "for automated depression screening and severity estimation. It is NOT a diagnostic tool "
            "and does not replace clinical evaluation by a licensed mental health professional."
        )
    }

    out_json = os.path.join(EXP_DIR, "explainability_summary.json")
    with open(out_json, "w") as f:
        json.dump(explainability_payload, f, indent=2)
    print(f"\n[SUCCESS] Explainability summary saved to: {out_json}")


if __name__ == "__main__":
    run_explainability()
