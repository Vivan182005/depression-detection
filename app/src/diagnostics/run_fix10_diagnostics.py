"""
Script: run_fix10_diagnostics.py
Description: Fix Prompt 10 Diagnostics Suite:
1. Feature Usability Check on Severe vs Minimal/Mild sessions.
2. Feature Space PCA 2D Class Separability Analysis.
3. Raw Model Output Probability Analysis on Test-Fold Severe Sessions.
"""

import sys
import os
import glob
import re
import json
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.feature_store import FeatureStore
from label_mapper import build_master_dataset_index, OFFICIAL_PHQ8_LABELS
from src.models.train_leak_free import split_joint_stratified_70_15_15, LeakFreeDataset, custom_collate
from src.models.depression_model import DualBranchDepressionNet, HAS_TORCH
from src.models.classical_trimodal import build_trimodal_pooled_feature_vector as build_pooled_feature_vector

if HAS_TORCH:
    import torch
    from torch.utils.data import DataLoader


def run_diagnostic_1(master_df, feature_store):
    print("\n==========================================================================")
    print("                DIAGNOSTIC 1: FEATURE USABILITY ANALYSIS")
    print("==========================================================================")

    severe_df = master_df[master_df['severity_class_id'] == 2].copy()
    mild_df = master_df[master_df['severity_class_id'] == 0].sample(n=5, random_state=42).copy()

    def analyze_sessions(df_sub, label):
        print(f"\n--- {label} Sessions ({len(df_sub)} sessions) ---")
        print(f"{'Session ID':<15} {'Frames':<8} {'NaN/Inf?':<10} {'MinVal':<10} {'MaxVal':<10} {'MeanVal':<10} {'Audio Duration'}")
        print("----------------------------------------------------------------------------------")

        audio_dir = "/Users/vivannaik/Desktop/CAPSTONE/data"

        for idx, row in df_sub.iterrows():
            sid = row['session_id']
            feats = feature_store.load_features(sid)
            t_33d = feats['temporal_33d']
            e_88d = feats['egemaps_88d']
            w_768d = feats['wav2vec_768d']

            vec = build_pooled_feature_vector(feats)
            has_nan_inf = np.isnan(vec).any() or np.isinf(vec).any()

            min_val = float(np.min(vec))
            max_val = float(np.max(vec))
            mean_val = float(np.mean(vec))
            num_frames = len(t_33d)

            # Locate WAV file duration
            wav_matches = glob.glob(os.path.join(audio_dir, "*", f"{sid}_AUDIO*.wav")) or glob.glob(os.path.join(audio_dir, f"{sid}*.wav"))
            dur_str = "N/A"
            if wav_matches:
                try:
                    info = sf.info(wav_matches[0])
                    dur_str = f"{info.duration:.1f}s"
                except Exception:
                    pass

            print(f"{sid:<15} {num_frames:<8} {str(has_nan_inf):<10} {min_val:<10.2f} {max_val:<10.2f} {mean_val:<10.2f} {dur_str}")

    analyze_sessions(severe_df, "Moderately Severe / Severe (Class 2)")
    analyze_sessions(mild_df, "Sampled Minimal / Mild (Class 0)")


def run_diagnostic_2(master_df, feature_store):
    print("\n==========================================================================")
    print("           DIAGNOSTIC 2: FEATURE SPACE PCA 2D SEPARABILITY")
    print("==========================================================================")

    X_all, y_all = [], []
    for idx, row in master_df.iterrows():
        feats = feature_store.load_features(row['session_id'])
        vec = build_pooled_feature_vector(feats)
        X_all.append(vec)
        y_all.append(int(row['severity_class_id']))

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Calculate centroid distances
    c0 = np.mean(X_pca[y_all == 0], axis=0)
    c1 = np.mean(X_pca[y_all == 1], axis=0)
    c2 = np.mean(X_pca[y_all == 2], axis=0)

    dist_0_2 = np.linalg.norm(c0 - c2)
    dist_1_2 = np.linalg.norm(c1 - c2)
    dist_0_1 = np.linalg.norm(c0 - c1)

    print(f"PCA Variance Explained by Top 2 Components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"Centroid Minimal/Mild (Class 0)         : [{c0[0]:.2f}, {c0[1]:.2f}]")
    print(f"Centroid Moderate (Class 1)             : [{c1[0]:.2f}, {c1[1]:.2f}]")
    print(f"Centroid Moderately Severe/Severe (2)   : [{c2[0]:.2f}, {c2[1]:.2f}]")
    print("Euclidean Distances Between Class Centroids:")
    print(f"  Class 0 <-> Class 1 : {dist_0_1:.3f}")
    print(f"  Class 0 <-> Class 2 : {dist_0_2:.3f}")
    print(f"  Class 1 <-> Class 2 : {dist_1_2:.3f}")

    # Plot & save diagnostic scatter plot
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    labels = ['Minimal/Mild (154)', 'Moderate (20)', 'Mod.Severe/Severe (9)']

    for c in range(3):
        mask = (y_all == c)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[c], label=labels[c], alpha=0.7, edgecolors='k', s=60)

    plt.scatter([c0[0], c1[0], c2[0]], [c0[1], c1[1], c2[1]], color='black', marker='X', s=150, label='Centroids')
    plt.title('Diagnostic 2: PCA 2D Feature Space Separability')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "feature_store", "diagnostic2_pca_scatter.png"))
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"PCA Scatter Plot Saved To: {out_path}")


def run_diagnostic_3(train_df, val_df, test_df, feature_store, store_dir):
    print("\n==========================================================================")
    print("         DIAGNOSTIC 3: TEST-FOLD SEVERE SESSIONS PROBABILITY ANALYSIS")
    print("==========================================================================")

    model_path = os.path.join(store_dir, "best_depression_model.pth")
    sidecar_path = os.path.join(store_dir, "best_depression_model_temperature.json")

    eff_temp = 1.44
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as f:
            eff_temp = json.load(f).get('optimal_temperature', 1.44)

    scaler = StandardScaler()
    train_dataset = LeakFreeDataset(train_df, feature_store, scaler=scaler, is_train=True)
    test_dataset = LeakFreeDataset(test_df, feature_store, scaler=scaler, is_train=False)

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=custom_collate)

    model = DualBranchDepressionNet(hidden_dim=64, num_classes=3, dropout=0.4)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)

    model.eval()

    severe_test_found = 0
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['egemaps'], batch['temporal'], batch['wav2vec'])
            raw_logits = outputs['severity_logits']
            cal_probs = torch.softmax(raw_logits / eff_temp, dim=-1).numpy()
            true_class = batch['class_id'].numpy()
            s_ids = batch['session_id']
            phq_true = batch['phq8_score'].numpy()
            phq_pred = outputs['phq8_score'].numpy()

            print(f"{'Session ID':<15} {'GT PHQ':<8} {'Pred PHQ':<10} {'Minimal/Mild %':<16} {'Moderate %':<14} {'Mod.Severe/Severe %':<20} {'Top Argmax'}")
            print("----------------------------------------------------------------------------------------------------------")

            for i in range(len(true_class)):
                if true_class[i] == 2: # Severe
                    severe_test_found += 1
                    p0, p1, p2 = cal_probs[i][0]*100, cal_probs[i][1]*100, cal_probs[i][2]*100
                    argmax_c = ["Minimal/Mild", "Moderate", "Mod.Severe/Severe"][int(np.argmax(cal_probs[i]))]
                    print(f"{s_ids[i]:<15} {phq_true[i]:<8.1f} {phq_pred[i]:<10.1f} {p0:<16.1f}% {p1:<14.1f}% {p2:<20.1f}% {argmax_c}")

            if severe_test_found == 0:
                print("  [NOTE] Test fold contained 0 Severe sessions (all Severe samples ended up in Train fold).")


def main():
    store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "feature_store"))
    feature_store = FeatureStore(store_dir=store_dir)
    master_df = build_master_dataset_index(store_dir=store_dir)

    train_df, val_df, test_df = split_joint_stratified_70_15_15(master_df, verbose=False)

    run_diagnostic_1(master_df, feature_store)
    run_diagnostic_2(master_df, feature_store)
    run_diagnostic_3(train_df, val_df, test_df, feature_store, store_dir)


if __name__ == "__main__":
    main()
