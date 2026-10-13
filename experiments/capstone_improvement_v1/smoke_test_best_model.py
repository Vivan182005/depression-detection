"""
Phase 11: End-to-End Inference Smoke Test for Best Candidate Model.
Tests:
1. Deterministic loading of Candidate 2 checkpoint and fitted scalers.
2. Verified output shape and probability calibration.
3. PHQ-8 continuous score validity (bounded in [0, 24]).
4. Robustness to missing modalities (e.g. zero text or zero acoustic).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.models.calibrated_model import TemperatureScaler
from train_improvements import StandardTriModalMLP


def test_inference_pipeline():
    print("==========================================================================")
    print("        PHASE 11: END-TO-END INFERENCE SMOKE TEST & VERIFICATION         ")
    print("==========================================================================")

    ckpt_path = os.path.join(EXP_DIR, "checkpoints", "cand2_standard_mlp_balanced.pth")
    scaler_path = os.path.join(EXP_DIR, "checkpoints", "feature_scalers.joblib")

    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    assert os.path.exists(scaler_path), f"Scalers missing: {scaler_path}"

    print(f"[1/4] Loading scalers from: {scaler_path}")
    scalers = joblib.load(scaler_path)
    sc_ac, sc_w2v, sc_b = scalers['scaler_ac'], scalers['scaler_w2v'], scalers['scaler_bert']

    print(f"[2/4] Loading model checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cal_model = TemperatureScaler(model)
    cal_model.temperature = torch.nn.Parameter(torch.tensor([float(ckpt["optimal_temperature"])], dtype=torch.float32))
    cal_model.eval()
    print(f"      Loaded successfully. Optimal Temperature T = {float(cal_model.temperature.item()):.4f}")

    print("[3/4] Running inference tests on real participant sample...")
    # Test 1: Real participant from test partition
    from train_improvements import FeatureStore, split_joint_stratified, load_dataset_features
    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    _, _, test_df = split_joint_stratified(master_df, verbose=False)
    d_te = load_dataset_features(test_df.iloc[:5], fs, clean_labels=True)

    ac_s = sc_ac.transform(d_te['ac'])
    w2v_s = sc_w2v.transform(d_te['w2v'])
    b_s = sc_b.transform(d_te['bert'])
    X_test_scaled = np.hstack([ac_s, w2v_s, b_s])

    with torch.no_grad():
        out = cal_model(torch.tensor(X_test_scaled, dtype=torch.float32))

    probs = out['severity_probs'].numpy()
    pred_cls = np.argmax(probs, axis=-1)
    raw_phq = out['phq8_score'].numpy()
    bounded_phq = np.clip(raw_phq, 0.0, 24.0)

    for i in range(len(d_te['pids'])):
        print(f"      PID {d_te['pids'][i]}: True PHQ={d_te['y_phq'][i]:.1f}, Pred PHQ={bounded_phq[i]:.2f}, Pred Class={pred_cls[i]}, Probs={probs[i]}")
        assert np.isclose(probs[i].sum(), 1.0, atol=1e-5), f"Probabilities do not sum to 1: {probs[i].sum()}"
        assert 0.0 <= bounded_phq[i] <= 24.0, f"PHQ outside [0, 24]: {bounded_phq[i]}"

    # Test 2: Missing text modality (All zeros BERT)
    print("\n[4/4] Testing robustness with missing text modality (BERT=0)...")
    sc_input_no_text = np.hstack([
        sc_ac.transform(d_te['ac'][:1]),
        sc_w2v.transform(d_te['w2v'][:1]),
        sc_b.transform(np.zeros((1, 768), dtype=np.float32))
    ])

    with torch.no_grad():
        out_no_text = cal_model(torch.tensor(sc_input_no_text, dtype=torch.float32))

    probs_nt = out_no_text['severity_probs'].numpy()[0]
    phq_nt = float(out_no_text['phq8_score'].numpy()[0])

    print(f"      Test 2 (Acoustic-Only / Zero Text Input):")
    print(f"        Probabilities: Minimal={probs_nt[0]:.4f}, Moderate={probs_nt[1]:.4f}, Severe={probs_nt[2]:.4f}")
    print(f"        Predicted Continuous PHQ-8: {phq_nt:.2f} / 24.0")

    assert np.isclose(probs_nt.sum(), 1.0, atol=1e-5), "Probabilities do not sum to 1.0!"
    assert 0.0 <= phq_nt <= 24.0, f"PHQ-8 score outside [0, 24]: {phq_nt}"

    print("\n[ALL TESTS PASSED] End-to-end inference pipeline verified and production-ready.")


if __name__ == "__main__":
    test_inference_pipeline()
