"""
Phase 9: Rigorous Evaluation on the Reserved Unseen Cohort (N=33).
Evaluates both the Baseline and Candidate 2 on exactly the same 33 unseen sessions.
Breaks down results by:
- Overall Unseen Cohort (N=33)
- Wizard-of-Oz Human Interviewer Cohort (N=9)
- Autonomous AI Interviewer Cohort (N=24)
Saves unseen_evaluation.json and detailed comparison predictions.
"""

import os
import sys
import glob
import re
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve, auc
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.egemaps_extractor import EgemapsExtractor
from src.features.wav2vec_extractor import Wav2VecExtractor
from src.features.bert_extractor import BertExtractor
from run_preprocessing import extract_features as extract_33d_features
from src.models.depression_model import CompactTriModalMLP
from src.models.calibrated_model import TemperatureScaler
from train_improvements import StandardTriModalMLP


def compute_metrics_unseen(y_true, y_pred, y_prob, phq_true, phq_pred):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)

    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    y_bin_true = (y_true >= 1).astype(int)
    prob_depressed = 1.0 - y_prob[:, 0]
    y_bin_pred = (prob_depressed >= 0.5).astype(int)

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


def extract_or_load_unseen_features(unseen_dir, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached unseen features from: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {
            'features': d['features'],
            'pids': d['pids'],
            'true_phq': d['true_phq'],
            'true_class': d['true_class'],
            'cohort': d['cohort']
        }

    print("Extracting Tri-Modal features for all 33 unseen sessions...")
    clean_labels_df = pd.read_csv(os.path.join(BASE_DIR, "new_audio_csv", "detailed_lables.csv"))
    clean_map = dict(zip(clean_labels_df['Participant'], clean_labels_df['Depression_severity']))

    per_session_cache_dir = os.path.join(EXP_DIR, "unseen_cache")
    os.makedirs(per_session_cache_dir, exist_ok=True)

    egemaps_ext = None
    wav2vec_ext = None
    bert_ext = None

    audio_files = sorted(glob.glob(os.path.join(unseen_dir, "**", "*.wav"), recursive=True))
    features_list, pids_list, phq_list, class_list, cohort_list = [], [], [], [], []

    for idx, wav_path in enumerate(audio_files):
        fn = os.path.basename(wav_path)
        pid = int(re.search(r'\d+', fn).group())
        cohort = "WoZ_Human" if pid < 600 else "AI_Autonomous"

        # Check per-session cache first
        per_cache = os.path.join(per_session_cache_dir, f"{pid}_features.npz")
        fs_direct = os.path.join(APP_DIR, "feature_store", f"{pid}_features.npz")
        fs_loud = os.path.join(APP_DIR, "feature_store", f"loud_audio{pid}_features.npz")

        if os.path.exists(per_cache):
            d = np.load(per_cache)
            f_fused = d['feature_vector']
            print(f"[{idx+1}/{len(audio_files)}] PID {pid} ({cohort}) loaded from local unseen cache.")
        elif os.path.exists(fs_direct) or os.path.exists(fs_loud):
            src = fs_direct if os.path.exists(fs_direct) else fs_loud
            d = np.load(src)
            e88 = d['egemaps_88d']
            t_mean = np.mean(d['temporal_33d'], axis=0) if len(d['temporal_33d']) > 0 else np.zeros(33, dtype=np.float32)
            t_std = np.std(d['temporal_33d'], axis=0) if len(d['temporal_33d']) > 0 else np.zeros(33, dtype=np.float32)
            w768 = d['wav2vec_768d']
            b768 = d['text_bert_768d']
            f_fused = np.hstack([e88, t_mean, t_std, w768, b768]).astype(np.float32)
            np.savez_compressed(per_cache, feature_vector=f_fused)
            print(f"[{idx+1}/{len(audio_files)}] PID {pid} ({cohort}) loaded from app/feature_store.")
        else:
            print(f"[{idx+1}/{len(audio_files)}] Extracting features for PID {pid} ({cohort})...", end=" ", flush=True)
            if egemaps_ext is None:
                egemaps_ext = EgemapsExtractor(sr=16000)
                wav2vec_ext = Wav2VecExtractor(device="cpu")
                bert_ext = BertExtractor(device="cpu")

            data_33d = extract_33d_features(wav_path)
            y, sr = data_33d['audio'], data_33d['sr']
            e88, _ = egemaps_ext.extract(y)

            # Fast chunked Wav2Vec extraction to prevent quadratic self-attention blowup on CPU
            max_chunk_samples = 15 * sr
            if len(y) <= max_chunk_samples:
                w768 = wav2vec_ext.extract_embedding(y, sr=sr)
            else:
                chunks = [y[i:i+max_chunk_samples] for i in range(0, len(y), max_chunk_samples) if len(y[i:i+max_chunk_samples]) >= sr]
                if len(chunks) > 6:
                    step = len(chunks) / 6.0
                    chunks = [chunks[int(i * step)] for i in range(6)]
                embs = [wav2vec_ext.extract_embedding(c, sr=sr) for c in chunks]
                w768 = np.mean(embs, axis=0).astype(np.float32) if embs else np.zeros(768, dtype=np.float32)

            # Transcript
            trans_candidates = [
                os.path.join(unseen_dir, "transcripts", f"{pid}_TRANSCRIPT.csv"),
                os.path.join(os.path.dirname(wav_path), f"{pid}_TRANSCRIPT.csv")
            ]
            t_text = ""
            for tc in trans_candidates:
                if os.path.exists(tc):
                    t_text, _ = bert_ext.extract_participant_text(tc)
                    break
            b768 = bert_ext.extract_from_text(t_text) if t_text else np.zeros(768, dtype=np.float32)

            t_mean = np.mean(data_33d['temporal_features'], axis=0) if len(data_33d['temporal_features']) > 0 else np.zeros(33, dtype=np.float32)
            t_std = np.std(data_33d['temporal_features'], axis=0) if len(data_33d['temporal_features']) > 0 else np.zeros(33, dtype=np.float32)

            f_fused = np.hstack([e88, t_mean, t_std, w768, b768]).astype(np.float32)
            np.savez_compressed(per_cache, feature_vector=f_fused)
            print(f"Done.")

        true_phq = float(clean_map.get(pid, 0.0))
        true_cls = 0 if true_phq <= 9.0 else (1 if true_phq <= 14.0 else 2)

        features_list.append(f_fused)
        pids_list.append(pid)
        phq_list.append(true_phq)
        class_list.append(true_cls)
        cohort_list.append(cohort)

    f_arr = np.array(features_list, dtype=np.float32)
    p_arr = np.array(pids_list, dtype=int)
    phq_arr = np.array(phq_list, dtype=np.float32)
    c_arr = np.array(class_list, dtype=int)
    coh_arr = np.array(cohort_list, dtype=object)

    np.savez_compressed(
        cache_path,
        features=f_arr,
        pids=p_arr,
        true_phq=phq_arr,
        true_class=c_arr,
        cohort=coh_arr
    )
    print(f"Saved cached unseen features to: {cache_path}")

    return {
        'features': f_arr,
        'pids': p_arr,
        'true_phq': phq_arr,
        'true_class': c_arr,
        'cohort': coh_arr
    }


def main():
    print("==========================================================================")
    print("        PHASE 9: STANDALONE EVALUATION ON 33 RESERVED UNSEEN SESSIONS     ")
    print("==========================================================================")

    unseen_dir = os.path.join(BASE_DIR, "unseen")
    cache_path = os.path.join(EXP_DIR, "unseen_feature_cache.npz")
    unseen_data = extract_or_load_unseen_features(unseen_dir, cache_path)

    X_unseen_raw = unseen_data['features']
    y_true = unseen_data['true_class']
    phq_true = unseen_data['true_phq']
    pids = unseen_data['pids']
    cohorts = unseen_data['cohort']

    # 1. Evaluate Baseline Model
    baseline_ckpt_path = os.path.join(APP_DIR, "feature_store", "best_depression_model.pth")
    baseline_scaler_path = os.path.join(APP_DIR, "feature_store", "fitted_acoustic_scaler.joblib")
    b_scaler = joblib.load(baseline_scaler_path)
    X_base_scaled = b_scaler.transform(X_unseen_raw)

    b_ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    b_model = CompactTriModalMLP(in_dim=1690, num_classes=3)
    b_model.load_state_dict(b_ckpt["model_state_dict"])
    b_model.eval()

    b_cal = TemperatureScaler(b_model)
    b_cal.temperature = torch.nn.Parameter(torch.tensor([float(b_ckpt.get("optimal_temperature", 1.2439))], dtype=torch.float32))
    b_cal.eval()

    with torch.no_grad():
        b_out = b_cal(torch.tensor(X_base_scaled, dtype=torch.float32))
        b_prob = b_out['severity_probs'].numpy()
        b_pred = np.argmax(b_prob, axis=-1)
        b_phq = b_out['phq8_score'].numpy()

    # 2. Evaluate Best Candidate (Candidate 2: Standard MLP with Balanced Loss)
    cand_ckpt_path = os.path.join(EXP_DIR, "checkpoints", "cand2_standard_mlp_balanced.pth")
    scalers_dict = joblib.load(os.path.join(EXP_DIR, "checkpoints", "feature_scalers.joblib"))
    sc_ac, sc_w2v, sc_b = scalers_dict['scaler_ac'], scalers_dict['scaler_w2v'], scalers_dict['scaler_bert']

    ac_s = sc_ac.transform(X_unseen_raw[:, :154])
    w2v_s = sc_w2v.transform(X_unseen_raw[:, 154:154+768])
    b_s = sc_b.transform(X_unseen_raw[:, 154+768:])
    X_cand_scaled = np.hstack([ac_s, w2v_s, b_s])

    c_ckpt = torch.load(cand_ckpt_path, map_location="cpu")
    c_model = StandardTriModalMLP(in_dim=1690, hidden_dim=128, num_classes=3)
    c_model.load_state_dict(c_ckpt["model_state_dict"])
    c_model.eval()

    c_cal = TemperatureScaler(c_model)
    c_cal.temperature = torch.nn.Parameter(torch.tensor([float(c_ckpt["optimal_temperature"])], dtype=torch.float32))
    c_cal.eval()

    with torch.no_grad():
        c_out = c_cal(torch.tensor(X_cand_scaled, dtype=torch.float32))
        c_prob = c_out['severity_probs'].numpy()
        c_pred = np.argmax(c_prob, axis=-1)
        c_phq = c_out['phq8_score'].numpy()

    # Compute overall and subgroup metrics
    is_woz = (cohorts == "WoZ_Human")
    is_ai = (cohorts == "AI_Autonomous")

    m_base_all = compute_metrics_unseen(y_true, b_pred, b_prob, phq_true, b_phq)
    m_cand_all = compute_metrics_unseen(y_true, c_pred, c_prob, phq_true, c_phq)

    m_base_woz = compute_metrics_unseen(y_true[is_woz], b_pred[is_woz], b_prob[is_woz], phq_true[is_woz], b_phq[is_woz])
    m_cand_woz = compute_metrics_unseen(y_true[is_woz], c_pred[is_woz], c_prob[is_woz], phq_true[is_woz], c_phq[is_woz])

    m_base_ai = compute_metrics_unseen(y_true[is_ai], b_pred[is_ai], b_prob[is_ai], phq_true[is_ai], b_phq[is_ai])
    m_cand_ai = compute_metrics_unseen(y_true[is_ai], c_pred[is_ai], c_prob[is_ai], phq_true[is_ai], c_phq[is_ai])

    print("\n" + "="*95)
    print(f"{'Unseen Cohort Partition':30s} | {'Baseline Acc / Macro-F1':28s} | {'Candidate Acc / Macro-F1':28s}")
    print("="*95)
    print(f"{'1. All 33 Unseen Sessions':30s} | {m_base_all['accuracy']:5.2f}% / {m_base_all['macro_f1']:5.2f}% (MAE {m_base_all['mae']:.2f})  | {m_cand_all['accuracy']:5.2f}% / {m_cand_all['macro_f1']:5.2f}% (MAE {m_cand_all['mae']:.2f})")
    print(f"{'2. WoZ Human Group (N=9)':30s} | {m_base_woz['accuracy']:5.2f}% / {m_base_woz['macro_f1']:5.2f}% (MAE {m_base_woz['mae']:.2f})  | {m_cand_woz['accuracy']:5.2f}% / {m_cand_woz['macro_f1']:5.2f}% (MAE {m_cand_woz['mae']:.2f})")
    print(f"{'3. AI Autonomous Group (N=24)':30s} | {m_base_ai['accuracy']:5.2f}% / {m_base_ai['macro_f1']:5.2f}% (MAE {m_base_ai['mae']:.2f})  | {m_cand_ai['accuracy']:5.2f}% / {m_cand_ai['macro_f1']:5.2f}% (MAE {m_cand_ai['mae']:.2f})")
    print("="*95)

    print("\nConfusion Matrix (Candidate 2 on 33 Unseen Sessions):")
    print(np.array(m_cand_all['confusion_matrix']))

    # Save detailed prediction comparison table
    df_compare = pd.DataFrame({
        'participant_id': pids,
        'cohort': cohorts,
        'true_phq': phq_true,
        'true_class': y_true,
        'baseline_pred_class': b_pred,
        'baseline_pred_phq': b_phq,
        'candidate_pred_class': c_pred,
        'candidate_pred_phq': c_phq,
        'cand_prob_minimal': c_prob[:, 0],
        'cand_prob_moderate': c_prob[:, 1],
        'cand_prob_severe': c_prob[:, 2]
    })
    preds_out = os.path.join(EXP_DIR, "predictions", "unseen_predictions_comparison.csv")
    df_compare.to_csv(preds_out, index=False)

    unseen_payload = {
        'total_evaluated_sessions': 33,
        'failed_sessions': 0,
        'baseline_model': {
            'overall': m_base_all,
            'woz_group': m_base_woz,
            'ai_group': m_base_ai
        },
        'best_candidate_model': {
            'model_name': 'Candidate 2 (Standard MLP with Smoothed Balanced Loss on Clean Ground-Truth)',
            'overall': m_cand_all,
            'woz_group': m_cand_woz,
            'ai_group': m_cand_ai
        }
    }

    out_json = os.path.join(EXP_DIR, "unseen_evaluation.json")
    with open(out_json, "w") as f:
        json.dump(unseen_payload, f, indent=2)

    print(f"\n[SUCCESS] Unseen evaluation saved to: {out_json}")
    print(f"[SUCCESS] Prediction comparison saved to: {preds_out}")


if __name__ == "__main__":
    main()
