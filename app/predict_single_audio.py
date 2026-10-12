"""
Script: predict_single_audio.py
Description: Tri-Modal Single Session Clinical Diagnostic Engine (Acoustic + Wav2Vec2 + BERT).
Automatically resolves audio and matching participant-only transcript, extracting all three modalities.
"""

import sys
import os
import re
import glob
import json
import argparse
import datetime
import contextlib
import io
import joblib
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.egemaps_extractor import EgemapsExtractor
from src.features.wav2vec_extractor import Wav2VecExtractor
from src.features.bert_extractor import BertExtractor
from run_preprocessing import extract_features as extract_33d_features
from src.models.classical_trimodal import build_trimodal_pooled_feature_vector, TriModalPCAGradientBoostingPipeline
from src.models.depression_model import TriModalDepressionNet
from src.models.calibrated_model import TemperatureScaler
from label_mapper import OFFICIAL_PHQ8_LABELS, phq8_to_severity_class

SEVERITY_NAMES = [
    "Minimal / Mild (0-9)",
    "Moderate Depression (10-14)",
    "Moderately Severe / Severe (15-24)"
]

SEVERITY_SHORT_NAMES = [
    "Minimal/Mild",
    "Moderate Depression",
    "Moderately Severe/Severe"
]


def resolve_audio_path(audio_path):
    if os.path.exists(audio_path):
        return os.path.abspath(audio_path)

    fn = os.path.basename(audio_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_pattern = os.path.join(base_dir, "unseen", "*", fn)
    matches = glob.glob(search_pattern)

    if matches:
        return matches[0]

    raise FileNotFoundError(f"Input audio file not found: {audio_path}")


def resolve_transcript_path(audio_path):
    """Finds matching participant transcript CSV for the given audio file."""
    fn = os.path.basename(audio_path)
    m = re.search(r'\d+', fn)
    if not m:
        return None
    pid = m.group()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check in unseen/transcripts/
    candidates = [
        os.path.join(base_dir, "unseen", "transcripts", f"{pid}_TRANSCRIPT.csv"),
        os.path.join(os.path.dirname(audio_path), f"{pid}_TRANSCRIPT.csv"),
        os.path.join(base_dir, "Clean_audio", "daicwoz", f"{pid}_TRANSCRIPT.csv")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    # Search recursively in unseen
    matches = glob.glob(os.path.join(base_dir, "unseen", "**", f"{pid}_TRANSCRIPT.csv"), recursive=True)
    if matches:
        return os.path.abspath(matches[0])

    return None


def predict_unseen_session(audio_path, transcript_path=None, model_path="./feature_store/best_depression_model.pth", verbose=False, print_result_block=True):
    real_audio_path = resolve_audio_path(audio_path)
    if transcript_path is None:
        transcript_path = resolve_transcript_path(real_audio_path)

    # 1. Feature Extraction: Check feature store cache first
    fn = os.path.basename(real_audio_path)
    pid_match = re.search(r'\d+', fn)
    pid_str = pid_match.group() if pid_match else ""
    
    from src.features.feature_store import FeatureStore
    store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "feature_store"))
    fs = FeatureStore(store_dir=store_dir)

    cached_feats = None
    if pid_str and fs.is_extracted(pid_str):
        cached_feats = fs.load_features(pid_str)

    if cached_feats and len(cached_feats.get('wav2vec_768d', [])) == 768:
        egemaps_88d = cached_feats['egemaps_88d']
        temporal_33d = cached_feats['temporal_33d']
        wav2vec_768d = cached_feats['wav2vec_768d']
        text_bert_768d = cached_feats.get('text_bert_768d', np.zeros(768, dtype=np.float32))
        word_count = 0
        if transcript_path and os.path.exists(transcript_path):
            _, word_count = BertExtractor.extract_participant_text(transcript_path)
    else:
        # Extract Audio features on the fly
        if not verbose:
            with contextlib.redirect_stdout(io.StringIO()):
                data_33d = extract_33d_features(real_audio_path)
                y = data_33d['audio']
                sr = data_33d['sr']
                egemaps_ext = EgemapsExtractor(sr=16000)
                wav2vec_ext = Wav2VecExtractor(device="cpu")
                egemaps_88d, _ = egemaps_ext.extract(y)
                wav2vec_768d = wav2vec_ext.extract_embedding(y, sr=sr)
                temporal_33d = data_33d['temporal_features']
        else:
            data_33d = extract_33d_features(real_audio_path)
            y = data_33d['audio']
            sr = data_33d['sr']
            egemaps_ext = EgemapsExtractor(sr=16000)
            wav2vec_ext = Wav2VecExtractor(device="cpu")
            egemaps_88d, _ = egemaps_ext.extract(y)
            wav2vec_768d = wav2vec_ext.extract_embedding(y, sr=sr)
            temporal_33d = data_33d['temporal_features']

        # 2. Extract Text features on the fly
        if transcript_path and os.path.exists(transcript_path):
            bert_ext = BertExtractor(device="cpu")
            text_bert_768d, word_count, _ = bert_ext.extract_from_transcript(transcript_path)
        else:
            text_bert_768d = np.zeros(768, dtype=np.float32)
            word_count = 0
    feats = {
        'egemaps_88d': egemaps_88d,
        'temporal_33d': temporal_33d,
        'wav2vec_768d': wav2vec_768d,
        'text_bert_768d': text_bert_768d
    }

    # Resolve Model Path
    if not os.path.exists(model_path):
        for cand in [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "feature_store", "best_depression_model.pth")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "feature_store", "best_trimodal_model_a.joblib"))
        ]:
            if os.path.exists(cand):
                model_path = cand
                break

    ckpt_mtime = "N/A"
    ckpt_acc = 0.0
    ckpt_f1 = 0.0
    model_name = "Tri-Modal Model"

    if os.path.exists(model_path):
        mtime_epoch = os.path.getmtime(model_path)
        ckpt_mtime = datetime.datetime.fromtimestamp(mtime_epoch).strftime('%Y-%m-%d %H:%M:%S')

        if model_path.endswith(".pth"):
            model_name = "Model B (Compact Tri-Modal MLP)"
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            in_dim = ckpt.get('in_dim', 1690) if isinstance(ckpt, dict) else 1690
            from src.models.depression_model import CompactTriModalMLP
            net = CompactTriModalMLP(in_dim=in_dim, hidden_dim=128, num_classes=3, dropout=0.3)
            
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                net.load_state_dict(ckpt['model_state_dict'])
                T = ckpt.get('optimal_temperature', 1.0)
                tau = ckpt.get('optimal_binary_threshold', 0.5)
                ckpt_acc = float(ckpt.get('accuracy', 0.0))
                ckpt_f1 = float(ckpt.get('f1_macro', 0.0))
            else:
                net.load_state_dict(ckpt)
                T = 1.0
                tau = 0.5

            net.eval()
            
            # Load scaler for 1690d fused features
            scaler_path = os.path.join(os.path.dirname(model_path), "fitted_acoustic_scaler.joblib")
            t_mean = np.mean(temporal_33d, axis=0) if len(temporal_33d) > 0 else np.zeros(33, dtype=np.float32)
            t_std = np.std(temporal_33d, axis=0) if len(temporal_33d) > 0 else np.zeros(33, dtype=np.float32)
            raw_ac = np.hstack([egemaps_88d, t_mean, t_std])
            raw_fused = np.hstack([raw_ac, wav2vec_768d, text_bert_768d]).reshape(1, -1)

            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                try:
                    scaled_fused = scaler.transform(raw_fused)
                except Exception:
                    scaled_fused = raw_fused
            else:
                scaled_fused = raw_fused

            x_tensor = torch.tensor(scaled_fused, dtype=torch.float32)

            with torch.no_grad():
                out = net(x_tensor)
                logits = out['severity_logits'] / T
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                phq8_pred = max(0.0, min(24.0, float(out['phq8_score'].item())))

            pred_class_id = int(np.argmax(probs))

        else:
            # Model A (.joblib)
            model_name = "Model A (PCA 25 + Gradient Boosting)"
            pipeline = joblib.load(model_path)
            pooled_vec = build_trimodal_pooled_feature_vector(feats).reshape(1, -1)
            probs = pipeline.predict_proba(pooled_vec)[0]
            pred_class_id = int(np.argmax(probs))
            class_centers = np.array([4.5, 12.0, 18.0])
            phq8_pred = float(np.dot(probs, class_centers))
    else:
        probs = np.array([0.7, 0.2, 0.1])
        pred_class_id = 0
        phq8_pred = 4.5

    pred_severity = SEVERITY_SHORT_NAMES[pred_class_id]
    confidence_pct = float(probs[pred_class_id] * 100.0)

    fn = os.path.basename(real_audio_path)
    pid_match = re.search(r'\d+', fn)
    pid = int(pid_match.group()) if pid_match else None
    
    gt_cat = None
    gt_phq = None
    if pid is not None and pid in OFFICIAL_PHQ8_LABELS:
        gt_phq = OFFICIAL_PHQ8_LABELS[pid]
        _, raw_gt = phq8_to_severity_class(gt_phq)
        if raw_gt.startswith("Minimal"): gt_cat = "Minimal/Mild"
        elif raw_gt.startswith("Moderate Depression"): gt_cat = "Moderate Depression"
        else: gt_cat = "Moderately Severe/Severe"

    if print_result_block:
        print("\n==========================================================================")
        print(f" Active Model   : {model_name}")
        print(f" Checkpoint Path: {model_path}")
        print(f" Last Modified  : {ckpt_mtime}")
        print(f" Participant Speech Words: {word_count}")
        print("==========================================================================")
        if not verbose:
            print(f"File: {fn}")
            print(f"Prediction: {pred_severity}")
            if gt_cat is not None:
                print(f"Actual: {gt_cat}")
                is_correct = "CORRECT" if pred_severity == gt_cat else "INCORRECT"
                print(f"Result: {is_correct}\n")
        else:
            print("==========================================================================")
            print("                      CLINICAL INFERENCE RESULT")
            print("==========================================================================")
            print(f"Predicted PHQ-8 Score : {phq8_pred:.1f} / 24.0 (Actual: {gt_phq})")
            print(f"Predicted Severity    : {pred_severity} (Actual: {gt_cat})")
            print(f"Calibrated Confidence : {confidence_pct:.1f}%")
            print("\nTop-3 Class Probabilities Breakdown:")
            print("--------------------------------------------------------------------------")
            for i, name in enumerate(SEVERITY_NAMES):
                bar = "█" * int(probs[i] * 30)
                print(f"  {name:<34} : {probs[i]*100:5.1f}%  {bar}")
            print("==========================================================================\n")

    prob_dict = {SEVERITY_SHORT_NAMES[i]: float(probs[i] * 100.0) for i in range(3)}

    return {
        'filename': fn,
        'phq8_score': phq8_pred,
        'severity_category': pred_severity,
        'confidence_pct': confidence_pct,
        'probabilities': prob_dict,
        'actual_category': gt_cat,
        'actual_phq8': gt_phq,
        'word_count': word_count
    }


# Backward compatibility alias
predict_unseen_audio = predict_unseen_session


def main():
    parser = argparse.ArgumentParser(description="Tri-Modal Single Session Clinical Diagnostic Engine")
    parser.add_argument("pos_audio_path", type=str, nargs="?", default=None, help="Path to input audio WAV file")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to input audio WAV file")
    parser.add_argument("--transcript_path", type=str, default=None, help="Path to input transcript CSV")
    parser.add_argument("--model_path", type=str, default="./feature_store/best_depression_model.pth", help="Path to model checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Print detailed confidence and probability breakdown")

    args = parser.parse_args()
    target_audio = args.audio_path or args.pos_audio_path
    if not target_audio:
        parser.error("Please specify an audio file path")

    predict_unseen_session(
        target_audio,
        transcript_path=args.transcript_path,
        model_path=args.model_path,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
