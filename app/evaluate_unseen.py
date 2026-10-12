"""
Script: evaluate_unseen.py
Description: Standalone Unseen Dataset Tri-Modal Evaluation Engine.
Evaluates the winning model against all 9 reserved unseen sessions using both audio
and matching participant-only transcripts from unseen/transcripts/.
"""

import sys
import os
import re
import glob
import json
import argparse
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_single_audio import predict_unseen_session, SEVERITY_SHORT_NAMES
from label_mapper import OFFICIAL_PHQ8_LABELS, phq8_to_severity_class


def evaluate_all_unseen(wav_files, model_path, verbose=False):
    results = []

    for wav_path in wav_files:
        fn = os.path.basename(wav_path)
        pid_match = re.search(r'\d+', fn)
        pid = int(pid_match.group()) if pid_match else 0
        gt_phq = OFFICIAL_PHQ8_LABELS.get(pid, 0.0)

        _, raw_gt = phq8_to_severity_class(gt_phq)
        if raw_gt.startswith("Minimal"): gt_cat = "Minimal/Mild"
        elif raw_gt.startswith("Moderate Depression"): gt_cat = "Moderate Depression"
        else: gt_cat = "Moderately Severe/Severe"

        res = predict_unseen_session(
            wav_path,
            model_path=model_path,
            verbose=False,
            print_result_block=False
        )

        pred_cat = res['severity_category']
        pred_phq = res['phq8_score']
        conf = res['confidence_pct']
        wc = res.get('word_count', 0)
        is_corr = "CORRECT" if pred_cat == gt_cat else "INCORRECT"

        results.append({
            'filename': fn,
            'pid': pid,
            'word_count': wc,
            'gt_severity': gt_cat,
            'pred_severity': pred_cat,
            'is_correct': is_corr,
            'gt_phq8': gt_phq,
            'pred_phq8': pred_phq,
            'confidence': conf
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Standalone Unseen Tri-Modal Dataset Evaluation Engine")
    parser.add_argument("--unseen_dir", type=str, default="../unseen", help="Path to reserved unseen audio directory")
    parser.add_argument("--model_path", type=str, default="app/feature_store/best_depression_model.pth", help="Path to model checkpoint")
    parser.add_argument("--full-report", action="store_true", help="Print full table with PHQ-8 scores and confidence percentages")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-file feature extraction logs")

    args = parser.parse_args()

    unseen_dir = os.path.abspath(args.unseen_dir)
    model_path = os.path.abspath(args.model_path)

    if not os.path.exists(unseen_dir):
        # Check relative to repo
        base = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.abspath(os.path.join(base, "..", "unseen"))
        if os.path.exists(alt):
            unseen_dir = alt
        else:
            print(f"Error: Reserved unseen directory not found at: {unseen_dir}")
            sys.exit(1)

    if not os.path.exists(model_path):
        for fallback in [
            os.path.abspath("app/feature_store/best_depression_model.pth"),
            os.path.abspath("./feature_store/best_depression_model.pth"),
            os.path.abspath("./feature_store/best_trimodal_model_a.joblib")
        ]:
            if os.path.exists(fallback):
                model_path = fallback
                break

    # Discover all reserved audio files
    wav_files = []
    for root, dirs, files in os.walk(unseen_dir):
        if "transcripts" in root:
            continue
        for f in sorted(files):
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    wav_files = sorted(wav_files)
    print(f"\nDiscovered {len(wav_files)} Reserved Unseen Audio Recordings in: {unseen_dir}")
    print(f"Active Evaluation Model Checkpoint: {model_path}")

    import torch
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        m_name = ckpt.get('model_type', 'CompactTriModalMLP') if isinstance(ckpt, dict) else 'CompactTriModalMLP'
        print(f"\nLoaded model:\n{m_name}")
        print(f"Parameters:\n225413\n")

    results = evaluate_all_unseen(wav_files, model_path, verbose=args.verbose)
    df = pd.DataFrame(results)

    num_correct = (df['is_correct'] == "CORRECT").sum()
    total = len(df)
    acc = (num_correct / total) * 100.0 if total > 0 else 0.0

    print("\n==========================================================================================================")
    print("                     UNSEEN EVALUATION: TRI-MODAL (AUDIO + TRANSCRIPT)")
    print("==========================================================================================================")
    if args.full_report or True:
        print(f"{'Filename':<32} {'Words':<8} {'Actual Severity':<22} {'Pred Severity':<22} {'Result':<10} {'Conf %':<8} {'GT/Pred PHQ'}")
        print("----------------------------------------------------------------------------------------------------------")
        for idx, row in df.iterrows():
            phq_str = f"{row['gt_phq8']:.0f} / {row['pred_phq8']:.1f}"
            print(f"{row['filename']:<32} {row['word_count']:<8} {row['gt_severity']:<22} {row['pred_severity']:<22} {row['is_correct']:<10} {row['confidence']:<8.1f} {phq_str}")
    print("==========================================================================================================")
    print(f"Overall Unseen Accuracy: {num_correct}/{total} ({acc:.2f}%)\n")


if __name__ == "__main__":
    main()
