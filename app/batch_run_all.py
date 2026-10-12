"""
Script: batch_run_all.py
Description: Batch Runner for processing a whole directory of audio files.
Creates a dedicated output subfolder for each audio session containing:
1. Clean audio without Ellie (.wav)
2. Attention Overlayed on Speech Waveform (.png)
3. Attention vs Loudness Over Time (.png)
4. Biomarker Profile Chart (.png)
5. Clinical Diagnostic Screening Report (.txt)
Also generates a global BATCH_SUMMARY_REPORT.csv listing all session results.
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_full_slide_pipeline import execute_slide_pipeline


def run_batch_all(audio_dir="../Audio", base_output_dir="./output"):
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    os.makedirs(base_output_dir, exist_ok=True)

    audio_files = [
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
        if f.endswith('.wav') and not f.endswith('_PARTICIPANT_ONLY.wav') and not f.endswith('_CLEAN.wav')
    ]
    audio_files.sort()

    print("\n==========================================================================")
    print(f"      BATCH MASTER PIPELINE: PROCESSING {len(audio_files)} RECORDINGS")
    print(f"      Output Directory: {os.path.abspath(base_output_dir)}")
    print("==========================================================================")

    batch_summary = []

    for i, audio_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] Batch Processing: {os.path.basename(audio_path)}")
        try:
            res = execute_slide_pipeline(audio_path, base_output_dir=base_output_dir)
            batch_summary.append({
                'session_id': res['session_id'],
                'predicted_category': res['category'],
                'confidence_score': f"{res['confidence']:.2f}%",
                'estimated_phq8': res['phq8'],
                'session_output_folder': res['output_dir']
            })
        except Exception as e:
            print(f"[ERROR] Failed to process {audio_path}: {e}")

    summary_df = pd.DataFrame(batch_summary)
    summary_path = os.path.join(base_output_dir, "BATCH_SUMMARY_REPORT.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n==========================================================================")
    print(f"BATCH PROCESSING COMPLETE!")
    print(f"Summary Table Saved To : {summary_path}")
    print(f"Individual Session Output Subfolders Created in : {os.path.abspath(base_output_dir)}")
    print("==========================================================================\n")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Batch Runner for Processing Audio Files into Dedicated Output Subfolders")
    parser.add_argument("--audio_dir", type=str, default="../Audio", help="Directory containing audio WAV files")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base output directory")

    args = parser.parse_args()
    run_batch_all(args.audio_dir, args.output_dir)


if __name__ == "__main__":
    main()
