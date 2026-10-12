"""
Pipeline: pipeline.py
Description: Master End-to-End Execution Pipeline for Automatic Depression Detection from Voice Cues.
Integrates speaker isolation, acoustic preprocessing, dual-branch feature extraction, 
BiLSTM-Attention inference, and Explainable AI (XAI) output visualizers.
"""

import sys
import os
import argparse
import numpy as np

# Add parent app directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.speaker_isolation import isolate_participant_audio
from src.preprocessing.audio_cleaner import preprocess_audio_signal
from src.features.egemaps_extractor import EgemapsExtractor
from src.features.wav2vec_extractor import Wav2VecExtractor
from src.models.dataset import DepressionSessionDataset
from src.models.depression_model import DualBranchDepressionNet, HAS_TORCH
from src.explainability.visualizer import (
    plot_temporal_attention,
    plot_prosodic_biomarkers,
    generate_diagnostic_summary
)


def run_pipeline(audio_path, transcript_path=None, output_dir="./output"):
    """
    Executes full depression detection pipeline on a given interview session.

    Parameters:
        audio_path (str): Path to session audio file (.wav)
        transcript_path (str, optional): Path to DAIC-WOZ transcript file (.csv/.tsv)
        output_dir (str): Directory to save clean audio, heatmap plots, and report
    """
    session_id = os.path.basename(audio_path).split('_')[0]
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print(f"       STARTING DEPRESSION DETECTION PIPELINE FOR SESSION {session_id}")
    print("="*80)

    # -------------------------------------------------------------------------
    # STAGE 1: Interviewer (Ellie) Removal & Speaker Isolation
    # -------------------------------------------------------------------------
    clean_audio_path = os.path.join(output_dir, f"{session_id}_PARTICIPANT_CLEAN.wav")
    
    if transcript_path and os.path.exists(transcript_path):
        print("\n[STAGE 1] Running Transcript-Driven Speaker Isolation...")
        raw_participant_audio, sr = isolate_participant_audio(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_path=None,
            sr=16000,
            crossfade_ms=5
        )
    else:
        print("\n[STAGE 1] Transcript not provided. Loading continuous audio signal...")
        try:
            import librosa
            raw_participant_audio, sr = librosa.load(audio_path, sr=16000)
        except ImportError:
            from scipy.io import wavfile
            sr, raw_participant_audio = wavfile.read(audio_path)
            if raw_participant_audio.ndim > 1: raw_participant_audio = raw_participant_audio.mean(axis=1)
            raw_participant_audio = raw_participant_audio.astype(np.float32) / (np.max(np.abs(raw_participant_audio)) + 1e-9)

    # -------------------------------------------------------------------------
    # STAGE 2: Signal Preprocessing & Denoising
    # -------------------------------------------------------------------------
    print("\n[STAGE 2] Applying Bandpass Filter, Spectral Denoising & RMS Normalization...")
    clean_audio = preprocess_audio_signal(raw_participant_audio, sr=sr, apply_denoise=True)

    # Save cleaned participant audio
    try:
        import soundfile as sf
        sf.write(clean_audio_path, clean_audio, sr)
        print(f"  -> Cleaned audio saved to: {clean_audio_path}")
    except Exception as e:
        print(f"  -> Audio save notice: {e}")

    # -------------------------------------------------------------------------
    # STAGE 3: Dual-Branch Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[STAGE 3] Initializing Feature Extractor Engines...")
    egemaps_ext = EgemapsExtractor(sr=sr)
    wav2vec_ext = Wav2VecExtractor(device="cpu")

    print("\n[STAGE 4] Creating 3.0s Sliding Window Dataset (50% Overlap)...")
    dataset = DepressionSessionDataset(
        audio_signal=clean_audio,
        sr=sr,
        window_sec=3.0,
        hop_sec=1.5,
        egemaps_extractor=egemaps_ext,
        wav2vec_extractor=wav2vec_ext
    )

    num_chunks = len(dataset)
    print(f"  -> Total Window Chunks Generated: {num_chunks}")

    if num_chunks == 0:
        print("[ERROR] Audio duration too short to generate window chunks.")
        return

    # Extract sequences for all chunks
    egemaps_list = []
    wav2vec_list = []
    chunk_times = []

    for item in dataset:
        egemaps_list.append(item['egemaps'])
        wav2vec_list.append(item['wav2vec'])
        chunk_times.append(item['time'])

    egemaps_seq = np.array(egemaps_list) # (N_chunks, 88)
    wav2vec_seq = np.array(wav2vec_list) # (N_chunks, 768)

    # Extract global session pause metrics
    _, pause_metrics = egemaps_ext.extract(clean_audio)

    # -------------------------------------------------------------------------
    # STAGE 5: Deep Learning Model Inference
    # -------------------------------------------------------------------------
    print("\n[STAGE 5] Running Dual-Branch BiLSTM + Self-Attention Model Inference...")
    
    if HAS_TORCH:
        import torch
        model = DualBranchDepressionNet()
        model.eval()

        egemaps_tensor = torch.tensor(egemaps_seq, dtype=torch.float32).unsqueeze(0) # (1, N_chunks, 88)
        wav2vec_tensor = torch.tensor(wav2vec_seq, dtype=torch.float32).unsqueeze(0) # (1, N_chunks, 768)

        with torch.no_grad():
            outputs = model(egemaps_tensor, wav2vec_tensor)
            depression_prob = float(outputs['depression_prob'].item())
            phq8_score = float(outputs['phq8_score'].item())
            attention_weights = outputs['attention_weights'].squeeze(0).cpu().numpy()
    else:
        model = DualBranchDepressionNet()
        outputs = model.predict_numpy(egemaps_seq, wav2vec_seq)
        depression_prob = outputs['depression_prob']
        phq8_score = outputs['phq8_score']
        attention_weights = outputs['attention_weights']

    # Normalize attention weights to sum to 1
    attention_weights = attention_weights / (np.sum(attention_weights) + 1e-9)

    # -------------------------------------------------------------------------
    # STAGE 6: Explainable AI (XAI) Visualizations & Diagnostic Report
    # -------------------------------------------------------------------------
    print("\n[STAGE 6] Generating Clinical Explainability Visualizations & Screening Report...")
    
    # 1. Plot Temporal Attention Timeline
    heatmap_path = os.path.join(output_dir, f"{session_id}_attention_heatmap.png")
    plot_temporal_attention(attention_weights, chunk_times, session_id=session_id, save_path=heatmap_path)

    # 2. Plot Vocal Biomarkers
    biomarker_path = os.path.join(output_dir, f"{session_id}_biomarkers.png")
    plot_prosodic_biomarkers(pause_metrics, save_path=biomarker_path)

    # 3. Print Diagnostic Summary Report
    report = generate_diagnostic_summary(depression_prob, phq8_score, pause_metrics, session_id=session_id)
    print(report)

    report_path = os.path.join(output_dir, f"{session_id}_diagnostic_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n[PIPELINE COMPLETE] Output artifacts saved to directory: {output_dir}")
    return {
        'session_id': session_id,
        'depression_prob': depression_prob,
        'phq8_score': phq8_score,
        'pause_metrics': pause_metrics,
        'clean_audio_path': clean_audio_path,
        'heatmap_path': heatmap_path,
        'biomarker_path': biomarker_path,
        'report_path': report_path
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Voice Cue Depression Detection Pipeline")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--transcript", type=str, default=None, help="Path to DAIC-WOZ transcript file (.csv/.tsv)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for clean audio & diagnostic visual artifacts")

    args = parser.parse_args()
    run_pipeline(args.audio, args.transcript, args.output_dir)


if __name__ == "__main__":
    main()
