"""
Script: extract_dataset_features.py
Description: Batch Feature Extraction & Feature Store Pipeline for 1,000+ Audio Files.
Processes pre-cleaned interviewee WAV files and saves compressed feature matrices (.npz) 
and dataset catalog index (.csv).
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.egemaps_extractor import EgemapsExtractor
from src.features.wav2vec_extractor import Wav2VecExtractor
from src.features.feature_store import FeatureStore
from run_preprocessing import extract_features as extract_33d_features

import soundfile as sf


def clean_session_id(audio_path):
    base = os.path.basename(audio_path)
    base = base.replace("_PARTICIPANT_ONLY.wav", "")
    base = base.replace("_AUDIO_interviewee.wav", "")
    base = base.replace("_AUDIO.wav", "")
    base = base.replace(".wav", "")
    return base


def process_audio_file(audio_path, feature_store, egemaps_ext, wav2vec_ext, force_reextract=False):
    session_id = clean_session_id(audio_path)
    
    if feature_store.is_extracted(session_id) and not force_reextract:
        print(f"[SKIP] Session {session_id} already extracted in feature store.")
        return feature_store.get_feature_path(session_id)

    print(f"\nProcessing Session: {session_id}")
    print(f"  -> Audio File: {audio_path}")

    # 1. Extract 33-dim temporal features (16 MFCC + 16 Delta MFCC + 1 Loudness)
    data_33d = extract_33d_features(audio_path)
    y = data_33d['audio']
    sr = data_33d['sr']
    duration_sec = len(y) / sr

    # 2. Extract 88-dim eGeMAPS Bio-markers & Pause Metrics
    egemaps_88d, pause_metrics = egemaps_ext.extract(y)

    # 3. Extract 768-dim Wav2Vec2 Deep Embeddings
    wav2vec_768d = wav2vec_ext.extract_embedding(y, sr=sr)

    # Assemble Feature Store Dictionary
    feature_dict = {
        'mfcc': data_33d['mfcc'],
        'delta_mfcc': data_33d['delta_mfcc'],
        'loudness': data_33d['loudness'],
        'temporal_33d': data_33d['temporal_features'],
        'egemaps_88d': egemaps_88d,
        'wav2vec_768d': wav2vec_768d
    }

    metadata = {
        'duration_sec': duration_sec,
        'pause_ratio': pause_metrics.get('pause_ratio', 0.0),
        'mean_pause_duration': pause_metrics.get('mean_pause_duration', 0.0),
        'pause_count': pause_metrics.get('pause_count', 0)
    }

    saved_path = feature_store.save_features(session_id, feature_dict, metadata)
    print(f"  [SUCCESS] Stored features for {session_id} -> {saved_path}")
    return saved_path


def run_batch_pipeline(audio_dir, store_dir="./feature_store", force_reextract=False):
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    feature_store = FeatureStore(store_dir=store_dir)
    egemaps_ext = EgemapsExtractor(sr=16000)
    wav2vec_ext = Wav2VecExtractor(device="cpu")

    audio_files = [
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
        if f.endswith('.wav')
    ]
    audio_files.sort()

    print("\n==========================================================================")
    print(f"       BATCH FEATURE STORE PIPELINE ({len(audio_files)} FILES)")
    print("==========================================================================")

    for i, audio_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}]", end=" ")
        try:
            process_audio_file(audio_path, feature_store, egemaps_ext, wav2vec_ext, force_reextract=force_reextract)
        except Exception as e:
            print(f"[ERROR] Failed to process {audio_path}: {e}")

    catalog = feature_store.load_catalog()
    print("\n==========================================================================")
    print(f"BATCH EXTRACTION COMPLETE! Catalog Summary:")
    print(f"Total Extracted Sessions : {len(catalog)}")
    print(f"Feature Store Directory  : {feature_store.store_dir}")
    print("==========================================================================\n")
    return catalog


def main():
    parser = argparse.ArgumentParser(description="Batch Feature Extraction Pipeline for 1,000+ Audio Files")
    parser.add_argument("--audio_dir", type=str, default="../Clean_audio/NEW_AUDIO", help="Directory containing input audio WAV files")
    parser.add_argument("--store_dir", type=str, default="./feature_store", help="Output directory for feature store (.npz archives)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of already stored feature files")

    args = parser.parse_args()
    run_batch_pipeline(args.audio_dir, args.store_dir, force_reextract=args.force)


if __name__ == "__main__":
    main()
