"""
Script: train_all_new_audio.py
Description: One-Click Master Pipeline for Processing & Training on ALL 187 Files in Clean_audio/NEW_AUDIO.
1. Batch extracts 33d temporal + 88d eGeMAPS + 768d Wav2Vec features for all 187 files.
2. Builds master dataset index with official DAIC-WOZ PHQ-8 ground-truth labels.
3. Trains Dual-Branch BiLSTM + Self-Attention Neural Network for 100 epochs.
4. Evaluates per-session clinical severity predictions.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_dataset_features import run_batch_pipeline
from label_mapper import build_master_dataset_index
from train_pipeline import train_model

NEW_AUDIO_DIR = "/Users/vivannaik/Desktop/CAPSTONE/Clean_audio/NEW_AUDIO"
STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_store")


def run_full_training_on_new_audio(epochs=100):
    print("\n" + "="*80)
    print("      ONE-CLICK MASTER TRAINING PIPELINE FOR ALL NEW CLEAN AUDIO FILES")
    print(f"      Input Audio Directory : {NEW_AUDIO_DIR}")
    print(f"      Feature Store Path    : {STORE_DIR}")
    print(f"      Training Epochs       : {epochs}")
    print("="*80 + "\n")

    # Step 1: Batch Feature Extraction for all 187 files
    print("[STEP 1/3] Extracting Features for all files in Clean_audio/NEW_AUDIO...")
    run_batch_pipeline(audio_dir=NEW_AUDIO_DIR, store_dir=STORE_DIR, force_reextract=False)

    # Step 2: Build Master Dataset Index with Official PHQ-8 Labels
    print("\n[STEP 2/3] Mapping Ground-Truth PHQ-8 Labels & 5 Severity Categories...")
    build_master_dataset_index(store_dir=STORE_DIR)

    # Step 3: Train Model
    print(f"\n[STEP 3/3] Training Dual-Branch BiLSTM Model for {epochs} Epochs...")
    train_model(epochs=epochs, store_dir=STORE_DIR)

    print("\n==========================================================================")
    print("            ALL NEW AUDIO FILES PROCESSED & MODEL TRAINED!")
    print("==========================================================================\n")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_full_training_on_new_audio(epochs=epochs)


if __name__ == "__main__":
    main()
