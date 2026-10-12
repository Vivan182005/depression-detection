"""
Script: run_preprocessing.py
Description: Pipeline preprocessor matching exact feature shapes:
- MFCC shape: (N_frames, 16)
- Delta MFCC shape: (N_frames, 16)
- Loudness shape: (N_frames, 1)
- Loudness stats shape: (N_segments, 6)
- Temporal feature shape: (N_frames, 33)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf
import librosa


def extract_features(audio_path):
    """
    Extracts frame-by-frame 33-dimensional temporal acoustic feature matrix:
    - 16 MFCCs
    - 16 Delta MFCCs
    - 1 Loudness (RMS)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=16000)

    # 1. Extract 16 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=256, n_fft=512).T # (N_frames, 16)

    # 2. Extract 16 Delta MFCCs
    delta_mfcc = librosa.feature.delta(mfcc, axis=0) # (N_frames, 16)

    # 3. Extract Frame Loudness (RMS Energy)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256).T # (N_frames, 1)

    # Align frame counts
    min_len = min(len(mfcc), len(delta_mfcc), len(rms))
    mfcc = mfcc[:min_len]
    delta_mfcc = delta_mfcc[:min_len]
    rms = rms[:min_len]

    # Combine into 33-dim temporal feature matrix
    temporal_features = np.hstack([mfcc, delta_mfcc, rms]) # (N_frames, 33)

    # 4. Compute Loudness Stats per segment (e.g., 200 frames per segment)
    seg_size = 200
    num_segs = max(1, min_len // seg_size)
    loudness_stats = []
    
    for i in range(num_segs):
        seg_rms = rms[i*seg_size : (i+1)*seg_size]
        stats = [
            np.mean(seg_rms),
            np.std(seg_rms),
            np.min(seg_rms),
            np.max(seg_rms),
            np.median(seg_rms),
            scipy_skew(seg_rms)
        ]
        loudness_stats.append(stats)

    loudness_stats = np.array(loudness_stats)

    print(f"MFCC shape: {mfcc.shape}")
    print(f"Delta MFCC shape: {delta_mfcc.shape}")
    print(f"Loudness shape: {rms.shape}")
    print(f"Loudness stats shape: {loudness_stats.shape}")
    print(f"Temporal feature shape: {temporal_features.shape}")
    print("Feature extraction + temporal preparation completed successfully.")

    return {
        'mfcc': mfcc,
        'delta_mfcc': delta_mfcc,
        'loudness': rms,
        'loudness_stats': loudness_stats,
        'temporal_features': temporal_features,
        'audio': y,
        'sr': sr
    }


def scipy_skew(arr):
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean((arr - np.mean(arr))**3) / (std**3))


def main():
    audio_path = "/Users/vivannaik/Desktop/CAPSTONE/Audio/loud_audio300.wav"
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        # Fallback to output folder clean audio
        audio_path = "/Users/vivannaik/Desktop/CAPSTONE/app/output/loud_audio300_PARTICIPANT_ONLY.wav"

    extract_features(audio_path)


if __name__ == "__main__":
    main()
