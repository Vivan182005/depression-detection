"""
Module: audio_cleaner.py
Description: Clean audio signal preprocessor applying peak loudness boosting (-1.0 dB FS) 
for crystal-clear, audible speech without distortion or fast-forwarding artifacts.
"""

import numpy as np


def normalize_loudness(audio, target_peak_db=-1.0):
    """
    Normalizes audio signal to maximum peak level (-1.0 dB FS),
    ensuring maximum clarity and audible volume without distortion or clipping.
    """
    max_val = np.max(np.abs(audio))
    if max_val < 1e-6:
        return audio

    target_amplitude = 10 ** (target_peak_db / 20.0) # ~0.891 for -1.0 dB FS
    gain = target_amplitude / max_val

    normalized = audio * gain
    return np.clip(normalized, -1.0, 1.0)


def preprocess_audio_signal(audio, sr=16000, target_peak_db=-1.0):
    """
    Pristine audio signal preprocessing pipeline with loudness boost.
    """
    # Apply Peak Loudness Boost (-1.0 dB FS)
    boosted_audio = normalize_loudness(audio, target_peak_db=target_peak_db)
    return boosted_audio
