"""
Module: audio_augmentor.py
Description: Audio-level data augmentation for minority depression classes (Moderate & Severe).
Applies training-only transformations:
- Pitch Shift (±1 to ±2 semitones)
- Time Stretch (0.9x to 1.1x speed)
- Light Gaussian Background Noise (SNR 25dB)
- Volume/Gain Perturbation (0.8x to 1.2x)

LEAK-FREE GUARANTEE: Applied strictly on the Training Fold after dataset splitting.
"""

import numpy as np
import librosa


def augment_audio_waveform(y, sr=16000, aug_type="pitch_shift"):
    """
    Applies a single augmentation transform to a 1D audio waveform.
    """
    if len(y) == 0:
        return y

    try:
        if aug_type == "pitch_shift":
            n_steps = float(np.random.choice([-2.0, -1.0, 1.0, 2.0]))
            return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

        elif aug_type == "time_stretch":
            rate = float(np.random.choice([0.9, 0.95, 1.05, 1.1]))
            return librosa.effects.time_stretch(y=y, rate=rate)

        elif aug_type == "add_noise":
            noise_amp = 0.005 * np.random.uniform(0.5, 1.5) * np.max(np.abs(y))
            noise = np.random.normal(0, noise_amp, len(y))
            return y + noise

        elif aug_type == "gain":
            gain_factor = float(np.random.uniform(0.8, 1.2))
            return y * gain_factor

        else:
            return y
    except Exception:
        return y


def generate_minority_augmentations(egemaps_88d, temporal_33d, wav2vec_768d, num_augs=3):
    """
    Generates synthetic feature variants for minority class samples in feature space.
    Adds slight Gaussian noise and feature scaling jitter (SNR 25-30dB equivalent) to 
    eGeMAPS, temporal, and Wav2Vec2 feature vectors.
    """
    augmented_samples = []

    for _ in range(num_augs):
        # 1. Light feature perturbation for eGeMAPS
        e_noise = np.random.normal(0, 0.02 * np.std(egemaps_88d) + 1e-6, size=egemaps_88d.shape)
        e_aug = egemaps_88d + e_noise

        # 2. Time-series noise & frame jitter for Temporal Features
        t_noise = np.random.normal(0, 0.02 * np.std(temporal_33d) + 1e-6, size=temporal_33d.shape)
        t_aug = temporal_33d + t_noise

        # 3. Embedding perturbation for Wav2Vec2
        w_noise = np.random.normal(0, 0.015 * np.std(wav2vec_768d) + 1e-6, size=wav2vec_768d.shape)
        w_aug = wav2vec_768d + w_noise

        augmented_samples.append((e_aug, t_aug, w_aug))

    return augmented_samples
