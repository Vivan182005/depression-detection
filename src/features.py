import numpy as np
import librosa


def extract_pitch(y, sr):
    """
    Extracts fundamental frequency (F0 / Pitch) using librosa.piptrack.
    Returns the mean F0 over the frames, ignoring zeros.
    """
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=512, hop_length=128)
    
    # Select the pitch corresponding to the maximum magnitude at each time step
    f0_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Skip zero values
            f0_values.append(pitch)
            
    if len(f0_values) == 0:
        return 0.0
    return float(np.mean(f0_values))


def extract_rms_energy(y):
    """Computes the root-mean-square (RMS) energy of the signal."""
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)
    return float(np.mean(rms))


def extract_mfcc(y, sr, n_mfcc=13):
    """Extracts Mel-Frequency Cepstral Coefficients (MFCCs)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=128)
    return mfcc


def extract_features_for_segment(y, sr, method="method2"):
    """
    Extracts a dense feature vector for a single audio segment (micro-frame).
    
    If method == "method1" (Baseline: 28-dimensional):
      - 13 MFCC means
      - 13 MFCC standard deviations
      - 1 RMS Energy
      - 1 Pitch (F0)
      Total = 28 features.
      
    If method == "method2" (Proposed: 33-dimensional):
      - 16 MFCC means
      - 16 Delta-MFCC means
      - 1 Loudness (RMS Energy)
      Total = 33 features.
    """
    if method == "method1":
        mfccs = extract_mfcc(y, sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        rms = extract_rms_energy(y)
        pitch = extract_pitch(y, sr)
        
        feature_vector = np.concatenate([mfcc_means, mfcc_stds, [rms, pitch]])
        return feature_vector

    else:  # method2 (Proposed 33 features)
        mfccs = extract_mfcc(y, sr, n_mfcc=16)
        mfcc_means = np.mean(mfccs, axis=1)
        
        # Calculate Delta MFCCs (first-order derivative)
        deltas = librosa.feature.delta(mfccs)
        delta_means = np.mean(deltas, axis=1)
        
        loudness = extract_rms_energy(y)
        
        feature_vector = np.concatenate([mfcc_means, delta_means, [loudness]])
        return feature_vector


def extract_pipeline_features(frames, sr, method="method2"):
    """
    Extracts features for all micro-frames in a session.
    Returns:
        NumPy array of shape (num_frames, feature_dimension).
    """
    feature_matrix = []
    for frame in frames:
        fv = extract_features_for_segment(frame, sr, method=method)
        feature_matrix.append(fv)
        
    return np.array(feature_matrix)
