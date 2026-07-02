import numpy as np
import pytest
from src.preprocessing import apply_bandpass_filter, normalize_volume, segment_audio
from src.features import extract_features_for_segment


def test_bandpass_filter():
    sr = 16000
    y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)  # 1s 440Hz sine wave
    y_filtered = apply_bandpass_filter(y, sr, low_cut=50, high_cut=8000)
    assert len(y_filtered) == len(y)
    assert isinstance(y_filtered, np.ndarray)


def test_normalize_volume():
    y = np.random.randn(16000) * 0.1
    y_norm = normalize_volume(y, target_db=-20)
    assert len(y_norm) == len(y)
    
    # Test zero input does not crash
    y_zero = np.zeros(16000)
    y_zero_norm = normalize_volume(y_zero)
    assert np.all(y_zero_norm == 0)


def test_segment_audio():
    sr = 16000
    y = np.random.randn(sr * 5)  # 5 seconds
    frames = segment_audio(y, sr, frame_len_sec=2.0, hop_len_sec=1.0)
    # Expected: 2.0s frame, 1.0s hop
    # Frame 1: 0 - 2
    # Frame 2: 1 - 3
    # Frame 3: 2 - 4
    # Frame 4: 3 - 5
    assert len(frames) == 4
    for frame in frames:
        assert len(frame) == int(2.0 * sr)


def test_feature_extraction():
    sr = 16000
    y = np.random.randn(int(2.0 * sr))  # 2 seconds frame
    
    # Test method1 features
    fv_m1 = extract_features_for_segment(y, sr, method="method1")
    assert len(fv_m1) == 28
    
    # Test method2 features
    fv_m2 = extract_features_for_segment(y, sr, method="method2")
    assert len(fv_m2) == 33
