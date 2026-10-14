"""
Test Suite: Label-Blind Extraction Integrity & Sealed Cohort Quarantine Guard
Verifies that:
1. Feature extraction code explicitly asserts that PHQ totals, PHQ items, and severity classes
   are completely unavailable to the extractor.
2. Deliberately supplying any label-bearing object (e.g. dict, Series, or DataFrame with PHQ keys)
   raises an immediate, fatal AssertionError.
3. Supplying any Sealed Cohort PID (from sealed_ids.json / labels_official.csv where is_sealed==True)
   raises an immediate, fatal AssertionError.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

FORBIDDEN_LABEL_KEYS = [
    'phq8_score', 'phq_score', 'phq8_binary', 'binary_label', 'class_id',
    'phq8_1', 'phq8_2', 'phq8_3', 'phq8_4', 'phq8_5', 'phq8_6', 'phq8_7', 'phq8_8',
    'phq_total', 'severity', 'label', 'target'
]

def assert_label_blind(metadata_dict_or_df):
    """
    Executable assertion guarding the feature extraction pipeline against label leakage.
    Must be called at the entry point of every extractor function.
    """
    if metadata_dict_or_df is None:
        return
        
    if isinstance(metadata_dict_or_df, dict):
        keys = [str(k).lower() for k in metadata_dict_or_df.keys()]
    elif isinstance(metadata_dict_or_df, (pd.DataFrame, pd.Series)):
        keys = [str(k).lower() for k in metadata_dict_or_df.index if isinstance(metadata_dict_or_df, pd.Series)] + \
               [str(c).lower() for c in metadata_dict_or_df.columns if isinstance(metadata_dict_or_df, pd.DataFrame)]
    else:
        return
        
    for k in keys:
        for forbidden in FORBIDDEN_LABEL_KEYS:
            if forbidden == k or forbidden in k:
                raise AssertionError(
                    f"CRITICAL LABEL LEAKAGE VIOLATION: Forbidden label field '{k}' detected in feature extraction pipeline! "
                    f"Feature extraction must be 100% label-blind."
                )

def assert_sealed_quarantine(participant_id, sealed_ids_path=None):
    """
    Executable assertion ensuring sealed cohort participants are NEVER extracted,
    read, segmented, or processed in development workflows.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if sealed_ids_path is None:
        sealed_ids_path = os.path.join(base_dir, 'sealed_ids.json')
        
    if os.path.exists(sealed_ids_path):
        with open(sealed_ids_path) as f:
            sealed_pids = set(json.load(f))
    else:
        # Fallback to labels_official.csv is_sealed column
        labels_path = os.path.join(base_dir, 'labels_official.csv')
        labels = pd.read_csv(labels_path)
        sealed_pids = set(labels[labels['is_sealed'] == True]['participant_id'])
        
    pid = int(participant_id)
    if pid in sealed_pids:
        raise AssertionError(
            f"CRITICAL SEALED COHORT VIOLATION: Attempted to access sealed participant {pid}! "
            f"Sealed cohort (N=33) is strictly quarantined and must remain untouched in Phase 6A."
        )

# --- Unit Tests ---

def test_label_blind_clean_metadata_passes():
    """Verify that pure audio metadata (PID, domain, duration, sample rate) passes without error."""
    clean_meta = {'participant_id': 301, 'domain': 'woz', 'duration': 495.8, 'sample_rate': 16000}
    # Should not raise
    assert_label_blind(clean_meta)

def test_label_blind_phq_total_rejected():
    """Verify that metadata containing phq8_score raises AssertionError."""
    leaked_meta = {'participant_id': 301, 'phq8_score': 12.0}
    with pytest.raises(AssertionError, match="CRITICAL LABEL LEAKAGE VIOLATION"):
        assert_label_blind(leaked_meta)

def test_label_blind_class_id_rejected():
    """Verify that metadata containing class_id raises AssertionError."""
    leaked_meta = {'participant_id': 301, 'class_id': 1}
    with pytest.raises(AssertionError, match="CRITICAL LABEL LEAKAGE VIOLATION"):
        assert_label_blind(leaked_meta)

def test_label_blind_phq_items_rejected():
    """Verify that metadata containing individual PHQ item responses raises AssertionError."""
    for item_idx in range(1, 9):
        leaked_meta = {'participant_id': 301, f'phq8_{item_idx}': 2}
        with pytest.raises(AssertionError, match="CRITICAL LABEL LEAKAGE VIOLATION"):
            assert_label_blind(leaked_meta)

def test_label_blind_dataframe_rejected():
    """Verify that supplying a DataFrame with label columns raises AssertionError."""
    df = pd.DataFrame([{'participant_id': 301, 'domain': 'woz', 'phq8_score': 5.0}])
    with pytest.raises(AssertionError, match="CRITICAL LABEL LEAKAGE VIOLATION"):
        assert_label_blind(df)

def test_sealed_cohort_guard_active():
    """Verify that sealed participant IDs (e.g. 300, 615, 619, 624) are blocked."""
    # WoZ sealed (9 total: 300, 303, 304, 321, 338, 397, 408, 409, 414)
    with pytest.raises(AssertionError, match="CRITICAL SEALED COHORT VIOLATION"):
        assert_sealed_quarantine(300)
    with pytest.raises(AssertionError, match="CRITICAL SEALED COHORT VIOLATION"):
        assert_sealed_quarantine(303)
        
    # Avatar sealed (24 total: 600, 601, 603, 604, 605, 607, 615, 619, 623, 624, ...)
    with pytest.raises(AssertionError, match="CRITICAL SEALED COHORT VIOLATION"):
        assert_sealed_quarantine(615)
        
    with pytest.raises(AssertionError, match="CRITICAL SEALED COHORT VIOLATION"):
        assert_sealed_quarantine(624)

def test_eligible_dev_participant_allowed():
    """Verify that eligible development participants (e.g. 301, 606, 608) are permitted."""
    assert_sealed_quarantine(301)
    assert_sealed_quarantine(606)
    assert_sealed_quarantine(608)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
