"""
Tests for sealed_guard.py verifying strict quarantine of sealed cohort (N=33).
Includes entry point testing for dataset loaders and label mappings.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import (
    SEALED_PIDS,
    extract_pid,
    assert_no_sealed_ids,
    guard_dataframe
)
from experiments.research_v2.dataset import load_development_dataset

class TestSealedGuard(unittest.TestCase):
    def test_sealed_pids_count(self):
        """Verify exactly 33 sealed participants are loaded."""
        self.assertEqual(len(SEALED_PIDS), 33)
        self.assertIn(624, SEALED_PIDS)
        self.assertIn(300, SEALED_PIDS)
        self.assertNotIn(301, SEALED_PIDS)

    def test_extract_pid(self):
        """Verify parsing participant IDs from various formats."""
        self.assertEqual(extract_pid(301), 301)
        self.assertEqual(extract_pid('301'), 301)
        self.assertEqual(extract_pid('301_AUDIO'), 301)
        self.assertEqual(extract_pid('624_TRANSCRIPT.csv'), 624)

    def test_assert_no_sealed_ids_clean(self):
        clean_ids = [301, 302, 305, 310, '315_AUDIO']
        assert_no_sealed_ids(clean_ids, context='test_clean', allow_sealed=False)

    def test_assert_no_sealed_ids_violation(self):
        leaked_ids = [301, 624, 305]
        with self.assertRaises(RuntimeError) as ctx:
            assert_no_sealed_ids(leaked_ids, context='test_leaked', allow_sealed=False)
        self.assertIn('CRITICAL DATA LEAKAGE VIOLATION', str(ctx.exception))
        self.assertIn('624', str(ctx.exception))

    def test_assert_no_sealed_ids_allowed_in_phase6(self):
        sealed_ids_list = [624, 300, 303]
        # In Phase 6, allow_sealed=True must not raise
        assert_no_sealed_ids(sealed_ids_list, context='phase6_eval', allow_sealed=True)

    def test_guard_dataframe(self):
        clean_df = pd.DataFrame({'participant_id': [301, 302, 305]})
        guard_dataframe(clean_df, allow_sealed=False)

        dirty_df = pd.DataFrame({'participant_id': [301, 600, 305]})
        with self.assertRaises(RuntimeError):
            guard_dataframe(dirty_df, allow_sealed=False)

    def test_entrypoint_load_development_dataset_zero_sealed(self):
        """Integration test: verify load_development_dataset() output has zero sealed IDs."""
        data = load_development_dataset()
        loaded_pids = set(data['pids'])
        overlap = loaded_pids.intersection(SEALED_PIDS)
        self.assertEqual(len(overlap), 0, f'Sealed participants found in loaded dataset: {overlap}')
        self.assertEqual(len(loaded_pids), 179)
        self.assertEqual(len(data['sids']), 183)

    def test_entrypoint_rejects_injected_sealed_id(self):
        """Verify that injectively passing a sealed PID into assert_no_sealed_ids triggers RuntimeError."""
        for sealed_pid in [300, 600, 624, 638, 414]:
            with self.assertRaises(RuntimeError):
                assert_no_sealed_ids([301, 302, sealed_pid], context='test_injection')

    def test_entrypoint_clean_map_sealed_exclusion(self):
        """Verify that the label sanitization strips sealed PIDs from clean_map."""
        clean_labels_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'detailed_lables.csv')
        clean_df = pd.read_csv(clean_labels_path)
        clean_df_filtered = clean_df[~clean_df['Participant'].isin(SEALED_PIDS)]
        clean_map = dict(zip(clean_df_filtered['Participant'], clean_df_filtered['Depression_severity']))
        
        for sp in SEALED_PIDS:
            self.assertNotIn(sp, clean_map, f'Sealed PID {sp} leaked into sanitized label dictionary!')

if __name__ == '__main__':
    unittest.main()