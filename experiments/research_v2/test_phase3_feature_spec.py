"""
Unit test suite: experiments/research_v2/test_phase3_feature_spec.py
Description: Validates Phase 3 canonical feature extraction specification, archive schemas,
exact 1690-dimensional compatibility with Candidate 2A, and strict sealed isolation.
"""

import os
import unittest
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
AVATAR_CACHE_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_cache_avatar_dev')
CATALOG_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'avatar_feature_store_catalog.csv')
APP_FEATURE_STORE = os.path.join(_REPO_ROOT, 'app', 'feature_store')

class TestPhase3FeatureSpec(unittest.TestCase):
    def test_feature_archive_schema_and_dimensions_all_61(self):
        """Verify all 61 extracted avatar NPZ features match Candidate 2A specification (1690-dim)."""
        df_cat = pd.read_csv(CATALOG_PATH)
        self.assertEqual(len(df_cat), 61, 'Expected exactly 61 avatar candidates in catalog')
        
        expected_keys = [
            'mfcc', 'delta_mfcc', 'loudness', 'temporal_33d',
            'egemaps_88d', 'wav2vec_768d', 'text_bert_768d',
            'text_bert_raw_768d', 'text_bert_trimmed_768d'
        ]
        
        for pid in df_cat['participant_id']:
            npz_path = os.path.join(AVATAR_CACHE_DIR, f'{pid}_features.npz')
            self.assertTrue(os.path.exists(npz_path), f'Missing feature file for PID {pid}')
            
            data = np.load(npz_path)
            for k in expected_keys:
                self.assertIn(k, data.files, f'Key {k} missing in {pid}_features.npz')
                
            t = data['temporal_33d']
            self.assertEqual(t.shape[1], 33)
            self.assertEqual(t.dtype, np.float32)
            
            e = data['egemaps_88d']
            self.assertEqual(e.shape, (88,))
            self.assertEqual(e.dtype, np.float32)
            
            w = data['wav2vec_768d']
            self.assertEqual(w.shape, (768,))
            self.assertEqual(w.dtype, np.float32)
            
            b = data['text_bert_768d']
            self.assertEqual(b.shape, (768,))
            self.assertEqual(b.dtype, np.float32)
            
            # Reconstruct Candidate 2A 1690-dim input vector
            t_mean = np.mean(t, axis=0)
            t_std = np.std(t, axis=0)
            input_vector = np.hstack([e, t_mean, t_std, w, b])
            self.assertEqual(input_vector.shape, (1690,))
            
            self.assertFalse(np.isnan(input_vector).any(), f'NaN values detected in PID {pid}')
            self.assertFalse(np.isinf(input_vector).any(), f'Inf values detected in PID {pid}')

    def test_no_overwrite_existing_feature_store(self):
        """Verify app/feature_store remains untouched and contains zero avatar files."""
        npz_files = [f for f in os.listdir(APP_FEATURE_STORE) if f.endswith('.npz')]
        self.assertGreaterEqual(len(npz_files), 183)
        for f in npz_files:
            prefix = f.split('_')[0]
            if prefix.isdigit():
                pid = int(prefix)
                self.assertFalse(600 <= pid <= 799, f'Avatar participant {pid} leaked into app/feature_store: {f}')

    def test_catalog_structure(self):
        """Verify avatar_feature_store_catalog.csv contains valid records."""
        self.assertTrue(os.path.exists(CATALOG_PATH), f'Catalog missing at {CATALOG_PATH}')
        df_cat = pd.read_csv(CATALOG_PATH)
        self.assertEqual(len(df_cat), 61)
        required_cols = [
            'participant_id', 'feature_path', 'duration_sec',
            'temporal_dim', 'egemaps_dim', 'wav2vec_dim', 'bert_dim',
            'trimming_status', 'defer_participant_only_extraction'
        ]
        for c in required_cols:
            self.assertIn(c, df_cat.columns)

    def test_sealed_guard_rejection_in_extraction(self):
        """Verify sealed PIDs are rejected if passed to avatar extraction."""
        from experiments.research_v2.sealed_guard import assert_no_sealed_ids
        for sealed_pid in [300, 624, 638]:
            with self.assertRaises(RuntimeError):
                assert_no_sealed_ids([sealed_pid], context='test_avatar_extraction')

if __name__ == '__main__':
    unittest.main()
