"""
Module: experiments/research_v2/test_feature_sanity.py
Description: Feature Quality and Gate Sanity Test Suite for Feature Store v2.
Enforces:
1. Cross-session Wav2Vec2 cosine similarity is well above 0 (> 0.50 for natural speech).
2. Acoustic dims 62-87 are non-zero (proving real OpenSMILE eGeMAPSv02).
3. Zero constant or all-zero dimensions across all 3 modalities.
4. All feature vectors are strictly finite (no NaN, no Inf).
5. Deterministic reproducibility on identical audio.
6. Dataset loader resolves to feature_version='v2' and acoustic_set='robust' with zero sealed leakage.
7. Robust acoustic features strictly drop the 19 amplitude/timing features (exact 135d).
"""

import os
import sys
import glob
import unittest
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from experiments.research_v2.dataset import load_development_dataset
from experiments.research_v2.extractors_v2 import (
    AcousticExtractorV2,
    get_robust_acoustic_135d,
    DROPPED_ACOUSTIC_INDICES,
    ROBUST_ACOUSTIC_MASK
)

FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')

class TestFeatureSanityV2(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.v2_files = sorted(glob.glob(os.path.join(FEATURE_STORE_V2_DIR, '*_v2.npz')))
        if len(cls.v2_files) < 3:
            raise unittest.SkipTest(f'Need at least 3 extracted v2 sessions, found {len(cls.v2_files)}')
            
    def test_dimensions_and_finiteness(self):
        for f in self.v2_files[:10]:
            d = np.load(f)
            ac = d['acoustic_154d']
            w2v = d['wav2vec_768d']
            bert = d['bert_768d']
            meta = d['meta_7d']
            
            self.assertEqual(ac.shape, (154,), f'Wrong acoustic shape in {f}')
            self.assertEqual(w2v.shape, (768,), f'Wrong w2v shape in {f}')
            self.assertEqual(bert.shape, (768,), f'Wrong bert shape in {f}')
            self.assertEqual(meta.shape, (7,), f'Wrong meta shape in {f}')
            
            self.assertTrue(np.isfinite(ac).all(), f'Non-finite acoustic values in {f}')
            self.assertTrue(np.isfinite(w2v).all(), f'Non-finite w2v values in {f}')
            self.assertTrue(np.isfinite(bert).all(), f'Non-finite bert values in {f}')
            self.assertTrue(np.isfinite(meta).all(), f'Non-finite meta values in {f}')

    def test_robust_acoustic_135d_structure(self):
        self.assertEqual(len(DROPPED_ACOUSTIC_INDICES), 19, 'Must drop exactly 19 features')
        self.assertEqual(int(ROBUST_ACOUSTIC_MASK.sum()), 135, 'Robust mask must retain 135 features')
        
        for f in self.v2_files[:10]:
            d = np.load(f)
            ac_154 = d['acoustic_154d']
            rob_135 = get_robust_acoustic_135d(ac_154)
            self.assertEqual(rob_135.shape, (135,))
            self.assertTrue(np.isfinite(rob_135).all())

    def test_egemaps_dims_62_87_nonzero(self):
        for f in self.v2_files[:10]:
            d = np.load(f)
            ac = d['acoustic_154d']
            dims_62_87 = ac[62:88]
            self.assertFalse(np.all(dims_62_87 == 0.0), f'Dims 62-87 are all zero in {f}! OpenSMILE failed.')
            self.assertGreater(np.max(np.abs(dims_62_87)), 1e-4)

    def test_wav2vec_cross_session_cosine_well_above_zero(self):
        w2vs = [np.load(f)['wav2vec_768d'] for f in self.v2_files[:5]]
        cos_sims = []
        for i in range(len(w2vs)):
            for j in range(i + 1, len(w2vs)):
                v1, v2 = w2vs[i], w2vs[j]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                cos_sims.append(cos)
                
        mean_cos = float(np.mean(cos_sims))
        self.assertGreater(mean_cos, 0.50, f'Expected cross-session Wav2Vec2 cosine > 0.50, got {mean_cos:.4f}')

    def test_reproducible_acoustic_reextraction(self):
        ext = AcousticExtractorV2(sr=16000)
        t = np.linspace(0, 1, 16000)
        sig = 0.5 * np.sin(2 * np.pi * 400 * t).astype(np.float32)
        feat1 = ext.extract(sig)
        feat2 = ext.extract(sig)
        np.testing.assert_allclose(feat1, feat2, rtol=1e-5, atol=1e-5)
        
        rob1 = ext.extract_robust(sig)
        rob2 = ext.extract_robust(sig)
        np.testing.assert_allclose(rob1, rob2, rtol=1e-5, atol=1e-5)
        self.assertEqual(rob1.shape, (135,))

    def test_dataset_loader_defaults_to_v2_and_robust(self):
        data = load_development_dataset(cohort='combined')
        assert_no_sealed_ids(data['pids'], context='test_dataset_loader_defaults_to_v2_and_robust')
        self.assertEqual(data['X_ac'].shape[1], 135, 'Default acoustic set must be robust 135d')
        self.assertEqual(data['X_w2v'].shape[1], 768)
        self.assertEqual(data['X_bert'].shape[1], 768)
        self.assertEqual(data['X_tri'].shape[1], 135 + 768 + 768)
        self.assertEqual(len(data['pids']), 239)
        
        # Test full acoustic set
        data_full = load_development_dataset(cohort='combined', acoustic_set='full')
        self.assertEqual(data_full['X_ac'].shape[1], 154)

    def test_no_constant_or_zero_padded_dimensions(self):
        data = load_development_dataset(cohort='combined')
        for mod_name, X in [('Acoustic_Robust', data['X_ac']), ('Wav2Vec2', data['X_w2v']), ('BERT', data['X_bert'])]:
            # Assert all finite
            self.assertTrue(np.all(np.isfinite(X)), f'Found non-finite values in {mod_name}')
            # Assert zero all-zero (zero-padded) dimensions
            all_zero_dims = np.where(np.all(X == 0.0, axis=0))[0]
            self.assertEqual(len(all_zero_dims), 0, f'Found zero-padded dimensions in {mod_name}: {all_zero_dims}')
            # Assert zero constant dimensions across sessions (peak-to-peak spread > 0)
            ptp = np.ptp(X, axis=0)
            constant_dims = np.where(ptp == 0.0)[0]
            self.assertEqual(len(constant_dims), 0, f'Found strictly constant dimensions in {mod_name}: {constant_dims}')

if __name__ == '__main__':
    unittest.main()
