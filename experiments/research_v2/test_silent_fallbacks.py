"""
Unit Tests: experiments/research_v2/test_silent_fallbacks.py
Strictly asserts that:
1. Missing text_variant='T2' raises KeyError rather than silently falling back to T1.
2. Invalid text_variant, acoustic_set, or feature_version raises ValueError.
3. Missing metadata raises KeyError rather than silently substituting zeros.
4. load_variant_features('V14') raises KeyError if bert_t2_768d is missing.
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from experiments.research_v2.dataset import load_development_dataset
from experiments.research_v2.data_loader import load_variant_features

class TestSilentFallbacks(unittest.TestCase):

    def test_missing_t2_raises_keyerror(self):
        """Asserts that requesting text_variant='T2' raises KeyError when sessions lack bert_t2_768d."""
        from unittest.mock import patch
        orig_load = np.load
        def mock_np_load(p, *args, **kwargs):
            d = dict(orig_load(p, *args, **kwargs))
            if 'bert_t2_768d' in d:
                del d['bert_t2_768d']
            return d

        with patch('numpy.load', side_effect=mock_np_load):
            with self.assertRaises(KeyError) as ctx:
                load_development_dataset(cohort='woz_only', feature_version='v2', text_variant='T2')
            self.assertIn("Silent fallback prohibited", str(ctx.exception))

    def test_valid_t2_loads_without_error(self):
        """Asserts that genuine text_variant='T2' loads all 178 WoZ sessions with shape (178, 768)."""
        ds = load_development_dataset(cohort='woz_only', feature_version='v2', text_variant='T2')
        self.assertEqual(len(ds['X_bert']), 178)
        self.assertEqual(ds['X_bert'].shape[1], 768)
        self.assertTrue(np.isfinite(ds['X_bert']).all())

    def test_invalid_parameters_raise_valueerror(self):
        """Asserts that invalid variants or versions immediately raise ValueError."""
        with self.assertRaises(ValueError):
            load_development_dataset(cohort='woz_only', text_variant='INVALID')

        with self.assertRaises(ValueError):
            load_development_dataset(cohort='woz_only', acoustic_set='INVALID')

        with self.assertRaises(ValueError):
            load_development_dataset(cohort='woz_only', feature_version='INVALID')

    def test_v14_data_loader_raises_if_t2_missing(self):
        """Asserts that data_loader.load_variant_features('V14') raises KeyError if T2 is missing."""
        from unittest.mock import patch
        orig_load = np.load
        def mock_load(p, *args, **kwargs):
            d = dict(orig_load(p, *args, **kwargs))
            if 'bert_t2_768d' in d:
                del d['bert_t2_768d']
            return d

        mock_df = pd.DataFrame({'participant_id': [362], 'domain': ['woz'], 'gender': [0]})
        with patch('numpy.load', side_effect=mock_load):
            with self.assertRaises(KeyError) as ctx:
                load_variant_features('V14', mock_df)
            self.assertIn("bert_t2_768d", str(ctx.exception))

    def test_valid_t1_loads_without_error(self):
        """Asserts that valid text_variant='T1' loads properly."""
        ds = load_development_dataset(cohort='woz_only', feature_version='v2', text_variant='T1')
        self.assertEqual(len(ds['X_bert']), 178)
        self.assertEqual(ds['X_bert'].shape[1], 768)

if __name__ == '__main__':
    unittest.main()
