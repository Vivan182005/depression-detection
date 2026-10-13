"""
Unit tests for Phase 1 harness: fold integrity, metrics, and artifact verification.
"""

import unittest
import os
import json
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_EXP_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')

class TestPhase1Harness(unittest.TestCase):
    def test_selection_folds_integrity(self):
        folds_path = os.path.join(_EXP_DIR, 'folds.json')
        self.assertTrue(os.path.exists(folds_path))
        with open(folds_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 5)
        for seed_str, folds in data.items():
            val_indices = []
            for fold in folds:
                tr_p = set(fold['train_pids'])
                val_p = set(fold['val_pids'])
                self.assertEqual(len(tr_p.intersection(val_p)), 0, f'Leakage in seed {seed_str} fold {fold["fold"]}')
                val_indices.extend(fold['val_indices'])
            self.assertEqual(len(val_indices), 183)
            self.assertEqual(len(set(val_indices)), 183)

    def test_confirmation_folds_integrity(self):
        folds_path = os.path.join(_EXP_DIR, 'folds_confirmation.json')
        self.assertTrue(os.path.exists(folds_path))
        with open(folds_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 5)
        for seed_str, folds in data.items():
            self.assertIn(int(seed_str), [100, 101, 102, 103, 104])
            for fold in folds:
                tr_p = set(fold['train_pids'])
                val_p = set(fold['val_pids'])
                self.assertEqual(len(tr_p.intersection(val_p)), 0)

    def test_experiments_log_completeness(self):
        log_path = os.path.join(_EXP_DIR, 'experiments_log.csv')
        self.assertTrue(os.path.exists(log_path))
        df = pd.read_csv(log_path)
        self.assertEqual(len(df), 17)
        self.assertIn('Candidate_2A_Baseline', df['variant_name'].values)
        self.assertIn('Sanity_Majority_Class', df['variant_name'].values)
        self.assertIn('Sanity_Metadata_Only', df['variant_name'].values)

    def test_oof_artifacts_exist(self):
        oof_dir = os.path.join(_EXP_DIR, 'oof')
        self.assertTrue(os.path.exists(oof_dir))
        files = os.listdir(oof_dir)
        self.assertGreaterEqual(len(files), 102)
        base_pooled = os.path.join(oof_dir, 'Candidate_2A_Baseline_pooled_oof.csv')
        self.assertTrue(os.path.exists(base_pooled))
        df_base = pd.read_csv(base_pooled)
        self.assertEqual(len(df_base), 183)
        self.assertIn('prob_0', df_base.columns)
        self.assertIn('prob_1', df_base.columns)
        self.assertIn('prob_2', df_base.columns)
        self.assertIn('expected_phq', df_base.columns)

if __name__ == '__main__':
    unittest.main()
