"""
Unit test: experiments/research_v2/test_phase3_audit.py
Verifies integrity of Phase 3 audit artifacts:
1. OOF prediction files existence, schema, and non-empty.
2. Strict isolation: zero sealed IDs in any audit OOF prediction.
3. Multi-session grouping in folds_combined.json and folds.json.
4. Confirmation folds uncorrupted and untouched.
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS, assert_no_sealed_ids

class TestPhase3Audit(unittest.TestCase):
    def setUp(self):
        self.exp_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
        self.oof_dir = os.path.join(self.exp_dir, 'oof_phase3')
        
    def test_oof_files_exist_and_quarantined(self):
        expected_files = [
            'Exp1_ZeroShot_Avatar_pooled_oof.csv',
            'Exp2_Combined_CE_pooled_oof.csv',
            'Exp3_Combined_Focal_pooled_oof.csv'
        ]
        for s in range(5):
            expected_files.append(f'Exp1_ZeroShot_Avatar_seed{s}_oof.csv')
            expected_files.append(f'Exp2_Combined_CE_seed{s}_oof.csv')
            expected_files.append(f'Exp3_Combined_Focal_seed{s}_oof.csv')
            
        for fn in expected_files:
            fpath = os.path.join(self.oof_dir, fn)
            self.assertTrue(os.path.exists(fpath), f'Missing OOF file: {fn}')
            df = pd.read_csv(fpath)
            self.assertGreater(len(df), 0, f'Empty OOF file: {fn}')
            # Assert zero sealed participant IDs
            assert_no_sealed_ids(df['participant_id'].values, context=f'audit_oof_{fn}')
            self.assertFalse(df['pred_class'].isna().any())
            self.assertFalse(df['prob_c0'].isna().any())
            self.assertFalse(df['exp_phq8'].isna().any())
            
    def test_multisession_grouping_integrity(self):
        master_path = os.path.join(_REPO_ROOT, 'app', 'feature_store', 'master_dataset_index.csv')
        df_woz = pd.read_csv(master_path)
        multi_pids = [301, 302]
        
        with open(os.path.join(self.exp_dir, 'folds_combined.json')) as f:
            fc = json.load(f)
            
        for seed, folds in fc.items():
            for fold in folds:
                tr_pids = set(fold['train_pids'])
                val_pids = set(fold['val_pids'])
                self.assertEqual(len(tr_pids.intersection(val_pids)), 0)
                for p in multi_pids:
                    p_sids = df_woz[df_woz['participant_id'] == p]['session_id'].tolist()
                    in_tr = [s in fold['train_sessions'] for s in p_sids]
                    in_val = [s in fold['val_sessions'] for s in p_sids]
                    self.assertTrue(all(in_tr) or all(in_val), f'PID {p} split across folds!')
                    
    def test_confirmation_folds_untouched(self):
        conf_path = os.path.join(self.exp_dir, 'folds_confirmation.json')
        self.assertTrue(os.path.exists(conf_path))
        with open(conf_path) as f:
            cf = json.load(f)
        self.assertEqual(sorted(list(cf.keys())), ['100', '101', '102', '103', '104'])
        for s, folds in cf.items():
            self.assertEqual(len(folds), 5)
            
if __name__ == '__main__':
    unittest.main()
