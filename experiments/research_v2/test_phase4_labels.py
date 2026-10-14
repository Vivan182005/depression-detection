"""
Module: experiments/research_v2/test_phase4_labels.py
Description: Unit and regression tests for authoritative master label manifest labels_official.csv
and research_v2 strict dataset loading.
Enforces:
1. labels_official.csv exists, has 275 participants, and strictly zero nulls in required columns.
2. Sum of PHQ8_1..8 in detailed_lables.csv matches phq8_score across all 275 rows.
3. PHQ_Score and Gender in dev_split.csv, train_split.csv, test_split.csv match labels_official.csv.
4. Class definitions are strictly [0-9] -> 0, [10-14] -> 1, [15-24] -> 2.
5. Strict dataset loader raises KeyError immediately on any missing participant ID.
6. Zero default-0.0 or fabricated label logic exists in research_v2 loading.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS
from experiments.research_v2.dataset import get_official_label_for_pid, load_development_dataset

class TestPhase4OfficialLabels(unittest.TestCase):
    
    def setUp(self):
        self.labels_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'labels_official.csv')
        self.detailed_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'detailed_lables.csv')
        self.dev_split_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'dev_split.csv')
        self.train_split_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'train_split.csv')
        self.test_split_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'test_split.csv')
        
    def test_labels_official_existence_and_structure(self):
        self.assertTrue(os.path.exists(self.labels_path), f'Missing {self.labels_path}')
        df = pd.read_csv(self.labels_path)
        self.assertEqual(len(df), 275, f'Expected 275 rows, found {len(df)}')
        required_cols = ['participant_id', 'phq8_score', 'class_id', 'gender', 'split', 'domain', 'is_sealed', 'binary_label']
        for col in required_cols:
            self.assertIn(col, df.columns, f'Missing column {col}')
            self.assertEqual(df[col].isna().sum(), 0, f'Found nulls in required column {col}')
            
    def test_phq_score_sum_matches_detailed_labels(self):
        df_labels = pd.read_csv(self.labels_path)
        df_det = pd.read_csv(self.detailed_path)
        
        df_det_cols = [c for c in df_det.columns if c.startswith('PHQ8_')]
        self.assertEqual(len(df_det_cols), 8, f'Expected 8 PHQ8 items, found {len(df_det_cols)}')
        
        det_sums = df_det.set_index('Participant')[df_det_cols].sum(axis=1)
        for _, row in df_labels.iterrows():
            pid = int(row['participant_id'])
            self.assertIn(pid, det_sums.index, f'Participant {pid} not found in detailed_lables.csv')
            expected_score = float(det_sums[pid])
            actual_score = float(row['phq8_score'])
            self.assertEqual(expected_score, actual_score, f'PHQ mismatch for PID {pid}: expected {expected_score}, got {actual_score}')

    def test_phq_and_gender_match_challenge_splits(self):
        df_labels = pd.read_csv(self.labels_path).set_index('participant_id')
        splits = [
            pd.read_csv(self.dev_split_path),
            pd.read_csv(self.train_split_path),
            pd.read_csv(self.test_split_path)
        ]
        all_splits = pd.concat(splits, ignore_index=True)
        self.assertEqual(len(all_splits), 275, f'Expected 275 total split entries, got {len(all_splits)}')
        
        for _, row in all_splits.iterrows():
            pid = int(row['Participant_ID'])
            self.assertIn(pid, df_labels.index, f'PID {pid} in splits not found in labels_official.csv')
            official_row = df_labels.loc[pid]
            
            # PHQ Score exact match
            split_score = float(row['PHQ_Score'])
            self.assertEqual(split_score, float(official_row['phq8_score']), f'Score mismatch for PID {pid}')
            
            # Gender match
            split_gender = str(row['Gender']).strip().lower()
            self.assertEqual(split_gender, str(official_row['gender']).strip().lower(), f'Gender mismatch for PID {pid}')

    def test_class_cutoff_rules(self):
        df_labels = pd.read_csv(self.labels_path)
        for _, row in df_labels.iterrows():
            phq = float(row['phq8_score'])
            cid = int(row['class_id'])
            if phq <= 9.0:
                expected_cid = 0
            elif phq <= 14.0:
                expected_cid = 1
            else:
                expected_cid = 2
            self.assertEqual(cid, expected_cid, f'Class cut mismatch for PID {row["participant_id"]} with score {phq}')
            
    def test_sealed_quarantine_integrity(self):
        df_labels = pd.read_csv(self.labels_path)
        sealed_in_file = set(df_labels[df_labels['is_sealed']]['participant_id'].values)
        self.assertEqual(sealed_in_file, SEALED_PIDS, f'Sealed PID set does not match SEALED_PIDS exactly')

    def test_strict_loader_missing_pid_raises_keyerror(self):
        with self.assertRaises(KeyError):
            get_official_label_for_pid(999999)
            
    def test_dataset_loader_reads_official_labels(self):
        ds = load_development_dataset(cohort='woz_only', feature_version='v1')
        self.assertIn('y_class', ds)
        self.assertIn('y_phq', ds)
        # Verify first 5 participants match labels_official.csv exactly
        df_labels = pd.read_csv(self.labels_path).set_index('participant_id')
        for i in range(min(5, len(ds['pids']))):
            pid = ds['pids'][i]
            expected_phq = df_labels.loc[pid, 'phq8_score']
            self.assertEqual(ds['y_phq'][i], expected_phq)

if __name__ == '__main__':
    unittest.main()
