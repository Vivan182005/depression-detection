"""
Unit test suite: experiments/research_v2/test_phase2_manifest.py
Description: Validates Phase 2 participant-level manifest integrity, sealed cohort quarantine,
PID uniqueness, audio and label availability, and 700-series provenance.
"""

import os
import json
import unittest
import pandas as pd
import soundfile as sf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MANIFEST_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'expanded_dataset_manifest.csv')
SEALED_PATH = os.path.join(_REPO_ROOT, 'sealed_ids.json')
DEV_STORE_INDEX = os.path.join(_REPO_ROOT, 'app', 'feature_store', 'master_dataset_index.csv')

class TestPhase2Manifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MANIFEST_PATH):
            raise FileNotFoundError(f'Manifest not found at {MANIFEST_PATH}')
        cls.df = pd.read_csv(MANIFEST_PATH)
        with open(SEALED_PATH) as f:
            cls.sealed_pids = set(json.load(f)['sealed_pids'])
        cls.dev_store_df = pd.read_csv(DEV_STORE_INDEX)
        cls.dev_store_pids = set(cls.dev_store_df['participant_id'])

    def test_total_participants_count(self):
        """Verify manifest accounts for all 275 catalogued participants."""
        self.assertEqual(len(self.df), 275, f'Expected 275 participants, found {len(self.df)}')

    def test_unique_participant_ids(self):
        """Verify all 275 participant IDs are strictly unique (no duplicates)."""
        self.assertEqual(self.df['participant_id'].nunique(), 275)

    def test_sealed_quarantine_boundary(self):
        """Verify exact 33 sealed participants are quarantined with zero dev leakage."""
        sealed_rows = self.df[self.df['cohort_role'] == 'sealed_quarantine']
        self.assertEqual(len(sealed_rows), 33)
        manifest_sealed_pids = set(sealed_rows['participant_id'])
        self.assertEqual(manifest_sealed_pids, self.sealed_pids)
        
        # Verify 0 sealed IDs in dev active or expansion
        dev_rows = self.df[self.df['cohort_role'] == 'dev_active_woz']
        exp_rows = self.df[self.df['cohort_role'] == 'avatar_candidate_expansion']
        
        dev_overlap = set(dev_rows['participant_id']).intersection(self.sealed_pids)
        exp_overlap = set(exp_rows['participant_id']).intersection(self.sealed_pids)
        
        self.assertEqual(len(dev_overlap), 0, f'Sealed leakage into dev_active_woz: {dev_overlap}')
        self.assertEqual(len(exp_overlap), 0, f'Sealed leakage into avatar_candidate_expansion: {exp_overlap}')

    def test_dev_active_woz_count(self):
        """Verify active dev store matches existing 179 participants exactly."""
        dev_rows = self.df[self.df['cohort_role'] == 'dev_active_woz']
        self.assertEqual(len(dev_rows), 179)
        self.assertEqual(set(dev_rows['participant_id']), self.dev_store_pids)

    def test_avatar_candidate_expansion_count(self):
        """Verify eligible non-sealed avatar cohort has exactly 61 participants."""
        exp_rows = self.df[self.df['cohort_role'] == 'avatar_candidate_expansion']
        self.assertEqual(len(exp_rows), 61)
        
        # All 61 must have audio available on disk
        self.assertTrue(exp_rows['audio_available'].all(), 'Some expansion participants lack audio')
        
        # All 61 must have valid non-null PHQ-8 scores
        self.assertTrue(exp_rows['phq8_score'].notna().all(), 'Some expansion participants lack PHQ-8 scores')
        
        # All 61 must have domain == autonomous_avatar
        self.assertTrue((exp_rows['domain'] == 'autonomous_avatar').all())

    def test_700_series_provenance_and_integrity(self):
        """Verify the 13 participants in 700-series are included in expansion cohort."""
        pids_700 = [702, 703, 705, 707, 708, 709, 710, 712, 713, 715, 716, 717, 718]
        rows_700 = self.df[self.df['participant_id'].isin(pids_700)]
        self.assertEqual(len(rows_700), 13)
        self.assertTrue((rows_700['cohort_role'] == 'avatar_candidate_expansion').all())
        self.assertTrue((rows_700['domain'] == 'autonomous_avatar').all())
        self.assertTrue((rows_700['dataset_release'] == 'E-DAIC_AVEC_2019').all())
        
        # Verify physical audio existence on disk
        for idx, r in rows_700.iterrows():
            audio_path = os.path.join(_REPO_ROOT, r['audio_path'])
            self.assertTrue(os.path.exists(audio_path), f'Audio file missing for PID {r["participant_id"]}')
            info = sf.info(audio_path)
            self.assertGreater(info.duration, 60.0, f'Audio duration too short for PID {r["participant_id"]}')

    def test_missing_data_exclusions(self):
        """Verify excluded participants are properly documented."""
        excluded_rows = self.df[self.df['cohort_role'] == 'excluded_missing_data']
        self.assertEqual(len(excluded_rows), 2)
        excluded_pids = set(excluded_rows['participant_id'])
        self.assertEqual(excluded_pids, {458, 602})

    def test_cohort_disjointness(self):
        """Verify all 4 cohorts form a strict mathematical partition of the 275 participants."""
        roles = self.df['cohort_role'].unique()
        self.assertEqual(set(roles), {'dev_active_woz', 'avatar_candidate_expansion', 'sealed_quarantine', 'excluded_missing_data'})
        total_sum = sum(len(self.df[self.df['cohort_role'] == r]) for r in roles)
        self.assertEqual(total_sum, 275)


    def test_roster_audit_integrity(self):
        """Verify participant_roster_audit.csv has exact counts and disjoint cohorts."""
        roster_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'participant_roster_audit.csv')
        self.assertTrue(os.path.exists(roster_path))
        df_roster = pd.read_csv(roster_path)
        self.assertEqual(len(df_roster), 275)
        
        dev_included = df_roster[df_roster['included_in_dev'] == True]
        self.assertEqual(len(dev_included), 240)
        
        # Verify 179 WoZ and 61 Avatar are disjoint
        woz_pids = set(df_roster[df_roster['cohort_role'] == 'dev_active_woz']['participant_id'])
        avatar_pids = set(df_roster[df_roster['cohort_role'] == 'avatar_candidate_expansion']['participant_id'])
        self.assertEqual(len(woz_pids), 179)
        self.assertEqual(len(avatar_pids), 61)
        self.assertEqual(len(woz_pids.intersection(avatar_pids)), 0)
        
        # Verify sealed cohort has 0 dev inclusions
        sealed_included = df_roster[(df_roster['cohort_role'] == 'sealed_quarantine') & (df_roster['included_in_dev'] == True)]
        self.assertEqual(len(sealed_included), 0)

    def test_trimming_manifest_and_versioning(self):
        """Verify trimming audit manifest and separate transcript versioning."""
        audit_csv = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'trimming_audit_manifest.csv')
        self.assertTrue(os.path.exists(audit_csv))
        df_audit = pd.read_csv(audit_csv)
        self.assertIn('status', df_audit.columns)
        self.assertIn('defer_participant_only_extraction', df_audit.columns)
        self.assertIn('interview_start_sec', df_audit.columns)
        
        # Verify raw transcripts are preserved and separate from trimmed
        raw_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev')
        trimmed_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev_trimmed')
        self.assertTrue(os.path.exists(raw_dir))
        self.assertTrue(os.path.exists(trimmed_dir))

if __name__ == '__main__':
    unittest.main()