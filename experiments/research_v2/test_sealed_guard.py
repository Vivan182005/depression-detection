"""
Tests for sealed_guard.py verifying strict quarantine of sealed cohort (N=33).
Supports both standard library unittest and pytest.
"""

import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiments.research_v2.sealed_guard import (
    SEALED_PIDS,
    extract_pid,
    assert_no_sealed_ids,
    guard_dataframe
)

class TestSealedGuard(unittest.TestCase):
    def test_sealed_pids_count(self):
        self.assertEqual(len(SEALED_PIDS), 33)
        self.assertIn(624, SEALED_PIDS)
        self.assertIn(300, SEALED_PIDS)
        self.assertNotIn(301, SEALED_PIDS)

    def test_extract_pid(self):
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

if __name__ == '__main__':
    unittest.main()
