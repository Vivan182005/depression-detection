"""
Script: experiments/research_v2/update_feature_store_t2.py
Step S2: Extracts BERT T2 embeddings from Whisper transcripts for all 239 development participants.
Ensures zero silent fallbacks, updates manifest_v3.csv with text_source_t2, and asserts 100% T2 coverage.
"""
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from test_label_blind_extraction import assert_sealed_quarantine, assert_label_blind
from extractors_v2 import BertExtractorV2, get_robust_acoustic_135d

FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
FEATURE_STORE_V3_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3')
WOZ_WHISPER_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'woz_whisper')
AVATAR_WHISPER_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev')
MANIFEST_V3_PATH = os.path.join(FEATURE_STORE_V3_DIR, 'manifest_v3.csv')

def main():
    print("======================================================================")
    print("Step S2: Update Feature Stores with Complete BERT-T2 Embeddings")
    print("======================================================================")

    manifest = pd.read_csv(MANIFEST_V3_PATH)
    assert len(manifest) == 239, f"Expected 239 dev participants, got {len(manifest)}"
    assert_no_sealed_ids(manifest['participant_id'].values)

    device = 'cpu'
    bert_extractor = BertExtractorV2(device=device)

    t0 = time.time()
    updated = 0
    already_done = 0

    for idx, row in manifest.iterrows():
        pid = int(row['participant_id'])
        domain = row['domain']

        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': str(domain)})

        npz_path = os.path.join(FEATURE_STORE_V2_DIR, f"{pid}_v2.npz")
        assert os.path.exists(npz_path), f"Missing v2 store file: {npz_path}"

        data = dict(np.load(npz_path))

        # Determine transcript path
        if domain == 'woz':
            trans_path = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        else:
            trans_path = os.path.join(AVATAR_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")

        assert os.path.exists(trans_path), f"Missing Whisper transcript for PID {pid}: {trans_path}"

        # Check if already present and valid
        if 'bert_t2_768d' in data and data['bert_t2_768d'] is not None and data['bert_t2_768d'].shape == (768,):
            already_done += 1
            continue

        df_t = pd.read_csv(trans_path)
        full_text = ' '.join(df_t['text'].dropna().astype(str).tolist())
        if not full_text.strip():
            raise ValueError(f"Whisper transcript for PID {pid} is completely empty: {trans_path}")

        bert_t2 = bert_extractor.extract_from_text(full_text)
        assert bert_t2.shape == (768,), f"Unexpected BERT shape: {bert_t2.shape}"
        assert np.isfinite(bert_t2).all(), f"Non-finite values in BERT T2 for PID {pid}"

        data['bert_t1_768d'] = data['bert_768d']
        data['bert_t2_768d'] = bert_t2
        if 'acoustic_robust_135d' not in data:
            data['acoustic_robust_135d'] = get_robust_acoustic_135d(data['acoustic_154d'])

        tmp_npz = f"{npz_path}_tmp.npz"
        np.savez_compressed(tmp_npz, **data)
        os.replace(tmp_npz, npz_path)

        # Mirror to v3 store as well
        npz_v3_path = os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz")
        if os.path.exists(npz_v3_path):
            v3_data = dict(np.load(npz_v3_path, allow_pickle=True))
            v3_data['bert_t2_768d'] = bert_t2
            tmp_v3 = f"{npz_v3_path}_tmp.npz"
            np.savez_compressed(tmp_v3, **v3_data)
            os.replace(tmp_v3, npz_v3_path)

        updated += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(manifest):
            print(f"[{idx+1}/239] PID {pid} ({domain}): Processed T2 | Updated: {updated} | Already done: {already_done}")

    # Final assertion: Verify that ALL 239 participants strictly possess bert_t2_768d
    print("\nVerifying 100% BERT-T2 coverage across active development cohort...")
    for pid in manifest['participant_id'].values:
        npz_p = os.path.join(FEATURE_STORE_V2_DIR, f"{pid}_v2.npz")
        d = np.load(npz_p)
        assert 'bert_t2_768d' in d, f"Fatal: 'bert_t2_768d' missing for PID {pid}"
        assert d['bert_t2_768d'].shape == (768,), f"Shape error for PID {pid}"
        assert np.isfinite(d['bert_t2_768d']).all(), f"Non-finite values for PID {pid}"

    # Update manifest_v3.csv with text_source_t2 column
    manifest['text_source_t2'] = 'whisper_tiny_local'
    manifest.to_csv(MANIFEST_V3_PATH, index=False)
    print(f"Updated {MANIFEST_V3_PATH} with 'text_source_t2' column.")

    print(f"\nSUCCESS: All 239 development participants verified with genuine BERT-T2 in {time.time() - t0:.2f}s!")

if __name__ == '__main__':
    main()
