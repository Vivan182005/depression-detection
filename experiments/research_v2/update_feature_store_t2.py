"""
Incrementally extracts BERT T2 embeddings for available WoZ Whisper transcripts.
Updates feature_store_v2 .npz files.
"""
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS, assert_no_sealed_ids
from experiments.research_v2.extractors_v2 import BertExtractorV2, get_robust_acoustic_135d

FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
WOZ_WHISPER_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'woz_whisper')

def main():
    csv_files = sorted(glob.glob(os.path.join(WOZ_WHISPER_DIR, "*_TRANSCRIPT_WHISPER.csv")))
    print(f"Checking {len(csv_files)} available WoZ Whisper transcripts...")
    
    device = 'cpu'
    bert_extractor = BertExtractorV2(device=device)
    
    updated = 0
    t0 = time.time()
    for f in csv_files:
        fn = os.path.basename(f)
        pid = int(fn.split('_')[0])
        npz_path = os.path.join(FEATURE_STORE_V2_DIR, f"{pid}_v2.npz")
        if not os.path.exists(npz_path):
            continue
            
        data = dict(np.load(npz_path))
        if 'bert_t2_768d' in data and data['bert_t2_768d'] is not None:
            # Already updated
            continue
            
        df = pd.read_csv(f)
        full_text = ' '.join(df['text'].dropna().astype(str).tolist())
        if not full_text.strip():
            continue
            
        bert_t2 = bert_extractor.extract_from_text(full_text)
        data['bert_t1_768d'] = data['bert_768d']
        data['bert_t2_768d'] = bert_t2
        if 'acoustic_robust_135d' not in data:
            data['acoustic_robust_135d'] = get_robust_acoustic_135d(data['acoustic_154d'])
            
        np.savez_compressed(npz_path, **data)
        updated += 1
        
    print(f"Updated {updated} npz files with BERT T2 in {time.time() - t0:.2f}s!")

if __name__ == '__main__':
    main()
