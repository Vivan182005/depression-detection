"""
Script: experiments/research_v2/extract_avatar_features.py
Description: Canonical Feature Extraction Specification for Avatar Cohort.
Extracts 1690-dim multi-modal representations matching Candidate 2A specification:
- 33-dim temporal acoustic features (MFCC 16 + Delta 16 + Loudness 1)
- 88-dim eGeMAPS prosodic biomarkers
- 768-dim Wav2Vec2 self-supervised speech embeddings (facebook/wav2vec2-base-960h on MPS/CPU with 30s chunk pooling)
- 768-dim BERT contextual text embeddings (bert-base-uncased on CPU)
Caches output into experiments/research_v2/feature_cache_avatar_dev/ with zero dev store overwrite.
Strictly quarantines sealed cohort via sealed_guard.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from app.src.features.egemaps_extractor import EgemapsExtractor
from app.src.features.wav2vec_extractor import Wav2VecExtractor
from app.src.features.bert_extractor import BertExtractor
from app.run_preprocessing import extract_features as extract_33d_features

AVATAR_CACHE_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_cache_avatar_dev')
RAW_TRANSCRIPT_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev')
TRIMMED_TRANSCRIPT_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev_trimmed')
TRIMMING_AUDIT_CSV = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'trimming_audit_manifest.csv')
CATALOG_CSV = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'avatar_feature_store_catalog.csv')

def extract_wav2vec_chunked(wav2vec_ext, y, sr=16000, chunk_sec=30):
    """Extracts Wav2Vec2 768d embedding via 30s chunk pooling to prevent GPU buffer overflow on long audio."""
    chunk_size = chunk_sec * sr
    if len(y) <= chunk_size:
        return wav2vec_ext.extract_embedding(y, sr=sr)
    
    chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size) if len(y[i:i+chunk_size]) >= sr]
    if not chunks:
        return wav2vec_ext.extract_embedding(y, sr=sr)
        
    all_vecs = []
    for c in chunks:
        vec = wav2vec_ext.extract_embedding(c, sr=sr)
        all_vecs.append(vec)
        
    return np.mean(all_vecs, axis=0).astype(np.float32)

def extract_single_avatar_session(pid, audio_path, egemaps_ext, wav2vec_ext, bert_ext, trimming_dict, overwrite=False):
    assert_no_sealed_ids([pid], context=f'extract_single_avatar_session_{pid}')
    
    os.makedirs(AVATAR_CACHE_DIR, exist_ok=True)
    out_npz = os.path.join(AVATAR_CACHE_DIR, f'{pid}_features.npz')
    
    if os.path.exists(out_npz) and not overwrite:
        print(f'PID {pid}: Features already extracted at {out_npz}, skipping.')
        return {'participant_id': pid, 'status': 'already_extracted', 'feature_path': out_npz}
        
    print(f'Processing Avatar PID {pid}:')
    print(f'  Audio: {audio_path}')
    t0 = time.time()
    
    # 1. Audio loading & 33d temporal extraction
    data_33d = extract_33d_features(audio_path)
    y = data_33d['audio']
    sr = data_33d['sr']
    duration_sec = len(y) / sr
    
    # 2. 88d eGeMAPS
    egemaps_88d, pause_metrics = egemaps_ext.extract(y)
    
    # 3. 768d Wav2Vec2 with 30s chunk pooling
    wav2vec_768d = extract_wav2vec_chunked(wav2vec_ext, y, sr=sr, chunk_sec=30)
    
    # 4. BERT text extraction (both raw and trimmed)
    raw_trans_path = os.path.join(RAW_TRANSCRIPT_DIR, f'{pid}_TRANSCRIPT_WHISPER.csv')
    trimmed_trans_path = os.path.join(TRIMMED_TRANSCRIPT_DIR, f'{pid}_TRANSCRIPT_WHISPER_TRIMMED.csv')
    
    bert_raw_768d = np.zeros(768, dtype=np.float32)
    bert_trimmed_768d = np.zeros(768, dtype=np.float32)
    raw_words = 0
    trimmed_words = 0
    
    if os.path.exists(raw_trans_path):
        try:
            bert_raw_768d, raw_words, _ = bert_ext.extract_from_transcript(raw_trans_path)
        except Exception as e:
            print(f'  [WARNING] PID {pid} raw BERT extraction failed: {e}')
            
    trim_info = trimming_dict.get(pid, {})
    trim_status = trim_info.get('status', 'not_evaluated')
    defer_trim = trim_info.get('defer_participant_only_extraction', True)
    
    if os.path.exists(trimmed_trans_path) and not defer_trim:
        try:
            bert_trimmed_768d, trimmed_words, _ = bert_ext.extract_from_transcript(trimmed_trans_path)
        except Exception as e:
            print(f'  [WARNING] PID {pid} trimmed BERT extraction failed: {e}')
            bert_trimmed_768d = bert_raw_768d
    else:
        bert_trimmed_768d = bert_raw_768d
        
    canonical_bert_768d = bert_trimmed_768d if (not defer_trim and np.count_nonzero(bert_trimmed_768d) > 0) else bert_raw_768d
    
    # Save NPZ archive
    np.savez_compressed(
        out_npz,
        mfcc=data_33d['mfcc'],
        delta_mfcc=data_33d['delta_mfcc'],
        loudness=data_33d['loudness'],
        temporal_33d=data_33d['temporal_features'],
        egemaps_88d=egemaps_88d.astype(np.float32),
        wav2vec_768d=wav2vec_768d.astype(np.float32),
        text_bert_768d=canonical_bert_768d.astype(np.float32),
        text_bert_raw_768d=bert_raw_768d.astype(np.float32),
        text_bert_trimmed_768d=bert_trimmed_768d.astype(np.float32)
    )
    
    elapsed = time.time() - t0
    print(f'  [SUCCESS] PID {pid} saved ({elapsed:.1f}s) -> {out_npz}')
    
    return {
        'participant_id': pid,
        'feature_path': os.path.relpath(out_npz, _REPO_ROOT),
        'duration_sec': round(duration_sec, 2),
        'total_frames': len(data_33d['temporal_features']),
        'temporal_dim': 33,
        'egemaps_dim': len(egemaps_88d),
        'wav2vec_dim': len(wav2vec_768d),
        'bert_dim': len(canonical_bert_768d),
        'raw_words': raw_words,
        'trimmed_words': trimmed_words,
        'trimming_status': trim_status,
        'defer_participant_only_extraction': defer_trim,
        'extraction_time_sec': round(elapsed, 2),
        'status': 'success'
    }

def run_avatar_extraction(pids_subset=None, overwrite=False):
    manifest_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'expanded_dataset_manifest.csv')
    df_manifest = pd.read_csv(manifest_path)
    target_df = df_manifest[df_manifest['cohort_role'] == 'avatar_candidate_expansion']
    
    if pids_subset:
        target_df = target_df[target_df['participant_id'].isin(pids_subset)]
        
    print(f'Starting Canonical Feature Extraction for {len(target_df)} Avatar Candidates...')
    assert_no_sealed_ids(target_df['participant_id'].values, context='avatar_feature_extraction_start')
    
    trimming_dict = {}
    if os.path.exists(TRIMMING_AUDIT_CSV):
        df_audit = pd.read_csv(TRIMMING_AUDIT_CSV)
        for idx, row in df_audit.iterrows():
            trimming_dict[int(row['participant_id'])] = row.to_dict()
            
    egemaps_ext = EgemapsExtractor(sr=16000)
    wav2vec_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    wav2vec_ext = Wav2VecExtractor(device=wav2vec_device)
    bert_ext = BertExtractor(device='cpu')
    
    catalog_records = []
    for idx, row in target_df.iterrows():
        pid = int(row['participant_id'])
        rel_audio = row['audio_path']
        full_audio = os.path.join(_REPO_ROOT, rel_audio)
        if not os.path.exists(full_audio):
            print(f'PID {pid}: Audio not found at {full_audio}, skipping.')
            continue
            
        res = extract_single_avatar_session(pid, full_audio, egemaps_ext, wav2vec_ext, bert_ext, trimming_dict, overwrite=overwrite)
        if res.get('status') == 'success':
            catalog_records.append(res)
            df_single = pd.DataFrame([res])
            if os.path.exists(CATALOG_CSV):
                df_old = pd.read_csv(CATALOG_CSV)
                df_comb = pd.concat([df_old, df_single], ignore_index=True).drop_duplicates(subset=['participant_id'], keep='last')
            else:
                df_comb = df_single
            df_comb.to_csv(CATALOG_CSV, index=False)
            
    print(f'Completed extraction for {len(catalog_records)} sessions.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pids', type=int, nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    run_avatar_extraction(pids_subset=args.pids, overwrite=args.overwrite)
