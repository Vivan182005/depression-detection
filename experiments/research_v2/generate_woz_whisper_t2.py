"""
Script: experiments/research_v2/generate_woz_whisper_t2.py
Transcribes 178 WoZ dev sessions using whisper-tiny to create Variant T2 text embeddings.
Updates all feature_store_v2 .npz files with bert_t1_768d and bert_t2_768d.
"""
import os
import sys
import time
import glob
import re
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import pipeline

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS, assert_no_sealed_ids
from experiments.research_v2.extractors_v2 import (
    normalize_text_for_bert,
    BertExtractorV2,
    get_robust_acoustic_135d
)

FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
TRANSCRIPTS_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2')
WOZ_WHISPER_DIR = os.path.join(TRANSCRIPTS_V2_DIR, 'woz_whisper')
os.makedirs(WOZ_WHISPER_DIR, exist_ok=True)

def main():
    print("=== Generating WoZ Whisper-tiny Transcripts & BERT T2 Features ===")
    manifest = pd.read_csv(os.path.join(FEATURE_STORE_V2_DIR, 'manifest.csv'))
    
    # 1. First, populate bert_t1_768d and bert_t2_768d for Avatar Dev and Sealed
    avatar_and_sealed = manifest[(manifest['domain'] != 'woz') | (manifest['is_sealed'])]
    print(f"Populating Avatar/Sealed bert_t1 and bert_t2 ({len(avatar_and_sealed)} sessions)...")
    for _, r in avatar_and_sealed.iterrows():
        if r['feature_file'] == 'none': continue
        npz_path = os.path.join(FEATURE_STORE_V2_DIR, r['feature_file'])
        data = dict(np.load(npz_path))
        if 'bert_t2_768d' not in data or 'bert_t1_768d' not in data:
            bert_orig = data['bert_768d']
            data['bert_t1_768d'] = bert_orig
            data['bert_t2_768d'] = bert_orig
            if 'acoustic_robust_135d' not in data:
                data['acoustic_robust_135d'] = get_robust_acoustic_135d(data['acoustic_154d'])
            np.savez_compressed(npz_path, **data)
    print("Avatar Dev and Sealed sessions synchronized.")

    # 2. Transcribe WoZ dev sessions with whisper-tiny
    woz_dev = manifest[(manifest['domain'] == 'woz') & (~manifest['is_sealed']) & (manifest['feature_file'] != 'none')]
    assert_no_sealed_ids(woz_dev['participant_id'].values, context='woz_whisper_transcription')
    print(f"Target WoZ dev sessions to transcribe: {len(woz_dev)}")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Initializing openai/whisper-tiny on {device}...")
    pipe = pipeline(
        'automatic-speech-recognition',
        model='openai/whisper-tiny',
        chunk_length_s=30,
        batch_size=16,
        device=device
    )
    
    print(f"Initializing BertExtractorV2 on {device}...")
    bert_extractor = BertExtractorV2(device=device)
    
    t0_start = time.time()
    completed = 0
    
    for idx, (_, r) in enumerate(woz_dev.iterrows()):
        pid = int(r['participant_id'])
        audio_rel = r['source_audio_path']
        audio_path = os.path.join(_REPO_ROOT, audio_rel)
        out_csv = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        npz_path = os.path.join(FEATURE_STORE_V2_DIR, r['feature_file'])
        
        # Check if transcript already exists
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 20:
            df_existing = pd.read_csv(out_csv)
            full_text = ' '.join(df_existing['text'].dropna().astype(str).tolist())
        else:
            t0 = time.time()
            res = pipe(audio_path, generate_kwargs={'language': 'english', 'task': 'transcribe'})
            full_text = res.get('text', '')
            df_out = pd.DataFrame([{
                'participant_id': pid,
                'text': full_text,
                'source_flag': 'whisper_tiny_woz_dev'
            }])
            df_out.to_csv(out_csv, index=False)
            t_infer = time.time() - t0
            dur = sf.info(audio_path).duration
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"[{idx+1}/{len(woz_dev)}] WoZ PID {pid} (dur: {dur:.1f}s) transcribed in {t_infer:.2f}s ({dur/t_infer:.1f}x RT)")

        # Extract BERT T2 embedding
        bert_t2 = bert_extractor.extract_from_text(full_text)
        
        # Update npz
        data = dict(np.load(npz_path))
        data['bert_t1_768d'] = data['bert_768d']
        data['bert_t2_768d'] = bert_t2
        if 'acoustic_robust_135d' not in data:
            data['acoustic_robust_135d'] = get_robust_acoustic_135d(data['acoustic_154d'])
        np.savez_compressed(npz_path, **data)
        completed += 1
        
        if (completed % 15 == 0) or (completed == len(woz_dev)):
            elapsed = time.time() - t0_start
            rate = elapsed / completed
            remaining = (len(woz_dev) - completed) * rate
            print(f"==> Progress: {completed}/{len(woz_dev)} sessions processed. Elapsed: {elapsed/60:.1f}m, Est remaining: {remaining/60:.1f}m")

    print(f"\nAll {completed} WoZ dev sessions successfully transcribed and updated with BERT T2!")

if __name__ == '__main__':
    main()
