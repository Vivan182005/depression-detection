# Canonical Feature Extraction Pipeline v2
import os
import sys
import glob
import re
import time
import json
import hashlib
import resource
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS
from experiments.research_v2.extractors_v2 import (
    Wav2Vec2ExtractorV2,
    AcousticExtractorV2,
    BertExtractorV2,
    load_canonical_audio,
    normalize_text_for_bert
)

FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
MANIFEST_PATH = os.path.join(FEATURE_STORE_V2_DIR, 'manifest.csv')
PROGRESS_PATH = os.path.join(FEATURE_STORE_V2_DIR, 'extraction_progress.json')

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def get_current_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

def extract_metadata_stats(transcript_path: str, audio_dur: float, domain: str) -> np.ndarray:
    try:
        try:
            tdf = pd.read_csv(transcript_path, sep='	')
            if len(tdf.columns) <= 1:
                tdf = pd.read_csv(transcript_path, sep=',')
        except Exception:
            tdf = pd.read_csv(transcript_path, sep=None, engine='python')
            
        tdf.columns = [c.strip().lower() for c in tdf.columns]
        speaker_col = next((c for c in tdf.columns if 'speaker' in c), None)
        text_col = next((c for c in tdf.columns if c in ['value', 'text', 'transcript', 'utterance']), None)
        if text_col is None:
            text_col = tdf.columns[2] if len(tdf.columns) >= 3 else tdf.columns[-1]

        if speaker_col is not None:
            part_tdf = tdf[tdf[speaker_col].astype(str).str.strip().str.lower() == 'participant']
        else:
            part_tdf = tdf

        turn_count = float(len(part_tdf))
        if 'start_time' in part_tdf.columns and 'stop_time' in part_tdf.columns:
            spk_durations = part_tdf['stop_time'].astype(float) - part_tdf['start_time'].astype(float)
            spk_dur = float(spk_durations.sum())
            sess_time = float(tdf['stop_time'].max()) if len(tdf) > 0 else audio_dur
        else:
            spk_dur = audio_dur
            sess_time = audio_dur
            
        raw_words = part_tdf[text_col].dropna().astype(str).apply(lambda s: len(normalize_text_for_bert(s).split())).sum()
        words = float(raw_words)
        speech_rate = words / max(1.0, spk_dur)
        speaking_ratio = spk_dur / max(1.0, sess_time)
        mean_turn_dur = spk_dur / max(1.0, turn_count)
        return np.array([turn_count, spk_dur, words, speech_rate, speaking_ratio, sess_time, mean_turn_dur], dtype=np.float32)
    except Exception:
        return np.array([0.0, audio_dur, 0.0, 0.0, 1.0, audio_dur, 0.0], dtype=np.float32)

def verify_cached_npz(npz_path: str) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        data = np.load(npz_path)
        required_keys = ['acoustic_154d', 'wav2vec_768d', 'bert_768d']
        for k in required_keys:
            if k not in data:
                return False
        ac = data['acoustic_154d']
        w2v = data['wav2vec_768d']
        bert = data['bert_768d']
        if ac.shape != (154,) or w2v.shape != (768,) or bert.shape != (768,):
            return False
        if not (np.isfinite(ac).all() and np.isfinite(w2v).all() and np.isfinite(bert).all()):
            return False
        if np.all(ac[62:88] == 0):
            return False
        return True
    except Exception:
        return False

def build_cohort_plan() -> List[Dict[str, Any]]:
    plan = []
    
    # 1. WoZ dev (Clean_audio/NEW_AUDIO/*.wav)
    woz_audios = sorted(glob.glob(os.path.join(_REPO_ROOT, 'Clean_audio', 'NEW_AUDIO', '*.wav')))
    for af in woz_audios:
        fn = os.path.basename(af)
        m = re.search(r'(\d+)', fn)
        if not m: continue
        pid = int(m.group(1))
        if pid in SEALED_PIDS:
            continue
            
        trans_path = os.path.join(_REPO_ROOT, 'Clean_audio', 'daicwoz', f'{pid}_TRANSCRIPT.csv')
        plan.append({
            'participant_id': pid,
            'session_id': str(pid),
            'domain': 'woz',
            'is_sealed': False,
            'audio_path': af,
            'transcript_path': trans_path,
            'status': 'pending'
        })
        
    # PID 306 special entry (no audio file on disk)
    plan.append({
        'participant_id': 306,
        'session_id': '306',
        'domain': 'woz',
        'is_sealed': False,
        'audio_path': None,
        'transcript_path': os.path.join(_REPO_ROOT, 'Clean_audio', 'daicwoz', '306_TRANSCRIPT.csv'),
        'status': 'excluded_missing_audio_on_disk'
    })
    
    # 2. Avatar dev (NEW_AUDIO_DAIC/*.wav)
    avatar_audios = sorted(glob.glob(os.path.join(_REPO_ROOT, 'NEW_AUDIO_DAIC', '*.wav')))
    for af in avatar_audios:
        fn = os.path.basename(af)
        m = re.search(r'(\d+)', fn)
        if not m: continue
        pid = int(m.group(1))
        if pid in SEALED_PIDS:
            continue
            
        trans_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev', f'{pid}_TRANSCRIPT_WHISPER.csv')
        plan.append({
            'participant_id': pid,
            'session_id': str(pid),
            'domain': 'avatar',
            'is_sealed': False,
            'audio_path': af,
            'transcript_path': trans_path,
            'status': 'pending'
        })
        
    # 3. Sealed cohort (unseen/**/*.wav) strictly label-blind
    sealed_audios = sorted(glob.glob(os.path.join(_REPO_ROOT, 'unseen', '**', '*.wav'), recursive=True))
    for af in sealed_audios:
        fn = os.path.basename(af)
        m = re.search(r'(\d+)', fn)
        if not m: continue
        pid = int(m.group(1))
        assert pid in SEALED_PIDS, f'Found non-sealed PID {pid} in unseen/ directory'
        
        candidates = [
            os.path.join(_REPO_ROOT, 'unseen', 'transcripts', f'{pid}_TRANSCRIPT.csv'),
            os.path.join(_REPO_ROOT, 'unseen', 'transcripts', '*', f'{pid}_TRANSCRIPT.csv'),
            os.path.join(_REPO_ROOT, 'unseen', '*', f'{pid}_TRANSCRIPT.csv')
        ]
        found_trans = None
        for c in candidates:
            matches = glob.glob(c)
            if matches:
                found_trans = matches[0]
                break
        assert found_trans and os.path.exists(found_trans), f'Missing transcript for sealed PID {pid}'
        
        plan.append({
            'participant_id': pid,
            'session_id': str(pid),
            'domain': 'sealed',
            'is_sealed': True,
            'audio_path': af,
            'transcript_path': found_trans,
            'status': 'pending'
        })
        
    return plan

def run_feature_extraction(limit: Optional[int] = None):
    os.makedirs(FEATURE_STORE_V2_DIR, exist_ok=True)
    
    plan = build_cohort_plan()
    print(f'Total cohort extraction plan: {len(plan)} sessions.')
    woz_count = sum(1 for p in plan if p['domain'] == 'woz' and p['audio_path'])
    avatar_count = sum(1 for p in plan if p['domain'] == 'avatar')
    sealed_count = sum(1 for p in plan if p['is_sealed'])
    print(f'  WoZ dev sessions: {woz_count} (+ 1 excluded PID 306)')
    print(f'  Avatar dev sessions: {avatar_count}')
    print(f'  Sealed sessions (label-blind): {sealed_count}')

    print('\nInitializing Canonical Extractors v2...')
    t0 = time.time()
    ac_extractor = AcousticExtractorV2(sr=16000)
    w2v_extractor = Wav2Vec2ExtractorV2()
    bert_extractor = BertExtractorV2()
    print(f'Extractors initialized in {time.time() - t0:.2f}s. Initial RAM: {get_current_ram_mb():.1f} MB\n')

    manifest_records = []
    durations = []
    completed_count = 0
    start_total_time = time.time()

    for idx, item in enumerate(plan):
        if limit is not None and completed_count >= limit:
            print(f'Reached limit of {limit} sessions. Stopping.')
            break
            
        pid = item['participant_id']
        sid = item['session_id']
        domain = item['domain']
        is_sealed = item['is_sealed']
        audio_path = item['audio_path']
        trans_path = item['transcript_path']
        
        out_npz = os.path.join(FEATURE_STORE_V2_DIR, f'{pid}_v2.npz')
        
        if audio_path is None:
            manifest_records.append({
                'participant_id': pid,
                'session_id': sid,
                'domain': domain,
                'is_sealed': is_sealed,
                'source_audio_path': 'none',
                'source_audio_hash': 'none',
                'audio_duration_sec': 0.0,
                'source_transcript_path': os.path.relpath(trans_path, _REPO_ROOT) if trans_path else 'none',
                'source_transcript_hash': compute_sha256(trans_path) if trans_path and os.path.exists(trans_path) else 'none',
                'feature_file': 'none',
                'wav2vec_model': 'none',
                'acoustic_config': 'none',
                'bert_model': 'none',
                'extractor_version': 'v2.0_canonical',
                'status': 'excluded_missing_audio_on_disk'
            })
            continue

        if verify_cached_npz(out_npz):
            info = sf.info(audio_path)
            manifest_records.append({
                'participant_id': pid,
                'session_id': sid,
                'domain': domain,
                'is_sealed': is_sealed,
                'source_audio_path': os.path.relpath(audio_path, _REPO_ROOT),
                'source_audio_hash': compute_sha256(audio_path),
                'audio_duration_sec': round(info.duration, 2),
                'source_transcript_path': os.path.relpath(trans_path, _REPO_ROOT),
                'source_transcript_hash': compute_sha256(trans_path),
                'feature_file': f'{pid}_v2.npz',
                'wav2vec_model': 'facebook/wav2vec2-base-960h',
                'acoustic_config': 'OpenSMILE_eGeMAPSv02_88d + temporal_33d_mean_std',
                'bert_model': 'bert-base-uncased_norm_text_window510_stride128',
                'extractor_version': 'v2.0_canonical',
                'status': 'cached_verified'
            })
            completed_count += 1
            continue

        sess_t0 = time.time()
        y, sr = load_canonical_audio(audio_path, target_sr=16000)
        audio_dur = len(y) / float(sr)
        
        ac_154 = ac_extractor.extract(y)
        w2v_768 = w2v_extractor.extract(y, sr=16000, chunk_sec=30)
        bert_768, word_count, norm_text = bert_extractor.extract_from_transcript_file(trans_path, domain=domain)
        meta_7d = extract_metadata_stats(trans_path, audio_dur, domain=domain)
        
        np.savez_compressed(
            out_npz,
            participant_id=pid,
            session_id=sid,
            domain=domain,
            is_sealed=is_sealed,
            acoustic_154d=ac_154,
            wav2vec_768d=w2v_768,
            bert_768d=bert_768,
            meta_7d=meta_7d,
            audio_duration_sec=float(audio_dur),
            word_count=int(word_count)
        )
        
        sess_time = time.time() - sess_t0
        durations.append(sess_time)
        completed_count += 1
        
        audio_hash = compute_sha256(audio_path)
        trans_hash = compute_sha256(trans_path)
        
        manifest_records.append({
            'participant_id': pid,
            'session_id': sid,
            'domain': domain,
            'is_sealed': is_sealed,
            'source_audio_path': os.path.relpath(audio_path, _REPO_ROOT),
            'source_audio_hash': audio_hash,
            'audio_duration_sec': round(audio_dur, 2),
            'source_transcript_path': os.path.relpath(trans_path, _REPO_ROOT),
            'source_transcript_hash': trans_hash,
            'feature_file': f'{pid}_v2.npz',
            'wav2vec_model': 'facebook/wav2vec2-base-960h',
            'acoustic_config': 'OpenSMILE_eGeMAPSv02_88d + temporal_33d_mean_std',
            'bert_model': 'bert-base-uncased_norm_text_window510_stride128',
            'extractor_version': 'v2.0_canonical',
            'status': 'freshly_extracted'
        })
        
        if completed_count == 3 or completed_count % 10 == 0 or completed_count == len(plan):
            avg_speed = np.mean(durations[-10:]) if durations else sess_time
            current_ram = get_current_ram_mb()
            print(f'[{completed_count}/{len(plan)}] PID {pid} ({domain}, {audio_dur:.1f}s) extracted in {sess_time:.2f}s | Avg speed: {avg_speed:.2f}s/sess | RAM: {current_ram:.1f} MB')
            
            df_curr = pd.DataFrame(manifest_records)
            df_curr.to_csv(MANIFEST_PATH, index=False)

    df_final = pd.DataFrame(manifest_records)
    df_final.to_csv(MANIFEST_PATH, index=False)
    total_time = time.time() - start_total_time
    print(f'\nExtraction completed in {total_time:.1f}s ({total_time/60.0:.2f} min).')
    print(f'Manifest written to: {MANIFEST_PATH}')
    print(f'Total entries: {len(df_final)}')
    print(df_final["status"].value_counts())
    return df_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit number of sessions')
    args = parser.parse_args()
    run_feature_extraction(limit=args.limit)
