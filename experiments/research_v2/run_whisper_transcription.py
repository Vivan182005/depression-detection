"""
Script: experiments/research_v2/run_whisper_transcription.py
Description: Robust, reproducible local Whisper transcription pipeline for DAIC-WOZ and E-DAIC.
Optimized for Apple Silicon M2 (8 GB RAM) using openai/whisper-tiny via HuggingFace transformers.
Strictly isolates sealed cohort transcripts into a quarantined directory.
"""

import os
import sys
import time
import argparse
import glob
import re
import pandas as pd
import soundfile as sf
import torch
from transformers import pipeline

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DEV_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev')
SEALED_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_sealed_isolated')
LOG_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcription_log.csv')

ELLIE_INTRO_PATTERNS = [
    re.compile(r"(hi,? I'm ellie)", re.IGNORECASE),
    re.compile(r"(thanks for coming in today)", re.IGNORECASE),
    re.compile(r"(safe and secure environment)", re.IGNORECASE),
    re.compile(r"(think of me as a friend)", re.IGNORECASE),
    re.compile(r"(how are you doing today)", re.IGNORECASE),
]

def detect_interview_start(chunks):
    """
    Detects the timestamp where Ellie begins the interview or where setup dialogue ends.
    Returns (start_timestamp, detection_method)
    """
    for chunk in chunks:
        text = chunk.get('text', '')
        ts = chunk.get('timestamp', (0.0, 0.0))
        t_start = ts[0] if ts and ts[0] is not None else 0.0
        for pat in ELLIE_INTRO_PATTERNS:
            if pat.search(text):
                return t_start, f'ellie_intro_phrase: "{pat.pattern}"'
    return 0.0, 'no_setup_phrase_detected_full_session_used'

def transcribe_file(pipe, audio_path, pid, output_dir, cohort_role, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f'{pid}_TRANSCRIPT_WHISPER.csv')
    
    if os.path.exists(out_csv) and not overwrite:
        print(f'PID {pid}: Transcript already exists at {out_csv}, skipping.')
        return {'participant_id': pid, 'status': 'already_exists', 'output_csv': out_csv}

    try:
        info = sf.info(audio_path)
        audio_dur = round(info.duration, 2)
        sr = info.samplerate
        ch = info.channels
    except Exception as e:
        audio_dur, sr, ch = 0.0, 0, 0

    t0 = time.time()
    try:
        res = pipe(audio_path)
        t_proc = time.time() - t0
        speed_factor = round(audio_dur / t_proc, 2) if t_proc > 0 else 0.0
        
        chunks = res.get('chunks', [])
        full_text = res.get('text', '').strip()
        word_count = len(full_text.split())
        
        # Detect setup trim
        interview_start_sec, trim_reason = detect_interview_start(chunks)
        effective_dur = round(max(0.0, audio_dur - interview_start_sec), 2)
        
        # Save chunk CSV
        chunk_rows = []
        for idx, c in enumerate(chunks):
            ts = c.get('timestamp', (None, None))
            s_time = ts[0] if ts and ts[0] is not None else None
            e_time = ts[1] if ts and ts[1] is not None else None
            txt = c.get('text', '').strip()
            chunk_rows.append({
                'participant_id': pid,
                'chunk_index': idx,
                'start_time': s_time,
                'stop_time': e_time,
                'text': txt
            })
            
        df_chunks = pd.DataFrame(chunk_rows)
        df_chunks.to_csv(out_csv, index=False)
        
        log_entry = {
            'participant_id': pid,
            'cohort_role': cohort_role,
            'audio_path': os.path.relpath(audio_path, _REPO_ROOT),
            'audio_duration_sec': audio_dur,
            'audio_sample_rate': sr,
            'audio_channels': ch,
            'transcript_csv': os.path.relpath(out_csv, _REPO_ROOT),
            'model_name': 'openai/whisper-tiny',
            'whisper_version': 'transformers-whisper-tiny',
            'chunk_length_s': 30,
            'batch_size': 8,
            'compute_device': 'cpu',
            'processing_time_sec': round(t_proc, 2),
            'speed_factor_realtime': speed_factor,
            'total_chunks': len(chunks),
            'word_count': word_count,
            'interview_start_sec': interview_start_sec,
            'setup_trimmed_duration_sec': effective_dur,
            'trim_detection_rule': trim_reason,
            'status': 'success',
            'error_message': ''
        }
        print(f'PID {pid}: Completed in {t_proc:.1f}s ({speed_factor}x RT) | {len(chunks)} chunks, {word_count} words -> {out_csv}')
        return log_entry
        
    except Exception as e:
        t_proc = time.time() - t0
        err_msg = str(e)
        print(f'PID {pid}: FAILED with error: {err_msg}')
        return {
            'participant_id': pid,
            'cohort_role': cohort_role,
            'audio_path': os.path.relpath(audio_path, _REPO_ROOT),
            'audio_duration_sec': audio_dur,
            'audio_sample_rate': sr,
            'audio_channels': ch,
            'transcript_csv': '',
            'model_name': 'openai/whisper-tiny',
            'whisper_version': 'transformers-whisper-tiny',
            'chunk_length_s': 30,
            'batch_size': 8,
            'compute_device': 'cpu',
            'processing_time_sec': round(t_proc, 2),
            'speed_factor_realtime': 0.0,
            'total_chunks': 0,
            'word_count': 0,
            'interview_start_sec': 0.0,
            'setup_trimmed_duration_sec': 0.0,
            'trim_detection_rule': 'error',
            'status': 'failed',
            'error_message': err_msg
        }

def run_transcription(cohort='dev_avatar', pids_subset=None, overwrite=False):
    manifest_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'expanded_dataset_manifest.csv')
    df_manifest = pd.read_csv(manifest_path)
    
    if cohort == 'dev_avatar':
        target_df = df_manifest[df_manifest['cohort_role'] == 'avatar_candidate_expansion']
        out_dir = DEV_OUTPUT_DIR
    elif cohort == 'sealed_avatar':
        target_df = df_manifest[(df_manifest['cohort_role'] == 'sealed_quarantine') & (df_manifest['domain'] == 'autonomous_avatar')]
        out_dir = SEALED_OUTPUT_DIR
    elif cohort == 'all_avatar':
        target_df = df_manifest[df_manifest['domain'] == 'autonomous_avatar']
        out_dir = None
    else:
        raise ValueError(f'Unknown cohort: {cohort}')
        
    if pids_subset:
        target_df = target_df[target_df['participant_id'].isin(pids_subset)]
        
    print(f'Starting Whisper transcription for cohort "{cohort}": {len(target_df)} participants...')
    print(f'Model: openai/whisper-tiny | Device: CPU (torch threads: {torch.get_num_threads()}) | batch_size: 8')
    
    pipe = pipeline(
        'automatic-speech-recognition',
        model='openai/whisper-tiny',
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=True,
        device='cpu'
    )
    
    log_entries = []
    for idx, row in target_df.iterrows():
        pid = int(row['participant_id'])
        rel_audio = row['audio_path']
        role = row['cohort_role']
        if pd.isna(rel_audio) or not rel_audio:
            print(f'PID {pid}: No audio file specified, skipping.')
            continue
            
        full_audio_path = os.path.join(_REPO_ROOT, rel_audio)
        if not os.path.exists(full_audio_path):
            print(f'PID {pid}: Audio file not found at {full_audio_path}, skipping.')
            continue
            
        current_out_dir = SEALED_OUTPUT_DIR if role == 'sealed_quarantine' else DEV_OUTPUT_DIR
        res = transcribe_file(pipe, full_audio_path, pid, current_out_dir, role, overwrite=overwrite)
        if res.get('status') != 'already_exists':
            log_entries.append(res)
            df_single = pd.DataFrame([res])
            if os.path.exists(LOG_PATH):
                df_old_log = pd.read_csv(LOG_PATH)
                df_combined = pd.concat([df_old_log, df_single], ignore_index=True).drop_duplicates(subset=['participant_id'], keep='last')
            else:
                df_combined = df_single
            df_combined.to_csv(LOG_PATH, index=False)
        
    print('Transcription run completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='dev_avatar', choices=['dev_avatar', 'sealed_avatar', 'all_avatar'])
    parser.add_argument('--pids', type=int, nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    run_transcription(cohort=args.cohort, pids_subset=args.pids, overwrite=args.overwrite)