"""
Task C2: Development-Only Silero VAD Parameter Validation
Evaluates frozen VAD parameters against Whisper-tiny word timestamps across:
- 20 eligible DAIC-WOZ development sessions
- 20 eligible E-DAIC Avatar development sessions
Total N = 40 sessions sampled across duration strata (quintiles).

Metrics:
- Frame agreement (10ms resolution, 100 frames/sec)
- IoU (Intersection over Union)
- Fraction of VAD segments with NO overlapping reference words (50ms tolerance)
- Segment duration distributions (mean, median, p10, p90)
- Speech coverage ratio
- VAD failure flags (n_segments == 0 or speech_sec < 2.0s)
"""

import os
import sys
import glob
import time
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torch
import silero_vad
from transformers import pipeline

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine

OUT_CSV = os.path.join(os.path.dirname(__file__), 'vad_validation_audit.csv')

def sample_sessions():
    labels_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'labels_official.csv'))
    
    # Strictly exclude sealed cohort (N=33)
    woz_dev = labels_df[(labels_df['domain'] == 'woz') & (labels_df['is_sealed'] == False) & (labels_df['participant_id'] != 306)].copy()
    av_dev = labels_df[(labels_df['domain'] == 'avatar') & (labels_df['is_sealed'] == False) & (labels_df['participant_id'] != 602)].copy()
    
    # Compute audio duration on disk for duration quintile sampling
    woz_durations = []
    for pid in woz_dev['participant_id']:
        p = os.path.join(_REPO_ROOT, f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav')
        if os.path.exists(p):
            woz_durations.append((pid, sf.info(p).duration))
    woz_df = pd.DataFrame(woz_durations, columns=['pid', 'duration']).sort_values('duration')
    
    av_durations = []
    for pid in av_dev['participant_id']:
        p = os.path.join(_REPO_ROOT, f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav')
        if os.path.exists(p):
            av_durations.append((pid, sf.info(p).duration))
    av_df = pd.DataFrame(av_durations, columns=['pid', 'duration']).sort_values('duration')
    
    def pick_quintiles(df, n_per_quintile=4, seed=42):
        df = df.copy()
        df['quintile'] = pd.qcut(df['duration'], q=5, labels=False, duplicates='drop')
        selected = []
        for q in sorted(df['quintile'].unique()):
            sub = df[df['quintile'] == q]
            chosen = sub.sample(n=min(n_per_quintile, len(sub)), random_state=seed + int(q))
            selected.append(chosen)
        return pd.concat(selected)
        
    sampled_woz = pick_quintiles(woz_df, 4, seed=101)
    sampled_av = pick_quintiles(av_df, 4, seed=202)
    
    return sampled_woz, sampled_av

def evaluate_session(wav_path, pid, domain, vad_model, asr_pipe):
    assert_sealed_quarantine(pid)
    assert_label_blind({'participant_id': pid, 'domain': domain})
    
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y[:, 0]
    total_dur = len(y) / sr
    
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        
    # 1. Silero VAD Speech Timestamps (Frozen Parameters)
    wav_tensor = torch.from_numpy(y.astype('float32'))
    raw_vad = silero_vad.get_speech_timestamps(
        wav_tensor, vad_model, sampling_rate=16000,
        threshold=0.50,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200
    )
    
    max_samples = int(15.0 * sr)
    vad_segments = []
    for seg in raw_vad:
        s = seg['start']
        e = seg['end']
        if (e - s) > max_samples:
            cur = s
            while cur < e:
                nxt = min(e, cur + max_samples)
                vad_segments.append({'start': cur, 'end': nxt})
                cur = nxt
        else:
            vad_segments.append({'start': s, 'end': e})
            
    # 2. Whisper-tiny Word Timestamps (Imperfect Reference)
    asr_res = asr_pipe(y, return_timestamps='word', generate_kwargs={'language': 'en', 'task': 'transcribe'})
    word_chunks = asr_res.get('chunks', [])
    valid_words = [c for c in word_chunks if c.get('timestamp') and c['timestamp'][0] is not None and c['timestamp'][1] is not None]
    
    frame_rate = 100
    n_frames = int(total_dur * frame_rate)
    
    vad_mask = np.zeros(n_frames, dtype=bool)
    for seg in vad_segments:
        sf_idx = int((seg['start'] / sr) * frame_rate)
        ef_idx = int((seg['end'] / sr) * frame_rate)
        vad_mask[max(0, sf_idx):min(n_frames, ef_idx)] = True
        
    ref_mask = np.zeros(n_frames, dtype=bool)
    ref_word_intervals = []
    for w in valid_words:
        st, et = w['timestamp']
        ref_word_intervals.append((st, et))
        sf_idx = int(st * frame_rate)
        ef_idx = int(et * frame_rate)
        ref_mask[max(0, sf_idx):min(n_frames, ef_idx)] = True
        
    tp = np.sum(vad_mask & ref_mask)
    tn = np.sum((~vad_mask) & (~ref_mask))
    fp = np.sum(vad_mask & (~ref_mask))
    fn = np.sum((~vad_mask) & ref_mask)
    frame_agreement = (tp + tn) / n_frames if n_frames > 0 else 0.0
    
    intersection = tp
    union = np.sum(vad_mask | ref_mask)
    iou = intersection / union if union > 0 else 0.0
    
    vad_speech_sec = np.sum(vad_mask) / frame_rate
    ref_speech_sec = np.sum(ref_mask) / frame_rate
    
    n_vad_segs = len(vad_segments)
    no_overlap_segs = 0
    seg_durations = []
    
    for seg in vad_segments:
        s_sec = seg['start'] / sr
        e_sec = seg['end'] / sr
        dur = e_sec - s_sec
        seg_durations.append(dur)
        
        has_overlap = any((w_end >= (s_sec - 0.05)) and (w_start <= (e_sec + 0.05)) for w_start, w_end in ref_word_intervals)
        if not has_overlap:
            no_overlap_segs += 1
            
    frac_no_ref = (no_overlap_segs / n_vad_segs) if n_vad_segs > 0 else 0.0
    
    seg_dur_mean = np.mean(seg_durations) if seg_durations else 0.0
    seg_dur_median = np.median(seg_durations) if seg_durations else 0.0
    seg_dur_p10 = np.percentile(seg_durations, 10) if seg_durations else 0.0
    seg_dur_p90 = np.percentile(seg_durations, 90) if seg_durations else 0.0
    
    return {
        'participant_id': pid, 'domain': domain, 'audio_duration_s': round(total_dur, 2),
        'n_vad_segments': n_vad_segs, 'n_ref_words': len(valid_words),
        'vad_speech_duration_s': round(vad_speech_sec, 2),
        'ref_speech_duration_s': round(ref_speech_sec, 2),
        'speech_coverage_ratio': round(vad_speech_sec / total_dur, 4) if total_dur > 0 else 0.0,
        'frame_agreement': round(frame_agreement, 4),
        'iou': round(iou, 4),
        'vad_only_duration_s': round(fp / frame_rate, 2),
        'ref_only_duration_s': round(fn / frame_rate, 2),
        'frac_vad_no_overlapping_words': round(frac_no_ref, 4),
        'seg_duration_mean_s': round(seg_dur_mean, 2),
        'seg_duration_median_s': round(seg_dur_median, 2),
        'seg_duration_p10_s': round(seg_dur_p10, 2),
        'seg_duration_p90_s': round(seg_dur_p90, 2),
        'vad_failure_flag': bool(n_vad_segs == 0 or vad_speech_sec < 2.0)
    }

def main():
    os.chdir(_REPO_ROOT)
    print('Sampling 20 WoZ Dev and 20 Avatar Dev sessions across duration strata...')
    woz_sample, av_sample = sample_sessions()
    
    print('\nInitializing Silero VAD and Whisper-tiny pipeline (chunk_length_s=30, stride_length_s=5)...')
    vad_model = silero_vad.load_silero_vad()
    # Memory-safe chunked pipeline (prevents O(T^2) quadratic memory explosion on long sessions)
    asr_pipe = pipeline(
        'automatic-speech-recognition',
        model='openai/whisper-tiny',
        return_timestamps='word',
        chunk_length_s=30,
        stride_length_s=5,
        device='cpu'
    )
    
    records = []
    existing_pids = set()
    if os.path.exists(OUT_CSV):
        try:
            old_df = pd.read_csv(OUT_CSV)
            records = old_df.to_dict('records')
            existing_pids = set(old_df['participant_id'])
            print(f'Loaded {len(records)} existing evaluated sessions from {OUT_CSV}')
        except Exception:
            pass
            
    print('\nEvaluating 20 DAIC-WOZ Development Sessions...')
    for idx, row in woz_sample.reset_index(drop=True).iterrows():
        pid = int(row['pid'])
        if pid in existing_pids:
            print(f'  [WoZ {idx+1}/20] PID {pid}: Already evaluated, skipping.')
            continue
        wav_p = os.path.join(_REPO_ROOT, f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav')
        res = evaluate_session(wav_p, pid, 'woz_dev', vad_model, asr_pipe)
        records.append(res)
        existing_pids.add(pid)
        pd.DataFrame(records).to_csv(OUT_CSV, index=False)
        print(f'  [WoZ {idx+1}/20] PID {pid}: dur={res["audio_duration_s"]}s, VAD segs={res["n_vad_segments"]}, agreement={res["frame_agreement"]*100:.1f}%, no_ref={res["frac_vad_no_overlapping_words"]*100:.1f}%')
        
    print('\nEvaluating 20 E-DAIC Avatar Development Sessions...')
    for idx, row in av_sample.reset_index(drop=True).iterrows():
        pid = int(row['pid'])
        if pid in existing_pids:
            print(f'  [Avatar {idx+1}/20] PID {pid}: Already evaluated, skipping.')
            continue
        wav_p = os.path.join(_REPO_ROOT, f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav')
        res = evaluate_session(wav_p, pid, 'avatar_dev', vad_model, asr_pipe)
        records.append(res)
        existing_pids.add(pid)
        pd.DataFrame(records).to_csv(OUT_CSV, index=False)
        print(f'  [Avatar {idx+1}/20] PID {pid}: dur={res["audio_duration_s"]}s, VAD segs={res["n_vad_segments"]}, agreement={res["frame_agreement"]*100:.1f}%, no_ref={res["frac_vad_no_overlapping_words"]*100:.1f}%')
        
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved VAD validation audit to {OUT_CSV}')
    
    print('\n=== Domain-Level Summary ===')
    summary = df.groupby('domain')[['frame_agreement', 'iou', 'frac_vad_no_overlapping_words', 'speech_coverage_ratio', 'seg_duration_mean_s', 'seg_duration_median_s']].mean().round(4)
    print(summary.to_string())

if __name__ == '__main__':
    main()
