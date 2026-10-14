"""
Audit B1: Provenance of Clean_audio/NEW_AUDIO/*_AUDIO_interviewee.wav
and Empirical Alignment Hypothesis Testing.

Tasks:
1. Inspect provenance: speaker_isolation.py, remove_ellie.py, audio_cleaner.py,
   WAV headers/metadata, file timestamps, git history, logs, and READMEs.
2. For all 178 WoZ dev sessions:
   - Stored WAV duration D vs sum of participant turn durations S_d
   - Per-session gap g = (D - S_d) / n_turns
   - Cumulative-offset boundaries: t_start^(k) = sum_{j=0}^{k-1} d_j + k * g
   - Energy at predicted boundaries vs shifted offsets via permutation test (B=200)
   - Determine per-session pass/fail (pass if g > 0 and E_gap < E_speech and p_perm < 0.05)
   - Do NOT slice stitched audio by original transcript timestamps.
   - Output: experiments/research_v2/woz_alignment_audit.csv
3. For Avatar dev:
   - Validate Silero VAD on a sample of sessions against Whisper timestamps.
"""

import os
import glob
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import silero_vad

def test_session_alignment(wav_path, transcript_path, n_perms=200, seed=42):
    rng = np.random.default_rng(seed)
    
    # Read audio
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y[:, 0]
    total_samples = len(y)
    total_dur = total_samples / sr
    
    # Read transcript
    sep = '\t' if '\t' in open(transcript_path).readline() else ','
    df = pd.read_csv(transcript_path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]
    
    spk_col = 'speaker' if 'speaker' in df.columns else df.columns[2]
    start_col = 'start_time' if 'start_time' in df.columns else df.columns[0]
    stop_col = 'stop_time' if 'stop_time' in df.columns else df.columns[1]
    
    p_df = df[df[spk_col].astype(str).str.lower().str.contains('participant')].copy()
    n_turns = len(p_df)
    
    if n_turns == 0:
        return {
            'dur': total_dur, 'sum_d': 0.0, 'diff': total_dur, 'n_turns': 0,
            'g': 0.0, 'e_gap': np.nan, 'e_speech': np.nan, 'energy_ratio': np.nan,
            'p_perm': 1.0, 'passed': False, 'fail_reason': 'zero_turns'
        }
        
    durations = (p_df[stop_col] - p_df[start_col]).values
    sum_d = float(np.sum(durations))
    diff = total_dur - sum_d
    g = diff / n_turns
    
    if g <= 0 or total_dur < sum_d:
        return {
            'dur': total_dur, 'sum_d': sum_d, 'diff': diff, 'n_turns': n_turns,
            'g': g, 'e_gap': np.nan, 'e_speech': np.nan, 'energy_ratio': np.nan,
            'p_perm': 1.0, 'passed': False, 'fail_reason': 'negative_or_zero_gap'
        }
        
    # Frame-level energy computation (20ms frames, 10ms hop)
    frame_len = int(0.020 * sr)  # 320 samples
    hop_len = int(0.010 * sr)    # 160 samples
    n_frames = (total_samples - frame_len) // hop_len + 1
    
    if n_frames <= 10:
        return {
            'dur': total_dur, 'sum_d': sum_d, 'diff': diff, 'n_turns': n_turns,
            'g': g, 'e_gap': np.nan, 'e_speech': np.nan, 'energy_ratio': np.nan,
            'p_perm': 1.0, 'passed': False, 'fail_reason': 'too_short'
        }
        
    # Efficient frame energy via strided representation
    shape = (n_frames, frame_len)
    strides = (y.strides[0] * hop_len, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    frame_rms = np.sqrt(np.mean(frames**2, axis=-1) + 1e-12)
    frame_times = np.arange(n_frames) * 0.010 + 0.010
    
    # Construct cumulative-offset predicted boundaries
    cum_d = np.concatenate(([0.0], np.cumsum(durations)[:-1]))
    pred_turn_starts = cum_d + np.arange(n_turns) * g
    pred_turn_stops = pred_turn_starts + durations
    
    # Gap k is between stop of turn k and start of turn k+1
    gap_starts = pred_turn_stops[:-1]
    gap_stops = pred_turn_starts[1:]
    
    # Frame masks
    speech_frame_mask = np.zeros(n_frames, dtype=bool)
    for s_t, e_t in zip(pred_turn_starts, pred_turn_stops):
        s_f = int(np.round(s_t / 0.010))
        e_f = int(np.round(e_t / 0.010))
        speech_frame_mask[max(0, s_f):min(n_frames, e_f)] = True
        
    gap_frame_mask = np.zeros(n_frames, dtype=bool)
    for s_t, e_t in zip(gap_starts, gap_stops):
        s_f = int(np.round(s_t / 0.010))
        e_f = int(np.round(e_t / 0.010))
        gap_frame_mask[max(0, s_f):min(n_frames, e_f)] = True
        
    gap_energies = frame_rms[gap_frame_mask]
    speech_energies = frame_rms[speech_frame_mask]
    
    if len(gap_energies) == 0 or len(speech_energies) == 0:
        return {
            'dur': total_dur, 'sum_d': sum_d, 'diff': diff, 'n_turns': n_turns,
            'g': g, 'e_gap': np.nan, 'e_speech': np.nan, 'energy_ratio': np.nan,
            'p_perm': 1.0, 'passed': False, 'fail_reason': 'empty_gap_or_speech'
        }
        
    e_gap = float(np.mean(gap_energies))
    e_speech = float(np.mean(speech_energies))
    energy_ratio = e_gap / e_speech
    
    # Permutation test: roll the gap frame mask by random frame shifts
    min_shift = int(1.0 / 0.010)  # at least 1s shift
    max_shift = n_frames - min_shift
    
    if max_shift > min_shift:
        random_shifts = rng.integers(min_shift, max_shift, size=n_perms)
        count_shifted_lower = 0
        for shift in random_shifts:
            shifted_mask = np.roll(gap_frame_mask, shift)
            e_shifted = np.mean(frame_rms[shifted_mask])
            if e_shifted <= e_gap:
                count_shifted_lower += 1
        p_perm = float((count_shifted_lower + 1) / (n_perms + 1))
    else:
        p_perm = 1.0
        
    passed = bool(p_perm < 0.05 and energy_ratio < 0.85)
    fail_reason = 'none' if passed else ('high_energy_ratio' if energy_ratio >= 0.85 else 'insignificant_perm')
    
    return {
        'dur': total_dur, 'sum_d': sum_d, 'diff': diff, 'n_turns': n_turns,
        'g': g, 'e_gap': e_gap, 'e_speech': e_speech, 'energy_ratio': energy_ratio,
        'p_perm': p_perm, 'passed': passed, 'fail_reason': fail_reason
    }

def main():
    print('Starting B1 Alignment Audit across all 178 DAIC-WOZ dev sessions...')
    transcript_files = sorted(glob.glob('Clean_audio/daicwoz/*_TRANSCRIPT.csv'))
    results = []
    
    for i, tf in enumerate(transcript_files):
        pid = os.path.basename(tf).split('_')[0]
        wav_path = f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav'
        if not os.path.exists(wav_path):
            continue
            
        res = test_session_alignment(wav_path, tf, n_perms=200)
        res['participant_id'] = pid
        results.append(res)
        
        if (len(results)) % 30 == 0:
            print(f'  Processed {len(results)}/178 sessions...')
            
    df = pd.DataFrame(results)
    cols = ['participant_id', 'dur', 'sum_d', 'diff', 'n_turns', 'g', 'e_gap', 'e_speech', 'energy_ratio', 'p_perm', 'passed', 'fail_reason']
    df = df[cols]
    
    out_csv = 'experiments/research_v2/woz_alignment_audit.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved alignment audit to {out_csv}')
    
    total = len(df)
    passed_count = df['passed'].sum()
    failed_count = total - passed_count
    pass_pct = (passed_count / total) * 100
    
    print(f'Total sessions: {total}')
    print(f'Passed alignment: {passed_count} ({pass_pct:.1f}%)')
    print(f'Failed alignment: {failed_count} ({100 - pass_pct:.1f}%)')
    print('\nFail reason breakdown:')
    print(df['fail_reason'].value_counts())
    
    valid_diff = df.dropna(subset=['diff', 'n_turns'])
    corr = np.corrcoef(valid_diff['diff'], valid_diff['n_turns'])[0, 1]
    print(f'Correlation (excess duration vs n_turns): {corr:.4f}')
    g_mean, g_std, g_med = df['g'].mean(), df['g'].std(), df['g'].median()
    print(f'Mean gap g: {g_mean:.4f}s (std {g_std:.4f}s, median {g_med:.4f}s)')
    
    # Validate Silero VAD on Avatar Dev Sample
    print('\nValidating Silero VAD on Avatar Dev Sample...')
    vad_model = silero_vad.load_silero_vad()
    
    avatar_samples = ['600', '602', '606', '613', '620', '625']
    vad_results = []
    
    for pid in avatar_samples:
        wav_path = f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav'
        tx_path = f'experiments/research_v2/transcripts_avatar_dev/{pid}_TRANSCRIPT_WHISPER.csv'
        if not os.path.exists(wav_path) or not os.path.exists(tx_path):
            continue
            
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        wav_tensor = torch.from_numpy(y.astype('float32'))
        speech_timestamps = silero_vad.get_speech_timestamps(wav_tensor, vad_model, sampling_rate=16000)
        
        frame_rate = 100
        n_frames = int(len(y) / sr * frame_rate)
        vad_mask = np.zeros(n_frames, dtype=bool)
        for seg in speech_timestamps:
            sf_idx = int((seg['start'] / sr) * frame_rate)
            ef_idx = int((seg['end'] / sr) * frame_rate)
            vad_mask[sf_idx:ef_idx] = True
            
        tx_df = pd.read_csv(tx_path)
        tx_mask = np.zeros(n_frames, dtype=bool)
        for _, row in tx_df.iterrows():
            sf_idx = int(row['start_time'] * frame_rate)
            ef_idx = int(row['stop_time'] * frame_rate)
            tx_mask[sf_idx:ef_idx] = True
            
        intersection = np.sum(vad_mask & tx_mask)
        union = np.sum(vad_mask | tx_mask)
        iou = intersection / union if union > 0 else 0.0
        acc = np.mean(vad_mask == tx_mask)
        vad_speech_dur = np.sum(vad_mask) / frame_rate
        tx_speech_dur = np.sum(tx_mask) / frame_rate
        
        vad_results.append({
            'pid': pid, 'audio_dur': round(len(y)/sr, 2),
            'vad_speech_dur': round(vad_speech_dur, 2), 'tx_speech_dur': round(tx_speech_dur, 2),
            'iou': round(iou, 4), 'frame_agreement': round(acc, 4), 'n_vad_segments': len(speech_timestamps)
        })
        
    vad_df = pd.DataFrame(vad_results)
    print(vad_df.to_string())
    vad_df.to_csv('experiments/research_v2/avatar_silero_vad_validation.csv', index=False)
    print('\nSaved avatar Silero VAD validation to experiments/research_v2/avatar_silero_vad_validation.csv')

if __name__ == '__main__':
    main()
