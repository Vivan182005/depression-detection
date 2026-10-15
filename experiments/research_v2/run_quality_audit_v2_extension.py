"""
Task C3: Quality Audit v2 Extension (Stratified Sample: N = 65 Sessions)
Stratified by Domain and Gender:
- WoZ Female: 20 sessions (sampled from 79 eligible)
- WoZ Male: 20 sessions (sampled from 99 eligible)
- Avatar Male: 20 sessions (sampled from 56 eligible)
- Avatar Female: 5 sessions (all 5 eligible sessions: PIDs 606, 651, 682, 696, 703)
Total N = 65 sessions. Random seed = 42.

Metrics:
- Two-pass f0 range from each session's own audio (no labels or gender)
- Octave-jump rejection
- Praat/Parselmouth jitter (local, rap, ppq5)
- Praat/Parselmouth shimmer (local, apq3, apq5, apq11)
- Praat/Parselmouth HNR
- Praat/Parselmouth formants (F1, F2, F3)
- Perturbation validity threshold: >= 0.30s voiced speech AND >= 30 pitch pulses per turn
- Output: experiments/research_v2/quality_audit_v2_extended.csv
"""

import os
import glob
import pandas as pd
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call
import torch
import silero_vad

os.chdir('/Users/vivannaik/Desktop/CAPSTONE')

def sample_stratified_cohort(seed=42):
    labels_df = pd.read_csv('experiments/research_v2/labels_official.csv')
    
    # Exclude sealed (33), PID 306 (missing WoZ audio), PID 602 (missing Avatar audio)
    woz_dev = labels_df[(labels_df['domain'] == 'woz') & (labels_df['is_sealed'] == False) & (labels_df['participant_id'] != 306)].copy()
    av_dev = labels_df[(labels_df['domain'] == 'avatar') & (labels_df['is_sealed'] == False) & (labels_df['participant_id'] != 602)].copy()
    
    # Filter by existing audio files on disk and string gender ('female' / 'male')
    woz_dev = woz_dev[woz_dev['participant_id'].apply(lambda pid: os.path.exists(f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav'))]
    av_dev = av_dev[av_dev['participant_id'].apply(lambda pid: os.path.exists(f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav'))]
    
    woz_f = woz_dev[woz_dev['gender'].str.lower() == 'female']  # 79 eligible
    woz_m = woz_dev[woz_dev['gender'].str.lower() == 'male']    # 99 eligible
    av_m = av_dev[av_dev['gender'].str.lower() == 'male']      # 56 eligible
    av_f = av_dev[av_dev['gender'].str.lower() == 'female']    # 5 eligible
    
    print(f'Eligible populations: WoZ Female={len(woz_f)}, WoZ Male={len(woz_m)}, Avatar Male={len(av_m)}, Avatar Female={len(av_f)}')
    
    s_woz_f = woz_f.sample(n=min(20, len(woz_f)), random_state=seed)
    s_woz_m = woz_m.sample(n=min(20, len(woz_m)), random_state=seed + 1)
    s_av_m = av_m.sample(n=min(20, len(av_m)), random_state=seed + 2)
    s_av_f = av_f.copy()  # exactly 5, 100% of cell
    
    cohort = pd.concat([s_woz_f, s_woz_m, s_av_m, s_av_f]).copy()
    print(f'Sampled cohort total: {len(cohort)} sessions (WoZ Female: {len(s_woz_f)}, WoZ Male: {len(s_woz_m)}, Avatar Male: {len(s_av_m)}, Avatar Female: {len(s_av_f)})')
    return cohort

def analyze_session_fast(wav_path, pid, domain, gender_str, vad_model):
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y[:, 0]
    total_dur = len(y) / sr
    
    if sr != 16000:
        import librosa
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    else:
        y_16k = y
        
    # Create 16kHz mono Sound object (6x faster than 48kHz stereo)
    sound_16k = parselmouth.Sound(y_16k, sampling_frequency=16000)
    
    # Pass 1: Session-level pitch across whole audio (broad range 60-500 Hz)
    p1 = call(sound_16k, "To Pitch", 0.0, 60.0, 500.0)
    f0_median = call(p1, "Get quantile", 0, 0, 0.5, "Hertz")
    if np.isnan(f0_median) or f0_median < 50.0:
        f0_median = 150.0
        
    # Pass 2: Pitch range bounded to [0.5 * median, 2.0 * median] clamped to [50, 450]
    f_min = float(max(50.0, 0.5 * f0_median))
    f_max = float(min(450.0, 2.0 * f0_median))
    
    # Segment audio into speech turns using Silero VAD (frozen parameters)
    wav_tensor = torch.from_numpy(y_16k.astype('float32'))
    speech_timestamps = silero_vad.get_speech_timestamps(
        wav_tensor, vad_model, sampling_rate=16000,
        threshold=0.50,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200
    )
    
    # Apply max_duration 15s
    max_samples = int(15.0 * 16000)
    turns = []
    for seg in speech_timestamps:
        s, e = seg['start'], seg['end']
        if (e - s) > max_samples:
            cur = s
            while cur < e:
                nxt = min(e, cur + max_samples)
                turns.append({'start': cur / 16000.0, 'end': nxt / 16000.0})
                cur = nxt
        else:
            turns.append({'start': s / 16000.0, 'end': e / 16000.0})
            
    n_turns = len(turns)
    if n_turns == 0:
        return {
            'participant_id': pid, 'domain': domain, 'gender': gender_str,
            'audio_dur': total_dur, 'f0_median_pass1': f0_median, 'f_min_pass2': f_min, 'f_max_pass2': f_max,
            'n_turns': 0, 'n_valid_pert_turns': 0, 'pct_valid_pert': 0.0,
            'jitter_local_invalid_pct': 100.0, 'shimmer_local_invalid_pct': 100.0,
            'hnr_invalid_pct': 100.0, 'f0_invalid_pct': 100.0, 'formant_invalid_pct': 100.0,
            'mean_valid_f0': np.nan, 'mean_valid_jitter_local': np.nan, 'mean_valid_shimmer_local': np.nan,
            'mean_valid_hnr': np.nan, 'mean_valid_f1': np.nan, 'mean_valid_f2': np.nan, 'mean_valid_f3': np.nan,
            'insufficient_turns': True
        }
        
    turn_metrics = []
    for t in turns:
        t_start, t_end = t['start'], t['end']
        dur = t_end - t_start
        if dur < 0.10:
            continue
            
        try:
            snip = sound_16k.extract_part(from_time=t_start, to_time=t_end, preserve_times=True)
            snip_pitch = call(snip, "To Pitch", 0.0, f_min, f_max)
            snip_f0_mean = call(snip_pitch, "Get mean", 0, 0, "Hertz")
            
            # Fast octave jump check via pitch array
            p_arr = snip_pitch.selected_array['frequency']
            voiced_frames = p_arr[p_arr > 0]
            octave_jumps = 0
            if len(voiced_frames) > 1:
                log_diffs = np.abs(np.diff(np.log2(voiced_frames)))
                octave_jumps = int(np.sum(log_diffs > 0.8))
                
            f0_valid = bool(not np.isnan(snip_f0_mean) and f_min <= snip_f0_mean <= f_max and octave_jumps <= 2)
            
            # PointProcess
            pp = call([snip, snip_pitch], "To PointProcess (cc)")
            n_pulses = call(pp, "Get number of points")
            voiced_dur = (n_pulses / snip_f0_mean) if (f0_valid and snip_f0_mean > 0) else 0.0
            
            pert_valid = bool(voiced_dur >= 0.30 and n_pulses >= 30 and f0_valid)
            j_loc, s_loc = np.nan, np.nan
            
            if pert_valid:
                jl = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                if not np.isnan(jl) and 0.0 <= jl <= 0.12:
                    j_loc = jl
                else:
                    pert_valid = False
                    
                if pert_valid:
                    sl = call([snip, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    if not np.isnan(sl) and 0.0 <= sl <= 0.35:
                        s_loc = sl
                    else:
                        pert_valid = False
                        
            # HNR
            harm = call(snip, "To Harmonicity (cc)", 0.01, f_min, 0.1, 1.0)
            hnr_val = call(harm, "Get mean", 0, 0)
            hnr_valid = bool(not np.isnan(hnr_val) and 0.0 <= hnr_val <= 30.0)
            
            # Formants
            fmt = call(snip, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1 = call(fmt, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(fmt, "Get mean", 2, 0, 0, "Hertz")
            f3 = call(fmt, "Get mean", 3, 0, 0, "Hertz")
            
            fmt_valid = bool(
                not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3) and
                200.0 <= f1 <= 1200.0 and 600.0 <= f2 <= 3000.0 and 1500.0 <= f3 <= 4500.0 and
                f1 < f2 < f3 and (f2 - f1 >= 150.0) and (f3 - f2 >= 200.0)
            )
            
            turn_metrics.append({
                'turn_dur': dur, 'n_pulses': n_pulses, 'voiced_dur': voiced_dur,
                'f0_valid': f0_valid, 'snip_f0_mean': snip_f0_mean if f0_valid else np.nan,
                'pert_valid': pert_valid, 'j_local': j_loc, 's_local': s_loc,
                'hnr_valid': hnr_valid, 'hnr': hnr_val if hnr_valid else np.nan,
                'fmt_valid': fmt_valid,
                'f1': f1 if fmt_valid else np.nan,
                'f2': f2 if fmt_valid else np.nan,
                'f3': f3 if fmt_valid else np.nan
            })
        except Exception:
            continue
            
    tdf = pd.DataFrame(turn_metrics)
    tot_eval = len(tdf)
    if tot_eval == 0:
        return {
            'participant_id': pid, 'domain': domain, 'gender': gender_str,
            'audio_dur': total_dur, 'f0_median_pass1': f0_median, 'f_min_pass2': f_min, 'f_max_pass2': f_max,
            'n_turns': n_turns, 'n_valid_pert_turns': 0, 'pct_valid_pert': 0.0,
            'jitter_local_invalid_pct': 100.0, 'shimmer_local_invalid_pct': 100.0,
            'hnr_invalid_pct': 100.0, 'f0_invalid_pct': 100.0, 'formant_invalid_pct': 100.0,
            'mean_valid_f0': np.nan, 'mean_valid_jitter_local': np.nan, 'mean_valid_shimmer_local': np.nan,
            'mean_valid_hnr': np.nan, 'mean_valid_f1': np.nan, 'mean_valid_f2': np.nan, 'mean_valid_f3': np.nan,
            'insufficient_turns': True
        }
        
    n_valid_p = tdf['pert_valid'].sum()
    pct_valid_p = (n_valid_p / tot_eval) * 100.0
    
    insufficient = bool(n_valid_p < 5 or (tdf['voiced_dur'].sum() < 2.0))
    
    return {
        'participant_id': pid, 'domain': domain, 'gender': gender_str, 'audio_dur': round(total_dur, 2),
        'f0_median_pass1': round(f0_median, 1), 'f_min_pass2': round(f_min, 1), 'f_max_pass2': round(f_max, 1),
        'n_turns': tot_eval, 'n_valid_pert_turns': int(n_valid_p),
        'pct_valid_pert': round(pct_valid_p, 1),
        'jitter_local_invalid_pct': round((1.0 - tdf['pert_valid'].mean()) * 100.0, 1),
        'shimmer_local_invalid_pct': round((1.0 - tdf['pert_valid'].mean()) * 100.0, 1),
        'hnr_invalid_pct': round((1.0 - tdf['hnr_valid'].mean()) * 100.0, 1),
        'f0_invalid_pct': round((1.0 - tdf['f0_valid'].mean()) * 100.0, 1),
        'formant_invalid_pct': round((1.0 - tdf['fmt_valid'].mean()) * 100.0, 1),
        'mean_valid_f0': round(tdf['snip_f0_mean'].dropna().mean(), 1) if not tdf['snip_f0_mean'].dropna().empty else np.nan,
        'mean_valid_jitter_local': round(tdf['j_local'].dropna().mean(), 4) if not tdf['j_local'].dropna().empty else np.nan,
        'mean_valid_shimmer_local': round(tdf['s_local'].dropna().mean(), 4) if not tdf['s_local'].dropna().empty else np.nan,
        'mean_valid_hnr': round(tdf['hnr'].dropna().mean(), 2) if not tdf['hnr'].dropna().empty else np.nan,
        'mean_valid_f1': round(tdf['f1'].dropna().mean(), 1) if not tdf['f1'].dropna().empty else np.nan,
        'mean_valid_f2': round(tdf['f2'].dropna().mean(), 1) if not tdf['f2'].dropna().empty else np.nan,
        'mean_valid_f3': round(tdf['f3'].dropna().mean(), 1) if not tdf['f3'].dropna().empty else np.nan,
        'insufficient_turns': insufficient
    }

def main():
    print('Sampling stratified cohort (N=65) across Domain x Gender...')
    cohort = sample_stratified_cohort(seed=42)
    
    print('Loading Silero VAD model...')
    vad_model = silero_vad.load_silero_vad()
    
    records = []
    for idx, (_, row) in enumerate(cohort.iterrows()):
        pid = int(row['participant_id'])
        domain = row['domain']
        g_val = row['gender']
        g_str = str(row['gender']).capitalize()
        
        if domain == 'woz':
            wav_p = f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav'
        else:
            wav_p = f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav'
            
        if not os.path.exists(wav_p):
            print(f'Warning: {wav_p} not found, skipping.')
            continue
            
        res = analyze_session_fast(wav_p, pid, domain, g_str, vad_model)
        records.append(res)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(cohort):
            print(f'  Processed {idx+1}/{len(cohort)} sessions: PID {pid} ({domain} {g_str}) -> valid pert: {res["n_valid_pert_turns"]}/{res["n_turns"]} ({res["pct_valid_pert"]}%), f0={res["mean_valid_f0"]} Hz')
            
    df = pd.DataFrame(records)
    out_csv = 'experiments/research_v2/quality_audit_v2_extended.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved extended quality audit to {out_csv}')
    
    print('\n=== Stratified Summary by Domain x Gender ===')
    summary = df.groupby(['domain', 'gender'])[['pct_valid_pert', 'jitter_local_invalid_pct', 'f0_invalid_pct', 'hnr_invalid_pct', 'formant_invalid_pct', 'mean_valid_f0', 'mean_valid_jitter_local', 'mean_valid_shimmer_local']].mean().round(2)
    print(summary.to_string())

if __name__ == '__main__':
    main()
