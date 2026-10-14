"""
Quality Audit v2 (Pilot: 30 Dev Sessions per Domain = 60 Sessions Total)
Per-frame / per-turn acoustic validity analysis using Praat / Parselmouth.

Metrics:
- Two-pass f0 range from each session's own audio (no labels or gender)
- Octave-jump rejection
- Praat/Parselmouth jitter (local, rap, ppq5)
- Praat/Parselmouth shimmer (local, apq3, apq5, apq11)
- Praat/Parselmouth HNR
- Praat/Parselmouth formants (F1, F2, F3)
- Perturbation validity threshold: >= 0.3s voiced speech AND >= 30 pitch cycles per turn
- Invalid / missingness rates by participant and domain
- Output: experiments/research_v2/quality_audit_v2_pilot.csv
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

def analyze_session(wav_path, pid, domain, vad_model):
    # Load audio
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y[:, 0]
    total_dur = len(y) / sr
    
    # Ensure 16 kHz for VAD
    if sr != 16000:
        import librosa
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    else:
        y_16k = y
        
    # Parselmouth Sound object
    sound = parselmouth.Sound(wav_path)
    
    # Pass 1: Session-level pitch across whole audio (broad range 60-500 Hz)
    p1 = call(sound, "To Pitch", 0.0, 60.0, 500.0)
    f0_mean = call(p1, "Get mean", 0, 0, "Hertz")
    f0_median = call(p1, "Get quantile", 0, 0, 0.5, "Hertz")
    
    if np.isnan(f0_median) or f0_median < 50.0:
        f0_median = 150.0
        
    # Pass 2: Pitch range bounded to [0.5 * median, 2.0 * median] clamped to [50, 450]
    f_min = float(max(50.0, 0.5 * f0_median))
    f_max = float(min(450.0, 2.0 * f0_median))
    
    # Segment audio into speech turns using Silero VAD
    wav_tensor = torch.from_numpy(y_16k.astype('float32'))
    speech_timestamps = silero_vad.get_speech_timestamps(
        wav_tensor, vad_model, sampling_rate=16000,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200
    )
    
    n_turns = len(speech_timestamps)
    if n_turns == 0:
        return {
            'participant_id': pid, 'domain': domain, 'audio_dur': total_dur,
            'f0_median_pass1': f0_median, 'f_min_pass2': f_min, 'f_max_pass2': f_max,
            'n_turns': 0, 'n_valid_perturbation_turns': 0, 'pct_valid_perturbation': 0.0,
            'jitter_local_invalid_pct': 100.0, 'shimmer_local_invalid_pct': 100.0,
            'hnr_invalid_pct': 100.0, 'f0_invalid_pct': 100.0, 'formant_invalid_pct': 100.0,
            'mean_valid_f0': np.nan, 'mean_valid_jitter_local': np.nan,
            'mean_valid_jitter_rap': np.nan, 'mean_valid_jitter_ppq5': np.nan,
            'mean_valid_shimmer_local': np.nan, 'mean_valid_shimmer_apq3': np.nan,
            'mean_valid_shimmer_apq5': np.nan, 'mean_valid_shimmer_apq11': np.nan,
            'mean_valid_hnr': np.nan, 'mean_valid_f1': np.nan, 'mean_valid_f2': np.nan,
            'mean_valid_f3': np.nan, 'insufficient_turns': True
        }
        
    turn_metrics = []
    
    for seg in speech_timestamps:
        t_start = seg['start'] / 16000.0
        t_end = seg['end'] / 16000.0
        turn_dur = t_end - t_start
        
        if turn_dur < 0.10:
            continue
            
        try:
            snippet = sound.extract_part(from_time=t_start, to_time=t_end, preserve_times=True)
            # Pass 2 Pitch on snippet
            snip_pitch = call(snippet, "To Pitch", 0.0, f_min, f_max)
            snip_f0_mean = call(snip_pitch, "Get mean", 0, 0, "Hertz")
            
            # Octave jump check: check frame pitch differences
            n_p_frames = call(snip_pitch, "Get number of frames")
            octave_jump_count = 0
            prev_f = None
            for fi in range(1, min(n_p_frames + 1, 100)):
                vf = call(snip_pitch, "Get value in frame", fi, "Hertz")
                if not np.isnan(vf) and vf > 0:
                    if prev_f is not None and prev_f > 0:
                        jump = abs(np.log2(vf / prev_f))
                        if jump > 0.8:
                            octave_jump_count += 1
                    prev_f = vf
                    
            f0_valid = bool(not np.isnan(snip_f0_mean) and f_min <= snip_f0_mean <= f_max and octave_jump_count <= 2)
            
            # Point process for pulses
            pp = call([snippet, snip_pitch], "To PointProcess (cc)")
            n_pulses = call(pp, "Get number of points")
            
            # Estimated voiced duration
            voiced_dur = (n_pulses / snip_f0_mean) if (not np.isnan(snip_f0_mean) and snip_f0_mean > 0) else 0.0
            
            # Perturbation rule: >= 0.3s voiced speech AND >= 30 pulses
            pert_valid = bool(voiced_dur >= 0.30 and n_pulses >= 30 and f0_valid)
            
            j_local, j_rap, j_ppq5 = np.nan, np.nan, np.nan
            s_local, s_apq3, s_apq5, s_apq11 = np.nan, np.nan, np.nan, np.nan
            
            if pert_valid:
                # Jitter
                j_l = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                j_r = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                j_p = call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
                
                # Validity bounds: jitter local <= 0.12 (12%)
                if not np.isnan(j_l) and 0.0 <= j_l <= 0.12:
                    j_local, j_rap, j_ppq5 = j_l, j_r, j_p
                else:
                    pert_valid = False
                    
                # Shimmer
                if pert_valid:
                    s_l = call([snippet, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    s_3 = call([snippet, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    s_5 = call([snippet, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    s_11 = call([snippet, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    
                    if not np.isnan(s_l) and 0.0 <= s_l <= 0.35:
                        s_local, s_apq3, s_apq5, s_apq11 = s_l, s_3, s_5, s_11
                    else:
                        pert_valid = False
                        
            # HNR
            harm = call(snippet, "To Harmonicity (cc)", 0.01, f_min, 0.1, 1.0)
            hnr_val = call(harm, "Get mean", 0, 0)
            hnr_valid = bool(not np.isnan(hnr_val) and 0.0 <= hnr_val <= 30.0)
            
            # Formants
            fmt = call(snippet, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            f1 = call(fmt, "Get mean", 1, 0, 0, "Hertz")
            f2 = call(fmt, "Get mean", 2, 0, 0, "Hertz")
            f3 = call(fmt, "Get mean", 3, 0, 0, "Hertz")
            
            fmt_valid = bool(
                not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3) and
                200.0 <= f1 <= 1200.0 and 600.0 <= f2 <= 3000.0 and 1500.0 <= f3 <= 4500.0 and
                f1 < f2 < f3 and (f2 - f1 >= 150.0) and (f3 - f2 >= 200.0)
            )
            
            turn_metrics.append({
                'turn_dur': turn_dur, 'n_pulses': n_pulses, 'voiced_dur': voiced_dur,
                'f0_valid': f0_valid, 'snip_f0_mean': snip_f0_mean if f0_valid else np.nan,
                'pert_valid': pert_valid,
                'j_local': j_local, 'j_rap': j_rap, 'j_ppq5': j_ppq5,
                's_local': s_local, 's_apq3': s_apq3, 's_apq5': s_apq5, 's_apq11': s_apq11,
                'hnr_valid': hnr_valid, 'hnr': hnr_val if hnr_valid else np.nan,
                'fmt_valid': fmt_valid,
                'f1': f1 if fmt_valid else np.nan,
                'f2': f2 if fmt_valid else np.nan,
                'f3': f3 if fmt_valid else np.nan
            })
        except Exception as e:
            continue
            
    tdf = pd.DataFrame(turn_metrics)
    tot_evaluated = len(tdf)
    
    if tot_evaluated == 0:
        return {
            'participant_id': pid, 'domain': domain, 'audio_dur': total_dur,
            'f0_median_pass1': f0_median, 'f_min_pass2': f_min, 'f_max_pass2': f_max,
            'n_turns': n_turns, 'n_valid_perturbation_turns': 0, 'pct_valid_perturbation': 0.0,
            'jitter_local_invalid_pct': 100.0, 'shimmer_local_invalid_pct': 100.0,
            'hnr_invalid_pct': 100.0, 'f0_invalid_pct': 100.0, 'formant_invalid_pct': 100.0,
            'mean_valid_f0': np.nan, 'mean_valid_jitter_local': np.nan,
            'mean_valid_jitter_rap': np.nan, 'mean_valid_jitter_ppq5': np.nan,
            'mean_valid_shimmer_local': np.nan, 'mean_valid_shimmer_apq3': np.nan,
            'mean_valid_shimmer_apq5': np.nan, 'mean_valid_shimmer_apq11': np.nan,
            'mean_valid_hnr': np.nan, 'mean_valid_f1': np.nan, 'mean_valid_f2': np.nan,
            'mean_valid_f3': np.nan, 'insufficient_turns': True
        }
        
    n_valid_pert = tdf['pert_valid'].sum()
    pct_valid_pert = (n_valid_pert / tot_evaluated) * 100.0
    
    f0_inv_pct = (1.0 - tdf['f0_valid'].mean()) * 100.0
    j_inv_pct = (1.0 - tdf['pert_valid'].mean()) * 100.0
    s_inv_pct = (1.0 - tdf['pert_valid'].mean()) * 100.0
    hnr_inv_pct = (1.0 - tdf['hnr_valid'].mean()) * 100.0
    fmt_inv_pct = (1.0 - tdf['fmt_valid'].mean()) * 100.0
    
    insufficient = bool(n_valid_pert < 5 or (tdf['voiced_dur'].sum() < 2.0))
    
    return {
        'participant_id': pid, 'domain': domain, 'audio_dur': round(total_dur, 2),
        'f0_median_pass1': round(f0_median, 1), 'f_min_pass2': round(f_min, 1), 'f_max_pass2': round(f_max, 1),
        'n_turns': tot_evaluated, 'n_valid_perturbation_turns': int(n_valid_pert),
        'pct_valid_perturbation': round(pct_valid_pert, 1),
        'jitter_local_invalid_pct': round(j_inv_pct, 1),
        'shimmer_local_invalid_pct': round(s_inv_pct, 1),
        'hnr_invalid_pct': round(hnr_inv_pct, 1),
        'f0_invalid_pct': round(f0_inv_pct, 1),
        'formant_invalid_pct': round(fmt_inv_pct, 1),
        'mean_valid_f0': round(tdf['snip_f0_mean'].dropna().mean(), 1) if not tdf['snip_f0_mean'].dropna().empty else np.nan,
        'mean_valid_jitter_local': round(tdf['j_local'].dropna().mean(), 4) if not tdf['j_local'].dropna().empty else np.nan,
        'mean_valid_jitter_rap': round(tdf['j_rap'].dropna().mean(), 4) if not tdf['j_rap'].dropna().empty else np.nan,
        'mean_valid_jitter_ppq5': round(tdf['j_ppq5'].dropna().mean(), 4) if not tdf['j_ppq5'].dropna().empty else np.nan,
        'mean_valid_shimmer_local': round(tdf['s_local'].dropna().mean(), 4) if not tdf['s_local'].dropna().empty else np.nan,
        'mean_valid_shimmer_apq3': round(tdf['s_apq3'].dropna().mean(), 4) if not tdf['s_apq3'].dropna().empty else np.nan,
        'mean_valid_shimmer_apq5': round(tdf['s_apq5'].dropna().mean(), 4) if not tdf['s_apq5'].dropna().empty else np.nan,
        'mean_valid_shimmer_apq11': round(tdf['s_apq11'].dropna().mean(), 4) if not tdf['s_apq11'].dropna().empty else np.nan,
        'mean_valid_hnr': round(tdf['hnr'].dropna().mean(), 2) if not tdf['hnr'].dropna().empty else np.nan,
        'mean_valid_f1': round(tdf['f1'].dropna().mean(), 1) if not tdf['f1'].dropna().empty else np.nan,
        'mean_valid_f2': round(tdf['f2'].dropna().mean(), 1) if not tdf['f2'].dropna().empty else np.nan,
        'mean_valid_f3': round(tdf['f3'].dropna().mean(), 1) if not tdf['f3'].dropna().empty else np.nan,
        'insufficient_turns': insufficient
    }

def main():
    print('Loading Silero VAD model...')
    vad_model = silero_vad.load_silero_vad()
    
    # Load labels to select 30 dev sessions per domain
    labels_df = pd.read_csv('experiments/research_v2/labels_official.csv')
    
    woz_dev_pids = labels_df[(labels_df['domain'] == 'woz') & (labels_df['is_sealed'] == False) & (labels_df['participant_id'] != 306)]['participant_id'].tolist()[:30]
    avatar_dev_pids = labels_df[(labels_df['domain'] == 'avatar') & (labels_df['is_sealed'] == False)]['participant_id'].tolist()[:30]
    
    print(f'Selected {len(woz_dev_pids)} WoZ Dev sessions and {len(avatar_dev_pids)} Avatar Dev sessions (Total: 60)')
    
    results = []
    
    print('\nAuditing WoZ Dev sessions...')
    for i, pid in enumerate(woz_dev_pids):
        wav_path = f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav'
        if not os.path.exists(wav_path):
            continue
        res = analyze_session(wav_path, pid, 'woz_dev', vad_model)
        results.append(res)
        print(f"  [WoZ {i+1}/30] PID {pid}: {res['n_turns']} turns, valid pert: {res['n_valid_perturbation_turns']} ({res['pct_valid_perturbation']}%), f0: {res['mean_valid_f0']} Hz")
        
    print('\nAuditing Avatar Dev sessions...')
    for i, pid in enumerate(avatar_dev_pids):
        wav_path = f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav'
        if not os.path.exists(wav_path):
            continue
        res = analyze_session(wav_path, pid, 'avatar_dev', vad_model)
        results.append(res)
        print(f"  [Avatar {i+1}/30] PID {pid}: {res['n_turns']} turns, valid pert: {res['n_valid_perturbation_turns']} ({res['pct_valid_perturbation']}%), f0: {res['mean_valid_f0']} Hz")
        
    df = pd.DataFrame(results)
    out_csv = 'experiments/research_v2/quality_audit_v2_pilot.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nSaved Quality Audit v2 to {out_csv}')
    
    # Print domain-level summary
    print('\n=== Domain-Level Summary ===')
    summary = df.groupby('domain')[['pct_valid_perturbation', 'jitter_local_invalid_pct', 'shimmer_local_invalid_pct', 'hnr_invalid_pct', 'f0_invalid_pct', 'formant_invalid_pct', 'insufficient_turns']].mean()
    print(summary.to_string())

if __name__ == '__main__':
    main()
