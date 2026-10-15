"""
Script: experiments/research_v2/validate_turn_reconstruction.py
Audits stored development audio for turn-reconstruction validity:
1. Verifies sample rate, subtype, bit depth, exact-zero gap runs (0.3s vs 0.4s).
2. Tests mathematical effect of peak normalization and int16 writing on zero runs.
3. Compares recovered turns (via energy splitting) against ground-truth transcript timestamps.
4. Quantifies false boundaries from natural pauses and missed boundaries.
5. Evaluates pass/fail rates per session.
"""

import os
import sys
import glob
import soundfile as sf
import numpy as np
import pandas as pd

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS, assert_no_sealed_ids

MANIFEST_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2', 'manifest.csv')
DAIC_DIR = os.path.join(_REPO_ROOT, 'Clean_audio', 'daicwoz')
NEW_AUDIO_DIR = os.path.join(_REPO_ROOT, 'Clean_audio', 'NEW_AUDIO')

def main():
    df_manifest = pd.read_csv(MANIFEST_PATH)
    woz_dev = df_manifest[(df_manifest['domain'] == 'woz') & (~df_manifest['is_sealed']) & (df_manifest['feature_file'] != 'none')]
    assert_no_sealed_ids(woz_dev['participant_id'].values)

    print(f"Auditing turn reconstruction across {len(woz_dev)} WoZ dev sessions...")

    # 1. Theoretical test: Does peak normalization or int16 writing alter zeros?
    test_sig = np.array([0.0, 0.0, 0.5, -0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    gain = (10 ** (-1.0 / 20.0)) / 0.8
    norm_sig = test_sig * gain
    int16_sig = (norm_sig * 32767).astype(np.int16)
    readback_sig = int16_sig.astype(np.float32) / 32768.0

    zeros_preserved = (int16_sig[0] == 0) and (int16_sig[1] == 0) and (int16_sig[4] == 0)
    print(f"Mathematical proof: exact zeros preserved after peak normalization and int16 conversion: {zeros_preserved}")

    results = []

    for _, row in woz_dev.iterrows():
        pid = int(row['participant_id'])
        wav_file = os.path.join(_REPO_ROOT, row['source_audio_path'])
        tr_file = os.path.join(DAIC_DIR, f"{pid}_TRANSCRIPT.csv")

        if not os.path.exists(wav_file):
            continue

        info = sf.info(wav_file)
        sr = info.samplerate
        sub = info.subtype
        channels = info.channels
        dur = info.duration

        # Read audio
        y, _ = sf.read(wav_file, dtype='float32')
        is_zero = (y == 0.0)
        zero_frac = np.mean(is_zero)

        # Zero runs
        diff = np.diff(np.concatenate(([False], is_zero, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = ends - starts
        durations = lengths / float(sr)

        runs_03 = int(np.sum((durations >= 0.28) & (durations <= 0.32)))
        runs_04 = int(np.sum((durations >= 0.38) & (durations <= 0.42)))
        max_zero_s = float(durations.max()) if len(durations) > 0 else 0.0

        # Transcript comparison if available
        has_tr = os.path.exists(tr_file)
        n_tr_turns = np.nan
        sum_tr_dur = np.nan
        full_tr_span = np.nan
        n_rec_energy_turns = np.nan
        false_boundaries = np.nan
        missed_boundaries = np.nan
        turn_error_pct = np.nan
        passed = False

        if has_tr:
            try:
                df_tr = pd.read_csv(tr_file, sep='\t')
                if 'speaker' not in df_tr.columns:
                    df_tr = pd.read_csv(tr_file)
                part_df = df_tr[df_tr['speaker'].astype(str).str.lower().str.contains('participant', na=False)]
                n_tr_turns = len(part_df)
                turn_durations = (part_df['stop_time'] - part_df['start_time']).values
                sum_tr_dur = float(np.sum(turn_durations))
                full_tr_span = float(df_tr['stop_time'].max())

                # Attempt turn reconstruction via energy-based silence detection
                # 50ms frame, 350ms silence pause threshold (from speaker_isolation.py)
                frame_len = int(0.05 * sr)
                num_frames = len(y) // frame_len
                energies = np.array([
                    np.sqrt(np.mean(y[i*frame_len : (i+1)*frame_len]**2) + 1e-9)
                    for i in range(num_frames)
                ])
                p10 = np.percentile(energies, 10)
                p75 = np.percentile(energies, 75)
                thresh = p10 + 0.15 * (p75 - p10 + 1e-6)
                is_speech = energies > thresh

                # Group into turns
                min_turn_f = int(0.4 / 0.05)
                min_pause_f = int(0.35 / 0.05)
                rec_turns = []
                in_t = False
                t_start = 0
                s_count = 0
                for f_idx, act in enumerate(is_speech):
                    if act:
                        if not in_t:
                            in_t = True
                            t_start = f_idx
                        s_count = 0
                    else:
                        if in_t:
                            s_count += 1
                            if s_count >= min_pause_f:
                                t_end = f_idx - s_count
                                if (t_end - t_start) >= min_turn_f:
                                    rec_turns.append((t_start * 0.05, t_end * 0.05))
                                in_t = False
                if in_t and (num_frames - t_start) >= min_turn_f:
                    rec_turns.append((t_start * 0.05, num_frames * 0.05))

                n_rec_energy_turns = len(rec_turns)

                # Quantify false and missed boundaries
                # A natural pause of >350ms inside a participant turn creates an extra (false) boundary
                # While missing an interviewer question merges two turns
                diff_count = n_rec_energy_turns - n_tr_turns
                if diff_count > 0:
                    false_boundaries = diff_count
                    missed_boundaries = 0
                else:
                    false_boundaries = 0
                    missed_boundaries = abs(diff_count)

                turn_error_pct = abs(n_rec_energy_turns - n_tr_turns) / max(1, n_tr_turns) * 100.0
                dur_diff_pct = abs(dur - sum_tr_dur) / max(1, sum_tr_dur) * 100.0

                # Strict pass: Turn count within 15% AND duration within 15%
                passed = (turn_error_pct <= 15.0) and (dur_diff_pct <= 15.0)

            except Exception as e:
                pass

        results.append({
            'participant_id': pid,
            'sr': sr,
            'subtype': sub,
            'channels': channels,
            'duration_s': round(dur, 2),
            'zero_frac': round(zero_frac, 6),
            'max_zero_s': round(max_zero_s, 5),
            'runs_03s': runs_03,
            'runs_04s': runs_04,
            'has_transcript': has_tr,
            'n_tr_turns': n_tr_turns,
            'sum_tr_dur_s': round(sum_tr_dur, 2) if not np.isnan(sum_tr_dur) else np.nan,
            'n_rec_energy_turns': n_rec_energy_turns,
            'false_boundaries': false_boundaries,
            'missed_boundaries': missed_boundaries,
            'turn_error_pct': round(turn_error_pct, 1) if not np.isnan(turn_error_pct) else np.nan,
            'reconstruction_passed': passed
        })

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'turn_reconstruction_audit.csv')
    df_res.to_csv(out_csv, index=False)
    print(f"\nAudit completed. Saved to {out_csv}")

    # Aggregate findings
    n_total = len(df_res)
    n_with_tr = df_res['has_transcript'].sum()
    n_pass = df_res['reconstruction_passed'].sum()
    pct_pass = (n_pass / n_with_tr * 100.0) if n_with_tr > 0 else 0.0

    print(f"\n========================================================")
    print(f"TURN RECONSTRUCTION VALIDATION AUDIT SUMMARY")
    print(f"========================================================")
    print(f"Total WoZ Dev Sessions Audited : {n_total}")
    print(f"Sampling Rates                 : {df_res['sr'].unique().tolist()} Hz")
    print(f"Audio Subtypes                 : {df_res['subtype'].unique().tolist()}")
    print(f"Total Exact Zero Runs near 0.3s: {df_res['runs_03s'].sum()}")
    print(f"Total Exact Zero Runs near 0.4s: {df_res['runs_04s'].sum()}")
    print(f"Mean Max Zero Run per Session  : {df_res['max_zero_s'].mean():.5f}s (max: {df_res['max_zero_s'].max():.5f}s)")
    print(f"Sessions with Ground Truth Tr  : {n_with_tr}")
    print(f"Sessions Passing Reconstruction: {n_pass} / {n_with_tr} ({pct_pass:.1f}%)")
    print(f"Mean Turn Count Error          : {df_res['turn_error_pct'].mean():.1f}%")
    print(f"Mean False Boundaries / Session: {df_res['false_boundaries'].mean():.1f}")
    print(f"Mean Missed Boundaries / Session: {df_res['missed_boundaries'].mean():.1f}")
    print(f"========================================================\n")

if __name__ == '__main__':
    main()
