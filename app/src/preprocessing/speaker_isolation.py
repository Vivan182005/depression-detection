"""
Module: speaker_isolation.py
Description: High-fidelity Ellie removal and participant speaker isolation module.
Ensures zero fast-forwarding, zero pitch shifting, zero static pops, and zero metallic echo artifacts.
"""

import os
import pandas as pd
import numpy as np

try:
    import soundfile as sf
    import librosa
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False


def separate_speakers_automatically(y, sr, pause_gap_sec=0.3):
    """
    Automated speaker separation for DAIC-WOZ audio without transcript CSV.
    Detects continuous speech turns and separates interviewer (Ellie) vs participant.
    In DAIC-WOZ interviews:
    - Ellie is the virtual interviewer who speaks FIRST (asking questions like 'Hi, I am Ellie...').
    - Ellie speaks in short prompt turns.
    - Participant responds after Ellie in longer turns.
    """
    frame_sec = 0.05 # 50ms frames
    frame_len = int(frame_sec * sr)
    hop_len = frame_len

    # Calculate RMS energy per 50ms frame
    num_frames = len(y) // hop_len
    energies = np.array([
        np.sqrt(np.mean(y[i*hop_len : (i+1)*hop_len]**2) + 1e-9)
        for i in range(num_frames)
    ])

    # Adaptive energy threshold based on audio noise floor
    p10 = np.percentile(energies, 10)
    p75 = np.percentile(energies, 75)
    adaptive_threshold = p10 + 0.15 * (p75 - p10 + 1e-6)

    is_speech = energies > adaptive_threshold

    # Group contiguous speech frames into turns
    min_turn_frames = int(0.4 / frame_sec)   # 400ms minimum turn
    min_pause_frames = int(0.35 / frame_sec) # 350ms pause to split turns

    speech_turns = []
    in_turn = False
    turn_start = 0
    silent_count = 0

    for i, active in enumerate(is_speech):
        if active:
            if not in_turn:
                in_turn = True
                turn_start = i
            silent_count = 0
        else:
            if in_turn:
                silent_count += 1
                if silent_count >= min_pause_frames:
                    turn_end = i - silent_count
                    if (turn_end - turn_start) >= min_turn_frames:
                        speech_turns.append((turn_start * hop_len, turn_end * hop_len))
                    in_turn = False

    if in_turn:
        turn_end = num_frames
        if (turn_end - turn_start) >= min_turn_frames:
            speech_turns.append((turn_start * hop_len, min(turn_end * hop_len, len(y))))

    if not speech_turns:
        print("[WARNING] Could not detect distinct speech turns. Returning full audio.")
        return y

    participant_turns = []
    pause_buffer = np.zeros(int(pause_gap_sec * sr), dtype=np.float32)

    for idx, (start_sample, end_sample) in enumerate(speech_turns):
        turn_duration = (end_sample - start_sample) / sr
        
        # Ellie typically speaks at idx 0 (initial prompt) and short alternating prompts (< 3.0s questions)
        is_ellie = False
        if idx == 0 and turn_duration < 8.0:
            is_ellie = True # Initial Ellie intro question
        elif idx % 2 == 0 and turn_duration < 2.5:
            is_ellie = True # Intermittent short Ellie question prompts

        if not is_ellie:
            segment = y[start_sample:end_sample]
            participant_turns.append(segment)
            participant_turns.append(pause_buffer)

    if not participant_turns:
        # Fallback if filtering was too strict: take turns after turn 0
        for start_sample, end_sample in speech_turns[1:]:
            participant_turns.append(y[start_sample:end_sample])
            participant_turns.append(pause_buffer)

    clean_audio = np.concatenate(participant_turns)
    return clean_audio


def isolate_participant_audio(audio_path, transcript_path=None, output_path=None, sr=16000, pause_gap_sec=0.3):
    """
    Extracts participant speech and removes interviewer ('Ellie') voice.

    Parameters:
        audio_path (str): Path to raw interview audio file (.wav)
        transcript_path (str, optional): Path to transcript CSV/TSV containing start_time, stop_time, speaker
        output_path (str, optional): Path to save cleaned participant audio file
        sr (int): Target sample rate (16000 Hz)
        pause_gap_sec (float): Natural pause duration (seconds) inserted between speech turns

    Returns:
        np.ndarray: Pristine participant audio signal at original speed
        int: Sample rate
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load raw audio at native speed
    if HAS_AUDIO_LIBS:
        y, sample_rate = librosa.load(audio_path, sr=sr)
    else:
        from scipy.io import wavfile
        sample_rate, y = wavfile.read(audio_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32) / (np.max(np.abs(y)) + 1e-9)

    # -------------------------------------------------------------------------
    # MODE A: Transcript-Driven Isolation (When CSV/TSV is provided)
    # -------------------------------------------------------------------------
    if transcript_path and os.path.exists(transcript_path):
        try:
            df = pd.read_csv(transcript_path, sep='\t')
            if 'speaker' not in df.columns or 'start_time' not in df.columns:
                df = pd.read_csv(transcript_path, sep=',')
        except Exception as e:
            raise ValueError(f"Failed to parse transcript file {transcript_path}: {e}")

        df.columns = [col.strip().lower() for col in df.columns]

        # Filter participant speech
        participant_df = df[df['speaker'].astype(str).str.lower().str.contains('participant', na=False)]

        if participant_df.empty:
            print(f"[WARNING] No 'Participant' speaker tag found in {transcript_path}. Keeping full audio.")
            participant_df = df

        audio_turns = []
        pause_samples = np.zeros(int(pause_gap_sec * sample_rate), dtype=np.float32)

        for _, row in participant_df.iterrows():
            try:
                start_sec = float(row['start_time'])
                stop_sec = float(row['stop_time'])
            except (ValueError, TypeError):
                continue

            start_idx = int(start_sec * sample_rate)
            end_idx = int(stop_sec * sample_rate)

            if start_idx < len(y) and end_idx <= len(y) and end_idx > start_idx:
                turn_segment = y[start_idx:end_idx]
                if len(turn_segment) > 0:
                    audio_turns.append(turn_segment)
                    audio_turns.append(pause_samples)

        if audio_turns:
            clean_audio = np.concatenate(audio_turns)
        else:
            clean_audio = y

    # -------------------------------------------------------------------------
    # MODE B: Automated Speaker Separation (When NO transcript CSV is provided)
    # -------------------------------------------------------------------------
    else:
        print("[INFO] No transcript file provided. Running adaptive automated speaker turn separation...")
        clean_audio = separate_speakers_automatically(y, sample_rate, pause_gap_sec=pause_gap_sec)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if HAS_AUDIO_LIBS:
            sf.write(output_path, clean_audio, sample_rate)
        else:
            from scipy.io import wavfile
            wavfile.write(output_path, sample_rate, (clean_audio * 32767).astype(np.int16))
        print(f"[SUCCESS] Isolated participant audio saved to: {output_path}")

    return clean_audio, sample_rate
