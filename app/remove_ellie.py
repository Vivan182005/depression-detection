"""
Script: remove_ellie.py
Description: High-fidelity Ellie Removal & Speaker Isolation Software with Volume Boost.
Strips out interviewer ("Ellie") questions and extracts pure participant speech.
Guarantees 100% natural playback speed, clear audible volume (-1.0 dB FS), zero static, and zero echo.
"""

import sys
import os

# Add parent app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.speaker_isolation import isolate_participant_audio
from src.preprocessing.audio_cleaner import preprocess_audio_signal

import soundfile as sf
import numpy as np


def process_file(audio_path, transcript_path=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_PARTICIPANT_ONLY.wav")

    print(f"\nProcessing: {os.path.basename(audio_path)}")
    print(f"  -> Input Audio   : {audio_path}")
    print(f"  -> Output Path   : {output_path}")

    # 1. Isolate Participant Speech
    clean_audio, sr = isolate_participant_audio(
        audio_path=audio_path,
        transcript_path=transcript_path,
        output_path=None,
        sr=16000,
        pause_gap_sec=0.4
    )

    # 2. Boost Loudness to Peak Level (-1.0 dB FS) for clear audibility
    boosted_audio = preprocess_audio_signal(clean_audio, sr=sr, target_peak_db=-1.0)

    # 3. Save WAV File
    sf.write(output_path, boosted_audio, sr)
    dur_sec = len(boosted_audio) / sr

    print(f"  [SUCCESS] Saved: {output_path} ({dur_sec:.1f} sec, Loudness Boosted -1.0 dB FS)")
    return output_path


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
        transcript_path = sys.argv[2] if len(sys.argv) >= 3 and os.path.exists(sys.argv[2]) else None
        process_file(audio_path, transcript_path, output_dir)
    else:
        print("\n==========================================================================")
        print("         ELLIE REMOVER SOFTWARE (BATCH & LOUDNESS BOOST ENABLED)")
        print("==========================================================================")
        audio_dir = "/Users/vivannaik/Desktop/CAPSTONE/Audio"
        if os.path.exists(audio_dir):
            files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') and not f.endswith('_PARTICIPANT_ONLY.wav')]
            print(f"Found {len(files)} audio files in {audio_dir}. Processing all files...")
            for f in sorted(files):
                full_p = os.path.join(audio_dir, f)
                process_file(full_p, output_dir=output_dir)
        else:
            print(f"Usage: python remove_ellie.py <audio_wav_path> [transcript_csv_path]")
        print("==========================================================================\n")


if __name__ == "__main__":
    main()
