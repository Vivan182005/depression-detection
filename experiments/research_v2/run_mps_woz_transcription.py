"""
Transcribes remaining WoZ dev sessions with whisper-tiny on Apple Silicon MPS.
"""
import os
import sys
import time
import glob
import pandas as pd
import soundfile as sf
import torch
from transformers import pipeline

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS, assert_no_sealed_ids
from experiments.research_v2.extractors_v2 import load_canonical_audio

WOZ_WHISPER_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'woz_whisper')
os.makedirs(WOZ_WHISPER_DIR, exist_ok=True)

def main():
    manifest = pd.read_csv(os.path.join(_REPO_ROOT, 'experiments/research_v2/feature_store_v2/manifest.csv'))
    woz_dev = manifest[(manifest['domain'] == 'woz') & (~manifest['is_sealed']) & (manifest['feature_file'] != 'none')]
    assert_no_sealed_ids(woz_dev['participant_id'].values)

    done_files = glob.glob(os.path.join(WOZ_WHISPER_DIR, "*.csv"))
    done_pids = set(int(os.path.basename(f).split('_')[0]) for f in done_files if not f.endswith('.tmp'))
    
    remaining = []
    for _, r in woz_dev.iterrows():
        pid = int(r['participant_id'])
        if pid not in done_pids:
            af = os.path.join(_REPO_ROOT, r['source_audio_path'])
            remaining.append((pid, af))
            
    print(f"Total WoZ dev: {len(woz_dev)}, Done: {len(done_pids)}, Remaining to transcribe: {len(remaining)}", flush=True)
    if not remaining:
        print("All WoZ dev sessions already transcribed!", flush=True)
        return

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading openai/whisper-tiny on {device}...", flush=True)
    pipe = pipeline(
        'automatic-speech-recognition',
        model='openai/whisper-tiny',
        chunk_length_s=30,
        batch_size=16,
        device=device
    )

    t0_all = time.time()
    for i, (pid, audio_path) in enumerate(remaining):
        out_csv = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 20:
            continue
            
        t0 = time.time()
        y, sr = load_canonical_audio(audio_path)
        res = pipe({'raw': y, 'sampling_rate': sr}, generate_kwargs={'language': 'english', 'task': 'transcribe'})
        text = res.get('text', '')
        
        tmp_csv = f"{out_csv}.tmp"
        df = pd.DataFrame([{'participant_id': pid, 'text': text, 'source_flag': 'whisper_tiny_woz_dev'}])
        df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, out_csv)
        
        t_infer = time.time() - t0
        dur = sf.info(audio_path).duration
        speed = dur / max(0.1, t_infer)
        elapsed = time.time() - t0_all
        rem_sec = (len(remaining) - (i + 1)) * (elapsed / (i + 1))
        print(f"[{i+1}/{len(remaining)}] WoZ PID {pid} ({dur:.1f}s) transcribed in {t_infer:.2f}s ({speed:.1f}x RT) | Remaining: {rem_sec/60:.1f}m", flush=True)

    print("All WoZ dev Whisper transcriptions completed!", flush=True)

if __name__ == '__main__':
    main()
