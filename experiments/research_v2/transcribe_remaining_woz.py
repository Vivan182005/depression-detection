"""
Script: experiments/research_v2/transcribe_remaining_woz.py
Step S2: Transcribes remaining WoZ dev sessions with whisper-tiny using multiprocessing.
Standardized exact settings matching Avatar transcripts:
- Model: openai/whisper-tiny
- Greedy decoding: temperature=0.0
- chunk_length_s=30, batch_size=8
- language='english', task='transcribe'
- Multiprocessing start method: spawn
"""
import os
import sys
import time
import glob
import multiprocessing as mp
import pandas as pd
import soundfile as sf
import torch
from transformers import pipeline

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from test_label_blind_extraction import assert_sealed_quarantine, assert_label_blind

WOZ_WHISPER_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'woz_whisper')
os.makedirs(WOZ_WHISPER_DIR, exist_ok=True)

def worker_fn(worker_id, pid_list, num_threads=4):
    torch.set_num_threads(num_threads)
    device = 'cpu'
    pipe = pipeline(
        'automatic-speech-recognition',
        model='openai/whisper-tiny',
        chunk_length_s=30,
        batch_size=8,
        device=device
    )
    
    for i, (pid, audio_path) in enumerate(pid_list):
        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': 'woz'})

        out_csv = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 20:
            continue
            
        t0 = time.time()
        res = pipe(
            audio_path,
            generate_kwargs={'language': 'english', 'task': 'transcribe', 'temperature': 0.0}
        )
        text = res.get('text', '')
        
        # Atomically write to avoid race
        tmp_csv = f"{out_csv}.tmp_{worker_id}"
        df = pd.DataFrame([{
            'participant_id': pid,
            'text': text,
            'source_flag': 'whisper_tiny_local',
            'model': 'openai/whisper-tiny',
            'decoding': 'greedy',
            'temperature': 0.0,
            'condition_on_previous_text': False,
            'chunk_length_s': 30,
            'batch_size': 8,
            'language': 'english',
            'task': 'transcribe',
            'transformers_version': '5.19.0',
            'torch_version': '2.14.1'
        }])
        df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, out_csv)
        
        t_infer = time.time() - t0
        dur = sf.info(audio_path).duration
        print(f"[Worker {worker_id} | {i+1}/{len(pid_list)}] PID {pid} ({dur:.1f}s) in {t_infer:.2f}s ({dur/t_infer:.1f}x RT)")

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    manifest = pd.read_csv(os.path.join(_REPO_ROOT, 'experiments/research_v2/feature_store_v2/manifest.csv'))
    woz_dev = manifest[(manifest['domain'] == 'woz') & (~manifest['is_sealed']) & (manifest['feature_file'] != 'none')]
    assert_no_sealed_ids(woz_dev['participant_id'].values)

    done_files = glob.glob(os.path.join(WOZ_WHISPER_DIR, "*_TRANSCRIPT_WHISPER.csv"))
    done_pids = set(int(os.path.basename(f).split('_')[0]) for f in done_files if not f.endswith('.tmp') and os.path.getsize(f) > 20)
    
    remaining_tasks = []
    for _, r in woz_dev.iterrows():
        pid = int(r['participant_id'])
        if pid not in done_pids:
            af = os.path.join(_REPO_ROOT, r['source_audio_path'])
            remaining_tasks.append((pid, af))
            
    print(f"Total remaining WoZ dev sessions to transcribe: {len(remaining_tasks)}")
    if not remaining_tasks:
        print("All WoZ dev sessions already transcribed!")
        return

    num_workers = 3
    chunks = [remaining_tasks[i::num_workers] for i in range(num_workers)]
    
    processes = []
    for w_id in range(num_workers):
        p = mp.Process(target=worker_fn, args=(w_id, chunks[w_id], 2))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("All workers finished transcription!")

if __name__ == '__main__':
    main()
