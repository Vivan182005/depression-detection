"""
Script: experiments/research_v2/transcribe_vad_chunked.py
Amendment v1.3c (S3 gate remediation): VAD-chunked Whisper-tiny transcription.

Applied UNIFORMLY to all 239 active development sessions (WoZ + Avatar), label-blind.
- Speech regions come from the frozen Silero VAD parameters of Amendment v1.2
  (threshold 0.50, min speech 250 ms, min silence 200 ms).
- Adjacent VAD segments are packed into chunks of <= 28 s (gap between packed
  segments <= 2.0 s is kept as-is; longer gaps start a new chunk), so Whisper
  never decodes long stretches of silence (the trigger of autoregressive loops).
- Each chunk is decoded independently (no conditioning on previous text),
  greedy decoding, openai/whisper-tiny, language='english', task='transcribe'.
- A deterministic loop-collapse filter (collapse_repeats) is applied to every
  transcript identically: any n-gram (n = 1..10) repeated consecutively more
  than 3 times is truncated to 3 repetitions.

Output: transcripts_v2/vad_chunked/{pid}_TRANSCRIPT_WHISPER_VAD.csv
"""
import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids
from test_label_blind_extraction import assert_sealed_quarantine, assert_label_blind

OUT_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'vad_chunked')
MANIFEST_V3_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3', 'manifest_v3.csv')

MAX_CHUNK_S = 28.0
MAX_JOIN_GAP_S = 2.0
MAX_CONSECUTIVE_REPEATS = 3


def audio_path_for(pid: int, domain: str) -> str:
    if domain == 'woz':
        return os.path.join(_REPO_ROOT, f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav')
    return os.path.join(_REPO_ROOT, f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav')


def collapse_repeats(words, max_n=10, max_reps=MAX_CONSECUTIVE_REPEATS):
    """Truncate any n-gram repeated consecutively more than max_reps times to max_reps copies."""
    words = list(words)
    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            i = 0
            out = []
            while i < len(words):
                gram = words[i:i + n]
                if len(gram) < n:
                    out.extend(gram)
                    break
                reps = 1
                while words[i + reps * n:i + (reps + 1) * n] == gram:
                    reps += 1
                if reps > max_reps:
                    out.extend(gram * max_reps)
                    changed = True
                    i += reps * n
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return words


def pack_segments(segs, sr):
    """Pack VAD segments into chunks of at most MAX_CHUNK_S seconds.
    Returns a list of (start, end, member_segments) tuples in samples."""
    max_len = int(MAX_CHUNK_S * sr)
    max_gap = int(MAX_JOIN_GAP_S * sr)
    chunks = []
    cur = None
    for s in segs:
        st, en = s['start'], s['end']
        if cur is None:
            cur = [st, en, [(st, en)]]
        elif (st - cur[1]) <= max_gap and (en - cur[0]) <= max_len:
            cur[1] = en
            cur[2].append((st, en))
        else:
            chunks.append(tuple(cur))
            cur = [st, en, [(st, en)]]
    if cur is not None:
        chunks.append(tuple(cur))
    return chunks


def has_loop(text):
    words = text.split()
    return len(collapse_repeats(words)) < len(words)


def worker_fn(worker_id, tasks, num_threads, overwrite):
    torch.set_num_threads(num_threads)
    import silero_vad
    from transformers import pipeline
    from extract_features_v3 import get_silero_vad_segments

    vad_model = silero_vad.load_silero_vad()
    pipe = pipeline('automatic-speech-recognition', model='openai/whisper-tiny', device='cpu')

    for i, (pid, domain) in enumerate(tasks):
        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': domain})
        out_csv = os.path.join(OUT_DIR, f'{pid}_TRANSCRIPT_WHISPER_VAD.csv')
        if os.path.exists(out_csv) and not overwrite:
            continue

        t0 = time.time()
        y, sr = sf.read(audio_path_for(pid, domain))
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        y = y.astype(np.float32)

        segs = get_silero_vad_segments(y, sr, vad_model)
        chunks = pack_segments(segs, sr)
        gen_kwargs = {'language': 'english', 'task': 'transcribe', 'do_sample': False, 'num_beams': 1}

        def decode(spans):
            if not spans:
                return []
            res = pipe([{'raw': y[a:b], 'sampling_rate': sr} for a, b in spans],
                       batch_size=8, generate_kwargs=gen_kwargs)
            return [r.get('text', '').strip() for r in res]

        texts = decode([(a, b) for a, b, _ in chunks])
        # Deterministic fallback: a chunk whose decode loops is re-decoded
        # segment-by-segment (shorter context, no silence padding).
        n_redecoded = 0
        for ci, (a, b, members) in enumerate(chunks):
            if has_loop(texts[ci]) and len(members) > 1:
                texts[ci] = ' '.join(t for t in decode(members) if t)
                n_redecoded += 1

        raw_text = ' '.join(t for t in texts if t)
        words = raw_text.split()
        clean_words = collapse_repeats(words)
        clean_text = ' '.join(clean_words)

        tmp_csv = f'{out_csv}.tmp_{worker_id}'
        pd.DataFrame([{
            'participant_id': pid,
            'domain': domain,
            'text': clean_text,
            'text_raw': raw_text,
            'n_words_raw': len(words),
            'n_words_clean': len(clean_words),
            'n_vad_segments': len(segs),
            'n_chunks': len(chunks),
            'n_chunks_redecoded': n_redecoded,
            'source_flag': 'whisper_tiny_local_vad_chunked',
            'model': 'openai/whisper-tiny',
            'decoding': 'greedy',
            'max_chunk_s': MAX_CHUNK_S,
            'max_join_gap_s': MAX_JOIN_GAP_S,
            'max_consecutive_repeats': MAX_CONSECUTIVE_REPEATS,
            'vad': 'silero-v6.2.3 frozen v1.2 params',
        }]).to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, out_csv)
        print(f'[W{worker_id} {i + 1}/{len(tasks)}] PID {pid} ({domain}) chunks={len(chunks)} '
              f'redecoded={n_redecoded} words={len(words)}->{len(clean_words)} in {time.time() - t0:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', type=str, default='', help='comma-separated subset (default: all 239)')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_V3_PATH)
    assert len(manifest) == 239
    assert_no_sealed_ids(manifest['participant_id'].values)
    if args.pids:
        keep = {int(p) for p in args.pids.split(',')}
        manifest = manifest[manifest['participant_id'].isin(keep)]
    tasks = [(int(r.participant_id), str(r.domain)) for r in manifest.itertuples()]
    print(f'Transcribing {len(tasks)} sessions with {args.workers} workers', flush=True)

    mp.set_start_method('spawn', force=True)
    procs = []
    for w in range(args.workers):
        p = mp.Process(target=worker_fn, args=(w, tasks[w::args.workers], args.threads, args.overwrite))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise SystemExit(f'Worker failures: {bad}')
    print('Done.')


if __name__ == '__main__':
    main()
