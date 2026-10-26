"""
Script: experiments/research_v2/extract_segments_phase8.py
Phase 8 (prereg_phase8_fewshot_augmentation.md): per-VAD-segment embeddings for all 272 participants.
  --pass clean : original audio
  --pass aug   : channel augmentation (20 dB SNR noise, +/-6 dB gain, band-pass 300-3400 Hz for half), PID-seeded
VAD boundaries always come from the clean audio. Output: feature_store_8/{pid}_{pass}.npz with
  w2v2_L6 (S, 768), hubert_large_er_L19 (S, 1024), hubert_large_er_L20 (S, 1024), frames (S,)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
from scipy.signal import butter, sosfiltfilt

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
RD = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
for p in [_REPO_ROOT, RD]:
    if p not in sys.path:
        sys.path.insert(0, p)

from transcribe_vad_chunked import audio_path_for
from run_sealed_evaluation import sealed_audio, LOCK

OUT_DIR = os.path.join(RD, 'feature_store_8')


def all_pids():
    import pandas as pd
    dev = pd.read_csv(os.path.join(RD, 'feature_store_v3', 'manifest_v3.csv'))[['participant_id', 'domain']]
    sealed = json.load(open(os.path.join(_REPO_ROOT, 'sealed_ids.json')))['sealed_pids']
    rows = [(int(p), d) for p, d in dev.itertuples(index=False)] + [(int(p), 'woz' if p < 600 else 'avatar') for p in sealed]
    return sorted(rows)


def load_audio(pid, domain, sealed):
    y, sr = sf.read(sealed_audio(pid) if pid in sealed else audio_path_for(pid, domain))
    if y.ndim > 1:
        y = y[:, 0]
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y.astype(np.float32), 16000


def augment(y, sr, pid):
    rng = np.random.RandomState(pid)
    y = y.copy()
    if rng.rand() < 0.5:
        y = sosfiltfilt(butter(4, [300, 3400], btype='bandpass', fs=sr, output='sos'), y).astype(np.float32)
    y = y * (10 ** (rng.uniform(-6, 6) / 20))
    p_sig = np.mean(y ** 2) + 1e-12
    y = y + rng.randn(len(y)).astype(np.float32) * np.sqrt(p_sig / (10 ** (20 / 10)))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def seg_layer_means(y, sr, segs, model, fe, layers):
    out = {l: [] for l in layers}
    frames = []
    for s in segs:
        snip = y[s['start']:s['end']]
        if len(snip) < int(0.10 * sr):
            continue
        inp = fe(snip, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            hs = model(inp.input_values, output_hidden_states=True).hidden_states
        for l in layers:
            out[l].append(hs[l].squeeze(0).mean(dim=0).numpy())
        frames.append(hs[-1].shape[1])
    return {l: np.array(v, dtype=np.float32) for l, v in out.items()}, np.array(frames, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pass', dest='which', choices=['clean', 'aug'], required=True)
    ap.add_argument('--pids', type=str, default='')
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    assert os.path.exists(LOCK), 'Phase 8 uses the formerly-sealed 33, allowed only after the one-shot sealed evaluation'
    torch.set_num_threads(a.threads)
    import silero_vad
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, HubertModel
    from extract_features_v3 import get_silero_vad_segments

    os.makedirs(OUT_DIR, exist_ok=True)
    sealed = set(json.load(open(os.path.join(_REPO_ROOT, 'sealed_ids.json')))['sealed_pids'])
    rows = all_pids()
    if a.pids:
        keep = {int(p) for p in a.pids.split(',')}
        rows = [r for r in rows if r[0] in keep]
    vad = silero_vad.load_silero_vad()
    w2v = (Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').eval(),
           Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h'))
    hub = (HubertModel.from_pretrained('superb/hubert-large-superb-er').eval(),
           AutoFeatureExtractor.from_pretrained('superb/hubert-large-superb-er'))
    for i, (pid, dom) in enumerate(rows):
        out = os.path.join(OUT_DIR, f'{pid}_{a.which}.npz')
        if os.path.exists(out):
            continue
        t0 = time.time()
        y, sr = load_audio(pid, dom, sealed)
        segs = get_silero_vad_segments(y, sr, vad)          # boundaries from clean audio
        if a.which == 'aug':
            y = augment(y, sr, pid)
        w, frames = seg_layer_means(y, sr, segs, *w2v, layers=[6])
        h, _ = seg_layer_means(y, sr, segs, *hub, layers=[19, 20])
        np.savez_compressed(out + '.tmp.npz', participant_id=np.int64(pid), domain=np.str_(dom), frames=frames,
                            w2v2_L6=w[6], hubert_large_er_L19=h[19], hubert_large_er_L20=h[20])
        os.replace(out + '.tmp.npz', out)
        print(f'[{i + 1}/{len(rows)}] {a.which} PID {pid} segs={len(frames)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
