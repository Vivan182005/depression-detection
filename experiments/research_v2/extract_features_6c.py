"""
Script: experiments/research_v2/extract_features_6c.py
Phase 6C label-blind feature extraction (see prereg_phase6c_exploratory.md sec 2).
- wav2vec2-base-960h: all 13 hidden states, frame-weighted mean + across-segment std of
  per-segment means, over the frozen v1.2 Silero VAD segments (identical to v3 V6-V8).
- whisper-base encoder: all 7 hidden states, mean over valid frames of packed <= 28 s chunks.
Output: feature_store_6c/{pid}_6c.npz
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
for p in [_REPO_ROOT, os.path.join(_REPO_ROOT, 'experiments', 'research_v2')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sealed_guard import assert_no_sealed_ids
from test_label_blind_extraction import assert_sealed_quarantine, assert_label_blind
from transcribe_vad_chunked import audio_path_for, pack_segments

OUT_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_6c')
MANIFEST_V3_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3', 'manifest_v3.csv')


def w2v2_features(y, sr, segs, model, fe):
    seg_means, weights = [], []
    for s in segs:
        snip = y[s['start']:s['end']]
        if len(snip) < int(0.10 * sr):
            continue
        inputs = fe(snip, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            hs = model(**inputs, output_hidden_states=True).hidden_states
        seg_means.append(np.stack([h.squeeze(0).mean(dim=0).numpy() for h in hs]))  # (13, 768)
        weights.append(hs[-1].shape[1])
    seg_means = np.array(seg_means, dtype=np.float32)  # (S, 13, 768)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    mean = np.einsum('s,sld->ld', w, seg_means).astype(np.float32)
    std = seg_means.std(axis=0).astype(np.float32)
    return mean, std


def whisper_features(y, sr, segs, model, fe):
    chunks = pack_segments(segs, sr)
    sums, n_tot = None, 0
    for a, b, _ in chunks:
        snip = y[a:b]
        feats = fe(snip, sampling_rate=16000, return_tensors='pt').input_features
        with torch.no_grad():
            hs = model(feats, output_hidden_states=True).hidden_states  # 7 x (1, 1500, 512)
        n_valid = max(1, min(1500, int(np.ceil(len(snip) / sr * 50))))
        layer_sums = np.stack([h[0, :n_valid].sum(dim=0).numpy() for h in hs])
        sums = layer_sums if sums is None else sums + layer_sums
        n_tot += n_valid
    return (sums / n_tot).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', type=str, default='')
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import silero_vad
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, WhisperFeatureExtractor, WhisperModel
    from extract_features_v3 import get_silero_vad_segments

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_V3_PATH)
    assert len(manifest) == 239
    assert_no_sealed_ids(manifest['participant_id'].values)
    if args.pids:
        keep = {int(p) for p in args.pids.split(',')}
        manifest = manifest[manifest['participant_id'].isin(keep)]

    vad = silero_vad.load_silero_vad()
    w_fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
    w_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').eval()
    wh_fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-base')
    wh_enc = WhisperModel.from_pretrained('openai/whisper-base').encoder.eval()

    for i, r in enumerate(manifest.itertuples()):
        pid, domain = int(r.participant_id), str(r.domain)
        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': domain})
        out = os.path.join(OUT_DIR, f'{pid}_6c.npz')
        if os.path.exists(out):
            continue
        t0 = time.time()
        y, sr = sf.read(audio_path_for(pid, domain))
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        y = y.astype(np.float32)
        segs = get_silero_vad_segments(y, sr, vad)
        w_mean, w_std = w2v2_features(y, sr, segs, w_model, w_fe)
        wh_mean = whisper_features(y, sr, segs, wh_enc, wh_fe)
        np.savez_compressed(out + '.tmp.npz', participant_id=np.int64(pid), w2v2_all_mean=w_mean,
                            w2v2_all_segstd=w_std, whisper_base_all_mean=wh_mean,
                            n_vad_segments=np.int64(len(segs)))
        os.replace(out + '.tmp.npz', out)
        print(f'[{i + 1}/{len(manifest)}] PID {pid} ({domain}) segs={len(segs)} in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
