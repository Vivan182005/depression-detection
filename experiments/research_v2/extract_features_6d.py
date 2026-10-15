"""
Script: experiments/research_v2/extract_features_6d.py
Phase 6D label-blind feature extraction (prereg_phase6d_emotion_mit.md sec 2).
  --stage audio : HuBERT-base-ER / HuBERT-large-ER all-layer means + MIT coordination eigenspectra
  --stage text  : emotion-english-roberta-large on v1.3c VAD-chunked Whisper transcripts
Output: feature_store_6d/{pid}_6d.npz (audio) and {pid}_6d_text.npz (text)
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
from transcribe_vad_chunked import audio_path_for

OUT_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_6d')
VAD_TX_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'vad_chunked')
MANIFEST_V3_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3', 'manifest_v3.csv')

N_DELAYS = 15
DELAY_SCALES = (1, 3, 7)
MIN_COORD_SEG_S = 1.0


def ssl_all_layer_mean(y, sr, segs, model, fe):
    seg_means, weights = [], []
    for s in segs:
        snip = y[s['start']:s['end']]
        if len(snip) < int(0.10 * sr):
            continue
        inputs = fe(snip, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            hs = model(inputs.input_values, output_hidden_states=True).hidden_states
        seg_means.append(np.stack([h.squeeze(0).mean(dim=0).numpy() for h in hs]))
        weights.append(hs[-1].shape[1])
    w = np.array(weights, dtype=np.float32)
    return np.einsum('s,sld->ld', w / w.sum(), np.array(seg_means, dtype=np.float32)).astype(np.float32)


class DelayCorr:
    """Accumulates the correlation matrix of time-delay-embedded multichannel tracks across segments."""

    def __init__(self, n_ch, scale):
        self.scale, self.dim = scale, n_ch * N_DELAYS
        self.n, self.s1, self.s2 = 0, np.zeros(self.dim), np.zeros((self.dim, self.dim))

    def add(self, X):  # X: (T, n_ch)
        span = (N_DELAYS - 1) * self.scale
        T = X.shape[0] - span
        if T < 10:
            return
        E = np.concatenate([X[d * self.scale:d * self.scale + T] for d in range(N_DELAYS)], axis=1)
        self.n += T
        self.s1 += E.sum(axis=0)
        self.s2 += E.T @ E

    def eigenspectrum(self):
        if self.n < 50:
            return np.full(self.dim, np.nan, dtype=np.float32)
        mu = self.s1 / self.n
        cov = self.s2 / self.n - np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / np.outer(sd, sd)
        return np.sort(np.linalg.eigvalsh(corr))[::-1].astype(np.float32)


def mit_coordination(y, sr, segs):
    import parselmouth
    form = [DelayCorr(3, s) for s in DELAY_SCALES]
    dmf = [DelayCorr(16, s) for s in DELAY_SCALES]
    for s in segs:
        snip = y[s['start']:s['end']]
        if len(snip) < int(MIN_COORD_SEG_S * sr):
            continue
        fm = parselmouth.Sound(snip.astype(np.float64), sampling_frequency=sr).to_formant_burg(
            time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
        times = fm.ts()
        F = np.array([[fm.get_value_at_time(k, t) for k in (1, 2, 3)] for t in times], dtype=np.float64)
        ok = np.isfinite(F).all(axis=1)
        if ok.mean() > 0.5:
            idx = np.arange(len(F))
            for k in range(3):
                F[:, k] = np.interp(idx, idx[ok], F[ok, k])
            for acc in form:
                acc.add(F)
        mf = librosa.feature.mfcc(y=snip, sr=sr, n_mfcc=16, n_fft=400, hop_length=160)
        D = librosa.feature.delta(mf).T
        for acc in dmf:
            acc.add(D)
    return (np.stack([a.eigenspectrum() for a in form]), np.stack([a.eigenspectrum() for a in dmf]))


def run_audio(manifest, threads):
    torch.set_num_threads(threads)
    import silero_vad
    from transformers import AutoFeatureExtractor, HubertModel
    from extract_features_v3 import get_silero_vad_segments
    vad = silero_vad.load_silero_vad()
    models = {}
    for key, name in [('hubert_base_er', 'superb/hubert-base-superb-er'),
                      ('hubert_large_er', 'superb/hubert-large-superb-er')]:
        models[key] = (HubertModel.from_pretrained(name).eval(), AutoFeatureExtractor.from_pretrained(name))
    for i, r in enumerate(manifest.itertuples()):
        pid, domain = int(r.participant_id), str(r.domain)
        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': domain})
        out = os.path.join(OUT_DIR, f'{pid}_6d.npz')
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
        feats = {k: ssl_all_layer_mean(y, sr, segs, m, fe) for k, (m, fe) in models.items()}
        f_eig, d_eig = mit_coordination(y, sr, segs)
        np.savez_compressed(out + '.tmp.npz', participant_id=np.int64(pid),
                            hubert_base_er_all_mean=feats['hubert_base_er'],
                            hubert_large_er_all_mean=feats['hubert_large_er'],
                            mit_formant_eig=f_eig, mit_dmfcc_eig=d_eig)
        os.replace(out + '.tmp.npz', out)
        print(f'[{i + 1}/{len(manifest)}] PID {pid} ({domain}) in {time.time() - t0:.1f}s', flush=True)


def run_text(manifest):
    from extractors_v2 import BertExtractorV2
    ext = BertExtractorV2(model_name='j-hartmann/emotion-english-roberta-large', device='cpu')
    for r in manifest.itertuples():
        pid = int(r.participant_id)
        assert_sealed_quarantine(pid)
        out = os.path.join(OUT_DIR, f'{pid}_6d_text.npz')
        if os.path.exists(out):
            continue
        text = str(pd.read_csv(os.path.join(VAD_TX_DIR, f'{pid}_TRANSCRIPT_WHISPER_VAD.csv')).iloc[0]['text'])
        emb = ext.extract_from_text(text)
        assert np.isfinite(emb).all()
        np.savez_compressed(out, participant_id=np.int64(pid), emo_roberta_large=emb.astype(np.float32))
    print('text done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['audio', 'text'], required=True)
    ap.add_argument('--pids', type=str, default='')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_V3_PATH)
    assert len(manifest) == 239
    assert_no_sealed_ids(manifest['participant_id'].values)
    if args.pids:
        keep = {int(p) for p in args.pids.split(',')}
        manifest = manifest[manifest['participant_id'].isin(keep)]
    if args.stage == 'audio':
        run_audio(manifest, args.threads)
    else:
        run_text(manifest)


if __name__ == '__main__':
    main()
