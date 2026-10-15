"""
Script: experiments/research_v2/extract_ssl_and_segments.py
Extracts Wav2Vec2 (L4, L8, L12), WavLM (L4, L8, L12), WavLM L8 segments, and Handcrafted segments
for all 239 eligible development participants in feature_store_v3.
Zero real labels touched. Sealed cohort strictly quarantined.
"""
import os
import sys
import time
import gc
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine
import extract_features_v3 as ef

CACHE_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3')
MANIFEST_PATH = os.path.join(CACHE_DIR, 'manifest_v3.csv')

def main():
    print("======================================================================")
    print("Phase 6A: Extract SSL (Wav2Vec2, WavLM) & Segment Features into Store v3")
    print("======================================================================")

    manifest = pd.read_csv(MANIFEST_PATH)
    assert len(manifest) == 239, f"Expected 239 manifest rows, got {len(manifest)}"
    assert_no_sealed_ids(manifest['participant_id'].values)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Compute Device: {device}")

    # Initialize models
    print("Loading models (Wav2Vec2-base-960h & WavLM-base-plus)...")
    t0 = time.time()
    w2v_fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
    w2v_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)
    w2v_model.eval()

    wavlm_fe = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus')
    wavlm_model = AutoModel.from_pretrained('microsoft/wavlm-base-plus').to(device)
    wavlm_model.eval()

    vad_model = ef.silero_vad.load_silero_vad()
    print(f"Models initialized in {time.time() - t0:.2f}s.")

    start_all = time.time()
    n_processed = 0

    for idx, row in manifest.iterrows():
        pid = int(row['participant_id'])
        domain = row['domain']

        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': str(domain)})

        npz_path = os.path.join(CACHE_DIR, f"{pid}_features.npz")
        assert os.path.exists(npz_path), f"Missing base feature file: {npz_path}"

        existing_data = dict(np.load(npz_path, allow_pickle=True))
        
        # Check if already complete
        required_new_keys = [
            'w2v2_layer_4_mean', 'w2v2_layer_8_mean', 'w2v2_layer_12_mean',
            'wavlm_layer_4_mean', 'wavlm_layer_8_mean', 'wavlm_layer_12_mean',
            'handcrafted_segments', 'wavlm_layer_8_segments'
        ]
        if all(k in existing_data for k in required_new_keys):
            continue

        t_start = time.time()

        if domain == 'woz':
            wav_path = os.path.join(_REPO_ROOT, f"Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav")
        else:
            wav_path = os.path.join(_REPO_ROOT, f"NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav")

        assert os.path.exists(wav_path), f"WAV not found: {wav_path}"

        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        total_dur = len(y) / sr
        session_rms = float(np.sqrt(np.mean(y**2)))

        segs = ef.get_silero_vad_segments(y, sr, vad_model)

        # 1. Handcrafted segment-level features (V12)
        seg_feats = []
        for s in segs:
            snip = y[s['start']:s['end']]
            f = ef.extract_segment_handcrafted(snip, sr, session_rms)
            if f is not None:
                seg_feats.append(f)
        
        if seg_feats:
            existing_data['handcrafted_segments'] = np.array(seg_feats, dtype=np.float32)
        else:
            existing_data['handcrafted_segments'] = np.zeros((0, 55), dtype=np.float32)

        # 2. Wav2Vec2 features (V6, V7, V8)
        w2v_feats = ef.extract_wav2vec2_features(y, sr, segs, w2v_model, w2v_fe, device)
        existing_data['w2v2_layer_4_mean'] = w2v_feats['w2v2_layer_4_mean']
        existing_data['w2v2_layer_8_mean'] = w2v_feats['w2v2_layer_8_mean']
        existing_data['w2v2_layer_12_mean'] = w2v_feats['w2v2_layer_12_mean']

        # 3. WavLM features (V9, V10, V11, V13)
        l4_segs, l8_segs, l12_segs, weights = [], [], [], []
        for s in segs:
            snip = y[s['start']:s['end']]
            if len(snip) < int(0.10 * sr):
                continue
            inputs = wavlm_fe(snip, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = wavlm_model(**inputs, output_hidden_states=True)
                h4 = outputs.hidden_states[4].squeeze(0).mean(dim=0).cpu().numpy()
                h8 = outputs.hidden_states[8].squeeze(0).mean(dim=0).cpu().numpy()
                h12 = outputs.hidden_states[12].squeeze(0).mean(dim=0).cpu().numpy()
                n_frames = outputs.hidden_states[12].shape[1]
                l4_segs.append(h4)
                l8_segs.append(h8)
                l12_segs.append(h12)
                weights.append(n_frames)

        if l12_segs:
            weights_arr = np.array(weights, dtype=np.float32)
            w_norm = weights_arr / (np.sum(weights_arr) if np.sum(weights_arr) > 0 else 1.0)
            existing_data['wavlm_layer_4_mean'] = np.sum(np.array(l4_segs) * w_norm[:, None], axis=0).astype(np.float32)
            existing_data['wavlm_layer_8_mean'] = np.sum(np.array(l8_segs) * w_norm[:, None], axis=0).astype(np.float32)
            existing_data['wavlm_layer_12_mean'] = np.sum(np.array(l12_segs) * w_norm[:, None], axis=0).astype(np.float32)
            existing_data['wavlm_layer_8_segments'] = np.array(l8_segs, dtype=np.float32)
        else:
            existing_data['wavlm_layer_4_mean'] = np.zeros(768, dtype=np.float32)
            existing_data['wavlm_layer_8_mean'] = np.zeros(768, dtype=np.float32)
            existing_data['wavlm_layer_12_mean'] = np.zeros(768, dtype=np.float32)
            existing_data['wavlm_layer_8_segments'] = np.zeros((0, 768), dtype=np.float32)

        # Atomically save
        tmp_npz = f"{npz_path}_tmp.npz"
        np.savez_compressed(tmp_npz, **existing_data)
        os.replace(tmp_npz, npz_path)

        n_processed += 1
        t_elapsed = time.time() - t_start
        tot_elapsed = time.time() - start_all
        avg_time = tot_elapsed / n_processed
        eta_min = (239 - (idx + 1)) * avg_time / 60.0

        print(f"[{idx+1}/239] PID {pid} ({domain}, {total_dur:.1f}s): Extracted in {t_elapsed:.2f}s | Avg: {avg_time:.2f}s | ETA: {eta_min:.1f}m")

    print("\n======================================================================")
    print(f"EXTRACTION COMPLETE: Processed {n_processed} sessions in {(time.time()-start_all)/60:.2f} minutes.")
    print("======================================================================")

if __name__ == '__main__':
    main()
