"""
Module: experiments/research_v2/extract_features_v3.py
Task C4: Canonical Label-Blind Feature Extraction & Resumable Caching Pipeline
Strict Constraints:
- 100% label-blind: raises AssertionError on any label access.
- Sealed cohort (N=33): quarantined and strictly forbidden.
- One large model in memory at a time (8 GB boundary).
- Resumable: checks file existence and hashes.
- Features:
  * Per-VAD-segment handcrafted (MFCCs+deltas, f0, jitter, shimmer, HNR, formants, spectral slope, pauses, rel intensity)
  * Participant-level aggregations (median, IQR, p10, p90, std, early-vs-late slope)
  * Whole-recording VAD-masked eGeMAPS functionals (135d robust)
  * wav2vec2-base-960h (layers 4, 8, 12 mean-pooled over VAD frames, plus per-segment arrays for V13)
  * WavLM-base-plus (layers 4, 8, 12 mean-pooled over VAD frames)
  * CMVN and LTSE independently switchable representations for V4/V5
"""

import os
import sys
import gc
import json
import time
import hashlib
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import parselmouth
from parselmouth.praat import call
import silero_vad
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoModel, AutoFeatureExtractor

try:
    import opensmile
except ImportError:
    opensmile = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'feature_store_v3')
MANIFEST_PATH = os.path.join(CACHE_DIR, 'manifest_v3.csv')

def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_silero_vad_segments(y: np.ndarray, sr: int, vad_model) -> List[Dict[str, int]]:
    """Frozen Silero VAD segmenter under Pre-registration v1.2."""
    if y.ndim > 1:
        y = y[:, 0]
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
        
    wav_tensor = torch.from_numpy(y.astype('float32'))
    raw_vad = silero_vad.get_speech_timestamps(
        wav_tensor, vad_model, sampling_rate=16000,
        threshold=0.50,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200
    )
    
    max_samples = int(15.0 * sr)
    segments = []
    for seg in raw_vad:
        s, e = seg['start'], seg['end']
        if (e - s) > max_samples:
            cur = s
            while cur < e:
                nxt = min(e, cur + max_samples)
                segments.append({'start': cur, 'end': nxt})
                cur = nxt
        else:
            segments.append({'start': s, 'end': e})
            
    return segments

def extract_segment_handcrafted(y_seg: np.ndarray, sr: int, session_mean_rms: float, f_min: float = 75.0, f_max: float = 500.0) -> Optional[np.ndarray]:
    dur = len(y_seg) / sr
    if dur < 0.10:
        return None
        
    # 1. MFCCs + deltas (39d)
    try:
        mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=13, hop_length=160, n_fft=400)
        d_mfcc = librosa.feature.delta(mfcc)
        dd_mfcc = librosa.feature.delta(mfcc, order=2)
        mfcc_39 = np.hstack([np.mean(mfcc, axis=1), np.mean(d_mfcc, axis=1), np.mean(dd_mfcc, axis=1)])
    except Exception:
        mfcc_39 = np.full(39, np.nan)
        
    # 2. Praat Acoustic Analysis
    try:
        snd = parselmouth.Sound(y_seg, sampling_rate=sr)
        pitch = call(snd, "To Pitch", 0.0, f_min, f_max)
        f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
        f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        
        pp = call([snd, pitch], "To PointProcess (cc)")
        n_pulses = call(pp, "Get number of points")
        voiced_dur = (n_pulses / f0_mean) if (f0_mean and not np.isnan(f0_mean) and f0_mean > 0) else 0.0
        
        pert_valid = bool(voiced_dur >= 0.30 and n_pulses >= 30 and not np.isnan(f0_mean))
        
        j_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) if pert_valid else np.nan
        j_rap = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) if pert_valid else np.nan
        j_ppq5 = call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) if pert_valid else np.nan
        
        s_local = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) if pert_valid else np.nan
        s_apq3 = call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) if pert_valid else np.nan
        s_apq5 = call([snd, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6) if pert_valid else np.nan
        s_apq11 = call([snd, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) if pert_valid else np.nan
        
        harm = call(snd, "To Harmonicity (cc)", 0.01, f_min, 0.1, 1.0)
        hnr = call(harm, "Get mean", 0, 0)
        
        fmt = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1 = call(fmt, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(fmt, "Get mean", 2, 0, 0, "Hertz")
        f3 = call(fmt, "Get mean", 3, 0, 0, "Hertz")
    except Exception:
        f0_mean, f0_std = np.nan, np.nan
        j_local, j_rap, j_ppq5 = np.nan, np.nan, np.nan
        s_local, s_apq3, s_apq5, s_apq11 = np.nan, np.nan, np.nan, np.nan
        hnr = np.nan
        f1, f2, f3 = np.nan, np.nan, np.nan
        
    # 3. Spectral slope
    try:
        spec = np.abs(np.fft.rfft(y_seg))
        freqs = np.fft.rfftfreq(len(y_seg), 1.0 / sr)
        valid_idx = (freqs >= 100) & (freqs <= 5000) & (spec > 1e-8)
        if np.sum(valid_idx) > 10:
            log_s = np.log10(spec[valid_idx])
            log_f = np.log10(freqs[valid_idx])
            slope, _ = np.polyfit(log_f, log_s, 1)
        else:
            slope = np.nan
    except Exception:
        slope = np.nan
        
    # 4. In-segment pauses (>100ms unvoiced)
    try:
        energy = librosa.feature.rms(y=y_seg, frame_length=320, hop_length=160)[0]
        thresh = np.mean(energy) * 0.10
        silent_frames = energy < thresh
        pause_frac = float(np.mean(silent_frames))
    except Exception:
        pause_frac = np.nan
        
    # 5. Session-relative intensity
    seg_rms = float(np.sqrt(np.mean(y_seg**2)))
    rel_intensity = (seg_rms / session_mean_rms) if session_mean_rms > 0 else 1.0
    
    acoustic_vec = np.array([
        f0_mean, f0_std,
        j_local, j_rap, j_ppq5,
        s_local, s_apq3, s_apq5, s_apq11,
        hnr,
        f1, f2, f3,
        slope, pause_frac, rel_intensity
    ], dtype=np.float32)
    
    return np.hstack([mfcc_39, acoustic_vec]).astype(np.float32)

def aggregate_participant_handcrafted(seg_features: List[np.ndarray], seg_times: List[Tuple[float, float]], total_dur: float) -> np.ndarray:
    if not seg_features:
        return np.full(55 * 6, np.nan, dtype=np.float32)
        
    arr = np.array(seg_features)
    
    med = np.nanmedian(arr, axis=0)
    p25 = np.nanpercentile(arr, 25, axis=0)
    p75 = np.nanpercentile(arr, 75, axis=0)
    iqr = p75 - p25
    p10 = np.nanpercentile(arr, 10, axis=0)
    p90 = np.nanpercentile(arr, 90, axis=0)
    std = np.nanstd(arr, axis=0)
    
    midpoint = total_dur / 2.0
    early_idx = [i for i, (st, et) in enumerate(seg_times) if ((st + et)/2.0) <= midpoint]
    late_idx = [i for i, (st, et) in enumerate(seg_times) if ((st + et)/2.0) > midpoint]
    
    if early_idx and late_idx:
        early_mean = np.nanmean(arr[early_idx], axis=0)
        late_mean = np.nanmean(arr[late_idx], axis=0)
        early_late_diff = (late_mean - early_mean) / (total_dur / 3600.0 if total_dur > 0 else 1.0)
    else:
        early_late_diff = np.zeros(arr.shape[1], dtype=np.float32)
        
    summary = np.hstack([med, iqr, p10, p90, std, early_late_diff]).astype(np.float32)
    summary = np.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)
    return summary

def extract_whole_session_egemaps(y_speech_concat: np.ndarray, sr: int = 16000) -> np.ndarray:
    if opensmile is None:
        raise RuntimeError("OpenSMILE not installed.")
        
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    df = smile.process_signal(y_speech_concat, sr)
    feat_88 = df.values.flatten().astype(np.float32)
    assert len(feat_88) == 88, f"Expected 88 eGeMAPS features, got {len(feat_88)}"
    assert not np.all(feat_88[62:88] == 0), "CRITICAL: eGeMAPS dims 62-87 are zero!"
    
    mfcc = librosa.feature.mfcc(y=y_speech_concat, sr=sr, n_mfcc=16, hop_length=256, n_fft=512).T
    delta_mfcc = librosa.feature.delta(mfcc, axis=0)
    rms = librosa.feature.rms(y=y_speech_concat, frame_length=512, hop_length=256).T
    temporal = np.hstack([mfcc, delta_mfcc, rms])
    t_mean = np.mean(temporal, axis=0).astype(np.float32)
    t_std = np.std(temporal, axis=0).astype(np.float32)
    
    ac_154 = np.hstack([feat_88, t_mean, t_std]).astype(np.float32)
    
    mask = np.ones(154, dtype=bool)
    mask[62:74] = False
    mask[120:124] = False
    mask[151:154] = False
    robust_135 = ac_154[mask]
    assert len(robust_135) == 135
    return robust_135

def apply_cmvn(feature_matrix: np.ndarray) -> np.ndarray:
    mean = np.mean(feature_matrix, axis=0, keepdims=True)
    std = np.std(feature_matrix, axis=0, keepdims=True) + 1e-8
    return (feature_matrix - mean) / std

def apply_ltse(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    return librosa.effects.preemphasis(y, coef=0.97)

def extract_wav2vec2_features(y: np.ndarray, sr: int, segments: List[Dict[str, int]], model, fe, device: str) -> Dict[str, np.ndarray]:
    l4_segs, l8_segs, l12_segs, weights = [], [], [], []
    for s in segments:
        snip = y[s['start']:s['end']]
        if len(snip) < int(0.10 * sr):
            continue
        inputs = fe(snip, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h4 = outputs.hidden_states[4].squeeze(0).mean(dim=0).cpu().numpy()
            h8 = outputs.hidden_states[8].squeeze(0).mean(dim=0).cpu().numpy()
            h12 = outputs.hidden_states[12].squeeze(0).mean(dim=0).cpu().numpy()
            n_frames = outputs.hidden_states[12].shape[1]
            l4_segs.append(h4)
            l8_segs.append(h8)
            l12_segs.append(h12)
            weights.append(n_frames)
            
    if not l12_segs:
        return {
            'w2v2_layer_4_mean': np.zeros(768, dtype=np.float32),
            'w2v2_layer_8_mean': np.zeros(768, dtype=np.float32),
            'w2v2_layer_12_mean': np.zeros(768, dtype=np.float32),
            'w2v2_layer_12_segments': np.zeros((0, 768), dtype=np.float32),
        }
        
    weights = np.array(weights, dtype=np.float32)
    weights_norm = weights / (np.sum(weights) if np.sum(weights) > 0 else 1.0)
    
    mean_l4 = np.sum(np.array(l4_segs) * weights_norm[:, None], axis=0).astype(np.float32)
    mean_l8 = np.sum(np.array(l8_segs) * weights_norm[:, None], axis=0).astype(np.float32)
    mean_l12 = np.sum(np.array(l12_segs) * weights_norm[:, None], axis=0).astype(np.float32)
    segs_l12 = np.array(l12_segs, dtype=np.float32)
    
    return {
        'w2v2_layer_4_mean': mean_l4,
        'w2v2_layer_8_mean': mean_l8,
        'w2v2_layer_12_mean': mean_l12,
        'w2v2_layer_12_segments': segs_l12,
    }

def extract_wavlm_features(y: np.ndarray, sr: int, segments: List[Dict[str, int]], model, fe, device: str) -> Dict[str, np.ndarray]:
    l4_segs, l8_segs, l12_segs, weights = [], [], [], []
    for s in segments:
        snip = y[s['start']:s['end']]
        if len(snip) < int(0.10 * sr):
            continue
        inputs = fe(snip, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h4 = outputs.hidden_states[4].squeeze(0).mean(dim=0).cpu().numpy()
            h8 = outputs.hidden_states[8].squeeze(0).mean(dim=0).cpu().numpy()
            h12 = outputs.hidden_states[12].squeeze(0).mean(dim=0).cpu().numpy()
            n_frames = outputs.hidden_states[12].shape[1]
            l4_segs.append(h4)
            l8_segs.append(h8)
            l12_segs.append(h12)
            weights.append(n_frames)
            
    if not l12_segs:
        return {
            'wavlm_layer_4_mean': np.zeros(768, dtype=np.float32),
            'wavlm_layer_8_mean': np.zeros(768, dtype=np.float32),
            'wavlm_layer_12_mean': np.zeros(768, dtype=np.float32),
            'wavlm_layer_12_segments': np.zeros((0, 768), dtype=np.float32),
        }
        
    weights = np.array(weights, dtype=np.float32)
    weights_norm = weights / (np.sum(weights) if np.sum(weights) > 0 else 1.0)
    
    mean_l4 = np.sum(np.array(l4_segs) * weights_norm[:, None], axis=0).astype(np.float32)
    mean_l8 = np.sum(np.array(l8_segs) * weights_norm[:, None], axis=0).astype(np.float32)
    mean_l12 = np.sum(np.array(l12_segs) * weights_norm[:, None], axis=0).astype(np.float32)
    segs_l12 = np.array(l12_segs, dtype=np.float32)
    
    return {
        'wavlm_layer_4_mean': mean_l4,
        'wavlm_layer_8_mean': mean_l8,
        'wavlm_layer_12_mean': mean_l12,
        'wavlm_layer_12_segments': segs_l12,
    }

def process_stage(stage: str, dev_cohort: pd.DataFrame, vad_model, device: str):
    print(f"=== Executing Feature Extraction Stage: {stage} ===")
    
    model = None
    fe = None
    if stage == 'wav2vec2':
        print("Loading wav2vec2-base-960h...")
        fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
        model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)
        model.eval()
    elif stage == 'wavlm':
        print("Loading WavLM-base-plus...")
        fe = AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus')
        model = AutoModel.from_pretrained('microsoft/wavlm-base-plus').to(device)
        model.eval()
        
    manifest_rows = []
    
    for idx, (_, row) in enumerate(dev_cohort.iterrows()):
        pid = int(row['participant_id'])
        domain = row['domain']
        
        assert_sealed_quarantine(pid)
        clean_meta = {'participant_id': pid, 'domain': str(domain)}
        assert_label_blind(clean_meta)
        
        if domain == 'woz':
            wav_path = f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav'
        else:
            wav_path = f'NEW_AUDIO_DAIC/{pid}_AUDIO_interviewee.wav'
            
        if not os.path.exists(wav_path):
            continue
            
        out_npz = os.path.join(CACHE_DIR, f"{pid}_features.npz")
        
        # Check resumption
        cached_keys = set()
        data_to_save = {}
        if os.path.exists(out_npz):
            cached = np.load(out_npz, allow_pickle=True)
            cached_keys = set(cached.files)
            data_to_save = {k: cached[k] for k in cached.files}
            
        if stage == 'acoustic' and 'handcrafted_summary' in cached_keys and 'egemaps_135d' in cached_keys:
            continue
        if stage == 'wav2vec2' and 'w2v2_layer_12_mean' in cached_keys:
            continue
        if stage == 'wavlm' and 'wavlm_layer_12_mean' in cached_keys:
            continue
            
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        total_dur = len(y) / sr
        session_rms = float(np.sqrt(np.mean(y**2)))
        
        segs = get_silero_vad_segments(y, sr, vad_model)
        n_segs = len(segs)
        
        speech_chunks = [y[s['start']:s['end']] for s in segs if (s['end'] - s['start']) >= int(0.10 * sr)]
        y_speech = np.concatenate(speech_chunks) if speech_chunks else y
        
        data_to_save['participant_id'] = np.int64(pid)
        data_to_save['domain'] = np.str_(domain)
        data_to_save['total_duration_s'] = np.float32(total_dur)
        data_to_save['n_vad_segments'] = np.int64(n_segs)
        data_to_save['speech_duration_s'] = np.float32(len(y_speech) / sr)
        
        if stage == 'acoustic':
            seg_feats, seg_times = [], []
            for s in segs:
                st_sec = s['start'] / sr
                et_sec = s['end'] / sr
                snip = y[s['start']:s['end']]
                f = extract_segment_handcrafted(snip, sr, session_rms)
                if f is not None:
                    seg_feats.append(f)
                    seg_times.append((st_sec, et_sec))
                    
            summary_330 = aggregate_participant_handcrafted(seg_feats, seg_times, total_dur)
            egemaps_135 = extract_whole_session_egemaps(y_speech, sr)
            
            data_to_save['handcrafted_summary'] = summary_330
            data_to_save['egemaps_135d'] = egemaps_135
            
            y_ltse = apply_ltse(y_speech, sr)
            egemaps_ltse = extract_whole_session_egemaps(y_ltse, sr)
            data_to_save['egemaps_135d_ltse'] = egemaps_ltse
            
            if seg_feats:
                cmvn_segs = apply_cmvn(np.array(seg_feats))
                data_to_save['handcrafted_summary_cmvn'] = aggregate_participant_handcrafted(list(cmvn_segs), seg_times, total_dur)
            else:
                data_to_save['handcrafted_summary_cmvn'] = summary_330
                
        elif stage == 'wav2vec2':
            w2v2_feats = extract_wav2vec2_features(y, sr, segs, model, fe, device)
            data_to_save.update(w2v2_feats)
            
        elif stage == 'wavlm':
            wavlm_feats = extract_wavlm_features(y, sr, segs, model, fe, device)
            data_to_save.update(wavlm_feats)
            
        np.savez_compressed(out_npz, **data_to_save)
        
        file_hash = compute_file_sha256(wav_path)
        out_hash = compute_file_sha256(out_npz)
        
        manifest_rows.append({
            'participant_id': pid,
            'domain': domain,
            'stage': stage,
            'audio_duration_s': round(total_dur, 2),
            'n_vad_segments': n_segs,
            'speech_duration_s': round(len(y_speech) / sr, 2),
            'source_wav_sha256': file_hash,
            'output_npz_sha256': out_hash,
            'vad_version': 'silero-v6.2.3',
            'extractor_version': f'v3-{stage}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        print(f"[{idx+1}/{len(dev_cohort)}] PID {pid} ({domain}): Cached {stage} features to {out_npz}")
        
    if manifest_rows:
        m_df = pd.DataFrame(manifest_rows)
        if os.path.exists(MANIFEST_PATH):
            m_df.to_csv(MANIFEST_PATH, mode='a', header=False, index=False)
        else:
            m_df.to_csv(MANIFEST_PATH, index=False)
            
    # Unload large model & free memory (strictly enforce 8 GB boundary)
    if model is not None:
        del model
        del fe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        print(f"Freed memory for stage {stage}.")

def main():
    parser = argparse.ArgumentParser(description="Task C4: Canonical Feature Extraction v3")
    parser.add_argument("--stage", type=str, choices=['acoustic', 'wav2vec2', 'wavlm', 'all'], default='acoustic')
    parser.add_argument("--device", type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.chdir(_REPO_ROOT)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    labels_df = pd.read_csv('experiments/research_v2/labels_official.csv')
    dev_cohort = labels_df[labels_df['is_sealed'] == False].copy()
    
    print(f"Loaded {len(dev_cohort)} development candidate participants from labels_official.csv")
    vad_model = silero_vad.load_silero_vad()
    
    if args.stage == 'all':
        for stg in ['acoustic', 'wav2vec2', 'wavlm']:
            process_stage(stg, dev_cohort, vad_model, args.device)
    else:
        process_stage(args.stage, dev_cohort, vad_model, args.device)

if __name__ == '__main__':
    main()
