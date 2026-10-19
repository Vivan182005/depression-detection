"""
Script: experiments/research_v2/extract_features_phase7_sealed.py
Phase 7 (prereg_phase7_full_cohort.md): extract the Phase 6C/6D feature sets for the 33 formerly-sealed
participants so they can join the N = 272 cross-validation pool. Allowed only because the one-shot
sealed evaluation has already run (sealed_eval.lock must exist).
Uses the identical development extraction functions; writes into feature_store_6c / feature_store_6d.
"""
import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import librosa
import torch

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
RD = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
for p in [_REPO_ROOT, RD]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_sealed_evaluation import sealed_audio, LOCK, STORE as SEALED_STORE


def main():
    assert os.path.exists(LOCK), 'Phase 7 may only use the sealed 33 after the one-shot sealed evaluation.'
    torch.set_num_threads(6)
    import silero_vad
    from transformers import (Wav2Vec2Model, Wav2Vec2FeatureExtractor, WhisperFeatureExtractor, WhisperModel,
                              AutoFeatureExtractor, HubertModel)
    from extract_features_v3 import get_silero_vad_segments
    from extract_features_6c import w2v2_features, whisper_features
    from extract_features_6d import ssl_all_layer_mean, mit_coordination
    from extractors_v2 import BertExtractorV2

    pids = sorted(json.load(open(os.path.join(_REPO_ROOT, 'sealed_ids.json')))['sealed_pids'])
    vad = silero_vad.load_silero_vad()
    w2v = (Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').eval(),
           Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h'))
    whs = (WhisperModel.from_pretrained('openai/whisper-base').encoder.eval(),
           WhisperFeatureExtractor.from_pretrained('openai/whisper-base'))
    hub = {k: (HubertModel.from_pretrained(n).eval(), AutoFeatureExtractor.from_pretrained(n))
           for k, n in [('hubert_base_er', 'superb/hubert-base-superb-er'),
                        ('hubert_large_er', 'superb/hubert-large-superb-er')]}

    for i, pid in enumerate(pids):
        out6c = os.path.join(RD, 'feature_store_6c', f'{pid}_6c.npz')
        out6d = os.path.join(RD, 'feature_store_6d', f'{pid}_6d.npz')
        if os.path.exists(out6c) and os.path.exists(out6d):
            continue
        t0 = time.time()
        y, sr = sf.read(sealed_audio(pid))
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        y = y.astype(np.float32)
        segs = get_silero_vad_segments(y, sr, vad)
        mean, std = w2v2_features(y, sr, segs, *w2v)
        np.savez_compressed(out6c, participant_id=np.int64(pid), w2v2_all_mean=mean, w2v2_all_segstd=std,
                            whisper_base_all_mean=whisper_features(y, sr, segs, *whs),
                            n_vad_segments=np.int64(len(segs)))
        h = {k: ssl_all_layer_mean(y, sr, segs, *m) for k, m in hub.items()}
        f_eig, d_eig = mit_coordination(y, sr, segs)
        np.savez_compressed(out6d, participant_id=np.int64(pid), hubert_base_er_all_mean=h['hubert_base_er'],
                            hubert_large_er_all_mean=h['hubert_large_er'], mit_formant_eig=f_eig, mit_dmfcc_eig=d_eig)
        print(f'[{i + 1}/{len(pids)}] audio features in {time.time() - t0:.0f}s', flush=True)

    # Text: reuse the one-shot run's VAD-chunked transcripts (same procedure as development)
    for key, model_name in [('bert_t2_768d', 'bert-base-uncased'), ('emo_roberta_large', 'j-hartmann/emotion-english-roberta-large')]:
        ext = BertExtractorV2(model_name=model_name, device='cpu')
        for pid in pids:
            p = os.path.join(RD, 'feature_store_6d', f'{pid}_6d_text.npz')
            d = dict(np.load(p)) if os.path.exists(p) else {'participant_id': np.int64(pid)}
            if key in d:
                continue
            text = str(np.load(os.path.join(SEALED_STORE, f'{pid}_sealed.npz'))['transcript_clean'])
            d[key] = ext.extract_from_text(text).astype(np.float32)
            np.savez_compressed(p, **d)
        del ext
    print('Phase 7 sealed extraction complete', flush=True)


if __name__ == '__main__':
    main()
