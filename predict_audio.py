"""
predict_audio.py: screen one interview recording with the final Phase 7 model.

    .venv_opensmile/bin/python predict_audio.py path/to/interview.wav [--domain woz|avatar]

Pipeline (identical to training): Silero VAD -> wav2vec2-base layer 6 + HuBERT-large-ER layers 19/20
(voice), Whisper-tiny VAD-chunked transcript -> emotion-RoBERTa-large (text) -> saved ensemble + stacker.
Research prototype for screening only. It is NOT a diagnosis.
"""
import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(ROOT, 'experiments', 'research_v2')
sys.path.insert(0, RD)

from final_model_phase7 import predict_bundle  # noqa: E402  (also needed to unpickle the bundle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav')
    ap.add_argument('--domain', choices=['woz', 'avatar'], default='woz',
                    help='recording setup; only matters for per-domain-scaled members')
    a = ap.parse_args()
    torch.set_num_threads(4)

    import silero_vad
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, HubertModel, pipeline
    from extract_features_v3 import get_silero_vad_segments
    from extract_features_6c import w2v2_features
    from extract_features_6d import ssl_all_layer_mean
    from transcribe_vad_chunked import transcribe_array
    from extractors_v2 import BertExtractorV2

    bundle = joblib.load(os.path.join(ROOT, 'models', 'final_phase7_p4.joblib'))
    needed = {m.base for m in bundle['voice_members']} | {bundle['text_member'].base}

    y, sr = sf.read(a.wav)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    y = y.astype(np.float32)
    vad = silero_vad.load_silero_vad()
    segs = get_silero_vad_segments(y, sr, vad)
    speech_s = sum(s['end'] - s['start'] for s in segs) / sr
    print(f'Audio {len(y) / sr:.0f}s, detected speech {speech_s:.0f}s in {len(segs)} segments')
    if speech_s < 30:
        print('Warning: under 30 s of speech; the estimate will be unreliable.')

    feats = {}
    if any(n.startswith('w2v2_L') for n in needed):
        mean, _ = w2v2_features(y, sr, segs, Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').eval(),
                                Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h'))
        for k in range(1, 13):
            feats[f'w2v2_L{k}'] = mean[k][None]
    for key, name in [('hubert_large_er', 'superb/hubert-large-superb-er'), ('hubert_base_er', 'superb/hubert-base-superb-er')]:
        if any(n.startswith(key) for n in needed):
            h = ssl_all_layer_mean(y, sr, segs, HubertModel.from_pretrained(name).eval(), AutoFeatureExtractor.from_pretrained(name))
            for k in range(1, h.shape[0]):
                feats[f'{key}_L{k}'] = h[k][None]
    text = transcribe_array(y, sr, pipeline('automatic-speech-recognition', model='openai/whisper-tiny', device='cpu'), vad)[-1]
    tb = bundle['text_member'].base
    model_name = 'j-hartmann/emotion-english-roberta-large' if tb == 'text_emo' else 'bert-base-uncased'
    feats[tb] = BertExtractorV2(model_name=model_name, device='cpu').extract_from_text(text)[None]

    p, flag_acc, flag_bal = predict_bundle(bundle, feats, [a.domain])
    thr = bundle['thresholds']
    print(f'\nTranscript (first 200 chars): {text[:200]}...')
    print(f'\nDepression risk score: {p[0]:.3f}  (0 = low, 1 = high)')
    print(f'  Max-accuracy threshold {thr["max_acc"]:.3f}: {"FLAG: likely PHQ-8 >= 10" if flag_acc[0] else "not flagged"}')
    print(f'  Balanced threshold     {thr["youden"]:.3f}: {"FLAG: likely PHQ-8 >= 10" if flag_bal[0] else "not flagged"}')
    print('\nResearch screening prototype (cross-validated AUC ~0.76). Not a medical diagnosis.')


if __name__ == '__main__':
    main()
