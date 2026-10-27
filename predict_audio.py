"""
predict_audio.py: screen one interview recording with the final Phase 9 model (F1: voice + text + LLM PHQ-8 items).

    .venv_opensmile/bin/python predict_audio.py path/to/interview.wav [--domain woz|avatar] [--model f1|p4]

Pipeline (identical to training): Silero VAD -> wav2vec2-base layer 6 + HuBERT-large-ER layers 19/20
(voice), Whisper-tiny VAD-chunked transcript -> emotion-RoBERTa-large (text) and local Mistral-7B-Instruct 4-bit
zero-shot PHQ-8 item scores (LLM, run in .venv_llm via MLX) -> saved members + stacker.
--model p4 uses the previous Phase 7 model (voice + text only, no LLM).
Research prototype for screening only. It is NOT a diagnosis.
"""
import os
import sys
import gc
import json
import argparse
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(ROOT, 'experiments', 'research_v2')
LLM_PY = os.path.join(ROOT, '.venv_llm', 'bin', 'python')
sys.path.insert(0, RD)

from final_model_phase7 import predict_bundle  # noqa: E402  (also needed to unpickle the bundles)
from final_model_phase9 import predict_bundle_f1  # noqa: E402

MODELS = {'f1': ('final_phase9_f1.joblib', 'Phase 9 F1, nested-CV AUC 0.84 / accuracy 79%; unseen-33 holdout 78.8%'),
          'p4': ('final_phase7_p4.joblib', 'Phase 7 P4, nested-CV AUC 0.76 / accuracy 74%')}


def llm_items(text, items):
    """Score the transcript with the local LLM in its own venv; returns an (1, 8) array in training item order."""
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(text)
    try:
        r = subprocess.run([LLM_PY, '-W', 'ignore', os.path.join(RD, 'score_phq_llm.py'), '--text-file', f.name],
                           capture_output=True, text=True, check=True)
    finally:
        os.unlink(f.name)
    s = json.loads(next(l for l in r.stdout.splitlines() if l.startswith('LLM_JSON '))[9:])
    if s['n_items_parsed'] < len(items):
        print(f'Warning: LLM returned {s["n_items_parsed"]}/8 items; missing items are set to 1.5 (scale midpoint).')
    return np.array([[s['items'].get(k, 1.5) for k in items]]), s['items']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav')
    ap.add_argument('--domain', choices=['woz', 'avatar'], default='woz',
                    help='recording setup; only matters for per-domain-scaled members')
    ap.add_argument('--model', choices=list(MODELS), default='f1')
    a = ap.parse_args()
    if a.model == 'f1' and not os.path.exists(LLM_PY):
        sys.exit(f'{LLM_PY} not found: create .venv_llm (pip install mlx-lm) or run with --model p4')
    torch.set_num_threads(4)

    import silero_vad
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, HubertModel, pipeline
    from extract_features_v3 import get_silero_vad_segments
    from extract_features_6c import w2v2_features
    from extract_features_6d import ssl_all_layer_mean
    from transcribe_vad_chunked import transcribe_array
    from extractors_v2 import BertExtractorV2

    bundle = joblib.load(os.path.join(ROOT, 'models', MODELS[a.model][0]))
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
    print(f'\nTranscript (first 200 chars): {text[:200]}...')

    if a.model == 'f1':
        gc.collect()        # free the speech models before the 4 GB LLM loads (8 GB Macs)
        print('Scoring PHQ-8 items with the local LLM (first run loads ~4 GB; about 1-3 min)...', flush=True)
        feats['llm_items'], raw = llm_items(text, bundle['llm_items'])
        print('LLM PHQ-8 item estimates (0-3): ' + ', '.join(f'{k} {v:.1f}' for k, v in raw.items())
              + f' | sum {feats["llm_items"].sum():.1f}')
        p, flag_acc, flag_bal = predict_bundle_f1(bundle, feats, [a.domain])
    else:
        p, flag_acc, flag_bal = predict_bundle(bundle, feats, [a.domain])

    thr = bundle['thresholds']
    print(f'\nDepression risk score: {p[0]:.3f}  (0 = low, 1 = high)')
    print(f'  Max-accuracy threshold {thr["max_acc"]:.3f}: {"FLAG: likely PHQ-8 >= 10" if flag_acc[0] else "not flagged"}')
    print(f'  Balanced threshold     {thr["youden"]:.3f}: {"FLAG: likely PHQ-8 >= 10" if flag_bal[0] else "not flagged"}')
    print(f'\nModel: {MODELS[a.model][1]}. Research screening prototype. Not a medical diagnosis.')


if __name__ == '__main__':
    main()
