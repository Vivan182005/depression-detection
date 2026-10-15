"""
Script: experiments/research_v2/run_sealed_evaluation.py
ONE-SHOT evaluation of the frozen models on the sealed 33 participants.

Order of operations (each step completes before the next starts):
 1. Refuse to run without --authorized, or if sealed_eval.lock exists (one-shot guarantee).
 2. Read final_frozen_config.json (written by run_phase6d_experiments.py).
 3. Fit the frozen audio and fusion models on all 239 development participants.
    The decision threshold is the Youden point of the development out-of-fold probabilities.
 4. Extract features for the 33 sealed participants with the identical development code
    (audio located by participant ID only; label-bearing folder names are never read or printed).
 5. Write sealed_predictions.csv and record its SHA-256 in the lock file BEFORE any sealed label is read.
 6. Read the sealed labels and report AUC (bootstrap CI, permutation p), balanced accuracy,
    sensitivity, specificity and the confusion matrix.
"""
import os
import sys
import json
import glob
import time
import hashlib
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
RD = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
for p in [_REPO_ROOT, RD]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_cohort_info, load_variant_features
from run_phase6c_experiments import load_bases as load_bases_6c, make_model, inner_oof, youden_threshold, W2V2_ENS_LAYERS
from run_phase6d_experiments import load_bases_6d
from run_phase6a_experiment import compute_cluster_bootstrap_ci

LOCK = os.path.join(RD, 'sealed_eval.lock')
FROZEN = os.path.join(RD, 'final_frozen_config.json')
STORE = os.path.join(RD, 'feature_store_sealed')
PRED_CSV = os.path.join(RD, 'sealed_predictions.csv')
RESULT_JSON = os.path.join(RD, 'sealed_results.json')
SEEDS = (0, 1, 2, 3, 4)
V19_POOL = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']
SSL_V = {'V6': 'w2v2_L4', 'V7': 'w2v2_L8', 'V8': 'w2v2_L12', 'V9': 'wavlm_L4', 'V10': 'wavlm_L8', 'V11': 'wavlm_L12'}


# ----------------------------------------------------------------------------- development side
def dev_matrix(name, cohort_df, pids, cache):
    if name in cache:
        return cache[name]
    if name in SSL_V.values() or name in V19_POOL or name == 'text_bert':
        vid = {'text_bert': 'V14'}.get(name, {v: k for k, v in SSL_V.items()}.get(name, name))
        cache[name] = load_variant_features(vid, cohort_df)
    else:
        if 'b6c' not in cache:
            cache['b6c'] = load_bases_6c(pids)
            cache['b6d'], cache['text_emo'] = load_bases_6d(pids)
        cache[name] = cache['text_emo'] if name == 'text_emo' else {**cache['b6c'], **cache['b6d']}[name]
    return cache[name]


def expand(name, y, cohort_df, pids, cache):
    """Resolve an audio candidate into its member base names (ensembles chosen on all 239)."""
    if name == 'ens_w2v2_L6-10':
        return [f'w2v2_L{k}' for k in W2V2_ENS_LAYERS]
    if name == 'ens_L8+best_whisper':
        wh = max([f'whisper_L{k}' for k in range(1, 7)], key=lambda n: np.mean(
            [roc_auc_score(y, inner_oof(dev_matrix(n, cohort_df, pids, cache), y, s)) for s in SEEDS]))
        return ['w2v2_L8', wh]
    if name in SSL_V:
        return [SSL_V[name]]
    return [name]


def oof_and_fit(members, y, cohort_df, pids, cache):
    """Dev OOF probabilities (seed-averaged) and fitted final models for an averaged set of members."""
    oof = np.mean([np.mean([inner_oof(dev_matrix(m, cohort_df, pids, cache), y, s) for s in SEEDS], axis=0)
                   for m in members], axis=0)
    fitted = {m: [make_model(s).fit(dev_matrix(m, cohort_df, pids, cache), y) for s in SEEDS] for m in members}
    return oof, fitted


def predict(fitted, feats):
    return np.mean([np.mean([mdl.predict_proba(feats[m])[:, 1] for mdl in mdls], axis=0)
                    for m, mdls in fitted.items()], axis=0)


# ----------------------------------------------------------------------------- sealed side
def sealed_audio(pid):
    if pid < 600:
        return os.path.join(_REPO_ROOT, f'Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav')
    hits = glob.glob(os.path.join(_REPO_ROOT, 'unseen', '**', f'{pid}_AUDIO_interviewee.wav'), recursive=True)
    assert len(hits) == 1, f'expected exactly one sealed audio file for PID {pid}, found {len(hits)}'
    return hits[0]


def extract_sealed(pids, needed, path_fn=None, store=STORE):
    path_fn = path_fn or sealed_audio
    import silero_vad
    from transformers import (Wav2Vec2Model, Wav2Vec2FeatureExtractor, WhisperFeatureExtractor, WhisperModel,
                              AutoFeatureExtractor, AutoModel, HubertModel, pipeline)
    from extract_features_v3 import get_silero_vad_segments, extract_wavlm_features
    from extract_features_6c import w2v2_features, whisper_features
    from extract_features_6d import ssl_all_layer_mean, mit_coordination
    from transcribe_vad_chunked import transcribe_array
    from extractors_v2 import BertExtractorV2

    fam = {n.split('_L')[0] if '_L' in n else n for n in needed}
    vad = silero_vad.load_silero_vad()
    m = {}
    if fam & {'w2v2'}:
        m['w2v2'] = (Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').eval(),
                     Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h'))
    if fam & {'whisper'}:
        m['whisper'] = (WhisperModel.from_pretrained('openai/whisper-base').encoder.eval(),
                        WhisperFeatureExtractor.from_pretrained('openai/whisper-base'))
    if fam & {'wavlm'}:
        m['wavlm'] = (AutoModel.from_pretrained('microsoft/wavlm-base-plus').eval(),
                      AutoFeatureExtractor.from_pretrained('microsoft/wavlm-base-plus'))
    for key, name in [('hubert_base_er', 'superb/hubert-base-superb-er'), ('hubert_large_er', 'superb/hubert-large-superb-er')]:
        if key in fam:
            m[key] = (HubertModel.from_pretrained(name).eval(), AutoFeatureExtractor.from_pretrained(name))
    need_text = bool(fam & {'text_bert', 'text_emo'})
    asr = pipeline('automatic-speech-recognition', model='openai/whisper-tiny', device='cpu') if need_text else None

    os.makedirs(store, exist_ok=True)
    rows = {}
    for i, pid in enumerate(pids):
        out = os.path.join(store, f'{pid}_sealed.npz')
        if os.path.exists(out):
            rows[pid] = dict(np.load(out))
            continue
        y, sr = sf.read(path_fn(pid))
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        y = y.astype(np.float32)
        segs = get_silero_vad_segments(y, sr, vad)
        f = {}
        if 'w2v2' in m:
            mean, std = w2v2_features(y, sr, segs, *m['w2v2'])
            for k in range(1, 13):
                f[f'w2v2_L{k}'] = mean[k]
                f[f'w2v2_L{k}_ms'] = np.concatenate([mean[k], std[k]])
        if 'whisper' in m:
            wh = whisper_features(y, sr, segs, *m['whisper'])
            for k in range(1, 7):
                f[f'whisper_L{k}'] = wh[k]
        if 'wavlm' in m:
            wl = extract_wavlm_features(y, sr, segs, m['wavlm'][0], m['wavlm'][1], 'cpu')
            for k in (4, 8, 12):
                f[f'wavlm_L{k}'] = wl[f'wavlm_layer_{k}_mean']
        for key in ('hubert_base_er', 'hubert_large_er'):
            if key in m:
                h = ssl_all_layer_mean(y, sr, segs, *m[key])
                for k in range(1, h.shape[0]):
                    f[f'{key}_L{k}'] = h[k]
        if 'mit' in {n.split('_')[0] for n in needed}:
            fe, de = mit_coordination(y, sr, segs)
            f['mit_formant'], f['mit_dmfcc'] = fe.ravel(), de.ravel()
            f['mit_both'] = np.concatenate([f['mit_formant'], f['mit_dmfcc']])
        if need_text:
            f['transcript_clean'] = np.array(transcribe_array(y, sr, asr, vad)[-1])
        np.savez_compressed(out, **f)
        rows[pid] = f
        print(f'  sealed extraction {i + 1}/{len(pids)} done', flush=True)

    if need_text:
        for key, model_name in [('text_bert', 'bert-base-uncased'), ('text_emo', 'j-hartmann/emotion-english-roberta-large')]:
            if key in fam:
                ext = BertExtractorV2(model_name=model_name, device='cpu')
                for pid in pids:
                    rows[pid][key] = ext.extract_from_text(str(rows[pid]['transcript_clean']))
    return {n: np.array([rows[p][n] for p in pids], dtype=np.float32) for n in needed}


# ----------------------------------------------------------------------------- metrics
def metrics(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    auc = float(roc_auc_score(y, prob))
    ci = compute_cluster_bootstrap_ci(y, prob, n_bootstraps=5000, seed=42)
    rng = np.random.RandomState(0)
    null = np.array([roc_auc_score(rng.permutation(y), prob) for _ in range(10000)])
    return {'auc': auc, 'auc_ci95': list(ci), 'auc_perm_p': float((1 + (null >= auc).sum()) / 10001),
            'threshold': float(thr), 'sensitivity': float(sens), 'specificity': float(spec),
            'balanced_accuracy': float((sens + spec) / 2), 'accuracy': float((tp + tn) / len(y)),
            'confusion': {'tp': int(tp), 'fn': int(fn), 'tn': int(tn), 'fp': int(fp)}}


def selftest(names):
    """Run the sealed extraction path on two DEVELOPMENT participants and compare with the dev stores."""
    from transcribe_vad_chunked import audio_path_for
    cohort_df, y, groups, _ = load_cohort_info(permute=False)
    pids_all, cache = cohort_df['participant_id'].values, {}
    test = [301, 679]
    feats = extract_sealed(test, names, path_fn=lambda p: audio_path_for(p, 'woz' if p < 600 else 'avatar'),
                           store=os.path.join(RD, 'feature_store_selftest'))
    idx = [int(np.where(pids_all == p)[0][0]) for p in test]
    for n in names:
        dev = dev_matrix(n, cohort_df, pids_all, cache)[idx]
        cos = [float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) for a, b in zip(feats[n], dev)]
        print(f'selftest {n}: cosine vs dev store = {[round(c, 6) for c in cos]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--authorized', action='store_true', help='user approved the one-shot sealed evaluation')
    ap.add_argument('--selftest', type=str, default='', help='comma-separated feature names to check on 2 dev PIDs')
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.selftest.split(','))
    if not args.authorized:
        raise SystemExit('Refusing: the sealed evaluation is one-shot and needs --authorized.')
    if os.path.exists(LOCK):
        raise SystemExit(f'Refusing: sealed evaluation already ran ({LOCK}).')

    frozen = json.load(open(FROZEN))
    sealed_pids = sorted(json.load(open(os.path.join(_REPO_ROOT, 'sealed_ids.json')))['sealed_pids'])
    cohort_df, y, groups, _ = load_cohort_info(permute=False)
    pids, cache = cohort_df['participant_id'].values, {}
    print('Frozen config:', json.dumps(frozen))

    # ---- audio model
    audio_members = expand(frozen['audio']['model'], y, cohort_df, pids, cache)
    a_oof, a_fit = oof_and_fit(audio_members, y, cohort_df, pids, cache)
    a_thr = youden_threshold(y, a_oof)

    # ---- fusion model
    if frozen['fusion']['model'] == 'V19':
        # V19 procedure on all 239: pick the best of V1-V11 by inner-CV AUC, stack with V14 (BERT-T2).
        pool_auc = {v: np.mean([roc_auc_score(y, inner_oof(dev_matrix(v, cohort_df, pids, cache), y, s)) for s in SEEDS])
                    for v in V19_POOL}
        best_v = max(pool_auc, key=pool_auc.get)
        if best_v not in SSL_V:
            raise SystemExit(f'V19 selected {best_v}; hand-crafted sealed extraction is not implemented. Stop and report.')
        f_audio, f_text = expand(best_v, y, cohort_df, pids, cache), 'text_bert'
        fusion_desc = {'V19_selected_audio': best_v, 'pool_inner_auc': pool_auc}
    else:
        f_audio, f_text = expand(frozen['fusion']['audio'], y, cohort_df, pids, cache), frozen['fusion']['text']
        fusion_desc = {}
    fa_oof, fa_fit = oof_and_fit(f_audio, y, cohort_df, pids, cache)
    ft_oof, ft_fit = oof_and_fit([f_text], y, cohort_df, pids, cache)
    meta = LogisticRegression(C=1.0).fit(np.column_stack([fa_oof, ft_oof]), y)
    f_thr = youden_threshold(y, meta.predict_proba(np.column_stack([fa_oof, ft_oof]))[:, 1])
    dev_summary = {'audio_members': audio_members, 'audio_dev_oof_auc': float(roc_auc_score(y, a_oof)),
                   'fusion_audio_members': f_audio, 'fusion_text': f_text, 'meta_coef': meta.coef_[0].tolist(),
                   'fusion_dev_oof_auc': float(roc_auc_score(y, meta.predict_proba(np.column_stack([fa_oof, ft_oof]))[:, 1])),
                   **fusion_desc}
    print('Dev-side summary:', json.dumps(dev_summary, default=float))

    # ---- sealed features and predictions (no labels touched)
    needed = sorted(set(audio_members) | set(f_audio) | {f_text})
    feats = extract_sealed(sealed_pids, needed)
    p_audio = predict(a_fit, feats)
    p_fus = meta.predict_proba(np.column_stack([predict(fa_fit, feats), predict(ft_fit, feats)]))[:, 1]
    pred = pd.DataFrame({'participant_id': sealed_pids, 'domain': ['woz' if p < 600 else 'avatar' for p in sealed_pids],
                         'prob_audio': p_audio, 'pred_audio': (p_audio >= a_thr).astype(int),
                         'prob_fusion': p_fus, 'pred_fusion': (p_fus >= f_thr).astype(int)})
    pred.to_csv(PRED_CSV, index=False)
    digest = hashlib.sha256(open(PRED_CSV, 'rb').read()).hexdigest()
    json.dump({'predictions_sha256': digest, 'written_before_labels_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'frozen': frozen, 'dev_summary': dev_summary}, open(LOCK, 'w'), indent=2, default=float)
    print(f'Predictions written and locked (sha256 {digest[:16]}...). Now reading sealed labels.')

    # ---- unblinding
    lab = pd.read_csv(os.path.join(RD, 'labels_official.csv')).set_index('participant_id').loc[sealed_pids]
    y_s = lab['binary_label'].values.astype(int)
    res = {'n': len(y_s), 'n_depressed': int(y_s.sum()), 'predictions_sha256': digest, 'dev_summary': dev_summary,
           'audio_model': {'frozen': frozen['audio'], **metrics(y_s, p_audio, a_thr)},
           'fusion_model': {'frozen': frozen['fusion'], **metrics(y_s, p_fus, f_thr)}}
    for key, prob in [('audio_model', p_audio), ('fusion_model', p_fus)]:
        for dom in ['woz', 'avatar']:
            msk = (pred['domain'] == dom).values
            if len(np.unique(y_s[msk])) == 2:
                res[key][f'auc_{dom}'] = float(roc_auc_score(y_s[msk], prob[msk]))
                res[key][f'n_{dom}'] = int(msk.sum())
    json.dump(res, open(RESULT_JSON, 'w'), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
