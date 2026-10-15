"""
Script: experiments/research_v2/run_phase6d_experiments.py
Phase 6D exploratory evaluation (prereg_phase6d_emotion_mit.md): Phase 6C pool + emotion-tuned
HuBERT layers + MIT coordination features, with nested selection; the fusion additionally chooses
between BERT-T2 (V14) and emotion-RoBERTa text inside each fold. Applies the pre-stated freeze rule
and writes final_frozen_config.json for the sealed evaluation.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
for p in [_REPO_ROOT, os.path.join(_REPO_ROOT, 'experiments', 'research_v2')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sealed_guard import assert_no_sealed_ids
from data_loader import load_cohort_info, load_variant_features
from pipeline_engine import run_pipeline
from run_phase6a_experiment import compute_cluster_bootstrap_ci, compute_paired_cluster_bootstrap_delta
from run_phase6c_experiments import (load_bases as load_bases_6c, inner_oof, fit_predict, youden_threshold,
                                     threshold_metrics, SEEDS, W2V2_ENS_LAYERS)

RD = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
STORE_6D = os.path.join(RD, 'feature_store_6d')
OUT_JSON = os.path.join(RD, 'phase6d_results.json')
OUT_OOF = os.path.join(RD, 'phase6d_oof_predictions.csv')
FINAL_FROZEN = os.path.join(RD, 'final_frozen_config.json')
TEXT_KEYS = ('text_bert', 'text_emo')
MARGIN = 0.02


def load_bases_6d(pids):
    a = [np.load(os.path.join(STORE_6D, f'{p}_6d.npz')) for p in pids]
    t = [np.load(os.path.join(STORE_6D, f'{p}_6d_text.npz')) for p in pids]
    bases = {}
    for k in range(1, 13):
        bases[f'hubert_base_er_L{k}'] = np.array([x['hubert_base_er_all_mean'][k] for x in a], dtype=np.float32)
    for k in range(1, 25):
        bases[f'hubert_large_er_L{k}'] = np.array([x['hubert_large_er_all_mean'][k] for x in a], dtype=np.float32)
    form = np.array([x['mit_formant_eig'].ravel() for x in a], dtype=np.float32)
    dmf = np.array([x['mit_dmfcc_eig'].ravel() for x in a], dtype=np.float32)
    bases['mit_formant'] = form
    bases['mit_dmfcc'] = dmf
    bases['mit_both'] = np.concatenate([form, dmf], axis=1)
    text_emo = np.array([x['emo_roberta_large'] for x in t], dtype=np.float32)
    return bases, text_emo


def audio_candidates(base_oof, y):
    scores = {n: (roc_auc_score(y, p), [n]) for n, p in base_oof.items() if n not in TEXT_KEYS}
    ens = [f'w2v2_L{k}' for k in W2V2_ENS_LAYERS]
    scores['ens_w2v2_L6-10'] = (roc_auc_score(y, np.mean([base_oof[n] for n in ens], axis=0)), ens)
    wh = max([f'whisper_L{k}' for k in range(1, 7)], key=lambda n: roc_auc_score(y, base_oof[n]))
    pair = ['w2v2_L8', wh]
    scores['ens_L8+best_whisper'] = (roc_auc_score(y, np.mean([base_oof[n] for n in pair], axis=0)), pair)
    return scores


def nested(bases, texts, y, groups):
    n = len(y)
    a_p, f_p = np.zeros((len(SEEDS), n)), np.zeros((len(SEEDS), n))
    a_b, f_b = np.zeros((len(SEEDS), n), dtype=int), np.zeros((len(SEEDS), n), dtype=int)
    picks = []
    for si, seed in enumerate(SEEDS):
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                                        .split(np.zeros(n), y, groups=groups)):
            t0 = time.time()
            rs, y_tr = seed * 100 + fold, y[tr]
            oof = {name: inner_oof(X[tr], y_tr, rs) for name, X in bases.items()}
            for tk, X in texts.items():
                oof[tk] = inner_oof(X[tr], y_tr, rs)
            scores = audio_candidates(oof, y_tr)
            best = max(scores, key=lambda k: (scores[k][0], -len(scores[k][1])))
            members = scores[best][1]
            best_text = max(TEXT_KEYS, key=lambda k: roc_auc_score(y_tr, oof[k]))

            pa_in = np.mean([oof[m] for m in members], axis=0)
            pa_out = np.mean([fit_predict(bases[m][tr], y_tr, bases[m][va], rs) for m in members], axis=0)
            a_p[si, va] = pa_out
            a_b[si, va] = (pa_out >= youden_threshold(y_tr, pa_in)).astype(int)

            meta = LogisticRegression(C=1.0).fit(np.column_stack([pa_in, oof[best_text]]), y_tr)
            pt_out = fit_predict(texts[best_text][tr], y_tr, texts[best_text][va], rs)
            pf_out = meta.predict_proba(np.column_stack([pa_out, pt_out]))[:, 1]
            pf_in = meta.predict_proba(np.column_stack([pa_in, oof[best_text]]))[:, 1]
            f_p[si, va] = pf_out
            f_b[si, va] = (pf_out >= youden_threshold(y_tr, pf_in)).astype(int)
            picks.append({'seed': seed, 'fold': fold, 'audio_pick': best, 'inner_auc': scores[best][0],
                          'text_pick': best_text, 'meta_coef': meta.coef_[0].tolist()})
            print(f'  seed {seed} fold {fold}: audio={best} ({scores[best][0]:.3f}) text={best_text} '
                  f'{time.time() - t0:.0f}s', flush=True)
    return a_p, f_p, a_b, f_b, picks


def main():
    cohort_df, y, groups, _ = load_cohort_info(permute=False)
    assert len(cohort_df) == 239
    assert_no_sealed_ids(groups)
    pids = cohort_df['participant_id'].values
    bases = load_bases_6c(pids)
    b6d, text_emo = load_bases_6d(pids)
    new_names = list(b6d)
    bases.update(b6d)
    texts = {'text_bert': load_variant_features('V14', cohort_df), 'text_emo': text_emo}

    res = {'candidates_new': {}}
    oof_df = pd.DataFrame({'participant_id': pids, 'domain': cohort_df['domain'].values})
    print('Part 1: descriptive AUC for new Phase 6D candidates')
    for name in new_names + ['text_emo']:
        X = texts['text_emo'] if name == 'text_emo' else bases[name]
        r = run_pipeline(X=X, y=y, groups=groups, seeds=SEEDS, variant_id='V7', strata=None)
        ci = compute_cluster_bootstrap_ci(y, r['oof_probs'], n_bootstraps=2000, seed=42)
        res['candidates_new'][name] = {'auc': r['auc'], 'ci95': list(ci)}
        oof_df[name] = r['oof_probs']
        print(f'  {name}: AUC={r["auc"]:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]', flush=True)

    print('Part 2: nested selection over the full pool')
    a_p, f_p, a_b, f_b, picks = nested(bases, texts, y, groups)
    p6a = pd.read_csv(os.path.join(RD, 'phase6a_oof_predictions.csv')).set_index('participant_id').loc[pids]
    ref_auc = {'V7': float(roc_auc_score(y, p6a['V7'])), 'V19': float(roc_auc_score(y, p6a['V19']))}
    for key, P, B, ref in [('nested_audio', a_p, a_b, 'V7'), ('nested_fusion', f_p, f_b, 'V19')]:
        probs = P.mean(axis=0)
        oof_df[key] = probs
        auc = float(roc_auc_score(y, probs))
        ci = compute_cluster_bootstrap_ci(y, probs, n_bootstraps=2000, seed=42)
        pv, d, dl, du = compute_paired_cluster_bootstrap_delta(y, probs, p6a[ref].values, n_bootstraps=10000, seed=7)
        res[key] = {'auc': auc, 'ci95': list(ci), 'reference': ref, 'reference_auc': ref_auc[ref],
                    'delta': d, 'delta_ci95': [dl, du], 'p_gt_reference': pv,
                    'seed_aucs': [float(roc_auc_score(y, s)) for s in P],
                    'threshold_metrics_youden_infold': threshold_metrics(y, B)}
        for dom in ['woz', 'avatar']:
            m = (cohort_df['domain'] == dom).values
            res[key][f'auc_{dom}'] = float(roc_auc_score(y[m], probs[m]))
        print(f'  {key}: AUC={auc:.4f} [{ci[0]:.3f}, {ci[1]:.3f}] vs {ref}={ref_auc[ref]:.4f}: '
              f'{d:+.4f} [{dl:+.3f}, {du:+.3f}] p={pv:.3f} | {res[key]["threshold_metrics_youden_infold"]}', flush=True)

    audio_counts = pd.Series([p['audio_pick'] for p in picks]).value_counts()
    pair_counts = pd.Series([f"{p['audio_pick']}|{p['text_pick']}" for p in picks]).value_counts()
    res['picks'] = picks
    res['audio_pick_counts'] = audio_counts.to_dict()
    res['pair_pick_counts'] = pair_counts.to_dict()

    # Pre-stated freeze rule (prereg_phase6d sec 5)
    cand_auc = {k: v['auc'] for k, v in json.load(open(os.path.join(RD, 'phase6c_results.json')))['candidates'].items()}
    cand_auc.update({k: v['auc'] for k, v in res['candidates_new'].items()})

    def most_frequent(counts):
        # Ties broken by the higher per-candidate AUC of the audio member (ensembles count as 0).
        top = counts[counts == counts.max()].index.tolist()
        return max(top, key=lambda k: cand_auc.get(k.split('|')[0], 0.0))

    frozen = {'rule': 'prereg_phase6d_emotion_mit.md sec 5', 'margin': MARGIN}
    if res['nested_audio']['auc'] >= ref_auc['V7'] + MARGIN:
        frozen['audio'] = {'model': most_frequent(audio_counts), 'source': 'phase6d'}
    else:
        frozen['audio'] = {'model': 'V7', 'source': 'phase6a'}
    if res['nested_fusion']['auc'] >= ref_auc['V19'] + MARGIN:
        a, t = most_frequent(pair_counts).split('|')
        frozen['fusion'] = {'model': 'stack', 'audio': a, 'text': t, 'source': 'phase6d'}
    else:
        frozen['fusion'] = {'model': 'V19', 'source': 'phase6a'}
    frozen['freeze_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    res['final_frozen'] = frozen
    json.dump(res, open(OUT_JSON, 'w'), indent=2, default=float)
    json.dump(frozen, open(FINAL_FROZEN, 'w'), indent=2)
    oof_df.to_csv(OUT_OOF, index=False)
    print('audio pick counts:', res['audio_pick_counts'])
    print('pair pick counts:', res['pair_pick_counts'])
    print('FINAL FROZEN:', json.dumps(frozen))


if __name__ == '__main__':
    main()
