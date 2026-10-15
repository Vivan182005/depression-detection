"""
Script: experiments/research_v2/run_phase6c_experiments.py
Phase 6C exploratory evaluation (prereg_phase6c_exploratory.md).

1. Descriptive 5-seed OOF AUC for every base candidate (run_pipeline, same as Phase 6A).
2. Nested-selection AUC: in each outer fold (seeds 0-4 x 5 folds, stratified on label), every
   candidate is scored by inner 4-fold CV AUC on the training participants only; the winner alone
   predicts the outer fold. Done for the audio pool (C1-C6) and for the fusion (C7).
3. Youden thresholds chosen on inner OOF probabilities -> balanced accuracy / sensitivity /
   specificity on the outer fold.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
for p in [_REPO_ROOT, os.path.join(_REPO_ROOT, 'experiments', 'research_v2')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sealed_guard import assert_no_sealed_ids
from data_loader import load_cohort_info, load_variant_features
from pipeline_engine import run_pipeline
from run_phase6a_experiment import compute_cluster_bootstrap_ci, compute_paired_cluster_bootstrap_delta

STORE_6C = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_6c')
OUT_JSON = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'phase6c_results.json')
OUT_OOF = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'phase6c_oof_predictions.csv')
SEEDS = (0, 1, 2, 3, 4)
C_GRID = {'logisticregression__C': [0.001, 0.01, 0.1, 1.0, 10.0]}
W2V2_ENS_LAYERS = [6, 7, 8, 9, 10]


def load_bases(pids):
    d = [np.load(os.path.join(STORE_6C, f'{p}_6c.npz')) for p in pids]
    bases = {}
    for k in range(1, 13):
        bases[f'w2v2_L{k}'] = np.array([x['w2v2_all_mean'][k] for x in d], dtype=np.float32)
    for k in range(6, 11):
        bases[f'w2v2_L{k}_ms'] = np.array([np.concatenate([x['w2v2_all_mean'][k], x['w2v2_all_segstd'][k]])
                                          for x in d], dtype=np.float32)
    for k in range(1, 7):
        bases[f'whisper_L{k}'] = np.array([x['whisper_base_all_mean'][k] for x in d], dtype=np.float32)
    return bases


def make_model(seed):
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
                        C_GRID, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                        scoring='roc_auc', n_jobs=1)


def inner_oof(X, y, seed):
    """4-fold inner OOF probabilities on the training participants (each fit tunes C by 3-fold CV)."""
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(4, shuffle=True, random_state=seed).split(X, y):
        p[va] = make_model(seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return p


def fit_predict(X_tr, y_tr, X_va, seed):
    return make_model(seed).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def audio_candidates(base_oof, y):
    """Inner-CV scores for every audio candidate, given inner OOF probabilities of the bases."""
    scores = {}
    for name, p in base_oof.items():
        if name != 'text':
            scores[name] = (roc_auc_score(y, p), [name])
    ens = [f'w2v2_L{k}' for k in W2V2_ENS_LAYERS]
    scores['ens_w2v2_L6-10'] = (roc_auc_score(y, np.mean([base_oof[n] for n in ens], axis=0)), ens)
    wh = max([f'whisper_L{k}' for k in range(1, 7)], key=lambda n: roc_auc_score(y, base_oof[n]))
    pair = ['w2v2_L8', wh]
    scores['ens_L8+best_whisper'] = (roc_auc_score(y, np.mean([base_oof[n] for n in pair], axis=0)), pair)
    return scores


def nested_selection(bases, X_text, y, groups):
    n = len(y)
    audio_probs = np.zeros((len(SEEDS), n))
    fusion_probs = np.zeros((len(SEEDS), n))
    audio_bin = np.zeros((len(SEEDS), n), dtype=int)
    fusion_bin = np.zeros((len(SEEDS), n), dtype=int)
    picks = []
    for si, seed in enumerate(SEEDS):
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(outer.split(np.zeros(n), y, groups=groups)):
            t0 = time.time()
            rs = seed * 100 + fold
            y_tr = y[tr]
            base_oof = {name: inner_oof(X[tr], y_tr, rs) for name, X in bases.items()}
            base_oof['text'] = inner_oof(X_text[tr], y_tr, rs)
            scores = audio_candidates(base_oof, y_tr)
            best = max(scores, key=lambda k: (scores[k][0], -len(scores[k][1])))
            members = scores[best][1]

            # Audio: average member probabilities (single candidate = one member)
            p_audio_inner = np.mean([base_oof[m] for m in members], axis=0)
            p_audio_outer = np.mean([fit_predict(bases[m][tr], y_tr, bases[m][va], rs) for m in members], axis=0)
            audio_probs[si, va] = p_audio_outer
            audio_bin[si, va] = (p_audio_outer >= youden_threshold(y_tr, p_audio_inner)).astype(int)

            # Fusion (C7): meta-LR on inner OOF [audio, text]
            meta = LogisticRegression(C=1.0).fit(np.column_stack([p_audio_inner, base_oof['text']]), y_tr)
            p_text_outer = fit_predict(X_text[tr], y_tr, X_text[va], rs)
            p_fus_outer = meta.predict_proba(np.column_stack([p_audio_outer, p_text_outer]))[:, 1]
            p_fus_inner = meta.predict_proba(np.column_stack([p_audio_inner, base_oof['text']]))[:, 1]
            fusion_probs[si, va] = p_fus_outer
            fusion_bin[si, va] = (p_fus_outer >= youden_threshold(y_tr, p_fus_inner)).astype(int)

            picks.append({'seed': seed, 'fold': fold, 'audio_pick': best, 'inner_auc': scores[best][0],
                          'meta_coef': meta.coef_[0].tolist()})
            print(f'  seed {seed} fold {fold}: pick={best} (inner AUC {scores[best][0]:.3f}) '
                  f'meta=[{meta.coef_[0][0]:+.2f},{meta.coef_[0][1]:+.2f}] {time.time() - t0:.0f}s', flush=True)
    return audio_probs, fusion_probs, audio_bin, fusion_bin, picks


def threshold_metrics(y, bins):
    out = []
    for b in bins:
        sens = float(np.mean(b[y == 1] == 1))
        spec = float(np.mean(b[y == 0] == 0))
        out.append({'sensitivity': sens, 'specificity': spec, 'balanced_accuracy': (sens + spec) / 2,
                    'accuracy': float(np.mean(b == y))})
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def main():
    cohort_df, y, groups, _ = load_cohort_info(permute=False)
    assert len(cohort_df) == 239
    assert_no_sealed_ids(groups)
    pids = cohort_df['participant_id'].values
    bases = load_bases(pids)
    X_text = load_variant_features('V14', cohort_df)
    X_v7 = load_variant_features('V7', cohort_df)
    assert np.allclose(bases['w2v2_L8'], X_v7, atol=1e-4), 'w2v2_L8 must reproduce V7'

    results = {'candidates': {}}
    oof = pd.DataFrame({'participant_id': pids, 'domain': cohort_df['domain'].values})
    print('Part 1: descriptive per-candidate AUC (5-seed OOF, Phase 6A engine)')
    for name, X in bases.items():
        r = run_pipeline(X=X, y=y, groups=groups, seeds=SEEDS, variant_id='V7', strata=None)
        ci = compute_cluster_bootstrap_ci(y, r['oof_probs'], n_bootstraps=2000, seed=42)
        results['candidates'][name] = {'auc': r['auc'], 'ci95': list(ci), 'seed_aucs': r['seed_aucs']}
        oof[name] = r['oof_probs']
        print(f'  {name}: AUC={r["auc"]:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]', flush=True)

    print('Part 2: nested selection (audio pool C1-C6, fusion C7)')
    a_p, f_p, a_b, f_b, picks = nested_selection(bases, X_text, y, groups)
    a_mean, f_mean = a_p.mean(axis=0), f_p.mean(axis=0)
    oof['nested_audio'] = a_mean
    oof['nested_fusion'] = f_mean
    p6a = pd.read_csv(os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'phase6a_oof_predictions.csv'))
    p6a = p6a.set_index('participant_id').loc[pids]
    for key, probs, bins in [('nested_audio', a_mean, a_b), ('nested_fusion', f_mean, f_b)]:
        auc = float(roc_auc_score(y, probs))
        ci = compute_cluster_bootstrap_ci(y, probs, n_bootstraps=2000, seed=42)
        ref = 'V7' if key == 'nested_audio' else 'V19'
        pval, d, dl, du = compute_paired_cluster_bootstrap_delta(y, probs, p6a[ref].values, n_bootstraps=10000, seed=7)
        results[key] = {'auc': auc, 'ci95': list(ci), 'seed_aucs': [float(roc_auc_score(y, s)) for s in (a_p if key == 'nested_audio' else f_p)],
                        f'delta_vs_{ref}': d, f'delta_vs_{ref}_ci95': [dl, du], f'p_gt_{ref}': pval,
                        'threshold_metrics_youden_infold': threshold_metrics(y, bins)}
        for dom in ['woz', 'avatar']:
            m = (cohort_df['domain'] == dom).values
            results[key][f'auc_{dom}'] = float(roc_auc_score(y[m], probs[m]))
        print(f'  {key}: AUC={auc:.4f} [{ci[0]:.3f}, {ci[1]:.3f}] | vs {ref}: {d:+.4f} [{dl:+.3f}, {du:+.3f}] p={pval:.3f} '
              f'| {results[key]["threshold_metrics_youden_infold"]}', flush=True)
    results['selection_picks'] = picks
    results['pick_counts'] = pd.Series([p['audio_pick'] for p in picks]).value_counts().to_dict()
    json.dump(results, open(OUT_JSON, 'w'), indent=2)
    oof.to_csv(OUT_OOF, index=False)
    print('pick counts:', results['pick_counts'])
    print(f'Saved {OUT_JSON}')


if __name__ == '__main__':
    main()
