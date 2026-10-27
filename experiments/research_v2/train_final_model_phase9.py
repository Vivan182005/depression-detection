"""
Script: experiments/research_v2/train_final_model_phase9.py
Fits the adopted Phase 9 model F1 (voice + text_emo + LLM PHQ-8 items) on all 272 participants with the same procedure
as each Phase 9 outer fold: ridge on PHQ-8 for each member, alpha by 4-fold CV AUC (most frequent best over seeds 0-4),
CDF reference = seed-averaged out-of-fold scores, logistic-regression stacker and max-accuracy / Youden thresholds on
the out-of-fold stacked scores. Saves models/final_phase9_f1.joblib (+ .json). The Phase 7 model is kept.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

RD = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(RD))
sys.path.insert(0, RD)
from run_phase7_experiments import load_pool, thresholds, to_cdf, A_GRID
from run_phase8_experiments import BASES, SEEDS, standardise
from score_phq_llm import ITEMS, OUT as LLM_CSV, MODEL as LLM_MODEL
from final_model_phase7 import ConfigModel
from final_model_phase9 import predict_bundle_f1

OUT = os.path.join(ROOT, 'models', 'final_phase9_f1.joblib')


def tune(X, y, phq):
    """Seed-averaged 4-fold out-of-fold ridge scores at the most frequently chosen alpha."""
    best, oofs = [], {}
    for seed in SEEDS:
        o = {al: np.zeros(len(y)) for al in A_GRID}
        for a, b in StratifiedKFold(4, shuffle=True, random_state=seed).split(X, y):
            Xa, Xb = standardise(X[a], X[b])
            for al in A_GRID:
                o[al][b] = Ridge(alpha=al).fit(Xa, phq[a]).predict(Xb)
        best.append(max(A_GRID, key=lambda al: roc_auc_score(y, o[al])))
        for al in A_GRID:
            oofs.setdefault(al, []).append(o[al])
    hp = max(set(best), key=best.count)
    oof = np.mean(oofs[hp], axis=0)
    return hp, oof, float(roc_auc_score(y, oof))


def main():
    df, X = load_pool()
    pids = df['participant_id'].values
    y, phq, dom = df['y'].values, df['phq'].values, df['domain'].values
    L = pd.read_csv(LLM_CSV).drop_duplicates('participant_id').set_index('participant_id').loc[pids]
    assert (L.n_items_parsed == 8).all()
    X['llm_items'] = L[ITEMS].values.astype(np.float32)

    members, info = {}, {}
    for b in BASES + ['text_emo', 'llm_items']:
        hp, oof, auc = tune(X[b].astype(np.float64), y, phq)
        members[b] = ConfigModel(b, 'reg', False, hp, X[b], y, phq, dom, oof)
        info[b] = (hp, oof, auc)
        print(f'{b:<22} alpha {hp:>7} oof AUC {auc:.3f}', flush=True)

    cdf = {b: to_cdf(info[b][1], info[b][1]) for b in info}
    A = np.column_stack([np.mean([cdf[b] for b in BASES], axis=0), cdf['text_emo'], cdf['llm_items']])
    meta = LogisticRegression(C=1.0).fit(A, y)
    stacked = meta.predict_proba(A)[:, 1]
    thr = thresholds(y, stacked, dom, per_domain=False)['all']

    p9 = json.load(open(os.path.join(RD, 'phase9_results.json')))['analyses']['F1_voice_text_llm']
    ho = json.load(open(os.path.join(RD, 'holdout_239_33_results.json')))['models']['F1_voice_text_llm']
    bundle = {'voice_members': [members[b] for b in BASES], 'text_member': members['text_emo'], 'llm_member': members['llm_items'],
              'meta': meta, 'thresholds': thr, 'llm_items': ITEMS, 'description': {
                  'members': {b: {'alpha': info[b][0], 'oof_auc': round(info[b][2], 3)} for b in info},
                  'llm': LLM_MODEL, 'meta_coef_voice_text_llm': meta.coef_[0].tolist(), 'trained_on': f'N={len(y)} (all 272)',
                  'phase9_nested_cv_auc': p9['auc'], 'phase9_nested_cv_accuracy': p9['operating_points']['max_acc']['accuracy'],
                  'holdout_239_33_auc': ho['auc'], 'holdout_239_33_accuracy': ho['operating_points']['max_acc']['accuracy']}}

    # Self-check: the saved bundle reproduces the training-pool scores from the raw features.
    p, _, _ = predict_bundle_f1(bundle, X, list(dom))
    print(f'in-sample AUC {roc_auc_score(y, p):.3f} (sanity only, NOT a performance number); stacked OOF AUC {roc_auc_score(y, stacked):.3f}')
    joblib.dump(bundle, OUT)
    json.dump({'thresholds': thr, **bundle['description']}, open(OUT.replace('.joblib', '.json'), 'w'), indent=2)
    print('meta coef (voice, text, llm)', np.round(meta.coef_[0], 3), 'thresholds', thr, '->', OUT)


if __name__ == '__main__':
    main()
