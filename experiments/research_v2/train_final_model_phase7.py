"""
Script: experiments/research_v2/train_final_model_phase7.py
Fits the Phase 7 headline procedure (P4: top-3 voice ensemble + best text, meta-LR stack) on all 272
participants and saves it to models/final_phase7_p4.joblib. The selection rules are the same as inside each
Phase 7 outer fold, applied once to the full pool (configs scored by 4-fold CV, averaged over seeds 0-4).
Thresholds (max-accuracy and Youden) come from the full-pool out-of-fold stacked scores.
"""
import os
import sys
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
RD = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
for p in [_REPO_ROOT, RD]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_phase7_experiments import (load_pool, tune_and_oof, fit_score, to_cdf, thresholds, VOICE, TEXT, SEEDS, N_JOBS)
from final_model_phase7 import ConfigModel, predict_bundle

OUT = os.path.join(_REPO_ROOT, 'models', 'final_phase7_p4.joblib')


def main():
    df, X = load_pool()
    y, phq, dom = df['y'].values, df['phq'].values, df['domain'].values
    cfgs = [(b, k, pd_) for b in VOICE + TEXT for k in ('cls', 'reg') for pd_ in (False, True)]
    # Score each config by seed-averaged 4-fold CV AUC on all 272 (hyperparameter = most frequent best over seeds)
    runs = Parallel(n_jobs=N_JOBS)(delayed(tune_and_oof)(X[b], y, phq, dom, k, pd_, s)
                                   for b, k, pd_ in cfgs for s in SEEDS)
    info = {}
    for i, c in enumerate(cfgs):
        rr = runs[i * len(SEEDS):(i + 1) * len(SEEDS)]
        hps = [r[0] for r in rr]
        info[c] = {'auc': float(np.mean([r[1] for r in rr])), 'hp': max(set(hps), key=hps.count),
                   'oof': np.mean([r[2] for r in rr], axis=0)}
    rank = lambda cs: sorted(cs, key=lambda c: -info[c]['auc'])
    top3, seen = [], set()
    for c in rank([c for c in cfgs if c[0] in VOICE]):
        if c[0] not in seen:
            top3.append(c)
            seen.add(c[0])
        if len(top3) == 3:
            break
    tbest = rank([c for c in cfgs if c[0] in TEXT])[0]

    members = {c: ConfigModel(*c, info[c]['hp'], X[c[0]], y, phq, dom, info[c]['oof']) for c in top3 + [tbest]}
    v_oof = np.mean([to_cdf(info[c]['oof'], info[c]['oof']) for c in top3], axis=0)
    t_oof = to_cdf(info[tbest]['oof'], info[tbest]['oof'])
    meta = LogisticRegression(C=1.0).fit(np.column_stack([v_oof, t_oof]), y)
    stacked = meta.predict_proba(np.column_stack([v_oof, t_oof]))[:, 1]
    thr = thresholds(y, stacked, dom, per_domain=False)['all']

    bundle = {'voice_members': [members[c] for c in top3], 'text_member': members[tbest], 'meta': meta,
              'thresholds': thr, 'description': {
                  'voice': [list(map(str, c)) + [f"cv_auc={info[c]['auc']:.3f}", f"hp={info[c]['hp']}"] for c in top3],
                  'text': list(map(str, tbest)) + [f"cv_auc={info[tbest]['auc']:.3f}", f"hp={info[tbest]['hp']}"],
                  'meta_coef': meta.coef_[0].tolist(), 'trained_on': f'N={len(y)} (all 272)',
                  'phase7_nested_cv_auc': json.load(open(os.path.join(RD, 'phase7_results.json')))['procedures']['P4_fusion']['auc']}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump(bundle, OUT)
    json.dump({'thresholds': thr, **bundle['description']}, open(OUT.replace('.joblib', '.json'), 'w'), indent=2)
    print(json.dumps(bundle['description'], indent=2))
    print('thresholds', thr, '->', OUT)


if __name__ == '__main__':
    main()
