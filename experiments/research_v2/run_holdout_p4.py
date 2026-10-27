"""
Script: experiments/research_v2/run_holdout_p4.py
Holdout rule for the Phase 7 P4 model: the full P4 procedure (config scoring by seed-averaged 4-fold CV, top-3 voice
configs + best text config, meta-LR, thresholds) run on the 239 non-sealed participants only, then one prediction of
the 33 in sealed_ids.json. Same as train_final_model_phase7.py, restricted to the 239. Writes holdout_p4_results.json.
"""
import os
import sys
import json
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score

RD = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(RD))
sys.path.insert(0, RD)
from run_phase7_experiments import load_pool, tune_and_oof, to_cdf, thresholds, apply_thr, binary_metrics, VOICE, TEXT, SEEDS, N_JOBS
from final_model_phase7 import ConfigModel, predict_bundle
from run_holdout_239_33 import boot_ci, perm_p


def main():
    df, X = load_pool()
    pids = df['participant_id'].values
    y, phq, dom = df['y'].values, df['phq'].values, df['domain'].values
    sealed = json.load(open(os.path.join(ROOT, 'sealed_ids.json')))['sealed_pids']
    te, tr = np.where(np.isin(pids, sealed))[0], np.where(~np.isin(pids, sealed))[0]
    assert len(tr) == 239 and len(te) == 33
    cfgs = [(b, k, pd_) for b in VOICE + TEXT for k in ('cls', 'reg') for pd_ in (False, True)]
    runs = Parallel(n_jobs=N_JOBS)(delayed(tune_and_oof)(X[b][tr], y[tr], phq[tr], dom[tr], k, pd_, s)
                                   for b, k, pd_ in cfgs for s in SEEDS)
    info = {}
    for i, c in enumerate(cfgs):
        rr = runs[i * len(SEEDS):(i + 1) * len(SEEDS)]
        hps = [r[0] for r in rr]
        info[c] = {'auc': float(np.mean([r[1] for r in rr])), 'hp': max(set(hps), key=hps.count), 'oof': np.mean([r[2] for r in rr], axis=0)}
    rank = lambda cs: sorted(cs, key=lambda c: -info[c]['auc'])
    top3, seen = [], set()
    for c in rank([c for c in cfgs if c[0] in VOICE]):
        if c[0] not in seen:
            top3.append(c)
            seen.add(c[0])
        if len(top3) == 3:
            break
    tbest = rank([c for c in cfgs if c[0] in TEXT])[0]
    members = {c: ConfigModel(*c, info[c]['hp'], X[c[0]][tr], y[tr], phq[tr], dom[tr], info[c]['oof']) for c in top3 + [tbest]}
    A = np.column_stack([np.mean([to_cdf(info[c]['oof'], info[c]['oof']) for c in top3], axis=0), to_cdf(info[tbest]['oof'], info[tbest]['oof'])])
    meta = LogisticRegression(C=1.0).fit(A, y[tr])
    thr = thresholds(y[tr], meta.predict_proba(A)[:, 1], dom[tr], per_domain=False)['all']
    bundle = {'voice_members': [members[c] for c in top3], 'text_member': members[tbest], 'meta': meta, 'thresholds': thr}
    so, _, _ = predict_bundle(bundle, {k: v[te] for k, v in X.items()}, list(dom[te]))
    yt = y[te]
    r = {'selected_voice': [list(map(str, c)) for c in top3], 'selected_text': list(map(str, tbest)),
         'auc': float(roc_auc_score(yt, so)), 'auc_ci95': boot_ci(yt, so), 'perm_p': perm_p(yt, so), 'operating_points': {}}
    for rule in ('max_acc', 'youden'):
        b = (so >= thr[rule]).astype(int)
        r['operating_points'][rule] = dict(binary_metrics(yt, b), mcc=float(matthews_corrcoef(yt, b)),
                                           macro_f1=float(f1_score(yt, b, average='macro')), n_correct=int((b == yt).sum()))
    print(json.dumps(r, indent=1))
    json.dump(r, open(os.path.join(RD, 'holdout_p4_results.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
