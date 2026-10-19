"""
Script: experiments/research_v2/run_phase7_experiments.py
Phase 7 (prereg_phase7_full_cohort.md): N = 272 repeated nested CV with
  CLS (logistic) vs REG (ridge on PHQ-8), per-domain standardisation (F1), per-domain thresholds (F2),
  procedures P1 (best CLS voice), P2 (best REG voice), P3 (top-3 voice ensemble), P4 (P3 + best text stack).
Everything (hyperparameters, config choice, ensemble members, stacking, thresholds) is learnt inside the
outer training fold. Writes phase7_results.json and phase7_oof_predictions.csv.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import Parallel, delayed

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
RD = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
for p in [_REPO_ROOT, RD]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_phase6a_experiment import compute_cluster_bootstrap_ci

SEEDS = (0, 1, 2, 3, 4)
N_JOBS = 6
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
A_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0]
VOICE = ([f'w2v2_L{k}' for k in range(5, 12)] + [f'hubert_large_er_L{k}' for k in range(16, 21)]
         + ['hubert_base_er_L9', 'whisper_L5', 'whisper_L6'])
TEXT = ['text_bert', 'text_emo']
OUT_JSON = os.path.join(RD, 'phase7_results.json')
OUT_OOF = os.path.join(RD, 'phase7_oof_predictions.csv')


# ----------------------------------------------------------------------------- data
def load_pool():
    assert os.path.exists(os.path.join(RD, 'sealed_eval.lock')), 'Phase 7 runs only after the one-shot sealed evaluation'
    dev = pd.read_csv(os.path.join(RD, 'feature_store_v3', 'manifest_v3.csv'))['participant_id'].tolist()
    sealed = json.load(open(os.path.join(_REPO_ROOT, 'sealed_ids.json')))['sealed_pids']
    pids = sorted(set(dev) | set(sealed))
    lab = pd.read_csv(os.path.join(RD, 'labels_official.csv')).set_index('participant_id').loc[pids]
    df = pd.DataFrame({'participant_id': pids, 'domain': lab['domain'].values, 'gender': lab['gender'].values,
                       'phq': lab['phq8_score'].values.astype(float), 'y': lab['binary_label'].values.astype(int),
                       'was_sealed': [p in set(sealed) for p in pids]})
    X = {}
    c6 = [np.load(os.path.join(RD, 'feature_store_6c', f'{p}_6c.npz')) for p in pids]
    d6 = [np.load(os.path.join(RD, 'feature_store_6d', f'{p}_6d.npz')) for p in pids]
    for k in range(5, 12):
        X[f'w2v2_L{k}'] = np.array([d['w2v2_all_mean'][k] for d in c6], dtype=np.float32)
    for k in (5, 6):
        X[f'whisper_L{k}'] = np.array([d['whisper_base_all_mean'][k] for d in c6], dtype=np.float32)
    for k in range(16, 21):
        X[f'hubert_large_er_L{k}'] = np.array([d['hubert_large_er_all_mean'][k] for d in d6], dtype=np.float32)
    X['hubert_base_er_L9'] = np.array([d['hubert_base_er_all_mean'][9] for d in d6], dtype=np.float32)
    bert, emo = [], []
    for p in pids:
        t = np.load(os.path.join(RD, 'feature_store_6d', f'{p}_6d_text.npz'))
        emo.append(t['emo_roberta_large'])
        if p in set(sealed):
            bert.append(t['bert_t2_768d'])
        else:
            bert.append(np.load(os.path.join(RD, 'feature_store_v2', f'{p}_v2.npz'))['bert_t2_768d'])
    X['text_bert'] = np.array(bert, dtype=np.float32)
    X['text_emo'] = np.array(emo, dtype=np.float32)
    return df, X


# ----------------------------------------------------------------------------- modelling
def prep(Xtr, dtr, Xte, dte, per_domain):
    """Standardise with training statistics; per_domain=True uses each domain's own training statistics."""
    Xtr, Xte = Xtr.astype(np.float64).copy(), Xte.astype(np.float64).copy()
    groups = sorted(set(dtr)) if per_domain else [None]
    for g in groups:
        mtr = (dtr == g) if g is not None else np.ones(len(dtr), bool)
        mte = (dte == g) if g is not None else np.ones(len(dte), bool)
        mu, sd = Xtr[mtr].mean(0), Xtr[mtr].std(0) + 1e-8
        Xtr[mtr] = (Xtr[mtr] - mu) / sd
        Xte[mte] = (Xte[mte] - mu) / sd
    return Xtr, Xte


def fit_score(Xtr, ytr, phq_tr, Xte, kind, hp):
    if kind == 'cls':
        m = LogisticRegression(C=hp, max_iter=3000).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    m = Ridge(alpha=hp).fit(Xtr, phq_tr)
    return m.predict(Xte)


def tune_and_oof(X, y, phq, dom, kind, per_domain, seed):
    """4-fold inner CV over the grid: returns (best_hp, inner AUC, inner OOF scores at best_hp)."""
    grid = C_GRID if kind == 'cls' else A_GRID
    folds = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(X, y))
    oofs = {hp: np.zeros(len(y)) for hp in grid}
    for tr, va in folds:
        Xa, Xb = prep(X[tr], dom[tr], X[va], dom[va], per_domain)
        for hp in grid:
            oofs[hp][va] = fit_score(Xa, y[tr], phq[tr], Xb, kind, hp)
    aucs = {hp: roc_auc_score(y, o) for hp, o in oofs.items()}
    best = max(aucs, key=aucs.get)
    return best, aucs[best], oofs[best]


def to_cdf(ref, x):
    """Map scores to [0, 1] by the empirical CDF of the training reference scores."""
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side='right') / len(ref)


def thresholds(y, s, dom, per_domain):
    def best(yy, ss):
        cand = np.unique(ss)
        acc = [((ss >= t).astype(int) == yy).mean() for t in cand]
        fpr, tpr, thr = roc_curve(yy, ss)
        return {'max_acc': float(cand[int(np.argmax(acc))]), 'youden': float(thr[int(np.argmax(tpr - fpr))])}
    if not per_domain:
        return {'all': best(y, s)}
    return {g: best(y[dom == g], s[dom == g]) for g in sorted(set(dom))}


def apply_thr(s, dom, thr, rule):
    if 'all' in thr:
        return (s >= thr['all'][rule]).astype(int)
    return np.array([int(v >= thr[d][rule]) for v, d in zip(s, dom)])


# ----------------------------------------------------------------------------- nested CV
def run(df, X):
    y, phq, dom = df['y'].values, df['phq'].values, df['domain'].values
    n = len(y)
    procs = ['P1_cls', 'P2_reg', 'P3_top3', 'P4_fusion']
    score = {p: np.zeros((len(SEEDS), n)) for p in procs}
    pred = {(p, rule, pdth): np.zeros((len(SEEDS), n), dtype=int)
            for p in procs for rule in ('max_acc', 'youden') for pdth in (False, True)}
    picks = []
    for si, seed in enumerate(SEEDS):
        for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                                        .split(np.zeros(n), y, groups=df['participant_id'].values)):
            t0, rs = time.time(), seed * 100 + fold
            cfgs = [(b, k, pd_) for b in VOICE + TEXT for k in ('cls', 'reg') for pd_ in (False, True)]
            outs = Parallel(n_jobs=N_JOBS)(delayed(tune_and_oof)(X[b][tr], y[tr], phq[tr], dom[tr], k, pd_, rs)
                                           for b, k, pd_ in cfgs)
            info = {c: {'hp': hp, 'auc': auc, 'oof': oof} for c, (hp, auc, oof) in zip(cfgs, outs)}

            def outer(cfg):
                base, kind, pd_ = cfg
                Xa, Xb = prep(X[base][tr], dom[tr], X[base][va], dom[va], pd_)
                return fit_score(Xa, y[tr], phq[tr], Xb, kind, info[cfg]['hp'])

            voice_cfgs = [c for c in info if c[0] in VOICE]
            text_cfgs = [c for c in info if c[0] in TEXT]
            rank = lambda cs: sorted(cs, key=lambda c: -info[c]['auc'])
            p1 = rank([c for c in voice_cfgs if c[1] == 'cls'])[0]
            p2 = rank([c for c in voice_cfgs if c[1] == 'reg'])[0]
            top3, seen = [], set()
            for c in rank(voice_cfgs):          # top-3 distinct voice bases
                if c[0] not in seen:
                    top3.append(c)
                    seen.add(c[0])
                if len(top3) == 3:
                    break
            tbest = rank(text_cfgs)[0]

            inner, outer_s = {}, {}
            for name, cfg in [('P1_cls', p1), ('P2_reg', p2)]:
                inner[name], outer_s[name] = info[cfg]['oof'], outer(cfg)
            inner['P3_top3'] = np.mean([to_cdf(info[c]['oof'], info[c]['oof']) for c in top3], axis=0)
            outer_s['P3_top3'] = np.mean([to_cdf(info[c]['oof'], outer(c)) for c in top3], axis=0)
            t_in = to_cdf(info[tbest]['oof'], info[tbest]['oof'])
            t_out = to_cdf(info[tbest]['oof'], outer(tbest))
            meta = LogisticRegression(C=1.0).fit(np.column_stack([inner['P3_top3'], t_in]), y[tr])
            inner['P4_fusion'] = meta.predict_proba(np.column_stack([inner['P3_top3'], t_in]))[:, 1]
            outer_s['P4_fusion'] = meta.predict_proba(np.column_stack([outer_s['P3_top3'], t_out]))[:, 1]

            for p in procs:
                score[p][si, va] = outer_s[p]
                for pdth in (False, True):
                    thr = thresholds(y[tr], inner[p], dom[tr], pdth)
                    for rule in ('max_acc', 'youden'):
                        pred[(p, rule, pdth)][si, va] = apply_thr(outer_s[p], dom[va], thr, rule)
            picks.append({'seed': seed, 'fold': fold, 'P1': list(map(str, p1)), 'P2': list(map(str, p2)),
                          'P3': [list(map(str, c)) for c in top3], 'text': list(map(str, tbest)),
                          'meta_coef': meta.coef_[0].tolist()})
            print(f'  seed {seed} fold {fold}: P1={p1} P2={p2} top3={[c[0] for c in top3]} text={tbest[0]} '
                  f'{time.time() - t0:.0f}s', flush=True)
    return score, pred, picks


def binary_metrics(y, b):
    sens, spec = float(np.mean(b[y == 1] == 1)), float(np.mean(b[y == 0] == 0))
    return {'accuracy': float(np.mean(b == y)), 'balanced_accuracy': (sens + spec) / 2, 'sensitivity': sens, 'specificity': spec}


def main():
    df, X = load_pool()
    y, dom, gen = df['y'].values, df['domain'].values, df['gender'].values
    print(f'Pool N={len(df)} (WoZ {int((dom == "woz").sum())}, Avatar {int((dom == "avatar").sum())}), '
          f'depressed {int(y.sum())}; majority-class accuracy {1 - y.mean():.3f}', flush=True)
    score, pred, picks = run(df, X)
    res = {'n': len(df), 'n_depressed': int(y.sum()), 'majority_class_accuracy': float(1 - y.mean()), 'procedures': {}}
    oof = df[['participant_id', 'domain', 'gender', 'was_sealed']].copy()
    for p, S in score.items():
        mean_s = S.mean(axis=0)
        oof[p] = mean_s
        auc = float(roc_auc_score(y, mean_s))
        ci = compute_cluster_bootstrap_ci(y, mean_s, n_bootstraps=2000, seed=42)
        r = {'auc': auc, 'auc_ci95': list(ci), 'seed_aucs': [float(roc_auc_score(y, s)) for s in S], 'operating_points': {}}
        for (pp, rule, pdth), B in pred.items():
            if pp != p:
                continue
            per_seed = [binary_metrics(y, b) for b in B]
            r['operating_points'][f'{rule}{"_per_domain" if pdth else ""}'] = {
                k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
        sub = {}
        for name, mask in [('woz', dom == 'woz'), ('avatar', dom == 'avatar'), ('female', gen == 'female'), ('male', gen == 'male')]:
            b = pred[(p, 'max_acc', True)]
            sub[name] = {'n': int(mask.sum()), 'auc': float(roc_auc_score(y[mask], mean_s[mask])),
                         **{k: float(np.mean([binary_metrics(y[mask], bb[mask])[k] for bb in b])) for k in ('accuracy', 'sensitivity', 'specificity')}}
        r['subgroups_max_acc_per_domain'] = sub
        res['procedures'][p] = r
        op = r['operating_points']
        print(f'{p}: AUC={auc:.4f} [{ci[0]:.3f}, {ci[1]:.3f}] | max-acc acc={op["max_acc"]["accuracy"]:.3f} '
              f'(bal {op["max_acc"]["balanced_accuracy"]:.3f}) | per-domain max-acc acc={op["max_acc_per_domain"]["accuracy"]:.3f} '
              f'(bal {op["max_acc_per_domain"]["balanced_accuracy"]:.3f}) | youden bal={op["youden"]["balanced_accuracy"]:.3f}', flush=True)
        print(f'    subgroups: ' + ', '.join(f'{k} AUC {v["auc"]:.3f}' for k, v in sub.items()), flush=True)
    res['picks'] = picks
    res['headline'] = max(res['procedures'], key=lambda k: res['procedures'][k]['auc'])
    json.dump(res, open(OUT_JSON, 'w'), indent=2)
    oof.to_csv(OUT_OOF, index=False)
    print('HEADLINE:', res['headline'], json.dumps(res['procedures'][res['headline']]['operating_points']), flush=True)


if __name__ == '__main__':
    main()
