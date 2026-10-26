"""
Script: experiments/research_v2/run_phase8b_preprocessing.py
Phase 8b (prereg_phase8b_preprocessing.md): one-shot speaker reference, few-shot domain adaptation, augmentation-averaged
features and statistics pooling as PREPROCESSING, each with the Phase 8 A0 learner. Writes phase8b_results.json.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
from run_phase7_experiments import load_pool, thresholds, apply_thr, binary_metrics, to_cdf, A_GRID
from run_phase6a_experiment import compute_cluster_bootstrap_ci, compute_paired_cluster_bootstrap_delta
from run_phase8_experiments import STORE, BASES, SEEDS, N_JOBS, standardise, text_unit

VARIANTS = ['R0_mean', 'P1_oneshot_ref', 'P2_fewshot_domain', 'P3_aug_avg', 'P4_stats_pool', 'P5_combined']
N_REF = 5


def wmean(f, X):
    return (f / f.sum()) @ X


def build(pids):
    """Label-free per-person features for every variant (P2 is the R0 mean; its domain centring happens in-fold)."""
    F = {(v, b): [] for v in VARIANTS for b in BASES}
    for p in pids:
        c, a = np.load(os.path.join(STORE, f'{p}_clean.npz')), np.load(os.path.join(STORE, f'{p}_aug.npz'))
        f = c['frames'].astype(np.float64)
        for b in BASES:
            m, ma = wmean(f, c[b]), wmean(f, a[b])
            diff = m - wmean(f[:N_REF], c[b][:N_REF])
            sd = np.sqrt(np.maximum(wmean(f, (c[b] - m) ** 2), 0))
            avg = 0.5 * (m + ma)
            for v, x in (('R0_mean', m), ('P1_oneshot_ref', np.r_[m, diff]), ('P2_fewshot_domain', m),
                         ('P3_aug_avg', avg), ('P4_stats_pool', np.r_[m, sd]), ('P5_combined', np.r_[avg, diff, sd])):
                F[(v, b)].append(x)
    return {k: np.array(v, dtype=np.float32) for k, v in F.items()}


def domain_centre(Xtr, dtr, *pairs):
    """Subtract each domain's training-fold mean (label-free)."""
    mu = {d: Xtr[dtr == d].mean(0) for d in set(dtr)}
    out = [Xtr - np.stack([mu[d] for d in dtr])]
    for X, d in pairs:
        out.append(X - np.stack([mu[g] for g in d]))
    return out


def fit_predict(X, tr, te, phq, dom, centre, alpha):
    Xtr, Xte = X[tr], X[te]
    if centre:
        Xtr, Xte = domain_centre(Xtr, dom[tr], (Xte, dom[te]))
    Xa, Xb = standardise(Xtr, Xte)
    return Ridge(alpha=alpha).fit(Xa, phq[tr]).predict(Xb)


def unit(v, b, F, tr, te, y, phq, dom, seed):
    X, centre = F[(v, b)], v == 'P2_fewshot_domain'
    folds = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(tr, y[tr]))
    oofs = {al: np.zeros(len(tr)) for al in A_GRID}
    for a, c in folds:
        for al in A_GRID:
            oofs[al][c] = fit_predict(X, tr[a], tr[c], phq, dom, centre, al)
    best = max(A_GRID, key=lambda al: roc_auc_score(y[tr], oofs[al]))
    return oofs[best], fit_predict(X, tr, te, phq, dom, centre, best)


def main():
    df, X7 = load_pool()
    pids = df['participant_id'].values
    y, phq, dom, gen = df['y'].values, df['phq'].values, df['domain'].values, df['gender'].values
    F = build(pids)
    Xt, n = X7['text_emo'], len(y)
    fused = {v: np.zeros((len(SEEDS), n)) for v in VARIANTS}
    voice = {v: np.zeros((len(SEEDS), n)) for v in VARIANTS}
    bins = {(v, r): np.zeros((len(SEEDS), n), dtype=int) for v in VARIANTS for r in ('max_acc', 'youden')}
    for si, seed in enumerate(SEEDS):
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y, groups=pids)):
            t1, rs = time.time(), seed * 100 + fold
            units = [(v, b) for v in VARIANTS for b in BASES]
            res = dict(zip(units, Parallel(n_jobs=N_JOBS, backend='threading')(
                delayed(unit)(v, b, F, tr, te, y, phq, dom, rs) for v, b in units)))
            t_in, t_out = text_unit(Xt, tr, te, y, phq, rs)
            t_in_c, t_out_c = to_cdf(t_in, t_in), to_cdf(t_in, t_out)
            for v in VARIANTS:
                v_in = np.mean([to_cdf(res[(v, b)][0], res[(v, b)][0]) for b in BASES], axis=0)
                v_out = np.mean([to_cdf(res[(v, b)][0], res[(v, b)][1]) for b in BASES], axis=0)
                meta = LogisticRegression(C=1.0).fit(np.column_stack([v_in, t_in_c]), y[tr])
                f_in = meta.predict_proba(np.column_stack([v_in, t_in_c]))[:, 1]
                f_out = meta.predict_proba(np.column_stack([v_out, t_out_c]))[:, 1]
                voice[v][si, te], fused[v][si, te] = v_out, f_out
                thr = thresholds(y[tr], f_in, dom[tr], per_domain=False)
                for r in ('max_acc', 'youden'):
                    bins[(v, r)][si, te] = apply_thr(f_out, dom[te], thr, r)
            print(f'  seed {seed} fold {fold} done in {time.time() - t1:.0f}s', flush=True)

    p7 = pd.read_csv(os.path.join(RD, 'phase7_oof_predictions.csv')).set_index('participant_id').loc[pids]['P4_fusion'].values
    r0 = fused['R0_mean'].mean(0)
    out = {'n': int(n), 'majority_class_accuracy': float(1 - y.mean()), 'variants': {}}
    for v in VARIANTS:
        vs, f = voice[v].mean(0), fused[v].mean(0)
        ci = compute_cluster_bootstrap_ci(y, f, n_bootstraps=2000, seed=42)
        pa, da, dal, dau = compute_paired_cluster_bootstrap_delta(y, f, r0, n_bootstraps=5000, seed=7)
        pp, dp, dpl, dpu = compute_paired_cluster_bootstrap_delta(y, f, p7, n_bootstraps=5000, seed=7)
        r = {'voice_auc': float(roc_auc_score(y, vs)), 'fused_auc': float(roc_auc_score(y, f)), 'fused_ci95': list(ci),
             'delta_vs_R0': da, 'delta_vs_R0_ci95': [dal, dau], 'p_gt_R0': pa,
             'delta_vs_P4': dp, 'delta_vs_P4_ci95': [dpl, dpu], 'operating_points': {},
             'subgroup_auc': {g: float(roc_auc_score(y[m], f[m])) for g, m in
                              [('woz', dom == 'woz'), ('avatar', dom == 'avatar'), ('female', gen == 'female'), ('male', gen == 'male')]}}
        for rule in ('max_acc', 'youden'):
            ms = [binary_metrics(y, b) for b in bins[(v, rule)]]
            r['operating_points'][rule] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        out['variants'][v] = r
        op, s = r['operating_points'], r['subgroup_auc']
        print(f'{v:<20} voice {r["voice_auc"]:.3f} | fused {r["fused_auc"]:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] vs R0 {da:+.3f} '
              f'[{dal:+.3f}, {dau:+.3f}] | acc {op["max_acc"]["accuracy"]:.1%} bal {op["max_acc"]["balanced_accuracy"]:.1%} | '
              f'youden sens {op["youden"]["sensitivity"]:.0%} | avatar {s["avatar"]:.3f} woz {s["woz"]:.3f}', flush=True)
    json.dump(out, open(os.path.join(RD, 'phase8b_results.json'), 'w'), indent=2)
    print('saved phase8b_results.json', flush=True)


if __name__ == '__main__':
    main()
