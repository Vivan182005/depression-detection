"""
Script: experiments/research_v2/run_phase9_llm.py
Phase 9 (prereg_phase9_llm_phq.md): adds zero-shot LLM PHQ-8 item scores to the Phase 8 A0 voice + emotion-RoBERTa
fusion. Same N = 272 repeated nested CV, folds, seeds and thresholds as Phase 8b. Writes phase9_results.json.
"""
import os
import sys
import json
import time
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
from run_phase7_experiments import load_pool, thresholds, apply_thr, binary_metrics, to_cdf
from run_phase6a_experiment import compute_cluster_bootstrap_ci, compute_paired_cluster_bootstrap_delta
from run_phase8_experiments import BASES, SEEDS, N_JOBS, text_unit
from run_phase8b_preprocessing import build, unit
from score_phq_llm import ITEMS, OUT as LLM_CSV, load_transcripts

FUSIONS = {'R0_voice_text': ('voice', 'text'), 'F1_voice_text_llm': ('voice', 'text', 'llm'), 'F2_voice_llm': ('voice', 'llm'),
           'L1_llm_ridge': ('llm',)}
DISCLOSURE = r"diagnosed with (depression|ptsd)|i('m| am) depressed|i have depression"


def main():
    df, X7 = load_pool()
    pids = df['participant_id'].values
    y, phq, dom, gen = df['y'].values, df['phq'].values, df['domain'].values, df['gender'].values
    L = pd.read_csv(LLM_CSV).drop_duplicates('participant_id').set_index('participant_id').loc[pids]
    Xl = L[ITEMS].values.astype(float)
    print(f'LLM rows {len(L)}, fully parsed {(L.n_items_parsed == 8).sum()}, missing cells {np.isnan(Xl).sum()}', flush=True)
    Xl = np.where(np.isnan(Xl), np.nanmedian(Xl, 0), Xl)    # label-free fill (column median)
    llm_sum = Xl.sum(1)
    F = build(pids)
    Xt, n = X7['text_emo'], len(y)
    fused = {f: np.zeros((len(SEEDS), n)) for f in FUSIONS}
    bins = {(f, r): np.zeros((len(SEEDS), n), dtype=int) for f in FUSIONS for r in ('max_acc', 'youden')}
    l0_bins = {r: np.zeros((len(SEEDS), n), dtype=int) for r in ('max_acc', 'youden')}
    for si, seed in enumerate(SEEDS):
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y, groups=pids)):
            t1, rs = time.time(), seed * 100 + fold
            res = Parallel(n_jobs=N_JOBS, backend='threading')(delayed(unit)('R0_mean', b, F, tr, te, y, phq, dom, rs) for b in BASES)
            v_in = np.mean([to_cdf(r[0], r[0]) for r in res], axis=0)
            v_out = np.mean([to_cdf(r[0], r[1]) for r in res], axis=0)
            t_in, t_out = text_unit(Xt, tr, te, y, phq, rs)
            l_in, l_out = text_unit(Xl, tr, te, y, phq, rs)
            S = {'voice': (v_in, v_out), 'text': (to_cdf(t_in, t_in), to_cdf(t_in, t_out)), 'llm': (to_cdf(l_in, l_in), to_cdf(l_in, l_out))}
            for f, parts in FUSIONS.items():
                A = np.column_stack([S[p][0] for p in parts])
                B = np.column_stack([S[p][1] for p in parts])
                meta = LogisticRegression(C=1.0).fit(A, y[tr])
                f_in, f_out = meta.predict_proba(A)[:, 1], meta.predict_proba(B)[:, 1]
                fused[f][si, te] = f_out
                thr = thresholds(y[tr], f_in, dom[tr], per_domain=False)
                for r in ('max_acc', 'youden'):
                    bins[(f, r)][si, te] = apply_thr(f_out, dom[te], thr, r)
            thr0 = thresholds(y[tr], llm_sum[tr], dom[tr], per_domain=False)      # L0: zero-training score, in-fold threshold
            for r in ('max_acc', 'youden'):
                l0_bins[r][si, te] = apply_thr(llm_sum[te], dom[te], thr0, r)
            print(f'  seed {seed} fold {fold} done in {time.time() - t1:.0f}s', flush=True)

    T = load_transcripts()
    no_disc = np.array([re.search(DISCLOSURE, T[p].lower()) is None for p in pids])
    r0 = fused['R0_voice_text'].mean(0)
    p7 = pd.read_csv(os.path.join(RD, 'phase7_oof_predictions.csv')).set_index('participant_id').loc[pids]['P4_fusion'].values

    def report(name, s, B):
        ci = compute_cluster_bootstrap_ci(y, s, n_bootstraps=2000, seed=42)
        pa, da, dal, dau = compute_paired_cluster_bootstrap_delta(y, s, r0, n_bootstraps=5000, seed=7)
        pp, dp, dpl, dpu = compute_paired_cluster_bootstrap_delta(y, s, p7, n_bootstraps=5000, seed=7)
        r = {'auc': float(roc_auc_score(y, s)), 'auc_ci95': list(ci), 'delta_vs_R0': da, 'delta_vs_R0_ci95': [dal, dau],
             'delta_vs_P4': dp, 'delta_vs_P4_ci95': [dpl, dpu],
             'auc_without_self_disclosure': float(roc_auc_score(y[no_disc], s[no_disc])),
             'subgroup_auc': {g: float(roc_auc_score(y[m], s[m])) for g, m in
                              [('woz', dom == 'woz'), ('avatar', dom == 'avatar'), ('female', gen == 'female'), ('male', gen == 'male')]},
             'operating_points': {}}
        for rule, BB in B.items():
            ms = [dict(binary_metrics(y, b), mcc=float(matthews_corrcoef(y, b)), macro_f1=float(f1_score(y, b, average='macro'))) for b in BB]
            r['operating_points'][rule] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        op = r['operating_points']
        print(f'{name:<20} AUC {r["auc"]:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] vs R0 {da:+.3f} [{dal:+.3f}, {dau:+.3f}] vs P4 {dp:+.3f} | '
              f'acc {op["max_acc"]["accuracy"]:.1%} bal {op["max_acc"]["balanced_accuracy"]:.1%} MCC {op["max_acc"]["mcc"]:.3f} '
              f'mF1 {op["max_acc"]["macro_f1"]:.3f} | youden sens {op["youden"]["sensitivity"]:.0%} bal {op["youden"]["balanced_accuracy"]:.1%} | '
              f'no-disclosure AUC {r["auc_without_self_disclosure"]:.3f} | avatar {r["subgroup_auc"]["avatar"]:.3f}', flush=True)
        return r

    out = {'n': int(n), 'majority_class_accuracy': float(1 - y.mean()), 'n_without_self_disclosure': int(no_disc.sum()),
           'llm_item_auc': {k: float(roc_auc_score(y, Xl[:, i])) for i, k in enumerate(ITEMS)},
           'llm_sum_phq_spearman': float(pd.Series(llm_sum).corr(pd.Series(phq), method='spearman')), 'analyses': {}}
    print('LLM item AUCs: ' + ', '.join(f'{k} {v:.3f}' for k, v in out['llm_item_auc'].items()), flush=True)
    print(f'LLM sum vs PHQ-8 Spearman {out["llm_sum_phq_spearman"]:.3f}', flush=True)
    out['analyses']['L0_llm_sum_zero_shot'] = report('L0_llm_sum', llm_sum, l0_bins)
    for f in FUSIONS:
        out['analyses'][f] = report(f, fused[f].mean(0), {r: bins[(f, r)] for r in ('max_acc', 'youden')})
    json.dump(out, open(os.path.join(RD, 'phase9_results.json'), 'w'), indent=2)
    print('saved phase9_results.json', flush=True)


if __name__ == '__main__':
    main()
