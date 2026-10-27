"""
Script: experiments/research_v2/run_holdout_239_33.py
Standing holdout rule (2026-10-27): every model is trained ONLY on the 239 non-sealed participants and tested once on
the 33 in sealed_ids.json. Inner-CV alphas, CDF maps, the stacker and the thresholds are all fit on the 239; nothing
touches the 33 until prediction. Seeds 0-4 vary only the inner CV and are averaged. Writes holdout_239_33_results.json.
Caveat: the 33 were already used once (2026-10-19) and the voice bases were chosen on all 272, so this is a repeat check.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score

RD = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(RD))
sys.path.insert(0, RD)
from run_phase7_experiments import load_pool, thresholds, apply_thr, binary_metrics, to_cdf
from run_phase8_experiments import BASES, SEEDS, text_unit
from run_phase8b_preprocessing import build, unit
from score_phq_llm import ITEMS, OUT as LLM_CSV

FUSIONS = {'R0_voice_text': ('voice', 'text'), 'L1_llm_ridge': ('llm',), 'F1_voice_text_llm': ('voice', 'text', 'llm'),
           'F2_voice_llm': ('voice', 'llm')}


def boot_ci(y, s, n=5000, seed=42):
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def perm_p(y, s, n=5000, seed=0):
    rng, a = np.random.default_rng(seed), roc_auc_score(y, s)
    return float((1 + sum(roc_auc_score(rng.permutation(y), s) >= a for _ in range(n))) / (n + 1))


def main():
    df, X7 = load_pool()
    pids = df['participant_id'].values
    y, phq, dom = df['y'].values, df['phq'].values, df['domain'].values
    sealed = set(json.load(open(os.path.join(ROOT, 'sealed_ids.json')))['sealed_pids'])
    te = np.where(np.isin(pids, list(sealed)))[0]
    tr = np.where(~np.isin(pids, list(sealed)))[0]
    assert len(tr) == 239 and len(te) == 33, (len(tr), len(te))
    L = pd.read_csv(LLM_CSV).drop_duplicates('participant_id').set_index('participant_id').loc[pids]
    Xl = L[ITEMS].values.astype(float)
    Xl = np.where(np.isnan(Xl), np.nanmedian(Xl[tr], 0), Xl)
    F, Xt = build(pids), X7['text_emo']
    f_in = {f: [] for f in FUSIONS}
    f_out = {f: [] for f in FUSIONS}
    for seed in SEEDS:
        res = [unit('R0_mean', b, F, tr, te, y, phq, dom, seed) for b in BASES]
        v_in = np.mean([to_cdf(r[0], r[0]) for r in res], axis=0)
        v_out = np.mean([to_cdf(r[0], r[1]) for r in res], axis=0)
        t_in, t_out = text_unit(Xt, tr, te, y, phq, seed)
        l_in, l_out = text_unit(Xl, tr, te, y, phq, seed)
        S = {'voice': (v_in, v_out), 'text': (to_cdf(t_in, t_in), to_cdf(t_in, t_out)), 'llm': (to_cdf(l_in, l_in), to_cdf(l_in, l_out))}
        for f, parts in FUSIONS.items():
            A = np.column_stack([S[p][0] for p in parts])
            B = np.column_stack([S[p][1] for p in parts])
            meta = LogisticRegression(C=1.0).fit(A, y[tr])
            f_in[f].append(meta.predict_proba(A)[:, 1])
            f_out[f].append(meta.predict_proba(B)[:, 1])
        print(f'  seed {seed} done', flush=True)

    yt = y[te]
    out = {'n_train': int(len(tr)), 'n_test': int(len(te)), 'test_depressed': int(yt.sum()),
           'test_majority_accuracy': float(1 - yt.mean()), 'models': {}, 'predictions': {'participant_id': pids[te].tolist(), 'y': yt.tolist()}}
    print(f'Holdout: train 239, test 33 ({yt.sum()} depressed), majority baseline {1 - yt.mean():.1%}', flush=True)
    for f in FUSIONS:
        si, so = np.mean(f_in[f], 0), np.mean(f_out[f], 0)
        thr = thresholds(y[tr], si, dom[tr], per_domain=False)
        r = {'auc': float(roc_auc_score(yt, so)), 'auc_ci95': boot_ci(yt, so), 'perm_p': perm_p(yt, so), 'operating_points': {}}
        for rule in ('max_acc', 'youden'):
            b = apply_thr(so, dom[te], thr, rule)
            r['operating_points'][rule] = dict(binary_metrics(yt, b), mcc=float(matthews_corrcoef(yt, b)),
                                               macro_f1=float(f1_score(yt, b, average='macro')), n_correct=int((b == yt).sum()))
        out['models'][f] = r
        out['predictions'][f] = so.tolist()
        m, yd = r['operating_points']['max_acc'], r['operating_points']['youden']
        print(f'{f:<20} AUC {r["auc"]:.3f} [{r["auc_ci95"][0]:.3f}, {r["auc_ci95"][1]:.3f}] perm p {r["perm_p"]:.4f} | '
              f'acc {m["accuracy"]:.1%} ({m["n_correct"]}/33) bal {m["balanced_accuracy"]:.1%} sens {m["sensitivity"]:.0%} '
              f'spec {m["specificity"]:.0%} | youden acc {yd["accuracy"]:.1%} bal {yd["balanced_accuracy"]:.1%} sens {yd["sensitivity"]:.0%}', flush=True)
    json.dump(out, open(os.path.join(RD, 'holdout_239_33_results.json'), 'w'), indent=2)
    print('saved holdout_239_33_results.json', flush=True)


if __name__ == '__main__':
    main()
