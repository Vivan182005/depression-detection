"""
Script: experiments/research_v2/run_phase8_experiments.py
Phase 8 (prereg_phase8_fewshot_augmentation.md): augmentation, few-shot (prototypical network) and one-shot-style
(nearest prototype, kNN) methods on the Phase 7 voice bases, each fused with emotion-RoBERTa text as in P4.
Same N = 272 repeated nested CV as Phase 7. Writes phase8_results.json and phase8_oof_predictions.csv.
"""
import os
import sys
import json
import time
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
RD = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
for p in [_REPO_ROOT, RD]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_phase7_experiments import load_pool, thresholds, apply_thr, binary_metrics, to_cdf, A_GRID
from run_phase6a_experiment import compute_cluster_bootstrap_ci, compute_paired_cluster_bootstrap_delta

STORE = os.path.join(RD, 'feature_store_8')
BASES = ['w2v2_L6', 'hubert_large_er_L19', 'hubert_large_er_L20']
METHODS = ['A0_ridge', 'A1_segbag', 'A2_segbag_audioaug', 'A3_aug_tta', 'A4_mixup', 'D1_balanced',
           'C1_nearest_prototype', 'C2_knn', 'B1_protonet']
SEEDS = (0, 1, 2, 3, 4)
N_VIEWS, VIEW_FRAC, MAX_SEGS = 10, 0.6, 40
N_JOBS = 6
_INIT_LOCK = threading.Lock()   # torch's global RNG is shared across threads


# ----------------------------------------------------------------------------- data
def build(pids):
    """Per participant and base: clean/aug frame-weighted means, 10 bagged views per pass, <= 40 clean segments."""
    D = {b: {'mean': [], 'mean_aug': [], 'views': [], 'views_aug': [], 'segs': []} for b in BASES}
    for p in pids:
        c, a = np.load(os.path.join(STORE, f'{p}_clean.npz')), np.load(os.path.join(STORE, f'{p}_aug.npz'))
        f = c['frames'].astype(np.float64)
        rng = np.random.RandomState(p)
        S = len(f)
        subsets = [rng.choice(S, max(1, int(round(VIEW_FRAC * S))), replace=False) for _ in range(N_VIEWS)]
        seg_pick = rng.choice(S, min(S, MAX_SEGS), replace=False)
        for b in BASES:
            for key, src in (('mean', c), ('mean_aug', a)):
                D[b][key].append((f / f.sum()) @ src[b])
            D[b]['views'].append(np.stack([(f[s] / f[s].sum()) @ c[b][s] for s in subsets]))
            D[b]['views_aug'].append(np.stack([(f[s] / f[s].sum()) @ a[b][s] for s in subsets]))
            D[b]['segs'].append(c[b][seg_pick].astype(np.float32))
    for b in BASES:
        for k in ('mean', 'mean_aug', 'views', 'views_aug'):
            D[b][k] = np.array(D[b][k], dtype=np.float32)
    return D


# ----------------------------------------------------------------------------- learners
def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others]


def ridge_variant_fit_predict(Db, tr, te, phq, y, variant, alpha, seed):
    """Train a ridge on participants `tr` (with the variant's augmentation) and score participants `te`."""
    Xs, ts, ws = [Db['mean'][tr]], [phq[tr]], [np.ones(len(tr))]
    if variant in ('A1_segbag', 'A2_segbag_audioaug', 'A3_aug_tta'):
        Xs.append(Db['views'][tr].reshape(-1, Db['views'].shape[-1]))
        ts.append(np.repeat(phq[tr], N_VIEWS))
        ws.append(np.ones(len(tr) * N_VIEWS))
    if variant in ('A2_segbag_audioaug', 'A3_aug_tta'):
        Xs.append(Db['mean_aug'][tr])
        ts.append(phq[tr])
        ws.append(np.ones(len(tr)))
        Xs.append(Db['views_aug'][tr].reshape(-1, Db['views_aug'].shape[-1]))
        ts.append(np.repeat(phq[tr], N_VIEWS))
        ws.append(np.ones(len(tr) * N_VIEWS))
    if variant == 'A4_mixup':
        rng = np.random.RandomState(seed)
        i, j = rng.randint(0, len(tr), 2 * len(tr)), rng.randint(0, len(tr), 2 * len(tr))
        lam = rng.beta(0.4, 0.4, 2 * len(tr))
        Xm = Db['mean'][tr]
        Xs.append(lam[:, None] * Xm[i] + (1 - lam[:, None]) * Xm[j])
        ts.append(lam * phq[tr][i] + (1 - lam) * phq[tr][j])
        ws.append(np.ones(2 * len(tr)))
    if variant == 'D1_balanced':
        pos = y[tr].mean()
        ws = [np.where(y[tr] == 1, 0.5 / pos, 0.5 / (1 - pos))]
    X, t, w = np.concatenate(Xs), np.concatenate(ts), np.concatenate(ws)
    Xa, Xb, Xc = standardise(X, Db['mean'][te], Db['mean_aug'][te])
    m = Ridge(alpha=alpha).fit(Xa, t, sample_weight=w)
    s = m.predict(Xb)
    if variant == 'A3_aug_tta':
        s = 0.5 * (s + m.predict(Xc))
    return s


def pca_space(Xtr, Xte, seed):
    Xa, Xb = standardise(Xtr, Xte)
    pca = PCA(n_components=min(32, len(Xtr) - 1), random_state=seed).fit(Xa)
    A, B = pca.transform(Xa), pca.transform(Xb)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return A, B


def nearest_prototype(Db, tr, te, y, seed):
    A, B = pca_space(Db['mean'][tr], Db['mean'][te], seed)
    p1, p0 = A[y[tr] == 1].mean(0), A[y[tr] == 0].mean(0)
    p1, p0 = p1 / np.linalg.norm(p1), p0 / np.linalg.norm(p0)
    return B @ p1 - B @ p0


def knn(Db, tr, te, y, k, seed):
    A, B = pca_space(Db['mean'][tr], Db['mean'][te], seed)
    sim = B @ A.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    return y[tr][idx].mean(1) + 1e-3 * np.take_along_axis(sim, idx, 1).mean(1)


class Encoder(nn.Module):
    """d -> 128 -> LN -> ReLU -> Dropout(0.2) -> 64. Dropout draws from a per-model generator so threads stay reproducible."""
    def __init__(self, d, gen):
        super().__init__()
        self.inp, self.ln, self.out, self.gen = nn.Linear(d, 128), nn.LayerNorm(128), nn.Linear(128, 64), gen

    def forward(self, x):
        h = torch.relu(self.ln(self.inp(x)))
        if self.training:
            h = h * (torch.rand(h.shape, generator=self.gen) >= 0.2) / 0.8
        return self.out(h)


def protonet(Db, tr, te, y, seed, episodes=300, shot=5, query=5):
    """Prototypical network over segment embeddings; participant embedding = mean of encoded segments."""
    rng = np.random.RandomState(seed)
    segs = Db['segs']
    allseg = np.concatenate([segs[i] for i in tr])
    mu, sd = allseg.mean(0), allseg.std(0) + 1e-6
    T = {i: torch.tensor((segs[i] - mu) / sd, dtype=torch.float32) for i in list(tr) + list(te)}
    with _INIT_LOCK:
        torch.manual_seed(seed)
        enc = Encoder(allseg.shape[1], torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    pos, neg = [i for i in tr if y[i] == 1], [i for i in tr if y[i] == 0]
    embed = lambda ids: torch.stack([enc(T[i]).mean(0) for i in ids])
    enc.train()
    for _ in range(episodes):
        sp = rng.choice(pos, shot + query, replace=len(pos) < shot + query)
        sn = rng.choice(neg, shot + query, replace=len(neg) < shot + query)
        protos = torch.stack([embed(sn[:shot]).mean(0), embed(sp[:shot]).mean(0)])
        q = embed(np.concatenate([sn[shot:], sp[shot:]]))
        tgt = torch.tensor([0] * query + [1] * query)
        loss = nn.functional.cross_entropy(-torch.cdist(q, protos) ** 2, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        protos = torch.stack([embed(neg).mean(0), embed(pos).mean(0)])
        logits = -torch.cdist(embed(list(te)), protos) ** 2
        return torch.softmax(logits, 1)[:, 1].numpy()


# ----------------------------------------------------------------------------- inner CV wrapper
def method_unit(method, base, D, tr, te, y, phq, seed):
    """Returns (inner OOF scores on tr, outer scores on te) for one method x base, tuning inside tr only."""
    torch.set_num_threads(1)
    Db = D[base]
    folds = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(tr, y[tr]))

    def run(a_tr, a_te, hp):
        if method == 'C1_nearest_prototype':
            return nearest_prototype(Db, a_tr, a_te, y, seed)
        if method == 'C2_knn':
            return knn(Db, a_tr, a_te, y, hp, seed)
        if method == 'B1_protonet':
            return protonet(Db, a_tr, a_te, y, seed)
        return ridge_variant_fit_predict(Db, a_tr, a_te, phq, y, 'A0_ridge' if method == 'A0_ridge' else method, hp, seed)

    grid = [5, 11, 21] if method == 'C2_knn' else ([None] if method in ('C1_nearest_prototype', 'B1_protonet') else A_GRID)
    oofs = {hp: np.zeros(len(tr)) for hp in grid}
    for a, b in folds:
        for hp in grid:
            oofs[hp][b] = run(tr[a], tr[b], hp)
    best = max(grid, key=lambda hp: roc_auc_score(y[tr], oofs[hp]))
    return oofs[best], run(tr, te, best)


def text_unit(Xt, tr, te, y, phq, seed):
    folds = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(tr, y[tr]))
    oofs = {a: np.zeros(len(tr)) for a in A_GRID}
    for a_, b_ in folds:
        Xa, Xb = standardise(Xt[tr[a_]], Xt[tr[b_]])
        for al in A_GRID:
            oofs[al][b_] = Ridge(alpha=al).fit(Xa, phq[tr[a_]]).predict(Xb)
    best = max(A_GRID, key=lambda al: roc_auc_score(y[tr], oofs[al]))
    Xa, Xb = standardise(Xt[tr], Xt[te])
    return oofs[best], Ridge(alpha=best).fit(Xa, phq[tr]).predict(Xb)


# ----------------------------------------------------------------------------- main
def main():
    df, X7 = load_pool()
    pids = df['participant_id'].values
    y, phq, dom, gen = df['y'].values, df['phq'].values, df['domain'].values, df['gender'].values
    t0 = time.time()
    D = build(pids)
    print(f'Built segment views for {len(pids)} participants in {time.time() - t0:.0f}s', flush=True)
    Xt = X7['text_emo']
    n = len(y)
    voice = {m: np.zeros((len(SEEDS), n)) for m in METHODS}
    fused = {m: np.zeros((len(SEEDS), n)) for m in METHODS}
    bins = {(m, r): np.zeros((len(SEEDS), n), dtype=int) for m in METHODS for r in ('max_acc', 'youden')}
    for si, seed in enumerate(SEEDS):
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y, groups=pids)):
            t1, rs = time.time(), seed * 100 + fold
            units = [(m, b) for m in METHODS for b in BASES]
            outs = Parallel(n_jobs=N_JOBS, backend='threading')(
                delayed(method_unit)(m, b, D, tr, te, y, phq, rs) for m, b in units)
            res = dict(zip(units, outs))
            t_in, t_out = text_unit(Xt, tr, te, y, phq, rs)
            t_in_c, t_out_c = to_cdf(t_in, t_in), to_cdf(t_in, t_out)
            for m in METHODS:
                v_in = np.mean([to_cdf(res[(m, b)][0], res[(m, b)][0]) for b in BASES], axis=0)
                v_out = np.mean([to_cdf(res[(m, b)][0], res[(m, b)][1]) for b in BASES], axis=0)
                meta = LogisticRegression(C=1.0).fit(np.column_stack([v_in, t_in_c]), y[tr])
                f_in = meta.predict_proba(np.column_stack([v_in, t_in_c]))[:, 1]
                f_out = meta.predict_proba(np.column_stack([v_out, t_out_c]))[:, 1]
                voice[m][si, te], fused[m][si, te] = v_out, f_out
                thr = thresholds(y[tr], f_in, dom[tr], per_domain=False)
                for r in ('max_acc', 'youden'):
                    bins[(m, r)][si, te] = apply_thr(f_out, dom[te], thr, r)
            print(f'  seed {seed} fold {fold} done in {time.time() - t1:.0f}s', flush=True)

    p7 = pd.read_csv(os.path.join(RD, 'phase7_oof_predictions.csv')).set_index('participant_id').loc[pids]['P4_fusion'].values
    out = {'n': int(n), 'majority_class_accuracy': float(1 - y.mean()),
           'reference_phase7_P4_auc': float(roc_auc_score(y, p7)), 'methods': {}}
    oof = df[['participant_id', 'domain', 'gender']].copy()
    for m in METHODS:
        v, f = voice[m].mean(0), fused[m].mean(0)
        oof[f'{m}_voice'], oof[f'{m}_fused'] = v, f
        ci = compute_cluster_bootstrap_ci(y, f, n_bootstraps=2000, seed=42)
        pv, d, dl, du = compute_paired_cluster_bootstrap_delta(y, f, p7, n_bootstraps=5000, seed=7)
        pa, da, dal, dau = compute_paired_cluster_bootstrap_delta(y, f, fused['A0_ridge'].mean(0), n_bootstraps=5000, seed=7)
        r = {'voice_auc': float(roc_auc_score(y, v)), 'fused_auc': float(roc_auc_score(y, f)), 'fused_ci95': list(ci),
             'delta_vs_P4': d, 'delta_vs_P4_ci95': [dl, du], 'p_gt_P4': pv,
             'delta_vs_A0': da, 'delta_vs_A0_ci95': [dal, dau], 'p_gt_A0': pa, 'operating_points': {}}
        for rule in ('max_acc', 'youden'):
            ms = [binary_metrics(y, b) for b in bins[(m, rule)]]
            r['operating_points'][rule] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        r['subgroup_auc'] = {g: float(roc_auc_score(y[msk], f[msk])) for g, msk in
                             [('woz', dom == 'woz'), ('avatar', dom == 'avatar'), ('female', gen == 'female'), ('male', gen == 'male')]}
        out['methods'][m] = r
        op = r['operating_points']
        print(f'{m:<22} voice AUC {r["voice_auc"]:.3f} | fused AUC {r["fused_auc"]:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] '
              f'vs P4 {d:+.3f} (p={pv:.2f}) vs A0 {da:+.3f} | acc {op["max_acc"]["accuracy"]:.1%} bal {op["max_acc"]["balanced_accuracy"]:.1%} | '
              f'youden bal {op["youden"]["balanced_accuracy"]:.1%} sens {op["youden"]["sensitivity"]:.0%}', flush=True)
    json.dump(out, open(os.path.join(RD, 'phase8_results.json'), 'w'), indent=2)
    oof.to_csv(os.path.join(RD, 'phase8_oof_predictions.csv'), index=False)
    print('saved phase8_results.json', flush=True)


if __name__ == '__main__':
    main()
