"""
Script: experiments/research_v2/domain_diagnostics_v2.py
Description: Domain diagnostics on canonical Feature Store v2 (Development cohorts only).
Strictly enforces:
1. Zero access to sealed cohort (assert_no_sealed_ids).
2. Evaluates domain shift (WoZ dev vs Avatar dev) on repaired representations:
   - Acoustic (real OpenSMILE eGeMAPSv02 88d + temporal 66d = 154d)
   - Wav2Vec2 (real facebook/wav2vec2-base-960h 768d)
   - BERT (normalized text representations 768d)
   - TriModal (1690d)
3. Computes Centroid Cosine Similarity, Linear MMD, and 5-fold StratifiedGroupKFold Domain Discriminator AUC.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids
from experiments.research_v2.dataset import load_development_dataset

def compute_linear_mmd(X1, X2):
    mean_diff = np.mean(X1, axis=0) - np.mean(X2, axis=0)
    return float(np.dot(mean_diff, mean_diff))

def compute_centroid_cosine_similarity(X1, X2):
    c1 = np.mean(X1, axis=0)
    c2 = np.mean(X2, axis=0)
    norm1 = np.linalg.norm(c1)
    norm2 = np.linalg.norm(c2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(c1, c2) / (norm1 * norm2))

def evaluate_domain_discriminator(X, y_domain, pids):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_domain))
    
    for train_idx, val_idx in sgkf.split(X, y_domain, groups=pids):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_domain[train_idx])
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
        
    auc = float(roc_auc_score(y_domain, oof_probs))
    return auc, oof_probs

def run_diagnostics_v2():
    print('Loading development cohorts from Feature Store v2...')
    data = load_development_dataset(cohort='combined', feature_version='v2')
    
    assert_no_sealed_ids(data['pids'], context='domain_diagnostics_v2')
    
    y_domain = np.array([1 if d == 'avatar' else 0 for d in data['domains']], dtype=np.int64)
    pids = data['pids']
    
    woz_mask = y_domain == 0
    avatar_mask = y_domain == 1
    
    print(f'Total sessions: {len(y_domain)} (WoZ: {woz_mask.sum()}, Avatar: {avatar_mask.sum()})')
    
    modalities = {
        'Acoustic_154d': data['X_ac'],
        'Wav2Vec2_768d': data['X_w2v'],
        'BERT_768d': data['X_bert'],
        'TriModal_1690d': data['X_tri']
    }
    
    results = {}
    print('
--- Representation Alignment Diagnostics (v2 Repaired) ---')
    for mod_name, X in modalities.items():
        X_woz = X[woz_mask]
        X_avatar = X[avatar_mask]
        
        centroid_cos = compute_centroid_cosine_similarity(X_woz, X_avatar)
        mmd = compute_linear_mmd(X_woz, X_avatar)
        auc, _ = evaluate_domain_discriminator(X, y_domain, pids)
        
        results[mod_name] = {
            'centroid_cosine_similarity': centroid_cos,
            'linear_mmd': mmd,
            'domain_discriminator_auc': auc
        }
        print(f'{mod_name:16s} | Centroid Cosine: {centroid_cos:.4f} | Linear MMD: {mmd:8.4f} | Domain AUC: {auc:.4f}')

    out_json = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'domain_diagnostics_v2.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'
Saved domain diagnostics to: {out_json}')
    return results

if __name__ == '__main__':
    run_diagnostics_v2()
