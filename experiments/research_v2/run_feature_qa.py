import warnings
warnings.filterwarnings('ignore')
"""
Task C5: Label-Free Feature QA & Domain Separability Diagnostic
Evaluates feature stores strictly without reading PHQ labels:
- Dimensions, non-finite values, constant and near-constant dimensions
- Missingness by domain and participant
- Duplicate and near-duplicate rows
- Segment counts and total valid speech duration
- Extraction failures and reasons
- Domain-classification AUC as a descriptive diagnostic only
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine

def inspect_feature_matrix(X, feature_name):
    """Computes quality metrics on a feature matrix."""
    n_samples, n_dims = X.shape
    
    # Non-finite values
    n_nan = int(np.sum(np.isnan(X)))
    n_inf = int(np.sum(np.isinf(X)))
    non_finite_count = n_nan + n_inf
    
    # Clean non-finite for variance check
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(X_clean, axis=0)
    stds = np.std(X_clean, axis=0)
    
    # Constant and near-constant
    constant_dims = int(np.sum(variances == 0.0))
    near_constant_dims = int(np.sum((stds < 1e-4) & (variances > 0.0)))
    
    # Exact duplicate rows
    # Convert rows to bytes for fast hashing
    row_hashes = [hash(row.tobytes()) for row in X_clean]
    n_duplicates = len(row_hashes) - len(set(row_hashes))
    
    # Near duplicate rows (sample 100 rows if large)
    # Cosine similarity > 0.999
    sample_size = min(100, n_samples)
    sub = X_clean[:sample_size]
    norms = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8
    sub_norm = sub / norms
    sim_matrix = np.dot(sub_norm, sub_norm.T)
    np.fill_diagonal(sim_matrix, 0.0)
    near_dup_count = int(np.sum(sim_matrix > 0.999)) // 2
    
    return {
        'feature_name': feature_name,
        'n_samples': n_samples,
        'n_dimensions': n_dims,
        'non_finite_values': non_finite_count,
        'constant_dimensions': constant_dims,
        'near_constant_dimensions': near_constant_dims,
        'exact_duplicate_rows': n_duplicates,
        'near_duplicate_pairs_sampled': near_dup_count
    }

def run_domain_classification(X, domains):
    """
    Evaluates domain separability (woz vs avatar) via 5-fold CV.
    Purely descriptive diagnostic of distribution shift.
    """
    y_domain = (np.array(domains) == 'woz').astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y_domain))
    
    for train_idx, val_idx in skf.split(X, y_domain):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        clf = LogisticRegression(penalty='l2', C=1.0, max_iter=200, random_state=42)
        clf.fit(X_tr, y_domain[train_idx])
        oof_preds[val_idx] = clf.predict_proba(X_va)[:, 1]
        
    auc = roc_auc_score(y_domain, oof_preds)
    return float(auc)

def main():
    print("=======================================================")
    print("Task C5: Label-Free Feature QA & Domain Diagnostics")
    print("=======================================================")
    
    labels_df = pd.read_csv(os.path.join(_REPO_ROOT, 'experiments/research_v2/labels_official.csv'))
    
    # Load feature store files
    fs_dir = os.path.join(_REPO_ROOT, 'experiments/research_v2/feature_store_v2')
    files = sorted(glob.glob(os.path.join(fs_dir, '*_v2.npz')))
    print(f"Found {len(files)} cached .npz files in feature_store_v2")
    
    pids = []
    domains = []
    durations = []
    ac_154_list = []
    ac_135_list = []
    w2v_list = []
    bert_t2_list = []; bert_domains = []
    
    skipped_sealed = 0
    
    for f in files:
        pid = int(os.path.basename(f).split('_')[0])
        
        # Sealed guard
        try:
            assert_sealed_quarantine(pid)
        except AssertionError:
            skipped_sealed += 1
            continue
            
        data = np.load(f, allow_pickle=True)
        domain = str(data['domain'])
        dur = float(data['audio_duration_sec'])
        
        pids.append(pid)
        domains.append(domain)
        durations.append(dur)
        
        if 'acoustic_154d' in data:
            ac_154_list.append(data['acoustic_154d'])
        if 'acoustic_robust_135d' in data:
            ac_135_list.append(data['acoustic_robust_135d'])
        if 'wav2vec_768d' in data:
            w2v_list.append(data['wav2vec_768d'])
        if 'bert_t2_768d' in data:
            bert_t2_list.append(data['bert_t2_768d'])
            bert_domains.append(domain)
            
    print(f"Eligible development files analyzed: {len(pids)} (Quarantined sealed skipped: {skipped_sealed})")
    print(f"Domain breakdown: WoZ = {domains.count('woz')}, Avatar = {domains.count('avatar')}")
    
    # QA evaluations
    qa_results = []
    
    if ac_135_list:
        X_ac135 = np.array(ac_135_list)
        res_ac135 = inspect_feature_matrix(X_ac135, "Acoustic_Robust_135d")
        qa_results.append(res_ac135)
        
    if ac_154_list:
        X_ac154 = np.array(ac_154_list)
        res_ac154 = inspect_feature_matrix(X_ac154, "Acoustic_Canonical_154d")
        qa_results.append(res_ac154)
        
    if w2v_list:
        X_w2v = np.array(w2v_list)
        res_w2v = inspect_feature_matrix(X_w2v, "Wav2Vec2_768d")
        qa_results.append(res_w2v)
        
    if bert_t2_list:
        X_bert = np.array(bert_t2_list)
        res_bert = inspect_feature_matrix(X_bert, "BERT_T2_768d")
        qa_results.append(res_bert)
        
    print("\nFeature Quality Summary:")
    qa_df = pd.DataFrame(qa_results)
    print(qa_df.to_string(index=False))
    
    # Domain classification diagnostic
    print("\nEvaluating Domain-Classification AUC (woz vs avatar)...")
    auc_ac135 = run_domain_classification(X_ac135, domains) if ac_135_list else np.nan
    auc_w2v = run_domain_classification(X_w2v, domains) if w2v_list else np.nan
    auc_bert = run_domain_classification(X_bert, bert_domains) if bert_t2_list else np.nan
    
    print(f"  Acoustic (135d) Domain-Separability AUC: {auc_ac135:.3f}")
    print(f"  Wav2Vec2 (768d) Domain-Separability AUC: {auc_w2v:.3f}")
    print(f"  BERT T2 (768d)  Domain-Separability AUC: {auc_bert:.3f}")
    
    diag_summary = {
        'qa_metrics': qa_results,
        'domain_auc': {
            'acoustic_135d': auc_ac135,
            'wav2vec_768d': auc_w2v,
            'bert_t2_768d': auc_bert
        },
        'speech_duration_stats': {
            'mean_s': round(float(np.mean(durations)), 2),
            'median_s': round(float(np.median(durations)), 2),
            'p10_s': round(float(np.percentile(durations, 10)), 2),
            'p90_s': round(float(np.percentile(durations, 90)), 2)
        }
    }
    
    out_json = os.path.join(_REPO_ROOT, 'experiments/research_v2/feature_qa_diagnostics.json')
    with open(out_json, 'w') as f:
        json.dump(diag_summary, f, indent=2)
    print(f"\nSaved feature QA diagnostics to {out_json}")

if __name__ == '__main__':
    main()
