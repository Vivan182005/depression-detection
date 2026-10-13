"""
Script: experiments/research_v2/domain_diagnostics.py
Description: Development-only domain diagnostics comparing Human-WoZ (N=179 participants, 183 sessions)
versus Autonomous Avatar (N=61 participants, 61 sessions).
Computes:
1. Demographic & Clinical distributions (Gender, Age, PHQ-8, 3-class distribution).
2. Modality-specific representation divergence (Acoustic 154d, Wav2Vec2 768d, BERT 768d, TriModal 1690d).
3. Maximum Mean Discrepancy (Linear MMD) and Centroid Cosine Similarities.
4. Domain Discriminator AUC (StratifiedGroupKFold on PID) to quantify domain separation.
5. Top divergent acoustic features (eGeMAPS + temporal).
Strictly development-only: Zero access to the sealed cohort.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
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
    """Computes unbiased estimate of Linear Maximum Mean Discrepancy squared."""
    mean_diff = np.mean(X1, axis=0) - np.mean(X2, axis=0)
    return float(np.dot(mean_diff, mean_diff))

def compute_centroid_cosine_similarity(X1, X2):
    """Computes cosine similarity between centroids of two distributions."""
    c1 = np.mean(X1, axis=0)
    c2 = np.mean(X2, axis=0)
    norm1 = np.linalg.norm(c1)
    norm2 = np.linalg.norm(c2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(c1, c2) / (norm1 * norm2))

def evaluate_domain_discriminator(X, y_domain, pids):
    """
    5-fold StratifiedGroupKFold cross-validated Logistic Regression to predict domain (0=WoZ, 1=Avatar).
    AUC measures the degree of domain separability in feature space.
    """
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

def run_diagnostics():
    print("=" * 80)
    print("       PHASE 3: DEVELOPMENT-ONLY DOMAIN DIAGNOSTICS (WoZ vs AVATAR)     ")
    print("=" * 80)
    
    # Load WoZ and Avatar cohorts
    print("[1/4] Loading WoZ and Avatar development cohorts...")
    woz = load_development_dataset(cohort='woz_only')
    avatar = load_development_dataset(cohort='avatar_only')
    comb = load_development_dataset(cohort='combined')
    
    assert_no_sealed_ids(woz['pids'], context='woz_domain_diagnostics')
    assert_no_sealed_ids(avatar['pids'], context='avatar_domain_diagnostics')
    assert_no_sealed_ids(comb['pids'], context='comb_domain_diagnostics')
    
    n_woz_sess = len(woz['pids'])
    n_woz_pids = len(np.unique(woz['pids']))
    n_avatar_sess = len(avatar['pids'])
    n_avatar_pids = len(np.unique(avatar['pids']))
    
    print(f"  WoZ: {n_woz_sess} sessions ({n_woz_pids} unique participants)")
    print(f"  Avatar: {n_avatar_sess} sessions ({n_avatar_pids} unique participants)")
    print(f"  Combined: {len(comb['pids'])} sessions ({len(np.unique(comb['pids']))} unique participants)")
    
    # 1. Demographic Comparison
    print("\n[2/4] Analyzing Demographic and Clinical Distributions...")
    # Gender
    woz_males = sum(1 for g in woz['genders'] if g == 'male')
    woz_females = sum(1 for g in woz['genders'] if g == 'female')
    avatar_males = sum(1 for g in avatar['genders'] if g == 'male')
    avatar_females = sum(1 for g in avatar['genders'] if g == 'female')
    
    # Chi-squared test on gender proportions
    gender_contingency = [[woz_females, woz_males], [avatar_females, avatar_males]]
    chi2_gender, p_gender, _, _ = stats.chi2_contingency(gender_contingency)
    
    # Age
    woz_ages = woz['ages'][~np.isnan(woz['ages'])]
    avatar_ages = avatar['ages'][~np.isnan(avatar['ages'])]
    t_age, p_age = stats.ttest_ind(woz_ages, avatar_ages, equal_var=False)
    
    # PHQ-8 score
    woz_phq = woz['y_phq']
    avatar_phq = avatar['y_phq']
    mw_phq, p_phq = stats.mannwhitneyu(woz_phq, avatar_phq)
    
    # Class distribution
    woz_cls_counts = np.bincount(woz['y_class'], minlength=3).tolist()
    avatar_cls_counts = np.bincount(avatar['y_class'], minlength=3).tolist()
    cls_contingency = [woz_cls_counts, avatar_cls_counts]
    chi2_cls, p_cls, _, _ = stats.chi2_contingency(cls_contingency)
    
    demo_results = {
        'woz': {
            'sessions': n_woz_sess,
            'participants': n_woz_pids,
            'female_count': woz_females,
            'male_count': woz_males,
            'female_pct': round(woz_females / n_woz_sess * 100, 2),
            'age_mean': round(float(np.mean(woz_ages)), 2),
            'age_std': round(float(np.std(woz_ages)), 2),
            'phq8_mean': round(float(np.mean(woz_phq)), 2),
            'phq8_std': round(float(np.std(woz_phq)), 2),
            'class_distribution': woz_cls_counts,
            'depressed_pct': round((woz_cls_counts[1] + woz_cls_counts[2]) / n_woz_sess * 100, 2)
        },
        'avatar': {
            'sessions': n_avatar_sess,
            'participants': n_avatar_pids,
            'female_count': avatar_females,
            'male_count': avatar_males,
            'female_pct': round(avatar_females / n_avatar_sess * 100, 2),
            'age_mean': round(float(np.mean(avatar_ages)), 2),
            'age_std': round(float(np.std(avatar_ages)), 2),
            'phq8_mean': round(float(np.mean(avatar_phq)), 2),
            'phq8_std': round(float(np.std(avatar_phq)), 2),
            'class_distribution': avatar_cls_counts,
            'depressed_pct': round((avatar_cls_counts[1] + avatar_cls_counts[2]) / n_avatar_sess * 100, 2)
        },
        'tests': {
            'gender_chi2': round(float(chi2_gender), 4),
            'gender_p_value': float(p_gender),
            'age_t_stat': round(float(t_age), 4),
            'age_p_value': float(p_age),
            'phq8_mw_u': round(float(mw_phq), 4),
            'phq8_p_value': float(p_phq),
            'class_chi2': round(float(chi2_cls), 4),
            'class_p_value': float(p_cls)
        }
    }
    
    print(f"  Gender: WoZ {woz_females}F/{woz_males}M ({woz_females/n_woz_sess*100:.1f}% F) vs Avatar {avatar_females}F/{avatar_males}M ({avatar_females/n_avatar_sess*100:.1f}% F) | Chi2 p={p_gender:.4e}")
    print(f"  Age: WoZ {np.mean(woz_ages):.1f}±{np.std(woz_ages):.1f} vs Avatar {np.mean(avatar_ages):.1f}±{np.std(avatar_ages):.1f} | t-test p={p_age:.4e}")
    print(f"  PHQ-8: WoZ {np.mean(woz_phq):.2f}±{np.std(woz_phq):.2f} vs Avatar {np.mean(avatar_phq):.2f}±{np.std(avatar_phq):.2f} | MW-U p={p_phq:.4e}")
    print(f"  Classes: WoZ {woz_cls_counts} vs Avatar {avatar_cls_counts} | Chi2 p={p_cls:.4e}")
    
    # 2. Representation Divergence
    print("\n[3/4] Quantifying Feature Representation Divergence across Modalities...")
    modalities = {
        'acoustic_154d': (woz['X_ac'], avatar['X_ac'], comb['X_ac']),
        'wav2vec_768d': (woz['X_w2v'], avatar['X_w2v'], comb['X_w2v']),
        'text_bert_768d': (woz['X_bert'], avatar['X_bert'], comb['X_bert']),
        'trimodal_1690d': (woz['X_tri'], avatar['X_tri'], comb['X_tri']),
    }
    
    y_domain = (comb['domains'] == 'avatar').astype(int)
    pids_comb = comb['pids']
    
    rep_results = {}
    for mod_name, (X_w, X_a, X_c) in modalities.items():
        cos_sim = compute_centroid_cosine_similarity(X_w, X_a)
        mmd_sq = compute_linear_mmd(X_w, X_a)
        auc, _ = evaluate_domain_discriminator(X_c, y_domain, pids_comb)
        
        rep_results[mod_name] = {
            'dim': X_w.shape[1],
            'centroid_cosine_similarity': round(cos_sim, 4),
            'linear_mmd_sq': round(mmd_sq, 4),
            'domain_discriminator_auc': round(auc, 4)
        }
        print(f"  {mod_name:15s} | CosSim: {cos_sim:.4f} | MMD^2: {mmd_sq:10.4f} | Domain AUC: {auc:.4f}")
        
    # 3. Top Divergent Acoustic Features
    print("\n[4/4] Identifying Top Divergent Acoustic Features...")
    w_ac_mean = np.mean(woz['X_ac'], axis=0)
    w_ac_std = np.std(woz['X_ac'], axis=0)
    a_ac_mean = np.mean(avatar['X_ac'], axis=0)
    a_ac_std = np.std(avatar['X_ac'], axis=0)
    
    # Pooled standard deviation & Cohen's d
    s_pooled = np.sqrt(((len(woz['X_ac']) - 1) * w_ac_std**2 + (len(avatar['X_ac']) - 1) * a_ac_std**2) / (len(woz['X_ac']) + len(avatar['X_ac']) - 2))
    s_pooled = np.where(s_pooled == 0, 1e-6, s_pooled)
    cohens_d = np.abs(w_ac_mean - a_ac_mean) / s_pooled
    
    top_indices = np.argsort(cohens_d)[::-1][:15]
    top_features = []
    for idx in top_indices:
        feat_type = 'eGeMAPS' if idx < 88 else ('Temporal_Mean' if idx < 121 else 'Temporal_Std')
        sub_idx = idx if idx < 88 else (idx - 88 if idx < 121 else idx - 121)
        top_features.append({
            'feature_index': int(idx),
            'feature_category': feat_type,
            'sub_index': int(sub_idx),
            'cohens_d': round(float(cohens_d[idx]), 4),
            'woz_mean': round(float(w_ac_mean[idx]), 4),
            'avatar_mean': round(float(a_ac_mean[idx]), 4)
        })
        print(f"  Idx {idx:3d} ({feat_type:14s} #{sub_idx:2d}) | Cohen's d: {cohens_d[idx]:.4f} | WoZ: {w_ac_mean[idx]:.4f} vs Avatar: {a_ac_mean[idx]:.4f}")
        
    full_report = {
        'demographics': demo_results,
        'representations': rep_results,
        'top_divergent_acoustic_features': top_features
    }
    
    out_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'domain_diagnostics.json')
    with open(out_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved domain diagnostics report to {out_path}")
    return full_report

if __name__ == '__main__':
    run_diagnostics()
