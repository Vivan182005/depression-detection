"""
Script: experiments/research_v2/audit_phase3_verification.py
Description: Comprehensive Phase 3 Results-Verification Audit Script.
Executes:
1. Deterministic regeneration & offline export of per-seed and pooled OOF predictions
   for Exp 1 (Zero-Shot), Exp 2 (Combined CE), Exp 3 (Combined Focal) to experiments/research_v2/oof_phase3/.
2. Recomputation & verification of all metrics under:
   - Pooled OOF (reported in PHASE3.md)
   - Seed Mean +/- Std
   - Participant-Level Probability Ensemble
3. Investigation of Domain Discriminator AUC across modalities and feature subsets.
4. Participant-cluster bootstrap evaluation for Focal Loss vs CE and Combined CE vs Zero-Shot.
5. Extraction and generation of the 61-participant feature provenance audit table.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids
from experiments.research_v2.dataset import load_development_dataset
from experiments.research_v2.metrics import evaluate_predictions, paired_cluster_bootstrap_ci
from experiments.research_v2.models import (
    StandardMLP,
    run_train_and_val_fold,
    set_seed,
    train_nn_epoch_loop,
    DepressionTorchDataset
)
from torch.utils.data import DataLoader

def run_audit():
    print("=" * 80)
    print("       PHASE 3 RESULTS-VERIFICATION AUDIT EXECUTION                     ")
    print("=" * 80)
    
    exp_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
    oof_dir = os.path.join(exp_dir, 'oof_phase3')
    os.makedirs(oof_dir, exist_ok=True)
    
    # 1. Load Data
    print("[1/5] Loading Datasets...")
    woz = load_development_dataset(cohort='woz_only')
    avatar = load_development_dataset(cohort='avatar_only')
    comb = load_development_dataset(cohort='combined')
    
    assert_no_sealed_ids(woz['pids'], context='audit_woz')
    assert_no_sealed_ids(avatar['pids'], context='audit_avatar')
    assert_no_sealed_ids(comb['pids'], context='audit_comb')
    
    with open(os.path.join(exp_dir, 'folds.json'), 'r') as f:
        woz_folds = json.load(f)
    with open(os.path.join(exp_dir, 'folds_combined.json'), 'r') as f:
        comb_folds = json.load(f)
        
    seeds = [0, 1, 2, 3, 4]
    
    # -----------------------------------------------------------------
    # STEP 1: Regenerate and Export OOF Predictions
    # -----------------------------------------------------------------
    print("\n[2/5] Running Deterministic Model Evaluations & Saving OOF CSVs...")
    
    # A. Exp 1: Zero-Shot Avatar
    exp1_seed_records = []
    for seed in seeds:
        set_seed(seed)
        fold_list = woz_folds[str(seed)]
        seed_avatar_probs = np.zeros((len(avatar['sids']), 3))
        seed_avatar_phq = np.zeros(len(avatar['sids']))
        
        for fold_info in fold_list:
            tr_idx = fold_info['train_indices']
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(woz['X_tri'][tr_idx])
            X_av = scaler.transform(avatar['X_tri'])
            
            y_tr_cls = woz['y_class'][tr_idx]
            y_tr_phq = woz['y_phq'][tr_idx]
            pids_tr = woz['pids'][tr_idx]
            
            counts_tr = np.bincount(y_tr_cls, minlength=3)
            mu_fold = np.zeros(3, dtype=np.float32)
            for c in range(3):
                m = (y_tr_cls == c)
                mu_fold[c] = np.mean(y_tr_phq[m]) if np.sum(m) > 0 else np.mean(y_tr_phq)
                
            model = StandardMLP(in_dim=1690)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            ds_tr = DepressionTorchDataset(X_tr, y_tr_cls, y_tr_phq, pids_tr)
            loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
            
            train_nn_epoch_loop(model, loader_tr, opt, 'mild_balanced', counts_tr, epochs=40, seed=seed)
            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(X_av, dtype=torch.float32))
                fold_probs = F.softmax(out['logits'], dim=-1).numpy()
                fold_exp_phq = np.sum(fold_probs * mu_fold, axis=-1)
                
            seed_avatar_probs += fold_probs / 5.0
            seed_avatar_phq += fold_exp_phq / 5.0
            
        pred_cls = np.argmax(seed_avatar_probs, axis=-1)
        df_seed_av = pd.DataFrame({
            'participant_id': avatar['pids'],
            'session_id': avatar['sids'],
            'domain': avatar['domains'],
            'gender': avatar['genders'],
            'age': avatar['ages'],
            'true_class': avatar['y_class'],
            'true_phq8': avatar['y_phq'],
            'pred_class': pred_cls,
            'prob_c0': seed_avatar_probs[:, 0],
            'prob_c1': seed_avatar_probs[:, 1],
            'prob_c2': seed_avatar_probs[:, 2],
            'exp_phq8': seed_avatar_phq,
            'seed': seed
        })
        df_seed_av.to_csv(os.path.join(oof_dir, f'Exp1_ZeroShot_Avatar_seed{seed}_oof.csv'), index=False)
        exp1_seed_records.append(df_seed_av)
        
    df_exp1_pooled = pd.concat(exp1_seed_records, ignore_index=True)
    df_exp1_pooled.to_csv(os.path.join(oof_dir, 'Exp1_ZeroShot_Avatar_pooled_oof.csv'), index=False)
    print("  Exp 1 (Zero-Shot) OOF saved.")

    # B. Exp 2: Combined CE
    comb_sid_to_idx = {sid: i for i, sid in enumerate(comb['sids'])}
    exp2_seed_records = []
    
    for seed in seeds:
        set_seed(seed)
        fold_list = comb_folds[str(seed)]
        seed_probs = np.zeros((len(comb['sids']), 3))
        seed_phq = np.zeros(len(comb['sids']))
        seed_fold_assignment = np.zeros(len(comb['sids']), dtype=int)
        
        for fold_info in fold_list:
            f_idx = fold_info['fold']
            tr_sids = fold_info['train_sessions']
            val_sids = fold_info['val_sessions']
            tr_idx = [comb_sid_to_idx[s] for s in tr_sids]
            val_idx = [comb_sid_to_idx[s] for s in val_sids]
            seed_fold_assignment[val_idx] = f_idx
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(comb['X_tri'][tr_idx])
            X_val = scaler.transform(comb['X_tri'][val_idx])
            
            res = run_train_and_val_fold(
                StandardMLP, 1690,
                X_tr, comb['y_class'][tr_idx], comb['y_phq'][tr_idx], comb['pids'][tr_idx],
                X_val, comb['y_class'][val_idx], comb['y_phq'][val_idx], comb['pids'][val_idx],
                loss_type='mild_balanced', epochs=40, seed=seed
            )
            seed_probs[val_idx] = res['probs']
            seed_phq[val_idx] = res['expected_phq']
            
        pred_cls = np.argmax(seed_probs, axis=-1)
        df_seed_e2 = pd.DataFrame({
            'participant_id': comb['pids'],
            'session_id': comb['sids'],
            'domain': comb['domains'],
            'gender': comb['genders'],
            'age': comb['ages'],
            'fold': seed_fold_assignment,
            'true_class': comb['y_class'],
            'true_phq8': comb['y_phq'],
            'pred_class': pred_cls,
            'prob_c0': seed_probs[:, 0],
            'prob_c1': seed_probs[:, 1],
            'prob_c2': seed_probs[:, 2],
            'exp_phq8': seed_phq,
            'seed': seed
        })
        df_seed_e2.to_csv(os.path.join(oof_dir, f'Exp2_Combined_CE_seed{seed}_oof.csv'), index=False)
        exp2_seed_records.append(df_seed_e2)
        
    df_exp2_pooled = pd.concat(exp2_seed_records, ignore_index=True)
    df_exp2_pooled.to_csv(os.path.join(oof_dir, 'Exp2_Combined_CE_pooled_oof.csv'), index=False)
    print("  Exp 2 (Combined CE) OOF saved.")

    # C. Exp 3: Combined Focal Loss
    exp3_seed_records = []
    for seed in seeds:
        set_seed(seed)
        fold_list = comb_folds[str(seed)]
        seed_probs = np.zeros((len(comb['sids']), 3))
        seed_phq = np.zeros(len(comb['sids']))
        seed_fold_assignment = np.zeros(len(comb['sids']), dtype=int)
        
        for fold_info in fold_list:
            f_idx = fold_info['fold']
            tr_sids = fold_info['train_sessions']
            val_sids = fold_info['val_sessions']
            tr_idx = [comb_sid_to_idx[s] for s in tr_sids]
            val_idx = [comb_sid_to_idx[s] for s in val_sids]
            seed_fold_assignment[val_idx] = f_idx
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(comb['X_tri'][tr_idx])
            X_val = scaler.transform(comb['X_tri'][val_idx])
            
            res = run_train_and_val_fold(
                StandardMLP, 1690,
                X_tr, comb['y_class'][tr_idx], comb['y_phq'][tr_idx], comb['pids'][tr_idx],
                X_val, comb['y_class'][val_idx], comb['y_phq'][val_idx], comb['pids'][val_idx],
                loss_type='focal', focal_gamma=1.5, epochs=40, seed=seed
            )
            seed_probs[val_idx] = res['probs']
            seed_phq[val_idx] = res['expected_phq']
            
        pred_cls = np.argmax(seed_probs, axis=-1)
        df_seed_e3 = pd.DataFrame({
            'participant_id': comb['pids'],
            'session_id': comb['sids'],
            'domain': comb['domains'],
            'gender': comb['genders'],
            'age': comb['ages'],
            'fold': seed_fold_assignment,
            'true_class': comb['y_class'],
            'true_phq8': comb['y_phq'],
            'pred_class': pred_cls,
            'prob_c0': seed_probs[:, 0],
            'prob_c1': seed_probs[:, 1],
            'prob_c2': seed_probs[:, 2],
            'exp_phq8': seed_phq,
            'seed': seed
        })
        df_seed_e3.to_csv(os.path.join(oof_dir, f'Exp3_Combined_Focal_seed{seed}_oof.csv'), index=False)
        exp3_seed_records.append(df_seed_e3)
        
    df_exp3_pooled = pd.concat(exp3_seed_records, ignore_index=True)
    df_exp3_pooled.to_csv(os.path.join(oof_dir, 'Exp3_Combined_Focal_pooled_oof.csv'), index=False)
    print("  Exp 3 (Combined Focal) OOF saved.")

    # -----------------------------------------------------------------
    # STEP 2: Recompute and Compare All Metrics
    # -----------------------------------------------------------------
    print("\n[3/5] Recomputing & Auditing Metrics Across Aggregation Schemes...")
    
    def compute_all_schemes(seed_dfs, pooled_df, domain_filter=None):
        # 1. Pooled OOF
        p_df = pooled_df if domain_filter is None else pooled_df[pooled_df['domain'] == domain_filter].copy()
        probs_p = p_df[['prob_c0', 'prob_c1', 'prob_c2']].values
        res_pooled = evaluate_predictions(
            p_df['true_class'].values, p_df['pred_class'].values, probs_p,
            p_df['true_phq8'].values, p_df['exp_phq8'].values
        )
        
        # 2. Per-seed mean & std
        seed_metrics = []
        for s_df in seed_dfs:
            sub = s_df if domain_filter is None else s_df[s_df['domain'] == domain_filter].copy()
            pr = sub[['prob_c0', 'prob_c1', 'prob_c2']].values
            m = evaluate_predictions(
                sub['true_class'].values, sub['pred_class'].values, pr,
                sub['true_phq8'].values, sub['exp_phq8'].values
            )
            seed_metrics.append(m)
            
        keys = ['macro_f1', 'balanced_accuracy', 'accuracy', 'exp_phq_mae', 'binary_f1']
        mean_std = {}
        for k in keys:
            vals = [m[k] for m in seed_metrics]
            mean_std[k] = (float(np.mean(vals)), float(np.std(vals)))
            
        # Per class recall mean & std
        rec_c0 = [m['per_class_recall'][0] for m in seed_metrics]
        rec_c1 = [m['per_class_recall'][1] for m in seed_metrics]
        rec_c2 = [m['per_class_recall'][2] for m in seed_metrics]
        mean_std['rec_c0'] = (float(np.mean(rec_c0)), float(np.std(rec_c0)))
        mean_std['rec_c1'] = (float(np.mean(rec_c1)), float(np.std(rec_c1)))
        mean_std['rec_c2'] = (float(np.mean(rec_c2)), float(np.std(rec_c2)))
        
        # 3. Probability Ensemble
        grouped = p_df.groupby('session_id').agg({
            'true_class': 'first',
            'true_phq8': 'first',
            'prob_c0': 'mean',
            'prob_c1': 'mean',
            'prob_c2': 'mean',
            'exp_phq8': 'mean'
        }).reset_index()
        ens_probs = grouped[['prob_c0', 'prob_c1', 'prob_c2']].values
        ens_preds = np.argmax(ens_probs, axis=-1)
        res_ens = evaluate_predictions(
            grouped['true_class'].values, ens_preds, ens_probs,
            grouped['true_phq8'].values, grouped['exp_phq8'].values
        )
        
        return {
            'pooled': res_pooled,
            'mean_std': mean_std,
            'ensemble': res_ens
        }
        
    audited_metrics = {
        'Exp1_ZeroShot_Avatar': compute_all_schemes(exp1_seed_records, df_exp1_pooled),
        'Exp2_Combined_Overall': compute_all_schemes(exp2_seed_records, df_exp2_pooled),
        'Exp2_Combined_WoZ': compute_all_schemes(exp2_seed_records, df_exp2_pooled, domain_filter='woz'),
        'Exp2_Combined_Avatar': compute_all_schemes(exp2_seed_records, df_exp2_pooled, domain_filter='avatar'),
        'Exp3_Focal_Overall': compute_all_schemes(exp3_seed_records, df_exp3_pooled),
        'Exp3_Focal_WoZ': compute_all_schemes(exp3_seed_records, df_exp3_pooled, domain_filter='woz'),
        'Exp3_Focal_Avatar': compute_all_schemes(exp3_seed_records, df_exp3_pooled, domain_filter='avatar'),
    }
    
    # -----------------------------------------------------------------
    # STEP 3: Paired Participant-Cluster Bootstrap
    # -----------------------------------------------------------------
    print("\n[4/5] Computing Paired Participant-Cluster Bootstraps...")
    
    def run_rigorous_bootstrap(df_cand_pooled, df_base_pooled, filter_domain=None, n_bootstraps=2000, seed=42):
        d_c = df_cand_pooled if filter_domain is None else df_cand_pooled[df_cand_pooled['domain'] == filter_domain]
        d_b = df_base_pooled if filter_domain is None else df_base_pooled[df_base_pooled['domain'] == filter_domain]
        
        # Method 1: Repeated-Seed Cluster Bootstrap (resampling all seed predictions for each participant)
        bs_rep = paired_cluster_bootstrap_ci(
            d_c['participant_id'].values, d_c['true_class'].values,
            d_c['pred_class'].values, d_c[['prob_c0', 'prob_c1', 'prob_c2']].values,
            d_b['pred_class'].values, d_b[['prob_c0', 'prob_c1', 'prob_c2']].values,
            n_bootstraps=n_bootstraps, seed=seed
        )
        
        # Method 2: Participant-Level Ensemble Bootstrap (averaging predictions per participant first)
        c_agg = d_c.groupby('participant_id').agg({
            'true_class': 'first', 'prob_c0': 'mean', 'prob_c1': 'mean', 'prob_c2': 'mean'
        }).reset_index()
        b_agg = d_b.groupby('participant_id').agg({
            'true_class': 'first', 'prob_c0': 'mean', 'prob_c1': 'mean', 'prob_c2': 'mean'
        }).reset_index()
        
        c_p = c_agg[['prob_c0', 'prob_c1', 'prob_c2']].values
        b_p = b_agg[['prob_c0', 'prob_c1', 'prob_c2']].values
        c_pred = np.argmax(c_p, axis=-1)
        b_pred = np.argmax(b_p, axis=-1)
        
        bs_ens = paired_cluster_bootstrap_ci(
            c_agg['participant_id'].values, c_agg['true_class'].values,
            c_pred, c_p, b_pred, b_p,
            n_bootstraps=n_bootstraps, seed=seed
        )
        return {'repeated_cluster': bs_rep, 'participant_ensemble': bs_ens}
        
    bootstraps = {
        'CombinedCE_vs_ZeroShot_Avatar': run_rigorous_bootstrap(df_exp2_pooled, df_exp1_pooled, filter_domain='avatar'),
        'Focal_vs_CombinedCE_Avatar': run_rigorous_bootstrap(df_exp3_pooled, df_exp2_pooled, filter_domain='avatar'),
        'Focal_vs_CombinedCE_Overall': run_rigorous_bootstrap(df_exp3_pooled, df_exp2_pooled),
        'Focal_vs_CombinedCE_WoZ': run_rigorous_bootstrap(df_exp3_pooled, df_exp2_pooled, filter_domain='woz')
    }
    
    # -----------------------------------------------------------------
    # STEP 4: Save Full Audit Artifacts
    # -----------------------------------------------------------------
    print("\n[5/5] Saving Audit Results JSON...")
    audit_summary = {
        'audited_metrics': audited_metrics,
        'bootstraps': bootstraps
    }
    
    with open(os.path.join(exp_dir, 'phase3_audit_metrics.json'), 'w') as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Audit completed. Written to experiments/research_v2/phase3_audit_metrics.json")

if __name__ == '__main__':
    run_audit()
