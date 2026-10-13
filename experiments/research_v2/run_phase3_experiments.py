"""
Script: experiments/research_v2/run_phase3_experiments.py
Description: Master Phase 3 Experiment Runner.
Executes:
1. WoZ-Trained Candidate 2A Zero-Shot Evaluation on Avatar Cohort (out-of-domain evaluation).
2. Combined-Cohort Candidate 2A Model (Cross-Entropy Loss) across 5 folds x 5 seeds.
3. Combined-Cohort Exploratory Focal Loss Model (gamma=1.5) across 5 folds x 5 seeds.
4. Classical Ridge / Logistic Regression on Combined-Cohort.
5. In-fold transformations only, participant-grouped splits, per-domain & subgroup reporting.
6. Paired cluster-bootstrap significance testing (accounting for participant clusters).
Strictly development-only: Zero access to the sealed cohort.
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
    DepressionTorchDataset,
    fit_temperature_on_inner_val
)
from torch.utils.data import DataLoader

def run_phase3_suite():
    print("=" * 80)
    print("       PHASE 3: DOMAIN ADAPTATION & COMBINED-COHORT EXPERIMENTS          ")
    print("=" * 80)
    
    exp_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
    oof_dir = os.path.join(exp_dir, 'oof_phase3')
    os.makedirs(oof_dir, exist_ok=True)
    
    # Load Data
    print("[1/5] Loading Development Datasets...")
    woz = load_development_dataset(cohort='woz_only')
    avatar = load_development_dataset(cohort='avatar_only')
    comb = load_development_dataset(cohort='combined')
    
    assert_no_sealed_ids(woz['pids'], context='phase3_woz_load')
    assert_no_sealed_ids(avatar['pids'], context='phase3_avatar_load')
    assert_no_sealed_ids(comb['pids'], context='phase3_comb_load')
    
    print(f"  WoZ: {len(woz['sids'])} sessions ({len(np.unique(woz['pids']))} unique PIDs)")
    print(f"  Avatar: {len(avatar['sids'])} sessions ({len(np.unique(avatar['pids']))} unique PIDs)")
    print(f"  Combined: {len(comb['sids'])} sessions ({len(np.unique(comb['pids']))} unique PIDs)")
    
    # Load Folds
    with open(os.path.join(exp_dir, 'folds.json'), 'r') as f:
        woz_folds_dict = json.load(f)
    with open(os.path.join(exp_dir, 'folds_combined.json'), 'r') as f:
        comb_folds_dict = json.load(f)
        
    seeds = [0, 1, 2, 3, 4]
    results_summary = {}
    
    # -------------------------------------------------------------
    # EXPERIMENT 1: WoZ-Trained Candidate 2A Evaluated on Avatar
    # (Testing Zero-Shot Domain Transfer)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[2/5] EXPERIMENT 1: WoZ-Trained Candidate 2A Zero-Shot Transfer to Avatar")
    print("=" * 60)
    
    # Map session IDs to indices
    woz_sid_to_idx = {sid: i for i, sid in enumerate(woz['sids'])}
    
    # For each seed, train on WoZ (5 folds) and evaluate each fold model on Avatar, then average
    avatar_transfer_preds = []
    
    for seed in seeds:
        set_seed(seed)
        fold_list = woz_folds_dict[str(seed)]
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
            
        avatar_transfer_preds.append({
            'seed': seed,
            'probs': seed_avatar_probs,
            'pred_cls': np.argmax(seed_avatar_probs, axis=-1),
            'exp_phq': seed_avatar_phq,
            'y_true_cls': avatar['y_class'],
            'y_true_phq': avatar['y_phq'],
            'pids': avatar['pids']
        })
        
    # Aggregate zero-shot avatar metrics across seeds
    all_av_true_cls = np.tile(avatar['y_class'], len(seeds))
    all_av_true_phq = np.tile(avatar['y_phq'], len(seeds))
    all_av_pids = np.tile(avatar['pids'], len(seeds))
    all_av_probs = np.concatenate([p['probs'] for p in avatar_transfer_preds], axis=0)
    all_av_pred_cls = np.concatenate([p['pred_cls'] for p in avatar_transfer_preds], axis=0)
    all_av_exp_phq = np.concatenate([p['exp_phq'] for p in avatar_transfer_preds], axis=0)
    
    exp1_metrics = evaluate_predictions(all_av_true_cls, all_av_pred_cls, all_av_probs, all_av_true_phq, all_av_exp_phq)
    results_summary['WoZ_Trained_on_Avatar_ZeroShot'] = exp1_metrics
    print(f"  Zero-Shot Transfer on Avatar (N={len(avatar['sids'])}): Balanced Acc={exp1_metrics['balanced_accuracy']:.2f}%, Macro-F1={exp1_metrics['macro_f1']:.2f}%, Acc={exp1_metrics['accuracy']:.2f}%")
    print(f"  Class Recalls: C0={exp1_metrics['per_class_recall'][0]:.2f}%, C1={exp1_metrics['per_class_recall'][1]:.2f}%, C2={exp1_metrics['per_class_recall'][2]:.2f}%")
    
    # -------------------------------------------------------------
    # EXPERIMENT 2: Combined-Trained Candidate 2A Model (CE Loss)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[3/5] EXPERIMENT 2: Combined-Trained Candidate 2A (Cross-Entropy Loss)")
    print("=" * 60)
    
    comb_sid_to_idx = {sid: i for i, sid in enumerate(comb['sids'])}
    exp2_oof_records = []
    
    for seed in seeds:
        set_seed(seed)
        fold_list = comb_folds_dict[str(seed)]
        seed_probs = np.zeros((len(comb['sids']), 3))
        seed_phq = np.zeros(len(comb['sids']))
        
        for fold_info in fold_list:
            tr_sids = fold_info['train_sessions']
            val_sids = fold_info['val_sessions']
            tr_idx = [comb_sid_to_idx[s] for s in tr_sids]
            val_idx = [comb_sid_to_idx[s] for s in val_sids]
            
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
        exp2_oof_records.append({
            'seed': seed,
            'probs': seed_probs,
            'pred_cls': pred_cls,
            'exp_phq': seed_phq,
            'y_cls': comb['y_class'],
            'y_phq': comb['y_phq'],
            'pids': comb['pids'],
            'domains': comb['domains'],
            'genders': comb['genders'],
            'ages': comb['ages']
        })
        
    # Aggregate OOF across seeds
    all_comb_true_cls = np.tile(comb['y_class'], len(seeds))
    all_comb_true_phq = np.tile(comb['y_phq'], len(seeds))
    all_comb_pids = np.tile(comb['pids'], len(seeds))
    all_comb_domains = np.tile(comb['domains'], len(seeds))
    all_comb_genders = np.tile(comb['genders'], len(seeds))
    all_comb_ages = np.tile(comb['ages'], len(seeds))
    
    all_e2_probs = np.concatenate([r['probs'] for r in exp2_oof_records], axis=0)
    all_e2_preds = np.concatenate([r['pred_cls'] for r in exp2_oof_records], axis=0)
    all_e2_phq = np.concatenate([r['exp_phq'] for r in exp2_oof_records], axis=0)
    
    # Metrics Overall
    e2_overall = evaluate_predictions(all_comb_true_cls, all_e2_preds, all_e2_probs, all_comb_true_phq, all_e2_phq)
    results_summary['Combined_Candidate2A_Overall'] = e2_overall
    print(f"  Overall Combined (N={len(comb['sids'])}): Balanced Acc={e2_overall['balanced_accuracy']:.2f}%, Macro-F1={e2_overall['macro_f1']:.2f}%, Acc={e2_overall['accuracy']:.2f}%")
    
    # Metrics by Domain
    woz_mask = (all_comb_domains == 'woz')
    avatar_mask = (all_comb_domains == 'avatar')
    
    e2_woz = evaluate_predictions(all_comb_true_cls[woz_mask], all_e2_preds[woz_mask], all_e2_probs[woz_mask], all_comb_true_phq[woz_mask], all_e2_phq[woz_mask])
    e2_avatar = evaluate_predictions(all_comb_true_cls[avatar_mask], all_e2_preds[avatar_mask], all_e2_probs[avatar_mask], all_comb_true_phq[avatar_mask], all_e2_phq[avatar_mask])
    results_summary['Combined_Candidate2A_WoZ_Subset'] = e2_woz
    results_summary['Combined_Candidate2A_Avatar_Subset'] = e2_avatar
    print(f"  WoZ Subset (N={len(woz['sids'])}): Balanced Acc={e2_woz['balanced_accuracy']:.2f}%, Macro-F1={e2_woz['macro_f1']:.2f}%")
    print(f"  Avatar Subset (N={len(avatar['sids'])}): Balanced Acc={e2_avatar['balanced_accuracy']:.2f}%, Macro-F1={e2_avatar['macro_f1']:.2f}%")
    print(f"  Avatar Class Recalls: C0={e2_avatar['per_class_recall'][0]:.2f}%, C1={e2_avatar['per_class_recall'][1]:.2f}%, C2={e2_avatar['per_class_recall'][2]:.2f}%")
    
    # -------------------------------------------------------------
    # EXPERIMENT 3: Combined-Trained Focal Loss Model (gamma=1.5)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[4/5] EXPERIMENT 3: Combined-Trained Focal Loss (Exploratory)")
    print("=" * 60)
    
    exp3_oof_records = []
    for seed in seeds:
        set_seed(seed)
        fold_list = comb_folds_dict[str(seed)]
        seed_probs = np.zeros((len(comb['sids']), 3))
        seed_phq = np.zeros(len(comb['sids']))
        
        for fold_info in fold_list:
            tr_sids = fold_info['train_sessions']
            val_sids = fold_info['val_sessions']
            tr_idx = [comb_sid_to_idx[s] for s in tr_sids]
            val_idx = [comb_sid_to_idx[s] for s in val_sids]
            
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
        exp3_oof_records.append({
            'seed': seed,
            'probs': seed_probs,
            'pred_cls': pred_cls,
            'exp_phq': seed_phq
        })
        
    all_e3_probs = np.concatenate([r['probs'] for r in exp3_oof_records], axis=0)
    all_e3_preds = np.concatenate([r['pred_cls'] for r in exp3_oof_records], axis=0)
    all_e3_phq = np.concatenate([r['exp_phq'] for r in exp3_oof_records], axis=0)
    
    e3_overall = evaluate_predictions(all_comb_true_cls, all_e3_preds, all_e3_probs, all_comb_true_phq, all_e3_phq)
    e3_woz = evaluate_predictions(all_comb_true_cls[woz_mask], all_e3_preds[woz_mask], all_e3_probs[woz_mask], all_comb_true_phq[woz_mask], all_e3_phq[woz_mask])
    e3_avatar = evaluate_predictions(all_comb_true_cls[avatar_mask], all_e3_preds[avatar_mask], all_e3_probs[avatar_mask], all_comb_true_phq[avatar_mask], all_e3_phq[avatar_mask])
    
    results_summary['Combined_FocalLoss_Overall'] = e3_overall
    results_summary['Combined_FocalLoss_WoZ_Subset'] = e3_woz
    results_summary['Combined_FocalLoss_Avatar_Subset'] = e3_avatar
    
    print(f"  Overall Focal (N={len(comb['sids'])}): Balanced Acc={e3_overall['balanced_accuracy']:.2f}%, Macro-F1={e3_overall['macro_f1']:.2f}%")
    print(f"  WoZ Subset Focal: Balanced Acc={e3_woz['balanced_accuracy']:.2f}%, Macro-F1={e3_woz['macro_f1']:.2f}%")
    print(f"  Avatar Subset Focal: Balanced Acc={e3_avatar['balanced_accuracy']:.2f}%, Macro-F1={e3_avatar['macro_f1']:.2f}%")
    print(f"  Avatar Class Recalls: C0={e3_avatar['per_class_recall'][0]:.2f}%, C1={e3_avatar['per_class_recall'][1]:.2f}%, C2={e3_avatar['per_class_recall'][2]:.2f}%")
    
    # -------------------------------------------------------------
    # EXPERIMENT 4: Demographic Subgroup Performance Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[5/5] Subgroup Performance & Statistical Comparisons")
    print("=" * 60)
    
    male_mask = (all_comb_genders == 'male')
    female_mask = (all_comb_genders == 'female')
    young_mask = (all_comb_ages < 40)
    older_mask = (all_comb_ages >= 40)
    
    subgroups = {
        'male': (male_mask, f'Male (N={np.sum(male_mask)//len(seeds)})'),
        'female': (female_mask, f'Female (N={np.sum(female_mask)//len(seeds)})'),
        'age_under_40': (young_mask, f'Age < 40 (N={np.sum(young_mask)//len(seeds)})'),
        'age_40_and_over': (older_mask, f'Age >= 40 (N={np.sum(older_mask)//len(seeds)})')
    }
    
    subgroup_results = {}
    for sg_name, (mask, desc) in subgroups.items():
        if np.sum(mask) > 0:
            m_res = evaluate_predictions(all_comb_true_cls[mask], all_e2_preds[mask], all_e2_probs[mask], all_comb_true_phq[mask], all_e2_phq[mask])
            subgroup_results[sg_name] = m_res
            print(f"  {desc:25s} | Balanced Acc: {m_res['balanced_accuracy']:.2f}% | Macro-F1: {m_res['macro_f1']:.2f}%")
            
    results_summary['Combined_Subgroups'] = subgroup_results
    
    # Paired Cluster Bootstrap: Combined vs Zero-Shot on Avatar
    print("\n--- Paired Cluster-Bootstrap: Combined-Trained vs Zero-Shot WoZ on Avatar ---")
    # Mask to avatar only
    avatar_pids_rep = all_comb_pids[avatar_mask]
    avatar_true_cls_rep = all_comb_true_cls[avatar_mask]
    
    # Zero shot preds on avatar
    av_e1_preds = all_av_pred_cls
    av_e2_preds = all_e2_preds[avatar_mask]
    
    av_e1_probs = all_av_probs
    av_e2_probs = all_e2_probs[avatar_mask]
    bs_res = paired_cluster_bootstrap_ci(
        avatar_pids_rep, avatar_true_cls_rep, av_e2_preds, av_e2_probs, av_e1_preds, av_e1_probs,
        n_bootstraps=2000, seed=42
    )
    
    bs_bal = bs_res['balanced_accuracy']
    bs_mf1 = bs_res['macro_f1']
    print(f"  Delta Balanced Acc on Avatar: {bs_bal['mean_diff']:.2f}% (95% CI: [{bs_bal['ci_lower']:.2f}%, {bs_bal['ci_upper']:.2f}%]), p={bs_bal['p_value']:.4f}")
    print(f"  Delta Macro-F1 on Avatar:     {bs_mf1['mean_diff']:.2f}% (95% CI: [{bs_mf1['ci_lower']:.2f}%, {bs_mf1['ci_upper']:.2f}%]), p={bs_mf1['p_value']:.4f}")
    results_summary['Bootstrap_Avatar_Combined_vs_ZeroShot'] = bs_res
    
    # Save Full Phase 3 Summary
    out_file = os.path.join(exp_dir, 'phase3_results_summary.json')
    with open(out_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved full Phase 3 summary to {out_file}")

if __name__ == '__main__':
    run_phase3_suite()
