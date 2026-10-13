"""
Script: experiments/research_v2/run_phase1_harness.py
Description: Master Phase 1 Evaluation Harness.
Runs 17 model variants across 5 seeds x 5 folds (StratifiedGroupKFold).
Computes all primary, secondary, and calibration metrics.
Computes paired cluster-bootstrap 95% CIs against Candidate 2A baseline.
Saves all OOF artifacts and experiments_log.csv.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.dataset import load_development_dataset
from experiments.research_v2.metrics import evaluate_predictions, paired_cluster_bootstrap_ci
from experiments.research_v2.models import StandardMLP, MAGNetMLP, run_train_and_val_fold

def run_phase1_suite():
    print("=" * 80)
    print("       PHASE 1: RIGOROUS EVALUATION HARNESS & REPLICATION SUITE         ")
    print("=" * 80)
    
    start_total = time.time()
    exp_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2')
    oof_dir = os.path.join(exp_dir, 'oof')
    os.makedirs(oof_dir, exist_ok=True)
    
    # 1. Load Data
    print("[1/4] Loading development dataset...")
    data = load_development_dataset()
    N = len(data['sids'])
    pids = data['pids']
    sids = data['sids']
    y_cls = data['y_class']
    y_phq = data['y_phq']
    
    X_ac = data['X_ac']
    X_w2v = data['X_w2v']
    X_bert = data['X_bert']
    X_tri = data['X_tri']
    X_meta = data['X_meta']
    
    print(f"  Loaded {N} sessions across {len(np.unique(pids))} participants.")
    print(f"  Class distribution: {np.bincount(y_cls).tolist()} (Majority: {np.max(np.bincount(y_cls))/N*100:.2f}%)")
    
    # 2. Load Selection Folds (seeds 0, 1, 2, 3, 4)
    folds_path = os.path.join(exp_dir, 'folds.json')
    with open(folds_path, 'r') as f:
        folds_dict = json.load(f)
    selection_seeds = sorted([int(s) for s in folds_dict.keys()])
    print(f"  Loaded selection folds for seeds: {selection_seeds} (5 folds per seed, 25 total folds per model)")
    
    # Define experiment variants
    variants = [
        # Baseline Candidate 2A
        {'name': 'Candidate_2A_Baseline', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'mild_balanced', 'jitter': False},
        
        # Sanity baselines
        {'name': 'Sanity_Majority_Class', 'type': 'majority'},
        {'name': 'Sanity_Shuffled_Labels', 'type': 'shuffled'},
        {'name': 'Sanity_Metadata_Only', 'type': 'classical', 'model': 'lr', 'feat': 'X_meta'},
        
        # Classical machine learning
        {'name': 'Classical_LR_Trimodal', 'type': 'classical', 'model': 'lr', 'feat': 'X_tri'},
        {'name': 'Classical_SVM_Trimodal', 'type': 'classical', 'model': 'svm', 'feat': 'X_tri'},
        {'name': 'Classical_SVM_BERT', 'type': 'classical', 'model': 'svm', 'feat': 'X_bert'},
        
        # Modality ablations
        {'name': 'Modality_Acoustic_Only', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 154, 'feat': 'X_ac', 'loss': 'mild_balanced', 'jitter': False},
        {'name': 'Modality_Wav2Vec_Only', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 768, 'feat': 'X_w2v', 'loss': 'mild_balanced', 'jitter': False},
        {'name': 'Modality_BERT_Only', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 768, 'feat': 'X_bert', 'loss': 'mild_balanced', 'jitter': False},
        
        # Loss & Imbalance variants
        {'name': 'Loss_Unweighted_CE', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'unweighted', 'jitter': False},
        {'name': 'Loss_Inverse_Weighted_CE', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'inverse', 'jitter': False},
        {'name': 'Loss_Focal_Gamma15', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'focal', 'focal_gamma': 1.5, 'jitter': False},
        {'name': 'Loss_Focal_Gamma20', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'focal', 'focal_gamma': 2.0, 'jitter': False},
        
        # Architecture: Gated fusion
        {'name': 'Architecture_MAGNet', 'type': 'nn', 'arch': MAGNetMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'mild_balanced', 'jitter': False},
        
        # Data Augmentation: Acoustic Jitter
        {'name': 'Augmentation_Jitter', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'mild_balanced', 'jitter': True},
        
        # Continuous Regression derivation
        {'name': 'Regression_Derived', 'type': 'nn', 'arch': StandardMLP, 'in_dim': 1690, 'feat': 'X_tri', 'loss': 'pure_regression', 'jitter': False}
    ]
    
    print(f"[2/4] Executing {len(variants)} experimental variants...")
    
    results_by_variant = {}
    oof_predictions_by_variant = {}
    
    for v_idx, var in enumerate(variants):
        var_name = var['name']
        v_type = var['type']
        t0 = time.time()
        print(f"--> [{v_idx+1}/{len(variants)}] Running variant: {var_name} (type: {v_type})")
        
        seed_metrics_list = []
        seed_oof_records = []
        
        # Storage for seed-level OOF matrices to pool across seeds
        all_seed_probs = np.zeros((len(selection_seeds), N, 3), dtype=np.float32)
        all_seed_preds = np.zeros((len(selection_seeds), N), dtype=int)
        all_seed_exp_phq = np.zeros((len(selection_seeds), N), dtype=np.float32)
        all_seed_raw_phq = np.zeros((len(selection_seeds), N), dtype=np.float32)
        all_seed_logits = np.zeros((len(selection_seeds), N, 3), dtype=np.float32)
        
        for s_i, seed in enumerate(selection_seeds):
            folds = folds_dict[str(seed)]
            
            oof_logits = np.zeros((N, 3), dtype=np.float32)
            oof_probs = np.zeros((N, 3), dtype=np.float32)
            oof_preds = np.zeros(N, dtype=int)
            oof_exp_phq = np.zeros(N, dtype=np.float32)
            oof_raw_phq = np.zeros(N, dtype=np.float32)
            oof_calib_T = np.zeros(N, dtype=np.float32)
            
            for f_info in folds:
                f_idx = f_info['fold']
                tr_idx = np.array(f_info['train_indices'])
                val_idx = np.array(f_info['val_indices'])
                
                # Training fold conditional means
                mu_fold = np.zeros(3, dtype=np.float32)
                for c in range(3):
                    m = (y_cls[tr_idx] == c)
                    mu_fold[c] = np.mean(y_phq[tr_idx][m]) if np.sum(m) > 0 else np.mean(y_phq[tr_idx])
                    
                if v_type == 'majority':
                    counts = np.bincount(y_cls[tr_idx], minlength=3)
                    maj_c = int(np.argmax(counts))
                    oof_preds[val_idx] = maj_c
                    oof_probs[val_idx, maj_c] = 1.0
                    oof_exp_phq[val_idx] = mu_fold[maj_c]
                    oof_calib_T[val_idx] = 1.0
                    
                elif v_type == 'shuffled':
                    rng = np.random.RandomState(seed * 10 + f_idx)
                    shuff_y = rng.permutation(y_cls[tr_idx])
                    val_preds = rng.choice(shuff_y, size=len(val_idx))
                    oof_preds[val_idx] = val_preds
                    for vi, c in zip(val_idx, val_preds):
                        oof_probs[vi, c] = 1.0
                    oof_exp_phq[val_idx] = np.array([mu_fold[c] for c in val_preds])
                    oof_calib_T[val_idx] = 1.0
                    
                elif v_type == 'classical':
                    X_raw = data[var['feat']]
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_raw[tr_idx])
                    X_val_s = scaler.transform(X_raw[val_idx])
                    
                    if var['model'] == 'lr':
                        clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=seed + f_idx)
                    elif var['model'] == 'svm':
                        clf = SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, random_state=seed + f_idx)
                    
                    clf.fit(X_tr_s, y_cls[tr_idx])
                    val_p = clf.predict_proba(X_val_s)
                    val_c = np.argmax(val_p, axis=1)
                    
                    oof_probs[val_idx] = val_p
                    oof_preds[val_idx] = val_c
                    oof_exp_phq[val_idx] = np.sum(val_p * mu_fold, axis=1)
                    oof_calib_T[val_idx] = 1.0
                    
                elif v_type == 'nn':
                    X_raw = data[var['feat']]
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_raw[tr_idx])
                    X_val_s = scaler.transform(X_raw[val_idx])
                    
                    fold_res = run_train_and_val_fold(
                        model_cls=var['arch'],
                        in_dim=var['in_dim'],
                        X_tr=X_tr_s,
                        y_tr_cls=y_cls[tr_idx],
                        y_tr_phq=y_phq[tr_idx],
                        pids_tr=pids[tr_idx],
                        X_val=X_val_s,
                        y_val_cls=y_cls[val_idx],
                        y_val_phq=y_phq[val_idx],
                        pids_val=pids[val_idx],
                        loss_type=var['loss'],
                        focal_gamma=var.get('focal_gamma', 1.5),
                        epochs=40,
                        use_jitter=var['jitter'],
                        seed=seed * 100 + f_idx
                    )
                    
                    oof_logits[val_idx] = fold_res['logits']
                    oof_probs[val_idx] = fold_res['probs']
                    oof_preds[val_idx] = fold_res['pred_cls']
                    oof_exp_phq[val_idx] = fold_res['expected_phq']
                    oof_raw_phq[val_idx] = fold_res['raw_phq']
                    oof_calib_T[val_idx] = fold_res['calibrated_T']
                    
            # Compute seed-level metrics
            seed_m = evaluate_predictions(y_cls, oof_preds, oof_probs, phq_true=y_phq, exp_phq=oof_exp_phq, raw_phq=oof_raw_phq if v_type == 'nn' else None)
            seed_metrics_list.append(seed_m)
            
            # Save seed matrices
            all_seed_probs[s_i] = oof_probs
            all_seed_preds[s_i] = oof_preds
            all_seed_exp_phq[s_i] = oof_exp_phq
            all_seed_raw_phq[s_i] = oof_raw_phq
            all_seed_logits[s_i] = oof_logits
            
            # Save per-seed OOF CSV
            df_seed_oof = pd.DataFrame({
                'participant_id': pids,
                'session_id': sids,
                'true_class': y_cls,
                'true_phq8': y_phq,
                'pred_class': oof_preds,
                'logit_0': oof_logits[:, 0],
                'logit_1': oof_logits[:, 1],
                'logit_2': oof_logits[:, 2],
                'prob_0': oof_probs[:, 0],
                'prob_1': oof_probs[:, 1],
                'prob_2': oof_probs[:, 2],
                'expected_phq': oof_exp_phq,
                'raw_phq': oof_raw_phq,
                'calibrated_T': oof_calib_T
            })
            df_seed_oof.to_csv(os.path.join(oof_dir, f'{var_name}_seed{seed}_oof.csv'), index=False)
            
        # Pooled ensemble prediction across 5 seeds
        pooled_probs = np.mean(all_seed_probs, axis=0)
        pooled_preds = np.argmax(pooled_probs, axis=1)
        pooled_exp_phq = np.mean(all_seed_exp_phq, axis=0)
        pooled_raw_phq = np.mean(all_seed_raw_phq, axis=0)
        pooled_logits = np.mean(all_seed_logits, axis=0)
        
        pooled_m = evaluate_predictions(y_cls, pooled_preds, pooled_probs, phq_true=y_phq, exp_phq=pooled_exp_phq, raw_phq=pooled_raw_phq if v_type == 'nn' else None)
        
        df_pooled_oof = pd.DataFrame({
            'participant_id': pids,
            'session_id': sids,
            'true_class': y_cls,
            'true_phq8': y_phq,
            'pred_class': pooled_preds,
            'logit_0': pooled_logits[:, 0],
            'logit_1': pooled_logits[:, 1],
            'logit_2': pooled_logits[:, 2],
            'prob_0': pooled_probs[:, 0],
            'prob_1': pooled_probs[:, 1],
            'prob_2': pooled_probs[:, 2],
            'expected_phq': pooled_exp_phq,
            'raw_phq': pooled_raw_phq
        })
        df_pooled_oof.to_csv(os.path.join(oof_dir, f'{var_name}_pooled_oof.csv'), index=False)
        
        # Calculate mean +- std over 5 seeds
        def mean_std(key):
            vals = [sm[key] for sm in seed_metrics_list if key in sm]
            return float(np.mean(vals)), float(np.std(vals))
            
        mf1_mean, mf1_std = mean_std('macro_f1')
        bacc_mean, bacc_std = mean_std('balanced_accuracy')
        acc_mean, acc_std = mean_std('accuracy')
        mae_mean, mae_std = mean_std('exp_phq_mae')
        ccc_mean, ccc_std = mean_std('exp_phq_ccc')
        bin_f1_mean, bin_f1_std = mean_std('binary_f1')
        ece_mean, ece_std = mean_std('ece')
        
        elapsed = time.time() - t0
        print(f"    Macro-F1: {mf1_mean:.2f} +- {mf1_std:.2f}% | Bal-Acc: {bacc_mean:.2f} +- {bacc_std:.2f}% | Acc: {acc_mean:.2f}% (Maj: 71.04%) | Exp MAE: {mae_mean:.2f} | Time: {elapsed:.1f}s")
        
        results_by_variant[var_name] = {
            'type': v_type,
            'macro_f1_mean': mf1_mean,
            'macro_f1_std': mf1_std,
            'balanced_accuracy_mean': bacc_mean,
            'balanced_accuracy_std': bacc_std,
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'majority_baseline': 71.04,
            'exp_phq_mae_mean': mae_mean,
            'exp_phq_mae_std': mae_std,
            'exp_phq_ccc_mean': ccc_mean,
            'exp_phq_ccc_std': ccc_std,
            'binary_f1_mean': bin_f1_mean,
            'binary_f1_std': bin_f1_std,
            'ece_mean': ece_mean,
            'ece_std': ece_std,
            'pooled_metrics': pooled_m,
            'seed_metrics': seed_metrics_list,
            'elapsed_sec': elapsed
        }
        
        oof_predictions_by_variant[var_name] = {
            'pooled_preds': pooled_preds,
            'pooled_probs': pooled_probs
        }
        
    # 3. Paired Cluster-Bootstrap Confidence Intervals vs Candidate 2A
    print("[3/4] Computing paired cluster-bootstrap 95% CIs vs Candidate 2A Baseline...")
    base_preds = oof_predictions_by_variant['Candidate_2A_Baseline']['pooled_preds']
    base_probs = oof_predictions_by_variant['Candidate_2A_Baseline']['pooled_probs']
    
    ci_comparisons = {}
    for var_name, v_data in oof_predictions_by_variant.items():
        if var_name == 'Candidate_2A_Baseline':
            continue
        cand_preds = v_data['pooled_preds']
        cand_probs = v_data['pooled_probs']
        
        ci_res = paired_cluster_bootstrap_ci(
            pids=pids,
            y_true=y_cls,
            cand_preds=cand_preds,
            cand_probs=cand_probs,
            base_preds=base_preds,
            base_probs=base_probs,
            n_bootstraps=1000,
            seed=42,
            ci_level=0.95
        )
        ci_comparisons[var_name] = ci_res
        results_by_variant[var_name]['paired_ci_vs_candidate2a'] = ci_res
        
        mf1_diff = ci_res['macro_f1']['mean_diff']
        mf1_low = ci_res['macro_f1']['ci_lower']
        mf1_up = ci_res['macro_f1']['ci_upper']
        sig_str = "*SIGNIFICANT*" if ci_res['macro_f1']['significant_at_05'] else "(crosses zero)"
        print(f"  {var_name:30s} | Delta Macro-F1: {mf1_diff:+.2f}% [95% CI: {mf1_low:+.2f}%, {mf1_up:+.2f}%] {sig_str}")
        
    # 4. Save experiments_log.csv and summary JSON
    print("[4/4] Writing experiments_log.csv and phase1_results_summary.json...")
    log_rows = []
    for var_name, res in results_by_variant.items():
        row = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'variant_name': var_name,
            'type': res['type'],
            'macro_f1_mean': round(res['macro_f1_mean'], 2),
            'macro_f1_std': round(res['macro_f1_std'], 2),
            'balanced_accuracy_mean': round(res['balanced_accuracy_mean'], 2),
            'balanced_accuracy_std': round(res['balanced_accuracy_std'], 2),
            'accuracy_mean': round(res['accuracy_mean'], 2),
            'majority_baseline': 71.04,
            'binary_f1_mean': round(res['binary_f1_mean'], 2),
            'binary_f1_std': round(res['binary_f1_std'], 2),
            'exp_phq_mae_mean': round(res['exp_phq_mae_mean'], 2),
            'exp_phq_mae_std': round(res['exp_phq_mae_std'], 2),
            'exp_phq_ccc_mean': round(res['exp_phq_ccc_mean'], 3),
            'ece_mean': round(res['ece_mean'], 2),
            'elapsed_sec': round(res['elapsed_sec'], 1)
        }
        if 'paired_ci_vs_candidate2a' in res:
            ci_m = res['paired_ci_vs_candidate2a']['macro_f1']
            ci_b = res['paired_ci_vs_candidate2a']['balanced_accuracy']
            row['delta_macro_f1_mean'] = round(ci_m['mean_diff'], 2)
            row['delta_macro_f1_ci95'] = f"[{ci_m['ci_lower']:+.2f}, {ci_m['ci_upper']:+.2f}]"
            row['delta_macro_f1_p_val'] = round(ci_m['p_value'], 4)
            row['delta_bacc_mean'] = round(ci_b['mean_diff'], 2)
            row['delta_bacc_ci95'] = f"[{ci_b['ci_lower']:+.2f}, {ci_b['ci_upper']:+.2f}]"
        else:
            row['delta_macro_f1_mean'] = 0.0
            row['delta_macro_f1_ci95'] = 'REFERENCE'
            row['delta_macro_f1_p_val'] = 1.0
            row['delta_bacc_mean'] = 0.0
            row['delta_bacc_ci95'] = 'REFERENCE'
            
        log_rows.append(row)
        
    df_log = pd.DataFrame(log_rows)
    log_csv_path = os.path.join(exp_dir, 'experiments_log.csv')
    df_log.to_csv(log_csv_path, index=False)
    print(f"  Logged {len(df_log)} variants to {log_csv_path}")
    
    summary_json_path = os.path.join(exp_dir, 'phase1_results_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(results_by_variant, f, indent=2)
    print(f"  Saved summary JSON to {summary_json_path}")
    
    total_elapsed = time.time() - start_total
    print(f"Phase 1 Suite Complete in {total_elapsed/60.0:.2f} minutes.")
    return results_by_variant

if __name__ == '__main__':
    run_phase1_suite()
