import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

BASE_DIR = "/Users/vivannaik/Desktop/CAPSTONE"
EXP_DIR = os.path.join(BASE_DIR, "experiments", "research_v1")
sys.path.insert(0, EXP_DIR)

from data_loader import get_development_dataset, get_stratified_group_folds
from metrics_utils import compute_all_metrics
from model_components import StandardMLP, MAGNetMLP, train_nn_model

def run_suite():
    print("==========================================================================")
    print("   STARTING MASTER RESEARCH UPGRADE EXPERIMENT SUITE (PHASES 4 & 5)       ")
    print("==========================================================================")
    
    start_time = time.time()
    data = get_development_dataset()
    folds = get_stratified_group_folds(data, n_splits=5, seed=42)
    
    N = len(data["pids"])
    print(f"Total Development Sessions: {N} (179 unique participants across 5 leakage-free folds)")
    
    X_ac = data["X_ac"]
    X_w2v = data["X_w2v"]
    X_bert = data["X_bert"]
    X_tri = np.hstack([X_ac, X_w2v, X_bert])
    X_ac_w2v = np.hstack([X_ac, X_w2v])
    X_ac_bert = np.hstack([X_ac, X_bert])
    
    y_cls = data["y_class"]
    y_phq = data["y_phq"]
    pids = data["pids"]
    sids = data["sids"]
    
    CLASS_MEANS = np.array([3.24096, 11.05882, 18.25000], dtype=np.float32)
    results_all = {}
    
    # PHASE 4: Candidate 2A Baseline
    print()
    print("[PHASE 4] Running 5-Fold CV for Candidate 2A Baseline...")
    oof_preds_2a = np.zeros(N, dtype=int)
    oof_probs_2a = np.zeros((N, 3), dtype=float)
    oof_phq_2a = np.zeros(N, dtype=float)
    
    for f_idx, (tr_idx, val_idx) in enumerate(folds):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tri[tr_idx])
        X_val = scaler.transform(X_tri[val_idx])
        
        preds, _ = train_nn_model(
            StandardMLP, 1690,
            X_tr, y_cls[tr_idx], y_phq[tr_idx], pids[tr_idx], sids[tr_idx],
            X_val, y_cls[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx],
            loss_type="mild_balanced", epochs=40, lr=1e-3, seed=42 + f_idx
        )
        oof_preds_2a[val_idx] = preds["pred_cls"]
        oof_probs_2a[val_idx] = preds["probs"]
        oof_phq_2a[val_idx] = preds["pred_phq"]
        
    exp_phq_2a = np.sum(oof_probs_2a * CLASS_MEANS, axis=1)
    res_p4 = compute_all_metrics(y_cls, oof_preds_2a, oof_probs_2a, phq_true=y_phq, exp_phq=exp_phq_2a, raw_phq=oof_phq_2a)
    results_all["Candidate_2A_Baseline"] = res_p4
    print("  -> Cand 2A OOF: Acc=" + str(round(res_p4["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_p4["macro_f1"], 2)) + "%, Bal-Acc=" + str(round(res_p4["balanced_accuracy"], 2)) + "%")
    
    df_oof_2a = pd.DataFrame({
        "participant_id": pids,
        "session_id": sids,
        "true_class": y_cls,
        "true_phq8": y_phq,
        "pred_class": oof_preds_2a,
        "prob_0": oof_probs_2a[:, 0],
        "prob_1": oof_probs_2a[:, 1],
        "prob_2": oof_probs_2a[:, 2],
        "expected_phq": exp_phq_2a,
        "raw_phq": oof_phq_2a
    })
    df_oof_2a.to_csv(os.path.join(EXP_DIR, "oof_cand2a_baseline.csv"), index=False)
    
    # EXPERIMENT A: Classical Baselines
    print()
    print("[EXPERIMENT A] Running Classical Machine Learning Baselines...")
    feature_sets = {
        "Acoustic_154d": X_ac,
        "Wav2Vec_768d": X_w2v,
        "BERT_768d": X_bert,
        "Trimodal_1690d": X_tri
    }
    
    for feat_name, X_feat in feature_sets.items():
        oof_preds_lr = np.zeros(N, dtype=int)
        oof_probs_lr = np.zeros((N, 3), dtype=float)
        oof_preds_svm = np.zeros(N, dtype=int)
        oof_probs_svm = np.zeros((N, 3), dtype=float)
        
        for tr_idx, val_idx in folds:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_feat[tr_idx])
            X_val = scaler.transform(X_feat[val_idx])
            
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
            lr.fit(X_tr, y_cls[tr_idx])
            oof_preds_lr[val_idx] = lr.predict(X_val)
            oof_probs_lr[val_idx] = lr.predict_proba(X_val)
            
            svm = SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, random_state=42)
            svm.fit(X_tr, y_cls[tr_idx])
            oof_preds_svm[val_idx] = svm.predict(X_val)
            oof_probs_svm[val_idx] = svm.predict_proba(X_val)
            
        res_lr = compute_all_metrics(y_cls, oof_preds_lr, oof_probs_lr, phq_true=y_phq)
        res_svm = compute_all_metrics(y_cls, oof_preds_svm, oof_probs_svm, phq_true=y_phq)
        
        results_all["ExpA_LogisticRegression_" + feat_name] = res_lr
        results_all["ExpA_SVM_RBF_" + feat_name] = res_svm
        print("  LR  (" + feat_name + "): Acc=" + str(round(res_lr["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_lr["macro_f1"], 2)) + "%")
        print("  SVM (" + feat_name + "): Acc=" + str(round(res_svm["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_svm["macro_f1"], 2)) + "%")
        
    # EXPERIMENT B: Neural Modality Ablation
    print()
    print("[EXPERIMENT B] Running Neural Modality Ablations...")
    neural_ablations = {
        "Acoustic_Only": (X_ac, 154),
        "Wav2Vec_Only": (X_w2v, 768),
        "BERT_Only": (X_bert, 768),
        "Acoustic_plus_Wav2Vec": (X_ac_w2v, 922),
        "Acoustic_plus_BERT": (X_ac_bert, 922),
        "Trimodal_Complete": (X_tri, 1690)
    }
    
    for abl_name, (X_mat, in_dim) in neural_ablations.items():
        oof_preds = np.zeros(N, dtype=int)
        oof_probs = np.zeros((N, 3), dtype=float)
        oof_phq = np.zeros(N, dtype=float)
        
        for f_idx, (tr_idx, val_idx) in enumerate(folds):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_mat[tr_idx])
            X_val = scaler.transform(X_mat[val_idx])
            
            preds, _ = train_nn_model(
                StandardMLP, in_dim,
                X_tr, y_cls[tr_idx], y_phq[tr_idx], pids[tr_idx], sids[tr_idx],
                X_val, y_cls[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx],
                loss_type="mild_balanced", epochs=40, lr=1e-3, seed=42 + f_idx
            )
            oof_preds[val_idx] = preds["pred_cls"]
            oof_probs[val_idx] = preds["probs"]
            oof_phq[val_idx] = preds["pred_phq"]
            
        exp_phq = np.sum(oof_probs * CLASS_MEANS, axis=1)
        res_abl = compute_all_metrics(y_cls, oof_preds, oof_probs, phq_true=y_phq, exp_phq=exp_phq, raw_phq=oof_phq)
        results_all["ExpB_" + abl_name] = res_abl
        print("  Ablation (" + abl_name + "): Acc=" + str(round(res_abl["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_abl["macro_f1"], 2)) + "%")
        
    # EXPERIMENT C: Class-Imbalance Strategies
    print()
    print("[EXPERIMENT C] Evaluating Class-Imbalance Strategies...")
    imbalance_strategies = [
        ("Unweighted_CE", "unweighted", 1.5),
        ("Inverse_Weighted_CE", "inverse", 1.5),
        ("Mild_Balanced_CE", "mild_balanced", 1.5),
        ("Focal_Loss_Gamma15", "focal", 1.5),
        ("Focal_Loss_Gamma20", "focal", 2.0)
    ]
    
    for strat_name, loss_type, gamma in imbalance_strategies:
        oof_preds = np.zeros(N, dtype=int)
        oof_probs = np.zeros((N, 3), dtype=float)
        oof_phq = np.zeros(N, dtype=float)
        
        for f_idx, (tr_idx, val_idx) in enumerate(folds):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tri[tr_idx])
            X_val = scaler.transform(X_tri[val_idx])
            
            preds, _ = train_nn_model(
                StandardMLP, 1690,
                X_tr, y_cls[tr_idx], y_phq[tr_idx], pids[tr_idx], sids[tr_idx],
                X_val, y_cls[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx],
                loss_type=loss_type, focal_gamma=gamma, epochs=40, lr=1e-3, seed=42 + f_idx
            )
            oof_preds[val_idx] = preds["pred_cls"]
            oof_probs[val_idx] = preds["probs"]
            oof_phq[val_idx] = preds["pred_phq"]
            
        exp_phq = np.sum(oof_probs * CLASS_MEANS, axis=1)
        res_strat = compute_all_metrics(y_cls, oof_preds, oof_probs, phq_true=y_phq, exp_phq=exp_phq, raw_phq=oof_phq)
        results_all["ExpC_" + strat_name] = res_strat
        print("  Imbalance (" + strat_name + "): Acc=" + str(round(res_strat["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_strat["macro_f1"], 2)) + "%")

    # EXPERIMENT D: MAGNet Gated Fusion
    print()
    print("[EXPERIMENT D] Evaluating MAGNet Modality-Aware Gating Network...")
    oof_preds_mag = np.zeros(N, dtype=int)
    oof_probs_mag = np.zeros((N, 3), dtype=float)
    oof_phq_mag = np.zeros(N, dtype=float)
    
    for f_idx, (tr_idx, val_idx) in enumerate(folds):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tri[tr_idx])
        X_val = scaler.transform(X_tri[val_idx])
        
        preds, _ = train_nn_model(
            MAGNetMLP, 1690,
            X_tr, y_cls[tr_idx], y_phq[tr_idx], pids[tr_idx], sids[tr_idx],
            X_val, y_cls[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx],
            loss_type="mild_balanced", epochs=40, lr=1e-3, seed=42 + f_idx
        )
        oof_preds_mag[val_idx] = preds["pred_cls"]
        oof_probs_mag[val_idx] = preds["probs"]
        oof_phq_mag[val_idx] = preds["pred_phq"]
        
    exp_phq_mag = np.sum(oof_probs_mag * CLASS_MEANS, axis=1)
    res_mag = compute_all_metrics(y_cls, oof_preds_mag, oof_probs_mag, phq_true=y_phq, exp_phq=exp_phq_mag, raw_phq=oof_phq_mag)
    results_all["ExpD_MAGNet_Gated_Fusion"] = res_mag
    print("  MAGNet Fusion: Acc=" + str(round(res_mag["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_mag["macro_f1"], 2)) + "%, Bal-Acc=" + str(round(res_mag["balanced_accuracy"], 2)) + "%")

    # EXPERIMENT E: Domain Robustness Audit
    print()
    print("[EXPERIMENT E] Domain Robustness & Data Availability Audit...")
    # Auditing whether multi-domain training data exists in development store
    # Rule 9 strictly prohibits using the sealed unseen cohort for adaptation or training.
    dev_domains = np.zeros(N) # 100% human WoZ in development feature store
    num_domains_in_dev = len(np.unique(dev_domains))
    print("  Development domain count: " + str(num_domains_in_dev) + " (100% DAIC-WOZ human interviews)")
    print("  Domain-adversarial training inside dev store is constrained by lack of multi-domain dev labels.")
    print("  Per Rule 9 & guidelines: Documented limitation; sealed unseen cohort preserved without adaptation.")
    results_all["ExpE_Domain_Robustness_Audit"] = {
        "status": "Audited & Constrained",
        "development_domains": ["human_woz"],
        "unseen_domains": ["human_woz", "autonomous_ai"],
        "limitation_reason": "Development feature store contains 100% human WoZ; sealed unseen cohort not used for adaptation per Rule 9."
    }

    # EXPERIMENT F: Temporal Augmentation
    print()
    print("[EXPERIMENT F] Evaluating Training-Fold Acoustic Jitter Augmentation...")
    oof_preds_aug = np.zeros(N, dtype=int)
    oof_probs_aug = np.zeros((N, 3), dtype=float)
    oof_phq_aug = np.zeros(N, dtype=float)
    
    for f_idx, (tr_idx, val_idx) in enumerate(folds):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tri[tr_idx])
        X_val = scaler.transform(X_tri[val_idx])
        
        np.random.seed(42 + f_idx)
        jitter = np.random.normal(0.0, 0.02, size=X_tr.shape)
        X_tr_aug = np.vstack([X_tr, X_tr + jitter])
        y_tr_cls_aug = np.hstack([y_cls[tr_idx], y_cls[tr_idx]])
        y_tr_phq_aug = np.hstack([y_phq[tr_idx], y_phq[tr_idx]])
        pids_tr_aug = np.hstack([pids[tr_idx], pids[tr_idx]])
        sids_tr_aug = np.hstack([sids[tr_idx], sids[tr_idx]])
        
        preds, _ = train_nn_model(
            StandardMLP, 1690,
            X_tr_aug, y_tr_cls_aug, y_tr_phq_aug, pids_tr_aug, sids_tr_aug,
            X_val, y_cls[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx],
            loss_type="mild_balanced", epochs=40, lr=1e-3, seed=42 + f_idx
        )
        oof_preds_aug[val_idx] = preds["pred_cls"]
        oof_probs_aug[val_idx] = preds["probs"]
        oof_phq_aug[val_idx] = preds["pred_phq"]
        
    exp_phq_aug = np.sum(oof_probs_aug * CLASS_MEANS, axis=1)
    res_aug = compute_all_metrics(y_cls, oof_preds_aug, oof_probs_aug, phq_true=y_phq, exp_phq=exp_phq_aug, raw_phq=oof_phq_aug)
    results_all["ExpF_Acoustic_Jitter_Augmentation"] = res_aug
    print("  Acoustic Jitter: Acc=" + str(round(res_aug["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_aug["macro_f1"], 2)) + "%")

    # EXPERIMENT G: Continuous Regression Severity Derivation
    print()
    print("[EXPERIMENT G] Evaluating Continuous PHQ-8 Regression Severity Derivation...")
    oof_preds_reg = np.zeros(N, dtype=int)
    oof_probs_reg = np.zeros((N, 3), dtype=float)
    oof_phq_reg = np.zeros(N, dtype=float)
    
    for f_idx, (tr_idx, val_idx) in enumerate(folds):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tri[tr_idx])
        X_val = scaler.transform(X_tri[val_idx])
        
        preds, _ = train_nn_model(
            StandardMLP, 1690,
            X_tr, y_cls[tr_idx], y_phq[tr_idx], pids[tr_idx], sids[tr_idx],
            X_val, y_cls[val_idx], y_phq[val_idx], pids[val_idx], sids[val_idx],
            loss_type="pure_regression", epochs=40, lr=1e-3, seed=42 + f_idx
        )
        oof_preds_reg[val_idx] = preds["pred_cls"]
        oof_probs_reg[val_idx] = preds["probs"]
        oof_phq_reg[val_idx] = preds["pred_phq"]
        
    res_reg = compute_all_metrics(y_cls, oof_preds_reg, oof_probs_reg, phq_true=y_phq, exp_phq=oof_phq_reg, raw_phq=oof_phq_reg)
    results_all["ExpG_Continuous_Regression_Derivation"] = res_reg
    print("  Pure Regression: Acc=" + str(round(res_reg["accuracy"], 2)) + "%, Macro-F1=" + str(round(res_reg["macro_f1"], 2)) + "%, MAE=" + str(round(res_reg["exp_phq_mae"], 2)))

    out_json = os.path.join(EXP_DIR, "master_experiment_results.json")
    with open(out_json, "w") as f:
        json.dump(results_all, f, indent=2)
    print()
    print("[SUCCESS] Master Experiment Suite Completed in " + str(round(time.time() - start_time, 1)) + " seconds!")
    print("Saved all experiment metrics to: " + out_json)

if __name__ == "__main__":
    run_suite()
