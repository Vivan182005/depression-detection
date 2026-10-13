"""
Module: experiments/research_v2/metrics.py
Description: Rigorous statistical metrics evaluation harness.
Computes multi-class, binary, continuous, calibration, and paired cluster-bootstrap CIs.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Lin's Concordance Correlation Coefficient (CCC).
    """
    if len(y_true) < 2:
        return 0.0
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-12:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return float(2.0 * covar / denom)

def compute_expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width probability bins.
    probs: (N, 3) predicted class probabilities
    y_true: (N,) true integer class labels
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece * 100.0)

def compute_multiclass_brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Multi-class Brier score = 1/N * sum_i sum_c (P_ic - y_ic)^2
    """
    N = len(y_true)
    y_one_hot = np.zeros_like(probs)
    for i, c in enumerate(y_true):
        y_one_hot[i, c] = 1.0
    return float(np.sum((probs - y_one_hot) ** 2) / N)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    phq_true: np.ndarray = None,
    exp_phq: np.ndarray = None,
    raw_phq: np.ndarray = None
) -> Dict[str, Any]:
    """
    Computes comprehensive metric suite for depression severity detection.
    """
    N = len(y_true)
    counts = np.bincount(y_true, minlength=3)
    majority_baseline = float(np.max(counts) / N * 100.0)
    
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0)
    
    prec_per_class = [float(p * 100.0) for p in precision_score(y_true, y_pred, average=None, zero_division=0)]
    rec_per_class = [float(r * 100.0) for r in recall_score(y_true, y_pred, average=None, zero_division=0)]
    f1_per_class = [float(f * 100.0) for f in f1_score(y_true, y_pred, average=None, zero_division=0)]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    # Binary classification (Class 0: 0-9 vs Class 1+2: >= 10)
    y_bin_true = (y_true >= 1).astype(int)
    prob_depressed = probs[:, 1] + probs[:, 2]
    y_bin_pred = (prob_depressed >= 0.5).astype(int)
    
    bin_acc = float(accuracy_score(y_bin_true, y_bin_pred) * 100.0)
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_prec = float(precision_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_sens = float(recall_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    
    tn = np.sum((y_bin_true == 0) & (y_bin_pred == 0))
    fp = np.sum((y_bin_true == 0) & (y_bin_pred == 1))
    bin_spec = float(tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    
    try:
        bin_auc = float(roc_auc_score(y_bin_true, prob_depressed) * 100.0)
    except Exception:
        bin_auc = 50.0
    try:
        bin_prauc = float(average_precision_score(y_bin_true, prob_depressed) * 100.0)
    except Exception:
        bin_prauc = 0.0
        
    ece = compute_expected_calibration_error(probs, y_true, n_bins=10)
    brier = compute_multiclass_brier_score(probs, y_true)
    
    res = {
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'accuracy': acc,
        'majority_baseline': majority_baseline,
        'weighted_f1': weighted_f1,
        'per_class_precision': prec_per_class,
        'per_class_recall': rec_per_class,
        'per_class_f1': f1_per_class,
        'confusion_matrix': cm,
        'binary_accuracy': bin_acc,
        'binary_f1': bin_f1,
        'binary_precision': bin_prec,
        'binary_sensitivity': bin_sens,
        'binary_specificity': bin_spec,
        'roc_auc': bin_auc,
        'pr_auc': bin_prauc,
        'ece': ece,
        'brier_score': brier,
        'class_distribution': [int(c) for c in counts]
    }
    
    # Continuous metrics
    if phq_true is not None and exp_phq is not None:
        exp_mae = float(np.mean(np.abs(exp_phq - phq_true)))
        exp_rmse = float(np.sqrt(np.mean((exp_phq - phq_true) ** 2)))
        exp_ccc = float(concordance_correlation_coefficient(phq_true, exp_phq))
        res['exp_phq_mae'] = exp_mae
        res['exp_phq_rmse'] = exp_rmse
        res['exp_phq_ccc'] = exp_ccc
        
    if phq_true is not None and raw_phq is not None:
        raw_mae = float(np.mean(np.abs(raw_phq - phq_true)))
        raw_rmse = float(np.sqrt(np.mean((raw_phq - phq_true) ** 2)))
        raw_ccc = float(concordance_correlation_coefficient(phq_true, raw_phq))
        res['raw_phq_mae'] = raw_mae
        res['raw_phq_rmse'] = raw_rmse
        res['raw_phq_ccc'] = raw_ccc
        
    return res

def paired_cluster_bootstrap_ci(
    pids: np.ndarray,
    y_true: np.ndarray,
    cand_preds: np.ndarray,
    cand_probs: np.ndarray,
    base_preds: np.ndarray,
    base_probs: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Cluster bootstrap grouped by participant ID to respect session-level dependency.
    Computes 95% confidence intervals on the paired difference (Candidate - Baseline)
    for Macro-F1, Balanced Accuracy, Accuracy, and Binary F1.
    """
    rng = np.random.RandomState(seed)
    unique_pids = np.unique(pids)
    M = len(unique_pids)
    
    # Pre-index PID -> session indices
    pid_to_indices = {pid: np.where(pids == pid)[0] for pid in unique_pids}
    
    diff_macro_f1 = []
    diff_bal_acc = []
    diff_acc = []
    diff_bin_f1 = []
    
    for _ in range(n_bootstraps):
        sampled_pids = rng.choice(unique_pids, size=M, replace=True)
        sampled_indices = np.concatenate([pid_to_indices[p] for p in sampled_pids])
        
        y_b = y_true[sampled_indices]
        c_pred_b = cand_preds[sampled_indices]
        b_pred_b = base_preds[sampled_indices]
        
        # Macro F1
        c_mf1 = f1_score(y_b, c_pred_b, average='macro', zero_division=0) * 100.0
        b_mf1 = f1_score(y_b, b_pred_b, average='macro', zero_division=0) * 100.0
        diff_macro_f1.append(c_mf1 - b_mf1)
        
        # Balanced Acc
        c_bacc = balanced_accuracy_score(y_b, c_pred_b) * 100.0
        b_bacc = balanced_accuracy_score(y_b, b_pred_b) * 100.0
        diff_bal_acc.append(c_bacc - b_bacc)
        
        # Accuracy
        c_acc = accuracy_score(y_b, c_pred_b) * 100.0
        b_acc = accuracy_score(y_b, b_pred_b) * 100.0
        diff_acc.append(c_acc - b_acc)
        
        # Binary F1
        y_bin_b = (y_b >= 1).astype(int)
        c_bin_b = (cand_probs[sampled_indices, 1] + cand_probs[sampled_indices, 2] >= 0.5).astype(int)
        b_bin_b = (base_probs[sampled_indices, 1] + base_probs[sampled_indices, 2] >= 0.5).astype(int)
        diff_bin_f1.append((f1_score(y_bin_b, c_bin_b, zero_division=0) - f1_score(y_bin_b, b_bin_b, zero_division=0)) * 100.0)
        
    alpha_half = (1.0 - ci_level) / 2.0
    
    def get_summary(diffs):
        arr = np.array(diffs)
        mean_val = float(np.mean(arr))
        lower = float(np.percentile(arr, alpha_half * 100.0))
        upper = float(np.percentile(arr, (1.0 - alpha_half) * 100.0))
        # Two-tailed empirical p-value for diff == 0
        p_val = float(np.mean(arr <= 0) if mean_val > 0 else np.mean(arr >= 0)) * 2.0
        p_val = min(1.0, max(0.0, p_val))
        return {
            'mean_diff': mean_val,
            'ci_lower': lower,
            'ci_upper': upper,
            'p_value': p_val,
            'significant_at_05': bool(lower > 0.0 or upper < 0.0)
        }
        
    return {
        'macro_f1': get_summary(diff_macro_f1),
        'balanced_accuracy': get_summary(diff_bal_acc),
        'accuracy': get_summary(diff_acc),
        'binary_f1': get_summary(diff_bin_f1)
    }
