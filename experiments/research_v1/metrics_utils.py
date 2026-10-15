import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, roc_auc_score, precision_recall_curve, auc
)

def compute_all_metrics(y_true, y_pred, y_probs=None, phq_true=None, exp_phq=None, raw_phq=None):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0)
    
    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    # Binary PHQ-8 >= 10 screening
    y_bin_true = (y_true >= 1).astype(int)
    if y_probs is not None:
        y_bin_prob = 1.0 - y_probs[:, 0]
        y_bin_pred = (y_bin_prob >= 0.5).astype(int)
    else:
        y_bin_pred = (y_pred >= 1).astype(int)
        y_bin_prob = None
        
    bin_acc = float(accuracy_score(y_bin_true, y_bin_pred) * 100.0)
    bin_p = float(precision_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_r = float(recall_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)
    
    tn = int(np.sum((y_bin_true == 0) & (y_bin_pred == 0)))
    fp = int(np.sum((y_bin_true == 0) & (y_bin_pred == 1)))
    fn = int(np.sum((y_bin_true == 1) & (y_bin_pred == 0)))
    tp = int(np.sum((y_bin_true == 1) & (y_bin_pred == 1)))
    sensitivity = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    
    if y_bin_prob is not None and len(np.unique(y_bin_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_bin_true, y_bin_prob) * 100.0)
        except Exception:
            roc_auc = None
        try:
            p_c, r_c, _ = precision_recall_curve(y_bin_true, y_bin_prob)
            pr_auc = float(auc(r_c, p_c) * 100.0)
        except Exception:
            pr_auc = None
    else:
        roc_auc, pr_auc = None, None
        
    # Regression metrics
    res_reg = {}
    if phq_true is not None and exp_phq is not None:
        res_reg['exp_phq_mae'] = float(mean_absolute_error(phq_true, exp_phq))
        res_reg['exp_phq_rmse'] = float(np.sqrt(mean_squared_error(phq_true, exp_phq)))
    if phq_true is not None and raw_phq is not None:
        res_reg['raw_phq_mae'] = float(mean_absolute_error(phq_true, raw_phq))
        res_reg['raw_phq_rmse'] = float(np.sqrt(mean_squared_error(phq_true, raw_phq)))
        
    # Calibration metrics
    brier_score = None
    ece = None
    if y_probs is not None:
        n_classes = y_probs.shape[1]
        y_one_hot = np.eye(n_classes)[y_true]
        brier_score = float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))
        
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = (predictions == y_true)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        for b_idx in range(n_bins):
            b_lower, b_upper = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
            in_bin = (confidences > b_lower) & (confidences <= b_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                conf_in_bin = np.mean(confidences[in_bin])
                ece_val += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
        ece = float(ece_val * 100.0)
        
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': [float(x) for x in p_per],
        'per_class_recall': [float(x) for x in r_per],
        'per_class_f1': [float(x) for x in f_per],
        'confusion_matrix': cm,
        'binary_accuracy': bin_acc,
        'binary_precision': bin_p,
        'binary_recall': bin_r,
        'binary_f1': bin_f1,
        'binary_sensitivity': sensitivity,
        'binary_specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier_score,
        'ece': ece,
        **res_reg
    }
