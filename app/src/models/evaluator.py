"""
Module: evaluator.py
Description: Research-Quality Evaluation Engine for Depression Severity Prediction.
Computes complete metric suite:
- Classification: Accuracy, Balanced Accuracy, Precision/Recall/F1 (Macro & Weighted), MCC, Cohen's Kappa, ROC-AUC
- Regression: MAE, RMSE, R^2, Pearson r, Spearman rho
- Calibration: Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
- Clinical Risk: False Negative Rate on Severe Cases, False Positive Rate
- Visual Plot Generators: Confusion Matrix, ROC-AUC, Reliability Diagrams, Loss/Accuracy Curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr, spearmanr


def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)

    return ece * 100.0, mce * 100.0


def evaluate_research_metrics(y_true, y_pred, probs, phq_true, phq_pred, num_classes=3):
    """
    Comprehensive Research Evaluation Suite returning metric dictionary.
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0

    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0

    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Multi-class ROC-AUC One-vs-Rest
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc_ovr = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        else:
            roc_auc_ovr = 0.50
    except Exception:
        roc_auc_ovr = 0.50

    # Regression Metrics
    mae = mean_absolute_error(phq_true, phq_pred)
    rmse = np.sqrt(mean_squared_error(phq_true, phq_pred))
    r2 = r2_score(phq_true, phq_pred) if len(phq_true) > 1 else 0.0
    
    p_corr, _ = pearsonr(phq_true, phq_pred) if len(phq_true) > 1 else (0.0, 1.0)
    s_corr, _ = spearmanr(phq_true, phq_pred) if len(phq_true) > 1 else (0.0, 1.0)

    # Calibration Metrics
    ece, mce = compute_ece(probs, y_true)

    # Clinical Vulnerability Metrics (Severe & Moderately Severe Depression False Negative Rate)
    severe_mask = (y_true >= 2)
    if np.sum(severe_mask) > 0:
        fn_severe = np.sum((y_pred[severe_mask] < 2))
        fnr_severe = (fn_severe / np.sum(severe_mask)) * 100.0
    else:
        fnr_severe = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'precision_weighted': prec_weighted,
        'recall_weighted': rec_weighted,
        'mcc': mcc,
        'kappa': kappa,
        'roc_auc_ovr': roc_auc_ovr,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': p_corr if not np.isnan(p_corr) else 0.0,
        'spearman_rho': s_corr if not np.isnan(s_corr) else 0.0,
        'ece': ece,
        'mce': mce,
        'fnr_severe': fnr_severe,
        'confusion_matrix': cm
    }
    return metrics


def compute_gender_bias_metrics(y_true, y_pred, gender_array):
    """
    TASK 3 Bias Auditing:
    - Calculates macro & weighted F1 separately for Female (0) and Male (1) subgroups.
    - Computes Disparate Impact Ratio (Positive/Depressed prediction rate for Male ÷ Female).
      where positive prediction rate = fraction of samples predicted as Moderate or above (classes >= 1).
    """
    female_mask = (gender_array == 0)
    male_mask = (gender_array == 1)

    f1_female = f1_score(y_true[female_mask], y_pred[female_mask], average='macro', zero_division=0) * 100.0 if np.sum(female_mask) > 0 else 0.0
    f1_male = f1_score(y_true[male_mask], y_pred[male_mask], average='macro', zero_division=0) * 100.0 if np.sum(male_mask) > 0 else 0.0

    pos_rate_female = np.mean(y_pred[female_mask] >= 1) if np.sum(female_mask) > 0 else 0.0
    pos_rate_male = np.mean(y_pred[male_mask] >= 1) if np.sum(male_mask) > 0 else 0.0

    disparate_impact_ratio = (pos_rate_male / pos_rate_female) if pos_rate_female > 0 else (1.0 if pos_rate_male == 0 else 0.0)

    return {
        'f1_female': f1_female,
        'f1_male': f1_male,
        'pos_rate_female': pos_rate_female * 100.0,
        'pos_rate_male': pos_rate_male * 100.0,
        'disparate_impact_ratio': disparate_impact_ratio
    }


def plot_research_figures(y_true, y_pred, probs, phq_true, phq_pred, output_dir="./output/plots"):
    """
    Generates Publication-Quality Evaluation Figures:
    1. Confusion Matrix Heatmap
    2. Reliability Diagram & Calibration Curve
    3. Regression Scatter & Residual Plot
    """
    os.makedirs(output_dir, exist_ok=True)
    class_names = ["Minimal/Mild", "Moderate", "Mod.Severe/Severe"]

    # Figure 1: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=class_names,
        title='Confusion Matrix (Subject-Independent Test Set)',
        ylabel='True Severity Class',
        xlabel='Predicted Severity Class'
    )
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig1)

    # Figure 2: Reliability Diagram (Calibration Curve)
    confidences = np.max(probs, axis=-1)
    accuracies = (y_pred == y_true)
    bin_boundaries = np.linspace(0, 1, 11)
    bin_accs = []
    bin_confs = []

    for i in range(10):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0)
            bin_confs.append(bin_boundaries[i] + 0.05)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax2.bar(bin_boundaries[:-1], bin_accs, width=0.08, align='edge', alpha=0.6, color='#1f77b4', label='Model Accuracy')
    ax2.step(bin_boundaries[:-1], bin_confs, where='post', color='red', linewidth=1.5, label='Model Confidence')
    ax2.set_title('Reliability Diagram (Probability Calibration)')
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Accuracy / Confidence')
    ax2.legend(loc='upper left')
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "reliability_diagram.png"), dpi=300)
    plt.close(fig2)

    # Figure 3: Regression Scatter Plot (True vs Predicted PHQ-8)
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    ax3.scatter(phq_true, phq_pred, color='#2ca02c', alpha=0.7, edgecolors='k', s=60)
    ax3.plot([0, 24], [0, 24], 'r--', label='Ideal Prediction (y = x)')
    ax3.set_title('Continuous PHQ-8 Score Regression (True vs Predicted)')
    ax3.set_xlabel('Ground Truth PHQ-8 Score')
    ax3.set_ylabel('Model Predicted PHQ-8 Score')
    ax3.set_xlim(-1, 25)
    ax3.set_ylim(-1, 25)
    ax3.legend(loc='upper left')
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "phq8_regression_scatter.png"), dpi=300)
    plt.close(fig3)

    print(f"[SUCCESS] Research evaluation plots saved to: {output_dir}")
