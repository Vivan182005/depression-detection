import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shap


def plot_attention_heatmap(attention_weights, save_path=None):
    """
    Plots a heatmap showing the distribution of attention weights over frames.
    """
    plt.figure(figsize=(12, 2))
    sns.heatmap(
        np.array(attention_weights).reshape(1, -1),
        cmap="Oranges",
        cbar=True,
        xticklabels=10,
        yticklabels=False
    )
    plt.title("LSTM Attention Weights Over Speech Chunks")
    plt.xlabel("Frame Index")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_attention_waveform_overlay(y, sr, attention_weights, frame_len_sec=2.0, hop_len_sec=1.0, save_path=None):
    """
    Plots the audio waveform and overlays the attention weights as a smooth, semi-transparent colored band.
    """
    duration = len(y) / sr
    time_axis = np.linspace(0, duration, len(y))
    
    # Calculate timestamps for each frame
    num_frames = len(attention_weights)
    frame_times = np.array([i * hop_len_sec + (frame_len_sec / 2.0) for i in range(num_frames)])
    
    # Interpolate attention weights to match audio time axis for seamless coloring
    # Normalize attention weights for transparency scaling
    weights = np.array(attention_weights)
    if weights.max() > 0:
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    else:
        norm_weights = weights
        
    interpolated_weights = np.interp(time_axis, frame_times, norm_weights)

    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot raw waveform in standard gray/blue
    ax.plot(time_axis, y, color='#95a5a6', alpha=0.6, label="Participant Speech")
    
    # Create colored fill indicating attention level
    # We will color sections above a threshold orange/red to indicate high focus/salience
    # Or plot a smooth line of attention weights
    ax2 = ax.twinx()
    ax2.plot(time_axis, interpolated_weights, color='#e74c3c', linewidth=2, label="Attention Weight (Salience)")
    ax2.fill_between(time_axis, 0, interpolated_weights, color='#e74c3c', alpha=0.15)
    ax2.set_ylabel("Attention Score (0 - 1)")
    ax2.set_ylim(-0.1, 1.1)

    ax.set_title("Vocal Energy Waveform with Salient Attention Overlay")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def explain_traditional_model(model, X_train_agg, X_explain_agg, feature_names=None, save_path=None):
    """
    Computes SHAP value explanations for traditional ML models (XGBoost/SVM).
    """
    # Background summary for faster SHAP execution
    background = shap.kmeans(X_train_agg, 5) if len(X_train_agg) > 5 else X_train_agg
    
    # KernelExplainer handles both SVM and XGBoost agnostic of implementation details
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_explain_agg)
    
    plt.figure()
    # If binary classification, shap_values is a list for both classes, select class 1
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
        
    shap.summary_plot(shap_values_class1, X_explain_agg, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Depression Predictors)")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
