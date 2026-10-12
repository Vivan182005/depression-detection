"""
Module: visualizer.py
Description: Explainable AI (XAI) and Clinical Visualization Suite for Depression Biomarkers.
Generates temporal attention heatmaps across the interview timeline and prosodic feature importance plots.
"""

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_temporal_attention(attention_weights, chunk_times, session_id="300", save_path="attention_heatmap.png"):
    """
    Plots timeline heatmaps highlighting critical speech segments where the model 
    detected high depressive acoustic cues.

    Parameters:
        attention_weights (np.ndarray or list): Attention weights alpha_t per chunk
        chunk_times (list of tuples): List of (start_sec, end_sec) per chunk
        session_id (str): Identifier of the session/subject
        save_path (str): Output image filepath
    """
    weights = np.array(attention_weights).flatten()
    num_chunks = len(weights)
    
    if num_chunks == 0:
        print("[WARNING] Empty attention weights provided.")
        return

    mid_times_min = [(t[0] + t[1]) / 120.0 for t in chunk_times] # Convert to minutes

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(14, 4))
        
        # Plot attention weights curve
        plt.plot(mid_times_min, weights, color='#d9534f', linewidth=2, label='Attention Weight (α_t)')
        plt.fill_between(mid_times_min, weights, color='#d9534f', alpha=0.3)
        
        # Highlight top 10% highest attention moments
        threshold = np.percentile(weights, 90)
        high_attention = weights >= threshold
        plt.scatter(
            np.array(mid_times_min)[high_attention], 
            weights[high_attention], 
            color='#c9302c', s=50, zorder=5, label='Salient Depressive Cue'
        )

        plt.title(f"Clinical Temporal Attention Timeline - Session {session_id}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Interview Timeline (Minutes)", fontsize=12)
        plt.ylabel("Self-Attention Weight", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Saved temporal attention heatmap to: {save_path}")
    else:
        print(f"[INFO] Matplotlib not installed. Attention summary: Max weight {np.max(weights):.4f} at chunk {np.argmax(weights)}")


def plot_prosodic_biomarkers(pause_metrics, egemaps_summary=None, save_path="biomarkers.png"):
    """Plots key prosodic biomarkers such as pause duration, pause ratio, and pitch variability."""
    if not HAS_MATPLOTLIB:
        print("[INFO] Matplotlib not installed. Skipping biomarker plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Pause Ratio
    axes[0].bar(['Pause Ratio'], [pause_metrics.get('pause_ratio', 0.0)], color='#5bc0de', width=0.4)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Ratio (0 to 1)")
    axes[0].set_title("Pause-to-Speech Ratio", fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Mean Pause Duration
    axes[1].bar(['Mean Pause (s)'], [pause_metrics.get('mean_pause_duration', 0.0)], color='#f0ad4e', width=0.4)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Mean Silence Duration", fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Total Pause Count
    axes[2].bar(['Pause Count'], [pause_metrics.get('pause_count', 0)], color='#5cb85c', width=0.4)
    axes[2].set_ylabel("Count")
    axes[2].set_title("Total Hesitation Pauses", fontweight='bold')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle("Clinical Vocal Biomarker Profile", fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved biomarker profile plot to: {save_path}")


def generate_diagnostic_summary(depression_prob, phq8_score, pause_metrics, session_id="300"):
    """Generates a text-based clinical screening report."""
    status = "DEPRESSED / HIGH RISK" if depression_prob >= 0.5 else "NON-DEPRESSED / LOW RISK"
    
    report = f"""
================================================================================
                    DEPRESSION SCREENING DIAGNOSTIC REPORT
================================================================================
Session ID               : {session_id}
Diagnostic Status        : {status}
Depression Probability   : {depression_prob * 100:.2f}%
Estimated PHQ-8 Score    : {phq8_score:.1f} / 24.0

VOCAL BIOMARKER SUMMARY:
--------------------------------------------------------------------------------
- Pause-to-Speech Ratio : {pause_metrics.get('pause_ratio', 0.0) * 100:.1f}%
- Mean Pause Duration   : {pause_metrics.get('mean_pause_duration', 0.0):.2f} seconds
- Total Pause Count     : {pause_metrics.get('pause_count', 0)} hesitations
================================================================================
"""
    return report
