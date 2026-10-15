# Research Experiment Plan & Protocol Specification

**Project**: Explainable Multi-Modal Speech & Language Depression Detection System  
**Working Repository**: `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Date**: October 13, 2026  
**Auditor**: Senior Machine Learning & Research Specialist  
**Status**: Approved & Active  

---

## 1. Experimental Protocol & Leakage-Free Design

To ensure scientific validity and reproducibility, all experiments adhere to the following principles:

1. **Sealed Unseen Holdout**: The 33 reserved unseen sessions (`unseen/`) are strictly sealed and excluded from all model training, hyperparameter sweeps, and feature selection.
2. **Participant-Level Grouped Splitting**: To avoid participant leakage from duplicated sessions (e.g., PIDs 301 and 302), all development data is split using **5-Fold StratifiedGroupKFold** grouped strictly on `participant_id`.
3. **Training-Fold-Only Fitting**:
   - Feature scalers (`StandardScaler`) are fit exclusively on the training partition of each fold.
   - Class balancing weights ($) and focal loss parameters are computed exclusively from training fold labels.
   - Data augmentations are applied solely to training fold samples.
4. **Validation-Only Model Selection & Calibration**:
   - Best checkpoint selection is governed by validation fold Macro-F1.
   - Temperature scaling ($) is calibrated using validation fold logits.

---

## 2. Master Experiment Pipeline

### Phase 4: Candidate 2A Baseline Reproduction
- **Model**: `StandardTriModalMLP` ( \rightarrow 128 \rightarrow 64 \rightarrow 3$ with auxiliary PHQ-8 and binary heads).
- **Loss**: Cross-Entropy with mild class balancing ( \propto 1 / \sqrt{N_c}$) + zsh.1 \times \text{MSE} + 0.2 \times \text{BCE}$.
- **Metrics Collected**:
  - 3-Class Accuracy, Macro-F1, Balanced Accuracy.
  - Per-class Precision, Recall, and F1 (Minimal/Mild, Moderate, Severe).
  - Confusion Matrix across all Out-Of-Fold (OOF) predictions.
  - Binary ($\text{PHQ} \ge 10$) Sensitivity, Specificity, Precision, F1, PR-AUC.
  - Expected PHQ-8 MAE, RMSE.
  - Calibration: Brier Score, Expected Calibration Error (ECE).

### Experiment A: Classical Machine Learning Baselines
- **Algorithms**: Support Vector Machine (RBF and Linear kernels) and L2-regularized Logistic Regression.
- **Configurations**:
  1. Unimodal Acoustic (154d)
  2. Unimodal Wav2Vec 2.0 (768d)
  3. Unimodal BERT (768d)
  4. Trimodal Concatenation (1690d)
- **Objective**: Establish whether deep neural representations genuinely outperform regularized linear and kernel baselines under identical participant folds.

### Experiment B: Modality Ablation Study
- Evaluates the marginal contribution of each modality under the deep MLP architecture:
  1. Acoustic Only (154d)
  2. Wav2Vec 2.0 Only (768d)
  3. BERT Only (768d)
  4. Acoustic + Wav2Vec 2.0 (922d)
  5. Acoustic + BERT (922d)
  6. Complete Trimodal (1690d)

### Experiment C: Class-Imbalance Strategies
- Compares loss formulations on identical folds:
  1. Standard Unweighted Cross-Entropy
  2. Inverse Frequency Class Weighting ( = N / (3 N_c)$)
  3. Smoothed Square-Root Weighting ( \propto 1 / \sqrt{N_c}$, Candidate 2A baseline)
  4. Focal Loss (Lin et al., $\gamma=1.5, 2.0$) with class-weight modulation.

### Experiment D: Modality-Aware Gating Network (MAGNet Architecture)
- Replaces rigid concatenation with dynamic confidence-based gating:
  4240g_{text} = \sigma(W_t h_t + b_t), \quad g_{audio} = \sigma(W_a h_a + b_a)4240
  4240\tilde{g}_m = \frac{\exp(g_m / \tau)}{\sum_k \exp(g_k / \tau)}4240
  4240h_{gated} = [\tilde{g}_{text} \cdot h_t \;\|\; \tilde{g}_{audio} \cdot h_a]4240
  Residual integration:
  4240h_{fused} = \text{LayerNorm}(h_{gated} + \text{Linear}([h_t, h_a]))4240

### Experiment E: Domain Robustness & Covariate Shift Analysis
- Evaluates distribution divergence between human Wizard-of-Oz and autonomous virtual avatar speech.
- Trains a domain classifier on acoustic and textual representations to measure domain separability ( \in \{0, 1\}$).
- Investigates Domain-Adversarial Neural Networks (DANN) using Gradient Reversal Layer (GRL).

### Experiment F: Segment-Level Temporal Aggregation & Audio Augmentation
- Evaluates temporal frame statistics vs. attentive pooling over temporal frames.
- Controlled acoustic noise/jitter augmentation applied exclusively to training fold samples.

### Experiment G: Severity Estimation & Calibration Consistency
- Compares:
  1. Independent classification and regression heads (Candidate 2A baseline).
  2. Single continuous PHQ-8 regression head deriving clinical severity classes from clinical boundaries (9, [10, 14], [15, 24]$).
  3. Joint ordinal logistic regression head.
