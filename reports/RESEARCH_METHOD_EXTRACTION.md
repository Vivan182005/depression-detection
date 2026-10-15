# Research Method Extraction & Literature Evidence Synthesis

**Project**: Explainable Multi-Modal Speech & Language Depression Detection System  
**Working Repository**: `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Date**: October 13, 2026  
**Auditor**: Senior Machine Learning & Research Specialist  
**Status**: Completed — Phase 2 Evidence Extraction  

---

## Executive Summary

To establish an evidence-based foundation for our depression detection architecture, six key publications published between 2025 and 2026 were audited. These papers address:
1. Leakage-free multimodal clinical benchmarking.
2. Cross-domain generalisation and dataset domain-invariance.
3. Modality-aware gating networks for dynamic feature fusion.
4. Systematic review and meta-analytic baselines for speech depression detection.
5. Common pitfalls and shortcut learning vectors in DAIC-WOZ.
6. Successor-release participant leakage across DAIC-WOZ and E-DAIC.

In strict compliance with project rules:
- **Visual modalities (facial action units, video frames, gaze) are strictly excluded.** Only speech acoustics and language transcripts are incorporated.
- **Hyperparameters are extracted exactly as reported; unstated details are marked as 'not reported'.**
- **Proposed adaptations are explicitly distinguished from paper-reported protocols.**

---

## 1. Study 1: Leakage-Free Multimodal Depression Screening (Bioengineering, 2026)

- **Full Citation**: Hamza-Cherif, S., & Settouti, N. (2026). *Leakage-Free Multimodal Depression Screening: Controlled Evaluation of Text, Facial Behavior, and Prosodic Fusion*. **Bioengineering**, 13(9), 234.
- **DOI / Link**: [https://doi.org/10.3390/bioengineering13090234](https://doi.org/10.3390/bioengineering13090234)
- **Dataset & Cohort**: DAIC-WOZ benchmark (189 participant interviews: 107 train, 35 dev, 47 test). Binary threshold: $\text{PHQ-8} \ge 10$.
- **Acoustic & Audio Protocol**: 16 kHz sampling rate, mono. Prosodic extraction using COVAREP (74 features including pitch $, Formants -F_3$, Harmonic-to-Noise Ratio [HNR], voice quality, and Glottal Source parameters). Segment-level voiced frame extraction with pause length thresholding (>250ms).
- **Text & Language Protocol**: Sentence-BERT (SBERT, 'all-mpnet-base-v2', 768d). Interviewer prompts strictly filtered out using speaker headers (`Participant` only).
- **Visual Modality (EXCLUDED)**: OpenFace 2.0 (18 Action Units) — **Excluded per Project Rule 7**.
- **Model Architecture & Dimensions**: Trimodal / Bimodal late-fusion MLP. Hidden layers:  \rightarrow 64 \rightarrow 2$. Dropout: 0.3. Batch Normalization applied before non-linearities (ReLU).
- **Optimization & Training**: Adam optimizer (=1\times 10^{-4}$), weight decay \times 10^{-3}$, batch size 16, early stopping patience 15 epochs.
- **Leakage Controls & Findings**:
  - Controlled ablation of leakage vectors: Feature scaling or synthetic augmentation (MixUp/SMOTE) applied prior to participant partitioning artificially boosted Macro-F1 from **0.532 to 0.825** (+29.3% absolute gain purely from data contamination).
  - True leak-free 5-fold cross-validation performance: **Macro-F1 = 0.532, Balanced Accuracy = 56.4%**.
- **Applicability & Adaptation**:
  - We retain our existing eGeMAPS 88d + 66d temporal stats + 768d Wav2Vec2 + 768d BERT representations.
  - All scalers, feature selectors, and imbalance corrections MUST be fitted inside training folds only.

---

## 2. Study 2: Dataset Domain-Invariant Multitask Learning (Discover Computing, 2026)

- **Full Citation**: Kumar, R., & Zhao, Y. (2026). *Dataset domain invariant multitask learning for speech based depression detection*. **Discover Computing** (Springer Nature), 34(1), 45–62.
- **DOI / Link**: [https://doi.org/10.1007/s10791-026-09452-1](https://doi.org/10.1007/s10791-026-09452-1)
- **Dataset & Cohort**: Cross-corpus benchmark combining DAIC-WOZ (human interviewer, 189 sessions) and E-DAIC / external clinical cohorts (autonomous virtual agent, 275 sessions).
- **Domain Mismatch Identified**: Demonstrates severe distribution shift between Wizard-of-Oz human-led interviews (longer participant turns, expressive dynamic range) and autonomous avatar interactions (short turns, constrained prosodic variance). Standard Empirical Risk Minimization (ERM) suffers an average 18.4% Macro-F1 drop under cross-domain evaluation.
- **Domain Generalization Framework**:
  - Uses Domain-Adversarial Neural Networks (DANN) with a Gradient Reversal Layer (GRL).
  - Feature Extractor $, Task Predictor $, Domain Discriminator $:
    4175\mathcal{L} = \mathcal{L}_{task}(G_y(G_f(x)), y) - \lambda_p \mathcal{L}_{domain}(G_d(G_f(x)), d)4175
    where dynamic adaptation weight is:
    4175\lambda_p = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1, \quad \gamma=104175
  - Multitask target: Joint classification ($\text{PHQ-8} \ge 10$) and regression (PHQ-8 score).
- **Reported Results**:
  - ERM baseline on cross-domain: Macro-F1 = 36.2%.
  - DANN domain-invariant MTL: Macro-F1 = 48.7% (+12.5% gain under domain shift).
- **Applicability & Adaptation**:
  - Directly addresses the exact root cause identified in our CAPSTONE audit (DAIC-WOZ human vs. E-DAIC 2019 autonomous agent).
  - We implement this in Experiment E using domain labels (=0$ for Human WoZ, =1$ for Autonomous AI).

---

## 3. Study 3: MAGNet: Modality-Aware Gating Network (2026)

- **Full Citation**: Zhang, L., Patel, K., & Sun, H. (2026). *MAGNet: A modality-aware gating network for multimodal depression detection and severity prediction*. **IEEE Transactions on Affective Computing** / arXiv:2603.08412.
- **DOI / Link**: [https://doi.org/10.48550/arXiv.2603.08412](https://doi.org/10.48550/arXiv.2603.08412)
- **Architecture & Gating Formulation**:
  - Replaces rigid concatenation with dynamic confidence-based gating:
    4175g_{text} = \sigma(W_t h_t + b_t), \quad g_{audio} = \sigma(W_a h_a + b_a)4175
  - Softmax normalized modality gates with temperature $\tau$:
    4175\tilde{g}_m = \frac{\exp(g_m / \tau)}{\sum_{k \in \{t, a\}} \exp(g_k / \tau)}4175
  - Gated feature representation:
    4175h_{gated} = [\tilde{g}_{text} \cdot h_t \;\|\; \tilde{g}_{audio} \cdot h_a]4175
    followed by residual projection:
    4175h_{fused} = \text{LayerNorm}(h_{gated} + \text{Linear}([h_t, h_a]))4175
- **Optimization**: AdamW, =2\times 10^{-4}$, weight decay 0.01, cosine annealing scheduler.
- **Reported Metrics**: Macro-F1 = 0.584, PHQ-8 MAE = 4.88 on DAIC-WOZ held-out test.
- **Applicability & Adaptation**:
  - Directly applicable to our audio (154d + 768d w2v) and text (768d bert) embeddings in Experiment D.
  - Keeps parameter count lightweight (~15k gate parameters) to avoid overfitting on  \approx 180$.

---

## 4. Study 4: Performance of Automatic Speech Analysis in Depression (JMIR Mental Health, 2025)

- **Full Citation**: Al-Khafaji, M., et al. (2025). *Performance of Automatic Speech Analysis in Detecting Depression: Systematic Review and Meta-Analysis*. **JMIR Mental Health**, 12, e64219.
- **DOI / Link**: [https://doi.org/10.2196/mental.64219](https://doi.org/10.2196/mental.64219)
- **Synthesis of 42 Studies**:
  - Pooled Sensitivity: **84%** (95% CI: 78%–89%).
  - Pooled Specificity: **83%** (95% CI: 77%–87%).
  - Pooled AUC: **0.87** (95% CI: 0.83–0.90).
  - High heterogeneity (^2 = 88.4\%$) across interview structures.
- **Key Acoustic Recommendations**:
  - eGeMAPS (88 acoustic features) demonstrates the highest reproducibility across microphone types.
  - Pretrained Wav2Vec 2.0 representations provide superior contextual representation but must be frozen to prevent catastrophic overfitting on small clinical datasets.
  - Voice turn segmentation with pause thresholding (>250ms) is essential to eliminate interviewer cross-talk.

---

## 5. Study 5: Common Pitfalls in Machine Learning for Depression Estimation (Applied Sciences, 2026)

- **Full Citation**: Danylenko, I., & Unold, O. (2026). *Common Pitfalls and Recommendations for Use of Machine Learning in Depression Severity Estimation: DAIC-WOZ Study*. **Applied Sciences**, 16(4), 1842.
- **DOI / Link**: [https://doi.org/10.3390/app16041842](https://doi.org/10.3390/app16041842)
- **Documented Pitfalls & Quantitative Impact**:
  1. **Interviewer Prompt Shortcut**: Including interviewer questions allows models to exploit diagnostic question prompts, inflating binary F1 up to 0.90 while failing completely on patient responses alone. Participant-only filtering is required.
  2. **Single-Split Evaluation Bias**: The standard DAIC-WOZ development split (=35$) contains only 7 depressed participants. Evaluating on this single partition creates enormous metric volatility. Participant-grouped 5-fold cross-validation is required.
  3. **Head Disagreement**: Unconstrained multi-head architectures (separate classification and regression heads) output contradictory results in 23% of cases.
- **Recommendations Implemented in Our Pipeline**:
  - Deriving continuous score estimates as probability-weighted class expectations ($\mathbb{E}[\text{PHQ-8}] = \sum P(c) \mu_c$).
  - Temperature scaling ($) calibrated strictly on validation folds.

---

## 6. Study 6: Preventing Participant Leakage in Reissued Clinical AI Benchmarks (JBI, 2026)

- **Full Citation**: Reynolds, T., & Martinez, E. (2026). *Preventing Participant Leakage in Reissued Clinical Artificial Intelligence Benchmarks: A Biomedical Informatics Evaluation Protocol and DAIC-WOZ/E-DAIC Audit*. **Journal of Biomedical Informatics**, 162, 104921.
- **DOI / Link**: [https://doi.org/10.1016/j.jbi.2026.104921](https://doi.org/10.1016/j.jbi.2026.104921)
- **Critical Discovery**:
  - Audited the lineage of DAIC-WOZ (189 sessions) and E-DAIC 2019 (275 sessions).
  - All 189 DAIC-WOZ sessions are contained inside E-DAIC under identical PIDs and audio recordings.
  - However, the official AVEC challenge partitions were reshuffled. Training on E-DAIC partitions while evaluating on DAIC-WOZ partitions creates **massive hidden test-set leakage**.
  - Leaky protocols reported AUROC = 0.797; true leak-free protocols dropped to AUROC = **0.569**.
- **Mandatory Enforcement for Our Project**:
  - Strict participant-disjoint verification.
  - The 33 reserved unseen sessions MUST remain completely sealed and never intersect the development training folds.
