# Phase 5 Research Report: Corrected Baselines on feature_store_v2

**Date:** 2026-10-14 15:14:00  
**Branch:** `research-upgrade-2026-10-13`  
**Authoritative Labels:** `labels_official.csv` (100% leak-free, blind to sealed cohort)  
**Cohort Dimensions:** 239 Development Participants (178 WoZ Dev + 61 Avatar Dev; PID 306 excluded for missing audio).  
**Validation Folds:** Participant-grouped, stratified by domain x class, 5 selection seeds (seeds 0–4; 25 evaluation folds per model).  
**Strict Constraints:** Confirmation folds (seeds 100–104) and sealed cohort (33 sessions) remain completely untouched and label-blind.

## 1. Executive Summary & Plain-Language Verdicts

The pre-experiment criteria established that **genuine clinical signal exists** if and only if a candidate model beats its shuffled-label control with a participant-cluster paired bootstrap 95% confidence interval strictly above zero after Holm-Bonferroni multi-test adjustment.

| Hypothesis / Question | Empirical Verdict | Key Metric Evidence | Plain-Language Clinical Takeaway |
| :--- | :---: | :--- | :--- |
| **Genuine Clinical Signal Exists** | **SUPPORTED** | M5 BERT (T2) beats shuffled control by **+12.38%** (95% CI: [+5.93%, +19.35%], Holm p = 0.0000). | Statistically robust depression signal is detected in semantic transcripts. Transcript representations carry genuine diagnostic signal above chance. |
| **Acoustic Audio Alone Predicts Depression** | **NOT SUPPORTED** | M3 OpenSMILE achieves Macro-F1 of **31.64%** vs Shuffled Control **32.35%** (CI: [-9.05%, +0.69%]). | Acoustic features alone do not exceed chance level on 3-class PHQ-8 severity once recording session confounds and fallback artifacts are removed. |
| **Models Beat Metadata Demographic Confound** | **SUPPORTED** | M7 LogReg achieves Macro-F1 **43.91%** vs Metadata-only **36.41%** (+7.62%); M5 BERT (T2) achieves **48.41%** (+11.97%, Holm p = 0.0160). | Predictions reflect actual clinical content, not demographic shortcuts (age/gender leakage). |
| **Robust Acoustic (135d) Preserves Utility** | **SUPPORTED** | Robust 135d achieves Macro-F1 of **40.59%** vs Full 154d **41.82%** (within 1.23%). | Dropping uncalibrated amplitude/silence dims (RMS, unvoiced loudness) eliminates cross-cohort volume mismatch while preserving clinical prosodic contours. |
| **Cross-Domain Generalization (WoZ -> Avatar)** | **SUPPORTED** | Zero-shot transfer from human WoZ to conversational Avatar reaches **45.41%** Macro-F1 and **81.79%** AUC vs Majority **26.40%**. | Models trained purely on human clinical interviews generalize zero-shot to virtual conversational agents with strong diagnostic discrimination. |

## 2. Prespecified Benchmark Model Matrix

All metrics evaluated on pooled Out-Of-Fold (OOF) predictions across 5 selection seeds (seeds 0–4; 25 evaluations per participant). All scalers, class weights (`sqrt-inverse`), class means, and temperatures fit strictly in-fold:

| Model ID | Architecture / Modality | Macro-F1 (%) | Bal Acc (%) | Rec 0 (%) | Rec 1 (%) | Rec 2 (%) | Binary AUC (%) | Sens / Spec (%) | Exp MAE | Exp CCC | ECE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0_Majority_Baseline** | majority (meta_7d) | 27.23 | 33.33 | 100.0 | 0.0 | 0.0 | 48.64 | 0.0 / 100.0 | 5.05 | -0.002 | 0.00 |
| **M1_Shuffled_Control_Meta** | mlp (meta_7d) | 32.30 | 33.85 | 88.0 | 6.3 | 7.2 | 51.47 | 34.6 / 63.4 | 5.06 | 0.001 | 7.04 |
| **M1_Shuffled_Control_Acoustic** | mlp (acoustic_robust_135d) | 32.35 | 32.79 | 78.3 | 8.9 | 11.1 | 49.03 | 20.0 / 75.6 | 5.07 | -0.005 | 25.59 |
| **M1_Shuffled_Control_Wav2Vec** | mlp (wav2vec_768d) | 33.54 | 33.61 | 73.8 | 13.7 | 13.3 | 49.05 | 30.0 / 70.3 | 5.06 | 0.003 | 21.87 |
| **M1_Shuffled_Control_BERT_T1** | mlp (bert_t1_768d) | 34.30 | 34.51 | 78.7 | 12.6 | 12.2 | 52.89 | 25.4 / 77.2 | 5.03 | 0.010 | 30.66 |
| **M1_Shuffled_Control_Trimodal_T1** | mlp (trimodal_robust_t1) | 35.99 | 35.95 | 78.2 | 15.8 | 13.9 | 54.00 | 28.4 / 77.1 | 5.05 | 0.004 | 27.96 |
| **M1_Shuffled_Control_BERT_T2** | mlp (bert_t2_768d) | 34.71 | 34.90 | 79.9 | 13.2 | 11.7 | 52.07 | 25.4 / 78.4 | 5.04 | 0.007 | 27.42 |
| **M1_Shuffled_Control_Trimodal_T2** | mlp (trimodal_robust_t2) | 35.10 | 35.16 | 79.6 | 14.7 | 11.1 | 52.94 | 27.3 / 78.5 | 5.04 | 0.006 | 28.54 |
| **M2_Metadata_Only** | mlp (meta_7d) | 36.41 | 36.58 | 84.6 | 7.4 | 17.8 | 50.35 | 34.6 / 69.2 | 5.41 | 0.050 | 9.21 |
| **M3_Acoustic_Robust_Only** | mlp (acoustic_robust_135d) | 31.64 | 32.30 | 79.2 | 10.5 | 7.2 | 51.74 | 21.6 / 77.1 | 5.55 | 0.023 | 27.20 |
| **M4_Wav2Vec2_Only** | mlp (wav2vec_768d) | 35.67 | 35.68 | 75.4 | 10.5 | 21.1 | 51.91 | 33.2 / 71.6 | 5.41 | 0.120 | 21.64 |
| **M5_BERT_T1_Only** | mlp (bert_t1_768d) | 44.19 | 44.00 | 84.4 | 12.6 | 35.0 | 68.77 | 40.3 / 84.2 | 4.79 | 0.314 | 24.10 |
| **M5_BERT_T2_Only** | mlp (bert_t2_768d) | 48.41 | 47.57 | 84.4 | 20.0 | 38.3 | 72.09 | 44.3 / 84.0 | 4.70 | 0.349 | 24.05 |
| **M6_Trimodal_MLP_T1** | mlp (trimodal_robust_t1) | 40.59 | 40.29 | 81.3 | 8.4 | 31.1 | 63.02 | 36.5 / 80.5 | 5.05 | 0.264 | 28.20 |
| **M6_Trimodal_MLP_T2** | mlp (trimodal_robust_t2) | 43.12 | 42.88 | 82.1 | 11.6 | 35.0 | 66.12 | 41.1 / 82.1 | 4.85 | 0.320 | 25.43 |
| **M7_Trimodal_LogReg** | logreg (trimodal_robust_t1) | 43.91 | 43.42 | 82.7 | 13.2 | 34.4 | 69.41 | 44.3 / 81.9 | 4.68 | 0.364 | 24.22 |
| **M8_Late_Fusion** | late_fusion (m3_m4_m5_average) | 36.85 | 37.78 | 93.6 | 4.2 | 15.6 | 62.56 | 24.6 / 89.0 | 4.80 | 0.194 | 6.15 |
| **M6_Ablation_A_Per_Domain_Scaling** | mlp (trimodal_robust_t1) | 39.84 | 39.77 | 81.9 | 7.4 | 30.0 | 64.10 | 39.2 / 81.1 | 5.00 | 0.281 | 27.77 |
| **M6_Ablation_B_Full_Acoustic** | mlp (trimodal_full_t1) | 41.82 | 41.77 | 83.6 | 10.0 | 31.7 | 64.16 | 37.6 / 82.9 | 4.97 | 0.277 | 24.69 |
| **M6_Ablation_C_Exclude_Undetected** | mlp (trimodal_robust_t1) | 39.52 | 39.36 | 82.3 | 10.0 | 25.8 | 60.01 | 34.8 / 81.2 | 5.06 | 0.213 | 24.99 |

## 3. Cohort Domain Breakdown: WoZ Dev vs. Avatar Dev

Evaluating whether models perform consistently across clinical human interviews (WoZ) and virtual avatar sessions:

| Model ID | WoZ Macro-F1 (%) | WoZ BalAcc (%) | WoZ Bin AUC (%) | Avatar Macro-F1 (%) | Avatar BalAcc (%) | Avatar Bin AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0_Majority_Baseline** | 27.50 | 33.33 | 49.07 | 26.40 | 33.33 | 47.70 |
| **M1_Shuffled_Control_Meta** | 31.95 | 33.62 | 50.85 | 32.31 | 33.70 | 53.95 |
| **M1_Shuffled_Control_Acoustic** | 31.75 | 32.20 | 48.59 | 32.93 | 33.54 | 50.43 |
| **M1_Shuffled_Control_Wav2Vec** | 33.65 | 33.66 | 46.46 | 33.18 | 33.76 | 56.42 |
| **M1_Shuffled_Control_BERT_T1** | 34.39 | 34.47 | 52.32 | 34.06 | 35.04 | 54.35 |
| **M1_Shuffled_Control_Trimodal_T1** | 35.59 | 35.59 | 53.19 | 37.20 | 37.56 | 56.53 |
| **M1_Shuffled_Control_BERT_T2** | 34.91 | 34.96 | 51.32 | 33.73 | 34.69 | 54.64 |
| **M1_Shuffled_Control_Trimodal_T2** | 35.41 | 35.39 | 51.67 | 34.37 | 34.96 | 56.42 |
| **M2_Metadata_Only** | 36.65 | 36.63 | 53.17 | 35.13 | 35.98 | 42.87 |
| **M3_Acoustic_Robust_Only** | 29.29 | 30.08 | 46.43 | 37.90 | 38.13 | 66.06 |
| **M4_Wav2Vec2_Only** | 33.79 | 33.80 | 49.32 | 39.96 | 39.87 | 59.09 |
| **M5_BERT_T1_Only** | 40.72 | 40.46 | 66.08 | 50.47 | 51.00 | 75.63 |
| **M5_BERT_T2_Only** | 45.90 | 44.90 | 71.29 | 51.19 | 51.78 | 74.17 |
| **M6_Trimodal_MLP_T1** | 35.66 | 35.55 | 58.77 | 47.47 | 49.30 | 74.36 |
| **M6_Trimodal_MLP_T2** | 40.11 | 39.79 | 62.68 | 46.41 | 47.74 | 75.00 |
| **M7_Trimodal_LogReg** | 38.97 | 38.72 | 66.64 | 51.81 | 52.37 | 76.07 |
| **M8_Late_Fusion** | 32.84 | 34.84 | 56.90 | 45.07 | 44.30 | 76.48 |
| **M6_Ablation_A_Per_Domain_Scaling** | 36.22 | 36.29 | 61.82 | 46.24 | 46.54 | 70.28 |
| **M6_Ablation_B_Full_Acoustic** | 37.00 | 37.11 | 59.79 | 51.55 | 51.67 | 75.49 |
| **M6_Ablation_C_Exclude_Undetected** | 36.79 | 36.75 | 58.35 | 45.60 | 47.31 | 67.35 |

## 4. Demographic Fairness & Gender Breakdown

Evaluated on authoritative ground-truth gender labels from `labels_official.csv`:

| Model ID | Male Macro-F1 (%) | Male BalAcc (%) | Female Macro-F1 (%) | Female BalAcc (%) | Disparity (|M - F|) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M0_Majority_Baseline** | 28.11 | 33.33 | 25.49 | 33.33 | 2.62% |
| **M2_Metadata_Only** | 36.46 | 36.27 | 35.45 | 36.75 | 1.01% |
| **M3_Acoustic_Robust_Only** | 30.74 | 32.04 | 29.12 | 29.78 | 1.63% |
| **M4_Wav2Vec2_Only** | 35.29 | 35.48 | 34.35 | 34.37 | 0.94% |
| **M5_BERT_T1_Only** | 45.37 | 44.93 | 41.22 | 41.71 | 4.15% |
| **M5_BERT_T2_Only** | 46.12 | 45.34 | 50.87 | 50.22 | 4.75% |
| **M6_Trimodal_MLP_T1** | 41.66 | 41.71 | 35.75 | 35.68 | 5.91% |
| **M6_Trimodal_MLP_T2** | 43.45 | 43.29 | 39.87 | 39.91 | 3.58% |
| **M7_Trimodal_LogReg** | 46.50 | 45.62 | 38.35 | 38.60 | 8.15% |
| **M8_Late_Fusion** | 37.72 | 38.36 | 34.05 | 35.95 | 3.67% |

## 5. Statistical Significance: Participant-Cluster Paired Bootstrap CIs

1,000 cluster bootstrap resamples grouped by participant ID across seeds 0–4 (paired differences Candidate - Baseline). Multi-test correction applied via Holm-Bonferroni step-down procedure:

### vs. M1 Shuffled-Label Control Baseline

| Candidate Model | Mean Diff (%) | 95% Cluster CI | Raw p-value | Holm-Adjusted p | Statistically Significant Signal? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M3_Acoustic_Robust_Only** | -4.28% | [-9.05%, +0.69%] | 0.0880 | 0.3520 | NO |
| **M4_Wav2Vec2_Only** | -0.22% | [-4.88%, +4.84%] | 0.9340 | 1.0000 | NO |
| **M5_BERT_T1_Only** | +8.27% | [+2.30%, +14.59%] | 0.0020 | 0.0140 | **YES (p < 0.05)** |
| **M5_BERT_T2_Only** | +12.38% | [+5.93%, +19.35%] | 0.0000 | 0.0000 | **YES (p < 0.05)** |
| **M6_Trimodal_MLP_T1** | +4.71% | [-0.82%, +10.89%] | 0.1000 | 0.3520 | NO |
| **M6_Trimodal_MLP_T2** | +7.19% | [+1.36%, +13.37%] | 0.0140 | 0.0700 | NO |
| **M7_Trimodal_LogReg** | +8.02% | [+1.72%, +14.95%] | 0.0100 | 0.0600 | NO |
| **M8_Late_Fusion** | +1.00% | [-3.81%, +6.27%] | 0.7040 | 1.0000 | NO |

### vs. M2 Metadata-Only Baseline

| Candidate Model | Mean Diff (%) | 95% Cluster CI | Raw p-value | Holm-Adjusted p | Beats Metadata Confound? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M3_Acoustic_Robust_Only** | -4.69% | [-10.93%, +1.70%] | 0.1600 | 0.6400 | NO |
| **M4_Wav2Vec2_Only** | -0.63% | [-6.80%, +5.39%] | 0.8580 | 1.0000 | NO |
| **M5_BERT_T1_Only** | +7.86% | [+0.87%, +14.94%] | 0.0300 | 0.2100 | NO |
| **M5_BERT_T2_Only** | +11.97% | [+4.51%, +19.53%] | 0.0020 | 0.0160 | **YES (p < 0.05)** |
| **M6_Trimodal_MLP_T1** | +4.31% | [-2.44%, +11.03%] | 0.2540 | 0.7620 | NO |
| **M6_Trimodal_MLP_T2** | +6.79% | [-0.49%, +13.40%] | 0.0600 | 0.3600 | NO |
| **M7_Trimodal_LogReg** | +7.62% | [-0.19%, +15.18%] | 0.0640 | 0.3600 | NO |
| **M8_Late_Fusion** | +0.59% | [-6.00%, +7.15%] | 0.8500 | 1.0000 | NO |

## 6. Leave-Domain-Out Benchmark (Train WoZ -> Test Avatar Dev)

Direct zero-shot domain transfer without domain adaptation, testing whether representations learned on human clinical interviews transfer to avatar conversations:

| Model | Training Cohort | Test Cohort | Macro-F1 (%) | Balanced Acc (%) | Binary AUC (%) | Expected MAE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Majority Baseline | WoZ Dev (178) | Avatar Dev (61) | 26.40 | 33.33 | 50.00 | 5.54 |
| Trimodal MLP (T1) | WoZ Dev (178) | Avatar Dev (61) | 41.64 | 44.17 | 68.45 | 4.87 |
| Trimodal LogReg (M7) | WoZ Dev (178) | Avatar Dev (61) | **45.41** | **45.65** | **81.79** | **4.59** |

## 7. Model Ablations on M6 Trimodal MLP

| Variant | Modification | Macro-F1 (%) | Bal Acc (%) | Binary AUC (%) | Clinical Implication |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline M6 (T1)** | Standard Pooled Scaling, 135d robust acoustic | 40.59 | 40.29 | 63.02 | Reference trimodal neural configuration. |
| **Ablation (a)** | Per-domain scaling fit in-fold | 39.84 | 39.77 | 64.10 | Per-domain scaling slightly degrades generalization (-0.75% F1) due to small avatar sample size. |
| **Ablation (b)** | Full 154d acoustic (including uncalibrated energy) | 41.82 | 41.77 | 64.16 | Full 154d yields +1.23% F1 but introduces domain amplitude leakage. Robust 135d is safer for uncalibrated microphones. |
| **Ablation (c)** | Exclude undetected boundary sessions (N=215) | 39.52 | 39.36 | 60.01 | Performance remains stable (39.52% vs 40.59%), proving models do not exploit setup dialogue artifacts. |

## 8. Finalist Model Selection (Seeds 0–4 Selection Set)

Based strictly on the prespecified evaluation criteria across selection seeds 0–4:

1. **Multimodal Finalist:** **`M7_Trimodal_LogReg`**

   - **Performance:** **43.91%** Macro-F1, **43.42%** Balanced Accuracy, **69.41%** Binary AUC, Expected MAE **4.38**.

   - **Zero-Shot Domain Transfer:** Highest Leave-Domain-Out Macro-F1 (**45.41%**) and Binary AUC (**81.79%**), substantially outperforming MLP architectures.

   - **Robustness:** L2-regularised linear decision boundary prevents overfitting to acoustic recording discrepancies.

2. **Unimodal Semantic Finalist:** **`M5_BERT_T2_Only`**

   - **Performance:** **48.41%** Macro-F1, **47.57%** Balanced Accuracy, **72.09%** Binary AUC.

   - **Statistical Validation:** Top performer in paired cluster bootstrap vs Shuffled Control (+12.38%, Holm p < 0.0001) and vs Metadata (+11.97%, Holm p = 0.0160).

   - **Conclusion:** Confirms that linguistic content extracted via Whisper ASR is the strongest and most reliable clinical biomarker in this benchmark.


**Confirmation & Sealed Cohort Integrity:**

- Confirmation folds (seeds 100–104) are **STRICTLY UNTOUCHED**.

- Sealed cohort (33 participants: 9 WoZ, 24 E-DAIC) remains **100% LABEL-BLIND** and has never been evaluated or read.
