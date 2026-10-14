# Pre-Registration Document: Phase 6A Research Protocol
## Voice-Primary (Acoustic) and Text-Secondary Depression Severity Detection

**Protocol Version:** 1.0  
**Registration Date:** 2026-10-14  
**Repository Branch:** `research-upgrade-2026-10-13`  
**Working Directory:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Execution Gate:** PRE-REGISTRATION LOCKED PRIOR TO ANY MODEL TRAINING  
**Governing Standard:** Formal empirical pre-registration for clinical voice ML research  

---

## 1. Scope, Objectives & Research Questions

### 1.1 Objective
The primary clinical objective of Phase 6A is to rigorously evaluate whether voice representations (acoustic prosody and self-supervised audio representations) contain genuine, generalizable clinical signal for detecting 3-class PHQ-8 depression severity (None/Mild [0–9], Moderate [10–14], Severe [15–24]), when tested against strict empirical null baselines and linguistic (text) representations.

### 1.2 Core Research Hypotheses
1. **$H_1$ (Acoustic Primary Signal):** Voice-only representations (`acoustic_robust_135d` and/or `wav2vec_768d`) statistically outperform both an in-fold shuffled-label control ($M_1$) and an interview length/demographic metadata control ($M_2$) with Holm-adjusted $p < 0.05$.
2. **$H_2$ (Valid-Turn Prosodic Filtering):** Turn-level prosodic aggregation filtered strictly by physiological validity rules (two-pass $f_0$, octave jump rejection, minimum voiced duration $\ge 100$ ms) significantly improves voice classification over raw unsegmented functionals.
3. **$H_3$ (Multimodal Synergy vs. Text Dominance):** Multimodal fusion (voice + text) yields a statistically significant increase in Macro-F1 over the best unimodal component (`M5_BERT_T2_Only`), rejecting the null hypothesis that text alone explains all detectable clinical variance.
4. **$H_4$ (Zero-Shot Cross-Domain Generalization):** Models trained on WoZ clinical interviews transfer to continuous avatar interviews above an in-fold transfer shuffled-label control, with bootstrap 95% confidence intervals strictly above zero.
5. **$H_5$ (Subgroup Equity):** The demographic performance disparity between male and female participants ($|\text{Macro-F1}_{\text{Male}} - \text{Macro-F1}_{\text{Female}}|$) remains $< 10.0$ percentage points without subgroup-sacrificing optimization.

---

## 2. Cohorts, Partitions & Data Boundary Gates

### 2.1 Active Development Cohort ($N = 239$)
All feature analysis, cross-validation, hyperparameter tuning, and model selection in Phase 6A are restricted exclusively to the 239 active development sessions:
- **DAIC-WOZ Development:** $N = 178$ sessions (PID 306 excluded due to missing source audio).
- **Avatar Development (E-DAIC Dev):** $N = 61$ sessions.
- **Demographics:** Female $N = 84$ (35.1%), Male $N = 155$ (64.9%).
- **Severity Distribution:** Class 0 (None/Mild) = 165 (69.0%), Class 1 (Moderate) = 38 (15.9%), Class 2 (Severe) = 36 (15.1%).

### 2.2 Strictly Frozen Cohorts & Partitions
1. **Sealed Evaluation Cohort ($N = 33$):**
   - Comprising 9 WoZ test sessions and 24 E-DAIC test sessions.
   - **STRICT PROTOCOL:** Must NOT be read, segmented, extracted, or evaluated in Phase 6A.
   - Remains completely label-blind, sequestered, and untouched.
2. **Confirmation Folds (Seeds 100–104):**
   - Confirmation seeds 100, 101, 102, 103, 104 are strictly locked.
   - No feature fitting, tuning, or evaluation may be executed on confirmation seeds during Phase 6A.
3. **Production Repository:**
   - Production path `/Users/vivannaik/Desktop/CAPSTONE` remains strictly read-only and untouched.
   - All work executed exclusively in `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`.

---

## 3. Fixed Feature Matrix & Representational Encodings

| Modality | Feature Representation | Dimensions | Source & Preprocessing Specification |
| :--- | :--- | :---: | :--- |
| **Control / Baseline** | `majority_baseline` | 0 | Majority class assignment (Class 0). |
| **Control / Baseline** | `shuffled_control_m1` | N/A | In-fold random permutation of training labels under identical model pipeline. |
| **Control / Baseline** | `meta_7d` | 7 | Interview duration, speech duration, turn count, mean turn duration, speaking rate, age, gender. |
| **Acoustic (Primary)** | `acoustic_robust_135d` | 135 | OpenSMILE eGeMAPSv02 session functionals with uncalibrated 19d loudness/silence functionals removed. |
| **Acoustic (Primary)** | `acoustic_valid_turn_135d` | 135 | Median/MAD aggregation across valid voiced turns ($\ge 100$ ms voiced duration, two-pass $f_0$, octave-jump filter). |
| **Audio SSL (Primary)** | `wav2vec_768d` | 768 | `facebook/wav2vec2-base-960h` mean-pooled transformer representations from participant speech. |
| **Audio Combined** | `audio_primary_903d` | 903 | Concatenation of `acoustic_robust_135d` + `wav2vec_768d`. |
| **Text (Secondary)** | `bert_t2_768d` | 768 | `bert-base-uncased` representations from Whisper-base / transcript text. |
| **Multimodal (Early)** | `multimodal_concat_1671d` | 1,671 | Concatenation of `acoustic_robust_135d` + `wav2vec_768d` + `bert_t2_768d`. |
| **Multimodal (Late)** | `multimodal_late_fusion` | N/A | Weighted ensemble of out-of-fold predicted class probabilities: $P = \alpha P_{\text{Audio}} + (1 - \alpha) P_{\text{Text}}$. |

---

## 4. Model Architectures & Hyperparameter Grids

All classifiers optimize multi-class cross-entropy or negative log-likelihood with balanced class weighting:
$$w_c = \frac{N}{3 \cdot N_c}$$

### 4.1 Regularized Multinomial Logistic Regression (`LogReg`)
- **Loss:** Multinomial cross-entropy with $L_2$ regularization.
- **Solver:** `lbfgs`, maximum iterations = 1,000.
- **Hyperparameter Grid (Tuned In-Fold):**
  - $C \in \{0.001, 0.01, 0.1, 1.0, 10.0, 100.0\}$.
  - Class weighting: `balanced`.

### 4.2 LayerNorm Multi-Layer Perceptron (`MLP`)
- **Architecture:** Input $\to$ Dense(128) $\to$ LayerNorm $\to$ LeakyReLU($\alpha=0.1$) $\to$ Dropout(0.3) $\to$ Dense(64) $\to$ LayerNorm $\to$ LeakyReLU($\alpha=0.1$) $\to$ Dropout(0.2) $\to$ Dense(3) $\to$ Softmax.
- **Optimizer:** AdamW ($\text{weight\_decay} = 0.01$).
- **Learning Rate:** $1 \times 10^{-3}$ with cosine annealing schedule over 60 epochs.
- **Batch Size:** 32.
- **Early Stopping:** Monitored on validation fold Macro-F1 (patience = 12 epochs).

### 4.3 Gradient Boosted Decision Trees (`HistGBDT` / `LightGBM`)
- **Architecture:** Histogram-based gradient boosting for multi-class classification.
- **Regularization Constraints (Tuned In-Fold):**
  - Max depth: $\{3, 4\}$.
  - Min child samples: $\{10, 15\}$.
  - Learning rate: $\{0.03, 0.05\}$.
  - L2 regularization: $\{1.0, 5.0\}$.

---

## 5. Leak-Free Cross-Validation Protocol

### 5.1 Partitioning Structure
- **5-Fold Stratified Group K-Fold Cross-Validation:**
  - Grouping by `participant_id` (ensures zero participant overlap across train and validation folds).
  - Stratified by clinical class label (proportions of Class 0, 1, 2 maintained across folds).
- **Selection Seeds:** Evaluated across 5 random seeds: `seed ∈ {0, 1, 2, 3, 4}` (total 25 fold evaluations per model).
- **Pooled Out-of-Fold (OOF) Prediction:** Concatenation of held-out validation fold predictions across the full development cohort ($N=239$).

### 5.2 Strict In-Fold Processing Pipeline
To eliminate data leakage, the following operations are computed strictly inside each training fold and applied statelessly to validation folds:
1. **Feature Standardization:** `StandardScaler` fitted on training fold indices only.
2. **Outlier Winsorisation:** $1^{\text{st}}$ and $99^{\text{th}}$ percentiles computed on training fold only.
3. **Median Imputation:** Feature medians computed on training fold only.
4. **Missing-Data Indicators:** Appended statelessly without target reference.
5. **Hyperparameter Selection:** Inner nested validation or in-fold grid search; zero validation data peeked.
6. **Class-Mean Calibration Targets:** Class centroid values for Expected MAE computed on training fold labels only.

---

## 6. Prespecified Endpoints & Evaluation Metrics

### 6.1 Primary Endpoint
- **Macro-Averaged F1 Score ($\text{Macro-F1}$):**
  $$\text{Macro-F1} = \frac{1}{3} \sum_{c=0}^{2} F1_c = \frac{1}{3} \sum_{c=0}^{2} \frac{2 \cdot \text{Recall}_c \cdot \text{Precision}_c}{\text{Recall}_c + \text{Precision}_c}$$
  Enforces equal weighting across None/Mild, Moderate, and Severe depression classes regardless of class imbalance.

### 6.2 Secondary Endpoints
1. **Balanced Accuracy:**
   $$\text{Bal-Acc} = \frac{1}{3} \sum_{c=0}^{2} \text{Recall}_c$$
2. **Binary Clinical AUC (ROC-AUC):**
   - Area under the ROC curve for detecting clinically significant depression (Class 0 [0–9] vs. Class 1+2 [10–24]).
   - Continuous score: $P(\text{Class } 1) + P(\text{Class } 2) = 1 - P(\text{Class } 0)$.
3. **Continuous Expected MAE (PHQ-8 Error):**
   - Mean absolute error between true continuous PHQ-8 score and expected score:
     $$\hat{y}_i = \sum_{c=0}^{2} P(Y_i = c) \cdot \bar{y}_c^{\text{train}}$$
     where $\bar{y}_c^{\text{train}}$ is the empirical mean PHQ-8 score of class $c$ in the training fold.
4. **Demographic Subgroup Disparity:**
   - $\text{Macro-F1}_{\text{Male}}$ ($N=155$) vs. $\text{Macro-F1}_{\text{Female}}$ ($N=84$).
   - Disparity Index: $\Delta_{\text{Gender}} = |\text{Macro-F1}_{\text{Male}} - \text{Macro-F1}_{\text{Female}}|$.

---

## 7. Statistical Testing & Decision Framework

### 7.1 Participant-Cluster Paired Bootstrap
Because repeated evaluation seeds share identical participants, standard i.i.d. tests are statistically invalid. We employ paired participant-cluster bootstrap:
1. Resample $N = 239$ participant IDs with replacement (10,000 bootstrap iterations).
2. For each bootstrap iteration $b$, compute the paired difference:
   $$\Delta^{(b)} = \text{Metric}(\text{Model})^{(b)} - \text{Metric}(\text{Control})^{(b)}$$
3. Compute the two-sided empirical percentile 95% confidence interval: $[P_{2.5}, P_{97.5}]$.
4. Compute the empirical $p$-value:
   $$p = 2 \cdot \min \left( \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\Delta^{(b)} \le 0), \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\Delta^{(b)} \ge 0) \right)$$

### 7.2 Multiplicity Correction
- All $p$-values adjusted using the **Holm-Bonferroni step-down procedure** across family-wise hypothesis tests.

### 7.3 Formal Gating Rules for Model Advancement
A model $M$ shall be certified as a validated research advancement IF AND ONLY IF:
1. $\Delta_{\text{Macro-F1}}(M - M_1) > 0$ with Holm-adjusted $p < 0.05$ (beats shuffled-label null).
2. $\Delta_{\text{Macro-F1}}(M - M_2) > 0$ with Holm-adjusted $p < 0.05$ (beats metadata length/demographic baseline).
3. For multimodal models ($M_{\text{Multi}}$): $\Delta_{\text{Macro-F1}}(M_{\text{Multi}} - M_{\text{Text}}) > 0$ with Holm-adjusted $p < 0.05$ (multimodal synergy gate).
4. Subgroup equity gate: $\Delta_{\text{Gender}} \le 12.0\%$, with neither male nor female Macro-F1 falling below $35.0\%$.

---

## 8. Cross-Domain Zero-Shot Transfer Protocol

To evaluate real-world generalization across interview modalities:
- **Training Domain:** DAIC-WOZ Development Cohort ($N = 178$).
- **Test Domain:** Avatar Development Cohort ($N = 61$).
- **Evaluation Requirements:**
  1. Train models on 178 WoZ sessions using in-fold hyperparameter selection.
  2. Evaluate out-of-domain on 61 Avatar sessions.
  3. **Transfer Shuffled Control:** Measure transfer performance of the same pipeline trained on shuffled labels.
  4. Compute cluster bootstrap 95% CIs on the domain-transfer delta ($\text{Model}_{\text{Transfer}} - \text{Shuffled}_{\text{Transfer}}$).
  5. Report transfer Macro-F1, Balanced Accuracy, Binary AUC, and Expected MAE.

---

## 9. Pre-Registration Verification & Execution Status

- **Status:** **FROZEN & COMMITTED BEFORE ANY PHASE 6A MODEL TRAINING.**
- **Model Training Execution:** **ZERO model runs have been initiated.**
- **Sealed Cohort Integrity:** **The 33 sealed participants remain unread, unsegmented, unextracted, and label-blind.**
- **Confirmation Folds:** **Seeds 100–104 remain strictly untouched.**

---
*Signed and registered on branch `research-upgrade-2026-10-13` before Phase 6A execution.*
