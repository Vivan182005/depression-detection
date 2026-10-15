# Pre-Registration Amendment (Version 1.3): Phase 6A Execution Protocol
## Voice-Primary Depression Detection (Audio Primary, Text Secondary)

**Protocol Amendment:** v1.3 (Supplements v1.1 and Amendment v1.2)  
**Date:** 2026-10-15  
**Repository Branch:** `research-upgrade-2026-10-13`  
**Working Directory:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Execution Gate:** LOCKED PRIOR TO ANY OBSERVATION OF REAL CLINICAL OUTCOMES  
**Outcome Status:** ZERO OBSERVED REAL-LABEL PREDICTIVE OUTCOMES SEEN  

---

## 1. Rationale & Purpose of Amendment v1.3

This amendment formally records four methodological locks prior to unblinding and evaluating any predictive outcomes on real clinical depression labels (PHQ-8 $\ge$ 10) across the 239 active development sessions:

1. **Inner-CV Tuning Metric:** Strictly locked to **ROC-AUC for binary $PHQ \ge 10$** (`scoring='roc_auc'`), superseding any ambiguous references to accuracy or balanced accuracy.
2. **Exact Prespecified Hyperparameter Grids:** Fully detailed mathematical specifications and candidate grids for all 19 variants.
3. **Pre-Stated Model Freeze Rule:** Algorithmic, deterministic selection rule for freezing exactly one primary audio model and one fusion model into `frozen_config.json`.
4. **Text Feature Representation Verification (V14):** Formal verification of 1:1 mapping onto all 239 development participants with zero exclusions.
5. **Analysis Code Tagging Requirement:** Code commit hash and Git tag (`prereg-exec-v1`) recorded prior to real-label evaluation.

---

## 2. Text Representation Verification (V14 Mapping)

- **Verification Target:** Complete 239 active development cohort ($N = 178$ DAIC-WOZ + $N = 61$ E-DAIC Avatar).
- **Representation Used:** Frozen `bert-base-uncased` pooled [CLS] representations across sliding windows (`bert_768d`) from `feature_store_v2/`.
- **Audit Finding:**
  - `bert_768d` is present, finite, non-zero, and valid for **239 of 239 participants (100.0%)**.
  - **Exclusions:** **ZERO exclusions** ($N_{\text{excluded}} = 0$).
  - *Provenance Note on Whisper-Tiny T2:* Fully local `openai/whisper-tiny` transcription (`bert_t2_768d`) is available for 116 sessions (all 61 Avatar + 55 WoZ), with 123 WoZ sessions untranscribed in pure T2 form. To maintain the pre-registered requirement of evaluating on the identical 239 participants across all 19 variants without participant exclusion, `bert_768d` serves as the complete canonical text feature matrix for V14, V18, and V19.

---

## 3. Exact Prespecified Hyperparameter Grids

All hyperparameter tuning is conducted strictly inside each outer training fold via an **inner 4-fold Stratified Group K-Fold cross-validation** using `scoring='roc_auc'`.

### 3.1 Linear & Regularized Models
- **V1 (eGeMAPS 135d + Elastic-Net Logistic Regression):**
  - Preprocessing: In-fold `StandardScaler` + winsorisation ($P_{01}^{\text{train}}, P_{99}^{\text{train}}$).
  - Estimator: `LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, tol=1e-3)`.
  - Grid:
    - $C \in \{0.01, 0.1, 1.0\}$
    - $\text{l1\_ratio} \in \{0.2, 0.5, 0.8\}$
    - Total candidate configurations: 9.
- **V2, V4, V6, V7, V8, V9, V10, V11, V14 ($L_2$ Logistic Regression):**
  - V2 (Confirmatory $H_1$, 330d), V4 (Normalised 330d), V6–V8 (W2V2 L4/8/12, 768d), V9–V11 (WavLM L4/8/12, 768d, including Confirmatory $H_2$ on V10), V14 (BERT Text, 768d).
  - Preprocessing: In-fold `StandardScaler` (+ winsorisation for handcrafted V2/V4).
  - Estimator: `LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)`.
  - Grid:
    - $C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$
    - Total candidate configurations: 5.

### 3.2 Gradient Boosted Trees
- **V3, V5 (HistGradientBoostingClassifier):**
  - V3 (Handcrafted 330d Raw), V5 (Handcrafted 330d Normalised).
  - Preprocessing: In-fold median imputation.
  - Estimator: `HistGradientBoostingClassifier(random_state=seed)`.
  - Grid:
    - `max_depth` $\in \{3, 5\}$
    - `learning_rate` $\in \{0.05, 0.1\}$
    - `min_samples_leaf` $\in \{5, 10\}$
    - Total candidate configurations: 8.

### 3.3 Neural Segment-Level Models (Fixed Epochs)
- **V12 (Handcrafted Segments 55d), V13 (WavLM L8 Segments 768d):**
  - Architecture: Small 2-layer MLP (`Linear(D, 64) -> LayerNorm -> ReLU -> Dropout(0.2) -> Linear(64, 1)`).
  - Training: Fixed **40 epochs**, Adam optimizer ($\text{lr} = 10^{-3}$), binary cross-entropy loss.
  - Early stopping: **STRICTLY PROHIBITED**.
  - Grouping: Segments are grouped strictly by participant. During CV, all segments belonging to training participants form the training set; validation participants are never seen during training.
  - Aggregation: For each validation participant, segment-level logits/probabilities are aggregated via **arithmetic mean** to yield exactly one participant-level prediction score $\hat{p}_i$.

### 3.4 Controls
- **V15 (Length-Only Control, 4 features):** Total audio duration (s), speech duration (s), $N$ VAD segments, mean segment duration (s). In-fold `StandardScaler`, `LogisticRegression` ($C \in \{0.01, 0.1, 1.0, 10.0\}$).
- **V16 (Gender-Only Control, 1 feature):** Binary indicator (`gender`). `LogisticRegression(C=1.0)`.
- **V17 (Domain-Only Control, 1 feature):** Binary indicator (`is_woz`). `LogisticRegression(C=1.0)`.

### 3.5 Multimodal Stacking Blenders
- **V18 (Confirmatory $H_3$):** Nested Stacking of V2 (Handcrafted 330d) + V14 (BERT 768d).
  - Within each outer training fold:
    1. Inner 4-fold CV generates out-of-fold predicted probabilities for V2 and V14.
    2. Meta-learner: `LogisticRegression(C=1.0)` fitted on $[\hat{p}_{\text{inner}}^{V2}, \hat{p}_{\text{inner}}^{V14}]$ against training fold labels.
    3. Retrain full V2 and V14 models on the full outer training fold.
    4. Predict outer validation fold with V2 and V14, and pass predictions through meta-learner to obtain outer validation probabilities.
- **V19 (Exploratory Fusion):** Nested Stacking of Inner-CV Selected Best Audio Variant + V14.
  - Within each outer training fold:
    1. Evaluate inner 4-fold CV ROC-AUC across all candidate audio variants (V1–V13).
    2. Select best audio variant $V^*$ (highest inner AUC; tie-breaker: fewer input dimensions).
    3. Generate inner-OOF predictions for $V^*$ and V14.
    4. Fit meta-learner: `LogisticRegression(C=1.0)` on $[\hat{p}_{\text{inner}}^{V^*}, \hat{p}_{\text{inner}}^{V14}]$.
    5. Retrain $V^*$ and V14 on full outer training fold and predict outer validation fold.

---

## 4. Pre-Stated Freeze Rule for Phase 6B Transition

Prior to observing any outcome values, the freeze rule for generating `frozen_config.json` is fixed as follows:

1. **Primary Audio Model Freeze:**
   - Evaluated across all audio-only variants: $\{V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13\}$.
   - Selection criterion: The audio variant with the **highest observed participant-level ROC-AUC** (averaged across seeds 0–4).
   - Deterministic tie-breaker: If two or more variants achieve identical AUC to 4 decimal places, select the variant with **fewer input dimensions**. If dimensions are equal, select the lower variant index.
   - Exploratory designation: If the selected primary audio variant is NOT V2 ($H_1$) or V10 ($H_2$), it must be formally designated as **`exploratory-derived`** in `frozen_config.json`.
2. **Fusion Model Freeze:**
   - Evaluated between the two multimodal stacking variants: $\{V18, V19\}$.
   - Selection criterion: The fusion variant with the **higher observed participant-level ROC-AUC**.
   - Deterministic tie-breaker: If identical, select V18 (confirmatory $H_3$).
   - Exploratory designation: If V19 is selected, mark as **`exploratory-derived`**.
3. **Strict Prohibition:**
   - Freezing these configurations authorizes **NO runs on confirmation seeds 100–104 or the sealed cohort of 33**.

---

## 5. Execution Hash & Audit Log

- **Tagged Commit Hash:** *(To be recorded upon tagging `prereg-exec-v1` following dry-run)*
- **Git Tag:** `prereg-exec-v1`
- **Execution Date:** 2026-10-15
- **Pre-Registration Status:** **LOCKED & IMMUTABLE PRIOR TO REAL LABEL EVALUATION.**

---
*Amendment v1.3 logged on branch `research-upgrade-2026-10-13` before real label evaluation.*
