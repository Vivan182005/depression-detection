# Pre-Registration Document (Version 1.1): Phase 6A Research Protocol
## Voice-Primary Depression Detection (Audio Primary, Text Secondary)

**Protocol Version:** 1.1 (Supersedes v1.0)  
**Registration Date:** 2026-10-14  
**Repository Branch:** `research-upgrade-2026-10-13`  
**Working Directory:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Execution Gate:** PRE-REGISTRATION LOCKED PRIOR TO ANY MODEL RUN  
**Current Run Status:** ZERO MODEL RUNS STARTED IN THIS TASK  

---

## 1. Study Overview & Primary Scientific Goal

The objective of Phase 6A is to definitively evaluate whether voice representations (hand-crafted prosody, spectral features, and self-supervised speech representations) contain genuine, generalizable clinical signal for detecting clinically significant depression (PHQ-8 $\ge$ 10), and whether multimodal audio-text fusion provides measurable incremental value over semantic text representations alone.

### 1.1 Factual Corrections from Historical Reports
1. **Text Variant T2:** Feature representation `BERT-T2` uses transcripts generated locally by **`openai/whisper-tiny`** (normalized with preserved apostrophes), matching the automated pipeline.
2. **Sealed Cohort Provenance:** The 33 sealed participants comprise exactly **16 test + 12 train + 5 dev** sessions across official challenge partitions (**9 DAIC-WOZ + 24 E-DAIC Avatar**). They remain strictly sequestered, unextracted, and label-blind.
3. **Turn-Alignment on Spliced Audio:** DAIC-WOZ audio files in `Clean_audio/NEW_AUDIO/` fail cumulative-offset alignment (98.3% failure rate). Raw transcript slicing is strictly prohibited. Sessions failing alignment are excluded from turn-level features only.

---

## 2. Cohorts, Partitions & Data Boundary Gates

### 2.1 Active Development Cohort ($N = 239$)
All feature analysis, cross-validation, hyperparameter tuning, stacking, and model selection in Phase 6A are restricted exclusively to the 239 active development sessions:
- **DAIC-WOZ Development:** $N = 178$ sessions (PID 306 excluded due to missing source audio).
- **Avatar Development (E-DAIC Dev):** $N = 61$ sessions.
- **Demographics:** Female $N = 84$ (35.1%), Male $N = 155$ (64.9%).
- **Clinical Labels:** None/Mild (PHQ 0–9) = 165 (69.0%), Moderate/Severe (PHQ $\ge$ 10) = 74 (31.0%).

### 2.2 Boundary Integrity Guards
1. **Sealed Cohort ($N = 33$):** Strictly unread, unsegmented, unextracted, and label-blind. No model selection or decision may reference the sealed 33.
2. **Confirmation Seeds (100–104):** Strictly locked. Seeds 100–104 re-split the **SAME 239 development participants** to test numerical stability across fold assignments; they do NOT represent independent generalisation. Results on seeds 100–104 must **NEVER be called 'confirmed'**.
3. **Freezing Rule:** At most **ONE audio configuration** and **ONE fusion configuration** may be frozen in `frozen_config.json` before touching seeds 100–104. Any frozen config that is not part of the confirmatory Holm family must be explicitly labeled as **exploratory-derived**.

---

## 3. Prespecified Endpoints & Decision Rules

### 3.1 Primary Endpoint
- **Participant-Level Binary AUC (ROC-AUC):**
  - Binary classification target: Clinically Significant Depression (PHQ-8 $\ge$ 10) vs. Non-Depressed (PHQ-8 $<$ 10).
  - Evaluated on all $N = 239$ development participants.
  - Calculated strictly from out-of-fold (OOF) predicted probabilities averaged across selection seeds $0, 1, 2, 3, 4$ (5 seeds $\times$ 5 folds = 25 runs per model).

### 3.2 Secondary Endpoints (Descriptive Only)
Secondary metrics provide qualitative diagnostic insight but carry **NO pass/fail gating authority**:
1. **3-Class Macro-Averaged F1 Score ($\text{Macro-F1}$):** Across None/Mild (0–9), Moderate (10–14), and Severe (15–24).
2. **Severe Recall:** Sensitivity on the Severe depression cohort (PHQ-8 $\ge$ 15).
3. **Subgroup AUCs with 95% Bootstrap Confidence Intervals:**
   - DAIC-WOZ Subgroup AUC ($N = 178$).
   - Avatar Subgroup AUC ($N = 61$).
   - Male Subgroup AUC ($N = 155$).
   - Female Subgroup AUC ($N = 84$).
   - *Equity Note:* Subgroup AUCs are monitored descriptively; no demographic equity pass/fail gate is enforced.

---

## 4. Complete 19-Variant Model Matrix

The Phase 6A evaluation matrix comprises exactly 19 prespecified variants. Each layer, pooling strategy, feature family, normalisation setting, and classification head constitutes a unique variant:

| Variant ID | Role / Family | Feature Representation & Preprocessing | Classifier Architecture |
| :--- | :--- | :--- | :--- |
| **V1** | Exploratory (Audio) | Whole-recording speech-only eGeMAPS functionals (135d robust, uncalibrated loudness/silence dropped) | Elastic-Net Logistic Regression |
| **V2** | **Confirmatory ($H_1$)** | Turn-aggregated hand-crafted voice features (MFCC+deltas, jitter [local/rap/ppq5], shimmer [local/apq3/apq5/apq11], HNR, two-pass f0, formants F1-F3, spectral slope, in-turn pauses, session-relative intensity) [Raw] | $L_2$ Logistic Regression |
| **V3** | Exploratory (Audio) | Same turn-aggregated hand-crafted voice features as V2 [Raw] | HistGradientBoosting (GBDT) |
| **V4** | Exploratory (Audio) | Same hand-crafted voice features as V2 + Cepstral Mean & Variance Normalisation (CMVN) + Long-Term-Spectrum Equalisation | $L_2$ Logistic Regression |
| **V5** | Exploratory (Audio) | Normalised hand-crafted voice features (as in V4) | HistGradientBoosting (GBDT) |
| **V6** | Exploratory (SSL) | `facebook/wav2vec2-base-960h` Layer 4 representations (mean-pooled over participant speech) | $L_2$ Logistic Regression |
| **V7** | Exploratory (SSL) | `facebook/wav2vec2-base-960h` Layer 8 representations (mean-pooled over participant speech) | $L_2$ Logistic Regression |
| **V8** | Exploratory (SSL) | `facebook/wav2vec2-base-960h` Layer 12 representations (mean-pooled over participant speech) | $L_2$ Logistic Regression |
| **V9** | Exploratory (SSL) | `microsoft/wavlm-base-plus` Layer 4 representations (mean-pooled over participant speech) | $L_2$ Logistic Regression |
| **V10** | **Confirmatory ($H_2$)** | `microsoft/wavlm-base-plus` Layer 8 representations (mean-pooled over participant speech) | $L_2$ Logistic Regression |
| **V11** | Exploratory (SSL) | `microsoft/wavlm-base-plus` Layer 12 representations (mean-pooled over participant speech) | $L_2$ Logistic Regression |
| **V12** | Exploratory (Neural) | Turn-level hand-crafted features ($\ge 0.3$s voiced turns) | Small MLP (2 layers, LayerNorm, Dropout, fixed epochs, no early stopping) |
| **V13** | Exploratory (Neural) | Turn-level WavLM Layer 8 representations | Small MLP (2 layers, LayerNorm, Dropout, fixed epochs, no early stopping) |
| **V14** | Benchmark (Text) | `bert-base-uncased` representations from Whisper-tiny transcripts (`BERT-T2`) | $L_2$ Logistic Regression |
| **V15** | Control (Metadata) | Length-only metadata control (total audio duration, total speech duration, turn count, mean turn duration, speaking rate) | $L_2$ Logistic Regression |
| **V16** | Control (Metadata) | Gender-only demographic control | $L_2$ Logistic Regression |
| **V17** | Control (Metadata) | Recording domain-only control (WoZ vs. Avatar indicator) | $L_2$ Logistic Regression |
| **V18** | **Confirmatory ($H_3$)** | Nested Stacking: V2 (Hand-Crafted Voice) + V14 (BERT-T2 Text) | Meta-Logistic Regression (in-fold stacking weights) |
| **V19** | Exploratory (Fusion) | Nested Stacking: Best Audio Variant (selected via inner grouped CV) + V14 (BERT-T2 Text) | Meta-Logistic Regression (in-fold stacking weights) |

*Definition of Length Metadata (V15):* Strictly restricted to physical interview length dynamics: total recording duration (s), total participant speech duration (s), number of detected speech turns, mean speech turn duration (s), and speech-to-pause ratio. Zero demographic features (age, gender) are included in V15.

---

## 5. Validation Protocol & Elimination of Leakage

### 5.1 Partitioning Structure
- **5-Fold Stratified Group K-Fold Cross-Validation:**
  - Grouped by `participant_id` (zero participant overlap across folds).
  - Stratified by binary label (PHQ-8 $\ge$ 10).
- **Selection Seeds:** Evaluated across 5 seeds: `0, 1, 2, 3, 4`. Out-of-fold predicted probabilities are averaged across the 5 seeds to form the final participant decision vector.
- **Fixed Epoch Training (No Validation Early Stopping):** Neural models (V12, V13) train for a strictly fixed budget of 40 epochs. Early stopping against validation fold loss/AUC is **strictly prohibited**.
- **Nested Grouped Inner CV:** All hyperparameter tuning (regularization $C$, GBDT tree depth, learning rate), stacking weights, and fusion blending parameters must be learned using an **inner 4-fold grouped cross-validation** within each training fold. Validation folds are evaluated purely out-of-sample.

### 5.2 Strict In-Fold Preprocessing
All data transformations are fitted strictly on training fold indices:
1. `StandardScaler` fitted on training fold only.
2. Outlier winsorisation limits ($P_{01}^{\text{train}}, P_{99}^{\text{train}}$) computed on training fold only.
3. Feature medians for missing-value imputation computed on training fold only.
4. Binary missing-indicator covariates appended statelessly.

---

## 6. Null Construction & Statistical Testing Framework

### 6.1 Stratified Permutation Null Hypothesis ($H_0$)
To establish an empirical null distribution preserving demographic and domain marginals:
- **Permutation Construction:** Each participant's complete clinical label record (PHQ-8 total score, binary label, 3-class severity, item responses) is permuted across participants **WITHIN domain $\times$ gender strata** (WoZ Female, WoZ Male, Avatar Female, Avatar Male). Permutation is **never performed within class**.
- **Full Pipeline Re-Execution:** For each permutation replication $r \in \{1, \dots, R\}$:
  1. Generate 5-fold stratified splits from the permuted labels.
  2. Re-fit all scalers, winsorisation bounds, and imputation medians inside each training fold.
  3. Re-tune all hyperparameters via inner grouped CV.
  4. Generate seed-averaged (seeds 0–4) out-of-fold AUC on the permuted labels.
- **Empirical $p$-Value Calculation:**
  $$p = \frac{1 + \sum_{r=1}^{R} \mathbb{I}(\text{AUC}_{\text{null}}^{(r)} \ge \text{AUC}_{\text{observed}})}{R + 1}$$
- **Documented Caveat of Stratified Null:** Because labels are permuted within domain $\times$ gender strata, any feature that acts as a proxy for gender or recording domain preserves its alignment with the stratum-level base rate of depression. Consequently, **gender-proxy or domain-correlated features can systematically score above 0.5 under this null**.

### 6.2 Confirmatory Family & Multiplicity Control ($m = 3$)
The formal confirmatory testing family consists of exactly 3 hypotheses, controlled at family-wise error rate $\alpha = 0.05$ via the **Holm-Bonferroni step-down procedure**:

1. **$H_1$ (Hand-Crafted Voice vs. Null):**
   $$\text{AUC}(V2) > \text{AUC}_{\text{null}} \quad (R = 999)$$
2. **$H_2$ (Self-Supervised Speech vs. Null):**
   $$\text{AUC}(V10) > \text{AUC}_{\text{null}} \quad (R = 999)$$
3. **$H_3$ (Multimodal Incremental Synergy):**
   $$V18 \text{ beats } V14 \text{ AND } V18 \text{ beats } V2 \text{ in } \Delta\text{-AUC}$$
   - Tested via an **Intersection-Union Test (IUT)** using paired participant-cluster bootstrap ($B = 10,000$ resamples clustered by participant):
     $$p = \max \left( p_{V18 > V14}, \, p_{V18 > V2} \right)$$
   - Both one-sided bootstrap tests must simultaneously reach significance for $H_3$ to be accepted.

**Computational Feasibility Gate:**
Before launching $R = 999$ permutations, execution runtime will be clocked on $R = 5$ permutations. If $R = 999$ requires $> 48$ hours of computation, $H_1$ and $H_2$ will be **reclassified as exploratory** rather than reducing $R$.

### 6.3 Exploratory Family (13 Variants)
Variants V1, V3–V9, V11–V14, and V19 are designated as exploratory:
- Evaluated via participant-cluster bootstrap 95% confidence intervals on binary AUC.
- No formal significance claims or Holm multiplicity adjustments are made.
- An optional null-spread diagnostic ($R = 99$) may be computed for descriptive baseline calibration.
- **Mandatory Reporting:** **ALL 19 variants must be reported in full**, including negative and null findings.
- **Leakage & Domain Audit Trigger:** Any variant achieving an unusually high AUC ($> 0.75$ for audio-only or $> 0.85$ for multimodal) triggers an immediate investigation for leakage and domain confounding, rather than automatic acceptance.

---

## 7. Quality & Validity Rules for Acoustic Features

Following Quality Audit v2, turn-level perturbation features must adhere to strict physiological validity constraints:
1. **Perturbation Validity Threshold:** Jitter and shimmer are calculated only on turns with **$\ge 0.30\text{ s}$ voiced speech AND $\ge 30$ pitch pulses**. Turns failing this threshold are assigned `NaN` for perturbation metrics.
2. **Two-Pass Pitch Estimation:**
   - Pass 1: Session-level pitch across full audio ($60–500\text{ Hz}$) to extract median $\tilde{f}_0$.
   - Pass 2: Pitch re-estimated within $[0.5 \tilde{f}_0, 2.0 \tilde{f}_0]$, clamped to $[50, 450\text{ Hz}]$. No labels or gender metadata used.
   - Octave-jump rejection: reject frames with $|\Delta \log_2(f_0)| > 0.8$ octaves within $\le 30\text{ ms}$.
3. **Plausibility Bounds:**
   - Jitter local $\le 0.12$; Shimmer local $\le 0.35$ (amplitude ratio); HNR $\in [0, 30\text{ dB}]$.
   - Formants strictly ordered: $200 \le F_1 \le 1200$, $600 \le F_2 \le 3000$, $1500 \le F_3 \le 4500\text{ Hz}$, with $F_2 - F_1 \ge 150\text{ Hz}$ and $F_3 - F_2 \ge 200\text{ Hz}$.
4. **Session Minimum Data Gate:** Any participant with fewer than 5 valid perturbation turns or $< 2.0\text{ s}$ total valid speech is flagged with an `insufficient_acoustic_data` indicator.

---

## 8. Pre-Registration Verification & Execution Status

- **Status:** **FROZEN & COMMITTED BEFORE ANY MODEL RUN.**
- **Model Training Execution:** **ZERO model runs have been initiated.**
- **Sealed Cohort Isolation:** **The 33 sealed participants remain unread, unsegmented, unextracted, and label-blind.**
- **Confirmation Folds:** **Seeds 100–104 remain strictly untouched.**

---
*Pre-registration v1.1 formally locked on branch `research-upgrade-2026-10-13` before Phase 6A execution.*
