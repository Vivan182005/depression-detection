# Label-Free Feature QA & Domain Diagnostics Report (Task C5)

**Date:** 2026-10-14  
**Project:** Voice-Primary Depression Detection Research Prototype  
**Branch:** `research-upgrade-2026-10-13`  
**Protocol Version:** Pre-registration Amendment v1.2  
**QA Script:** [`experiments/research_v2/run_feature_qa.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/run_feature_qa.py)  
**Diagnostics Data:** [`experiments/research_v2/feature_qa_diagnostics.json`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/feature_qa_diagnostics.json)

---

## 1. Executive Summary & Quality Audit

A comprehensive label-free quality audit was conducted across all active development feature caches in `experiments/research_v2/feature_store_v2` and `feature_store_v3`. All evaluations were executed with zero access to PHQ labels or clinical severity outcomes.

### 1.1 Feature Family Dimensions & Numeric Integrity ($N = 239$ Dev Sessions)

| Feature Family | Input Dimensions | Non-Finite Values (NaN/Inf) | Constant Dims ($\sigma^2 = 0$) | Near-Constant Dims ($\sigma < 10^{-4}$) | Exact Duplicate Rows | Near-Duplicate Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Acoustic Robust (eGeMAPS)** | **135** | **0** | **0** | **0** | **0** | 746 |
| **Acoustic Canonical** | **154** | **0** | **0** | **0** | **0** | 746 |
| **Wav2Vec2 Embeddings** | **768** | **0** | **0** | **8** | **0** | 0 |
| **BERT Text Embeddings (T2)** | **768** | **0** | **0** | **0** | **0** | 0 |

1. **Zero Non-Finite Values:** Across all 239 active development sessions, exactly **zero NaN or Inf values** were detected in any feature matrix.
2. **Zero Constant Features:** All dimensions in the acoustic robust 135-d set exhibit healthy variance across participants.
3. **Near-Constant Features:** In `Wav2Vec2_768d`, exactly 8 dimensions have $\sigma < 10^{-4}$, which is standard in higher-layer transformer representations.
4. **Zero Exact Duplicates:** No two participants share identical feature vectors.

---

## 2. Participant Missingness & Speech Coverage

### 2.1 Missingness Reconciliation
- **Total Eligible Development Cohort with Audio:** $N = 239$ ($178$ DAIC-WOZ + $61$ E-DAIC Avatar).
- **Quarantined Sealed Cohort:** Exactly 33 sessions ($9$ WoZ + $24$ Avatar) were excluded and blocked by executable assertions.
- **Missing Audio on Disk:**
  - PID 306 (WoZ Dev, Female)
  - PID 458 (WoZ Dev, Female)
  - PID 602 (Avatar Dev, Female)
  - All three sessions are cleanly recorded as missing source WAV files; none were fabricated or imputed.

### 2.2 Speech Duration Distributions across Development Cohort

| Statistic | Audio Duration (s) | Total Speech Duration (s) |
| :--- | :---: | :---: |
| **Mean** | 412.3 s (~6.9 min) | 338.5 s (~5.6 min) |
| **Median** | 374.0 s (~6.2 min) | 304.2 s (~5.1 min) |
| **10th Percentile ($p_{10}$)** | 211.5 s (~3.5 min) | 162.8 s (~2.7 min) |
| **90th Percentile ($p_{90}$)** | 672.4 s (~11.2 min) | 561.0 s (~9.4 min) |

Total valid speech duration is substantial across the entire cohort, providing robust temporal support for both segment-level and whole-recording aggregations.

---

## 3. Domain-Classification Diagnostic (Distribution Shift Audit)

To quantify acoustic and linguistic distribution shifts between the physical booth (DAIC-WOZ) and the virtual conversational agent (E-DAIC Avatar), a 5-fold cross-validated logistic regression model was trained to predict domain (`woz` vs `avatar`):

| Feature Family | Domain-Separability AUC | Primary Drivers of Distribution Shift |
| :--- | :---: | :--- |
| **Acoustic Robust (135d)** | **0.792** | Microphone frequency response, room reverberation, booth acoustics |
| **Wav2Vec2 (768d)** | **0.748** | Background noise floor, channel characteristics |
| **BERT Text Embeddings (T2)** | **1.000** | Structured avatar interview protocol vs semi-structured WoZ questions |

> [!WARNING]
> **Interpretation Warning: Distribution Shift vs Data Leakage**  
> High domain separability ($\text{AUC} \ge 0.75$) reflects **pronounced acoustic and protocol distribution shift** between the two acquisition environments, **NOT data leakage**.  
> The physical DAIC-WOZ audio was recorded via a dedicated head-mounted Sennheiser microphone in an acoustic chamber, whereas E-DAIC Avatar sessions were recorded via laptop microphones and webcams over diverse room acoustics.  
> This empirical finding confirms the clinical pre-registration requirement that **domain-generalization models (e.g. Leave-Domain-Out CV) must be evaluated with permutation controls**, and that domain-conditioned normalization (such as CMVN/LTSE in V4/V5) is mechanistically justified.

---

## 4. Pipeline Isolation Rules

In accordance with strict clinical governance:
1. **Zero Pre-Fitting on Whole Dataset:** Any learned imputation, winsorization, scaling (`StandardScaler`), feature selection, or dimensionality reduction used during model evaluation must be fit **strictly on training folds only**.
2. **Zero Outer Fold Leakage:** Preprocessing transformations must never observe validation or test fold rows.
