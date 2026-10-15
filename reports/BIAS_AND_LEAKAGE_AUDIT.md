# Comprehensive Bias & Leakage Audit Report

**Project**: Explainable Multi-Modal Speech & Language Depression Detection System  
**Working Repository**: `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Date**: October 13, 2026  
**Auditor**: Senior Machine Learning & Clinical Integrity Specialist  
**Status**: Completed — Phase 3 Integrity Verification  

---

## Executive Summary

Before implementing any research experiments or new training runs, an empirical audit was conducted to identify potential data leakage vectors, demographic biases, lineage discrepancies, and label inconsistencies across the DAIC-WOZ and E-DAIC 2019 releases.

Key findings include:
1. **Zero Contamination of Unseen Holdout**: There is **0% participant overlap** between the 183 development sessions (`app/feature_store/master_dataset_index.csv`) and the 33 reserved holdout sessions (`unseen/`).
2. **Critical Discovery in Development Index**: Two participants (`301` and `302`) possess duplicate session entries (`301_AUDIO`, `301`, `loud_audio301`, and `302_AUDIO`, `302`, `loud_audio302`), resulting in 183 rows across 179 unique participants. Any random row-based splitting would cause **participant leakage**. All validation splits **must be grouped strictly by participant ID**.
3. **Severe Cross-Domain Disparity**: 100% of the training feature store is composed of human Wizard-of-Oz interviews (PIDs 301–492), whereas 72.7% (24/33) of the unseen evaluation cohort consists of 600-series autonomous virtual avatar interviews.
4. **Demographic Representation**: Across the 275 cataloged participants in `detailed_lables.csv`, gender distribution is 61.5% male (169), 38.2% female (105), and 0.4% unknown (1). Mean age is .2 \pm 12.7$ years (range: 18 to 69).

---

## 1. Lineage & Participant Overlap Audit

| Corpus Partition | Total Rows / Files | Unique Participants | Participant ID Range | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Development Feature Store** (`master_dataset_index.csv`) | 183 | **179** | PIDs 301 – 492 | DAIC-WOZ human Wizard-of-Oz interviews. Contains duplicate sessions for PIDs 301 & 302. |
| **Authoritative Extended Labels** (`detailed_lables.csv`) | 275 | **275** | PIDs 300 – 718 | Full E-DAIC 2019 metadata catalog including question-level PHQ-8 items, gender, and age. |
| **Reserved Unseen Cohort** (`unseen/`) | 33 | **33** | 9 WoZ (300-414), 24 AI (600-662) | Sealed out-of-sample evaluation benchmark. |

### Overlap Analysis:
4214\text{Development PIDs} \cap \text{Unseen PIDs} = \emptyset \quad (0 \text{ participants})4214
- **Verification**: Complete set disjointness confirmed. The unseen cohort has never been touched by gradient descent, feature scalers, or model selection.

---

## 2. Repeated Sessions & Participant-Level Grouping Requirement

In `app/feature_store/master_dataset_index.csv`:
- Participant **301** appears 3 times:
  - Row 0: `session_id = "301_AUDIO"`
  - Row 1: `session_id = "301"`
  - Row 180: `session_id = "loud_audio301"`
- Participant **302** appears 3 times:
  - Row 2: `session_id = "302_AUDIO"`
  - Row 3: `session_id = "302"`
  - Row 181: `session_id = "loud_audio302"`

### Leakage Risk:
If cross-validation or train/validation splitting is conducted using standard row indices (`KFold(shuffle=True)`), repeated sessions of participant 301 or 302 can end up in both training and validation folds simultaneously.
### Enforcement:
All experimental splits MUST use **`GroupKFold`** or **`StratifiedGroupKFold`** keyed strictly on `participant_id`, guaranteeing that all sessions of any participant reside exclusively within a single fold.

---

## 3. Label Consistency & Stratification Audit

### PHQ-8 Severity Thresholds:
- **Class 0 (Minimal/Mild)**: $\text{PHQ-8} \in [0, 9]$
- **Class 1 (Moderate)**: $\text{PHQ-8} \in [10, 14]$
- **Class 2 (Severe)**: $\text{PHQ-8} \in [15, 24]$
- **Binary Screening Threshold**: $\text{PHQ-8} \ge 10$

### Distribution in Development Cohort (183 sessions):
- Class 0 (Minimal/Mild): $ sessions (.8\%$)
- Class 1 (Moderate): $ sessions (.5\%$)
- Class 2 (Severe): $ sessions (.7\%$)
- Binary Positive ($\text{PHQ} \ge 10$): $ sessions (.2\%$)
- Binary Negative ($\text{PHQ} < 10$): $ sessions (.8\%$)

### Distribution in Reserved Unseen Cohort (33 sessions):
- Class 0 (Minimal/Mild): $ sessions (.7\%$)
- Class 1 (Moderate): $ sessions (.1\%$)
- Class 2 (Severe): $ sessions (.2\%$)
- Binary Positive ($\text{PHQ} \ge 10$): $ sessions (.3\%$)
- Binary Negative ($\text{PHQ} < 10$): $ sessions (.7\%$)

### Audit of Missing / Corrupted Labels:
- Zero NaN values in `phq8_score` or `severity_class_id` across development and unseen datasets.

---

## 4. Transcript Formatting & Shortcut Leakage Audit

### Interviewer Turn Filtering:
- Prior literature (Burdisso et al., Danylenko & Unold 2026) demonstrated that when interviewer turns are included, models exploit diagnostic questions as predictive shortcuts.
- **Audit Result**: `BertExtractor.extract_participant_text()` strictly ignores non-participant dialogue turns. In tab-separated DAIC-WOZ transcripts, it filters for `Speaker == "Participant"`. In E-DAIC transcripts, participant responses are isolated.

### Zero-Embedding Defect (Audited & Cleared):
- Previous runs experienced silent parser failures on comma-delimited E-DAIC transcripts, causing 24 unseen sessions to produce zero BERT embeddings (zsh.0$ norm).
- In the active verified cache (`artifacts/final_model/unseen_feature_cache_repaired.npz`), all 33 sessions contain non-zero BERT embeddings (norms: .83$ to .44$).
- In the 183 development sessions, all 183 contain valid non-zero BERT embeddings.

---

## 5. Pipeline Preprocessing & Scaler Leakage Audit

- **Standardization**: Feature standardizers (`scaler_ac`, `scaler_w2v`, `scaler_bert`) must be fitted **strictly inside the training fold**.
- **Audit of Candidate 2A Baseline**:
  - `feature_scalers.joblib` was fitted strictly on the 107 training sessions of the official DAIC-WOZ split.
  - Zero validation, test, or unseen samples were involved in computing mean or variance statistics.
- **Temperature Calibration**:
  - Calibrated strictly on validation fold logits using L-BFGS with temperature bounded to  \in [0.1, 5.0]$.

---

## 6. Demographic Representation & Generalizability Audit

| Demographic Variable | Available in Dataset? | Breakdown | Representation Assessment |
| :--- | :---: | :--- | :--- |
| **Gender** | Yes (`detailed_lables.csv`) | Male: 169 (61.5%), Female: 105 (38.2%), Unknown: 1 (0.4%) | Modest male skew; both genders have adequate representation for subgroup evaluation. |
| **Age** | Yes (`detailed_lables.csv`) | Mean: 41.2 years, Range: 18 to 69 years | Broad adult representation spanning early adulthood to seniors. |
| **Ethnicity / Race** | **No (Unmeasured)** | Not collected or reported in DAIC metadata | **Unverified limitation**. Cannot claim racial fairness without ground-truth labels. |
| **Language / Accent** | Partially | All interviews conducted in English | Dialect and accent diversity unquantified. |
| **Interview Condition** | Yes | Human Wizard-of-Oz vs. Autonomous Avatar | Severe domain shift: training cohort is 100% WoZ; unseen cohort is 72.7% autonomous avatar. |

---

## 7. Audit Conclusions & Experimental Guardrails

1. **Mandatory Participant Grouping**: All research cross-validation MUST use `StratifiedGroupKFold(n_splits=5)` grouped on `participant_id`.
2. **Strict Training-Only Transformations**: Scalers, feature selectors, class weighting, and any data augmentation must be computed exclusively within each training fold.
3. **Sealed Unseen Cohort**: The 33 unseen participants will remain sealed until the final candidate model is fully trained and selected via cross-validation.
