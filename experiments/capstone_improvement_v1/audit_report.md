# 🔬 Comprehensive System & Pipeline Audit Report (Phase 1)

**Date**: 2026-10-12  
**Environment**: macOS (Apple Silicon / Darwin), Python 3.14.6 / PyTorch 2.0+  
**Repository**: `https://github.com/Vivan182005/depression-detection`  
**Git Branch**: `main` | **Head Commit**: `9fd65d8` (Clean working tree)  

---

## 1. Executive Summary & Audit Scope

This audit provides a complete, forensic examination of the local codebase, dataset structures, feature caches, training logic, evaluation protocols, and saved checkpoints for the **Explainable Multi-Modal Speech & Language Depression Detection System**.

All historical claims were checked against actual local files, binary weights, and runtime logs.

---

## 2. Verified Repository Architecture & Datasets

### 2.1 File & Directory Inventory
* **Core Codebase**: Located in [`app/`](../../app), with submodules for preprocessing ([`app/src/preprocessing`](../../app/src/preprocessing)), feature extraction ([`app/src/features`](../../app/src/features)), model architectures ([`app/src/models`](../../app/src/models)), and diagnostics.
* **Ground-Truth Metadata**: Located in [`new_audio_csv/`](../../new_audio_csv) containing:
  - `detailed_lables.csv` (275 participants: Age, Gender, 8 PHQ-8 itemized ratings, PTSD symptoms, split).
  - `train_split.csv` (163 participants), `dev_split.csv` (56 participants), `test_split.csv` (56 participants: AI 600-series).
  - `Detailed_PHQ8_Labels.csv` (220 participants).
* **Audio Collections**:
  - `Clean_audio/NEW_AUDIO/`: 187 participant audio files (`XXX_AUDIO_interviewee.wav`), WoZ cohort.
  - `NEW_AUDIO_DAIC/`: 61 participant audio files (remaining after unseen isolation), autonomous AI 600-series cohort.
* **Reserved Unseen Cohort** ([`unseen/`](../../unseen)):
  - Structurally isolated holdout set consisting of **33 sessions** (9 original WoZ sessions + 24 autonomous AI sessions from the 600 series).
  - Categorized into subdirectories: `minimal` (13), `mild` (9), `moderate` (4), `moderately_severe` (5), `severe` (2), with corresponding transcripts.

---

## 3. Discrepancy & Root Cause Analysis

### 3.1 Historical PHQ-8 MAE Ambiguities
* **Historical Observation**: Across different reports, PHQ-8 MAE was inconsistently reported as **3.51**, **4.23**, and **3.07/3.69**.
* **Audit Finding**:
  1. **`4.23 pts`** (exact: `4.22995`): The held-out test split regression MAE stored inside the active PyTorch checkpoint payload ([`app/feature_store/best_depression_model.pth`](../../app/feature_store/best_depression_model.pth)).
  2. **`3.51 pts`** (`3.51 ± 0.14`): The 5-fold cross-validation out-of-sample mean MAE produced by [`run_research_validation.py`](../../app/run_research_validation.py).
  3. **`3.69 pts`**: The classical model (Model A: PCA + Gradient Boosted Trees) test MAE saved in [`capstone_execution_deliverable_report.md`](../../capstone_execution_deliverable_report.md).
* **Resolution**: These values reflect different models and evaluation splits, not a single contradictory metric.

### 3.2 Unseen Cohort Count Discrepancy (9 vs. 33)
* **Historical Observation**: Early documentation references 9 unseen sessions, whereas recent updates reference 33.
* **Audit Finding**:
  - Originally, 9 sessions (`300, 303, 304, 321, 338, 397, 408, 409, 414`) from the human Wizard-of-Oz cohort were isolated in `unseen/`.
  - Upon downloading the E-DAIC 2019 dataset, 24 additional sessions from the 600-series autonomous AI interviewer were carved out and placed into `unseen/` to ensure the final sanity check rigorously tests both human and AI interviewer domains. Total = 33 sessions.

### 3.3 Critical Label Corruption in Legacy `OFFICIAL_PHQ8_LABELS`
* **Audit Finding**:
  - In [`app/label_mapper.py`](../../app/label_mapper.py), the hardcoded dictionary `OFFICIAL_PHQ8_LABELS` had **66 participant scores that differed from the official clinical records** in `new_audio_csv/detailed_lables.csv` and `Detailed_PHQ8_Labels.csv`!
  - Example 1: Participant `300` was hardcoded as **12.0** ("Moderate Depression"), but the true clinical survey score is **2.0** (0+0+1+0+1+0+0+0 = 2).
  - Example 2: Participant `308` was hardcoded as **11.0**, but the true score is **22.0** (Severe Depression).
  - Example 3: Participant `311` was hardcoded as **1.0**, but the true score is **21.0** (Severe Depression).
* **Impact**: Models trained on this legacy dictionary learned corrupted class boundaries for a significant portion of minority cases.
* **Action Required**: The training pipeline must ingest labels dynamically from `new_audio_csv/detailed_lables.csv`.

### 3.4 Duplicate Participant Records in `master_dataset_index.csv`
* **Audit Finding**:
  - `master_dataset_index.csv` lists 183 sessions but only 179 unique participants.
  - Participant `301` was indexed 3 times (`301`, `301_AUDIO`, `loud_audio301`).
  - Participant `302` was indexed 3 times (`302`, `302_AUDIO`, `loud_audio302`).
* **Impact**: If split randomly without group constraints, data leakage occurs. Even under `StratifiedGroupKFold`, duplicate records artificially weight participant 301 and 302 by 3x compared to all other participants.
* **Action Required**: Deduplicate sessions so each unique participant has exactly one canonical session.

---

## 4. Feature Extraction & Leakage Audit

1. **Acoustic Features (154d)**:
   - eGeMAPS (88d) + Temporal frame statistics (33d mean + 33d std = 66d).
   - Computed from interviewee-only audio. Checked for NaNs/Infs: none found.
2. **Speech Foundation Model (768d)**:
   - Extracted using `facebook/wav2vec2-base-960h` with frozen weights. Mean-pooled across time frames.
3. **Clinical Language Model (768d)**:
   - Extracted using `bert-base-uncased` with frozen weights.
   - Text parsing in [`app/src/features/bert_extractor.py`](../../app/src/features/bert_extractor.py) strictly filters for `speaker.lower() == 'participant'`, verifying **zero interviewer verbal content leakage**.
4. **Data Normalization**:
   - `StandardScaler` in [`train_leak_free.py`](../../app/src/models/train_leak_free.py) is fitted strictly on `train_dataset` and transformed on `val_dataset` and `test_dataset`.

---

## 5. Checkpoint Inventory & Provenance

| Checkpoint Path | Architecture | Input Dims | Parameters | Optimal Temp $T^*$ | Saved Test Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [`app/feature_store/best_depression_model.pth`](../../app/feature_store/best_depression_model.pth) | `CompactTriModalMLP` | 1,690 | 225,413 | 1.2439 | **83.78%** |
| [`app/feature_store/best_trimodal_model_a.joblib`](../../app/feature_store/best_trimodal_model_a.joblib) | PCA (25) + GBDT | 1,592 | N/A | N/A | **81.48%** |
| [`app/feature_store/fitted_acoustic_scaler.joblib`](../../app/feature_store/fitted_acoustic_scaler.joblib) | `StandardScaler` | 1,690 | N/A | N/A | N/A |

---

## 6. Audit Verdict & Action Plan

1. **Baseline Preservation**: The existing checkpoint `best_depression_model.pth` is verified intact and serves as our immutable baseline.
2. **Label Integrity**: Switch ground-truth mapping to `new_audio_csv/detailed_lables.csv`, eliminating the 66 scrambled label discrepancies.
3. **Dataset Deduplication**: Eliminate the duplicate 301/302 session artifacts to enforce true 1-participant-to-1-sample group isolation.
4. **Independent Experiment Directory**: All new models, splits, metrics, and logs will be saved under `experiments/capstone_improvement_v1/`.
