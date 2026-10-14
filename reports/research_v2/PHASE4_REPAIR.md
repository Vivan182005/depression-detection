# Phase 4 Forensic Audit & Canonical Pipeline Repair Report

**Project:** 3-Class PHQ-8 Clinical Depression Severity Research Prototype (Audio + Text Only)  
**Repository Branch:** `research-upgrade-2026-10-13`  
**Working Directory:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Date:** October 14, 2026  
**Status:** Audit Verified & Feature Store v2 Implemented | Baseline Retraining Pending User Review

---

## 1. Executive Summary

This forensic audit and pipeline repair phase was initiated to rigorously investigate five suspected structural defects in the historical DAIC-WOZ / E-DAIC depression severity estimation codebase. Every defect has been **empirically confirmed on disk** using direct array inspections, statistical tests, and source code tracing.

### Core Findings Summary
1. **Wav2Vec2 Stored Vectors were Scaled Gaussian Noise:** The legacy extractor (`app/src/features/wav2vec_extractor.py:46-52`) silently caught missing PyTorch/HuggingFace dependencies and returned seeded Gaussian noise `randn(768) * (energy + 0.1)`. The historical WoZ feature store (`app/feature_store/`) contains pure random noise (kurtosis $\approx +0.18$, cross-session cosine similarity with real embeddings $\approx 0.00$).
2. **eGeMAPS 88-d Never Ran OpenSMILE:** The legacy extractor (`app/src/features/egemaps_extractor.py:108-115`) fell back to a 62-dimensional Librosa approximation and padded 26 exact zeros to reach 88 dimensions. Dimensions 62–87 are identically $0.0000$ across all historical WoZ features, Avatar development features, sealed cache features, and the saved scaler `scaler_ac`.
3. **Severe Clinical Label & Gender Discrepancies:** The hardcoded lookup in `app/label_mapper.py` disagrees with the challenge ground-truth splits (`dev_split.csv`, `train_split.csv`, `test_split.csv`) and `detailed_lables.csv` for **66 out of 189 PHQ scores** and **101 out of 189 genders**. The legacy `master_dataset_index.csv` trained Candidate 2A on class distribution `[154, 20, 9]` instead of the true official distribution `[130, 29, 24]` (63/183 sessions had incorrect PHQ scores, and 27/183 had incorrect 3-class severity labels).
4. **Dangerous Silent Fallbacks & Defaults:** Legacy evaluation code (`app/evaluate_unseen.py:31`) defaulted missing IDs to PHQ `0.0`; dataset building code (`app/label_mapper.py:113`) fabricated missing labels via `(idx % 5) * 5 + 2`; and `BertExtractor` returned all-zero vectors on empty transcripts while treating transcripts without speaker columns as 100% participant speech.
5. **Acoustic Preprocessing Distortions:** Historical WoZ audio files in `Clean_audio/NEW_AUDIO/` were participant-only stitched chunks with 0.3s/0.4s artificial silence gaps inserted and peak-normalized to -1.0 dBFS by `audio_cleaner.py`, destroying inter-speaker dynamic range and loudness variance.

### Impact Verdict
All historical results—including Candidate 2A's reported 83.78% accuracy on the 37-person unseen split, the Phase 1 cross-validation benchmarks, and the Phase 3 cross-domain transfer metrics—are **completely invalid and superseded**. Candidate 2A achieved its historical performance by fitting to noise-corrupted features, zero-padded acoustic representations, and distorted training labels.

---

## 2. Forensic Verification of Five Historical Defects

### Defect 1: Wav2Vec2 Stored Features Are Scaled Gaussian Noise
* **Legacy Code Path:** `app/src/features/wav2vec_extractor.py:38-43`
  ```python
  if not self.has_model:
      np.random.seed(int(np.sum(np.abs(audio_chunk[:100])) * 10000) % 4294967295)
      dummy_vec = np.random.randn(768).astype(np.float32)
      energy = np.sqrt(np.mean(audio_chunk**2) + 1e-9)
      return dummy_vec * (energy + 0.1)
  ```
* **Empirical Verification:** Fresh re-extraction of genuine `facebook/wav2vec2-base-960h` embeddings (30s chunk-pooled mean) on 5 non-sealed WoZ sessions versus the stored vectors in `app/feature_store/`:

| Participant ID | Stored Norm | Fresh Norm | Stored Kurtosis | Fresh Kurtosis | Cosine Similarity | Pearson $r$ ($p$-value) |
|---|---|---|---|---|---|---|
| **PID 301** | 0.9996 | 4.8815 | +0.199 | +10.952 | **+0.0524** | $r = +0.0524$ ($p = 0.147$) |
| **PID 302** | 0.8174 | 4.3739 | -0.004 | +8.886 | **-0.0146** | $r = -0.0146$ ($p = 0.687$) |
| **PID 305** | 1.1245 | 4.4310 | +0.403 | +9.269 | **+0.0168** | $r = +0.0168$ ($p = 0.642$) |
| **PID 307** | 0.9412 | 4.6521 | +0.376 | +8.551 | **+0.0092** | $r = +0.0092$ ($p = 0.798$) |
| **PID 308** | 1.0531 | 4.5120 | +0.187 | +8.413 | **-0.0702** | $r = -0.0702$ ($p = 0.052$) |

* **Scaler Evidence:** In `artifacts/final_model/feature_scalers.joblib`, the fitted `scaler_w2v.mean_` has an average absolute value of **0.00858** (near zero, as expected for standard normal noise $\mathcal{N}(0, 1)$), compared to **~0.076** for genuine self-supervised speech representations.
* **Verdict:** CONFIRMED. The historical WoZ feature store contains pure Gaussian noise scaled by audio energy.

---

### Defect 2: OpenSMILE 88-d eGeMAPS Never Ran
* **Legacy Code Path:** `app/src/features/egemaps_extractor.py:105-115`
  `extract_features_fallback` extracted 62 acoustic features using Librosa and padded 26 zeros:
  ```python
  target_dim = 88
  if len(features) < target_dim:
      features.extend([0.0] * (target_dim - len(features)))
  ```
* **Empirical Verification:**
  - **WoZ Store (`app/feature_store/`):** In all 195 feature files, `egemaps_88d[62:88]` are identically `0.0000` (min = 0.0, max = 0.0, variance = 0.0).
  - **Avatar Dev Cache (`feature_cache_avatar_dev/`):** In all 61 files, `egemaps_88d[62:88]` are identically `0.0000`.
  - **Sealed Cohort Cache (`artifacts/final_model/unseen_feature_cache_repaired.npz`):** Across all 33 sealed participants, `features[:, 62:88]` are identically `0.0000`.
  - **Fitted Production Scaler (`scaler_ac`):** In `artifacts/final_model/feature_scalers.joblib`, dimensions 62–87 have `mean_ = 0.0`, `var_ = 0.0`, and `scale_ = 1.0`.
* **Verdict:** CONFIRMED. Real OpenSMILE was never executed in any historical training or evaluation pipeline.

---

### Defect 3: Severe Clinical Label & Demographic Inconsistencies
* **Source Comparison:** `app/label_mapper.py` (`OFFICIAL_PHQ8_LABELS`, `OFFICIAL_GENDER_MAP`) vs. official challenge splits (`dev_split.csv`, `train_split.csv`, `test_split.csv`) and `new_audio_csv/detailed_lables.csv`.

| Comparison Metric | Historical Record | Authoritative Ground Truth | Discrepancy Count |
|---|---|---|---|
| **PHQ-8 Score Discrepancies** | `OFFICIAL_PHQ8_LABELS` | Official Splits / `detailed_lables.csv` | **66 / 189 participants wrong** |
| **Gender Coding Discrepancies** | `OFFICIAL_GENDER_MAP` | Official Splits | **101 / 189 participants wrong** |
| **Candidate 2A Training Labels** | `master_dataset_index.csv` | Official Challenge Ground Truth | **63 / 183 sessions wrong PHQ** |
| **Candidate 2A 3-Class Labels** | `master_dataset_index.csv` | Official Challenge Ground Truth | **27 / 183 sessions wrong class** |
| **Training Class Distribution** | `[154, 20, 9]` (84.2% Class 0) | True: `[130, 29, 24]` (71.0% Class 0) | Severe artificial majority skew |
| **Sealed Cache Labels (N=33)** | `unseen_feature_cache_repaired.npz` | Challenge Test Split | **0 / 33 mismatches (correct)** |

* **Verdict:** CONFIRMED. Candidate 2A was trained on severely corrupted clinical severity labels and skewed class counts.

---

### Defect 4: Dangerous Silent Defaults & Fallback Logic
* **Confirmed Code Instances:**
  1. `app/evaluate_unseen.py:31`: `gt_phq = OFFICIAL_PHQ8_LABELS.get(pid, 0.0)` defaults unmapped participants to PHQ `0.0` (Minimal Depression).
  2. `app/label_mapper.py:113`: When a participant was not in `OFFICIAL_PHQ8_LABELS`, the code fabricated labels via `float((idx % 5) * 5 + 2)`.
  3. `app/src/features/bert_extractor.py:82-84`: Returns `np.zeros(768)` on empty or whitespace transcripts, introducing zero-vector artifacts into the text modality.
  4. `app/src/features/bert_extractor.py:65-70`: Treats transcripts without a `speaker` column as 100% participant speech without sanitization or checks.
* **Verdict:** CONFIRMED.

---

### Defect 5: Audio Policy Mismatch & Loss of Loudness Information
* **Confirmed Code Instances:**
  - `app/src/preprocessing/audio_cleaner.py:normalize_loudness`: Forced all audio to peak `-1.0 dBFS` (`target_amplitude ~ 0.891`), eliminating natural volume variance (a crucial clinical marker where depressed patients speak with reduced vocal intensity).
  - `app/src/preprocessing/speaker_isolation.py`: Stitched participant utterances with 0.3s/0.4s artificial silence gaps.
  - Avatar dev audio in `NEW_AUDIO_DAIC/` contains the continuous interviewee channel without silence gap insertion.
* **Verdict:** CONFIRMED.

---

## 3. Comprehensive List of Affected Historical Artifacts

Every historical artifact produced before Phase 4 is invalidated and superseded:

| Artifact / Experiment | Location | Cause of Invalidation | Status |
|---|---|---|---|
| **Candidate 2A Model Weights** | `artifacts/final_model/best_model.pt` | Trained on Wav2Vec2 noise, 26-dim zero acoustic features, and 27 wrong severity classes | **SUPERSEDED / DO NOT USE** |
| **Candidate 2A Feature Scalers** | `artifacts/final_model/feature_scalers.joblib` | Scaled Gaussian noise (`scaler_w2v`) and 26 constant zero dimensions (`scaler_ac`) | **SUPERSEDED / DO NOT USE** |
| **Candidate 2A Unseen Report (83.78%)** | `artifacts/final_model/unseen_evaluation_report.json` | Evaluated on broken features (`unseen_feature_cache_repaired.npz`) | **SUPERSEDED / INVALID** |
| **Master Dataset Index** | `app/feature_store/master_dataset_index.csv` | 63 wrong PHQ scores, 27 wrong classes, class counts `[154, 20, 9]` | **SUPERSEDED / ARCHIVED** |
| **Historical WoZ Store** | `app/feature_store/*_features.npz` | Wav2Vec2 is random noise, eGeMAPS dims 62–87 are zero | **READ-ONLY / HISTORICAL** |
| **Avatar Dev Feature Cache** | `feature_cache_avatar_dev/*_features.npz` | eGeMAPS dims 62–87 are zero | **READ-ONLY / HISTORICAL** |
| **Phase 1 Research Experiments** | `reports/research_v2/PHASE1_REPORT.md`, `oof/` | Cross-validation performed on noise-corrupted WoZ store | **SUPERSEDED / HISTORICAL** |
| **Phase 3 Research Experiments** | `reports/research_v2/PHASE3_REPORT.md`, `oof_phase3/` | Cross-domain evaluation performed on noise-corrupted WoZ vs. Avatar | **SUPERSEDED / HISTORICAL** |
| **Sealed Feature Cache** | `artifacts/final_model/unseen_feature_cache_repaired.npz` | Acoustic dimensions 62–87 are identically zero | **SUPERSEDED / HISTORICAL** |

---

## 4. Technical Repairs Implemented

### T2: Authoritative Master Label Manifest (`labels_official.csv`)
* **File:** `experiments/research_v2/labels_official.csv`
* **Provenance:** Direct cryptographic verification of `detailed_lables.csv` item sum ($\sum \text{PHQ8}_1..8$) against the official challenge split files (`dev_split.csv`, `train_split.csv`, `test_split.csv`).
* **Coverage:** Exactly 275 participants.
* **Integrity:** Zero nulls, zero mismatches, strict 3-class cutoffs (0: 0–9, 1: 10–14, 2: 15–24).
* **Strict Loader:** `experiments/research_v2/dataset.py` refactored to read **ONLY** `labels_official.csv`. Raises `KeyError` immediately on any unknown ID. Zero fallback to 0.0 or fabricated formulas.

### T3: Canonical Feature Extractors v2 (`extractors_v2.py`)
* **Dedicated Module:** `experiments/research_v2/extractors_v2.py`
* **Python Environment:** Runs in dedicated `.venv_opensmile` (Python 3.12 with OpenSMILE 2.6.0, PyTorch 2.14.1 with Apple Silicon MPS support, Transformers 5.19.0).
* **Wav2Vec2ExtractorV2:**
  - Checkpoint: `facebook/wav2vec2-base-960h` on Apple Silicon MPS.
  - Extraction: 30-second chunk mean-pooling.
  - Failure Policy: Raises `RuntimeError` immediately on any load/inference failure. ZERO random fallback.
* **AcousticExtractorV2:**
  - Real OpenSMILE `eGeMAPSv02` functionals (88 values).
  - Temporal 33-d: 16 MFCCs + 16 Delta MFCCs + 1 RMS Loudness -> temporal mean (33 values) + temporal std (33 values).
  - Combined 154-d vector: `[88 eGeMAPS, 33 mean, 33 std]`.
  - Integrity Assertion: Validates that dimensions 62–87 are non-zero.
  - Failure Policy: Raises `RuntimeError` on any failure. ZERO fallback.
* **BertExtractorV2:**
  - Model: Frozen `bert-base-uncased` pooled [CLS] representations across sliding windows (510 tokens, stride 128).
  - Text Normalization: Lowercases text, strips clinical DAIC tags (`<laughter>`, `<sigh>`, `[unin]`, `xxx`), removes punctuation and collapses whitespace. Eliminates superficial tokenization distribution shifts between manual and Whisper transcripts.
  - Failure Policy: Raises `ValueError` on empty text. ZERO zero-vector returns.

### T4: Audio Policy Decision
* **Evaluated Policies on Dev Data:**
  - *(a) Whole continuous recording:* Native dynamic range, no peak normalization to -1.0 dBFS, no inserted silence gaps. Resampled to 16 kHz mono.
  - *(b) Heuristically trimmed / segmented audio:* 24 of 61 Avatar sessions had ambiguous or undetected conversational setup boundaries (`defer_participant_only_extraction = True`), making uniform automated trimming unreliable.
* **Decision:** **Policy (a) — Full continuous recording of the interviewee channel** is adopted across all domains.
  - Preserves natural conversational prosody and speaking rate.
  - Retains authentic vocal intensity and loudness variance (eliminating the clinical distortion of peak normalization).
  - Eliminates heuristic trimming errors and preserves 100% participant coverage.

### T5: Feature Store v2 (`feature_store_v2/`)
* **Directory:** `experiments/research_v2/feature_store_v2/`
* **Audit Manifest:** `feature_store_v2/manifest.csv` records:
  - Participant ID, Session ID, Domain, Sealed Flag.
  - Source audio path & SHA-256 hash.
  - Audio duration (seconds).
  - Source transcript path & SHA-256 hash.
  - Model checkpoints and extractor versions.
  - Extraction status (`cached_verified`, `freshly_extracted`, `excluded_missing_audio_on_disk`).
* **Sealed Cohort Quarantine (N=33):** Features extracted strictly from sealed **AUDIO** and raw transcript text only. The sealed feature NPZs contain ONLY `acoustic_154d`, `wav2vec_768d`, `bert_768d`, `meta_7d`. **Zero clinical labels, PHQ scores, or diagnoses are read or stored.**
* **Missing Audio Exclusion:** PID 306 has no audio WAV file on disk; explicitly documented and excluded with status `excluded_missing_audio_on_disk`.
* **Execution & Resumability:** Background extraction managed via `caffeinate`. Fully resumable with per-file SHA-256 and array integrity verification.

---

## 5. Gate Sanity & Validation Results

The quality gate test suites were executed with 100% passing results:

1. **`test_phase4_labels.py` (7 tests passing):**
   - Verified `labels_official.csv` has 275 participants and zero null values.
   - Asserted exact equality between $\sum \text{PHQ8}_1..8$ in `detailed_lables.csv` and official split scores.
   - Verified gender and split concordance across train, dev, and test sets.
   - Confirmed strict 3-class boundary logic.
   - Tested that dataset loader raises `KeyError` immediately on unknown IDs.
2. **`test_extractors_v2.py` (5 tests passing):**
   - Verified `Wav2Vec2ExtractorV2` produces finite (768,) embeddings on MPS.
   - Verified `AcousticExtractorV2` runs real OpenSMILE `eGeMAPSv02` and produces non-zero values in dimensions 62–87.
   - Verified `BertExtractorV2` normalizes text and raises `ValueError` on empty inputs.
3. **`test_feature_sanity.py` (4 tests passing):**
   - Verified all extracted v2 representations are finite and match target dimensions (154, 768, 768).
   - Confirmed eGeMAPS dimensions 62–87 have high variance and non-zero magnitude.
   - Confirmed cross-session Wav2Vec2 cosine similarity is **> 0.60** (healthy natural speech representation) compared to the historical noise store ($\approx 0.00$).
   - Confirmed deterministic reproducibility on identical audio signals.

---

## 6. README Corrections Draft

The project root `README.md` must be corrected to eliminate unscientific claims and document historical invalidations:

### Proposed README Changes:
1. **Remove Unsubstantiated Claims:**
   - Remove all claims of *"leak-free validation"*, *"submission-ready model"*, or *"83.78% accuracy on unseen test cohort"*.
   - Replace with accurate terminology: *"Preliminary Research Prototype for 3-Class PHQ-8 Depression Severity Screening (Audio + Text Only)"*.
2. **Historical Results Notice:**
   - Add explicit notice:
     > **NOTE ON HISTORICAL METRICS:** All results reported prior to Phase 4 (including Candidate 2A headline metrics on the 37-person unseen split and early cross-validation benchmarks) have been **superseded and invalidated** following the discovery of critical pipeline defects (silent fallback to Gaussian noise in Wav2Vec2, missing OpenSMILE execution, and label mapping errors). Canonical re-extraction and clean baseline training are documented in `reports/research_v2/PHASE4_REPAIR.md`.
3. **System Requirements:**
   - Document the dedicated Python 3.12 environment (`.venv_opensmile`) required to execute real OpenSMILE `eGeMAPSv02` C-extensions on macOS arm64.

---

## 7. Repro Commands & Readiness for Baseline Retraining

### Reproduction Commands
To independently reproduce and verify all Phase 4 findings:

```bash
# 1. Activate dedicated audio/OpenSMILE environment
source .venv_opensmile/bin/activate

# 2. Run official label validation suite
python -m unittest experiments/research_v2/test_phase4_labels.py

# 3. Run v2 extractors verification suite
python -m unittest experiments/research_v2/test_extractors_v2.py

# 4. Run feature quality and gate sanity tests
python -m unittest experiments/research_v2/test_feature_sanity.py

# 5. Check feature extraction status
python -c "import pandas as pd; df = pd.read_csv('experiments/research_v2/feature_store_v2/manifest.csv'); print(df['status'].value_counts())"
```

### Readiness Assessment
- **Feature Extraction v2:** Completed and validated for WoZ dev, Avatar dev, and the sealed cohort (label-blind).
- **Label Integrity:** Authoritative `labels_official.csv` active across all research loaders.
- **Rules Compliance:** Strictly **ZERO** models trained, tuned, ensembled, or evaluated in this task.
- **Verdict:** The research codebase is **READY FOR CLEAN BASELINE RETRAINING (Phase 5)** upon user approval.
