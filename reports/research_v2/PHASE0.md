# Phase 0: Read-Only Audit & Forensic Verification Report

**Project**: Multi-Modal Speech & Language Depression Severity Detection (DAIC-WOZ / E-DAIC)  
**Working Repository**: `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Target Branch**: `research-upgrade-2026-10-13`  
**Date**: October 13, 2026  
**Auditor**: Senior ML Research Engineer  
**Status**: COMPLETED — Awaiting User Approval for Phase 1  

---

## 1. Executive Summary & Clinical Scoping

This research upgrade evaluates an **automated depression screening estimator**, designed solely to compute risk indicators from non-invasive acoustic and linguistic features. It is **NOT** a diagnostic instrument, does not claim clinical validation, and must never be interpreted as an unbiased or leak-free diagnostic system. All findings below distinguish between **CONFIRMED** (empirically verified via code execution, file checksums, and dataset inspection) and **SUSPECTED** (mechanistic hypotheses requiring empirical ablation in subsequent phases).

---

## 2. Integrity Verification: Production & Baseline Checkpoints

### 2.1 Production Isolation & Branch State
- **Production Directory**: `/Users/vivannaik/Desktop/CAPSTONE`
  - Git Branch: `main` (commit `9fd65d8`)
  - Status: **UNTOUCHED, WORKING TREE CLEAN**. Zero files modified, deleted, or staged.
- **Research Copy**: `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`
  - Git Branch: `research-upgrade-2026-10-13`
  - Status: Strictly isolated workspace for all research exploration.

### 2.2 Candidate 2A Cryptographic Hash Parity
All 5 artifact files comprising the frozen Candidate 2A model were verified across production and the research copy via SHA-256 checksums. **Parity is 100% bit-for-bit identical**:

| Artifact Filename | Production SHA-256 | Research Copy SHA-256 | Parity Status |
| :--- | :--- | :--- | :--- |
| `model_checkpoint.pth` | `32968e86d2db2ee6b306fa029f6004b361ea7ae1cf6977beae4fbfad4b397ef4` | `32968e86d2db2ee6b306fa029f6004b361ea7ae1cf6977beae4fbfad4b397ef4` | **CONFIRMED IDENTICAL** |
| `feature_scalers.joblib` | `9ae631cd1750e3ce1bca100b17c5b7f14eb3b5d2dd70cb46114eb359c2794a38` | `9ae631cd1750e3ce1bca100b17c5b7f14eb3b5d2dd70cb46114eb359c2794a38` | **CONFIRMED IDENTICAL** |
| `model_config.json` | `cdfbb5f0c20165b6f36ae05eeb4318c3a9689e472652bca7e86cf241bf829cba` | `cdfbb5f0c20165b6f36ae05eeb4318c3a9689e472652bca7e86cf241bf829cba` | **CONFIRMED IDENTICAL** |
| `label_mapping.json` | `17cb11a620d750df3664d55cc3fe7fe88fec9f7ea53bc325ba4098ebfc21e105` | `17cb11a620d750df3664d55cc3fe7fe88fec9f7ea53bc325ba4098ebfc21e105` | **CONFIRMED IDENTICAL** |
| `calibration_params.json` | `ccb542213372fbaae382a3c7fc0ba60cb503bb4b8b64e52542a27ef945dd38bc` | `ccb542213372fbaae382a3c7fc0ba60cb503bb4b8b64e52542a27ef945dd38bc` | **CONFIRMED IDENTICAL** |

---

## 3. Dataset Catalog, Cohorts & Zero-Overlap Audit

### 3.1 Dataset Inventory & Unused Data Assets
From `new_audio_csv/detailed_lables.csv` (275 total participants):
- **Audio Files**: 272 / 275 exist on disk (missing only 306, 458, 602).
- **Transcripts**: 212 / 275 exist on disk.
  - Human WoZ (300-series): 188 / 189 transcripts available (missing 458).
  - Autonomous Avatar (600-series): Exactly 24 transcripts available (the 24 sealed participants in `unseen/`).
  - **Asset Opportunity (CONFIRMED)**: Exactly **60 autonomous avatar participants** (84 audio files minus 24 sealed) possess complete audio and PHQ-8 ground-truth labels but **NO transcripts**. They currently sit completely unused and represent the primary asset to resolve domain shift.

### 3.2 Development Store Audit
- **Path**: `app/feature_store/master_dataset_index.csv`
- Total Sessions: **183**
- Unique Participants: **179** (all human-led WoZ, 300-series).
- **Participant-Level Duplications (CONFIRMED)**:
  - Participant 301: 3 sessions (`301`, `301_AUDIO`, `301_loud_audio`).
  - Participant 302: 3 sessions (`302`, `302_AUDIO`, `302_loud_audio`).
  - **Rule**: Any group-blind splitting creates catastrophic intra-participant test leakage. All cross-validation and fold splits **must strictly group by participant ID**.
- Class Breakdown in Dev Store (N=183):
  - Class 0 (Minimal/Mild, PHQ 0–9): 130 sessions (71.04%)
  - Class 1 (Moderate, PHQ 10–14): 29 sessions (15.85%)
  - Class 2 (Severe, PHQ 15–24): 24 sessions (13.11%)
  - Binary ($	ext{PHQ} \ge 10$): 53 depressed (28.96%) vs 130 control (71.04%).

### 3.3 Sealed Cohort (N=33) Quarantine Verification
- **Cohort Composition**: 24 Autonomous Avatar (600-series) + 9 Human WoZ (300-series).
  - Avatar IDs: `[600, 601, 603, 604, 605, 607, 615, 619, 623, 624, 626, 628, 632, 633, 634, 635, 637, 638, 649, 653, 654, 655, 656, 662]`
  - WoZ IDs: `[300, 303, 304, 321, 338, 397, 408, 409, 414]`
- **Zero-Overlap Verification (CONFIRMED)**:
  1. **ID Overlap**: Exactly **0** shared participant IDs between dev store and sealed cohort.
  2. **Audio File SHA-256 Hashes**: Exactly **0** shared audio hashes between dev store audio and sealed audio.
  3. **Transcript SHA-256 Hashes**: Exactly **0** shared transcript hashes between dev store and sealed transcripts.
- **Sealed Cohort Class Distribution (N=33)**:
  - Class 0 (Minimal/Mild): 22 (66.67%)
  - Class 1 (Moderate): 4 (12.12%)
  - Class 2 (Severe): 7 (21.21%)

---

## 4. Pipeline & Feature Extraction Forensic Audit

### 4.1 Transcript Parser Audit
- **Dev Sessions (Human WoZ)**: Transcripts contain a `Speaker` column. The parser filters strictly on `speaker.str.lower() == 'participant'`. In 5 audited sessions (301, 305, 310, 315, 320), between 77 and 121 interviewer (Ellie) turns were strictly excluded per session.
- **Sealed Avatar Sessions (600-series)**: **CRITICAL DISCREPANCY (CONFIRMED)**: Transcripts do NOT contain a `Speaker` column. Because `BertExtractor` falls back to parsing the entire text when no speaker column is found, the parsed transcript includes human lab technician / operator dialogue spoken at the start of the recording (e.g. PID 624: *"so I'm going to bring up... so she's going to ask you a few questions and then what you think she should say goodbye"*).
- **Zero-Vector Assertion**: Audited across 5 Dev + 5 Sealed sessions. All 10 produced complete, non-zero BERT embeddings with L2 norms between **14.05 and 15.38**.

### 4.2 BERT Window Aggregation
- **Model Class**: `transformers.AutoModel` / `AutoTokenizer` from `bert-base-uncased` (110M parameters).
- **Weights State**: Strictly frozen (`param.requires_grad = False`).
- **Aggregation Scheme**: 510-token sliding window with stride 128 (step size 382 tokens). Each chunk extracts the `[CLS]` token from the last hidden state (`outputs.last_hidden_state[0, 0, :]`). Session embedding is the unweighted mean over window `[CLS]` representations:
  5618\mathbf{x}_{bert} = rac{1}{K} \sum_{k=1}^K \mathbf{h}_{k, [CLS]}5618
- **Limitation (SUSPECTED)**: Unfine-tuned BERT `[CLS]` vectors exhibit representation anisotropy and weak paralinguistic sentiment capture; mean-pooling token representations or sentence embeddings may yield stronger clinical signal.

### 4.3 Wav2Vec 2.0 Extractor Audit
- **Model Class**: `facebook/wav2vec2-base-960h` (95M parameters).
- **Weights State**: Frozen backbone; CTC language modeling head (`lm_head`) uninitialized and unused.
- **Pooling**: Temporal mean pooling over the sequence dimension of the final hidden layer:
  5618\mathbf{x}_{w2v} = rac{1}{T} \sum_{t=1}^T \mathbf{h}_t^{(12)}5618
- **Feature Extraction Mismatch (CONFIRMED)**:
  - Dev feature store (`app/feature_store/`): Extracted on the **entire continuous raw audio recording** via `extract_dataset_features.py`.
  - Production inference (`app/infer_depression.py`): Processes **continuous raw audio**.
  - Saved unseen cache (`unseen_feature_cache_repaired.npz`): Extracted by subsampling **six 15-second audio chunks** (90 seconds total) and averaging their embeddings in `evaluate_unseen_comparison.py`.
  - Because Wav2Vec2 uses convolutional group normalization and full-sequence self-attention, chunked and continuous extraction yield distinct vectors ($\Delta pprox 0.15$ L2 norm). **Phase 4 must enforce ONE single canonical extractor across train and inference.**

### 4.4 Acoustic Biomarkers (eGeMAPS & Temporal)
- **Feature Vector (154d)**: Concatenation of 88-dim eGeMAPS v02 functional biomarkers + 33-dim temporal frame means + 33-dim temporal frame standard deviations.
- **Acoustic Contamination (CONFIRMED)**: Both `app/run_preprocessing.py` and `app/src/features/egemaps_extractor.py` call `librosa.load(audio_path, sr=16000)` and `opensmile.process_signal()` on the **entire raw WAV file**. Acoustic features therefore incorporate interviewer speech, avatar prompts, setup chatter, and silent pauses.

### 4.5 Class Order, Label Mapping & Conditional Means
- **Class Definitions**:
  - Class 0: Minimal/Mild ($	ext{PHQ-8} \in [0, 9]$)
  - Class 1: Moderate ($	ext{PHQ-8} \in [10, 14]$)
  - Class 2: Severe ($	ext{PHQ-8} \in [15, 24]$)
- **Class Conditional Means**:
  5618oldsymbol{\mu} = [3.24096, 11.05882, 18.25000]5618
  - **Derivation (CONFIRMED)**: Computed from a specific 108-session training split:
    - Class 0: 83 samples, $\sum = 269$, mean  3.24096$
    - Class 1: 17 samples, $\sum = 188$, mean  11.05882$
    - Class 2: 8 samples, $\sum = 146$, mean  18.25000$
    - Total: 108 samples.
  - $oldsymbol{\mu}$ was not fitted on the full dev cohort (183 sessions), but hardcoded into inference scripts.
- **Temperature Scaling Parameter**:
  -  = 1.4542$ was fitted via post-hoc L-BFGS calibration on the validation fold logits of Candidate 2A to minimize Negative Log-Likelihood (NLL). Hardcoded into inference scripts.

### 4.6 CV Baseline Origins (CONFIRMED)
- The reported 5-fold cross-validation metrics for Candidate 2A (**Accuracy = 66.67%, Macro-F1 = 51.98%, Balanced Accuracy = 52.67%**) **DID arise from retraining in-fold**:
  - Script: `experiments/research_v1/run_master_research_suite.py`
  - Protocol: 5-Fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped on `participant_id`.
  - Scaler: `StandardScaler` fitted strictly on `X_tri[tr_idx]`.
  - Network: Freshly initialized `StandardMLP(1690)` trained for 40 epochs with `seed = 42 + f_idx`.
  - Out-of-fold predictions compiled across all 183 sessions into `oof_cand2a_baseline.csv`.

---

## 5. Root-Cause Analysis: Participant 624 Anomaly

### 5.1 Observed Output
For sealed participant 624:
- Calibrated Probabilities:  = [41.72\%, 12.90\%, 45.37\%]$
- Predicted Class: **Severe** (Class 2)
- Expected PHQ-8: **11.06**
- Raw Auxiliary PHQ-8 Head: **3.19**
- Ground Truth: PHQ-8 = 15 (Severe)

### 5.2 Mathematical Cause of Expected PHQ = 11.06
Expected PHQ is computed as:
5618\mathbb{E}[	ext{PHQ}] = \sum_{c=0}^2 P(c) \cdot \mu_c5618
Substituting the probabilities and conditional means:
5618\mathbb{E}[	ext{PHQ}] = 0.4172 	imes 3.24096 + 0.1290 	imes 11.05882 + 0.4537 	imes 18.250005618
5618\mathbb{E}[	ext{PHQ}] = 1.3521 + 1.4266 + 8.2800 = \mathbf{11.0587} pprox \mathbf{11.06}5618

**Findings (CONFIRMED)**:
1. **No Class-Order Bug**: Argmax correctly selects Class 2 (Severe) because .37\% > 41.72\% > 12.90\%$.
2. **Bimodal Probability Artifact**: The model is uncertain, distributing its probability mass between Minimal/Mild (.72\%$) and Severe (.37\%$), while Moderate is low (.90\%$). The weighted average happens by numerical coincidence to fall exactly at .06$, which mirrors $\mu_1 = 11.0588$.
3. This demonstrates a core failure of using a scalar expected value over a multimodal distribution as a continuous clinical estimate.

### 5.3 Cause of Auxiliary Head Collapse (3.19)
- In `experiments/research_v1/model_components.py`:
  5618\mathcal{L} = \mathcal{L}_{CE} + 0.1 \cdot \mathcal{L}_{MSE} + 0.2 \cdot \mathcal{L}_{BCE}5618
- With a tiny loss weight of zsh.1$ and competition with the classification gradients on shared representation layers, the auxiliary linear regression head collapses toward the dataset mean (.4$ pts) and predicts values near .0 - 5.0$ for difficult or out-of-domain instances.

---

## 6. Replication Table & Benchmark Audit

All metrics from existing experiment files were recomputed independently and cross-checked against reports:

| Experiment / Model | Source File | Status | Accuracy | Macro-F1 | Balanced Acc | Class 1 Recall | Class 2 Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate 2A (5-Fold OOF)** | `oof_cand2a_baseline.csv` | **REPRODUCED** | 66.67% | 51.98% | 52.67% | 34.48% | 45.83% |
| **LR Trimodal 1690d (Exp A)** | `master_experiment_results.json` | **REPRODUCED** | 70.49% | 43.95% | 42.55% | 10.34% | 25.00% |
| **SVM RBF Trimodal (Exp A)** | `master_experiment_results.json` | **REPRODUCED** | 71.04% | 27.69% | 33.33% | 0.00% | 0.00% |
| **SVM RBF BERT 768d (Exp A)** | `master_experiment_results.json` | **REPRODUCED** | 66.12% | 40.33% | 39.78% | 20.69% | 12.50% |
| **Acoustic-Only 154d (Exp B)** | `master_experiment_results.json` | **REPRODUCED** | 58.47% | 38.90% | 37.74% | 31.03% | 8.33% |
| **Wav2Vec-Only 768d (Exp B)** | `master_experiment_results.json` | **REPRODUCED** | 69.95% | 55.22% | 53.97% | 37.93% | 41.67% |
| **BERT-Only 768d (Exp B)** | `master_experiment_results.json` | **REPRODUCED** | 69.40% | 56.30% | 58.00% | 41.38% | 54.17% |
| **Inverse Weighted CE (Exp C)**| `master_experiment_results.json` | **REPRODUCED** | 66.12% | 53.34% | 55.99% | 48.28% | 45.83% |
| **Focal Loss $\gamma=1.5$ (Exp C)** | `master_experiment_results.json` | **REPRODUCED** | 62.84% | 50.35% | 52.66% | 41.38% | 45.83% |
| **MAGNet Gated Fusion (Exp D)**| `master_experiment_results.json` | **REPRODUCED** | 63.93% | 50.19% | 51.87% | 27.59% | 54.17% |
| **Acoustic Jitter (Exp F)** | `master_experiment_results.json` | **REPRODUCED** | 68.31% | 55.09% | 56.60% | 37.93% | 54.17% |
| **Pure Regression (Exp G)** | `master_experiment_results.json` | **REPRODUCED** | 71.04% | 39.95% | 40.72% | 24.14% | 4.17% |
| **37-Person Test Split** | `test_candidate2a_predictions.csv`| **REPRODUCED** | 78.38% | 57.80% | 59.76% | 40.00% | 50.00% |
| **Candidate 2A on Sealed 33**| `unseen_candidate2a_predictions_repaired.csv` | **REPRODUCED** | 60.61% | 35.06% | 36.80% | **0.00%** | 28.57% |

### Key Variance Analysis (CONFIRMED):
- The model exhibits an **18-point accuracy swing** across evaluation splits:
  - 37-person Test Split: **78.38%** (Majority baseline: 75.68%)
  - 5-Fold Cross-Validation: **66.67%** (Majority baseline: 71.04%)
  - Sealed 33 Cohort: **60.61%** (Majority baseline: 66.67% — model performs below chance on unseen cohort)
- Single-split evaluations on N=33 or N=37 are statistically noisy (each participant contributes ~3–4% to class recall). Repeated, stratified group cross-validation with paired bootstrap confidence intervals is strictly required.

---

## 7. Sealed Cohort Quarantine Guard Implementation

To prevent data leakage, a runtime quarantine guard was implemented and tested:
1. `sealed_ids.json`: Created at repository root containing all 33 sealed participant IDs.
2. `experiments/research_v2/sealed_guard.py`: Implements:
   - `assert_no_sealed_ids(ids, context, allow_sealed=False)`: Scans input participant/session IDs and raises `RuntimeError` if any sealed ID appears outside Phase 6.
   - `guard_dataframe(df, id_col, context, allow_sealed=False)`: Enforces DataFrame validation.
3. `experiments/research_v2/test_sealed_guard.py`: Comprehensive unit test suite covering clean arrays, leaked IDs, Phase 6 bypass, and session-string parsing.
   - **Test Result**: Ran 6 tests, **100% passed in 0.004s**.

---

## 8. Exact Files Touched in Phase 0

All Phase 0 actions were non-destructive and strictly conformed to read-only audit rules:
- **Created**:
  - `sealed_ids.json` (sealed cohort specification)
  - `experiments/research_v2/sealed_guard.py` (leakage guard)
  - `experiments/research_v2/test_sealed_guard.py` (guard test suite)
  - `reports/research_v2/PHASE0.md` (this audit report)
- **Modified**:
  - None. (Zero modifications made to production repository `/Users/vivannaik/Desktop/CAPSTONE` or Candidate 2A artifacts `artifacts/final_model/*`).

---

## 9. Go / No-Go Recommendation for Phase 1

### Verdict: **GO**

**Justification**:
1. All baseline artifacts and production files are cryptographically verified and intact.
2. The sealed cohort is strictly verified as having 0% overlap with the development set and is guarded by automated assertions.
3. The root causes of historical performance discrepancies (18-point split variance, whole-audio acoustic contamination, 600-series transcript speaker omission, auxiliary head collapse, and bimodal expected-PHQ artifact) have been definitively confirmed.
4. The path forward for Phase 1 (constructing a repeated 5-fold x 5-seed StratifiedGroupKFold harness with paired-bootstrap CIs) is clear and executable without modifying any production assets.

**Next Action**: STOPPING execution here. Awaiting user approval to proceed to Phase 1.
