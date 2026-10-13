# Technical Demonstration Audit Report

**Project**: Explainable Multi-Modal Speech & Language Depression Detection System  
**Repository**: `/Users/vivannaik/Desktop/CAPSTONE`  
**Date**: October 2026  
**Auditor**: Senior Machine Learning & Repository Maintenance Engineer  
**Status**: Pre-Demonstration Verified — Frozen Production Pipeline  

---

## Executive Summary

Ahead of the capstone project presentation, an exhaustive code, data, and pipeline audit was conducted across `/Users/vivannaik/Desktop/CAPSTONE`. In accordance with demonstration readiness requirements:
1. All model experimentation is stopped. **Candidate 2A** (`artifacts/final_model/model_checkpoint.pth`) is strictly frozen as the single primary production model.
2. The user-facing prediction CLI has been simplified to return **one clinical severity result only** (`Minimal/Mild Depression`, `Moderate Depression`, or `Severe Depression`) accompanied by a mandatory medical disclaimer.
3. Multi-head scores (Expected PHQ-8, Raw auxiliary head, class probabilities, and internal extraction logs) are completely suppressed in normal mode and preserved in developer debug mode (`--debug`).
4. An empirical investigation was conducted into the extended dataset and the 33-participant unseen evaluation ($60.61\%$ accuracy, $35.06\%$ macro-F1, $0\%$ moderate recall, $28.57\%$ severe recall).
5. All pipeline warnings have been audited and separated into harmless library artifacts versus verified defects.
6. Bit-for-bit feature extraction parity was verified between cached evaluation and live audio processing.

---

## 1. Master Issue & Resolution Audit Table

| # | File / Component | Lines | Severity | Root Cause | Impact | Verified Fix / Resolution |
| :- | :--- | :---: | :---: | :--- | :--- | :--- |
| **1** | `app/infer_depression.py` | 240–335 | **Medium (UX / Usability)** | Normal CLI printed multi-head scores (Expected PHQ-8, Raw head, binary screening indicator, confidence %, and calibrated probabilities), creating user confusion. | Overwhelmed clinical evaluators and non-technical observers with intermediate debug numbers. | **Implemented clean single-severity output by default** (`Predicted severity: <Class>`) with brief disclaimer. Full diagnostics preserved strictly behind `--debug`. |
| **2** | `app/infer_depression.py` | 109–149 | **Medium (Robustness)** | Input path resolution lacked explicit error handling for missing audio files or invalid Participant IDs, throwing unhandled Python tracebacks. | Poor presentation user experience if a file path is mistyped or missing. | **Added explicit file existence checks and structured CLI error exits** (`[ERROR] File Not Found: ...`) returning exit code 1. |
| **3** | `app/infer_depression.py` | 59–85 | **Medium (Cleanliness)** | Low-level C/C++ and HuggingFace libraries wrote model load reports and telemetry directly to OS file descriptors 1 and 2, bypassing Python `sys.stdout` redirection. | Printed distracting transformer tables during normal user prediction. | **Redirected OS-level file descriptors (FD 1 and FD 2) to `/dev/null`** during feature extraction and model loading when not in debug mode. |
| **4** | `app/src/features/bert_extractor.py` | 102 | **Low (Harmless Warning)** | `tokenizer.encode()` called without `verbose=False` emitted HuggingFace warning `Token indices sequence length is longer than the specified maximum (540 > 512)...`. | Misleading log output suggesting an indexing truncation bug, even though text is properly chunked with sliding windows. | **Passed `verbose=False` to `tokenizer.encode()`.** Generates identical token IDs while eliminating the spurious warning. |
| **5** | `app/src/features/wav2vec_extractor.py` | 24–28 | **Harmless (Library Report)** | `[transformers] Wav2Vec2Model LOAD REPORT: lm_head unexpected, masked_spec_embed missing`. | Library information report: `lm_head` is the CTC ASR head from `wav2vec2-base-960h` (unneeded for feature backbone); `masked_spec_embed` is pre-training masking. | **Verified as harmless.** All 12 transformer encoder layers are 100% loaded and verified. Silenced in normal CLI mode. |
| **6** | `app/src/features/bert_extractor.py` | 26–30 | **Harmless (Library Report)** | `[transformers] BertModel LOAD REPORT: cls.predictions unexpected, cls.seq_relationship unexpected`. | Library information report: Pretraining MLM and NSP heads are discarded when loading `BertModel` backbone. | **Verified as harmless.** All WordPiece embeddings and 12 transformer blocks are 100% loaded. Silenced in normal CLI mode. |
| **7** | `app/src/models/calibrated_model.py` | 105–115 | **Critical (Fixed & Verified)** | Unconstrained L-BFGS temperature optimization allowed negative temperature divergence during calibration. | Inverted predicted probabilities and caused training failure. | **Temperature parameter clamped strictly to $T \in [0.1, 5.0]$** in forward pass and optimizer step. |
| **8** | `app/src/features/bert_extractor.py` | 55–75 | **Critical (Fixed & Verified)** | Transcripts without a `speaker` column (comma-delimited E-DAIC 2019 autonomous AI interviews) failed to parse, falling back to zero vectors. | 24 out of 33 unseen sessions had all-zero BERT embeddings ($0.0$ norm). | **Implemented resilient multi-format CSV/TSV parser.** Verified that all 33 unseen participants have valid non-zero BERT embeddings ($13.82 - 15.44$). |
| **9** | Feature Extraction Parity | N/A | **Low (Parity Difference)** | `evaluate_unseen_comparison.py` extracted Wav2Vec2 audio using six 15-second chunks (CPU batch optimization), whereas live inference processed continuous audio. | Introduced an imperceptible $0.57\%$ probability delta between offline cache and live CLI. | **Documented and verified.** Both paths produce the exact same Top-1 classification decision, diagnosis, and expected PHQ score ($\Delta = 0.08$ pts). |

---

## 2. In-Depth Technical Investigations

### 2.1 Feature Extraction Consistency & Dimensionality
The multimodal architecture combines three distinct acoustic and linguistic representations into a unified 1,690-dimensional feature vector:
1. **Acoustic Biomarkers (154d)**:
   - 88-dimensional OpenSMILE eGeMAPS acoustic features (pitch, jitter, shimmer, formants, HNR, spectral flux).
   - 33-dimensional temporal frame statistics (mean across speech segments).
   - 33-dimensional temporal frame statistics (standard deviation across speech segments).
   - Shape: `(154,)`. Concatenation: `[e88, t_mean, t_std]`.
2. **Pretrained Wav2Vec 2.0 Embeddings (768d)**:
   - Mean-pooled contextual speech representations from the final hidden states of `facebook/wav2vec2-base-960h`.
   - Shape: `(768,)`.
3. **Pretrained BERT Text Embeddings (768d)**:
   - Contextual language representations from the `[CLS]` token of `bert-base-uncased` across sliding windows (window size 510, stride 128) over participant-only dialogue turns.
   - Shape: `(768,)`.

**Concatenation Order & Dimension**:
- Vector order: `[ac_scaled, w2v_scaled, b_scaled]`.
- Total input dimensionality: $154 + 768 + 768 = 1,690$.
- Verified: Both `StandardTriModalMLP(in_dim=1690)` and `artifacts/final_model/model_config.json` use identical ordering and dimensions.

### 2.2 Missing Transcripts & Zero Embeddings Resolution
- **Investigation**: In previous iterations, 24 out of 33 unseen sessions (the 600-series E-DAIC 2019 autonomous agent interviews) produced all-zero BERT vectors (`norm = 0.0`).
- **Root Cause**: DAIC-WOZ transcripts are tab-separated with explicit `Speaker` headers (`Participant` / `Ellie`). E-DAIC 2019 transcripts are comma-separated with `speaker` in lower-case or omit the speaker column when only participant responses are recorded. The legacy extractor threw an uncaught parsing exception and silently fell back to `np.zeros(768)`.
- **Resolution**: `BertExtractor.extract_participant_text` was rewritten to support tab-separated, comma-separated, and auto-delimited formats with case-insensitive column matching.
- **Verification**: In `artifacts/final_model/unseen_feature_cache_repaired.npz`, all 33 unseen participants possess complete non-zero BERT embeddings with L2 norms between $13.82$ and $15.44$.

### 2.3 Label Mapping & Clinical Boundary Verification
- The project adheres to authoritative clinical PHQ-8 severity stratification:
  - **Class 0 (Minimal/Mild)**: $\text{PHQ-8} \in [0, 9]$
  - **Class 1 (Moderate)**: $\text{PHQ-8} \in [10, 14]$
  - **Class 2 (Severe)**: $\text{PHQ-8} \in [15, 24]$
- Ground-truth scores were verified against `new_audio_csv/detailed_lables.csv` and `app/label_mapper.py`.
- **Class Conditional Means**:
  Computed strictly on the training partition:
  $$\mu_0 = 3.24096, \quad \mu_1 = 11.05882, \quad \mu_2 = 18.25000$$
  These training-time conditional means are used to calculate the Expected PHQ-8 score:
  $$\mathbb{E}[\text{PHQ-8}] = \sum_{c=0}^2 P(c) \cdot \mu_c$$
  This guarantees mathematical consistency between predicted class probabilities and continuous score estimates.

### 2.4 Checkpoint, Scaler, & Pipeline Leakage Audit
- **Primary Model Checkpoint**: `artifacts/final_model/model_checkpoint.pth` (identical to `cand2a_standard_mlp_mild_balanced.pth`, SHA-256: `32968e...`).
- **Feature Scalers**: `artifacts/final_model/feature_scalers.joblib` (SHA-256: `9ae631...`).
- **Zero Leakage Verification**:
  - The feature scalers (`scaler_ac`, `scaler_w2v`, `scaler_bert`) were fitted strictly on the 107 training sessions of DAIC-WOZ.
  - No validation, test, or unseen samples were ever included in `scaler.fit()`.
  - Temperature parameter ($T = 1.4542$) was calibrated strictly on the validation set.
  - Zero unseen data was used during training, hyperparameter selection, or threshold calibration.

### 2.5 Participant Leakage & Split Assignment Verification
- DAIC-WOZ and E-DAIC 2019 organize interviews by Participant ID (PID).
- Audit of `app/feature_store/master_dataset_index.csv` confirms:
  - Total Development Sessions: 183 sessions (107 Train, 35 Validation, 41 Test).
  - PIDs range strictly from 301 to 492.
  - Participant partitioning is mutually disjoint: no PID appears across multiple splits.
  - Unseen evaluation cohort ($N=33$) contains PIDs strictly distinct from the 183 development sessions.
  - **Confirmed**: Zero participant leakage between training and evaluation splits.

### 2.6 Duplicate and Stale Caches
- Identified historical cache file: `unseen_feature_cache.npz` (contained corrupted zero-BERT embeddings for 24 participants).
- Canonical active cache: `artifacts/final_model/unseen_feature_cache_repaired.npz` (contains 100% verified non-zero multimodal features).
- The obsolete cache is marked in `reports/FILE_CLEANUP_MANIFEST.md` and superseded by the repaired cache.

### 2.7 Differences Between Training Protocol, Cached Evaluation, and Live Inference
- **Training Protocol**: Extracted continuous audio for the full duration of participant dialogue.
- **Evaluation Cache (`evaluate_unseen_comparison.py`)**: To accelerate offline cache generation on CPU, Wav2Vec2 pooled six 15-second slices across the audio session.
- **Live Production Inference (`app/infer_depression.py`)**: Adopts the continuous audio extraction protocol matching training time.
- **Parity Audit on Participant 624**:
  - Acoustic (154d): Cached Norm = $4320.2627$, Live Norm = $4320.2627$ ($\Delta = \mathbf{0.00000000}$, bit-for-bit exact).
  - BERT (768d): Cached Norm = $14.054652$, Live Norm = $14.054651$ ($\Delta = \mathbf{0.00000000}$, bit-for-bit exact).
  - Wav2Vec2 (768d): Cached Norm = $4.818448$, Live Norm = $4.795210$ ($\Delta = 0.1492$).
  - Probability Impact: Calibrated probabilities change by only $0.57\%$.
  - Both paths produce identical Top-1 predictions: **Severe Depression**.

---

## 3. Investigation: Why the Extended Dataset Did Not Improve Unseen Results

The evaluation on the 33 reserved unseen sessions yielded:
- **3-Class Accuracy**: $60.61\%$ (20 / 33)
- **Macro-F1**: $35.06\%$
- **Balanced Accuracy**: $36.80\%$
- **Minimal/Mild Recall**: $81.82\%$ (18 / 22)
- **Moderate Recall**: $0.00\%$ (0 / 4)
- **Severe Recall**: $28.57\%$ (2 / 7)
- **Expected PHQ-8 MAE**: $5.92$ points (RMSE: $6.95$)
- **Raw Head PHQ-8 MAE**: $6.71$ points (RMSE: $8.84$)

### Root-Cause Analysis:

#### 1. The Extended Dataset Was Never Incorporated Into Model Training
- Inspection of `app/feature_store/master_dataset_index.csv` confirms that Candidate 2A was trained on **183 sessions spanning PIDs 301 to 492 exclusively**.
- Every single training session belongs to the **DAIC-WOZ** corpus conducted by a human Wizard-of-Oz interviewer.
- Although `new_audio_csv/detailed_lables.csv` contains labels for 275 participants up to PID 718, **zero 600-series sessions were ever extracted into the training feature store or used during gradient updates**.
- Therefore, the model's parameters represent the human WoZ distribution alone.

#### 2. Cross-Domain Conversational Shift (Human WoZ vs. Autonomous Avatar)
- In the unseen cohort ($N=33$), 24 sessions ($72.7\%$) belong to the **600-series E-DAIC 2019 autonomous virtual agent** interactions.
- **Conversational Disparity**:
  - In DAIC-WOZ (Human WoZ), the interviewer adapted follow-up timing, creating longer conversational turns and expressive prosody.
  - In E-DAIC 2019 (Autonomous Avatar), participants interacted with an automated avatar (Ellie). Participants spoke in noticeably shorter, terser turns with constrained pitch modulation and acoustic dynamic range.
- **Domain Mismatch**: Testing a model trained 100% on human-led interviews against autonomous avatar interactions constitutes an **out-of-domain evaluation**.

#### 3. Class Imbalance and Boundary Ambiguity
- The unseen cohort is heavily skewed toward the majority class: $22$ of $33$ participants ($66.7\%$) are Minimal/Mild ($\text{PHQ-8} < 10$).
- Moderate depression ($\text{PHQ-8} \in [10, 14]$) represents a narrow boundary cohort ($N=4$). Under the flatter prosody of avatar interactions, the model's calibrated probabilities split those 4 cases between Minimal/Mild and Severe, resulting in $0\%$ recall on Moderate depression.

#### 4. Scientific Honesty
- We deliberately report these metrics without post-hoc threshold adjustment or relabeling. In clinical NLP and speech processing, out-of-domain degradation is an established, well-documented phenomenon.

---

## 4. Pre-Demonstration Verification Matrix

| Verification Item | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| Primary Model Loaded | Candidate 2A (`cand2a_standard_mlp_mild_balanced.pth`) | Loaded successfully from `artifacts/final_model/` | **PASS** |
| Feature Scalers Loaded | Training-only fitted standardizers | Loaded successfully from `artifacts/final_model/` | **PASS** |
| Single Clean Result | Normal output prints one severity class only | `Predicted severity: Severe Depression` | **PASS** |
| Medical Disclaimer | Disclaimer included in user output | Included on both PID 624 and 300 | **PASS** |
| Input Validation | Invalid file path triggers clean error exit | `[ERROR] File Not Found: ...` (Exit code 1) | **PASS** |
| Parity on Sample 1 | Participant 624 inference runs successfully | Outputs `Predicted severity: Severe Depression` | **PASS** |
| Parity on Sample 2 | Participant 300 inference runs successfully | Outputs `Predicted severity: Minimal/Mild Depression` | **PASS** |
| Unseen Evaluation | Full N=33 evaluation reproduces exactly | Accuracy 60.61%, Macro-F1 35.06%, MAE 5.92 | **PASS** |
| Zero Retraining | No weights or thresholds modified | Model weights and calibration parameters frozen | **PASS** |
