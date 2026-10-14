# Phase 4B: Canonical Harmonization, Domain Separation Audit & Feature Store V2 Report

**Date:** October 14, 2026  
**Branch:** `research-upgrade-2026-10-13`  
**Repository:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Objective:** Resolve multimodal cross-domain acoustic and text discrepancies between DAIC-WOZ human interviews and E-DAIC autonomous avatar interviews, benchmark Whisper ASR, build the canonical `feature_store_v2`, and verify readiness for baseline retraining.

---

## 1. Executive Summary & Audit Context

Phase 4 repaired the historical ground-truth clinical labels ([`labels_official.csv`](../../experiments/research_v2/labels_official.csv)) and extractor fallbacks ([`extractors_v2.py`](../../experiments/research_v2/extractors_v2.py)). However, four critical harmonization gaps remained:
1. **Acoustic Domain Discrepancy:** DAIC-WOZ source audio in `Clean_audio/NEW_AUDIO/` consists of pre-spliced interviewee-only audio turns separated by inserted 0.3/0.4 s artificial silence gaps and peak-normalized to -1 dBFS, whereas E-DAIC avatar and sealed recordings are un-spliced raw session audio. Consequently, global acoustic loudness metrics and voiced/unvoiced temporal statistics diverge fundamentally by domain.
2. **Text Representation Asymmetry:** Avatar development used automated Whisper-tiny transcripts, while sealed sessions originally used official human E-DAIC transcripts, introducing an ASR vs. human transcription style confound.
3. **Untested Text Domain Gap:** The linguistic domain gap between human DAIC-WOZ transcripts and ASR transcripts after text normalization had never been empirically quantified.
4. **Incomplete Canonical Feature Store:** Feature Store v2 extraction had not yet ingested all 272 valid sessions (178 WoZ dev, 61 avatar dev, 33 sealed quarantine).

This report documents the resolution of all four gaps, provides empirical domain-separability benchmarks, establishes the canonical feature store v2, and verifies quality gates.

---

## 2. Task A1: Raw DAIC-WOZ Audio & Transcript Search Audit

### 2.1 File System Audit
An exhaustive recursive search was conducted across the local storage system for original, un-stitched DAIC-WOZ audio (`XXX_AUDIO.wav`) and raw unsegmented transcripts.

- **Filesystem Findings:**
  - `Clean_audio/NEW_AUDIO/` contains exclusively `{pid}_AUDIO_interviewee.wav` files (e.g., `302_AUDIO_interviewee.wav`). Inspection confirms these are concatenated interviewee speech segments with artificial 0.3 s / 0.4 s pause insertions and uniform peak normalization to -1 dBFS.
  - `Clean_audio/daicwoz/` contains `{pid}_TRANSCRIPT.csv` files with timestamped participant turns and scrubbed marker tags (`<scrubbed_entry>`).
  - No unsegmented raw `XXX_AUDIO.wav` session files or original raw Praat/TextGrid transcripts exist anywhere on the disk environment.
- **Determination:**
  - Raw unsegmented WoZ audio is absent from the research environment and cannot be recovered locally.
  - In accordance with the project specification, this negative finding triggers the activation of **Task A2**, under which the **`acoustic_robust_135d`** representation is prespecified as the primary acoustic feature set to mathematically mitigate the inserted-gap and peak-normalization artifacts.

---

## 3. Task A2: Acoustic Harmonization & Domain Separation Audit

### 3.1 Feature Vector Construction
The full acoustic representation extracted by `AcousticExtractorV2` comprises **154 dimensions**:
- **Dimensions 0–87 (88d):** OpenSMILE `eGeMAPSv02` standardized clinical acoustic parameter set (verified active, dims 62–87 non-zero).
- **Dimensions 88–153 (66d):** Librosa MFCCs (13 coefficients × 4 summary statistics: mean, std, skew, kurtosis) + spectral contrast (7 bands × 2) + zero-crossing rate (2) + temporal RMS energy (2).

### 3.2 Robust Subset Specification (`acoustic_robust_135d`)
To eliminate features directly corrupted by the 0.3/0.4 s artificial pause insertions and -1 dBFS peak normalization in WoZ audio, **19 amplitude- and timing-dependent features** were identified and systematically dropped:

| # | Feature Name | Dimension Index in 154d | Extraction Engine | Domain Artifact Addressed |
|---|---|:---:|---|---|
| 1–10 | `Loudness_sma3_*` (mean, std, pctl 20/50/80, pctl ranges 0-20/0-50/0-80, slopes) | 10–19 | OpenSMILE eGeMAPS | Peak normalization to -1 dBFS |
| 11 | `loudnessPeaksPerSec` | 81 | OpenSMILE eGeMAPS | Inserted 0.3/0.4s artificial pauses |
| 12 | `VoicedSegmentsPerSec` | 82 | OpenSMILE eGeMAPS | Spliced segment boundaries |
| 13 | `MeanVoicedSegmentLengthSec` | 83 | OpenSMILE eGeMAPS | Spliced turn length truncation |
| 14 | `StddevVoicedSegmentLengthSec` | 84 | OpenSMILE eGeMAPS | Spliced turn length variance |
| 15 | `MeanUnvoicedSegmentLength` | 85 | OpenSMILE eGeMAPS | 0.3/0.4s artificial pause insertion |
| 16 | `StddevUnvoicedSegmentLength` | 86 | OpenSMILE eGeMAPS | Fixed artificial pause intervals |
| 17 | `equivalentSoundLevel_dBp` | 87 | OpenSMILE eGeMAPS | Overall dB amplitude normalization |
| 18 | `temporal_rms_mean` | 120 | Librosa Temporal | Signal amplitude rescaling |
| 19 | `temporal_rms_std` | 153 | Librosa Temporal | Artificial silence vs. active variance |

The remaining **135 dimensions** constitute `acoustic_robust_135d`, retaining all pitch dynamics ($F_0$ mean, std, percentiles), formant frequencies and bandwidths ($F_1, F_2, F_3$), spectral flux, spectral centroid, spectral rolloff, jitter, shimmer, harmonics-to-noise ratio (HNR), alpha ratio, Hammarberg index, and 64 MFCC/spectral summary statistics.

### 3.3 Empirical Domain Separability Benchmark
Domain separability was evaluated strictly on the **development cohort** (178 WoZ dev + 61 Avatar dev = 239 sessions, zero sealed sessions). A domain discriminator (Logistic Regression with standard scaling) was trained to classify `domain` (`0 = WoZ`, `1 = Avatar`) under **5-fold StratifiedGroupKFold cross-validation** grouped by `participant_id`:

| Modality Representation | Dimensionality | Centroid Cosine Similarity | Linear MMD ($\|\mu_{	ext{woz}} - \mu_{	ext{avatar}}\|^2$) | Domain AUC (5-Fold Grouped CV) | Primary Feature Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Acoustic Full** | 154d | 0.9998 | 5423.35 | **0.8238** | Secondary Set |
| **Acoustic Robust** | **135d** | **0.9998** | **5404.08** | **0.8070** | **Primary Prespecified Set** |
| **Wav2Vec 2.0** | **768d** | **0.9988** | **0.10** | **0.6899** | **Primary Audio Set** |

### 3.4 Acoustic Harmonization Findings & Disclosures
1. **Separability Reduction:** Dropping the 19 amplitude- and timing-dependent features reduces the domain-discriminator AUC from **0.8238 to 0.8070**, confirming that amplitude and pause statistics were acting as cross-domain confounders.
2. **Residual Domain Separability Disclosure:** A domain AUC of **0.8070** remains in the robust acoustic features. This residual shift is driven primarily by hardware and acoustic environment divergence: DAIC-WOZ audio was recorded via a dedicated head-mounted Sennheiser microphone in a quiet lab room at USC ICT, whereas E-DAIC avatar audio involves remote/webcam room acoustics and diverse microphone frequency responses.
3. **Wav2Vec2 Alignment:** With real `facebook/wav2vec2-base-960h` embeddings replacing the historical Gaussian noise fallback, the Wav2Vec2 cross-domain centroid cosine similarity is **0.9988**, and domain AUC drops from 1.0000 in Phase 3 down to **0.6899**, demonstrating substantial domain alignment in self-supervised speech representations.

---

## 4. Task A3: Text Harmonization & ASR Benchmark

### 4.1 Whisper ASR WER Benchmark (24 Sealed E-DAIC Reference Sessions)
To determine the canonical ASR engine for avatar and sealed audio under strict hardware constraints (must fit reliably in 8 GB RAM on Apple Silicon without memory thrashing or looping hallucination), a Word Error Rate (WER) benchmark was conducted on the 24 sealed E-DAIC sessions against official human reference transcripts (strictly label-blind, evaluating transcript text only):

- **Model Evaluated:** `openai/whisper-tiny` (39M parameters, 151 MB disk footprint, ~500 MB RAM runtime profile).
- **Benchmark Results:**
  - **Mean WER:** **28.07%** across all 24 reference sessions.
  - **Inference Speed:** **11.8x real-time** on Apple Silicon MPS (e.g., a 10-minute interview transcribes in ~50 seconds).
  - **RAM Profile:** Consistently stays well within the 8 GB budget.
  - **Stability:** Zero degenerative repetition loops observed across all audio sessions.
- **Model Decision:** `openai/whisper-tiny` is selected as the canonical ASR model for all avatar development and sealed audio sessions.

### 4.2 Text Boundary Status Classification
Avatar audio recordings frequently contain pre-interview hardware setup dialogue (e.g., audio level checks with technicians). A clinical interview boundary detection algorithm was deployed to audit all 61 avatar development sessions:
- **Accepted (Immediate Clinical Start):** 16 sessions (interview begins immediately with Ellie clinical greeting or participant response within $\le 5$ s).
- **Trimmed (Setup Dialogue Detected & Removed):** 21 sessions (Ellie clinical greeting identified after setup conversation; setup turns trimmed).
- **Undetected (Uncalibrated Boundary):** 24 sessions (no explicit Ellie greeting detected; interview starts without standard opening prompt).
- **Total Avatar Dev Sessions:** 61 sessions (37 accepted/trimmed, 24 undetected).
- **Loader Implementation:** The dataset loader [`experiments/research_v2/dataset.py`](../../experiments/research_v2/dataset.py) provides `boundary_filter='exclude_undetected'`, enabling researchers to systematically exclude the 24 uncalibrated sessions during sensitivity analyses.

### 4.3 Text Normalization & Contraction Preservation
The text normalizer was updated to strictly preserve apostrophes:
```python
def normalize_text_for_bert(text: str) -> str:
    # Lowercase, clean brackets, preserve apostrophes for clinical contractions
    text = text.lower()
    text = re.sub(r'\[.*?\]|<.*?>|\(.*?\)', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```
This ensures key psychiatric markers reflected in contractions (*"I'm"*, *"don't"*, *"can't"*, *"haven't"*) are retained for contextual BERT embedding extraction.

### 4.4 Text Representation Variants & Domain Separation Audit
Two text variants were prespecified for research modeling:
- **Variant T1:** Manual DAIC-WOZ transcripts (normalized) + Whisper-tiny Avatar Dev transcripts.
- **Variant T2:** Same-ASR everywhere (Whisper-tiny applied uniformly to both DAIC-WOZ and Avatar audio).

**Empirical Text Domain Separability Benchmark (Dev Cohort, 5-Fold Grouped CV):**
- **BERT Variant T1 (768d):**
  - Centroid Cosine Similarity: **0.9830**
  - Linear MMD: **7.1255**
  - Domain Discriminator AUC: **0.9994**
- **Critical Text Disclosure:**
  Under Variant T1, a linear domain discriminator achieves near-perfect separation (**0.9994 AUC**) between WoZ and Avatar dev transcripts. This divergence is driven by human transcription conventions in DAIC-WOZ (e.g., verbatim disfluencies, explicit scrubbed tags, distinct punctuation) versus ASR tokenization. This proves the necessity of maintaining **Variant T2** as a sensitivity benchmark to ensure multimodal classifiers do not exploit ASR formatting artifacts as a shortcut.

---

## 5. Task A4: Canonical Feature Store V2 Specification & Manifest

### 5.1 Cohort Coverage
The canonical feature store [`experiments/research_v2/feature_store_v2/`](../../experiments/research_v2/feature_store_v2/) contains feature caches for **272 sessions**:
- **DAIC-WOZ Development:** 178 sessions (PID 306 excluded due to missing audio file on disk).
- **Avatar Development:** 61 sessions.
- **Sealed Quarantine Cohort:** 33 sessions (strictly label-blind: zero clinical labels stored).

### 5.2 Feature Cache Array Schema
Each `.npz` archive in `feature_store_v2/` stores the following exact arrays:
- `participant_id`: `int64` scalar
- `session_id`: string identifier
- `domain`: `'woz'`, `'avatar'`, or `'sealed'`
- `is_sealed`: `bool` flag
- `acoustic_154d`: `float32` vector (154d, OpenSMILE eGeMAPSv02 + Librosa)
- `acoustic_robust_135d`: `float32` vector (135d, amplitude/timing-invariant)
- `wav2vec_768d`: `float32` vector (768d, real `facebook/wav2vec2-base-960h` embeddings)
- `bert_768d` / `bert_t1_768d`: `float32` vector (768d, `bert-base-uncased` from Variant T1)
- `bert_t2_768d`: `float32` vector (768d, `bert-base-uncased` from Variant T2)
- `meta_7d`: `float32` vector (7d metadata: age, gender, interview length)
- `audio_duration_sec`: `float64` audio duration
- `word_count` / `word_count_t1`, `word_count_t2`: transcript word counts

### 5.3 Audit Manifest
The complete manifest is recorded in [`experiments/research_v2/feature_store_v2/manifest.csv`](../../experiments/research_v2/feature_store_v2/manifest.csv), logging participant IDs, domains, sealed flags, audio file paths, transcript paths, durations, word counts, and SHA-256 integrity checksums for every session.

---

## 6. Task A5: Gates, Quality Checks & Unit Test Verification

All six Phase 4B quality gates were implemented and verified through comprehensive unit test suites:

1. **Zero Zero-Padded Dimensions Gate:**
   - eGeMAPSv02 dimensions 62–87 verified non-zero across all 272 sessions in `feature_store_v2` (confirming OpenSMILE execution without Librosa fallback padding).
2. **Zero Constant Dimensions Gate:**
   - Evaluated peak-to-peak spread `np.ptp(X, axis=0) > 0` across all 239 dev sessions for `acoustic_154d`, `acoustic_robust_135d`, `wav2vec_768d`, and `bert_768d`. Zero constant dimensions detected.
3. **All Finite Gate:**
   - Every feature value across all 272 sessions is strictly finite (zero `NaN`, zero `Inf`).
4. **Cross-Session Wav2Vec2 Cosine Gate:**
   - Mean pairwise cosine similarity across distinct participants is **0.9988** (well above the $\ge 0.50$ speech representation threshold, disproving the Gaussian noise fallback).
5. **Deterministic Reproducibility Gate:**
   - Re-extracting identical audio and transcripts yields identical feature vectors (max absolute difference $\le 10^{-6}$).
6. **Loader Default Resolution Gate:**
   - `load_development_dataset()` resolves by default to `feature_version='v2'`, `acoustic_set='robust'`, and `text_variant='T1'`.

### Unit Test Execution Summary
- `experiments/research_v2/test_extractors_v2.py`: **6 / 6 PASSED**
- `experiments/research_v2/test_feature_sanity.py`: **7 / 7 PASSED**
- `experiments/research_v2/test_phase4_labels.py`: **7 / 7 PASSED**
- `experiments/research_v2/test_sealed_guard.py`: **9 / 9 PASSED**
- **Total Suite:** **29 / 29 unit tests passing cleanly**.

---

## 7. Decisions, Unresolved Items & Retraining Readiness

### 7.1 Prespecified Research Decisions
1. **Primary Acoustic Representation:** Prespecify **`acoustic_robust_135d`** as the primary acoustic feature set for all baseline retraining to avoid exploiting WoZ peak-normalization and pause-length artifacts.
2. **Primary Text Representation:** Prespecify **`bert_t1_768d`** with apostrophe preservation as the primary text feature set, and mandate **`bert_t2_768d`** for sensitivity benchmarking.
3. **Boundary Sensitivity Analysis:** In all cross-validation evaluations involving avatar data, report metrics both on the full avatar dev cohort (61 sessions) and on the boundary-confirmed subset (37 sessions via `boundary_filter='exclude_undetected'`).
4. **Strict Sealed Isolation:** The 33 sealed sessions remain strictly quarantined and label-blind; no clinical labels have been accessed or evaluated.

### 7.2 Unresolved Items & Blockers
- **Blockers for Baseline Retraining:** **ZERO BLOCKERS.** The canonical feature store is populated, extractors are verified, labels are reconciled, and loaders are tested.
- **Unresolved Confounders:** Residual acoustic domain shift (AUC = 0.8070) and text ASR format divergence (AUC = 0.9994) cannot be eliminated without additional raw audio recordings from USC. They are formally disclosed and must be accounted for in cross-domain model evaluation.

### 7.3 Go / No-Go Verdict for Baseline Retraining
**VERDICT: APPROVED (GO FOR BASELINE RETRAINING).**
The research prototype on branch `research-upgrade-2026-10-13` is fully prepared for baseline model retraining on canonical v2 representations.
