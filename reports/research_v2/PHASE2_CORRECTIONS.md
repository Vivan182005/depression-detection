# PHASE 2 CORRECTIVE AUDIT REPORT — COHORT, TRIMMING, GUARD, AND STATISTICAL AUDIT

**Repository:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Branch:** `research-upgrade-2026-10-13`  
**Date:** 2026-10-13  
**Status:** COMPLETE (Corrective Audit Executed, Defect Fixes Tested, Statistical Assertions Corrected)  
**Recommendation:** CONDITIONAL GO FOR PHASE 3 (Strictly Awaiting User Review)

---

## 1. Executive Summary & Non-Negotiable Guardrails

This report documents the **Phase 2 Corrective Audit** conducted inside the isolated research copy. As mandated:
1. **Production repository** (`/Users/vivannaik/Desktop/CAPSTONE`) and **Candidate 2A artifacts** (`artifacts/final_model/*`) remain completely untouched and frozen.
2. The original `reports/research_v2/PHASE2.md` has **not** been rewritten or sanitized to conceal uncertainties; all newly confirmed findings, failure modes, and statistical corrections are recorded independently here in `reports/research_v2/PHASE2_CORRECTIONS.md`.
3. No models were trained, tuned, or evaluated in this audit.

---

## 2. Task 1: Audit of Cohort Counts and Manifest Integrity

### 2.1 Direct Recomputation from Raw Files (=275$)
Every participant entry in `new_audio_csv/detailed_lables.csv` was verified against raw filesystem assets:
- **Total Catalogue Entries:** Exactly **275 unique PIDs**.
- **Label Coverage:** 274 participants have complete clinical PHQ-8 items; 1 participant (PID 458) has null severity.
- **Audio Coverage:** 272 participants have verified uncorrupted WAV audio files on disk (187 in `Clean_audio/`, 61 in `NEW_AUDIO_DAIC/`, 33 in `unseen/` with 9 intentional duplicates). 3 participants lack raw audio (PID 306 has cached features; PIDs 458 and 602 have no audio on disk).
- **Transcript Coverage:** 189 Wizard-of-Oz sessions have official turn-annotated human transcripts in `Clean_audio/daicwoz/`. The 61 avatar expansion sessions are transcribed via our local Whisper pipeline into `experiments/research_v2/transcripts_avatar_dev/`.

### 2.2 Complete Auditable Cohort Partition
The complete auditable participant list is exported in [`experiments/research_v2/participant_roster_audit.csv`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/participant_roster_audit.csv):

| Cohort Role | Domain | Provenance | Total PIDs | Included in Dev | Audio Status | Label Status | Explicit Roster Reason |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **`dev_active_woz`** | Human WoZ | DAIC-WOZ 2016 | **179** | **True** | 178 on-disk + 1 cached (306) | Valid PHQ-8 (179) | Active Wizard-of-Oz dev store participants matching historical index. |
| **`avatar_candidate_expansion`** | Autonomous Avatar | E-DAIC AVEC 2019 | **61** | **True** | 61 on-disk | Valid PHQ-8 (61) | Eligible non-sealed avatar interviews (48 in 600s + 13 in 700s). |
| **`sealed_quarantine`** | Mixed | Combined Holdout | **33** | **False** | 33 on-disk | Quarantined | Exact 33 PIDs in `sealed_ids.json` strictly quarantined for Phase 6 only. |
| **`excluded_missing_data`** | Mixed | Missing Data | **2** | **False** | Missing (0) | 1 Valid (602), 1 Null (458) | Excluded: PID 458 (audio missing on disk), PID 602 (audio missing on disk). |
| **TOTAL** | — | — | **275** | **240** | **272** | **274** | **100% Partitioned with Zero Overlap.** |

### 2.3 Verification of Disjointness & Sealed IDs
- **Disjointness:** $\{dev\_active\_woz\} \cap \{avatar\_candidate\_expansion\} = \emptyset$ (verified in code and unit tests).
- **Sealed Quarantine:** Exactly matches the canonical 33 PIDs in `sealed_ids.json`. Zero sealed IDs appear in development data, feature store indexing, or training manifests.
- **Audio Hash Checks:** MD5 hashing of all 281 WAV files confirms **zero unexpected duplicates**. Exactly 9 identical hashes exist between `Clean_audio/NEW_AUDIO/` and `unseen/`, reflecting the original mirroring of the 9 WoZ sealed participants.
- **Speaker Re-Enrollment Caveat:** While identical file hashes across cohorts are ruled out, matching hashes **cannot rule out the same physical individual appearing under different participant IDs across different years** (e.g., if a 2014 participant re-enrolled in 2017/2019). This cannot be verified without biometric voiceprint identification, and they are treated as independent sessions per the AVEC challenge standard.

---

## 3. Task 2: Transcript and Trimming Validation Audit

### 3.1 Raw Transcripts Preservation and Trimming Architecture
- **Raw Transcripts Preserved:** All raw, continuous Whisper transcripts remain intact in [`experiments/research_v2/transcripts_avatar_dev/`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/transcripts_avatar_dev) and are **never overwritten**.
- **Separately Versioned Trimmed Transcripts:** Version-separated trimmed transcripts are written to [`experiments/research_v2/transcripts_avatar_dev_trimmed/`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/transcripts_avatar_dev_trimmed) only for sessions with validated start boundaries.
- **Auditable Manifest:** Every decision is logged in [`experiments/research_v2/trimming_audit_manifest.csv`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/trimming_audit_manifest.csv).

### 3.2 Manual Inspection of Avatar Audio/Transcript Pairs
Manual inspection across completed avatar sessions revealed crucial real-world failure modes:
1. **Failure of Simple Greeting Detection:** The virtual agent Ellie speaks through room speakers or headphones. In lapel microphone recordings, **Ellie's voice is frequently not captured on the participant's microphone**, or participant silence during Ellie's intro produces zero transcript tokens. Thus, regex searching for *"Hi, I'm Ellie"* failed in ~85–90% of sessions despite legitimate interviews.
2. **Technician Setup Dialogue:** In ~30–40% of sessions (e.g., PIDs 612, 660, 661, 664), the opening 15–90 seconds record the research assistant adjusting the screen, testing the microphone, or explaining the virtual agent (*e.g., "there she is", "shrink her back down", "she looks like the Sims", "frozen or something"*).
3. **Immediate Clinical Responders:** In ~50% of sessions (e.g., PIDs 608, 666, 702, 703), the recording begins after setup, and the participant immediately answers clinical depression questions (*e.g., "Summer to Dino, California", "Yes, good, down to beach", "I'm fine, a little tired"*). Blindly cutting an arbitrary 60s would discard critical clinical answers.
4. **Whisper Diarization Reality:** Whisper transcribes all audible speech into text; **it does not perform speaker diarization**. We explicitly do not claim participant-only separation in the avatar audio or transcripts.

### 3.3 Trimming Classification & Defers Across 35 Audited Sessions
Using our multi-pattern transition auditor ([`experiments/research_v2/validate_transcripts_and_trimming.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/validate_transcripts_and_trimming.py)):

| Trimming Status | Description | Count (=35$) | % of Cohort | Defer Participant-Only Extraction? |
| :--- | :--- | :---: | :---: | :---: |
| **`accepted_immediate_start`** | Immediate clinical response at 0.0s; no setup phrases detected. | **18** | 51.4% | False (Safe to use full audio/text) |
| **`accepted_setup_trimmed`** | Setup dialogue or Ellie intro detected; clean transition boundary found. | **5** | 14.3% | False (Safe to use trimmed audio/text) |
| **`uncertain`** | Setup dialogue present, but transition boundary is ambiguous. | **4** | 11.4% | **True (DEFERRED)** |
| **`not_detected`** | Conversational banter or silent opening; neither setup nor greeting found. | **8** | 22.9% | **True (DEFERRED)** |
| **TOTAL** | — | **35** | **100%** | **12 Deferred (34.3%)** |

> [!IMPORTANT]
> **Defensive Extraction Policy:**  
> For the **12 sessions (34.3%)** flagged as `uncertain` or `not_detected`, participant-only feature extraction is **strictly deferred**. They will not be fed to fine-grained participant-only feature estimators without confirmed boundary validation.

---

## 4. Task 3: Sealed-Cohort Guard Deep Audit

### 4.1 Development Loader Entry Point Audit
We audited all loader paths in `experiments/research_v2/dataset.py` and `run_phase1_harness.py`:
- **Defect Identified:** In `dataset.py`, `clean_labels_path` was reading `new_audio_csv/detailed_lables.csv` (which contains all 275 rows) into `clean_map` before checking `df['participant_id']`. While `df` was guarded, the in-memory `clean_map` dictionary held ground-truth labels for the 33 sealed participants.
- **Defect Fixed:** In `experiments/research_v2/dataset.py`, `clean_df` is now explicitly filtered against `SEALED_PIDS` at load time:
  ```python
  clean_df = clean_df[~clean_df['Participant'].isin(SEALED_PIDS)]
  clean_map = dict(zip(clean_df['Participant'], clean_df['Depression_severity']))
  ```
  This guarantees that `clean_map` never contains a single sealed participant ID or label in memory.

### 4.2 Expanded Unit Test Verification (/9$ Tests Passing)
In [`experiments/research_v2/test_sealed_guard.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/test_sealed_guard.py), we added direct integration tests testing the actual entry points:
- `test_entrypoint_load_development_dataset_zero_sealed`: Verifies that `load_development_dataset()` output has exactly 0 sealed IDs in `pids`, `sids`, or any feature array.
- `test_entrypoint_rejects_injected_sealed_id`: Verifies that attempting to pass sealed PIDs (300, 600, 624, 638, 414) immediately triggers a `RuntimeError`.
- `test_entrypoint_clean_map_sealed_exclusion`: Verifies that querying any sealed PID against the sanitized label dictionary raises `KeyError`.

---

## 5. Task 4: Demographic Domain-Shift Plan (Development Data Only)

### 5.1 Development-Only Demographic Breakdown (=240$)
To ensure the blind boundary is preserved, **no sealed cohort labels or evaluation outcomes are used to formulate the development mitigation plan**.

On development data only (=240$ participants):
| Domain | Total N | Female (%) | Male (%) | Unknown | Age Mean $\pm$ Std | Missingness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`human_woz`** | 179 (74.6%) | **79 (44.1%)** | **100 (55.9%)** | 0 | .9 \pm 12.4$ | 0% |
| **`autonomous_avatar`** | 61 (25.4%) | **5 (8.2%)** | **55 (90.2%)** | 1 (1.6%) | .4 \pm 12.0$ | 0% |
| **Combined Development** | **240 (100%)** | **84 (35.0%)** | **155 (64.6%)** | **1 (0.4%)** | **.3 \pm 12.9* | **0%** |

### 5.2 Subgroup Analysis Discipline & Methodological Constraints
1. **Sample Size Reality of Female Avatars (=5$):** The female avatar cohort in development comprises only **5 individuals**. Any subgroup metric computed on =5$ has massive sampling variance (a single error shifts accuracy by 20 percentage points). Therefore, female avatar performance on development data must be treated as **purely exploratory and descriptive**, never as a primary optimization target.
2. **No Demographic Downsampling or Pruning:** We will **not** discard eligible male avatar participants or artificially rebalance development data to match sealed demographics. Discarding eligible data would degrade model sample efficiency.
3. **Domain Shift is Multi-Faceted:** Gender distribution is only one component of domain shift. DAIC-WOZ and E-DAIC differ fundamentally in conversational dynamics (adaptive human follow-ups vs fixed prompt trees), acoustic channel (headset vs lapel microphones, speaker bleed), and duration (.7$ min vs .7$ min). **We do not claim that gender stratification alone will resolve the domain mismatch.**

---

## 6. Task 5: Correction of Phase 1 Statistical Claims

### 6.1 Rechecking Focal Loss vs Baseline under Clustered Resampling
Using the saved pooled out-of-fold (OOF) predictions in `experiments/research_v2/oof/`, we audited the paired difference between Focal Loss and Candidate 2A across 10,000 participant-clustered bootstrap resamples (=179$ clusters):

| Comparison | Metric | Baseline | Candidate | Observed Delta | 95% Clustered CI (=179$) | Nominal p-value | Bonferroni p-value (=16$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Loss_Focal_Gamma15` vs Base** | **Balanced Acc** | **42.02%** | **48.13%** | **+6.11%** | **[+1.01%, +11.97%]** | **0.0118** | **0.1888 (p > 0.05)** |
| **`Loss_Focal_Gamma15` vs Base** | **Macro-F1** | **42.04%** | **46.90%** | **+4.86%** | **[+0.28%, +10.06%]** | **0.0370** | **0.5920 (p > 0.05)** |
| **`Loss_Focal_Gamma20` vs Base** | **Balanced Acc** | **42.02%** | **45.96%** | **+3.94%** | **[-0.83%, +9.56%]** | **0.1146** | **1.0000 (p > 0.05)** |
| **`Loss_Focal_Gamma20` vs Base** | **Macro-F1** | **42.04%** | **44.62%** | **+2.58%** | **[-1.72%, +7.60%]** | **0.2752** | **1.0000 (p > 0.05)** |

### 6.2 Key Statistical Corrections
1. **Significance is Strictly Exploratory:** While Focal Loss ($\gamma = 1.5$) achieves a nominal  = 0.0118$ unadjusted, **after Bonferroni correction for the 16 exploratory pairwise comparisons evaluated in Phase 1, the p-value is  = 0.1888$ (not significant at $lpha = 0.05$)**. The focal loss result is therefore classified as **exploratory and hypothesis-generating**, requiring rigorous confirmation on the reserved confirmation folds (`folds_confirmation.json`, seeds 100–104).
2. **Validation Cohort Alignment Verified:** Both Candidate 2A, Focal Loss, and the Majority Class Baseline were verified on the **exact same 183 sessions / 179 participants** (100% participant alignment).
3. **Repeated-Seed Independence Audit:** The bootstrap resamples participant clusters (=179$), not individual session predictions. Predictions for repeated sessions or seeds are never treated as independent participant observations.
4. **Metadata Baseline Confound, Not Leakage:** The metadata-only baseline (45.78% balanced accuracy from turn counts and session duration) is a behavioral clinical confound (depressed individuals have slower speech rates and shorter turns), **not proof of feature leakage or prohibited data paths**.

---

## 7. Verification & Automated Integrity Tests (/19$ Passing)

The complete test suite was executed:
```bash
python3 -m unittest experiments/research_v2/test_phase2_manifest.py experiments/research_v2/test_sealed_guard.py
```
**Result:**
```
...................
----------------------------------------------------------------------
Ran 19 tests in 5.642s

OK
```

---

## 8. Artifact Ledger (Files Changed & Created)

| Artifact Path | Description |
| :--- | :--- |
| `experiments/research_v2/participant_roster_audit.csv` | Complete 275-participant auditable roster with inclusion flags and explicit reasons. |
| `experiments/research_v2/validate_transcripts_and_trimming.py` | Transcript auditor classifying start boundaries and version-separating trimmed transcripts. |
| `experiments/research_v2/trimming_audit_manifest.csv` | Trimming metadata manifest tracking status, detection rules, confidence, and defer flags. |
| `experiments/research_v2/transcripts_avatar_dev_trimmed/` | Directory containing version-separated trimmed transcripts (raw transcripts preserved). |
| `experiments/research_v2/dataset.py` | Fixed development data loader to strip sealed PIDs from in-memory label dictionary. |
| `experiments/research_v2/test_sealed_guard.py` | Expanded with loader entrypoint integration tests (9 tests). |
| `experiments/research_v2/test_phase2_manifest.py` | Expanded with roster audit and trimming integrity tests (10 tests). |
| `reports/research_v2/PHASE2_CORRECTIONS.md` | This comprehensive corrective audit report. |

---

## 9. Go / No-Go Decision

### Decision: **CONDITIONAL GO FOR PHASE 3 (Awaiting User Review)**

**Scientific Justification:**
1. All cohort counts (=275$) and manifest relationships are 100% verified and auditable.
2. The sealed quarantine boundary (=33$) is reinforced against in-memory label leaks and validated via integration tests.
3. Trimming failure modes are documented; raw transcripts are preserved untouched; and uncertain sessions (.3\%$) are explicitly deferred from participant-only extraction.
4. The statistical claims for Focal Loss have been corrected: its nominal gain is confirmed exploratory ({	ext{adj}} = 0.1888$), not a proven winner prior to confirmation folds.
5. All 19 tests pass without weakening any guards.

*End of Phase 2 Corrective Audit Report. Stopping execution and awaiting user review.*