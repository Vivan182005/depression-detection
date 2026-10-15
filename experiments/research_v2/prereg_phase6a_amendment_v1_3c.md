# Pre-Registration Amendment (Version 1.3c): S3 Gate Remediation & Pre-Outcome Code Corrections
## Voice-Primary Depression Detection (Audio Primary, Text Secondary)

**Protocol Amendment:** v1.3c (Supplements v1.1, v1.2, v1.3, v1.3b)
**Date:** 2026-10-15
**Repository Branch:** `research-upgrade-2026-10-13`
**Execution Gate:** LOGGED BEFORE ANY REAL-LABEL MODEL RUN (Step E1 not yet started)
**Outcome Status:** ZERO REAL-LABEL PREDICTIVE OUTCOMES SEEN

---

## 1. Trigger

The v1.3b S3 transcript audit (`transcript_qa_extended_report.json`) halted execution:
42 / 239 sessions (17.57 %; 32 / 61 Avatar, 10 / 178 WoZ) contained Whisper-tiny
repetition loops (10-gram repeated >= 5 times), exceeding the pre-stated 5 % stop rule.
All other gates passed (0 near-empty transcripts, median WoZ WER 0.197, batch AUC 0.524).

Root cause: the HF pipeline was fed whole sessions with fixed 30 s windows. Windows
dominated by silence/background noise send the autoregressive decoder into loops.

## 2. Remediation (Option 3 + Option 1, applied uniformly to all 239 sessions)

Script: `transcribe_vad_chunked.py`. Every development session is re-transcribed with
one identical procedure (no subset treatment, so no new batch/domain artefact):

| Parameter | Value |
| :--- | :--- |
| ASR model | `openai/whisper-tiny`, greedy (`do_sample=False`, `num_beams=1`), `language='english'`, `task='transcribe'` |
| Speech regions | Frozen v1.2 Silero VAD (threshold 0.50, min speech 250 ms, min silence 200 ms) |
| Chunking | Consecutive VAD segments packed into chunks <= 28 s; a gap > 2.0 s starts a new chunk; each chunk decoded independently |
| Loop fallback | A chunk whose decode contains any n-gram (n <= 10) repeated > 3 times consecutively is re-decoded segment-by-segment |
| Post-filter | Deterministic collapse: any n-gram (n = 1..10) repeated > 3 times consecutively is truncated to 3 repetitions |

The same S3 gates (v1.3b sec. 4) are re-applied by `audit_s3_vad_and_update_t2.py`. Only if every
gate passes is `bert_t2_768d` rebuilt from the new transcripts (previous embeddings archived as
`bert_t2_fulllen_768d`, unused by any variant). V14/V18/V19 definitions are otherwise unchanged.

All parameters were fixed from ASR behaviour on three loop-affected sessions (PIDs 679, 710, 432)
and one clean session (301). No label, label-derived quantity or classifier output was consulted.

## 3. Pre-Outcome Code Corrections (found during review, before any real-label run)

| # | File | Defect | Correction |
| :--- | :--- | :--- | :--- |
| C1 | `data_loader.py` | `labels_official.csv` stores gender as `'male'`/`'female'`; V16 (`astype(float32)`) crashed and subgroup endpoints compared against 0/1, so the E1 dry-run could never complete | Gender mapped to female=0 / male=1 with a hard error on unknown values; `phq_score` column added (runner expected it; source column is `phq8_score`) |
| C2 | `run_phase6a_experiment.py`, `run_dryrun_permuted.py` | Outer folds were stratified on domain x gender, contradicting v1.1 sec. 5.1 ("stratified by binary label") | `strata=None`, so `run_pipeline` stratifies on the label it is given (observed labels for the observed run, permuted labels for each null replication, as v1.1 sec. 6.1 requires) |
| C3 | `pipeline_engine.py` (V19) | Candidate keys are lower-case (`'v2'`), so `'V2' in cand` never matched and V2/V4 winsorisation was skipped inside V19 | Compare `cand.upper()` against `('V2', 'V4')` |
| C4 | `run_phase6a_experiment.py` | The v1.1 sec. 6.2 R=5 feasibility gate was not implemented | R=5 timing for V2 and V10 before R=999; H1/H2 reclassified as exploratory if the projection exceeds 48 h |
| C6 | `pipeline_engine.py` (V12/V13 segment MLP) | Segment hand-crafted features are NaN by design on turns failing the v1.2 validity thresholds (present in all 239 sessions), but the MLP path never imputed them, so V12 produced NaN probabilities and crashed (found by a permuted-label smoke test) | In-fold median imputation computed on training-fold segments only, applied to train and validation segments (v1.1 sec. 5.2) |
| C5 | `transcribe_remaining_woz.py` (historical) | Metadata records `condition_on_previous_text=False`, but that argument was never passed to the pipeline | Documented only; superseded by Sec. 2 |

## 4. Integrity Log / Deviations

1. **Aggregate label exposure (2026-10-15, during code review):** while checking how gender is
   coded, one diagnostic printed `binary_label` counts and means grouped by domain x gender over
   **all 272 rows of `labels_official.csv`, including the 33 sealed participants**. No
   per-participant label was displayed, no model was fit and no analysis choice was based on it.
   Recorded here for transparency. Every later check filters to the 239 manifest participants
   before touching any label column.
2. **Sealed-cohort transcripts already exist:** `transcripts_v2/` contains Whisper transcripts for
   sealed PIDs (for example 300, 303, 304, 600-607, 632-662), produced by the Phase 2 transcription
   pipeline. This contradicts the "unextracted" wording of v1.1 sec. 2.2. The files hold no labels and
   no Phase 6A code reads them, but the sealed set should be described as **label-blind and unused for
   modelling** rather than "unextracted".
3. **Data-licence note:** participant transcripts and PHQ labels are licensed DAIC-WOZ / E-DAIC data.
   New transcript outputs are git-ignored and are not pushed to the (public) GitHub remote.

## 5. Confirmatory Family

Unchanged (m = 3: H1 on V2, H2 on V10, H3 IUT on V18 vs V14 and V2; Holm alpha = 0.05).

## 6. S3 Re-Audit Result (label-blind, `transcript_qa_v1_3c_report.json`)

| Gate (v1.3b sec. 4) | Threshold | v1.3b full-length | v1.3c VAD-chunked | Status |
| :--- | :---: | :---: | :---: | :---: |
| Near-empty (< 20 words) | <= 5 % | 0.00 % | 0.00 % | PASS |
| Repetition loops (10-gram x5) | <= 5 % | 17.57 % (42) | 0.84 % (2; 2.51 % before the collapse filter) | PASS |
| Median WoZ WER vs manual | <= 0.60 | 0.197 | 0.202 (IQR 0.167-0.249) | PASS |
| Text-domain AUC on `bert_t2_768d` (descriptive) | - | 1.000 | 0.900 | reduced |

`bert_t2_768d` was rebuilt for all 239 sessions from the VAD-chunked transcripts. The previous
embedding is archived as `bert_t2_fulllen_768d`. Step E1 is authorised.
