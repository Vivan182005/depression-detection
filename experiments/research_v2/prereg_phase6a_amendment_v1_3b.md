# Pre-Registration Amendment (Version 1.3b): True BERT-T2 Text Benchmark & Fallback Elimination
## Voice-Primary Depression Detection (Audio Primary, Text Secondary)

**Protocol Amendment:** v1.3b (Supplements v1.1, v1.2, and v1.3)  
**Date:** 2026-10-15  
**Repository Branch:** `research-upgrade-2026-10-13`  
**Working Directory:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Execution Gate:** LOCKED PRIOR TO ANY OUTCOME EVALUATION ON REAL LABELS  
**Outcome Status:** ZERO REAL-LABEL PREDICTIVE OUTCOMES SEEN  

---

## 1. Rationale & Decision Record

Following a rigorous text feature audit, an unflagged defect in historical data loaders was discovered:
In Phase 5, `dataset.py` implemented an undocumented fallback whereby requested `text_variant='T2'` silently fell back to `bert_t1_768d` (human manual transcripts) for the 123 DAIC-WOZ development participants lacking Whisper transcripts.

Under executive authorization, **Option A** has been selected:
- **Variant V14 is strictly locked to TRUE BERT-T2 across 100% of the 239 active development sessions.**
- All 102 remaining DAIC-WOZ development sessions will be transcribed using `openai/whisper-tiny` under the exact parameters matching the E-DAIC Avatar development cohort.
- All silent fallbacks have been eliminated from data loaders and verified with unit tests.

---

## 2. Whisper-Tiny Transcription Specification (Standardized Across Cohorts)

To eliminate transcription modality confounding between DAIC-WOZ and E-DAIC Avatar, all 239 active development sessions will utilize transcripts produced by the identical local ASR model and decoding configuration:

| Parameter | Specification | Purpose / Rationale |
| :--- | :--- | :--- |
| **Model** | `openai/whisper-tiny` (39M parameters) | Standardized lightweight local open-source ASR |
| **Framework** | HuggingFace `transformers` ASR pipeline | Fully deterministic local inference |
| **Chunk Length** | 30.0 seconds (`chunk_length_s=30`) | Standard long-form audio chunking |
| **Batch Size** | 8 | Memory-efficient CPU batching |
| **Decoding Strategy** | Greedy decoding (`temperature=0.0`, `do_sample=False`) | Zero stochasticity; fully reproducible text |
| **Language & Task** | `language='english'`, `task='transcribe'` | Standard English transcription |
| **Timestamps** | `return_timestamps=True` | Generates word/segment-level timestamps |
| **Post-Processing** | Lowercase normalization, apostrophes preserved, punctuation stripped | Uniform input format for `bert-base-uncased` |

---

## 3. Fallback Bug Elimination & Zero-Tolerance Policy

### 3.1 Historical Bug Description
In `experiments/research_v2/dataset.py`:
```python
# HISTORICAL CODE (DEFECTIVE):
if text_variant == 'T2' and 'bert_t2_768d' in data:
    bert = data['bert_t2_768d'].astype(np.float32)
elif 'bert_t1_768d' in data:
    bert = data['bert_t1_768d'].astype(np.float32)  # <-- SILENT FALLBACK TO MANUAL TRANSCRIPTS
else:
    bert = data['bert_768d'].astype(np.float32)
```
This allowed 123 of 239 participants to evaluate on human manual transcripts while claiming an automated ASR evaluation.

### 3.2 Authoritative Zero-Fallback Standard
In `dataset.py` and `data_loader.py`, all silent fallbacks have been replaced with strict runtime assertions:
```python
# CORRECTED CODE (ZERO FALLBACK):
if text_variant == 'T2':
    if 'bert_t2_768d' not in data:
        raise KeyError(f"Requested text_variant='T2' but 'bert_t2_768d' is missing for PID {pid} in {feat_path}. Silent fallback prohibited.")
    bert = data['bert_t2_768d'].astype(np.float32)
elif text_variant == 'T1':
    if 'bert_t1_768d' in data:
        bert = data['bert_t1_768d'].astype(np.float32)
    elif 'bert_768d' in data:
        bert = data['bert_768d'].astype(np.float32)
    else:
        raise KeyError(f"Requested text_variant='T1' but neither 'bert_t1_768d' nor 'bert_768d' is present for PID {pid} in {feat_path}")
else:
    raise ValueError(f"Unknown text_variant '{text_variant}'. Must be 'T1' or 'T2'.")
```
- Missing variants immediately raise `KeyError`.
- Missing metadata (`meta_7d`) immediately raises `KeyError` (zero-substitution prohibited).
- Verified by unit tests in [`test_silent_fallbacks.py`](test_silent_fallbacks.py).

---

## 4. Transcript QA & Stop Thresholds (Pre-Stated)

Following Whisper transcription of the remaining 102 WoZ dev sessions and prior to Step E1 dry-run, an automated label-blind transcript QA audit will be conducted against the following **pre-stated stopping rules**:

1. **Empty / Near-Empty Transcript Threshold:**
   - Any transcript with $< 20$ total words is flagged as invalid.
   - **Stop Rule:** If $> 5\%$ of sessions ($> 11$ sessions) are empty/near-empty, **STOP and report**.
2. **Repetition Loop Threshold:**
   - Any transcript containing any 10-gram repeated $\ge 5$ consecutive times is flagged for ASR hallucination/looping.
   - **Stop Rule:** If $> 5\%$ of sessions ($> 11$ sessions) exhibit repetition loops, **STOP and report**.
3. **Word Error Rate (WER) Against Manual Reference (WoZ Dev Only):**
   - Transcripts will be aligned with official WoZ human transcripts to evaluate Word Error Rate.
   - **Stop Rule:** If the median WoZ WER exceeds **$0.60$ (60%)**, **STOP and report**.
4. **Text Domain Confounding Check:**
   - Evaluate classification domain-AUC on `bert_t2_768d` to verify that the extreme 0.9994 domain separation observed in T1 has been substantially reduced.

---

## 5. Status of the Confirmatory Testing Family ($m = 3$)

- **Verdict: FULLY PRESERVED & STRENGTHENED.**
  - **$H_1$ (Hand-Crafted Voice V2 vs Null):** Unaffected.
  - **$H_2$ (WavLM Layer 8 V10 vs Null):** Unaffected.
  - **$H_3$ (Multimodal Incremental Synergy V18 vs Constituents):** V18 stacks V2 with true automated ASR text (V14 = BERT-T2). This evaluates genuine multimodal clinical value in a realistic automated clinical intake scenario without relying on human manual transcription.

---
*Amendment v1.3b locked on branch `research-upgrade-2026-10-13` before real label evaluation.*
