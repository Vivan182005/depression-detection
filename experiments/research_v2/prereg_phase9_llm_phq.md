# Phase 9: Local LLM PHQ-8 Item Scoring (Declared Before Running)

**Date:** 2026-10-27
**Status:** EXPLORATORY. This is the top-ranked recommendation of `reports/Depression detection literature review.md`
(Ishikawa & Duke 2026 report that zero-shot LLM PHQ-8 item scoring is the strongest text signal under subject-disjoint CV).

## 0. Pre-run checks (done before any LLM output was produced)

- Binary labels recomputed from PHQ-8 >= 10: 0 mismatches out of 275 rows.
- Corpus confound: an avatar-vs-WoZ indicator alone has AUC 0.512 (prevalence 0.329 vs 0.305), so it is negligible. Gender alone has AUC 0.561.
- Interviewer leakage: 10 verbatim Ellie prompts were searched across all 272 transcripts. Only 3 transcripts contain any of them, so leakage is negligible.
- 17 participants mention a diagnosis or being depressed in their own words. This is legitimate participant speech, but a
  robustness AUC excluding them is reported.
- Phase 7 P4 extra metrics (max-accuracy point): MCC 0.368, macro-F1 0.679.

## 1. LLM features (label-free, `score_phq_llm.py`)

- **Model:** Mistral-7B-Instruct-v0.3, 4-bit MLX (`mlx-community/Mistral-7B-Instruct-v0.3-4bit`). It was chosen because it was the
  best small open model in zero-shot DAIC-WOZ tests (Merzougui 2025, macro-F1 0.78), and it runs locally, so no transcript leaves the machine.
- **Input:** the same v1.3c VAD-chunked Whisper transcripts (participant speech only) used for the emotion-RoBERTa features.
- **Output:** for each of the 8 PHQ-8 items, the probability-weighted score E[s] = sum_k k * p(k), k in {0,1,2,3}, read from the
  model's next-token distribution at that item's value, with greedy decoding of a fixed JSON template. This gives 8 continuous
  features plus their sum. The decoding is deterministic, the prompt is fixed, and there is no prompt tuning: one prompt, written before any output was seen.

## 2. Analyses (same pool, folds and seeds as Phase 8b; reference R0 = Phase 8 A0 fused)

| ID | Description |
| :--- | :--- |
| L0 | LLM item sum alone, zero training (AUC directly, thresholds in-fold) |
| L1 | Ridge on the 8 LLM items (alpha tuned in inner CV, PHQ target) |
| F1 | **Primary:** voice (R0 three bases) + emotion-RoBERTa text + L1, meta-LR stacker on in-fold CDF scores |
| F2 | Voice + L1 only (LLM replaces emotion-RoBERTa) |

**Adoption:** F1 replaces the final model if its fused AUC is at least R0 + 0.02 with a paired-bootstrap 95% CI excluding zero, or its accuracy is at least
R0 + 2 points without lowering AUC. Reported regardless: accuracy, balanced accuracy, MCC, macro-F1, subgroup AUCs, and AUC without the 17 self-disclosure participants.
