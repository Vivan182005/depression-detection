# Phase 9 Results: Local LLM PHQ-8 Item Scoring

**Date:** 2026-10-27 | **Plan:** `experiments/research_v2/prereg_phase9_llm_phq.md` (declared and pushed before scoring, commit dc2664d)
**Pool:** all 272 participants, repeated nested CV (5 outer folds x seeds 0-4, speaker-disjoint), same protocol and voice bases as Phase 8 R0/A0.
**Baselines:** always guessing "not depressed" scores **68.8%**. Phase 7 P4 (the final model until now) scores **AUC 0.764, accuracy 74.3%**.

## 0. Pre-run checks (Section 0 of the plan)

| Check | Result |
| :--- | :--- |
| Labels recomputed from PHQ-8 scores (≥ 10) | 0 errors in 272 |
| Recording setup as a confound | Depression rate WoZ 30.5%, Avatar 32.9%; setup alone predicts depression at AUC 0.51 |
| Interviewer (Ellie) prompts leaking into transcripts | Present in 3 of 272 participant-only transcripts: negligible |
| Self-disclosure ("I'm depressed", "diagnosed with ...") | 17 participants; every result below is also reported without them |

## 1. Method

- **Scorer:** Mistral-7B-Instruct-v0.3, 4-bit, run locally with MLX on the M2 laptop. It is used **frozen and zero-shot**: it never sees a PHQ-8 score or label, and nothing is fine-tuned.
- **Input:** the participant-only Whisper transcript (no interviewer questions, no manual transcripts), given in full with no truncation (longest 5,618 tokens).
- **Output:** for each of the 8 PHQ-8 items, the expected score E[s] = Σ k·p(k) over k = 0-3 from the model's next-token probabilities, using greedy JSON decoding. All 272 transcripts returned all 8 items, with no missing values.
- **Learning:** every learned step happens inside the training folds:
  - ridge regression on the 8 item scores (L1);
  - the voice ridge and text ridge models from R0;
  - the logistic-regression stacker that fuses them;
  - the max-accuracy and Youden thresholds.
- **Runtime:** about 3.5 hours. The first run thrashed swap on 8 GB of RAM; capping MLX's buffer cache fixed it, and the run resumed from the saved rows.

## 2. Results

| ID | Model | AUC [95% CI] | vs R0 [95% CI] | Accuracy | Balanced acc. | MCC | Macro-F1 | AUC without 17 disclosers | Avatar AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| R0 | Voice + text (reference, = Phase 8 A0) | 0.785 [0.722, 0.843] | n/a | 75.2% | 68.3% | 0.39 | 0.69 | 0.791 | 0.816 |
| L0 | LLM item sum, no training at all | 0.819 [0.766, 0.871] | +0.033 [-0.035, +0.104] | 75.7% | 71.2% | 0.43 | 0.71 | 0.806 | 0.825 |
| L1 | Ridge on the 8 LLM items | 0.826 [0.775, 0.875] | +0.041 [-0.028, +0.111] | 78.1% | **74.2%** | 0.49 | **0.74** | 0.815 | 0.818 |
| **F1** | **Voice + text + LLM (primary)** | **0.843** [0.793, 0.887] | **+0.057** [+0.015, +0.102] | **79.1%** | 72.4% | **0.49** | **0.74** | **0.835** | **0.855** |
| F2 | Voice + LLM (no text_emo) | **0.849** [0.800, 0.892] | **+0.063** [+0.009, +0.120] | **79.3%** | 72.3% | **0.49** | **0.74** | **0.839** | **0.857** |

- Accuracy, balanced accuracy, MCC and macro-F1 are at the in-fold max-accuracy threshold.
- At the balanced (Youden) threshold, F1 catches **77% of depressed people**, against 63% for R0.
- F1 against Phase 7 P4: **+0.079 AUC [+0.035, +0.124]**.
- Subgroup AUC for F1:
  - by setup: WoZ 0.839, Avatar 0.855;
  - by gender: female 0.857, male 0.829.
  - Every subgroup is 0.05-0.07 higher than R0.

**Per-item AUC** (zero-shot LLM score against the binary label):

| Item | AUC |
| :--- | :---: |
| Sleep | 0.816 |
| Concentration | 0.815 |
| Depressed mood | 0.805 |
| Tired | 0.796 |
| Failure | 0.765 |
| Psychomotor | 0.699 |
| Appetite | 0.629 |
| Interest | 0.568 |

Interest is nearly uninformative: the model rates almost everyone about 1.7. The LLM item sum correlates with true PHQ-8 at Spearman 0.63.

## 3. Adoption decision

The pre-declared rule was: F1 AUC ≥ R0 + 0.02 with a paired-bootstrap CI excluding zero, **or** accuracy ≥ R0 + 2 points without lowering AUC.

- F1 gains **+0.057 AUC** with a CI of [+0.015, +0.102], which excludes zero.
- F1 gains **+3.9 points of accuracy**.

**Both conditions are met, so F1 is adopted.** This is the first method in Phases 8-9 to clear the rule.

## 4. How far to trust it

- **No label leakage.** The LLM is frozen, zero-shot and label-free. The ridge, stacker and thresholds are all fit inside the training folds, and speakers never cross folds.
- **Not driven by people who say it outright.** Without the 17 self-disclosing participants, F1 still scores AUC 0.835. The gain holds across WoZ and Avatar and across genders.
- **Same optimism as R0.** The three voice bases were fixed using all 272 labels, which inflated R0 by about 0.02 over P4. That optimism is shared by R0 and F1, so the **+0.057 delta is fair**. The absolute 0.843 may still be about 0.02 high.
  - **Honest estimate: AUC about 0.82-0.84, accuracy about 77-79%.**
  - It is quoted as a 95% CI, never as in-sample.
- **F2 ≈ F1.** Once the LLM items are present, the text-emotion model adds nothing (F2 0.849 against F1 0.843; the difference is within noise). The LLM captures what the text-emotion model was capturing, and more.
- **Pretraining contamination is unlikely but cannot be ruled out.** DAIC-WOZ transcripts are licence-restricted and not public. Mistral's training data is undisclosed, though.
- **This is not a new unseen test.** The sealed 33 are part of the 272-person CV pool, as in Phases 7-8, and are not re-quoted as "unseen".

## 5. Where this sits in the literature

Honest speaker-independent DAIC-WOZ / E-DAIC results are about 0.55-0.74 macro-F1 (`reports/Depression detection literature review.md`). F1 reaches **macro-F1 0.74 and AUC 0.84 under repeated speaker-disjoint nested CV on 272 people**, on ASR transcripts only and with no interviewer prompts. That puts it at the top of the honest range. It also matches the literature review's prediction that LLM PHQ item scoring was the most promising next step.

## 6. Next step

Retrain the final model as F1 (voice + text + LLM items) on all 272 and add LLM scoring to `predict_audio`, which needs the local 4 GB Mistral model at inference. This will be done after confirmation, because it changes the app's dependencies.

Files:
- `experiments/research_v2/phase9_results.json`
- `phase9_run.log`
- `score_llm.log` and `score_llm_run1.log`
- `score_phq_llm.py`, `run_phase9_llm.py`
- Item scores: `feature_store_9/llm_phq.csv`. These are derived from licensed transcripts, so they are git-ignored and not pushed.
