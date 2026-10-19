# Phase 7: Full-Cohort Accuracy Optimisation (N = 272, Post-Unblinding)

**Date:** 2026-10-19
**Status:** EXPLORATORY. Declared after the one-shot sealed evaluation (V7 AUC 0.694, V19 0.698 on 33 participants).

## 1. Why the design changes

The sealed 33 were evaluated once (`sealed_eval.lock`). Their labels are now known, so they can **no longer serve as an
unbiased held-out test**, and re-testing a modified model on them would overstate performance. Phase 7 therefore:

1. Merges the 239 development and 33 formerly-sealed participants into one pool of **N = 272**.
2. Estimates performance only by **repeated nested cross-validation** (5 outer folds x 5 seeds, speaker-disjoint, stratified on
   label). Every prediction is made by a model that never saw that participant. All selection, tuning, and threshold
   choice happen inside the training folds.
3. Reports the sealed-test result (Section 6 of FINAL_RESULTS.md) unchanged as the only true held-out estimate. Phase 7 numbers are
   cross-validated estimates on a larger sample and are labelled as such.

## 2. Candidate pool (restricted to families that showed signal in Phases 6A-6D, to limit selection noise)

Voice bases: wav2vec2-base layers 5-11 (mean), HuBERT-large-ER layers 16-20, HuBERT-base-ER layer 9, Whisper-base layers 5-6.
Text bases: BERT-T2 (V14), emotion-RoBERTa-large.

Models per base:
- **CLS**: L2 logistic regression on the binary label (as before).
- **REG**: ridge regression on the PHQ-8 total score (0-24), alpha grid {1, 10, 100, 1000, 10000} tuned by inner CV; the
  predicted score is the ranking for AUC and is thresholded for classification. New in Phase 7.

Procedures compared (each chosen and scored by nested CV):
- P1 single best voice base (CLS); P2 single best voice base (REG)
- P3 top-3 voice ensemble: average of the 3 best voice bases by inner AUC (CLS or REG, whichever has the higher inner AUC)
- P4 fusion: meta-LR stack of the P3 voice score + the best text base score

## 3. Operating points

For every procedure, two thresholds are chosen on inner out-of-fold scores of the training participants only:
- **max-accuracy** threshold (the metric the course grades), and
- **Youden** threshold (balanced sensitivity/specificity).

Reported for each: accuracy, balanced accuracy, sensitivity, specificity, plus AUC with a participant bootstrap CI.
The majority-class baseline accuracy (always "not depressed") is reported alongside for reference.

## 4. Selection of the reported model

The procedure with the highest nested AUC is the headline model. Its max-accuracy operating point is the
headline accuracy, reported together with balanced accuracy and the majority-class baseline.

## 5. Fixes motivated by the sealed-set error analysis (added before any Phase 7 run)

The sealed error analysis (descriptive only) showed that the frozen voice model separates WoZ participants well
(AUC 0.95, n = 9) but not Avatar participants (0.57, n = 24). 5 of the 6 missed depressed participants and 4 of the 5 false alarms
were Avatar sessions, and 4 of the 6 misses were borderline (PHQ 10-12). Checks ruled out a stereo-channel error (channels identical,
r = 1.0) and showed only minor interviewer bleed (3 of 61 Avatar dev transcripts contain 3 or more Ellie prompts). The working
explanation is recording-domain shift with few Avatar training examples. Fixes evaluated in Phase 7:

- **F1 per-domain standardisation:** features z-scored separately within WoZ and within Avatar, with statistics from
  training-fold participants of that domain only (label-free domain adaptation).
- **F2 per-domain thresholds:** the max-accuracy / Youden threshold is chosen separately per domain inside the training fold.
- **F3 larger Avatar pool:** N = 272 (Avatar 85 vs 61).
- **F4 PHQ regression target** (Section 2, REG).
- Each procedure P1-P4 is run with and without F1; F2 is evaluated as an alternative operating point.

**Bias / fairness reporting:** AUC, sensitivity, and specificity per gender and per domain for the headline model.
Because these fixes were motivated by the 33 formerly-sealed participants, the Phase 7 estimates include those participants
and are labelled accordingly. The one-shot sealed result (V7 0.694) remains the untouched held-out figure.
