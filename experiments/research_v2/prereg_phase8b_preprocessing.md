# Phase 8b: One-Shot, Few-Shot and Augmentation as Preprocessing (Declared Before Running)

**Date:** 2026-10-26
**Status:** EXPLORATORY follow-up to Phase 8. The supervisor asked whether these ideas help when they are applied in
**preprocessing** (to the features) rather than as the learner. Declared before any Phase 8b number was computed.
**Pool and protocol:** identical to Phase 8 (N = 272, 5 outer folds x seeds 0-4, inner 4-fold alpha tuning, same three voice
bases, ridge on PHQ-8, CDF averaging, meta-LR fusion with emotion-RoBERTa text, in-fold thresholds).
**Reference:** R0 = Phase 8 A0 (fused AUC 0.785, accuracy 75.2%). Because the bases are fixed, every comparison is made
against R0, not against P4.

| ID | Idea | Preprocessing applied to each person's segment embeddings (label-free) |
| :--- | :--- | :--- |
| R0 | Reference | Frame-weighted mean of all segments |
| P1 | One-shot speaker reference | One reference "shot" = the mean of the person's first 5 speech segments (neutral rapport questions). Features = [mean, mean - reference], which removes speaker and channel identity |
| P2 | Few-shot domain adaptation | Inside each training fold, the mean of each recording setup (WoZ / Avatar) is estimated from the training people only and subtracted (unlabelled domain alignment, CORAL-lite) |
| P3 | Augmentation-averaged features | Features = average of clean and augmented-audio embeddings, for both training and test people |
| P4 | Statistics pooling | [mean, standard deviation across segments], capturing how much the voice varies |
| P5 | All of P1 + P3 + P4 | [P3 mean, P1 difference, P4 std] concatenated |

**Adoption:** fused AUC >= R0 + 0.02 with a paired-bootstrap 95% CI excluding zero, or accuracy >= R0 + 2 points without
lowering AUC. Even then the honest headline number stays the Phase 7 estimate unless the gain also holds against P4.
