# Phase 8: Augmentation, Few-Shot and One-Shot Learning (Declared Before Running)

**Date:** 2026-10-26
**Status:** EXPLORATORY. Suggested by the course supervisor; declared before any Phase 8 feature was extracted.
**Pool:** the same N = 272 participants and the same repeated nested CV as Phase 7 (5 outer folds x seeds 0-4, speaker-disjoint,
stratified on label). Every augmentation, model fit, hyperparameter, and threshold lives inside the training folds.
**Reference:** Phase 7 P4 (AUC 0.764, accuracy 74.3%). Note: this is a further search on the same pool, so gains below
about +0.02 AUC are within selection noise.

## 1. New features (label-free, `extract_segments_phase8.py`)

Per Silero-VAD speech segment (frozen v1.2 parameters, boundaries from the clean audio), the frame-mean of:
`w2v2_L6` (wav2vec2-base, 768d), `hubert_large_er_L19` and `hubert_large_er_L20` (HuBERT-large-superb-ER, 1024d), plus segment frame counts.

- **clean** pass: original audio. Sanity check: the frame-weighted mean of the clean segments must reproduce the Phase 7
  participant-level features (cosine >= 0.999).
- **aug** pass (audio augmentation, deterministic per participant): additive Gaussian noise at 20 dB SNR, a random gain
  of -6 to +6 dB, and, for half of the participants (chosen by a PID-seeded coin), a 300-3400 Hz band-pass (telephone channel).
  These channel augmentations target the WoZ/Avatar recording-domain shift. Speed and pitch changes are excluded because
  speaking rate and pitch are themselves depression cues.

## 2. Methods (voice bases; each is then fused with emotion-RoBERTa text exactly as in Phase 7 P4)

| ID | Family | Method |
| :--- | :--- | :--- |
| A0 | Reference | Phase 7 P4 reproduced from the new clean segment store |
| A1 | Augmentation (feature space) | Segment-bagging: 10 extra pooled "views" per training participant from random 60% subsets of their segments |
| A2 | Augmentation (audio) | A1 + the same views from the audio-augmented pass (training participants only) |
| A3 | Augmentation (test time) | A2 + the prediction averaged over the clean and augmented views of the test participant |
| A4 | Mixup | Participant-level mixup (lambda ~ Beta(0.4, 0.4)) of training embeddings and PHQ targets |
| B1 | Few-shot | Prototypical network: 2-layer MLP encoder on segment embeddings, episodic training (2-way, 5-shot, 5-query participants per class), fixed 300 episodes; distance-to-prototype score |
| C1 | One-shot style | Nearest class prototype (cosine distance to the depressed and non-depressed centroids after PCA to 32 dimensions) |
| C2 | One-shot style | k-nearest neighbours (k in {5, 11, 21} by inner CV; cosine; PCA 32) |
| D1 | Class balance | Ridge/LR with class-balanced sample weights |

Ridge (PHQ target) is the base learner for A0-A4 and D1, with alpha tuned by inner CV as in Phase 7; each uses the same three voice
bases and the same top-3-style averaging of calibrated scores.

## 3. Evaluation and adoption rule

- Nested-CV AUC with bootstrap CI, accuracy and balanced accuracy at in-fold max-accuracy and Youden thresholds, and subgroup AUCs.
- Paired participant bootstrap of each method's fused score against Phase 7 P4.
- **Adoption:** a method replaces P4 as the final model only if its fused nested AUC exceeds 0.764 by >= 0.02 **or** its accuracy
  exceeds 74.3% by >= 2 points without lowering AUC. All methods are reported, including those that do not help.

## 4. Pre-run corrections (2026-10-26, before any Phase 8 outcome was computed)

- **ProtoNet reproducibility:** the runner trains in parallel threads, which share torch's global RNG. Weight initialisation
  now runs under a lock with the unit's seed, and dropout draws from a per-model generator. The architecture is unchanged.
- **Extra comparison vs A0:** the three voice bases were picked by the Phase 7 final model using all 272 labels, which
  slightly favours every Phase 8 method over P4 (P4 re-selects its bases inside each fold). Each method is therefore also
  compared with A0 (same fixed bases, no new technique). That isolates the gain from the technique itself. The adoption
  rule is unchanged, but a method whose gain over P4 does not also appear over A0 will be reported as base selection, not
  as a benefit of the technique.
