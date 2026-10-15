# Phase 6C: Exploratory Accuracy Improvement (Declared Before Running)

**Date:** 2026-10-15
**Status:** EXPLORATORY. Declared after the Phase 6A observed AUCs were seen (V7 = 0.711,
V10 = 0.644, V14 = 0.704, V19 = 0.714, hand-crafted V1-V5/V12 at about 0.50), before any
Phase 6C feature was extracted or evaluated.
**Data:** the same 239 development participants. The sealed 33 and seeds 100-104 remain untouched.

## 1. Why a separate phase

Phase 6A is the pre-registered confirmatory study, and its results stand as reported. Any further
model search on the same 239 participants is selection on observed outcomes, so:

1. Every Phase 6C number is labelled **exploratory**.
2. The headline Phase 6C estimate is the **nested-selection AUC**: inside each outer fold, the
   candidate is chosen by inner 4-fold CV AUC on the training participants only, and only that
   candidate predicts the outer fold. This scores the whole "pick the best model" procedure, not
   the luckiest candidate.
3. Per-candidate AUCs are reported for every candidate (including losers) but are
   **optimistically biased** if read as the best-of-N.
4. The only unbiased test of the final model is a single evaluation on the sealed 33, after the
   Phase 6C winner is frozen.

## 2. New features (label-blind extraction, `extract_features_6c.py`)

Same audio, same frozen v1.2 Silero VAD segments, same per-segment processing as v3:

| Key | Model | Content |
| :--- | :--- | :--- |
| `w2v2_all_mean` (13 x 768) | `facebook/wav2vec2-base-960h` | Frame-weighted mean over speech segments of hidden states 0-12 |
| `w2v2_all_segstd` (13 x 768) | same | Std across segments of the per-segment means (statistics pooling) |
| `whisper_base_all_mean` (7 x 512) | `openai/whisper-base` encoder | Mean over valid (non-padding) frames of hidden states 0-6, over packed <= 28 s VAD chunks |

Sanity check (must pass before evaluation): the re-extracted `w2v2_all_mean[8]` must match the stored
`w2v2_layer_8_mean` (V7) with cosine similarity >= 0.999 for every participant.

## 3. Candidate pool (all L2 logistic regression, C grid {0.001, 0.01, 0.1, 1, 10}, in-fold scaling)

| ID | Candidate |
| :--- | :--- |
| C1 | V7 baseline: wav2vec2 layer 8 mean |
| C2 | wav2vec2 layer k mean, k in {1..12} (one sub-candidate per layer) |
| C3 | wav2vec2 layer k mean + segment-std, k in {6, 7, 8, 9, 10} |
| C4 | wav2vec2 layer-ensemble: average of the probabilities of the layer 6-10 mean models |
| C5 | Whisper-base encoder layer k mean, k in {1..6} |
| C6 | Voice ensemble: average of the probabilities of C1 and the best Whisper layer chosen in inner CV |
| C7 | Fusion: nested stack (meta-LR) of the inner-CV-selected audio candidate + V14 text |

## 4. Reporting

- All candidates, with 5-seed OOF AUC and participant-bootstrap 95% CI.
- Nested-selection AUC for the audio pool (C1-C6) and for the fusion (C7).
- Paired bootstrap of the nested-selection AUC against V7 (Phase 6A).
- Threshold metrics (balanced accuracy, sensitivity, specificity) at the **Youden-optimal threshold
  chosen inside each training fold**, never on the evaluated fold.

## 5. Freeze rule for the sealed evaluation

The model taken to the sealed 33 is the procedure with the highest nested-selection AUC. If that
does not exceed V7's AUC by at least 0.02, **V7 (Phase 6A frozen primary audio) is kept**, since a
smaller gain is within selection noise.
