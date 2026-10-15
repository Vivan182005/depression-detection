# Phase 6D: Emotion-Tuned Encoders and MIT Vocal-Coordination Features (Declared Before Running)

**Date:** 2026-10-15
**Status:** EXPLORATORY. Declared after Phases 6A and 6C, before any Phase 6D feature was extracted.
**Data:** the same 239 development participants. The sealed 33 and seeds 100-104 remain untouched.

## 1. Motivation (literature review, 2026-10-15)

- **Gomez-Zaragoza et al., Interspeech 2025:** speech and text foundation models fine-tuned for
  **emotion recognition (ER)** gave the best speaker-independent depression results on E-DAIC/DAIC+.
  Intermediate block 9 beat the final block. Checkpoints: `superb/hubert-base-superb-er`,
  `superb/hubert-large-superb-er`, `j-hartmann/emotion-english-roberta-large`.
- **Williamson et al., AVEC 2016 (MIT Lincoln Laboratory + Harvard):** articulatory/spectral
  **coordination** features (eigenspectra of time-delay-embedded formant and delta-MFCC correlation
  matrices) reached a dev AUC of 0.71 for formants on 35 participants.
- **Al Hanai, Ghassemi and Glass, Interspeech 2018 (MIT CSAIL):** a COVAREP + LSTM voice model gave F1 0.63
  on the 35-person dev set. This is not adopted: hand-crafted acoustic statistics were at chance in Phase 6A
  (V1-V5), and question-conditioned weighting relies on interviewer prompts, a known shortcut
  (arXiv 2404.14463).

## 2. New features (label-blind, `extract_features_6d.py`, same frozen v1.2 Silero VAD segments)

| Key | Source | Content |
| :--- | :--- | :--- |
| `hubert_base_er_all_mean` (13 x 768) | `superb/hubert-base-superb-er` encoder | Frame-weighted mean over speech segments of hidden states 0-12 |
| `hubert_large_er_all_mean` (25 x 1024) | `superb/hubert-large-superb-er` encoder | Same, hidden states 0-24 |
| `mit_formant_eig` (3 x 45) | Parselmouth Burg formants F1-F3, 10 ms frames | Eigenvalues (descending) of the correlation matrix of 15-delay embeddings, delay scales 1, 3, 7 frames, pooled over segments of at least 1 s |
| `mit_dmfcc_eig` (3 x 240) | librosa 16 MFCC deltas, 10 ms frames | Same construction for 16 channels x 15 delays |
| `emo_roberta_large` (1024) | `j-hartmann/emotion-english-roberta-large` encoder on the v1.3c VAD-chunked Whisper transcripts | Mean of first-token embeddings over 512-token windows (stride 128), as in BERT-T2 |

## 3. Candidates added to the Phase 6C pool (all L2 logistic regression, C grid {0.001, 0.01, 0.1, 1, 10}, in-fold scaling)

| ID | Candidate |
| :--- | :--- |
| D1 | HuBERT-base-ER layer k mean, k in {1..12} |
| D2 | HuBERT-large-ER layer k mean, k in {1..24} |
| D3a | MIT formant coordination eigenspectra (135d) |
| D3b | MIT delta-MFCC coordination eigenspectra (720d) |
| D3c | D3a + D3b concatenated |
| D4 | Emotion-RoBERTa-large text (alternative to V14 in the fusion) |

## 4. Evaluation

- Per-candidate 5-seed OOF AUC with bootstrap CI (descriptive).
- **Nested selection** over the full audio pool (Phase 6C pool C1-C6 + D1-D3): in each outer fold the audio
  candidate with the best inner-CV AUC is chosen; the fusion chooses the text representation (V14 or D4)
  by inner-CV AUC as well and stacks it with the chosen audio candidate (meta-LR).
- Youden-threshold metrics chosen inside each training fold.

## 5. Freeze rule for the sealed evaluation (pre-stated)

- **Primary audio model:** if the Phase 6D nested-audio AUC exceeds V7's Phase 6A AUC (0.711) by at least 0.02,
  freeze the **most frequently selected Phase 6D audio candidate** (ties: higher per-candidate AUC).
  Otherwise keep **V7**.
- **Fusion model:** if the Phase 6D nested-fusion AUC exceeds V19's (0.714) by at least 0.02, freeze the most
  frequently selected audio + text pair with meta-LR stacking. Otherwise keep **V19**.
- The sealed 33 are then evaluated **once** with both frozen models, trained on all 239 development participants.
