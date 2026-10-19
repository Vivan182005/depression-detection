# Final Results: Voice-Primary Depression Detection (PHQ-8 >= 10)

**Status:** complete. Sealed-cohort evaluation run once on 2026-10-19 with the frozen models.

## 1. Headline: performance on the 33 sealed (never-seen) participants

33 participants (11 depressed, 22 not; 9 DAIC-WoZ, 24 E-DAIC Avatar). Models were trained on all 239
development participants. The decision threshold is the Youden point of the development out-of-fold
probabilities, fixed before unblinding. Predictions were written and hashed (SHA-256 `1704a72487416f20...`)
before any sealed label was read.

| Model | AUC | 95% CI | Permutation p | Sensitivity | Specificity | Balanced acc. | Accuracy | WoZ AUC | Avatar AUC |
|:--|--:|:--|--:|:--|:--|--:|--:|:--|:--|
| V7 voice only (wav2vec2-base layer 8 + LR) | **0.694** | [0.48, 0.88] | 0.035 | 5/11 (45%) | 17/22 (77%) | 61.4% | 66.7% | 0.95 (n=9) | 0.57 (n=24) |
| V19 voice + text (V7 + BERT on Whisper-tiny transcripts, stacked) | **0.698** | [0.50, 0.88] | 0.035 | 5/11 (45%) | 16/22 (73%) | 59.1% | 63.6% | 0.80 (n=9) | 0.66 (n=24) |

**Development vs sealed:** V7 0.711 (dev, 5x5 CV) vs 0.694 (sealed); V19 0.714 vs 0.698. The sealed result matches the
development estimate, so the model was not overfitted to the development data. With n = 33 the CI is wide
([0.48, 0.88]); both models beat chance (permutation p = 0.035). Subgroup AUCs (9 WoZ / 24 Avatar) are
descriptive only.

## 2. How we got here

| Phase | Outcome |
| :--- | :--- |
| 0-4 Forensic audit | The legacy "83.78%" Candidate 2A was invalid: random-noise wav2vec features, zero-padded eGeMAPS, 66/189 wrong PHQ labels, non-independent "unseen" split. Rebuilt from scratch. |
| 5 Corrected baselines | Multimodal fusion did not beat metadata; a silent BERT fallback to manual transcripts was found and retracted. |
| 6A Pre-registered study (19 variants, N = 239) | H2 supported: WavLM-L8 beats a 999-permutation null (p = 0.003, Holm). H1 (hand-crafted voice, AUC 0.504) and H3 (fusion beats both parts) not supported. Freeze: V7 (0.711), V19 (0.714). |
| S3 / v1.3c transcript fix | Whisper repetition loops 17.6% -> 0.8% via VAD-chunked decoding, applied to all sessions. |
| 6C Layer/pooling/Whisper search | Selection-honest voice AUC 0.694; no gain over V7. |
| 6D Emotion-tuned HuBERT/RoBERTa + MIT/Harvard coordination features | Voice 0.684, fusion 0.720 (+0.007, p = 0.39); below the +0.02 margin, so V7/V19 kept. MIT coordination features at chance. |
| **Sealed evaluation** | **V7 0.694, V19 0.698 on 33 unseen participants.** |

## 3. Model in one paragraph

Silero VAD keeps only the participant's speech. Each speech segment goes through the frozen self-supervised model
`facebook/wav2vec2-base-960h`, and the layer-8 hidden states are averaged over all speech into one 768-number voice
profile per person. An L2 logistic regression (C tuned by inner CV) maps that profile to a depression probability.
The fusion model additionally transcribes the speech with Whisper-tiny, embeds it with BERT and combines the two
probabilities with a small logistic-regression stacker.

## 4. Honest limitations

- AUC of about 0.70 and balanced accuracy of about 60%: a screening aid at best, not a diagnostic tool. At the fixed
  threshold it catches 5 of 11 depressed participants; a lower threshold would catch more at the cost of more false alarms.
- 272 participants in total (two related corpora, US English). External validity to other populations is untested.
- Hand-crafted acoustic features (jitter, shimmer, pitch, MFCC, eGeMAPS, MIT coordination) were at chance under
  speaker-independent evaluation, contrary to many small-sample reports.
- The sealed set can no longer be used as an unbiased test: any future model change needs new held-out data.

## 5. Provenance

Pre-registration `prereg_phase6a.md` with amendments v1.2, v1.3, v1.3b, v1.3c; exploratory plans `prereg_phase6c_exploratory.md`
and `prereg_phase6d_emotion_mit.md`; analysis tag `prereg-exec-v1`; reports `PHASE6A_RESULTS.md`, `PHASE6C_RESULTS.md`,
`PHASE6D_RESULTS.md`; sealed artefacts `sealed_predictions.csv`, `sealed_eval.lock` (written 2026-10-19 11:18:26),
`sealed_results.json`, `sealed_run.log`.

## 6. Update: Phase 7 improved model (all 272 participants, cross-validated)

After the one-shot sealed test, the 33 sealed participants joined the training pool (N = 272) and the model was rebuilt with
PHQ-score regression, a top-3 voice ensemble (emotion-tuned HuBERT-large + wav2vec2) and emotion-RoBERTa text. Repeated nested CV
(every person predicted by a model that never saw them):

| | AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity |
|:--|--:|--:|--:|--:|--:|
| Phase 7 P4, max-accuracy threshold | **0.764** | **74.3%** | 67.0% | 48% | 87% |
| Phase 7 P4, balanced (Youden) threshold | 0.764 | 69.7% | **68.1%** | 64% | 72% |
| Majority-class baseline | 0.500 | 68.8% | 50.0% | 0% | 100% |

Avatar-domain AUC improved from 0.57 (sealed V7) to 0.79. Gender/domain subgroup AUCs lie within 0.75-0.79. Details: `PHASE7_RESULTS.md`.
These are cross-validated estimates on a pool that includes the formerly-sealed participants; the untouched held-out figure remains Section 1.
