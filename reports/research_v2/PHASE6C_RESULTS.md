# Phase 6C Results: Exploratory Accuracy Improvement (Nested Selection)

Protocol: `experiments/research_v2/prereg_phase6c_exploratory.md` (declared before extraction and evaluation). Cohort: the same 239 development participants. The sealed 33 and seeds 100-104 remain untouched. **All numbers are exploratory.**

## 1. Headline: nested-selection estimates (selection inside each outer fold)

| Procedure | AUC | 95% CI | vs Phase 6A frozen | Balanced acc. | Sensitivity | Specificity | WoZ AUC | Avatar AUC |
|:--|--:|:--|:--|--:|--:|--:|--:|--:|
| Voice: best of C1-C6 chosen by inner CV | **0.694** | [0.623, 0.761] | -0.016 vs V7 [-0.048, +0.015], p=0.84 | 0.616 | 0.565 | 0.668 | 0.658 | 0.817 |
| Fusion C7: inner-chosen voice + V14 text | **0.724** | [0.654, 0.790] | +0.010 vs V19 [-0.017, +0.038], p=0.24 | 0.628 | 0.632 | 0.623 | 0.680 | 0.856 |

Threshold metrics use a Youden threshold chosen on inner out-of-fold probabilities of the training participants, applied to the outer fold (averaged over seeds 0-4).

Inner-CV picks across the 25 outer folds: {'w2v2_L8': 8, 'ens_w2v2_L6-10': 7, 'w2v2_L6': 6, 'w2v2_L6_ms': 3, 'ens_L8+best_whisper': 1}

## 2. Freeze-rule decision (Phase 6C sec. 5)

Neither procedure exceeds its Phase 6A counterpart by the pre-declared margin of +0.02 AUC. **The Phase 6A frozen models are retained: V7 (wav2vec2 layer 8 mean, voice-only) and V19 (inner-CV best audio + BERT-T2 stacking).**

## 3. All candidates (descriptive 5-seed OOF AUC; best-of-N reading is optimistic)

| Candidate | AUC | 95% CI |
|:--|--:|:--|
| w2v2_L1 | 0.535 | [0.455, 0.621] |
| w2v2_L2 | 0.527 | [0.450, 0.611] |
| w2v2_L3 | 0.569 | [0.490, 0.651] |
| w2v2_L4 | 0.577 | [0.494, 0.655] |
| w2v2_L5 | 0.675 | [0.598, 0.748] |
| w2v2_L6 | 0.712 | [0.640, 0.779] |
| w2v2_L7 | 0.665 | [0.591, 0.735] |
| w2v2_L8 | 0.709 | [0.637, 0.775] |
| w2v2_L9 | 0.690 | [0.619, 0.757] |
| w2v2_L10 | 0.689 | [0.616, 0.755] |
| w2v2_L11 | 0.673 | [0.595, 0.743] |
| w2v2_L12 | 0.573 | [0.494, 0.649] |
| w2v2_L6_ms | 0.674 | [0.599, 0.746] |
| w2v2_L7_ms | 0.588 | [0.509, 0.667] |
| w2v2_L8_ms | 0.638 | [0.562, 0.715] |
| w2v2_L9_ms | 0.623 | [0.547, 0.700] |
| w2v2_L10_ms | 0.545 | [0.468, 0.624] |
| whisper_L1 | 0.474 | [0.400, 0.550] |
| whisper_L2 | 0.525 | [0.447, 0.603] |
| whisper_L3 | 0.553 | [0.475, 0.629] |
| whisper_L4 | 0.609 | [0.528, 0.687] |
| whisper_L5 | 0.659 | [0.579, 0.736] |
| whisper_L6 | 0.662 | [0.584, 0.738] |

## 4. Findings

1. **Layer profile:** wav2vec2-base carries depression signal in a broad mid-depth plateau (layers 5-11, AUC 0.67-0.71). It is near chance at layers 1-4 and at the final layer 12, which is specialised for ASR.
2. **Statistics pooling hurts:** appending the across-segment standard deviation lowered the AUC at every layer tested (6-10). With N = 239, doubling the dimensionality adds variance without adding signal.
3. **Whisper-base encoder:** peaks at 0.66 (layers 5-6), below wav2vec2 mid layers.
4. **Selection-honest estimate:** picking the best of 25 voice candidates inside each fold gives 0.694, consistent with V7 (0.711). The V7 result is not an artefact of choosing one lucky layer, and further model search on these 239 participants is unlikely to add reliable accuracy.
5. **Next lever:** more data (for example, E-DAIC train-split sessions not yet in the development pool) or the single sealed-33 evaluation, not more search on the same data.
