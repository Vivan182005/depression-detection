# Phase 6D Results: Emotion-Tuned Encoders and MIT Vocal Coordination

Protocol: `experiments/research_v2/prereg_phase6d_emotion_mit.md` (declared before extraction). Same 239 development participants; sealed 33 and seeds 100-104 untouched. **Exploratory.**

## 1. Headline: nested selection over the full pool (Phase 6C pool + emotion-tuned HuBERT + MIT features; fusion also chooses the text encoder)

| Procedure | AUC | 95% CI | vs frozen | Balanced acc. | Sensitivity | Specificity |
|:--|--:|:--|:--|--:|--:|--:|
| Voice (best of ~60 candidates) | **0.683** | [0.609, 0.754] | -0.027 vs V7 (0.710) [-0.065, +0.010], p=0.92 | 0.612 | 0.549 | 0.675 |
| Fusion (best voice + best text) | **0.720** | [0.648, 0.791] | +0.007 vs V19 (0.714) [-0.045, +0.057], p=0.39 | 0.618 | 0.559 | 0.676 |

Voice picks across 25 folds: {'ens_w2v2_L6-10': 5, 'w2v2_L8': 5, 'hubert_large_er_L20': 3, 'w2v2_L6': 3, 'w2v2_L6_ms': 2, 'hubert_large_er_L19': 2, 'hubert_large_er_L21': 1, 'ens_L8+best_whisper': 1, 'hubert_large_er_L18': 1, 'hubert_large_er_L11': 1, 'hubert_large_er_L16': 1}

Voice|text pair picks: {'ens_w2v2_L6-10|text_bert': 3, 'w2v2_L6|text_emo': 3, 'w2v2_L8|text_emo': 3, 'ens_w2v2_L6-10|text_emo': 2, 'w2v2_L6_ms|text_emo': 2, 'w2v2_L8|text_bert': 2, 'hubert_large_er_L20|text_emo': 2, 'hubert_large_er_L19|text_emo': 2, 'hubert_large_er_L20|text_bert': 1, 'hubert_large_er_L21|text_emo': 1, 'ens_L8+best_whisper|text_emo': 1, 'hubert_large_er_L18|text_bert': 1, 'hubert_large_er_L11|text_emo': 1, 'hubert_large_er_L16|text_emo': 1}

## 2. Freeze decision (pre-stated, margin +0.02 AUC)

`final_frozen_config.json`: audio = **V7**, fusion = **V19** (both from Phase 6A). Neither Phase 6D procedure cleared the margin.

## 3. New candidates (descriptive 5-seed OOF AUC)

| Candidate | AUC | 95% CI |
|:--|--:|:--|
| hubert_base_er_L1 | 0.521 | [0.439, 0.606] |
| hubert_base_er_L2 | 0.529 | [0.450, 0.613] |
| hubert_base_er_L3 | 0.532 | [0.452, 0.610] |
| hubert_base_er_L4 | 0.557 | [0.479, 0.634] |
| hubert_base_er_L5 | 0.588 | [0.508, 0.666] |
| hubert_base_er_L6 | 0.614 | [0.533, 0.687] |
| hubert_base_er_L7 | 0.641 | [0.564, 0.714] |
| hubert_base_er_L8 | 0.656 | [0.578, 0.734] |
| hubert_base_er_L9 | 0.657 | [0.576, 0.730] |
| hubert_base_er_L10 | 0.671 | [0.593, 0.743] |
| hubert_base_er_L11 | 0.629 | [0.552, 0.705] |
| hubert_base_er_L12 | 0.612 | [0.534, 0.690] |
| hubert_large_er_L1 | 0.516 | [0.435, 0.596] |
| hubert_large_er_L2 | 0.539 | [0.460, 0.618] |
| hubert_large_er_L3 | 0.518 | [0.435, 0.599] |
| hubert_large_er_L4 | 0.514 | [0.429, 0.591] |
| hubert_large_er_L5 | 0.514 | [0.434, 0.595] |
| hubert_large_er_L6 | 0.573 | [0.498, 0.647] |
| hubert_large_er_L7 | 0.598 | [0.521, 0.670] |
| hubert_large_er_L8 | 0.550 | [0.473, 0.629] |
| hubert_large_er_L9 | 0.572 | [0.495, 0.647] |
| hubert_large_er_L10 | 0.590 | [0.511, 0.663] |
| hubert_large_er_L11 | 0.606 | [0.528, 0.678] |
| hubert_large_er_L12 | 0.649 | [0.570, 0.722] |
| hubert_large_er_L13 | 0.657 | [0.577, 0.733] |
| hubert_large_er_L14 | 0.659 | [0.580, 0.734] |
| hubert_large_er_L15 | 0.676 | [0.599, 0.748] |
| hubert_large_er_L16 | 0.676 | [0.601, 0.749] |
| hubert_large_er_L17 | 0.689 | [0.615, 0.761] |
| hubert_large_er_L18 | 0.697 | [0.622, 0.766] |
| hubert_large_er_L19 | 0.711 | [0.636, 0.780] |
| hubert_large_er_L20 | 0.713 | [0.638, 0.781] |
| hubert_large_er_L21 | 0.719 | [0.651, 0.783] |
| hubert_large_er_L22 | 0.658 | [0.581, 0.735] |
| hubert_large_er_L23 | 0.623 | [0.545, 0.700] |
| hubert_large_er_L24 | 0.643 | [0.562, 0.717] |
| mit_formant | 0.482 | [0.404, 0.562] |
| mit_dmfcc | 0.516 | [0.434, 0.596] |
| mit_both | 0.498 | [0.418, 0.583] |
| text_emo | 0.712 | [0.638, 0.781] |

## 4. Findings

1. **Emotion-tuned speech encoders do not beat wav2vec2-base:** HuBERT-large-ER peaks at layers 18-20 (about 0.70); HuBERT-base-ER peaks at layer 9 (0.66).
2. **MIT/Harvard vocal-coordination eigenspectra are at chance** on 239 speaker-independent participants (formant 0.48, dMFCC 0.52), versus a dev AUC of 0.71 on 35 participants in Williamson et al. 2016.
3. **Emotion-tuned text helps slightly:** emotion-RoBERTa-large reaches 0.712 vs BERT-T2 0.704, and the inner CV chose it in most fusion folds, but the fusion gain (+0.007) is within noise.
4. **A larger search pool lowered the selection-honest voice estimate** (0.694 in 6C, 0.684 in 6D): more candidates means more chances to choose a fold-specific winner. The data supports a voice AUC of about 0.68-0.71.
