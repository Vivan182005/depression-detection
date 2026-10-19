# Phase 7 Results: Full-Cohort Accuracy Optimisation (N = 272)

Protocol: `experiments/research_v2/prereg_phase7_full_cohort.md`. Pool: 239 development + 33 formerly-sealed participants (N = 272, 85 depressed; WoZ 187, Avatar 85). Repeated nested CV (5 folds x 5 seeds, speaker-disjoint); every hyperparameter, config choice, ensemble member, stacking weight, and threshold is learnt inside the training folds. **Exploratory, cross-validated estimates.** The one-shot held-out result remains V7 AUC 0.694 on the sealed 33 (FINAL_RESULTS.md).

Majority-class accuracy (always 'not depressed'): **68.8%**.

## 1. Procedures

| Procedure | AUC | 95% CI | Acc. (max-acc) | Bal. acc. (max-acc) | Sens. / Spec. (max-acc) | Acc. (Youden) | Bal. acc. (Youden) | Sens. / Spec. (Youden) |
|:--|--:|:--|--:|--:|:--|--:|--:|:--|
| P1 best single voice (logistic) | **0.719** | [0.652, 0.784] | 69.3% | 59.5% | 33% / 86% | 63.2% | 63.1% | 63% / 64% |
| P2 best single voice (PHQ regression) | **0.741** | [0.677, 0.800] | 70.4% | 60.9% | 36% / 86% | 65.9% | 65.2% | 64% / 67% |
| P3 top-3 voice ensemble | **0.742** | [0.674, 0.803] | 72.2% | 64.1% | 42% / 86% | 66.3% | 65.0% | 62% / 68% |
| P4 top-3 voice + best text (stack) | **0.764** | [0.698, 0.824] | 74.3% | 67.0% | 48% / 87% | 69.7% | 68.1% | 64% / 72% |

## 2. Bias check (headline P4, per-domain max-accuracy threshold)

| Group | n | AUC | Accuracy | Sensitivity | Specificity |
|:--|--:|--:|--:|--:|--:|
| woz | 187 | 0.750 | 73.6% | 48% | 85% |
| avatar | 85 | 0.794 | 72.5% | 50% | 84% |
| female | 102 | 0.769 | 70.6% | 53% | 81% |
| male | 170 | 0.751 | 74.8% | 44% | 86% |

Subgroup AUCs lie within about ±0.04 of each other (0.75-0.79): no large domain or gender bias. Men have somewhat lower sensitivity.

## 3. Effect of the fixes

- **Avatar domain:** the frozen V7 reached AUC 0.57 on the 24 sealed Avatar participants; P4 reaches 0.79 across all 85 Avatar participants.
- **PHQ regression (F4)** is the largest single gain (P1 0.719 -> P2 0.741).
- **Ensembling** adds accuracy (P3 72.2%), and **text fusion** adds AUC (P4 0.764).
- **Per-domain standardisation (F1)** was chosen inside some folds but not consistently. **Per-domain thresholds (F2)** did not beat a single threshold (P4 73.2% vs 74.3%).

## 4. Final deployable model (`models/final_phase7_p4.joblib`)

P4 refit on all 272 participants: voice = hubert_large_er_L19 (reg), hubert_large_er_L20 (reg), w2v2_L6 (reg); text = text_emo (reg); meta weights 2.11 / 2.11. Thresholds: max-accuracy 0.538, Youden 0.292. Its expected performance on new people is the nested-CV estimate above (AUC ~0.76), **not** its in-sample fit.

## 5. Caveats

- The fixes were motivated by errors on the 33 formerly-sealed participants, who are part of this CV pool, so a small optimistic bias is possible.
- No untouched test set remains. A new external sample (for example DAIC+) is needed to confirm 0.76.
- Choosing P4 as the headline after comparing four procedures on the same CV adds mild selection optimism.
