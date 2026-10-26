# Phase 8 Results: Augmentation, Few-Shot and One-Shot Learning

**Date:** 2026-10-26 | **Plan:** `experiments/research_v2/prereg_phase8_fewshot_augmentation.md` (declared before running, including the Section 4 pre-run corrections)
**Pool:** all 272 participants, repeated nested CV (5 outer folds x seeds 0-4, speaker-disjoint). Every augmentation, fit, hyperparameter and threshold was set inside the training folds.
**Baselines:** always guessing "not depressed" scores **68.8%**. Phase 7 P4 (the current final model) scores **AUC 0.764, accuracy 74.3%**.

## 1. Results

| ID | Method (supervisor's idea) | Voice AUC | Fused AUC [95% CI] | vs A0 (same bases) | Accuracy | Balanced acc. | Youden sens. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A0 | Ridge, fixed 3 bases (no new technique) | 0.759 | 0.785 [0.722, 0.843] | n/a | 75.2% | 68.3% | 63% |
| A1 | Augmentation: segment bagging | 0.756 | 0.782 [0.718, 0.839] | -0.003 | 75.4% | 67.5% | 65% |
| A2 | Augmentation: + audio noise/gain/telephone | 0.750 | 0.778 [0.713, 0.837] | -0.008 | 75.1% | 67.2% | **68%** |
| A3 | Augmentation: + test-time averaging | 0.748 | 0.778 [0.714, 0.835] | -0.007 | **75.8%** | 68.8% | 62% |
| A4 | Mixup (extra) | 0.762 | 0.787 [0.725, 0.844] | +0.001 | 75.0% | 67.8% | 66% |
| D1 | Class-balanced weights (extra) | 0.765 | **0.788** [0.726, 0.844] | +0.002 | 75.0% | 67.4% | 65% |
| C1 | One-shot: nearest class prototype | 0.614 | 0.744 [0.676, 0.804] | **-0.042** | 72.8% | 63.9% | 64% |
| C2 | One-shot: k-nearest neighbours | 0.614 | 0.746 [0.677, 0.807] | **-0.040** | 73.2% | 64.2% | 62% |
| B1 | Few-shot: prototypical network | 0.685 | 0.755 [0.687, 0.815] | **-0.031** | 73.5% | 65.2% | 62% |

Accuracy and balanced accuracy are at the in-fold max-accuracy threshold. Youden sensitivity is the share of depressed people caught at the balanced threshold. **Bold** in the "vs A0" column marks gains or losses whose 95% CI excludes zero (C1, C2, B1 are significantly worse).

Subgroup AUCs (fused): A0 is WoZ 0.771, Avatar 0.816, female 0.800, male 0.771. A4 and D1 are within ±0.006 of A0 everywhere. C1, C2 and B1 are 0.03-0.06 lower in every subgroup.

## 2. Verdict on each idea

- **Augmentation (A1-A3): no reliable gain.** AUC moved -0.003 to -0.008 against A0 (CIs include zero). Accuracy rose by at most +0.6 points (A3, 75.8%), well inside noise. The one real effect is that audio augmentation (A2) catches more depressed people at the balanced threshold (68% vs 63%) at the same accuracy. It does not close the Avatar gap, which A0 had already narrowed (Avatar AUC 0.816).
- **Few-shot, prototypical network (B1): significantly worse** (-0.031 AUC, CI excludes zero). Few-shot methods are designed for a handful of examples per class. With 272 labelled people, a regularised regression uses the data better.
- **One-shot, nearest prototype and kNN (C1, C2): significantly worse** (about -0.04 AUC). Comparing a new person to a class average or to their nearest neighbours throws away most of the information in the 1024-dimensional embeddings.
- **Extras, mixup and class balancing (A4, D1): neutral** (+0.001 and +0.002 AUC).

## 3. Adoption decision

The pre-declared rule (AUC ≥ 0.784 or accuracy ≥ 76.3%) is technically met on AUC by A0, A4 and D1. But **A0 uses no new technique**: its +0.021 over P4 comes from fixing the three voice bases. Those bases were chosen by the final Phase 7 model using all 272 labels, which is the optimism Section 4 of the plan warned about. A4 and D1 add nothing beyond A0. Under the Section 4 correction this counts as base selection, not a benefit of the technique, so **no Phase 8 method is adopted**.

**The final model is unchanged** (`models/final_phase7_p4.joblib`, honest estimate AUC 0.764, accuracy 74.3%). It already uses exactly the A0 voice bases. The 0.785 figure is not quoted as an accuracy estimate, because it is optimistic.

## 4. What this means for the project

On this dataset, with about 270 people, the accuracy ceiling comes from the signal itself, not the learning method: four different families of techniques all land within ±1.5 points of 75%. This matches the DAIC-WOZ literature, where speaker-independent voice models report roughly 70-78%. A defensible statement: *"Supervisor-suggested augmentation, few-shot (prototypical network) and one-shot (prototype / kNN) learning were each tested under leakage-free nested cross-validation. None improved on the regression baseline, and the few- and one-shot methods were significantly worse, which is consistent with the data being sufficient for standard supervised learning."*

Files: `experiments/research_v2/phase8_results.json`, `phase8_oof_predictions.csv`, `phase8_run.log`.
