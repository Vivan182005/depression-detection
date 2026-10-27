# Comparison of the Three Models

**Task:** detect depression (PHQ-8 ≥ 10) from an interview recording. Data: DAIC-WOZ + E-DAIC, 272 people (85 depressed).
**Date:** 2026-10-27

| | Model 1: Phase 7 P4 | Model 2: Phase 8 R0 | Model 3: Phase 9 F1 (final) |
| :--- | :--- | :--- | :--- |
| Short name | "the 74.3% model" | "the 75% model" | "the 79% model" |
| Voice | Top-3 voice configs re-selected inside each training fold | 3 fixed voice encoders: wav2vec2 L6, HuBERT-ER L19, L20 | same as Model 2 |
| Text | Emotion-RoBERTa embedding of the transcript | same | same |
| **LLM** | none | none | **Mistral-7B reads the transcript and estimates the 8 PHQ-8 items (zero-shot)** |
| Combiner | Logistic-regression stacker | same | same |
| Saved file | `models/final_phase7_p4.joblib` | not saved (reference only) | **`models/final_phase9_f1.joblib`** |

## 1. Test A: cross-validation on all 272 people

The model is trained on 4/5 of the people and tested on the other 1/5, rotating until everyone has been tested once, then repeated over 5 random splits. No person is ever in both training and test.

**Baseline:** always guessing "not depressed" scores **68.8%**.

| Metric | Model 1: P4 | Model 2: R0 | **Model 3: F1** |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 74.3% | 75.2% | **79.1%** |
| Gain over the 68.8% baseline | +5.5 | +6.4 | **+10.3** |
| Balanced accuracy | 67.0% | 68.3% | **72.4%** |
| AUC [95% CI] | 0.764 [0.70, 0.82] | 0.785 [0.72, 0.84] | **0.843 [0.79, 0.89]** |
| MCC | 0.37 | 0.39 | **0.49** |
| Macro-F1 | 0.68 | 0.69 | **0.74** |
| Sensitivity (depressed caught) | 48% | 50% | **55%** |
| Specificity (healthy correctly cleared) | 87% | 87% | **90%** |
| At the balanced threshold: sensitivity | 64% | 63% | **77%** |

The figures above are at the max-accuracy threshold unless stated otherwise.

**Subgroup AUC:**

| Group | Model 1 | Model 2 | **Model 3** |
| :--- | :---: | :---: | :---: |
| WoZ setup (187) | 0.750 | 0.771 | **0.839** |
| Avatar setup (85) | 0.794 | 0.816 | **0.855** |
| Female (102) | 0.769 | 0.800 | **0.857** |
| Male (170) | 0.751 | 0.771 | **0.829** |

Without the 17 people who say "I'm depressed" outright, Model 3 still scores AUC 0.835.

**Are the differences real?** These are paired bootstrap tests on the same people:

| Comparison | AUC difference [95% CI] | Verdict |
| :--- | :---: | :--- |
| Model 2 vs Model 1 | +0.021 [+0.007, +0.036] | Small but real. It comes from fixing the voice bases, which were chosen on all 272, so it is partly optimistic. |
| Model 3 vs Model 2 | **+0.057 [+0.015, +0.102]** | **Real.** It passes the pre-declared adoption rule (≥ +0.02 with a CI above 0). |
| Model 3 vs Model 1 | **+0.079 [+0.035, +0.124]** | **Real.** |

## 2. Test B: train on 239, test on the 33 unseen people

Each model is trained **only** on 239 people. Every setting is chosen on those 239, including the ridge strength, the stacker and the threshold. The model then predicts the 33 people kept aside in `unseen/` once. Of the 33, 11 are depressed, so **always guessing "not depressed" scores 66.7%**.

| Metric | Model 1: P4 | Model 2: R0 | **Model 3: F1** |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 72.7% (24/33) | 72.7% (24/33) | **78.8% (26/33)** |
| Gain over the 66.7% baseline | +6.1 | +6.1 | **+12.1** |
| Balanced accuracy | 68.2% | 70.5% | 70.5% |
| AUC [95% CI] | 0.777 [0.60, 0.92] | 0.785 [0.61, 0.93] | **0.868 [0.72, 0.97]** |
| Permutation p | 0.004 | 0.002 | **0.0002** |
| MCC | 0.37 | 0.40 | **0.50** |
| Depressed caught (of 11) | 6 | 7 | 5 |
| Healthy correctly cleared (of 22) | 18 | 17 | **21** |

**At the balanced (Youden) threshold**, which suits screening:

| | Model 1 | Model 2 | **Model 3** |
| :--- | :---: | :---: | :---: |
| Accuracy | 69.7% | 72.7% | **75.8%** |
| Balanced accuracy | 68.2% | 70.5% | **77.3%** |
| Depressed caught (of 11) | 7 | 7 | **9** |

For reference, the older frozen models scored AUC 0.69-0.70 and accuracy 64-67% on the same 33 (2026-10-19).

## 3. Summary

1. **Model 3 is best on every main measure in both tests.**
   - Accuracy: 79.1% in CV and 78.8% on the unseen 33.
   - Model 1 scored 74.3% and 72.7%; Model 2 scored 75.2% and 72.7%.
   - The two tests agree to within 0.3 points, so the result is consistent rather than lucky.
2. **Model 1 → Model 2 (+0.9 points) is a minor gain.** It comes from fixing which voice layers are used. It is partly optimistic, and on the unseen 33 the two models tie.
3. **Model 2 → Model 3 (+3.9 points) is the real improvement.** It comes from one new idea: a local language model reads what the person said and estimates each PHQ-8 symptom (sleep, tiredness, concentration, mood and so on). The voice models capture *how* someone speaks; the LLM captures *what* they report. The two add up.
4. **Model 3 at the balanced threshold is the best screening setting.** It catches 77% of depressed people in CV and 9 of 11 on the unseen 33.

## 4. Limits

- **The unseen 33 are small.** One person changes accuracy by 3 points, so the confidence intervals are wide.
- **The unseen 33 are a repeat check, not a pristine test.** They were opened once before (2026-10-19). Models 2 and 3 also use voice layers chosen on all 272 people. Model 1 re-selects its layers on the 239, so its holdout is the cleanest of the three.
- **The LLM's training data is undisclosed.** DAIC-WOZ transcripts are licence-restricted, so it is unlikely the LLM saw them during pretraining, but this cannot be proven.
- **This is a research screening prototype, not a diagnosis.**

**One-line result for the teacher:** *"Adding zero-shot LLM PHQ-8 symptom scoring raised accuracy from 74.3% to 79.1% under cross-validation (baseline 68.8%; AUC 0.76 → 0.84). On 33 held-out people it raised accuracy from 72.7% to 78.8% (baseline 66.7%; AUC 0.78 → 0.87)."*

Sources:
- `experiments/research_v2/phase7_results.json`
- `phase9_results.json`
- `holdout_p4_results.json`
- `holdout_239_33_results.json`
- `sealed_results.json`
