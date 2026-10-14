# Synthetic Validation of Complete Testing Machinery

**Date:** 2026-10-14  
**Project:** Voice-Primary Depression Detection Research Prototype  
**Branch:** `research-upgrade-2026-10-13`  
**Protocol Version:** Pre-registration Amendment v1.2  
**Test Suite:** [`experiments/research_v2/test_synthetic_machinery.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/test_synthetic_machinery.py)  
**Results Data:** [`experiments/research_v2/synthetic_validation_results.json`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/synthetic_validation_results.json)

---

## 1. Governance & Boundary Verification

> [!IMPORTANT]
> **Strict Label-Blind Execution:**  
> In accordance with Task 6A-1 constraints, **zero real PHQ-8 scores, item totals, or depression severity labels** were read, processed, or evaluated during this validation. All tests operated exclusively on synthetic features and synthetic binary outcomes generated from fixed random seeds.

---

## 2. Section A: Noise-Null Calibration Smoke Test

### 2.1 Protocol & Generation Setup
- **Sample Size:** $N = 239$ synthetic participants (matching the exact active development cohort size), with multi-segment clustering (1 to 5 segments per participant, total $N_{\text{rows}} \approx 720$).
- **Features:** 30 independent standard Gaussian noise dimensions ($X \sim \mathcal{N}(0, I)$). Under this data-generating process, the true population association between features and outcome is strictly zero.
- **Outcomes:** Synthetic Bernoulli outcomes at participant level ($y \sim \text{Bernoulli}(0.50)$).
- **Cross-Validation:** 5-fold `StratifiedGroupKFold` ensuring all segments of any participant are assigned to exactly one fold.
- **Permutations:** $R = 99$ participant-level label permutations per dataset.
- **Independent Synthetic Datasets Evaluated:** $M = 40$ datasets (seeds 2000 through 2039).

### 2.2 Calibration Results & Type I Error

$$\text{Finite-Sample Permutation } p\text{-value: } p = \frac{1 + \sum_{r=1}^R \mathbb{I}(s^{(r)} \ge s^{(0)})}{1 + R}$$

- **Nominal Significance Level ($\alpha$):** $0.05$
- **Observed False Positive Rejections:** **$3$ out of $40$ datasets** ($\text{FPR} = 7.5\%$)
- **Numerator / Denominator:** $3 / 40$
- **95% Wilson Score Confidence Interval:** $[2.6\%, 19.9\%]$  
  *(The 95% confidence interval cleanly covers the nominal 5.0% error rate, confirming empirical calibration under the null).*
- **Mean Observed AUC under Null:** $0.502 \pm 0.038$ (Theoretical expectation: $0.500$)
- **Mean Permutation $p$-value:** $0.476 \pm 0.281$ (Theoretical expectation under uniform distribution: $0.500$)
- **Failed Runs:** $0 / 40$ (100% completion across all 40 independent datasets).

---

## 3. Section B: Injected-Signal Sensitivity

To confirm that the testing pipeline has statistical power to detect true clinical associations when present, a prespecified signal was injected into feature 0:

- **Signal Association:** $x_{i, 0} = 0.8 \cdot y_i + \epsilon_i$, representing a moderate-to-strong effect size (Cohen's $d \approx 0.80$).
- **Zero-Signal Control:** $x_{i, 0} = \epsilon_i$ where $\epsilon \sim \mathcal{N}(0, 1)$.

| Condition | Observed AUC | Permutation $p$-value ($R=99$) | Decision ($\alpha = 0.05$) | Pipeline Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Zero-Signal Control** | **0.513** | **$p = 0.410$** | Fail to Reject Null | Correctly avoids false positive |
| **Injected-Signal Test** | **0.776** | **$p = 0.010$** | **Reject Null ($p \le 0.01$)** | Correctly and sensitively detects signal |

Both conditions behaved exactly as expected: the zero-signal control showed no significant effect ($p = 0.410$), whereas the injected signal produced a substantial AUC ($0.776$) and achieved the minimum possible $p$-value under $R=99$ ($p = \frac{1+0}{1+99} = 0.010$).

---

## 4. Section C: Leakage & Implementation Assertions

Six formal, executable assertions were evaluated to audit the pipeline against subtle implementation bugs and data leakage:

1. **Sealed-Cohort Quarantine Guard:**
   - Evaluated against sealed participant IDs ($300, 303, 304, 615, 619, 624, 656$).
   - **Result:** PASSED. Supplying any sealed ID raises an immediate, fatal `AssertionError: CRITICAL SEALED COHORT VIOLATION`.
2. **Label-Blind Extraction Guard:**
   - Evaluated by injecting forbidden metadata keys (`phq8_score`, `phq_score`, `class_id`, `binary_label`, `phq8_1`, `severity`, `phq_total`).
   - **Result:** PASSED. Deliberately supplying label fields raises an immediate, fatal `AssertionError: CRITICAL LABEL LEAKAGE VIOLATION`.
3. **Participant-Grouped Fold Isolation:**
   - Evaluated across all 5 folds of `StratifiedGroupKFold`.
   - **Result:** PASSED. Participant intersection between train and validation folds was confirmed to be identically empty ($\text{Train} \cap \text{Val} = \emptyset$).
4. **Preprocessing Isolation (Training-Data Only):**
   - Evaluated on feature transformation (`StandardScaler`).
   - **Result:** PASSED. Scaler parameters ($\mu, \sigma^2$) are fit strictly on training fold rows; validation rows are never observed during parameter estimation.
5. **Finite-Sample Permutation Correction:**
   - Verified that the permutation $p$-value formula incorporates the $+1$ numerator and denominator correction ($\frac{1 + \text{count}}{1 + R}$).
   - **Result:** PASSED. Minimum achievable $p$-value with $R=99$ is strictly $0.010$, preventing erroneous claims of $p = 0.000$.
6. **Participant-Level Out-of-Fold Aggregation:**
   - Evaluated on clustered segment predictions.
   - **Result:** PASSED. Multiple segments per session are averaged to exactly one prediction per participant before computing ROC AUC.
