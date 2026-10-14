# Synthetic Statistical Validation Audit Report (Task 6A-1R / R2)

**Date:** 2026-10-14  
**Project:** Voice-Primary Depression Detection Research Prototype  
**Branch:** `research-upgrade-2026-10-13`  
**Protocol Version:** Pre-registration Amendment v1.2  
**Audit Script:** [`experiments/research_v2/test_synthetic_machinery.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/test_synthetic_machinery.py)  
**Results Data:** [`experiments/research_v2/synthetic_validation_results.json`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/synthetic_validation_results.json)

---

## 1. Executive Summary & Review Scope (R2)

A technical review of the preliminary synthetic validation report required an audit to:
1. Trace the exact execution path of `test_synthetic_machinery.py`.
2. Determine whether the synthetic test exercises the exact confirmatory pipeline (participant-grouped outer folds, nested tuning, in-fold preprocessing, OOF aggregation, finite-sample p-value formula, Holm-Bonferroni adjustment over $m=3$ hypotheses).
3. Identify stages that were mocked, simplified, or bypassed.
4. Verify the injected-signal sensitivity and zero-signal control.
5. Clarify that 40 noise datasets serve as smoke-test evidence, **not asymptotic proof of calibration at $\alpha = 0.05$**.

---

## 2. Execution Path & Pipeline Stage Audit

| Pipeline Stage | Confirmatory Preregistered Standard | Synthetic Test Implementation | Status / Audit Finding |
| :--- | :--- | :--- | :--- |
| **Outer CV Partitioning** | 5-Fold `StratifiedGroupKFold` grouped by participant | 5-Fold `StratifiedGroupKFold` | **Exercised:** 100% participant fold isolation asserted across all folds. |
| **Label-Dependent Fold Generation** | Folds regenerated dynamically for each permutation $y^{(r)}$ | Folds regenerated dynamically per permutation | **Exercised:** Permutation loop re-splits folds based on permuted labels. |
| **In-Fold Preprocessing** | `StandardScaler`, winsorisation, imputation fitted on train fold only | `StandardScaler` fitted on train fold only | **Partially Exercised:** Scaler is strictly in-fold; winsorisation/imputation not needed on Gaussian noise. |
| **Nested Hyperparameter Tuning** | Inner 4-fold grouped CV tuning $C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$ | Fixed $C=1.0$ (no inner CV) | **Bypassed:** Omitted inside the 40-dataset calibration loop to maintain test suite speed. |
| **Seed Aggregation** | Averaged across 5 selection seeds ($0, 1, 2, 3, 4$) | Single seed ($s=\text{seed}$) | **Bypassed:** Only 1 seed used in the 40-dataset noise loop. |
| **Out-of-Fold Aggregation** | Participant-level mean of segment prediction probabilities | Groupby `participant_id` mean of prediction probabilities | **Exercised:** Multi-segment rows correctly collapsed to 1 score per participant. |
| **Permutation Null Construction** | Permutation strictly within domain $\times$ gender strata | Global permutation across unique participants | **Simplified:** Did not partition synthetic noise into domain/gender strata. |
| **Finite-Sample $p$-Value Formula** | $p = \frac{1 + \sum \mathbb{I}(s_{\text{perm}} \ge s_{\text{obs}})}{R + 1}$ | $p = \frac{1 + \sum \mathbb{I}(s_{\text{perm}} \ge s_{\text{obs}})}{R + 1}$ | **Exercised:** Exact finite-sample correction strictly asserted ($p_{\min} = 1/(1+R)$). |
| **Multiplicity Control (Holm)** | Step-down procedure over $m=3$ confirmatory hypotheses | Simulated over $m=3$ hypotheses ($N=1,000$ runs) | **Exercised:** Added in R2 audit; confirms FWER is bounded at $\alpha \le 0.05$. |

---

## 3. Distinguishing Evidence Levels

### 3.1 Level 1: Unit-Test Correctness (100% Passed)
The test suite enforces 6 deterministic structural assertions:
1. **Sealed Cohort Quarantine Guard:** Confirms that any attempt to query or process any of the 33 sealed participant IDs (`[300, 303, 304, ...]`) raises an immediate fatal `AssertionError` (`CRITICAL SEALED COHORT VIOLATION`).
2. **Label-Blindness Guard:** Confirms that querying metadata with forbidden outcome keys (`phq8_score`, `phq_score`, `class_id`, `binary_label`, `severity`) raises `CRITICAL LABEL LEAKAGE VIOLATION`.
3. **Fold Isolation:** Asserts zero participant intersection between training and validation folds ($\text{Train} \cap \text{Val} = \emptyset$).
4. **Preprocessing Isolation:** Asserts `StandardScaler` fitted on training fold has zero contamination from validation fold.
5. **Finite-Sample Correction:** Asserts minimum p-value under $R=99$ is $0.010$, never $0.000$.
6. **OOF Score Aggregation:** Asserts multi-segment predictions aggregate to exactly one prediction per participant.

### 3.2 Level 2: Smoke-Test Machinery Feasibility (Passed)
- **Zero-Signal Control:** Gaussian noise features evaluated against random labels:
  - Observed AUC: $0.513$
  - Permutation $p$-value: $p = 0.410$ (Failed to reject null, as expected).
- **Injected-Signal Sensitivity:** Cohen's $d \approx 0.8$ signal added to feature 0:
  - Observed AUC: $0.776$
  - Permutation $p$-value: $p = 0.010$ (Exact minimum p-value under $R=99$, strong rejection).
  - Assertion passed: Pipeline detects prespecified true signal without failure.

### 3.3 Level 3: Statistical Calibration Evidence (Preliminary Smoke-Test, Not Asymptotic Proof)
- **Design:** $N = 40$ independent synthetic datasets, $R = 99$ permutations per dataset.
- **Empirical Rejections:** $3$ false-positive rejections out of $40$ datasets ($\text{FPR} = 7.5\%$).
- **95% Wilson Score Confidence Interval:** $[2.60\%, 19.86\%]$.
- **Mean Observed AUC under Null:** $0.502$ (Theoretical expectation: $0.500$).
- **Mean Permutation $p$-Value:** $0.476$ (Theoretical expectation for uniform $U(0,1)$: $0.500$).

> [!WARNING]
> **Statistical Power Limitation:**  
> While the empirical false positive rate ($7.5\%$) and its 95% Wilson interval ($[2.6\%, 19.9\%]$) encompass the nominal rate $\alpha = 0.05$, **$N = 40$ datasets is statistically underpowered to conclusively establish asymptotic calibration**.  
> The standard error with $N = 40$ is $\approx 3.5\%$. A pipeline with a true 12% error rate would fail to be rejected at $N = 40$.  
> Therefore, these findings demonstrate that the permutation engine operates without gross algorithmic bias; they do not constitute mathematical proof of exact asymptotic $\alpha$-calibration.

---

## 4. Multiplicity Control Audit (Holm-Bonferroni Step-Down, $m=3$)

The confirmatory family comprises $m = 3$ hypotheses ($H_1$: Handcrafted vs. Null, $H_2$: WavLM vs. Null, $H_3$: Stacking Synergy vs. Unimodal). Under the global null:
- Without multiplicity control, the family-wise error rate across 3 independent tests is:
  $$\text{FWER}_{\text{unadjusted}} = 1 - (1 - 0.05)^3 \approx 14.3\%$$
  Empirical simulation ($N = 1,000$ null runs): **$13.2\%$**.
- Under the **Holm-Bonferroni step-down procedure**:
  - Test ordered $p$-values: $p_{(1)} \le p_{(2)} \le p_{(3)}$.
  - Thresholds: $\alpha/3 = 0.0167$, $\alpha/2 = 0.025$, $\alpha/1 = 0.050$.
  - Empirical simulation ($N = 1,000$ null runs): **$5.0\%$**.
  - **Verdict:** Multiplicity control strictly enforces the family-wise error rate at $\le 0.05$.

---

## 5. Conclusion & Protocol Decision

> [!NOTE]
> **Synthetic Validation Status:**  
> The synthetic testing machinery is verified for code correctness, structural leakage guards, signal sensitivity ($p=0.01$), and multiplicity control ($5.0\%$).  
> The 40-dataset noise calibration is formally designated as **smoke-test validation of pipeline mechanics**, acknowledging the wide confidence interval ($[2.6\%, 19.9\%]$) inherent to $N=40$.  
> No modifications to the confirmatory testing family ($m=3$) or permutation count ($R=999$) are made.
