# Corrected Runtime Feasibility Benchmark Report (Task 6A-1R / R1)

**Date:** 2026-10-14  
**Project:** Voice-Primary Depression Detection Research Prototype  
**Branch:** `research-upgrade-2026-10-13`  
**Protocol Version:** Pre-registration Amendment v1.2  
**Benchmark Script:** [`experiments/research_v2/run_runtime_feasibility.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/run_runtime_feasibility.py)  
**Results Data:** [`experiments/research_v2/runtime_feasibility_results.json`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/runtime_feasibility_results.json)

---

## 1. Review Finding & Root Cause Resolution (R1)

### 1.1 Initial Benchmark Mismatch
The preliminary Task 6A-1 runtime feasibility report described:
- **V2** as eGeMAPS (135d) + handcrafted summaries (165d) with a Ridge / Logistic model (300d).
- **V10** as concatenated Wav2Vec2 layer 12 (768d) + WavLM layer 12 (768d) + eGeMAPS (135d) = 1671d.

### 1.2 Preregistered Confirmatory Hypotheses
The preregistration (Amendment v1.2) strictly specifies:
- **Variant V2 ($H_1$):** VAD-segment-aggregated handcrafted voice features (median, IQR, p10, p90, std, early-vs-late slope across 55 base features = **330 dimensions**) + $L_2$ Logistic Regression with nested hyperparameter tuning ($C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$) via inner 4-fold CV, 5-fold outer StratifiedGroupKFold, and out-of-fold predictions averaged across 5 selection seeds ($0, 1, 2, 3, 4$).
- **Variant V10 ($H_2$):** `microsoft/wavlm-base-plus` Layer 8 representations mean-pooled over participant speech frames (**768 dimensions**) + $L_2$ Logistic Regression with the identical nested tuning, 5-fold outer CV, and 5-seed averaging.

### 1.3 Root Cause Classification
The discrepancy was caused by:
1. **Model Configuration Proxy:** The preliminary benchmark script evaluated synthetic placeholder matrices (300d and 1671d) with fixed $C=1.0$ (omitting inner CV hyperparameter search) and evaluated only 1 seed rather than averaging 5 selection seeds.
2. **Text Artifacts:** The report header retained phrasing from Phase 5 (Ridge regression, Wav2Vec2 Layer 12) rather than the Phase 6A preregistered Logistic Regression specifications.
3. **Correction:** The benchmark script was rewritten to load the exact $239 \times 330$ cached feature matrix from `feature_store_v3/` for V2 and the exact $239 \times 768$ float32 matrix for V10, execute the full nested inner 4-fold tuning grid, fit all 5 outer folds across all 5 selection seeds ($5 \times 5 \times 21 = 525$ fits per replication), and permute labels strictly within Domain $\times$ Gender strata.

---

## 2. Hardware & Execution Environment

The corrected benchmark was executed on the local development workstation:
- **System Architecture:** Darwin macOS (`arm64`, Apple Silicon, Darwin 27.0.0)
- **CPU Cores:** 8 physical/logical cores
- **RAM Constraint:** 8.0 GB system boundary constraint
- **Python Environment:** Python 3.12.8 in `.venv_opensmile` (scikit-learn 1.6.0, NumPy 2.2.0, PyTorch 2.14.1)
- **Label Status:** Permuted clinical labels only (Domain $\times$ Gender stratum permutation); **zero observed predictive evaluation against real PHQ outcomes**.

---

## 3. Pipeline Stages & Computational Timing

Each replication executes the full computationally required pipeline:
1. **Fold Generation:** 5-fold outer splits from permuted labels across 5 selection seeds ($s \in \{0, 1, 2, 3, 4\}$).
2. **In-Fold Preprocessing:** `StandardScaler` and outlier winsorisation ($P_{01}^{\text{train}}, P_{99}^{\text{train}}$) fitted strictly on the outer training fold.
3. **Nested Hyperparameter Tuning:** Inner 4-fold cross-validation grid search over $C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$ using `GridSearchCV(scoring='roc_auc')`.
4. **Model Fitting:** Refitting best estimator on full outer training fold.
5. **Out-of-Fold Scoring:** Probabilistic prediction on outer validation fold.
6. **Seed Aggregation:** Averaging OOF predicted probabilities across the 5 selection seeds.
7. **Permutation Statistic:** Calculating participant-level ROC-AUC.

### Why Feature Caches are Legitimately Reused
Acoustic feature extraction (OpenSMILE eGeMAPS functionals, Parselmouth pitch/jitter/shimmer/formants, and WavLM forward passes) depends purely on raw acoustic waveforms and frozen model weights. It has **zero dependency on participant depression labels**. Caching these representations in `feature_store_v3/` is completely label-independent and does not reuse any label-dependent fitted state. All data transformations that depend on training fold membership or labels (scaling, winsorisation, inner CV tuning, estimator fitting, and metric computation) are executed afresh inside each replication.

---

## 4. Measured Runtimes & Extrapolation to $R = 999$

### 4.1 Measured $R = 5$ Permuted Dry-Runs

| Variant | Input Dims | Cohort Size ($N$) | Run 1 (s) | Run 2 (s) | Run 3 (s) | Run 4 (s) | Run 5 (s) | Mean $\pm$ SD (s) | Total 5 Runs (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2 (Handcrafted)** | **330** | 239 | 1.616 | 1.576 | 1.625 | 1.645 | 1.601 | **1.6125 $\pm$ 0.023 s** | **8.062 s** |
| **V10 (WavLM L8)** | **768** | 239 | 1.240 | 1.237 | 1.233 | 1.253 | 1.244 | **1.2415 $\pm$ 0.007 s** | **6.207 s** |

### 4.2 Per-Stage Wall-Clock Time Breakdown

| Pipeline Stage | Variant V2 (330d) | Variant V10 (768d) | Notes |
| :--- | :---: | :---: | :--- |
| **Stratified Fold Generation** | 0.0031 s (0.2%) | 0.0013 s (0.1%) | 5 outer splits $\times$ 5 seeds |
| **In-Fold Preprocessing** | 0.0335 s (2.1%) | 0.0181 s (1.5%) | StandardScaler + Winsorisation on train fold |
| **Nested Inner CV Tuning** | **1.5735 s (97.6%)** | **1.2196 s (98.2%)** | Inner 4-fold CV across 5 values of $C$ |
| **Out-of-Fold Scoring** | 0.0017 s (0.1%) | 0.0018 s (0.1%) | `predict_proba` on validation fold |
| **Seed Aggregation & Metric** | 0.0006 s (<0.1%) | 0.0006 s (<0.1%) | 5-seed mean & ROC-AUC calculation |
| **Total per Replication** | **1.6125 s** | **1.2415 s** | $525$ model fits per replication |

### 4.3 Extrapolation to $R = 999$ Permutations

$$\text{Total Time for Complete Experiment: } T = 1000 \times \bar{T}_{\text{perm}}$$
*(1 unpermuted baseline run + 999 permutation replications)*

| Metric | Variant V2 ($H_1$) | Variant V10 ($H_2$) | Protocol Feasibility Boundary |
| :--- | :---: | :---: | :---: |
| **Average Time per Run** | **1.6125 seconds** | **1.2415 seconds** | — |
| **Extrapolated Wall-Clock Time ($R=999$)** | **1612.49 seconds** | **1241.49 seconds** | $\le 172,800\text{ s}$ (48 hours) |
| **Extrapolated Time in Minutes** | **26.87 minutes** | **20.69 minutes** | $\le 2880\text{ min}$ |
| **Extrapolated Time in Hours** | **0.4479 hours** | **0.3449 hours** | $\le 48.0\text{ hours}$ |
| **Budget Utilization** | **0.93% of limit** | **0.72% of limit** | — |
| **Pre-registered Stop Condition** | **NOT TRIGGERED** | **NOT TRIGGERED** | Feasible for $R=999$ |

### 4.4 Extrapolation Assumptions & Limitations
1. **Sequential vs. Parallel Execution:** The extrapolation assumes single-threaded sequential execution (`n_jobs=1`). Under 4-core parallelization, runtime would reduce to approximately **7 minutes for V2** and **5 minutes for V10**.
2. **Linear Scaling Assumption:** Permutation runs are statistically independent. Linear scaling holds because memory consumption ($< 150$ MB) does not cause swapping or cache thrashing on the 8 GB machine.
3. **Linear Extrapolation is Not Exact:** Small operating system scheduling jitter, thermal throttling, and garbage collection introduce slight variance (observed SD $\approx 0.02$ s). However, even a 50% slowdown would still leave execution under 45 minutes, well below the 48-hour gate.

---

## 5. Conclusion & Protocol Decision

> [!NOTE]
> **Confirmatory Feasibility Confirmed:**  
> The complete 999-permutation pipeline for both confirmatory variants (V2 and V10), including nested inner 4-fold hyperparameter tuning and 5-seed aggregation, requires **under 27 minutes for V2** and **under 21 minutes for V10**.  
> The 48-hour pre-registered runtime threshold is **not exceeded**.  
> **No reduction in $R$ ($999$) or reclassification of confirmatory hypotheses $H_1/H_2$ to exploratory is required.**
