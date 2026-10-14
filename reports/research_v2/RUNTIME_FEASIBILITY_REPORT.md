# Runtime Feasibility Benchmark Report (Task C7)

**Date:** 2026-10-14  
**Project:** Voice-Primary Depression Detection Research Prototype  
**Branch:** `research-upgrade-2026-10-13`  
**Protocol Version:** Pre-registration Amendment v1.2  
**Benchmark Script:** [`experiments/research_v2/run_runtime_feasibility.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/run_runtime_feasibility.py)  
**Results Data:** [`experiments/research_v2/runtime_feasibility_results.json`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/runtime_feasibility_results.json)

---

## 1. Hardware & Environment Specifications

The runtime feasibility benchmark was conducted on the development workstation with zero label access:
- **System Architecture:** Darwin macOS (`arm64`, Apple Silicon)
- **CPU Cores:** 8 physical/logical cores
- **RAM Allocation:** 8.0 GB system boundary constraint
- **Python Environment:** Python 3.12.8 in `.venv_opensmile` (PyTorch 2.14.1 with MPS support, scikit-learn 1.6.0, NumPy 2.2.0)
- **Outcome Labels:** Synthetic / permuted binary labels across $N = 239$ development participants.

---

## 2. Benchmark Methodology

The benchmark measured the wall-clock time of the complete, leak-free 5-fold `StratifiedGroupKFold` cross-validation pipeline across $R = 5$ permuted runs for the two primary confirmatory variants:
1. **Variant V2 (Acoustic Primary Baseline):** OpenSMILE eGeMAPS robust functionals ($135\text{d}$) + participant-level handcrafted acoustic segment summaries ($165\text{d}$) = **$300$ total input dimensions**.
2. **Variant V10 (Pretrained Audio Fusion):** Concatenated Wav2Vec2 layer 12 ($768\text{d}$) + WavLM layer 12 ($768\text{d}$) + eGeMAPS ($135\text{d}$) = **$1671$ total input dimensions**.

Each permutation run executes:
1. Participant-level label shuffling (`RandomState`).
2. Stratified group K-fold splitting ($K = 5$).
3. Feature scaling (`StandardScaler`) fit strictly on training folds.
4. Estimator fitting (`LogisticRegression` with L2 regularization).
5. Out-of-fold probabilistic scoring and participant-level aggregation.
6. Metric computation (`roc_auc_score`).

---

## 3. Timing Results & Extrapolation to $R = 999$

### 3.1 Per-Stage Wall-Clock Timing

$$\text{Extrapolated Time for } R=999: \quad T_{R=999} = T_{\text{baseline}} + 999 \times \bar{T}_{\text{perm}}$$

| Benchmark Metric | Variant V2 (Audio Handcrafted + eGeMAPS, 300d) | Variant V10 (Pretrained Embeddings + eGeMAPS, 1671d) |
| :--- | :---: | :---: |
| **Number of Development Participants** | $N = 239$ | $N = 239$ |
| **Total Segment Rows Evaluated** | 956 rows | 956 rows |
| **Input Feature Dimensionality** | 300 dimensions | 1671 dimensions |
| **Baseline (Unpermuted) 5-Fold CV Time** | 0.954 s | 0.095 s |
| **Total Time for $R = 5$ Permuted Runs** | 0.388 s | 0.581 s |
| **Average Wall-Clock Time per Permutation** | **0.0775 s** | **0.1161 s** |
| **Extrapolated Wall-Clock Time for $R = 999$** | **78.38 seconds** | **116.05 seconds** |
| **Extrapolated Time in Hours** | **0.0218 hours (~1.3 minutes)** | **0.0322 hours (~1.9 minutes)** |
| **Pre-registered Stop Threshold** | 48.0 hours | 48.0 hours |
| **Feasibility Assessment** | **FEASIBLE (Exceeds by 0.04% of limit)** | **FEASIBLE (Exceeds by 0.07% of limit)** |

### 3.2 Extrapolation Assumptions & Robustness
- **Caching State Assumption:** Pre-computed feature representations (eGeMAPS 135d, handcrafted summaries, and frozen pretrained embeddings) are cached in `feature_store_v3/` prior to model evaluation. Thus, feature extraction occurs once and is not repeated inside the permutation loop.
- **Linear Scaling Assumption:** Permutation runs are statistically independent. The extrapolation assumes linear scaling $O(R)$ for fold splitting and linear estimator fitting, which holds strictly under non-saturating CPU load.
- **Memory Footprint:** Peak memory usage during CV fitting was $< 120$ MB, well within the 8 GB device boundary.

---

## 4. Conclusion & Decision

> [!NOTE]
> **Stop Condition Not Triggered:**  
> The complete 999-permutation pipeline for both confirmatory variants (V2 and V10) requires **less than 2 minutes each** to execute on the full $N=239$ development cohort.  
> The 48-hour runtime threshold is **not exceeded**.  
> **No reduction in $R$ (999) or reclassification of confirmatory hypotheses $H_1/H_2$ is required.**
