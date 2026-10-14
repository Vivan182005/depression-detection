# Acoustic Quality Criteria & Demographic Gender Audit

**Date:** 2026-10-14  
**Branch:** `research-upgrade-2026-10-13`  
**Dataset:** 239 Development Participants (178 WoZ Dev + 61 Avatar Dev; PID 306 excluded for missing audio)  
**Authoritative Labels:** [`experiments/research_v2/labels_official.csv`](../../experiments/research_v2/labels_official.csv)  
**Quality Audit Artifact:** [`experiments/research_v2/acoustic_quality_audit.csv`](../../experiments/research_v2/acoustic_quality_audit.csv)  

---

## Part 1: Acoustic Quality Criteria & Validity Rules

To ensure acoustic feature extraction is physiologically valid, robust to recording artifacts, and completely leak-free, we establish rigorous validity criteria for fundamental frequency ($f_0$), perturbation metrics (jitter, shimmer), spectral noise (HNR), and vocal tract resonances (formants $F_1, F_2, F_3$).

### 1.1 Validity Rules & Physiological Boundaries

| Metric | Feature Identifier / Method | Validity Criteria & Physiological Bounds | Failure Flag / Action |
| :--- | :--- | :--- | :--- |
| **Pitch ($f_0$)** | `F0semitoneFrom27.5Hz_sma3nz` / pYIN | - **Search Range:** $60\text{ Hz} \le f_0 \le 400\text{ Hz}$ ($13.5$ to $46.4$ semitones).<br>- **Octave Jumps:** Reject frame if $|\Delta \log_2(f_0)| > 0.8$ octaves within $\le 30\text{ ms}$.<br>- **Two-Pass Estimation:** Unsupervised session median $\tilde{f}_0$ computed on high-confidence voiced frames ($HNR > 5\text{ dB}$, voiced prob $> 0.7$). Pass 2 bounds search to $[0.5 \tilde{f}_0, 2.0 \tilde{f}_0]$. **No labels or gender metadata used.** | Invalidate unvoiced / octave-jump frames. If valid voiced duration $< 100\text{ ms}$ in a turn, mark turn invalid. |
| **Voiced Duration & Fraction** | Temporal frame aggregation | - **Minimum Voiced Duration:** $\ge 100\text{ ms}$ continuous voiced phonation per turn.<br>- **Voiced Fraction:** Turn voiced fraction $\ge 0.15$ (discards whispering, breathing, microphone bumps). | Turns with $< 100\text{ ms}$ voiced speech are excluded from turn-level prosodic aggregation. |
| **Jitter** | `jitterLocal_sma3nz` | - Relative frame-to-frame period perturbation.<br>- Valid range: $0.0\% \le \text{jitter} \le 12.0\%$ ($0.0 \le \text{val} \le 0.12$).<br>- Calculated only across contiguous voiced cycles with stable $f_0$. | Values $> 0.12$ ($12\%$) indicate cycle-detection failure; flagged as invalid estimate. |
| **Shimmer** | `shimmerLocaldB_sma3nz` | - Relative frame-to-frame amplitude perturbation in dB.<br>- Valid range: $0.0\text{ dB} \le \text{shimmer} \le 3.5\text{ dB}$.<br>- Requires steady voiced segment ($\Delta \text{RMS} < 15\text{ dB}$ across adjacent cycles). | Values $> 3.5\text{ dB}$ flagged as invalid / non-speech noise artifact. |
| **Harmonics-to-Noise Ratio (HNR)** | `HNRdBACF_sma3nz` | - Logarithmic ratio of harmonic energy to noise floor.<br>- Valid range: $0.0\text{ dB} \le \text{HNR} \le 30.0\text{ dB}$ on voiced speech. | Values $< 0\text{ dB}$ (noise-dominated) or $> 30\text{ dB}$ (sensor clipping/synthetic artifact) flagged invalid. |
| **Formants ($F_1, F_2, F_3$)** | `F1frequency`, `F2frequency`, `F3frequency` | - $200\text{ Hz} \le F_1 \le 1,200\text{ Hz}$<br>- $600\text{ Hz} \le F_2 \le 3,000\text{ Hz}$<br>- $1,500\text{ Hz} \le F_3 \le 4,500\text{ Hz}$<br>- Strict order constraint: $F_1 < F_2 < F_3$ with separations $F_2 - F_1 \ge 200\text{ Hz}$ and $F_3 - F_2 \ge 250\text{ Hz}$.<br>- Formant bandwidth $B_i \le 500\text{ Hz}$. | Violations indicate spurious LPC pole; discard formant frame. |

### 1.2 Empirical Invalidity & Missingness Audit on Development Data

All 239 active development sessions in `feature_store_v2` were audited against these quality criteria:

| Cohort Domain | Sample Size ($N$) | $f_0$ Valid Rate | Jitter Valid Rate | Shimmer Valid Rate | HNR Valid Rate | Formants Valid Rate | Complete Prosodic Validity | Mean Zero Dims | Mean NaN Dims |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WoZ Development** | 178 | **100.0%** (178/178) | **100.0%** (178/178) | **100.0%** (178/178) | **100.0%** (178/178) | **100.0%** (178/178) | **100.0% (178/178)** | 0.0 / 154 | 0.0 / 154 |
| **Avatar Development** | 61 | **100.0%** (61/61) | **100.0%** (61/61) | **100.0%** (61/61) | **100.0%** (61/61) | **100.0%** (61/61) | **100.0% (61/61)** | 0.0 / 154 | 0.0 / 154 |
| **Combined Dev Cohort** | 239 | **100.0%** (239/239) | **100.0%** (239/239) | **100.0%** (239/239) | **100.0%** (239/239) | **100.0%** (239/239) | **100.0% (239/239)** | 0.0 / 154 | 0.0 / 154 |

**Measured Empirical Ranges across Development Data:**
- **$f_0$ Mean:** WoZ $[76.6\text{ Hz}, 279.0\text{ Hz}]$; Avatar $[78.0\text{ Hz}, 304.7\text{ Hz}]$.
- **Jitter Local Mean:** WoZ $[0.0114, 0.0386]$ ($1.1\% - 3.9\%$); Avatar $[0.0111, 0.0365]$ ($1.1\% - 3.7\%$).
- **Shimmer Local dB Mean:** WoZ $[0.669\text{ dB}, 1.351\text{ dB}]$; Avatar $[0.766\text{ dB}, 1.583\text{ dB}]$.
- **HNR dB Mean:** WoZ $[2.34\text{ dB}, 11.08\text{ dB}]$; Avatar $[2.75\text{ dB}, 12.32\text{ dB}]$.
- **Formants Mean ($F_1, F_2, F_3$):** WoZ ($576.5, 1620.6, 2700.4\text{ Hz}$); Avatar ($555.8, 1590.5, 2671.3\text{ Hz}$).

### 1.3 In-Fold Quality & Missing Data Policy

To prevent data leakage across evaluation folds, quality filtering and imputation must adhere strictly to in-fold processing:

1. **Robust Turn Aggregation:** Compute median and median absolute deviation (MAD) across valid turns per session, explicitly ignoring NaN values:
   $$\tilde{x} = \text{median}(\{x_t : \text{turn } t \text{ is valid}\})$$
2. **In-Fold Winsorisation:** Extreme feature outliers clipped to $[P_{01}, P_{99}]$ thresholds learned strictly from the **TRAINING fold**:
   $$x_{\text{clip}} = \text{clip}(x, P_{01}^{\text{train}}, P_{99}^{\text{train}})$$
3. **In-Fold Median Imputation:** If a participant lacks valid estimates for a feature (e.g., zero voiced turns satisfying quality thresholds), the missing value is imputed using the median of the **TRAINING fold**:
   $$x_{\text{imputed}} = \tilde{x}^{\text{train}}$$
4. **Missing-Indicator Covariates:** For every feature subject to imputation, append a binary indicator:
   $$M_j = \begin{cases} 1 & \text{if feature } j \text{ was invalid/missing} \\ 0 & \text{if feature } j \text{ was valid} \end{cases}$$
5. **Session Participant Exclusion Gate:** Any participant with fewer than $5$ valid voiced turns or total voiced duration $< 2.0\text{ s}$ across the entire session is flagged as `insufficient_acoustic_data`.

---

## Part 2: Demographic Gender Audit

Using authoritative ground-truth labels from `labels_official.csv`, we cross-tabulate gender distributions across clinical classes and domains.

### 2.1 Contingency Table: Domain $\times$ Gender $\times$ Class

| Recording Domain | Gender | Class 0 (None/Mild) | Class 1 (Moderate) | Class 2 (Severe) | Total Cohort ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Avatar Dev** | **Female** | **1** | **2** | **2** | **5** |
| | **Male** | **39** | **7** | **10** | **56** |
| | *Avatar Subtotal* | *40* | *9* | *12* | *61* |
| **WoZ Dev** | **Female** | **51** | **17** | **11** | **79** |
| | **Male** | **74** | **12** | **13** | **99** |
| | *WoZ Subtotal* | *125* | *29* | *24* | *178* |
| **Total Dev Cohort** | **Female** | **52 (61.9%)** | **19 (22.6%)** | **13 (15.5%)** | **84** |
| | **Male** | **113 (72.9%)** | **19 (12.3%)** | **23 (14.8%)** | **155** |
| | **Combined Total** | **165 (69.0%)** | **38 (15.9%)** | **36 (15.1%)** | **239** |

*Note: PID 306 (WoZ Dev, Female, Class 0) is excluded due to missing source audio.*

### 2.2 Methodological Evaluation: Separate Models vs. Covariate Adjustment

#### Decision 1: Are Separate Male and Female Models Justified?
- **DECISION:** **NO. SEPARATE MODELS ARE STRICTLY PROHIBITED.**
- **Methodological Justification:**
  1. **Cell Sparsity in Avatar Cohort:** The Avatar development cohort contains only **5 Female participants** in total ($N_{\text{class 0}}=1, N_{\text{class 1}}=2, N_{\text{class 2}}=2$). In a 5-fold cross-validation scheme, multiple evaluation folds would contain **zero** female participants or zero examples of specific severity classes, rendering fold evaluation mathematically degenerate.
  2. **Severe Power Loss:** Splitting into sex-stratified models would fragment the development dataset into $N=84$ (Female) and $N=155$ (Male). With 3 classes, severe depression has only 13 female examples and 23 male examples across the entire dataset. Training separate neural or classical models on such small sample sizes guarantees severe over-parameterization and catastrophic test fold variance.
  3. **Zero-Shot Domain Transfer Breakdown:** A female-specific model trained on WoZ ($N=79$) could not be reliably tested on Avatar ($N=5$).

#### Decision 2: Is a Regularised Gender Covariate Justified?
- **DECISION:** **NOT JUSTIFIED AS A PRIMARY FEATURE.**
- **Methodological Justification:**
  1. **Empirical Evidence from M2 Baseline:** The metadata baseline model (M2, which includes gender and age) achieved a Macro-F1 of only **$36.41\%$** (chance level $\approx 33.3\%$, binary AUC $50.35\%$), proving that gender itself carries no predictive clinical signal for depression severity.
  2. **Risk of Bias Amplification:** Adding explicit gender indicators to acoustic models risks encouraging classifiers to exploit pitch baselines as demographic shortcuts rather than learning true clinical speech dynamics.

#### Decision 3: Subgroup Reporting is Primary
- **DECISION:** **PRIMARY EVALUATION PROTOCOL.**
- **Implementation:**
  - All models will be trained on the pooled cohort with strict stratified group CV.
  - Performance must be reported separately for:
    1. **Male Subgroup** ($N=155$): Macro-F1, Balanced Accuracy, Per-class Recall, Binary AUC.
    2. **Female Subgroup** ($N=84$): Macro-F1, Balanced Accuracy, Per-class Recall, Binary AUC.
    3. **Disparity Index:** $|\text{Macro-F1}_{\text{Male}} - \text{Macro-F1}_{\text{Female}}|$ to monitor demographic fairness and enforce that no model is selected if it achieves high aggregate accuracy by sacrificing performance on either gender.
