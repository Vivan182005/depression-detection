# Quality Audit v2: Pilot Audit on 60 Development Sessions
## Per-Frame & Per-Turn Acoustic Validity Analysis (Praat / Parselmouth)

**Date:** 2026-10-14  
**Branch:** `research-upgrade-2026-10-13`  
**Pilot Cohort:** 60 Development Sessions (30 DAIC-WOZ Dev + 30 E-DAIC Avatar Dev)  
**Authoritative Labels:** [`experiments/research_v2/labels_official.csv`](../../experiments/research_v2/labels_official.csv)  
**Audit Artifact:** [`experiments/research_v2/quality_audit_v2_pilot.csv`](../../experiments/research_v2/quality_audit_v2_pilot.csv)  
**Audit Script:** [`experiments/research_v2/run_quality_audit_v2.py`](../../experiments/research_v2/run_quality_audit_v2.py)  

---

## 1. Objectives & Methodological Shift from v1

Quality Audit v1 evaluated static session-level summary functionals from OpenSMILE `feature_store_v2`. In contrast, **Quality Audit v2** conducts a granular, **per-frame and per-turn acoustic analysis** directly using Praat (via Parselmouth C-API) to establish true physiological validity, invalidity rates, and missingness boundaries prior to Phase 6A model runs.

### 1.1 Acoustic Quality Criteria & Validity Bounds

| Acoustic Metric | Praat / Parselmouth Method | Validity Rules & Boundary Constraints | Failure Flag / Action |
| :--- | :--- | :--- | :--- |
| **Pitch ($f_0$)** | `To Pitch (cc)` | - **Two-Pass Estimation:** Pass 1 runs unconstrained ($60–500\text{ Hz}$) across session audio to compute median pitch $\tilde{f}_0$. Pass 2 bounds pitch range dynamically to $[0.5 \tilde{f}_0, 2.0 \tilde{f}_0]$, clamped to $[50, 450\text{ Hz}]$. **Zero labels or gender metadata used.**<br>- **Octave-Jump Filter:** Reject frame if $|\Delta \log_2(f_0)| > 0.8$ octaves within $\le 30\text{ ms}$. | Unvoiced / jump frames invalidated. If turn voiced duration $< 0.1\text{ s}$, turn is excluded. |
| **Voiced Speech Duration & Pulses** | `To PointProcess (cc)` | - Minimum voiced duration $\ge 0.30\text{ s}$ per turn.<br>- Minimum pitch pulses $N_{\text{pulses}} \ge 30$ cycles per turn. | Turns failing $\ge 0.30\text{ s}$ or $< 30$ pulses are **strictly excluded from perturbation measures** (jitter & shimmer marked `NaN`). |
| **Jitter** | `Get jitter (local, rap, ppq5)` | - Valid range: $0.0\% \le \text{jitter} \le 12.0\%$ ($0.0 \le \text{val} \le 0.12$).<br>- Calculated only across contiguous voiced cycles on turns meeting the perturbation threshold. | Values $> 0.12$ ($12\%$) flagged as cycle-tracking errors / invalid estimates. |
| **Shimmer** | `Get shimmer (local, apq3, apq5, apq11)` | - Valid range: $0.0 \le \text{amplitude ratio} \le 0.35$ ($0.0\text{ dB} \le \text{shimmer} \le 3.5\text{ dB}$).<br>- Evaluated on turns meeting the perturbation threshold. | Values $> 0.35$ flagged as non-speech noise or clipping artifacts. |
| **Harmonics-to-Noise Ratio (HNR)** | `To Harmonicity (cc)` | - Valid range: $0.0\text{ dB} \le \text{HNR} \le 30.0\text{ dB}$ on voiced speech. | Values $< 0\text{ dB}$ (noise-dominated) or $> 30\text{ dB}$ (synthetic/sensor anomaly) flagged invalid. |
| **Formants ($F_1, F_2, F_3$)** | `To Formant (burg)` | - $200\text{ Hz} \le F_1 \le 1,200\text{ Hz}$<br>- $600\text{ Hz} \le F_2 \le 3,000\text{ Hz}$<br>- $1,500\text{ Hz} \le F_3 \le 4,500\text{ Hz}$<br>- Strict acoustic separation: $F_1 < F_2 < F_3$ with $F_2 - F_1 \ge 150\text{ Hz}$ and $F_3 - F_2 \ge 200\text{ Hz}$. | LPC spurious pole / ordering violation; discard formant frame. |

---

## 2. Pilot Findings on 60 Development Sessions

A representative pilot cohort of 60 development sessions (30 DAIC-WOZ Dev + 30 E-DAIC Avatar Dev) was segmented and analyzed per-turn using Silero VAD and Parselmouth:

### 2.1 Domain-Level Aggregate Comparison

| Metric / Rate | DAIC-WOZ Development ($N = 30$) | E-DAIC Avatar Development ($N = 30$) | Combined Pilot Cohort ($N = 60$) |
| :--- | :---: | :---: | :---: |
| **Turns Meeting Perturbation Threshold ($\ge 0.3$s & $\ge 30$ pulses)** | **88.70%** | **87.69%** | **88.20%** |
| **Jitter (local, rap, ppq5) Invalid / Missing Rate** | **11.30%** | **12.31%** | **11.80%** |
| **Shimmer (local, apq3, apq5, apq11) Invalid / Missing Rate** | **11.30%** | **12.31%** | **11.80%** |
| **HNR Invalid / Missing Rate** | **0.37%** | **0.06%** | **0.21%** |
| **Pitch ($f_0$) Invalid / Unvoiced Rate** | **5.67%** | **7.68%** | **6.67%** |
| **Formants ($F_1, F_2, F_3$) Invalid / Ordering Violation Rate** | **0.09%** | **0.00%** | **0.04%** |
| **Sessions Flagged with `insufficient_turns` ($< 5$ valid turns)** | **0.0% (0 / 30)** | **0.0% (0 / 30)** | **0.0% (0 / 60)** |

### 2.2 Empirical Distribution of Valid Acoustic Estimates
Across all valid turns in the pilot cohort:
- **Two-Pass $f_0$ Mean:** WoZ = $126.95\text{ Hz}$ (range: $82.5–198.9\text{ Hz}$); Avatar = $127.07\text{ Hz}$ (range: $82.6–250.5\text{ Hz}$).
- **Jitter Local Mean:** WoZ = $0.0238$ ($2.38\%$); Avatar = $0.0259$ ($2.59\%$).
- **Shimmer Local Mean:** WoZ = $0.1027$ ($10.27\%$); Avatar = $0.1143$ ($11.43\%$).
- **HNR Mean:** WoZ = $10.82\text{ dB}$; Avatar = $9.74\text{ dB}$.
- **Formants Mean ($F_1, F_2, F_3$):** 
  - WoZ: $F_1 = 652.97\text{ Hz}, F_2 = 1734.39\text{ Hz}, F_3 = 2795.37\text{ Hz}$.
  - Avatar: $F_1 = 643.12\text{ Hz}, F_2 = 1735.47\text{ Hz}, F_3 = 2792.40\text{ Hz}$.

---

## 3. Strict In-Fold Quality & Missing Data Protocol (Phase 6A)

To prevent cross-validation leakage during turn-level feature extraction in Phase 6A:
1. **Robust Turn Aggregation:** For each session, summary functionals (median, MAD) are calculated across **valid turns only**, explicitly ignoring `NaN` values resulting from short/unvoiced turns:
   $$\tilde{x}_i = \text{median}(\{x_{i,t} : t \in \text{valid turns}\})$$
2. **In-Fold Winsorisation:** Extreme feature outliers are clipped to $[P_{01}^{\text{train}}, P_{99}^{\text{train}}]$ thresholds learned strictly from the **TRAINING fold**:
   $$x_{\text{clip}} = \text{clip}(x, P_{01}^{\text{train}}, P_{99}^{\text{train}})$$
3. **In-Fold Median Imputation:** If a participant lacks valid turns for a specific feature, the missing value is imputed using the training fold feature median:
   $$x_{\text{imputed}} = \tilde{x}^{\text{train}}$$
4. **Missing-Indicator Covariates:** For every feature subject to imputation, append a binary indicator:
   $$M_j = \begin{cases} 1 & \text{if feature } j \text{ was invalid/missing} \\ 0 & \text{if feature } j \text{ was valid} \end{cases}$$
5. **Session Participant Exclusion Gate:** Any participant with fewer than $5$ valid voiced turns or total voiced duration $< 2.0\text{ s}$ across the entire session is flagged as `insufficient_acoustic_data`. In the 60 pilot sessions, **0% of participants were excluded under this gate**.

---
*Audit recorded and finalized on branch `research-upgrade-2026-10-13`.*
