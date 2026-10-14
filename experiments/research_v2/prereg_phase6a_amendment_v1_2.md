# Pre-Registration Amendment (Version 1.2): Phase 6A Protocol Update
## Proposed Protocol Amendment & Dated Change Log

**Amendment Document Version:** 1.2  
**Date of Drafting:** 2026-10-14  
**Reference Document:** [`experiments/research_v2/prereg_phase6a.md`](prereg_phase6a.md) (Version 1.1, Preserved Historical)  
**Repository Branch:** `research-upgrade-2026-10-13`  
**Working Directory:** `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Execution Gate:** AMENDMENT PROPOSED & LOGGED BEFORE OUTCOME EXPERIMENT EXECUTION  
**Outcome Evaluation Status:** ZERO OBSERVED OUTCOME RESULTS HAVE BEEN COMPUTED AGAINST REAL PHQ LABELS  

---

## 1. Context & Rationale for Amendment v1.2

In Pre-registration Protocol v1.1 (committed under hash `3526432`), the primary endpoint (participant-level binary AUC on PHQ $\ge$ 10 across 239 dev participants), the 19-variant matrix, the stratified permutation null within domain $\times$ gender strata, and the Confirmatory Holm family ($m=3$: H1 on V2, H2 on V10, H3 on V18 IUT $\Delta$-AUC) were formally locked.

However, subsequent empirical audits (Audit B1 and Quality Audit v2) revealed two critical methodological findings:
1. **Cumulative Alignment Failure on WoZ Audio:** The hypothesis that `Clean_audio/NEW_AUDIO/*_AUDIO_interviewee.wav` could be segmented into conversational turns via cumulative-offset gaps failed for **98.3% of development sessions (175 / 178)** due to timing drift that caused predicted inter-turn pause boundaries to land squarely within active speech phonation. Raw transcript slicing is strictly prohibited. Slicing with cumulative offsets would corrupt turn-level features with artificial speech truncation.
2. **Superiority of Unified Neural VAD:** Validating **Silero VAD** against Whisper word timestamps demonstrated $83.3\% - 87.5\%$ frame agreement on continuous conversational recordings, reliably isolating true participant speech from background noise and setup dialogue without requiring external transcripts.

To maintain strict scientific integrity, this amendment (v1.2) formalizes the transition to an objective, transcript-independent, domain-harmonized feature definition while preserving the core hypotheses and the historical v1.1 document.

---

## 2. Formal Specification of Amendments

### Amendment 1: Redefinition of Feature-Extraction "Turn" as a Silero VAD Speech Segment
- **Old Definition (v1.1):** Turn-level features were defined over reconstructed transcript turns using cumulative-offset boundaries on DAIC-WOZ audio.
- **Amended Definition (v1.2):** A feature-extraction "turn" is formally redefined as a **Silero VAD speech segment**, detected directly from the continuous audio signal.
- **Frozen Unified Parameter Set (Identical across DAIC-WOZ and E-DAIC Avatar):**
  - **VAD Architecture:** Silero VAD (PyTorch JIT model).
  - **Software / Version:** `silero-vad == 6.2.3`, PyTorch `2.14.1`.
  - **Speech Probability Threshold (`threshold`):** `0.50`
  - **Minimum Speech Duration (`min_speech_duration_ms`):** `250 ms`
  - **Minimum Silence Duration (`min_silence_duration_ms`):** `200 ms`
  - **Maximum Segment Duration (`max_speech_duration_s`):** `15.0 s` (speech segments continuously exceeding 15.0 seconds are segmented at the internal minimum-energy frame).
  - **Speech Segment Padding / Merge Gap:** `30 ms` on speech boundaries.
  - **Audio Sampling Rate:** Uniformly converted to `16,000 Hz` mono float32 prior to VAD scoring.
- **Integrity Rule:** These parameters are frozen *a priori* based purely on acoustic voice detection physics. **Zero tuning of VAD parameters was performed using PHQ labels or observed classifier performance.**

### Amendment 2: Removal of the Cumulative-Offset Alignment Exclusion Rule
- **Old Rule (v1.1):** Sessions failing cumulative-offset alignment ($p_{\text{perm}} \ge 0.05$ or $E_{\text{gap}} \ge 0.85 E_{\text{speech}}$) were excluded from turn-level features. Because 98.3% of WoZ sessions failed this test, V2 would have been evaluated on only 3 WoZ sessions.
- **Amended Rule (v1.2):** The cumulative-offset alignment-exclusion rule is **completely removed from the feature definition pipeline**. Because all turn-level features are now extracted directly from verified Silero VAD speech segments on the actual audio, **all 239 active development sessions (178 DAIC-WOZ + 61 Avatar) are 100% eligible for turn-level feature extraction**.
- **Prohibition Preserved:** Slicing stored WAVs using original transcript timestamps remains **STRICTLY PROHIBITED**.

### Amendment 3: Explicit Declaration of Audio Provenance
- It is formally recorded as an empirical fact that the exact code, command invocation, and join structure that produced `Clean_audio/NEW_AUDIO/*_AUDIO_interviewee.wav` remain **UNRESOLVED**.
- No claim is made that these files are known to contain concatenated participant turns with a particular gap or normalization procedure. They are treated simply as pre-existing mono 16 kHz audio files containing interviewee speech.

### Amendment 4: Unified Table of Validity Thresholds & Discrepancy Reconciliation
To resolve discrepancies between previous text mentions (e.g. shimmer amplitude ratio $0.35$ vs. $3.5\text{ dB}$, formant separation $150/200$ vs. $200/250\text{ Hz}$), all validity criteria are unified in the following authoritative standard:

| Feature Dimension | Extraction Method & Tool | Unit | Formal Definition | Minimum Duration / Cycles | Valid Range Bounds | Action on Violation |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Pitch ($f_0$)** | Parselmouth `To Pitch (cc)` | Hz | Fundamental frequency from acoustic autocorrelation | $\ge 0.10\text{ s}$ voiced | $f_0 \in [f_{\min}, f_{\max}]$ where $[f_{\min}, f_{\max}] = [\max(50, 0.5 \tilde{f}_0), \min(450, 2.0 \tilde{f}_0)]$ derived from Pass 1 session median $\tilde{f}_0$ | Frame marked unvoiced; reject octave jumps ($|\Delta \log_2 f_0| > 0.8$ octaves in $\le 30\text{ ms}$) |
| **Jitter (local)** | Parselmouth `Get jitter (local)` | Ratio | Relative mean absolute difference of consecutive pitch periods: $\frac{1}{N-1} \sum \|T_i - T_{i-1}\| / \bar{T}$ | $\ge 0.30\text{ s}$ voiced AND $\ge 30$ pulses | $0.0 \le \text{jitter} \le 0.12$ ($0\% - 12\%$) | Marked `NaN` for segment |
| **Jitter (rap)** | Parselmouth `Get jitter (rap)` | Ratio | Relative Average Perturbation quotient across 3 periods | $\ge 0.30\text{ s}$ voiced AND $\ge 30$ pulses | $0.0 \le \text{jitter}_{\text{rap}} \le 0.10$ | Marked `NaN` for segment |
| **Jitter (ppq5)** | Parselmouth `Get jitter (ppq5)` | Ratio | Five-point Period Perturbation Quotient | $\ge 0.30\text{ s}$ voiced AND $\ge 30$ pulses | $0.0 \le \text{jitter}_{\text{ppq5}} \le 0.10$ | Marked `NaN` for segment |
| **Shimmer (local)** | Parselmouth `Get shimmer (local)` | Ratio | Relative mean absolute difference of consecutive peak amplitudes: $\frac{1}{N-1} \sum \|A_i - A_{i-1}\| / \bar{A}$ | $\ge 0.30\text{ s}$ voiced AND $\ge 30$ pulses | $0.0 \le \text{shimmer} \le 0.35$ ($0\% - 35\%$) | Marked `NaN` for segment |
| **Shimmer (apq3, apq5, apq11)** | Parselmouth `Get shimmer (apq*)` | Ratio | Amplitude perturbation quotients across 3, 5, 11 periods | $\ge 0.30\text{ s}$ voiced AND $\ge 30$ pulses | $0.0 \le \text{shimmer} \le 0.35$ | Marked `NaN` for segment |
| **Harmonics-to-Noise Ratio (HNR)** | Parselmouth `To Harmonicity (cc)` | dB | Logarithmic ratio of periodic acoustic energy to aperiodic noise | $\ge 0.10\text{ s}$ voiced | $0.0\text{ dB} \le \text{HNR} \le 30.0\text{ dB}$ | Marked `NaN` for segment |
| **Formants ($F_1, F_2, F_3$)** | Parselmouth `To Formant (burg)` | Hz | Vocal tract resonance pole frequencies via Burg LPC | $\ge 0.10\text{ s}$ voiced | $F_1 \in [200, 1200]\text{ Hz}$<br>$F_2 \in [600, 3000]\text{ Hz}$<br>$F_3 \in [1500, 4500]\text{ Hz}$ | Enforce strict acoustic separation: $F_2 - F_1 \ge 150\text{ Hz}$ and $F_3 - F_2 \ge 200\text{ Hz}$; discard violation |
| **Spectral Slope** | Linear regression on FFT spectrum | dB/oct | Spectral tilt from $0–5000\text{ Hz}$ | $\ge 0.10\text{ s}$ voiced | $[-30.0, +10.0]\text{ dB/oct}$ | Marked `NaN` for segment |
| **Session-Relative Intensity** | Root-Mean-Square (RMS) | dB | $20 \log_{10}(\text{RMS}_{\text{segment}} / \text{RMS}_{\text{session}})$ | Full segment | $[-40.0, +20.0]\text{ dB}$ | Unbounded / clipped |

*Resolution of Discrepancies:*
- **Shimmer:** In Praat/Parselmouth, `Get shimmer (local)` returns the dimensionless amplitude perturbation quotient ratio. The physiological valid threshold is $\le 0.35$ (35% perturbation). In decibels, this corresponds to $\le 3.5\text{ dB}$ (via $20 \log_{10}(1 + \text{ratio})$). Both descriptions refer to the same underlying physical boundary; the canonical representation in code is the Praat ratio with upper bound $0.35$.
- **Formant Separation:** Formant separation bounds are strictly standardized to $F_2 - F_1 \ge 150\text{ Hz}$ and $F_3 - F_2 \ge 200\text{ Hz}$.

---

## 3. Impact on 19-Variant Matrix & Confirmatory Hypotheses

### 3.1 Variants Directly Affected
The redefinition of a turn as a Silero VAD segment directly updates the feature extraction pipeline for:
- **V2 (Confirmatory $H_1$):** Hand-crafted voice features aggregated across valid Silero VAD segments (median, IQR, p10, p90, std, early-vs-late slope) + $L_2$ Logistic Regression.
- **V3:** Same VAD-aggregated hand-crafted voice features + HistGBDT.
- **V4:** Same hand-crafted features + CMVN + Long-Term-Spectrum Equalisation + $L_2$ Logistic Regression.
- **V5:** Normalised hand-crafted features + HistGBDT.
- **V12:** Turn-level hand-crafted features + Small MLP (fixed 40 epochs).
- **V13:** Turn-level WavLM Layer 8 representations across VAD segments + Small MLP (fixed 40 epochs).
- **V18 (Confirmatory $H_3$):** Nested Stacking of amended V2 + V14 (BERT-T2).

### 3.2 Variants Preserved Exactly
- Whole-recording functionals: **V1** (speech-only eGeMAPS 135d robust).
- Global SSL speech embeddings: **V6–V8** (wav2vec2-base-960h Layers 4, 8, 12 mean-pooled over VAD speech frames).
- Global SSL speech embeddings: **V9–V11** (WavLM-base-plus Layers 4, 8, 12 mean-pooled over VAD speech frames, including Confirmatory **$H_2$ on V10**).
- Text benchmark: **V14** (`BERT-T2` with Whisper-tiny transcripts).
- Controls: **V15** (length-only), **V16** (gender-only), **V17** (domain-only).
- Exploratory Fusion: **V19** (Best Audio + V14).

### 3.3 Status of the Confirmatory Testing Family ($m = 3$)
- **Verdict: PRESERVED IN FULL.**
- The amended feature definitions for V2 and V18 **faithfully preserve the original scientific intent** of Hypotheses $H_1$, $H_2$, and $H_3$:
  - $H_1$: Tests whether hand-crafted acoustic voice features carry predictive signal above the empirical null.
  - $H_2$: Tests whether self-supervised speech representations (WavLM Layer 8) carry predictive signal above the empirical null.
  - $H_3$: Tests whether multimodal fusion (Voice + Text) achieves statistically significant incremental benefit over both unimodal constituents.
- Redefining turns via neural VAD rather than flawed transcript alignment **increases the ecological validity and reproducibility** of V2 and V18 without introducing label leakage.

---

## 4. Dated Change Log & Pre-Registration Integrity

| Version | Date | Status | Description of Changes |
| :---: | :---: | :---: | :--- |
| **v1.0** | 2026-10-14 | Superseded | Initial pre-registration protocol draft. |
| **v1.1** | 2026-10-14 | Preserved Historical | Corrected primary endpoint to participant-level binary AUC (PHQ $\ge$ 10, seeds 0–4 averaged); descriptive secondary metrics; fixed-epoch MLPs; stratified permutation null within domain $\times$ gender; Confirmatory Holm family ($m=3$); 19-variant matrix; factual corrections (Whisper-tiny T2, sealed set 16+12+5). |
| **v1.2** | 2026-10-14 | **Proposed Amendment** | Redefined turn as a Silero VAD segment with frozen parameters; removed cumulative-offset alignment exclusion rule (all 239 dev participants eligible); recorded unresolved WoZ WAV provenance; unified validity table resolving shimmer and formant discrepancies; documented impact on V2/V3–V5/V12–V13 while preserving Confirmatory family. **Logged prior to outcome experiment.** |

---
*Amendment proposed and recorded on branch `research-upgrade-2026-10-13` before Phase 6A outcome experiment execution.*
