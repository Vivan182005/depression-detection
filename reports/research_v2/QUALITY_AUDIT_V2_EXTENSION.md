# Quality Audit v2 Extension (Stratified Sample: $N = 65$)

**Date:** 2026-10-14  
**Project:** Voice-Primary Depression Detection Research Prototype  
**Branch:** `research-upgrade-2026-10-13`  
**Protocol Version:** Pre-registration Amendment v1.2  
**Audit Dataset:** [`experiments/research_v2/quality_audit_v2_extended.csv`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/quality_audit_v2_extended.csv)  
**Execution Script:** [`experiments/research_v2/run_quality_audit_v2_extension.py`](file:///Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13/experiments/research_v2/run_quality_audit_v2_extension.py)

---

## 1. Executive Summary & Reconciliation of 59-vs-60 Pilot Discrepancy

### 1.1 Root-Cause Reconciliation of 59 vs 60 Pilot Rows
In the initial pilot audit, exactly 59 rows were produced instead of 60 intended (30 WoZ + 30 Avatar). Forensics revealed:
1. `labels_official.csv` contains 62 total E-DAIC Avatar development rows.
2. Participant **PID 602** (Female, Avatar Dev) was logged in metadata, but has **no corresponding audio file** on disk (`NEW_AUDIO_DAIC/602_AUDIO_interviewee.wav` is missing).
3. The pilot script indexed the first 30 entries of the manifest, but silently skipped PID 602 upon detecting a missing file, resulting in 29 Avatar + 30 WoZ = 59 sessions.

### 1.2 Development Audio Population Reconciliation
The exact reconciliation of development audio on disk across both domains is as follows:
- **DAIC-WOZ Development Cohort ($N = 178$ active audio files):**
  - Total labels: 189 participants.
  - Sealed cohort: 9 participants (PIDs 300, 303, 304, 321, 338, 397, 408, 409, 414) strictly quarantined.
  - Development candidates: 180 participants.
  - Missing audio on disk: 2 participants (PID 306, Female; PID 458, Female).
  - **Eligible Development Population with Audio:** $N = 178$ ($99$ Male, $79$ Female).
- **E-DAIC Avatar Development Cohort ($N = 61$ active audio files):**
  - Total labels: 86 participants.
  - Sealed cohort: 24 participants (PIDs 600, 601, 603, 604, 605, 607, 615, 619, 623, 624, 626, 628, 632, 633, 634, 635, 637, 638, 649, 653, 654, 655, 656, 662) strictly quarantined.
  - Development candidates: 62 participants.
  - Missing audio on disk: 1 participant (PID 602, Female).
  - **Eligible Development Population with Audio:** $N = 61$ ($56$ Male, $5$ Female).
- **Combined Active Development Cohort:** $178 + 61 = 239$ eligible sessions.

---

## 2. Sampling Methodology & Cohort Structure

To avoid manufacturing false balance or distorting domain demographics, a reproducible stratified sampling protocol was executed across the four Domain $\times$ Gender cells:
- Target: At least 20 sessions per cell where eligible population permits; if a cell has $< 20$, include **100% of eligible sessions** in that cell without synthetic padding or cross-cell substitution.
- Master Seed: `42` (Cell seeds: WoZ F = 42, WoZ M = 43, Avatar M = 44, Avatar F = 100% census).

| Cell Index | Domain | Gender | Eligible Population | Sampled Sessions | Sampling Rate | Sampled Participant IDs |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Cell 1** | DAIC-WOZ | Female | 79 | **20** | 25.3% | 395, 307, 345, 410, 483, 340, 415, 473, 477, 461, 350, 440, 476, 428, 327, 472, 335, 467, 452, 474 |
| **Cell 2** | DAIC-WOZ | Male | 99 | **20** | 20.2% | 453, 447, 464, 463, 455, 429, 392, 332, 351, 319, 449, 459, 448, 482, 357, 362, 369, 444, 421, 372 |
| **Cell 3** | E-DAIC Avatar | Male | 56 | **20** | 35.7% | 622, 658, 659, 686, 640, 672, 685, 712, 702, 648, 621, 627, 611, 663, 675, 680, 646, 669, 718, 678 |
| **Cell 4** | E-DAIC Avatar | Female | 5 | **5** | **100.0%** | **606, 651, 682, 696, 703** (100% census of Avatar female dev) |
| **Total** | **Combined** | **Both** | **239** | **65** | **27.2%** | **65 Distinct Sessions** |

---

## 3. Acoustic Quality & Validity Results

Acoustic turns were extracted using frozen Silero VAD parameters (Speech threshold 0.50, min speech 250 ms, min silence 200 ms, max duration 15.0 s, padding 30 ms). Per-turn acoustic measures were computed via Praat/Parselmouth with a two-pass pitch range tailored to each session's own voiced spectrum.

### 3.1 Stratified Summary Table

| Domain | Gender | $N$ | Valid Perturbation Turn Rate (%) | Jitter Invalid (%) | Shimmer Invalid (%) | $f_0$ Invalid (%) | HNR Invalid (%) | Formant Invalid (%) | Mean $f_0$ (Hz) | Mean Jitter (local) | Mean Shimmer (local) | Mean HNR (dB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E-DAIC Avatar** | Female | 5 | 58.7% | 41.3% | 41.3% | 30.6% | 0.22% | 0.00% | 170.0 Hz | 0.026 | 0.108 | 10.23 dB |
| **E-DAIC Avatar** | Male | 20 | 65.7% | 34.3% | 34.3% | 30.7% | 0.12% | 0.00% | 106.0 Hz | 0.025 | 0.106 | 9.39 dB |
| **DAIC-WOZ** | Female | 20 | 57.5% | 42.6% | 42.6% | 38.6% | 0.18% | 0.00% | 181.8 Hz | 0.024 | 0.084 | 14.38 dB |
| **DAIC-WOZ** | Male | 20 | 79.6% | 20.5% | 20.5% | 15.2% | 0.12% | 0.04% | 104.4 Hz | 0.024 | 0.098 | 10.35 dB |

### 3.2 Key Findings by Measure

1. **Fundamental Frequency ($f_0$) Sanity Check:**
   - Female sessions exhibit mean $f_0$ of **170.0 Hz** (Avatar) and **181.8 Hz** (WoZ).
   - Male sessions exhibit mean $f_0$ of **106.0 Hz** (Avatar) and **104.4 Hz** (WoZ).
   - The physiological separation is pronounced and clean across both domains, validating the two-pass ceiling/floor estimation without gender conditioning.
2. **Perturbation Validity Thresholds:**
   - Requiring $\ge 0.30$ s voiced speech and $\ge 30$ pitch pulses successfully rejects brief, unvoiced, or glottalized bursts.
   - Valid perturbation turn rates average **57.5%–79.6%**, providing high statistical power per session.
   - Turn-level shimmer and jitter invalidity tracks $f_0$ unvoiced/octave-jump segments closely.
3. **Harmonicity-to-Noise Ratio (HNR):**
   - HNR invalidity is minimal ($< 0.25\%$ across all cells). Mean HNR ranges from $9.39$ dB to $14.38$ dB.
   - DAIC-WOZ female sessions exhibit slightly higher mean HNR ($14.38$ dB) due to closer microphone placement in the physical WoZ booth compared to Avatar remote recording.
4. **Formant Validity:**
   - Under the unified pre-registration v1.2 rule ($200 \le F_1 \le 1200$, $600 \le F_2 \le 3000$, $1500 \le F_3 \le 4500$, $F_2 - F_1 \ge 150$, $F_3 - F_2 \ge 200$), formant validity is **$100.0\%$ in 3 of 4 cells** and **$99.96\%$ in WoZ Male**.
   - Zero sessions exhibited systemic formant collapse.

---

## 4. Participant-Level Exclusions & Sufficiency

A session is flagged as `insufficient_turns = True` if it contains $< 5$ valid perturbation turns or $< 2.0$ s total voiced speech.
- Out of 65 audited sessions, **63 sessions (96.9%) passed all sufficiency requirements**.
- Exactly **2 sessions** were flagged:
  1. **PID 473 (WoZ Female):** Total turns = 15; valid perturbation turns = 4 ($26.7\%$). Voiced duration = 1.62 s.
  2. **PID 622 (Avatar Male):** Total turns = 19; valid perturbation turns = 4 ($21.1\%$). Voiced duration = 1.84 s.
- **Protocol Action:** Neither session is dropped from whole-recording eGeMAPS or pretrained embeddings; both remain eligible for non-turn representations. For segment-level perturbation features, participant aggregation falls back to whole-session eGeMAPS functionals.

---

## 5. Governance & Boundary Compliance

> [!IMPORTANT]
> **Gender Auditing Boundary:** Gender metadata was used exclusively for descriptive auditing and sanity checking of $f_0$ distributions. In accordance with pre-registration v1.2, **gender is strictly forbidden as an input feature** in any primary voice model.
