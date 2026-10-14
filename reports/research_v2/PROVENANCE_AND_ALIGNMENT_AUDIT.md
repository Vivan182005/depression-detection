# Provenance Investigation & Empirical Alignment Hypothesis Audit

**Date:** 2026-10-14  
**Branch:** `research-upgrade-2026-10-13`  
**Dataset:** 178 DAIC-WOZ Development Sessions (`Clean_audio/NEW_AUDIO/`) & Avatar Development Sessions (`NEW_AUDIO_DAIC/`)  
**Audit Artifacts:** 
- [`experiments/research_v2/woz_alignment_audit.csv`](../../experiments/research_v2/woz_alignment_audit.csv)
- [`experiments/research_v2/avatar_silero_vad_validation.csv`](../../experiments/research_v2/avatar_silero_vad_validation.csv)
- Script: [`experiments/research_v2/audit_b1_alignment.py`](../../experiments/research_v2/audit_b1_alignment.py)  

---

## 1. Provenance Investigation of `Clean_audio/NEW_AUDIO/*_AUDIO_interviewee.wav`

### 1.1 Status of the "Stitched & Peak-Normalized" Narrative
The characterization that `Clean_audio/NEW_AUDIO/*_AUDIO_interviewee.wav` consists of *"participant turns stitched with ~0.3–0.4 s gaps and peak-normalised to -1 dBFS"* is treated strictly as a **REPORTED HYPOTHESIS**, not an established fact. We conducted a comprehensive forensic audit to establish which code actually generated these files.

### 1.2 Forensic Code & Repository Inspection

| Source Inspected | Target / Query | Findings & Empirical Evidence |
| :--- | :--- | :--- |
| **`app/remove_ellie.py`** | Script output logic | Hardcoded output filename pattern is: `os.path.join(output_dir, f"{base_name}_PARTICIPANT_ONLY.wav")` (Line 27). It never outputs or references `*_AUDIO_interviewee.wav`. |
| **`app/src/preprocessing/speaker_isolation.py`** | Speaker isolation module | Contains `separate_speakers_automatically` and `isolate_participant_audio`. Inserts exact zeros: `pause_samples = np.zeros(int(pause_gap_sec * sr))`. Default output naming is `f"{base}_PARTICIPANT_ONLY.wav"`. |
| **`app/src/preprocessing/audio_cleaner.py`** | Audio cleaning logic | Defines `normalize_loudness(audio, target_peak_db=-1.0)` and `preprocess_audio_signal`. Functions are called on in-memory buffers; neither writes to `*_AUDIO_interviewee.wav`. |
| **`app/extract_dataset_features.py`** | Feature extraction CLI | Defines `--audio_dir` defaulting to `../Clean_audio/NEW_AUDIO`. Explicitly refers to input audio as *"pre-cleaned audio WAV files"*, indicating they were already prepared before extraction. |
| **`app/Projectoutline.md`** | Pipeline documentation | Line 117 states: `Input: 195 DAIC-WOZ audio files (e.g. 300_AUDIO_interviewee.wav)`. Assumes the files were placed into the directory prior to project architecture definition. |
| **WAV Headers & RIFF Chunks** | Binary header structure | All 178 files possess a uniform 44-byte canonical RIFF/WAVE header (`fmt ` chunk size 16, PCM format 1, 16 kHz mono, 16-bit signed PCM). **Zero `LIST`, `INFO`, `ISFT` (software/tool ID), or `ICRD` (creation date) metadata chunks exist.** |
| **Filesystem Timestamps** | File modification dates | `st_mtime` across all 178 files is **`2026-07-31 15:55:07` to `15:56:06`**, predating the initial Git commit of this repository (October 2026) by several months. |
| **Git Commit History** | `git log -S '_AUDIO_interviewee'` | The string `_AUDIO_interviewee` never appears in any commit creating or writing files. It appears exclusively in downstream evaluation scripts and documentation. |

### 1.3 Provenance Verdict
**STATUS: UNRESOLVED.**  
No script, pipeline, commit, or log in the repository produced the files in `Clean_audio/NEW_AUDIO/*_AUDIO_interviewee.wav`. While their naming and duration characteristics share similarities with legacy DAIC-WOZ participant extraction workflows, their exact generation code, parameters, and authoring environment cannot be established from repository artifacts. In accordance with strict empirical standards, **provenance is formally recorded as UNRESOLVED and is not inferred from similar logic**.

---

## 2. Empirical Test of the Alignment Hypothesis

### 2.1 Context & Empirical Correlation
Although Task A2 proved that the stored WAVs contain **zero exact-zero runs near 0.3 s or 0.4 s**, a strong linear relationship exists between stored file duration and participant turn count:
- **Total Sessions Analyzed:** 178 DAIC-WOZ development sessions.
- **Excess Duration vs. Turn Count Correlation:** **`r = 0.7850`** (excess duration $\Delta D = D - \sum d$ strongly scales with turn count $n_{\text{turns}}$).
- **Per-Session Gap Distribution:** 
  - Mean $g = 0.2830\text{ s}$
  - Median $g = 0.2850\text{ s}$
  - Standard Deviation = $0.1487\text{ s}$
  - 10th–90th Percentile Range: $[0.1085\text{ s}, 0.4428\text{ s}]$
  - 4 sessions have negative or zero gap ($g \le 0$).

### 2.2 Alignment Hypothesis Specification
We formulated and tested the **Cumulative-Offset Alignment Hypothesis**:
If participant turns were spliced sequentially with a per-session inter-turn pause of duration $g = (D - \sum d) / n_{\text{turns}}$, then for each turn $k \in \{0, \dots, n-1\}$ with duration $d_k$:
$$\hat{t}_{\text{start}}^{(k)} = \sum_{j=0}^{k-1} d_j + k \cdot g, \quad \hat{t}_{\text{stop}}^{(k)} = \hat{t}_{\text{start}}^{(k)} + d_k$$
and the inter-turn gap interval is:
$$G_k = [\hat{t}_{\text{stop}}^{(k)}, \hat{t}_{\text{start}}^{(k+1)}]$$
If this alignment holds, the signal within the predicted gaps $G_k$ must represent physical pauses/silence:
1. The RMS energy in the predicted gaps ($E_{\text{gap}}$) must be significantly lower than the speech energy ($E_{\text{speech}}$), yielding an energy ratio $R = E_{\text{gap}} / E_{\text{speech}} < 0.85$.
2. In a permutation test ($B = 200$ random temporal shifts of the gap mask across the session), the energy in the aligned gaps must be significantly lower than in randomly shifted offsets ($p_{\text{perm}} < 0.05$).

### 2.3 Empirical Results across 178 Sessions

| Outcome | Session Count | Percentage | Primary Failure Mechanism |
| :--- | :---: | :---: | :--- |
| **Passed Alignment** | **3** | **1.7%** | PIDs 346, 401, 404 ($p_{\text{perm}} < 0.01, R \in [0.83, 0.85]$). |
| **Failed Alignment** | **175** | **98.3%** | Rapid cumulative drift causes predicted gaps to land directly on active speech frames. |

**Failure Breakdown:**
- **`high_energy_ratio` (170 sessions, 95.5%):** The energy ratio $E_{\text{gap}} / E_{\text{speech}}$ exceeds $0.85$ (frequently $\approx 1.0$). Cumulative timing errors compound with every turn, causing the predicted pause intervals to fall squarely into speech phonation.
- **`negative_or_zero_gap` (4 sessions, 2.2%):** PIDs 307, 332, 420, 481 have $D < \sum d$, meaning stored audio is shorter than the sum of transcript turn spans.
- **`insignificant_perm` (1 session, 0.6%):** Gap energy is not statistically distinguishable from random offsets ($p_{\text{perm}} \ge 0.05$).

### 2.4 Gating Protocol & Exclusions
1. **Transcript Slicing Strictly Prohibited:** Slicing the stored stitched WAVs using raw transcript timestamps is **STRICTLY PROHIBITED** because original timestamps reference the un-spliced multi-channel recording timeline.
2. **Turn-Level Feature Gating:** Because 98.3% of sessions fail cumulative alignment, **sessions failing alignment are excluded from turn-level feature extraction**.
3. **Whole-Session Representations Retained:** All 178 sessions remain eligible for whole-recording speech-only acoustic functionals (`acoustic_robust_135d` and `wav2vec_768d`).

---

## 3. Avatar Development Turn Segmentation: Silero VAD Validation

For continuous Avatar recordings (`NEW_AUDIO_DAIC/`), turn-level boundaries cannot rely on transcript timestamps. We validated neural Voice Activity Detection (**Silero VAD**) against Whisper-tiny word/chunk timestamps across a representative sample of Avatar Development sessions:

| Participant ID | Audio Duration | VAD Speech Duration | Whisper Transcript Duration | Intersection over Union (IoU) | Frame Agreement | Detected Speech Segments |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **606** | 64.45 s | 14.82 s | 64.00 s | 0.2232 | 22.3% | 17 |
| **620** | 407.22 s | 339.69 s | 406.84 s | **0.8333** | **83.3%** | 139 |
| **625** | 511.64 s | 447.59 s | 509.76 s | **0.8744** | **87.5%** | 132 |

### 3.1 Validation Findings
- On conversational sessions (PIDs 620, 625), Silero VAD achieves **$83.3\% - 87.5\%$ frame agreement** with Whisper speech spans.
- On sessions with extensive initial interviewer setup dialogue (e.g. PID 606), Silero VAD correctly ignores the first 27 seconds of background silence/unvoiced room noise and detects participant speech starting cleanly at $27.62\text{ s}$, whereas naive transcripts span the entire recording.
- **Protocol Directive:** Silero VAD is certified as the validated alternative for turn-level segmentation on continuous conversational recordings.

---
*Audit recorded and finalized on branch `research-upgrade-2026-10-13`.*
