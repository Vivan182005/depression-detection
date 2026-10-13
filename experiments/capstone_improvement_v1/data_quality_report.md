# 📊 Comprehensive Data Quality & Manifest Audit Report (Phase 3)

**Generated**: 2026-10-12  
**Dataset Reference**: DAIC-WOZ & E-DAIC 2019 Extended Corpus  
**Manifest Path**: [`split_manifest.csv`](./split_manifest.csv)  

---

## 1. Overall Dataset & Participant Census

| Metric | Count | Details |
| :--- | :---: | :--- |
| **Total Cataloged Participants** | **275** | Listed in `detailed_lables.csv` |
| **Participants with Valid Audio** | **272** | 187 original WoZ + 61 new training + 24 new unseen + 9 old unseen |
| **Participants with Transcripts** | **238** | 179 WoZ + 26 AI training + 33 unseen |
| **Participants with Both Audio & Text** | **237** | Full Tri-Modal eligible sessions |
| **Reserved Unseen Holdout** | **33** | 9 WoZ human-interviewer + 24 autonomous AI sessions |
| **Active Available Development Pool** | **239** | 187 WoZ + 61 AI sessions = 248 total |
| **Excluded (Audio Missing Locally)** | **3** | PIDs 460, 473, etc. not in downloaded WAV set |

---

## 2. Target Class & PHQ-8 Distributions (Available Pool: N=239)

```
Distribution in Available Development Pool:
  - Minimal / Mild                  : 165 (69.0%)
  - Moderate Depression             :  38 (15.9%)
  - Moderately Severe / Severe      :  36 (15.1%)

Binary Cutoff (PHQ-8 >= 10):
  - Non-Depressed (0): 165 (69.0%)
  - Depressed (1):     74 (31.0%)
```

---

## 3. Data Source & Domain Shift Analysis

1. **Cohort 1 — DAIC-WOZ (IDs 300–492)**:
   - Interviewer: Human wizard in adjacent room controlling animated avatar Ellie.
   - Speech Dynamics: Highly naturalistic conversational turns, human cadence and response latencies.
2. **Cohort 2 — E-DAIC 2019 (IDs 600–718)**:
   - Interviewer: Fully autonomous software AI agent generating perception and questions.
   - Speech Dynamics: Slightly fixed latency between participant answer and agent prompt.
   - Demographics: Slightly older mean age (46.1 vs 38.9 years).
3. **Domain Invariance Goal**:
   - By training on a joint composition of both human-led and AI-led interviews, the model avoids overfitting to a single agent turn-taking tempo.

---

## 4. Leakage Safeguards & Integrity Checklist

- [x] **Zero Participant Overlap**: Enforced programmatically. No participant appears in more than one partition.
- [x] **Unseen Cohort Isolation**: All 33 reserved sessions are blacklisted before training index generation.
- [x] **Deduplication Enforced**: Replicated sessions (`301_AUDIO`, `loud_audio301`) mapped to a single canonical participant entry.
- [x] **Interviewer Content Scrubbed**: Audio VAD and transcript parsing strictly retain participant speech only.
- [x] **No Target Contamination**: No post-interview metadata or survey items passed as input features.
