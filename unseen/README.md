# Unseen Sanity Check Dataset (Holdout Validation)

This directory contains structurally isolated audio recordings and participant-only transcripts reserved exclusively for post-training validation and real-world sanity checking.
None of these sessions or participant IDs are accessed during feature store indexing, training, validation, or hyperparameter selection.

## Cohort Overview
- **Total Reserved Sessions**: 33
  - **Wizard-of-Oz Cohort (300-492)**: 9 sessions
  - **Autonomous AI Cohort (600-718)**: 24 sessions
- **Severity Breakdown**:
  - Severe ($20 - 24$ pts): 2 sessions
  - Moderately Severe ($15 - 19$ pts): 5 sessions
  - Moderate ($10 - 14$ pts): 4 sessions
  - Mild ($5 - 9$ pts): 9 sessions
  - Minimal ($0 - 4$ pts): 13 sessions

---

## Complete Participant Roster

| Participant ID | Cohort | Severity Category | Category Folder | Ground-Truth PHQ-8 | Gender | Age |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **624** | AI (600s) | Severe | `unseen/severe/` | **22.0** | Male | 53 |
| **321** | WoZ (300s) | Severe | `unseen/severe/` | **20.0** | Female | 51 |
| **638** | AI (600s) | Moderately Severe | `unseen/moderately_severe/` | **19.0** | Female | 29 |
| **649** | AI (600s) | Moderately Severe | `unseen/moderately_severe/` | **19.0** | Male | 50 |
| **637** | AI (600s) | Moderately Severe | `unseen/moderately_severe/` | **16.0** | Female | 33 |
| **414** | WoZ (300s) | Moderately Severe | `unseen/moderately_severe/` | **16.0** | Female | 21 |
| **338** | WoZ (300s) | Moderately Severe | `unseen/moderately_severe/` | **15.0** | Female | 47 |
| **655** | AI (600s) | Moderate | `unseen/moderate/` | **12.0** | Female | 48 |
| **604** | AI (600s) | Moderate | `unseen/moderate/` | **12.0** | Male | 28 |
| **633** | AI (600s) | Moderate | `unseen/moderate/` | **10.0** | Female | 36 |
| **409** | WoZ (300s) | Moderate | `unseen/moderate/` | **10.0** | Male | 56 |
| **623** | AI (600s) | Mild | `unseen/mild/` | **9.0** | Male | 48 |
| **635** | AI (600s) | Mild | `unseen/mild/` | **9.0** | Male | 43 |
| **656** | AI (600s) | Mild | `unseen/mild/` | **8.0** | Female | 50 |
| **607** | AI (600s) | Mild | `unseen/mild/` | **7.0** | Female | 30 |
| **619** | AI (600s) | Mild | `unseen/mild/` | **6.0** | Female | 33 |
| **654** | AI (600s) | Mild | `unseen/mild/` | **6.0** | Male | 33 |
| **304** | WoZ (300s) | Mild | `unseen/mild/` | **6.0** | Female | 22 |
| **600** | AI (600s) | Mild | `unseen/mild/` | **5.0** | Female | 42 |
| **397** | WoZ (300s) | Mild | `unseen/mild/` | **5.0** | Male | 60 |
| **653** | AI (600s) | Minimal | `unseen/minimal/` | **4.0** | Female | 58 |
| **603** | AI (600s) | Minimal | `unseen/minimal/` | **4.0** | Male | 30 |
| **615** | AI (600s) | Minimal | `unseen/minimal/` | **3.0** | Male | 38 |
| **300** | WoZ (300s) | Minimal | `unseen/minimal/` | **2.0** | Male | 33 |
| **634** | AI (600s) | Minimal | `unseen/minimal/` | **2.0** | Female | 45 |
| **605** | AI (600s) | Minimal | `unseen/minimal/` | **2.0** | Male | 57 |
| **662** | AI (600s) | Minimal | `unseen/minimal/` | **2.0** | Female | 47 |
| **632** | AI (600s) | Minimal | `unseen/minimal/` | **1.0** | Female | 26 |
| **628** | AI (600s) | Minimal | `unseen/minimal/` | **0.0** | Female | 36 |
| **626** | AI (600s) | Minimal | `unseen/minimal/` | **0.0** | Male | 28 |
| **303** | WoZ (300s) | Minimal | `unseen/minimal/` | **0.0** | Female | 41 |
| **601** | AI (600s) | Minimal | `unseen/minimal/` | **0.0** | Male | 43 |
| **408** | WoZ (300s) | Minimal | `unseen/minimal/` | **0.0** | Female | 41 |
