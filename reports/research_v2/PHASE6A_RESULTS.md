# Phase 6A Results: Voice-Primary Depression Detection (PHQ-8 >= 10)

Executed 2026-10-15 19:40:32 | total runtime 123.4 min | protocol prereg v1.1 + amendments v1.2, v1.3, v1.3b, v1.3c

Cohort: N=239 dev participants (178 WoZ, 61 Avatar; 84 F / 155 M; 74 depressed / 165 not). Sealed 33 and seeds 100-104 untouched.

Primary endpoint: participant-level ROC-AUC of out-of-fold probabilities averaged over seeds 0-4 (5 x 5-fold stratified CV, all tuning in inner 4-fold CV). 95% CI = participant bootstrap (B=2000).

## 1. All 19 variants

| Variant | Role | Description | AUC | 95% CI | WoZ AUC | Avatar AUC | Female AUC | Male AUC | 3-class macro-F1 | Severe recall |
|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| V1 | Exploratory | eGeMAPS 135d speech-only + elastic-net LR | **0.478** | [0.400, 0.558] | 0.453 | 0.532 | 0.410 | 0.468 | 0.307 | 0.056 |
| V2 | Confirmatory | Hand-crafted VAD-aggregated 330d + L2 LR | **0.504** | [0.426, 0.583] | 0.543 | 0.380 | 0.630 | 0.416 | 0.303 | 0.250 |
| V3 | Exploratory | Hand-crafted 330d + HistGBDT | **0.505** | [0.423, 0.588] | 0.510 | 0.477 | 0.515 | 0.475 | 0.358 | 0.056 |
| V4 | Exploratory | Hand-crafted CMVN 330d + L2 LR | **0.447** | [0.369, 0.524] | 0.465 | 0.401 | 0.502 | 0.412 | 0.336 | 0.000 |
| V5 | Exploratory | Hand-crafted CMVN 330d + HistGBDT | **0.507** | [0.425, 0.588] | 0.515 | 0.473 | 0.543 | 0.470 | 0.358 | 0.028 |
| V6 | Exploratory | wav2vec2 L4 mean + L2 LR | **0.578** | [0.496, 0.657] | 0.538 | 0.682 | 0.572 | 0.556 | 0.375 | 0.500 |
| V7 | Exploratory | wav2vec2 L8 mean + L2 LR | **0.710** | [0.639, 0.777] | 0.655 | 0.868 | 0.671 | 0.724 | 0.460 | 0.611 |
| V8 | Exploratory | wav2vec2 L12 mean + L2 LR | **0.572** | [0.492, 0.648] | 0.582 | 0.548 | 0.556 | 0.568 | 0.366 | 0.389 |
| V9 | Exploratory | WavLM L4 mean + L2 LR | **0.521** | [0.438, 0.604] | 0.475 | 0.664 | 0.508 | 0.503 | 0.364 | 0.194 |
| V10 | Confirmatory | WavLM L8 mean + L2 LR | **0.644** | [0.566, 0.721] | 0.633 | 0.699 | 0.623 | 0.646 | 0.444 | 0.417 |
| V11 | Exploratory | WavLM L12 mean + L2 LR | **0.571** | [0.492, 0.649] | 0.558 | 0.625 | 0.624 | 0.519 | 0.348 | 0.194 |
| V12 | Exploratory | Segment hand-crafted 55d + MLP | **0.483** | [0.401, 0.558] | 0.448 | 0.570 | 0.394 | 0.485 | 0.309 | 0.083 |
| V13 | Exploratory | Segment WavLM L8 + MLP | **0.590** | [0.510, 0.669] | 0.548 | 0.712 | 0.548 | 0.566 | 0.366 | 0.111 |
| V14 | Exploratory | BERT on Whisper-tiny (VAD-chunked) text + L2 LR | **0.704** | [0.632, 0.773] | 0.694 | 0.731 | 0.688 | 0.706 | 0.434 | 0.444 |
| V15 | Control | Length-only control | **0.574** | [0.496, 0.649] | 0.561 | 0.604 | 0.630 | 0.547 | 0.402 | 0.000 |
| V16 | Control | Gender-only control | **0.479** | [0.399, 0.557] | 0.483 | 0.435 | 0.302 | 0.365 | 0.325 | 0.000 |
| V17 | Control | Domain-only control | **0.419** | [0.340, 0.499] | 0.358 | 0.144 | 0.448 | 0.411 | 0.311 | 0.000 |
| V18 | Confirmatory | Stack V2 + V14 | **0.685** | [0.609, 0.758] | 0.675 | 0.714 | 0.703 | 0.663 | 0.395 | 0.028 |
| V19 | Exploratory | Stack inner-CV best audio + V14 | **0.714** | [0.640, 0.781] | 0.674 | 0.812 | 0.689 | 0.709 | 0.489 | 0.333 |

## 2. Confirmatory family (Holm, m = 3, alpha = 0.05)

- **H1: V2 hand-crafted voice vs stratified permutation null**: observed AUC 0.504; null mean 0.502, null 95th pct 0.591; empirical p = 0.4890 (R=999)
- **H2: V10 WavLM-L8 vs stratified permutation null**: observed AUC 0.644; null mean 0.503, null 95th pct 0.586; empirical p = 0.0030 (R=999)
- **H3: V18 beats V14 AND V2**: dAUC(V18-V14) = -0.018 [-0.047, +0.011], p=0.8930; dAUC(V18-V2) = +0.181 [+0.087, +0.276], p=0.0002; IUT p = 0.8930

| Step | Hypothesis | raw p | Holm threshold | Decision |
|--:|:--|--:|--:|:--|
| 1 | H2 (WavLM Layer 8 V10 vs Null) | 0.0030 | 0.0167 | SUPPORTED (Statistically Significant) |
| 2 | H1 (Handcrafted Voice V2 vs Null) | 0.4890 | 0.0250 | NOT SUPPORTED (Failed to exceed threshold) |
| 3 | H3 (Multimodal Synergy V18 vs Constituents) | 0.8930 | 0.0500 | NOT SUPPORTED (Failed to exceed threshold) |

## 3. Pre-stated freeze rule (v1.3 sec 4)

- Primary audio: **V7** (AUC 0.710, exploratory-derived)
- Fusion: **V19** (AUC 0.714, exploratory-derived)
- Freezing authorises no run on seeds 100-104 or the sealed 33 (next step requires explicit sign-off).

## 4. Leakage / confound audit triggers (v1.1 sec 6.3)

- Variants above the audit thresholds: none

- Best metadata control AUC: 0.574. Any variant whose CI overlaps this value has not shown signal beyond length/gender/domain.


## 5. Provenance

- Analysis code frozen before the real-label run at git tag `prereg-exec-v1` (commit `ae41ae4`).
- Raw numbers: `experiments/research_v2/phase6a_results.json`, `phase6a_oof_predictions.csv`, `frozen_config.json`, `phase6a_run.log`.
- Folder note: the project was renamed from `~/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13` to `~/Desktop/CAPSTONE` after this run; dated protocol documents keep the original path.
