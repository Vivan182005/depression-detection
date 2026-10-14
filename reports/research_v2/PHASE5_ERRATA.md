# Phase 5 Research Errata & Methodological Clarifications

**Date:** 2026-10-14  
**Branch:** `research-upgrade-2026-10-13`  
**Reference Report:** [`reports/research_v2/PHASE5.md`](PHASE5.md)  
**Status:** Authoritative Errata Notice  

---

### 1. Correction to Hypothesis Verdict: M7 vs. Metadata Baseline (M2)

In [`PHASE5.md`](PHASE5.md), Section 1 (Table 1, Hypothesis 3) stated:
> *"Models Beat Metadata Demographic Confound: SUPPORTED — M7 LogReg achieves Macro-F1 43.91% vs Metadata-only 36.41% (+7.62%)."*

**Correction & Re-evaluation:**
- **Empirical Status:** **NOT STATISTICALLY SUPPORTED** for M7.
- **Statistical Evidence:** While M7 achieved an empirical mean difference of $+7.62\%$ over M2, the participant-cluster paired bootstrap 95% confidence interval is **`[-0.19%, +15.18%]`** with a raw two-tailed empirical $p$-value of **$0.0640$** and a Holm-Bonferroni adjusted $p$-value of **$0.3600$** (see Section 5 Table). Because the 95% cluster CI crosses zero and the Holm-adjusted $p$-value exceeds $0.05$, M7 **does not** beat the metadata baseline with statistical significance under the prespecified criteria.
- **Clarification on `meta_7d`:** The `meta_7d` feature vector is not purely demographic; it consists of audio duration, total speech duration, turn count, mean turn duration, and speaking rate in addition to age and gender. Therefore, M2 captures interview length and speech density dynamics, not merely demographic shortcuts.
- **Only Model Beating Both Controls:** **`M5_BERT_T2_Only`** is the **sole** candidate model that statistically outperforms both the shuffled-label control M1 ($+12.38\%$, 95% CI: `[+5.93%, +19.35%]`, Holm $p < 0.0001$) and the metadata control M2 ($+11.97\%$, 95% CI: `[+4.51%, +19.53%]`, Holm $p = 0.0160$).

---

### 2. Leave-Domain-Out Benchmark: Methodological Requirements

In [`PHASE5.md`](PHASE5.md), Section 6 reported point estimates for zero-shot domain transfer (Train on WoZ Dev $N=178 \to$ Test on Avatar Dev $N=61$):
- Majority Baseline: Macro-F1 = 26.40%, Balanced Acc = 33.33%, AUC = 50.00%, Expected MAE = 5.54
- Trimodal MLP (T1): Macro-F1 = 41.64%, Balanced Acc = 44.17%, AUC = 68.45%, Expected MAE = 4.87
- Trimodal LogReg (M7): Macro-F1 = 45.41%, Balanced Acc = 45.65%, AUC = 81.79%, Expected MAE = 4.59

**Errata & Protocol Addition:**
- Zero-shot transfer point estimates must not be judged solely against the majority class. A rigorous domain-transfer evaluation requires:
  1. An explicit **in-fold shuffled-label control** for the transfer setting (to measure transfer under the empirical null hypothesis).
  2. **Cluster bootstrap confidence intervals** on transfer performance differences (Trimodal Transfer - Transfer Shuffled Control).
- These rigorous statistical controls are prespecified for execution in Phase 6A.

---

### 3. Finalist Model Selection: Sole Validated Candidate

In [`PHASE5.md`](PHASE5.md), Section 8 designated both `M7_Trimodal_LogReg` (multimodal) and `M5_BERT_T2_Only` (unimodal semantic) as finalists.

**Correction:**
- Under the prespecified rule requiring a candidate model to beat its control with a Holm-adjusted 95% CI strictly above zero, **`M5_BERT_T2_Only` is the ONLY validated finalist model from Phase 5**.
- `M7_Trimodal_LogReg` failed the multi-test significance gate against metadata ($p_{\text{Holm}} = 0.3600$). It represents an unvalidated research exploration rather than a statistically certified multimodal deliverable.
- Consequently, Phase 5 proved that clinical signal resides predominantly in semantic text transcripts rather than acoustic or multimodal feature stores in their current form.

---

### 4. Interpretation of Ablation (c): Sensitivity Check

In [`PHASE5.md`](PHASE5.md), Section 1 and Section 7 discussed Ablation (c) (excluding 24 uncalibrated avatar sessions, yielding 39.52% vs 40.59% Macro-F1).

**Correction:**
- Ablation (c) is a **sensitivity check**, not proof of complete boundary invariance or clinical robustness.
- The 1.07% drop in Macro-F1 and 3.01% drop in Binary AUC demonstrate that boundary uncertainty does introduce variance. It indicates that models are sensitive to conversational boundary quality, underscoring the necessity of objective turn-reconstruction validation.

---

### 5. Reconciliation of M7 Expected MAE

In [`PHASE5.md`](PHASE5.md), Section 8 text erroneously noted an Expected MAE of 4.38 for M7, whereas Section 2 Table recorded 4.68.

**Correction:**
- The verified pooled OOF Expected MAE for `M7_Trimodal_LogReg` is **`4.68`** (MAE between continuous PHQ-8 ground truth and in-fold class-mean weighted expected predictions). The value 4.38 was a typographic draft transcription error and is formally retracted.

---

### 6. Scientific Tone & Terminology Standards

To maintain rigorous scientific standards, the following overreaching terms used in [`PHASE5.md`](PHASE5.md) are formally retracted and replaced with measured, precise descriptions:

| Overreaching Term | Retraction Rationale | Standard Scientific Replacement |
| :--- | :--- | :--- |
| **"100% leak-free"** | Absolute claims are unverifiable and scientifically untenable. | *"Strict in-fold fitting of scalers, class weights, and calibration parameters without label disclosure."* |
| **"Clinical biomarker"** | BERT text representations reflect linguistic and semantic content of conversational answers, not biological or physiological biomarkers. | *"Semantic text representations"* or *"linguistic behavioral features"*. |
| **"Proving models do not exploit artifacts"** | Observational sensitivity analyses cannot establish negative causal proofs. | *"Sensitivity analysis showing performance is stable within 1.07% F1 when uncalibrated sessions are omitted."* |
| **"Multimodal Finalist" (for M7)** | M7 did not beat M2 metadata baseline with statistical significance ($p_{\text{Holm}}=0.3600$). | *"Exploratory multimodal linear baseline (did not achieve significance over length/demographic metadata)."* |

---

### 7. Summary of Verified Phase 5 Status

1. **Genuine Signal:** Confirmed exclusively in text representations (`M5_BERT_T2_Only`, Macro-F1 48.41%, Holm $p < 0.05$ vs both M1 and M2).
2. **Acoustic Alone:** Fails to exceed chance level (M3 Macro-F1 31.64% vs Shuffled 32.35%).
3. **Multimodal Fusion:** Currently dominated by text; adding acoustic and Wav2Vec2 degrades performance from 48.41% (BERT T2) down to 43.12% (Trimodal MLP T2) and 43.91% (Trimodal LogReg), failing to beat metadata with statistical significance.
4. **Frozen Artifacts:** Confirmation folds (seeds 100–104) and sealed cohort (33 participants) remain strictly untouched.
