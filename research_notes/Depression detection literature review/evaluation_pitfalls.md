# Evaluation Pitfalls, Data Leakage and Reproducibility in Automatic Depression Detection (DAIC-WOZ / E-DAIC)

Project context: binary PHQ-8>=10, speaker-disjoint repeated nested CV, N=272, accuracy 74.3% (majority-class 68.8%), AUC 0.764, balanced accuracy 67%.

Research note: about 17 tool calls. Several key 2025-2026 sources were behind 403 (ACM DL, MDPI, ResearchGate), so for those I had only abstracts or search snippets. Where that is the case, it is flagged.

---

## What accuracy/F1/AUC is realistic under speaker-independent evaluation?

### Takeaway
Under strictly speaker-disjoint evaluation, current speech and text models on DAIC-WOZ/E-DAIC reach roughly 0.55-0.74 macro-F1 and roughly 55-73% accuracy/UAR. Official-test-split text results can be lower still (0.40-0.68 macro-F1). Accuracies of 90% or more almost always come from speaker overlap, interviewer-prompt shortcuts or tuning on the 35-person dev set. Your 74.3% accuracy, 0.764 AUC and 67% balanced accuracy fall within the honest range.

### Cited Findings
- **Controlled speaker-leakage experiment on DAIC-WOZ, Yeh et al. 2026** (submitted to Interspeech 2026). The test set was the same in both conditions: 714 segments from 38 speakers. Results, speaker-disjoint (Set A) vs speaker-overlapped (Set B):
  - Fine-tuned Wav2Vec-SLS: F1 0.738 / accuracy 70.3% in Set A vs F1 0.983 / accuracy 98.3% in Set B.
  - Fine-tuned Wav2Vec linear probe: F1 0.562 / accuracy 58.7% vs F1 0.976 / accuracy 97.7%.
  - Frozen wav2vec linear probe: F1 0.528 / accuracy 54.1% vs F1 0.765 / accuracy 76.8%.
  - Under overlap, a speaker-ID probe reached 91-96% accuracy. DANN did not close the gap: fine-tuned Wav2Vec-SLS+DANN scored F1 0.659 speaker-disjoint vs 0.965 overlapped.
  - The authors say papers reporting more than 90% accuracy on DAIC-WOZ "may be inflated by leakage." [Yeh et al. 2026, arXiv 2604.14354](https://arxiv.org/html/2604.14354v1)
- **Subject-disjoint leave-one-subject-out CV on E-DAIC, Ishikawa & Duke 2026.** A text+LLM-score hybrid reached macro-F1 0.723, which the authors call the highest reported under this protocol. [Ishikawa & Duke 2026, arXiv 2605.23977](https://arxiv.org/abs/2605.23977)
- **Text models on official test splits, Watawana et al. 2026** (participant-only macro-F1):
  - DAIC-WOZ: 0.68 (Longformer) and 0.59 (GCN).
  - E-DAIC: 0.40 (Longformer) and 0.54 (GCN).
  - The same models scored 0.71/0.85 on the DAIC-WOZ dev set, a dev-to-test drop of up to 0.26 F1. [Watawana et al. 2026, arXiv 2603.24651](https://arxiv.org/html/2603.24651v2)
- **Acoustic models evaluated per speaker on the DAIC-WOZ test set, Gonzalez-Machorro et al. 2025.** Best SVM: UAR 73% (95% CI 57-84), F1 67. The wide CI reflects the small test set. [Gonzalez-Machorro et al. 2025, arXiv 2508.18092](https://arxiv.org/html/2508.18092v1)
- **JMIR Mental Health meta-analysis, Maran et al. 2025** (105 studies, speech-based depression detection).
  - Pooled *highest-reported* accuracy 0.81 (95% CI 0.79-0.83), sensitivity 0.84, specificity 0.83.
  - Pooled *lowest-reported* accuracy 0.66 (0.63-0.69), sensitivity 0.63, specificity 0.60.
  - Heterogeneity was extreme (I² 94-99.9%).
  - 47.6% of studies had high risk of bias in at least one domain. Only 43.8% were low risk in the analysis domain, and 93.3% gave too little detail on preprocessing.
  - DAIC-WOZ was the most-used dataset (35/105 studies). [Maran et al. 2025, JMIR Ment Health 12:e67802](https://pmc.ncbi.nlm.nih.gov/articles/PMC12590051/)
- **A second deep-learning speech meta-analysis** pooled accuracy at 0.87, but only 8 of its 25 studies entered the meta-analysis. [Strathclyde / PMC11413444](https://pmc.ncbi.nlm.nih.gov/articles/PMC11413444/)
- **Berisha et al., Interspeech 2022** (77 clinical-speech ML papers, dementia). Reported accuracy *decreases* as sample size grows, which the authors attribute to publication bias plus overfitting. A follow-up tutorial found the same negative correlation across 59 classifiers and named data leakage and analytic flexibility as causes. [Berisha et al. 2022](https://www.isca-archive.org/interspeech_2022/berisha22_interspeech.html); [Tutorial, arXiv 2410.21640](https://arxiv.org/html/2410.21640v1)

### Inferences
- An honest speaker-independent band for binary PHQ-8>=10 on DAIC-WOZ/E-DAIC is about macro-F1 0.55-0.74, accuracy/UAR 55-75% and AUC about 0.65-0.80. The AUC range is inferred, since few papers report AUC. Your results (AUC 0.764, balanced accuracy 67%) sit at the upper-middle of that band.
- Your N=272 with repeated nested CV is statistically sturdier than results on the 35- or 47-person official splits, where a 95% CI spans about 25 percentage points (see the 57-84% UAR CI above).
- Berisha's sample-size finding supports the argument that small-N, single-split, high-accuracy papers are the least trustworthy.

### Gaps
- I found no meta-analysis that pools results restricted to speaker-independent DAIC-WOZ evaluations. Maran et al. do not code for split type.
- AUC is rarely reported, so I found no sourced "honest AUC" range.

---

## Which highly cited papers have been shown to leak or be irreproducible?

### Takeaway
Burdisso et al. 2024 identify the high-F1 DAIC-WOZ papers that feed interviewer (Ellie) prompts into the model; these papers report F1 of 0.84-0.96. Systematic reproducibility audits (Danylenko & Unold 2026) find that only 5 of 66 DAIC-WOZ papers meet minimal reproducibility standards. Without subject leakage, models did worse than a mean predictor.

### Cited Findings
- **Papers Burdisso et al. 2024 list as using Ellie/therapist prompts**, with their reported F1:
  - Dai et al. 2021 (multimodal): 0.96
  - Niu et al. 2021 (multimodal): 0.92
  - Zhuang et al. 2024 (multimodal): 0.88
  - Toto et al. 2021 (AudiBERT): 0.86
  - Shen et al. 2022: 0.83 (text) and 0.85 (multimodal)
  - Williamson et al. 2016: 0.84
  - Milintsevich et al. 2023: 0.80
  - Agarwal & Dias 2024: 0.77
  - The authors note that prompt-using works "exhibit an overall superior performance." [Burdisso et al. 2024, ClinicalNLP](https://arxiv.org/html/2404.14463v1). Caveat: I reconstructed the mapping of papers to the prompt column from a flattened table; verify it against the PDF.
- **Danylenko & Unold, *Applied Sciences* 2026, 16(1):422** (systematic review of DAIC-WOZ depression-severity / MAE papers up to Sept 2025).
  - 536 papers were found, 414 after deduplication, and 66 entered quality assessment.
  - Only 5 met minimal reproducibility standards: clear data partitioning, model description and training protocol.
  - The review names subject leakage as a critical flaw: "without leakage, the model consistently performed worse than a simple mean predictor."
  - Reported MAEs are about the size of PHQ-8 variability. [Search snippet of Danylenko & Unold 2026](https://www.mdpi.com/2076-3417/16/1/422); [ResearchGate entry](https://www.researchgate.net/publication/399279944_Common_Pitfalls_and_Recommendations_for_Use_of_Machine_Learning_in_Depression_Severity_Estimation_DAIC-WOZ_Study). Full text was 403; this is from abstract/snippets only.
- **A preliminary large-scale reproducibility study at ACM** argues that most DAIC-WOZ classifiers are invalid. In its framing, they learn general distress or paralinguistic cues (e.g., flat affect, vocal strain) rather than task-specific features. [ACM DL 10.1145/3747327.3763034](https://dl.acm.org/doi/full/10.1145/3747327.3763034). Full text was 403; authors and numbers are unverified.
- **Bailey & Plumbley, EUSIPCO 2021:** gender bias in DAIC-WOZ "can lead to an overreporting of performance." Data re-distribution and raw-audio features reduced the effect. The abstract gives no figures. [Bailey & Plumbley 2021, arXiv 2010.15120](https://arxiv.org/abs/2010.15120)
- **Leaderboard instability on the official E-DAIC split** (96 configurations):
  - The best-by-CV configuration ranked 20th on the official test.
  - The official-test winner ranked 41st by CV.
  - Top-3 overlap between the two rankings was zero.
  - The "winner" was rank-1 in only 32.3% of subject bootstraps. [Ishikawa & Duke 2026](https://arxiv.org/abs/2605.23977)
- **Speaker-disentanglement papers by Ravi et al.** report DAIC-WOZ audio F1 of 0.73-0.80 (0.83 with text fusion). These are the authors' own figures and I found no independent replication. They are useful as the "upper end" of audio-only claims. [Ravi et al. 2024, CSL](https://pmc.ncbi.nlm.nih.gov/articles/PMC10836190/); [Wang et al. Interspeech 2023](https://www.isca-archive.org/interspeech_2023/wang23pa_interspeech.html)

### Inferences
- Any DAIC-WOZ paper above about 0.85 F1 should be checked for three things. Does it include Ellie's turns? Does it split at segment level? Does it report on the 35-person dev set with tuning on that set? The Dai 2021, Niu 2021 and Zhuang 2024 results (0.88-0.96) combine prompt use with dev-set reporting.

### Gaps
- I could not access the ACM reproducibility study for its original-vs-reproduced numbers.
- I found no GitHub-issue evidence of failed reproductions of specific repos. Not searched in depth; the idiap/bias_in_daic-woz repo reproduces Burdisso's own numbers: [GitHub](https://github.com/idiap/bias_in_daic-woz).

---

## Which interview questions/prompts leak label information, and how big is the effect?

### Takeaway
Ellie's scripted follow-ups about mental-health history and therapy are only asked conditionally, so they reveal the label. Examples are "Do you still go to therapy?", whether therapy is useful, "How do you cope with that?" and "Have you been diagnosed with depression?". Interviewer-only models match or beat participant-only models on the dev set (DAIC-WOZ GCN 0.88 vs 0.85 F1). Combining both gives 0.90 F1, the highest text-only result on DAIC-WOZ. The effect is unstable on the test set.

### Cited Findings
- **Burdisso et al. 2024, DAIC-WOZ evaluation set (35 files: 23 control, 12 depressed), macro-F1:**

  | Model | Ellie-prompt-only | Participant-only |
  |---|---|---|
  | GCN | 0.88 | 0.85 |
  | longBERT | 0.84 | 0.72 |

  - The P-GCN ∧ E-GCN ensemble reached 0.90, the highest reported text-only result on DAIC-WOZ, obtained by deliberately exploiting the bias.
  - E-GCN evidence clusters in the second half of interviews, where Ellie asks about past mental-health experiences. One example relied on only four Ellie questions, including "Have you been diagnosed with depression?"
  - E-longBERT scored 0.84 F1 on the second half of interviews vs 0.60 on the first half. [Burdisso et al. 2024](https://arxiv.org/html/2404.14463v1)
- **Watawana et al. 2026 (DAIC-WOZ, E-DAIC, ANDROIDS):**
  - The interviewer-only GCN relies mainly on "How do you cope with that?", "Do you still go to therapy?" and a follow-up on whether therapy is useful. It largely ignores PTSD-screening and symptom questions.
  - Dev-set interviewer-minus-participant gap: +0.03 F1 on DAIC-WOZ, +0.04 on E-DAIC and up to +0.19 on ANDROIDS (Longformer: interviewer-only 0.98 vs participant-only 0.79).
  - On test the gap is inconsistent: DAIC-WOZ GCN +0.03 but Longformer -0.15; E-DAIC Longformer +0.16.
  - Interviewer-only models show narrow, high-contrast evidence bands at fixed positions in the interview. [Watawana et al. 2026](https://arxiv.org/html/2603.24651v2)
- **Symptom-content shortcut in E-DAIC text models:** text scores rise sharply on symptom-dense slices while audio stays flat. The gap was positive across all 5 seeds. [Ishikawa & Duke 2026](https://arxiv.org/abs/2605.23977)

### Inferences
- The effect is largest on the dev set and shrinks or reverses on the test set. That suggests part of the prompt advantage is itself dev-set overfitting. Either way, prompt inclusion is not a legitimate source of signal.
- For your project, any text features should come from participant turns only. If your Whisper-based transcripts (`woz_whisper/`) do not separate speakers, Ellie's prompts may be leaking in; check diarization.
- Even participant-only text can carry "proxy" leakage: participants' *answers* to the conditional therapy questions ("yes, I was diagnosed...") encode the label. A within-transcript masking ablation would quantify this.

### Gaps
- I found no paper quantifying leakage from participant answers to those specific questions, as opposed to Ellie's prompts.

---

## What evaluation protocol is recommended?

### Takeaway
The consensus recommendations come from Burdisso 2024, Watawana 2026, Yeh 2026, Ishikawa & Duke 2026, Danylenko & Unold 2026 and Maran 2025:
- Split speaker-disjoint (subject-level) only.
- Use participant-only inputs.
- Use subject-disjoint CV or LOSO with out-of-fold reporting and bootstrap CIs, not a single official split.
- Report macro-F1, AUC, MCC or balanced accuracy, plus a majority-class baseline.
- Do not tune on the reported set.
- Validate externally on a second corpus.
- Report full preprocessing details.

### Cited Findings
- **Speaker independence:** use "strictly speaker-independent evaluation." Conventional protocols "overestimate generalization and clinical utility." [Yeh et al. 2026](https://arxiv.org/abs/2604.14354)
- **Participant-only inputs and evidence localization:** use participant turns only, or always report participant-only results next to participant+interviewer results. Localize decision evidence by time and speaker. [Watawana et al. 2026](https://arxiv.org/html/2603.24651v2); [Burdisso et al. 2024](https://arxiv.org/html/2404.14463v1)
- **Out-of-fold estimates:** use subject-disjoint LOSO and report out-of-fold results. Do not rank fine-grained configurations on one official split; check stability with subject bootstraps; validate externally. [Ishikawa & Duke 2026](https://arxiv.org/abs/2605.23977)
- **Metrics under imbalance:** accuracy, sensitivity and specificity "may not fully capture model performance" under imbalance. Use F1, AUC and MCC. DAIC-WOZ is about 4:1 non-depressed to depressed and has a gender imbalance in prevalence. [Maran et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12590051/)
- **Reproducibility minimum:** document data partitioning, model description and training protocol. [Danylenko & Unold 2026](https://www.mdpi.com/2076-3417/16/1/422)
- **Gender:** report gender-stratified results and rebalance. [Bailey & Plumbley 2021](https://arxiv.org/abs/2010.15120)

### Inferences
- Your protocol (speaker-disjoint repeated nested CV, majority baseline reported, AUC and balanced accuracy) already meets or exceeds these recommendations. Three additions would close the remaining gaps:
  - bootstrap CIs on AUC and balanced accuracy;
  - macro-F1 and MCC;
  - gender-stratified metrics and an external or cross-corpus test.

---

## Supporting pitfall details (segment splits, dev-set reporting, class imbalance, cross-dataset, AVEC test labels)

### Takeaway
The official DAIC-WOZ split is 107 train / 35 dev / 47 test, and test labels were withheld for AVEC. As a result, most papers report on the 35-person dev set, often after tuning on it. On that set a trivial majority classifier scores about 66% accuracy, and a single misclassified participant moves accuracy by about 3 points. Cross-corpus transfer costs several points even in careful work.

### Cited Findings
- **Official split:** 189 sessions = 107 train / 35 dev / 47 test. Test labels are hidden "due to the AVEC competition." Papers report dev-set results "consistent with previous research" and "to ensure fair comparisons." [arXiv 2210.02527](https://arxiv.org/pdf/2210.02527); [arXiv 2210.15261](https://arxiv.org/pdf/2210.15261); [Burdisso et al. 2024 footnote](https://arxiv.org/html/2404.14463v1). Source attribution is inconsistent (AVEC 2016 vs 2017).
- **Dev-set composition:** 23 control / 12 depressed. [Burdisso et al. 2024](https://arxiv.org/html/2404.14463v1)
- **Tuning on the reported set:** Burdisso et al. ran 100 Optuna trials maximizing macro-F1, and the paper does not state which split was used for that search. This is an example of the ambiguity common in the literature. [Burdisso et al. 2024](https://arxiv.org/html/2404.14463v1)
- **Dev-to-test drop:** dev-set F1 does not carry over to test, e.g. DAIC-WOZ GCN participant-only 0.85 on dev vs 0.59 on test. [Watawana et al. 2026](https://arxiv.org/html/2603.24651v2)
- **Segment-level splits:** speaker overlap inflates F1 from about 0.53-0.74 to 0.76-0.98 (see table under the first question). [Yeh et al. 2026](https://arxiv.org/html/2604.14354v1). Older work makes the same point: when speaker IDs and depression state are perfectly correlated, detectors behave like speaker identifiers. [Zuo et al. 2023, Pattern Recognition Letters (PolyU)](https://research.polyu.edu.hk/en/publications/avoiding-dominance-of-speaker-features-in-speech-based-depression/)
- **Cross-corpus examples:**
  - DAIC-WOZ English to German multiple-sclerosis speech: UAR fell from 73% in-domain to 66% in transfer (74% vs 79% with feature selection). [Gonzalez-Machorro et al. 2025](https://arxiv.org/html/2508.18092v1)
  - CMDC and ANDROIDS baselines that are near-ceiling in-domain are "substantially weaker" zero-shot on external corpora. [Ishikawa & Duke 2026](https://arxiv.org/abs/2605.23977)
  - E-DAIC to MODMA transfer is confounded by age, recording conditions, interview structure and PHQ-8 vs PHQ-9 labels. [arXiv 2607.02920](https://arxiv.org/pdf/2607.02920)

### Inferences
- **Majority-class arithmetic** (computed here, not from a source):
  - On the 35-person dev set, predicting "not depressed" gives 23/35 = 65.7% accuracy but a minority-class F1 of 0.
  - Each participant is worth 2.9 accuracy points.
  - Papers reporting the F1 of the majority (non-depressed) class, or weighted F1, can show high numbers with poor depressed-class detection. Burdisso's table separates the depressed-class F1 (D) from the control-class F1 (C): D is consistently 0.06-0.10 lower.
- Your majority baseline (68.8%) vs accuracy (74.3%) is a +5.5-point gain. Framing results through balanced accuracy and AUC, as you do, is the honest choice.
- Watawana et al. 2026 report DAIC-WOZ *test-set* scores, which suggests test labels are now available to researchers. I could not confirm when or how they were released.

### Gaps
- I could not confirm whether, or when, DAIC-WOZ / E-DAIC test labels were publicly released (e.g., in later corpus distributions).
- I found no sourced, quantified list of specific papers that used segment-level random splits. Yeh et al. allude to papers [8-12] reporting more than 90% but do not name their protocols in the fetched text.
- No IEEE T-AFFC, npj Digital Medicine or Neuroscience & Biobehavioral Reviews review was retrieved in this pass; only JMIR (Maran 2025) and *Applied Sciences* (Danylenko & Unold 2026) are covered.
