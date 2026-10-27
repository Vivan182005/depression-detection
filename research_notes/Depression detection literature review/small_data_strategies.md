# Small-data strategies for depression detection from speech/text (DAIC-WOZ / E-DAIC focus)

Scope note: project baseline for comparison is binary PHQ-8>=10, N=272 (DAIC-WOZ + E-DAIC), speaker-disjoint nested CV, frozen SSL speech + emotion-RoBERTa text, ridge + logistic stacker: accuracy 74.3%, AUC 0.764 (majority 68.8%). Already tried without benefit: audio augmentation, segment bagging, mixup, TTA, ProtoNet few-shot, kNN/prototype one-shot, speaker-reference normalisation, domain mean centring, mean+std pooling, class-balanced weights, per-domain thresholds.

Research budget: about 15 tool calls. Most numbers come from abstracts, HTML full text, or search-engine summaries of papers. Where a figure came only from a search snippet and was not checked in the paper, it is marked "(snippet, unverified)".

## What has actually increased honest accuracy/F1/AUC on DAIC-WOZ or E-DAIC, and by how much?

### Takeaway
Few gains hold up under strict speaker-disjoint evaluation. The best documented honest reference is macro-F1 0.723 on E-DAIC under leave-one-subject-out CV, from text plus LLM-derived scores. Most reported improvements are +0.02 to +0.05 F1 on the 35-speaker DAIC-WOZ dev set or the 47-speaker test set, which is within seed and split noise. Many headline results (F1 0.85 to 0.92) come from interviewer-prompt shortcuts or from dev-set selection.

### Cited Findings

**Honest-evaluation reference points and audits**
- A 2026 multi-probe audit covering DAIC/E-DAIC, CMDC, ANDROIDS, MODMA and PDCH reports macro-F1 = 0.723 for a lightweight text plus LLM-score model under leave-one-subject-out CV on E-DAIC. The authors call this the highest reported under that protocol, to their knowledge, and frame it as a conservative out-of-fold reference. — [Multi-Probe Audit, arXiv 2605.23977](https://arxiv.org/abs/2605.23977)
- The same audit swept 96 configurations. CV rankings and official-test rankings agreed only moderately: the best-CV configuration ranked 20th on the official test, the official-test winner ranked 41st in CV, top-3 overlap was zero, and the apparent winner was rank 1 in only 32.3% of subject bootstraps. — [arXiv 2605.23977](https://arxiv.org/abs/2605.23977)
- In that audit, text-model scores rose sharply on symptom-dense interview slices while audio-model scores stayed nearly flat, so text carries most of the signal. — [arXiv 2605.23977](https://arxiv.org/abs/2605.23977)
- Burdisso et al. (ClinicalNLP 2024) show that models using Ellie's (interviewer) prompts exploit a late interview segment where mental-health-history questions are asked. Deliberately exploiting this bias reached F1 0.90 text-only, the highest text-only DAIC-WOZ result at the time. Models using participant responses alone draw on the whole interview. — [Burdisso et al., arXiv 2404.14463](https://arxiv.org/html/2404.14463v1)
- Ellie draws from a fixed repertoire of 191 prompts. — [When Consistency Becomes Bias, arXiv 2603.24651](https://www.alphaxiv.org/abs/2603.24651)

**(a) Multi-task learning (PHQ-8 items, regression + classification)**
- QuestMF (E-DAIC) predicts each PHQ-8 question separately and trains with an Imbalanced Ordinal Log-Loss (ImbOLL). Mean CCC over 3 seeds: total-score regression 0.602±0.015; item-wise MSE 0.602±0.024 (no gain); item-wise OLL 0.640±0.018; item-wise ImbOLL 0.662±0.022. RMSE was not better (5.25 vs 5.10 for total regression) and MAE was about equal (3.95 vs 3.99). Binary F1 was not reported. — [QuestMF, arXiv 2503.20496](https://arxiv.org/html/2503.20496v1)
- In QuestMF, text alone reached CCC 0.615 and audio alone 0.273. Text+audio reached 0.643 and all three modalities 0.662. — [arXiv 2503.20496](https://arxiv.org/html/2503.20496v1)
- In QuestMF, E-DAIC per-item labels exist only for train and validation, not test. — [arXiv 2503.20496](https://arxiv.org/html/2503.20496v1)
- U-Fair treats each PHQ-8 item as a task, with gender-based uncertainty reweighting. Its abstract says multitask learning improves performance and fairness over unitask, but gains are "not always consistent", with negative transfer and a reduced Pareto frontier. Uncertainty reweighting eases this "to a certain extent". — [U-Fair, arXiv 2501.09687](https://arxiv.org/abs/2501.09687)
- PhqCML / Explain-DAIC uses PHQ-8 item scores only as training-time supervision for PHQ-aware representation learning. Item vectors are not inference inputs. — [arXiv 2501.16106](https://arxiv.org/pdf/2501.16106) (snippet, unverified)

**(c)/(d) SSL representations and fine-tuning (closest evidence on in-domain adaptation)**
- Wu, Zhang & Woodland (ICASSP 2023), DAIC-WOZ dev (35 speakers), mean over 20 seeds, best layer:
  - Frozen layers: wav2vec2 F1-avg 0.627, HuBERT 0.667, WavLM 0.700.
  - Fine-tuned: WavLM on emotion recognition 0.720 (+0.020 over frozen); wav2vec2 on ASR 0.639; wav2vec2 on emotion 0.640.
  - Seed std was 0.024 to 0.036, so these fine-tuning gains are within noise.
  — [Wu et al., arXiv 2305.12263](https://arxiv.org/html/2305.12263v2)
- In Wu et al., the best middle layer beat the final layer: WavLM block 8 was best when frozen. — [arXiv 2305.12263](https://arxiv.org/html/2305.12263v2)
- In Wu et al., concatenating RoBERTa (on ASR hypotheses) with WavLM features gave F1-avg 0.725 using the frozen WavLM and 0.756 using the emotion-fine-tuned WavLM. A speech-only ensemble of wav2vec2+HuBERT+WavLM gave 0.800. A full audio+emotion+text ensemble gave 0.829 avg / 0.886 max. All results are on the dev set, which was also used for layer selection, so they are likely optimistic. — [arXiv 2305.12263](https://arxiv.org/html/2305.12263v2)
- In Wu et al., sub-dialogue shuffling (random contiguous sub-dialogues, class-balanced counts) mattered a lot for end-to-end training. F1-avg rose from 0.451 at M+=100 to 0.679 at M+=1000, and std fell from 0.131 to 0.027. This applies to sequence models trained on segments, not to frozen-embedding + ridge pipelines. — [arXiv 2305.12263](https://arxiv.org/html/2305.12263v2)

**(d) LoRA / LLM approaches**
- M3L (Interspeech 2025) fine-tunes Whisper + XLM-RoBERTa with LoRA on question-answer segments, sampling class-balanced 10-pair chunks, for DAIC-WOZ and EATD. Its LAFT step reportedly "consistently boosts performance" on both corpora. Exact numbers were not retrieved because the PDF did not parse. — [M3L, ISCA Archive](https://www.isca-archive.org/interspeech_2025/you25_interspeech.pdf) (snippet, unverified)
- A PubMed-indexed study reports a fine-tuned, quantized LLM trained generatively (no classification head) reaching F1 84% on DAIC-WOZ, above prompt-engineered models. The split and speaker handling were not confirmed. — [PubMed 41337372](https://pubmed.ncbi.nlm.nih.gov/41337372/) (snippet, unverified)
- LLM-extracted interpretable features plus linear regression reportedly gave PHQ-8 MAE 2.91 on the DAIC-WOZ test set and 2.86 on E-DAIC. — [PMC12885269](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12885269/) (snippet, unverified)
- Sadeghi et al. report E-DAIC dev MAE 3.65 with a language-model-only approach (BHI 2023). Their 2024 npj Mental Health Research follow-up reports MAE 2.85 / RMSE 4.02 for LLM-extracted text indicators combined with speech-quality features. — [FAU CRIS (2023)](https://cris.fau.de/publications/314369033); [DOAJ npj MHR 2024](https://doaj.org/article/b9b5632407f544db89511e702ed06aa8) (snippet, unverified)
- Dep-LLM (2026) is training-free. It reports macro-F1 across 21 LLMs and notes that zero-shot approaches are sensitive to distribution shift. — [arXiv 2606.10796](https://arxiv.org/html/2606.10796v1) (snippet, unverified)

**(e) Question/topic-aware modelling**
- HiQuE (CIKM 2024) models primary vs follow-up question hierarchy.
  - DAIC-WOZ: macro-F1 0.79 vs 0.75 for the best baseline (GRU/BiLSTM). The ablation shows the hierarchical question embedding adds +0.03 macro-F1 (0.76 → 0.79). The split is presumably the 47-speaker test set.
  - E-DAIC: F1 0.70 vs 0.67 for the best baseline. E-DAIC lacks interviewer transcripts, so questions were mapped to the predefined list with BERTScore.
  — [HiQuE, arXiv 2408.03648](https://arxiv.org/html/2408.03648v1)
- Topic-attentive transformer: models trained on single topics performed poorly. The whole-interview model (F1 64.7) beat ten topic-subset models. — [arXiv 2206.13256](https://arxiv.org/pdf/2206.13256)
- Hierarchical attention on DAIC-WOZ transcripts reached test UAR 0.66 (Mallol-Ragolta et al., Interspeech 2019). HCAG (ICASSP 2021) claims F1 0.92, but question/interviewer-structure exploitation of this kind is exactly what Burdisso et al. flag. — [TUM portal](https://portal.fis.tum.de/en/publications/a-hierarchical-attention-network-based-approach-for-depression-de/); [ICASSP 2021](https://rc.signalprocessingsociety.org/conferences/icassp-2021/spsicassp21vid1081); [Burdisso et al.](https://arxiv.org/html/2404.14463v1)

**(h) Synthetic data**
- DepressGEN (Interspeech 2025) generates synthetic depressed-patient dialogue plus TTS speech. It reports a +9.9% F1 gain on EATD, not DAIC-WOZ. — [ISCA Archive liang25](https://www.isca-archive.org/interspeech_2025/liang25_interspeech.pdf) (snippet, unverified)
- One pipeline uses an LLM to generate synopses of DAIC-WOZ transcripts conditioned on target depression scores, to rebalance the severity distribution. The authors frame this as privacy-preserving augmentation, not a speaker-independent benchmark gain. — [arXiv 2411.17672](https://arxiv.org/pdf/2411.17672)
- Persona-conditioned synthetic sessions have been used against DAIC-WOZ class imbalance. — [arXiv 2604.24126](https://arxiv.org/pdf/2604.24126) (snippet, unverified)

**(g) Label noise near the PHQ cutoff**
- One preprocessing study found 20 E-DAIC instances with PHQ score >=10 but binary label 0, and recommends correcting them. — [search summary citing arXiv 2412.10417 / related](https://arxiv.org/pdf/2412.10417) (snippet, unverified; source attribution uncertain)

### Inferences
- **Most actionable for this project: (1) and (2).**
  1. Recompute the binary label from the PHQ-8 score (a label audit). This is free, and the reported 20 E-DAIC mismatches could materially affect a 272-speaker study.
  2. Exclude interviewer turns or interview-position shortcuts, so the 74.3% figure stays honest and comparable to the 0.723 LOSO macro-F1 reference.
- **Ordinal item-level targets (QuestMF-style).** These are the most promising multi-task option and are CPU-feasible with frozen embeddings, for example ridge/ordinal-logistic heads per PHQ-8 item, summed to a total and then thresholded. However:
  - the reported gain is in CCC (+0.04 to +0.06), not RMSE;
  - item decomposition with MSE alone gave nothing.
  Expect at most a small accuracy/AUC change.
- **SSL layer choice and emotion fine-tuning give at most about +0.02 F1, within noise.** Using a middle layer (for example WavLM block 8 to 10) rather than the last layer is cheap to check if not already done. Full fine-tuning on CPU is impractical in hours.
- **Multi-model ensembling (several SSL backbones plus text) gave the largest gains in Wu et al.:** 0.72 single model vs 0.80 to 0.83 ensemble. Those were dev-set-selected, but stacking heterogeneous frozen backbones is CPU-feasible: extraction costs hours per backbone, while training costs seconds.
- **The 74.3% / AUC 0.764 result is probably near the honest ceiling for this data size.** Given the audit's leaderboard instability, any new gain under +2 to 3 accuracy points on N=272 should come with a bootstrap CI or a permutation test.
- **LoRA fine-tuning of Whisper/XLM-R or of an LLM is not CPU-feasible in hours.** Prompting a small local LLM to produce PHQ-item scores from transcripts, used as stacker features (as in the 0.723 audit model and in Sadeghi et al.), is feasible on CPU, slowly. Running a 7 to 8B model quantized over 272 transcripts probably takes several hours.

### Gaps
- No 2022 to 2026 paper found reports a statistically tested gain on combined DAIC-WOZ+E-DAIC with speaker-disjoint nested CV, the project's exact protocol.
- No evidence found on ensembling/calibration tricks (temperature scaling, isotonic regression) improving accuracy on DAIC-WOZ.
- No evidence found for contrastive or self-supervised pretraining on in-domain unlabelled DAIC speech.
- No LoRA/adapter result found with confirmed speaker-independent evaluation and exact numbers. M3L figures were not extracted.
- No direct paper found on soft labels or label smoothing near the PHQ-8 = 10 cutoff.

## Which public depression speech corpora could be added for pretraining or extra training?

### Takeaway
Candidates include Androids (Italian, clinically diagnosed), EATD (Mandarin, SDS self-report, GitHub), MODMA (Mandarin, small) and CMDC. Pooling corpora is hazardous: corpus identity is nearly perfectly recoverable from speech embeddings, and differences in prevalence alone produce a corpus-only AUC of about 0.70. Transfer from DAIC-WOZ to other corpora is partial and depends on elicitation type. No paper found shows that adding a foreign corpus improves honest DAIC-WOZ/E-DAIC accuracy.

### Cited Findings
- **Androids:** 228 recordings from 118 native Italian speakers, 64 diagnosed with depression by psychiatrists. Labels are clinical diagnoses, not questionnaires. Recordings were made with laptop microphones in clinics. The read subset used elsewhere has 112 speakers (58 depressed). — [Interspeech 2023 tao23](https://www.isca-archive.org/interspeech_2023/tao23_interspeech.pdf)
- **EATD:** 162 Chinese students, interviewed online via an app, labelled with the self-report SDS. Public on GitHub. — [EATD paper, arXiv 2202.08210](https://arxiv.org/pdf/2202.08210)
- **MODMA:** EEG plus speech. Speech covers 24 clinically depressed patients and 29 controls, with interview, reading and picture-description tasks. — [MODMA, arXiv 2002.09283](https://arxiv.org/pdf/2002.09283)
- **Other datasets:** A 2025 survey lists DAIC-WOZ, E-DAIC, D-VLOG, EATD, MODMA and CMDC as the main depression datasets; D-VLOG is limited to 861 videos. — [Survey, arXiv 2506.18915](https://arxiv.org/pdf/2506.18915) (snippet, unverified)
- **Pooled-corpus confound study (2026):** pooled DAIC-WOZ (141, 29% positive), EATD (162, 19%), MPDD-Elderly (87, 13%), MPDD-Young (88, 11%) and Androids-Read (112, 52%), 590 speakers in total.
  - Corpus identity was recoverable at 97.5 to 99.8% accuracy from six representations.
  - A corpus-identity + prevalence oracle reached AUC 0.697 with no speech signal.
  - Pooled 5-fold CV beat leave-one-corpus-out by +0.154 (eGeMAPS) to +0.291 (XLS-R) AUC.
  - Equalising prevalence cut the corpus-only AUC to 0.501 and shrank the gap.
  — [arXiv 2610.31247](https://arxiv.org/html/2610.31247)
- In the same study, adding DAIC-WOZ raised the corpus-only AUC from 0.567 to 0.621, and adding Androids-Read raised it to 0.697. The authors recommend reporting corpus composition, the corpus-only AUC and corpus-held-out results. — [arXiv 2610.31247](https://arxiv.org/html/2610.31247)
- Public CMDC and ANDROIDS baselines that are near ceiling in-domain are "substantially weaker" in zero-shot transfer. — [arXiv 2605.23977](https://arxiv.org/abs/2605.23977)
- A DAIC-WOZ-trained model "transfers reasonably well" to Androids and "acceptably" to EATD (EMNLP 2025 Findings, interviewer bias). — [ACL Anthology 2025.findings-emnlp.650](https://aclanthology.org/2025.findings-emnlp.650.pdf) (snippet, unverified)
- A 2026 paper notes that prior work finds only partial, elicitation-dependent transfer for speech-based depression prediction. — [arXiv 2604.01533](https://arxiv.org/html/2604.01533) (snippet, unverified)
- A dataset-domain-invariant multitask method has been trained across Androids, EATD and a third corpus (Springer 2026). Results were not retrievable because of a paywall/redirect. — [Springer 10.1007/s10791-026-10517-y](https://link.springer.com/article/10.1007/s10791-026-10517-y)
- Deep Covariance Alignment Network, for English-to-Chinese cross-lingual transfer, is evaluated on Androids; the evaluation details are unclear. — [ScienceDirect S0165032725011814](https://www.sciencedirect.com/science/article/abs/pii/S0165032725011814) (snippet, unverified)

### Inferences
- **The project already pools DAIC-WOZ and E-DAIC, so the same confound applies in miniature.** The project's own result that domain mean centring and per-domain thresholds did not help suggests the confound is not driving its accuracy. Still, computing the corpus-only AUC from N and prevalence of the two sources is a cheap honesty check worth reporting.
- **Adding non-English, non-interview corpora with different labels is unlikely to raise DAIC/E-DAIC accuracy.** Androids uses diagnosis labels, EATD uses SDS self-report, and both are in other languages and task types. They are better suited to external-validation experiments (train on DAIC+E-DAIC, test zero-shot) than to extra training data.
- **If extra corpora are used anyway:** use them only as a pretraining or auxiliary-task signal, and evaluate strictly on held-out DAIC/E-DAIC speakers.

### Gaps
- No licence or access terms retrieved for Androids, CMDC, DEPAC, SH2, Sonde, Ellipsis/Kintsugi or Thymia. Several of these (Sonde, Ellipsis, Kintsugi, Thymia) appear to be industry data, not public. This is unconfirmed.
- No sizes or details found for DEPAC or AVEC 2013/2014 German within budget.
- No quantified gain on DAIC/E-DAIC from cross-corpus pretraining was found.

## Is PHQ-8 item-level multi-task learning beneficial?

### Takeaway
It helps modestly when items are trained with an ordinal loss. On E-DAIC, CCC rose from 0.602 to 0.662, but RMSE and MAE did not improve, and decomposing into items with plain MSE gave no gain. Multi-task benefits are "not always consistent", with negative transfer reported. No binary-accuracy gain from item-level MTL under speaker-disjoint CV was found.

### Cited Findings
- **QuestMF (3-seed means, E-DAIC):**
  - Total regression: CCC 0.602, RMSE 5.10, MAE 3.99.
  - Item-wise MSE: CCC 0.602, RMSE 5.36.
  - Item-wise OLL: CCC 0.640, RMSE 5.14, MAE 3.88.
  - Item-wise ImbOLL: CCC 0.662, RMSE 5.25, MAE 3.95.
  - The gain comes from ordinal training, not from item decomposition alone.
  — [arXiv 2503.20496](https://arxiv.org/html/2503.20496v1)
- **QuestMF per-item modality pattern (validation):** text is strongest for interest (Q1) and appetite (Q5); text+audio for depressed mood (Q2); text+video for sleep (Q3) and concentration (Q7). — [arXiv 2503.20496](https://arxiv.org/html/2503.20496v1)
- **U-Fair:** PHQ-8 item MTL improves over unitask, but not consistently, with negative transfer. Uncertainty-based task reweighting partly mitigates this. — [arXiv 2501.09687](https://arxiv.org/abs/2501.09687)
- **Training-time item supervision:** item scores used purely as training-time supervision preserve binary-label comparability. — [arXiv 2501.16106](https://arxiv.org/pdf/2501.16106) (snippet, unverified)
- **Joint regression + classification:** a vowel-based ensemble for speech depression identification trains regression and classification jointly, and argues that item-level targets aid interpretability. — [arXiv 2410.18298](https://arxiv.org/pdf/2410.18298) (snippet, unverified)
- **LLM item-score agents:** HiMA-MDD has LLM agents predict the eight item scores (0 to 3), sums them, and thresholds at 10. — [arXiv 2608.21868](https://arxiv.org/pdf/2608.21868) (snippet, unverified)

### Inferences
- **For this project, a cheap test is feasible in minutes on CPU:**
  1. Fit 8 per-item ordinal or ridge heads on frozen embeddings inside the nested CV.
  2. Use their predictions, or their sum, as extra stacker inputs alongside the direct binary/total-score heads.
- **Item labels:** DAIC-WOZ provides item-level PHQ-8 for train/dev participants. E-DAIC item labels may be missing for the test portion, per QuestMF. Check coverage across all 272 speakers before relying on them.
- **Expected gain is small,** around 0 to 2 accuracy points by analogy to the CCC result, and should be judged with paired bootstrap.

### Gaps
- No study found reports binary accuracy/AUC gains from item-level MTL under speaker-disjoint nested CV.
- No study found tests emotion-recognition auxiliary heads trained jointly, as opposed to emotion fine-tuning of the encoder, on DAIC/E-DAIC with honest CV.
