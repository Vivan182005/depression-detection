# Speech/acoustic-only depression detection on DAIC-WOZ and E-DAIC (SSL and classic features)

Scope note: DAIC-WOZ official AVEC 2016/2017 split is 107 train / 35 dev (dev has 12 depressed); the DAIC-WOZ test labels were not originally released, so almost all "DAIC-WOZ F1" numbers are on the **35-person dev set**. E-DAIC (AVEC 2019) official split is 163 train / 56 dev / 56 test, and the E-DAIC test set is AI-controlled-avatar interviews only. Verification legend: **[V]** = numbers read from the primary paper (or its HTML/PDF) in this session; **[T]** = number taken from a comparison table in another verified paper (secondary); **[M]** = from prior knowledge of the paper, not re-verified this session (the writer should treat as unverified or move to gaps).

## Master table (25+ papers)

| # | Paper (authors, year, venue) | Dataset / split | Model | Features | Reported metric(s) | Speaker-disjoint? | Src |
|---|---|---|---|---|---|---|---|
| 1 | Valstar et al., 2016, AVEC'16 (baseline) | DAIC-WOZ dev (n=35) | linear SVM / trees | COVAREP + formants | Audio dev F1 dep/non-dep ≈0.46/0.68 | Yes (official split) | [M] [AVEC 2016](https://arxiv.org/abs/1605.01600) |
| 2 | Ma, Yang, Chen, Huang, Wang, 2016, AVEC'16 "DepAudioNet" | DAIC-WOZ dev (35) | CNN+LSTM on segments, random sampling to balance | Mel filterbanks | F1 dep 0.52 / non-dep 0.70; macro-F1 0.61, recall 0.77, precision 0.68 (as tabulated by Li et al.); re-implemented macro-F1 0.6081 by Wang et al. | Yes | [T] [Li et al. 2025](https://arxiv.org/html/2510.08593); [T] [Wang et al. 2023](https://arxiv.org/html/2306.01861) |
| 3 | Ringeval et al., 2017, AVEC'17 (baseline) | DAIC-WOZ dev/test (PHQ regression) | random forest | COVAREP/eGeMAPS-style audio | Audio dev RMSE ≈6.74, test RMSE ≈7.78 | Yes | [M] [AVEC 2017](https://dl.acm.org/doi/10.1145/3133944.3133953) |
| 4 | Gong & Poellabauer, 2017, AVEC'17 | DAIC-WOZ dev (35) | topic-modelling, multimodal | audio+video+text (reference transcripts) | F1-max 0.70 (multimodal) | Yes | [T] [Wu et al. 2023](https://arxiv.org/html/2305.12263) |
| 5 | Al Hanai, Ghassemi, Glass, 2018, Interspeech | DAIC-WOZ dev (35) | LSTM over question/answer sequence | COVAREP-style audio + text | multimodal F1 0.77 (Wu Table 6); audio-only F1 ≈0.63 | Yes | [T] [Wu et al. 2023](https://arxiv.org/html/2305.12263); audio-only number [M] |
| 6 | Ringeval et al., 2019, AVEC'19 DDS (baseline) | E-DAIC test (n=56) | GRU-style baselines | MFCC, eGeMAPS, BoAW, Deep Spectrum (DS-DNet, DS-VGG) | Best baseline test RMSE 6.37 (CCC primary metric) | Yes | [V-secondary] [cyZhu98 E-DAIC list](https://github.com/cyZhu98/depression_papers); [AVEC 2019](https://arxiv.org/pdf/1907.11510) |
| 7 | Yin et al., 2019, AVEC'19 "Multi-modal hierarchical RNN" | E-DAIC test | hierarchical BiLSTM | audio+video+text | test CCC 0.442, RMSE 5.50 (multimodal) | Yes | [T] [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 8 | Fan et al., 2019, AVEC'19 "Multi-scale temporal dilated CNNs" | E-DAIC test | dilated CNN | multimodal | test CCC 0.430, RMSE 5.91 | Yes | [T] [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 9 | Ray et al., 2019, AVEC'19 "Multi-level attention network" | E-DAIC **dev only** | attention fusion | text, audio, video | dev-only (flagged by list) | Yes | [T] [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 10 | Rodrigues Makiuchi et al., 2019, AVEC'19 "BERT-CNN + gated CNN" | E-DAIC | gated CNN on speech + BERT text | multimodal | numbers not captured | Yes | [T] [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 11 | Huang, Epps, Joachim, 2020, ICASSP "FVTC-CNN" | DAIC-WOZ dev (35) | dilated CNN | formant vocal-tract coordination | macro-F1 0.64 (F1 ND 0.46 / D 0.82) | Yes | [T] [Wang et al. 2023](https://arxiv.org/html/2306.01861) |
| 12 | Speech SimCLR / CPC baselines (Ravi et al. comparison) | DAIC-WOZ dev (35) | contrastive SSL on Mel | Mel-spectrogram | macro-F1 0.658 (SimCLR), 0.676 (CPC) | Yes | [T] [Wang et al. 2023](https://arxiv.org/html/2306.01861) |
| 13 | Ravi, Wang, Flint, Alwan, 2022, Interspeech "Unsupervised instance discriminative learning" | DAIC-WOZ dev (35) | CNN-LSTM, 5-model average | Mel-spectrogram | F1-avg 0.69 (as cited by Wu) | Yes | [T] [Wu et al. 2023](https://arxiv.org/html/2305.12263); [arXiv 2206.13016](https://arxiv.org/pdf/2206.13016) |
| 14 | Dumpala et al., 2022/2023 (Sci. Rep. 2023) "Depression overlaps with speaker identity" | DAIC-WOZ dev (35) | CNN-LSTM / MK-CNN, segment-level training, **majority vote** per file (context 16 segments) | speaker embeddings (x-vector/ECAPA) + openSMILE | macro-F1 0.685 (ND 0.86 / D 0.51) | Yes | [T] [Wang et al. 2023](https://arxiv.org/html/2306.01861); [Sci Rep 2023](https://www.nature.com/articles/s41598-023-35184-7) |
| 15 | Chen et al., 2022, Interspeech "SpeechFormer" | DAIC-WOZ dev (35) | hierarchical speech Transformer | wav2vec / HuBERT features (sources disagree) | macro-F1 0.694 (dev); under 5-fold subject CV macro-F1 0.533, AUC 0.557 (re-run by Li et al.) | dev: Yes; CV: Yes | [T] [Wang et al. 2023](https://arxiv.org/html/2306.01861); [T] [Li et al. 2025](https://arxiv.org/html/2510.08593) |
| 16 | Feng & Chaspari, 2022 "Vowel-based approach with augmentation" | DAIC-WOZ dev (35) | CNN on vowel segments | Mel-spectrogram | macro-F1 0.70 (ND 0.84 / D 0.56) | Yes | [T] [Wang et al. 2023](https://arxiv.org/html/2306.01861) |
| 17 | Vlad-GRU, 2022 | DAIC-WOZ dev (35) | NetVLAD + GRU | Mel filterbanks | macro-F1 0.77 (recall 1.00, prec 0.63) | Yes | [T] [Li et al. 2025](https://arxiv.org/html/2510.08593) |
| 18 | Shen, Yang, Lin, 2022, ICASSP (EATD corpus paper) | DAIC-WOZ dev (35) | GRU/BiLSTM, multimodal | audio + **reference transcripts** | F1-max 0.85 (multimodal) | Yes | [T] [Wu et al. 2023](https://arxiv.org/html/2305.12263) |
| 19 | Wu, Wu, Zhang, Woodland, 2022, ICASSP "Climate and weather" (emotion → depression) | DAIC-WOZ dev (35) | emotion-pretrained encoders + Transformer | speech + reference transcripts | F1-max 0.87 (multimodal) | Yes | [T] [Wu et al. 2023](https://arxiv.org/html/2305.12263) |
| 20 | **Wu, Zhang, Woodland, 2023, ICASSP "Self-supervised representations in speech-based depression detection"** | DAIC-WOZ train 107 / dev 35 | frozen SSL → mean pool per utterance → 2-layer Transformer over utterance sequence; 20 seeds; sub-dialogue augmentation (M+=500) | wav2vec2-base, HuBERT-base, WavLM-Base+, ASR-FT and emotion-FT (MSP-Podcast) variants, per block | Best single: WavLM_AER block 10 F1-avg 0.720 (max 0.769); WavLM_PT block 8 0.700; HuBERT_PT block 10 0.667 (max 0.762); W2V2_PT block 8 0.627; 3-SSL ensemble F1-avg 0.800 (max 0.857); +ASR-text ensemble 0.829 (max 0.886) | Yes (dev) | [V] [arXiv 2305.12263](https://arxiv.org/html/2305.12263) |
| 21 | MSCDR, 2023 | DAIC-WOZ dev (35) | multi-scale CNN | LPC-MFCC | macro-F1 0.75 | Yes | [T] [Li et al. 2025](https://arxiv.org/html/2510.08593) |
| 22 | DALF, 2023 | DAIC-WOZ dev (35) | learnable front-end | raw speech | macro-F1 0.78 | Yes | [T] [Li et al. 2025](https://arxiv.org/html/2510.08593) |
| 23 | **Wang, Ravi, Alwan, 2023, Interspeech "Non-uniform speaker disentanglement (NUSD)"** | DAIC-WOZ 107 / 35 dev; **participant audio only** | ECAPA-TDNN on raw audio, adversarial speaker removal; 3.84 s segments; 5-model average | raw waveform (beat Mel and ComParE16 in pilots) | macro-F1 0.7349 (ND 0.833 / D 0.636), speaker-ID acc 4.68%; no-disentanglement baseline 0.633; DepAudioNet+NUSD 0.709 | Yes | [V] [arXiv 2306.01861](https://arxiv.org/html/2306.01861) |
| 24 | Ravi et al., 2024, Computer Speech & Language "Enhancing accuracy and privacy … speaker disentanglement" | DAIC-WOZ 107 / 35 dev | CNN-LSTM, segment scores rounded | wav2vec2 features / Mel | macro-F1 ≈0.68–0.80 depending on loss and λ (exact best not verified) | Yes | [T/snippet] [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0885230823001249) |
| 25 | Yang et al., 2023 "Multimodal multi-order factor fusion" | E-DAIC (split unclear) | factor fusion | audio branch | audio-only CCC 0.434, RMSE 5.44 | unclear | [snippet] [arXiv 2301.00254](https://arxiv.org/pdf/2301.00254) |
| 26 | "Improving speech depression detection using transfer learning with wav2vec 2.0 in low-resource environments", 2024, Sci. Rep. | DAIC-WOZ | **fine-tuned** wav2vec2 (base/large) + self-attention | raw audio | F1 79.0% (full FT of large + attention); 74.72% without attention; full-layer FT best | split/speaker-disjointness not verified | [snippet] [Sci Rep 2024](https://www.nature.com/articles/s41598-024-60278-1) |
| 27 | Huang et al., 2024, Sci. Rep. "Depression recognition using voice-based pre-training model" | DAIC-WOZ | fine-tuned wav2vec2 | raw audio | accuracy 0.9649 (classification "RMSE" 0.1875) — almost certainly segment-level / speaker-overlap; cited as an example of leakage | likely **No** | [snippet] [Sci Rep 2024](https://www.nature.com/articles/s41598-024-63556-0); [Sun et al. 2026](https://arxiv.org/html/2604.14354v1) |
| 28 | SFTN, 2024 | DAIC-WOZ dev (35) | spatial-frequency Transformer | Mel filterbanks | macro-F1 0.76 | Yes | [T] [Li et al. 2025](https://arxiv.org/html/2510.08593) |
| 29 | Maji et al., 2024, Interspeech "Layer-wise speech representations in SSL … depression" | DAIC-WOZ + cross-lingual | layer-wise probing, pooling variants | HuBERT vs WavLM layers | WavLM features generalise better than HuBERT (numbers not captured) | — | [snippet] [ISCA 2024](https://www.isca-archive.org/interspeech_2024/maji24_interspeech.pdf) |
| 30 | DMPF, 2025 | DAIC-WOZ dev (35) | — | raw speech | macro-F1 0.80 | Yes | [T] [Li et al. 2025](https://arxiv.org/html/2510.08593) |
| 31 | **Gómez-Zaragozá, Marín-Morales, Alcañiz, Soleymani, 2025, Interspeech** | DAIC-WOZ: train 107 → tested on 35 dev; E-DAIC: train 219 → **official test 56**; DAIC+ (459 spk) test 92 | frozen HuBERT (last block or block 9), mean-pooled per turn; GRU over windows of W=20 turns (hop 10); 5-fold SI CV for HP; majority vote across folds and per speaker; 100 bootstraps | HuBERT-base/large and **superb-ER** fine-tuned variants (the same hubert-large-superb-er you use) | Speech-only test macro-F1: DAIC-WOZ best HuBERT-base-L9 0.553±0.066 (ER-L9 0.487, large-ER 0.395); E-DAIC best **HuBERT-large-er 0.670±0.058** (base-L9 0.611); DAIC+ best HuBERT-base-er-L9 0.584. Multimodal best: E-DAIC 0.840, DAIC-WOZ 0.643, DAIC+ 0.753. Female speakers significantly worse (gap 0.05–0.1 on E-DAIC/DAIC+) | Yes | [V] [ISCA 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf) |
| 32 | **Li, Chng, Guan, 2025/26, arXiv "HAREN-CTC"** | DAIC-WOZ dev (35) **and** merged 189-subject stratified 5-fold CV × 5 seeds | frozen WavLM-Large, all 24 layers softly grouped (shallow/deep) + cross-attention, CTC aux loss on HuBERT k-means pseudo-labels; random 10 s crops; subject = confidence-weighted mean of 20 longest utterances | WavLM-Large | Dev: macro-F1 0.81. **5-fold CV: macro-F1 0.576, UAR 0.592, AUC 0.599**; DepAudioNet CV F1 0.533 / AUC 0.566; SpeechFormer CV F1 0.533 / AUC 0.557 | Yes | [V] [arXiv 2510.08593](https://arxiv.org/html/2510.08593) |
| 33 | **Sun et al. (JHU / U. Michigan; Mower Provost, Sisman), 2026, arXiv "Who is speaking or who is depressed?"** | DAIC-WOZ, 189 subj; segments of 5 consecutive participant utterances; test 714 segments / 38 speakers; Set A speaker-independent vs Set B speaker-overlapped | wav2vec2 linear probe, SLS (weighted layer sum), XLS-R+eGeMAPS; frozen vs fine-tuned; ± DANN | wav2vec2 / XLS-R / eGeMAPS | Segment-level, speaker-independent macro-F1: frozen linear 0.528 → fine-tuned 0.562; frozen SLS 0.637 → **fine-tuned SLS 0.738 (acc 70.3%)**; frozen XLSR-eGeMAPS 0.710. Same models with speaker overlap: 0.76–0.98 (fine-tuned linear acc 97.65% vs 58.74% speaker-independent) | A: Yes / B: No | [V] [arXiv 2604.14354](https://arxiv.org/html/2604.14354v1) |
| 34 | **Ishikawa & Duke, 2026, arXiv "Multi-probe audit of clinical-interview depression benchmarks"** | E-DAIC LOSO (275 subj) + official split (163/56/56) | 96 configs (modality bundles × pooling × learners) | audio, vision, text, LLM scores | LOSO E-DAIC: text+LLM macro-F1 0.723 / AUROC 0.768; LLM-only AUROC 0.825; text-only AUROC 0.647. Official split: best-CV config ranked 20th on test, test-best ranked 41st by CV, top-3 overlap 0, Spearman 0.70; test winner rank-1 in only 32.3% of bootstraps. Audio scores flat on symptom-dense vs symptom-light slices (shift −0.004 vs text +0.422) | Yes (LOSO) | [V] [arXiv 2605.23977](https://arxiv.org/pdf/2605.23977) |

Additional papers known but not verified this session (move to gaps if needed): Williamson et al. 2016 AVEC (vocal-tract coordination, AVEC 2016 winner) [M]; Rejaibi et al. 2022 "MFCC-based RNN for depression" Biomed. Signal Proc. (audio accuracy ≈76%) [M]; Niu et al. 2019 Interspeech lp-norm pooling (PHQ regression) [M]; Egas-López et al. 2022 ICASSP x-vectors [M]; Bailey & Plumbley 2021 EUSIPCO "Gender bias in depression detection using audio features" [M]; Toto et al. 2021 CIKM "AudiBERT" [M]; Seneviratne & Espy-Wilson 2021 articulatory coordination [M]; Burdisso et al. 2024 "On the validity of using therapist's prompts in DAIC-WOZ" (text interviewer-bias) [M].

## Q1. What are the best honestly-evaluated audio-only results on DAIC-WOZ and E-DAIC, and what models produced them?

### Takeaway
On the 35-person DAIC-WOZ dev set, audio-only macro-F1 has climbed from ~0.61 (DepAudioNet, 2016) to 0.70–0.81 (WavLM/HuBERT intermediate layers, raw-waveform ECAPA with speaker disentanglement, HAREN-CTC). Under subject-level CV or a real held-out test, audio-only results fall to macro-F1 ≈0.55–0.67 and AUC ≈0.60. The best figures under stricter protocols are E-DAIC official test F1 0.670 (frozen HuBERT-large-superb-ER + GRU) and speaker-independent segment F1 0.738 (fine-tuned wav2vec2 with weighted layer sum). Seen against this, the project's AUC ~0.76 under repeated nested CV on N=272 is at or above the honest literature range.

### Cited Findings
- Wu et al. (ICASSP 2023): the best single frozen SSL model on the DAIC-WOZ dev set (n=35, 12 depressed) is WavLM emotion-FT block 10, with F1-avg 0.720 over 20 seeds. WavLM-PT block 8 reaches 0.700, and a three-SSL ensemble reaches 0.800 (max 0.857). These are speech-only. Adding ASR-hypothesis text raises the ensemble to 0.829/0.886 — [Wu et al. 2023](https://arxiv.org/html/2305.12263)
- Wu et al. report F1 std of 0.02–0.06 across seeds, and F1-max exceeds F1-avg by up to ~0.1. Many prior papers report only the best seed (F1-max) — [Wu et al. 2023](https://arxiv.org/html/2305.12263)
- Wang, Ravi, Alwan (Interspeech 2023): raw-waveform ECAPA-TDNN with non-uniform speaker disentanglement reaches macro-F1 0.7349 on the DAIC-WOZ dev set using participant audio only. It beats SpeechFormer (0.694), vowel-based CNN (0.70), and DepAudioNet (0.608) — [Wang et al. 2023](https://arxiv.org/html/2306.01861)
- Li, Chng, Guan (HAREN-CTC): dev-set macro-F1 is 0.81, but stratified subject-level 5-fold CV (×5 seeds) on all 189 DAIC-WOZ subjects gives only macro-F1 0.576 and AUC 0.599. Re-running DepAudioNet and SpeechFormer under the same CV gives AUC 0.566 and 0.557 — [Li et al. 2025](https://arxiv.org/html/2510.08593)
- Gómez-Zaragozá et al. (Interspeech 2025) give speech-only, speaker-level, bootstrapped results. The DAIC-WOZ dev set gets macro-F1 0.553 (HuBERT-base L9). The **E-DAIC official test (n=56)** gets **0.670** (HuBERT-large-superb-ER, last layer). The larger DAIC+ test (n=92) gets 0.584. The authors attribute DAIC-WOZ's lower scores and higher variance to its 35-person evaluation set — [Gómez-Zaragozá et al. 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf)
- Sun et al. (2026): under strict speaker independence, segment-level DAIC-WOZ accuracy across wav2vec2 variants is 54–70%; the best is fine-tuned wav2vec2-SLS at macro-F1 0.738 / acc 70.3%. With speaker overlap, the same models reach 62–98% — [Sun et al. 2026](https://arxiv.org/html/2604.14354v1)
- Papers reporting more than 90% accuracy on DAIC-WOZ audio are attributed to speaker leakage. Examples are Huang et al. at 96.5% with fine-tuned wav2vec2 and Rezaee et al. at 93.9% — [Sun et al. 2026](https://arxiv.org/html/2604.14354v1); [Huang et al. 2024](https://www.nature.com/articles/s41598-024-63556-0)
- A review of 66 studies found that at least 9.1% had subject leakage and 24.2% could not be verified — [Ishikawa & Duke 2026](https://arxiv.org/pdf/2605.23977)
- E-DAIC: the AVEC 2019 baseline test RMSE is 6.37. Multimodal challenge systems reach test RMSE 5.50 (CCC 0.442) and 5.91 (CCC 0.430) — [cyZhu98 E-DAIC list](https://github.com/cyZhu98/depression_papers). One 2023 paper reports an audio-only branch at CCC 0.434 / RMSE 5.44, but the split is unclear — [arXiv 2301.00254](https://arxiv.org/pdf/2301.00254)
- On the E-DAIC official split (56 test subjects), the ranking from development-side CV and the test ranking correlate only at Spearman 0.70. The test winner is rank-1 in only 32.3% of subject bootstraps, and its 95% rank interval spans positions 1–19 — [Ishikawa & Duke 2026](https://arxiv.org/pdf/2605.23977)

### Inferences
- Dev-set (n=35) numbers inflate audio-only results by about 0.15–0.25 macro-F1 compared with CV on the same data (HAREN-CTC: 0.81 dev vs 0.576 CV). Dev-only results should not be compared with the project's repeated nested CV.
- The most comparable external reference points for the project are: HAREN-CTC CV AUC 0.60 (DAIC-WOZ only); Gómez-Zaragozá E-DAIC test F1 0.67 and DAIC+ test F1 0.58 with the *same* hubert-large-superb-ER encoder; and Sun et al. speaker-independent segment accuracy 54–70%. Voice-only AUC 0.76 on N=272 is competitive or better. Further gains are more likely to come from evaluation-variance reduction or more data (DAIC+ style) than from architecture.

### Gaps
- No paper found that reports audio-only **AUC** on the E-DAIC official test with SSL features. Most report macro-F1 or CCC.
- The audio-only rows of the AVEC 2019 baseline table (DS-VGG, BoAW etc.) could not be extracted.
- The audio-only numbers for the AVEC 2016/2017 baselines and for Al Hanai 2018 are from memory [M].

## Q2. Does fine-tuning SSL models (vs frozen embeddings) help on such small data? By how much?

### Takeaway
The evidence is mixed. Fine-tuning gives a modest gain under speaker-independent evaluation, about +0.03 to +0.10 macro-F1 at segment level, and a large *spurious* gain whenever speakers overlap, because fine-tuning amplifies speaker identity. Using a model fine-tuned for emotion recognition, as a frozen feature extractor, gives small and inconsistent gains: +0.02 F1 for WavLM on the DAIC-WOZ dev set, +0.06 for HuBERT-large on the E-DAIC test, and a drop on the DAIC-WOZ dev set in another study.

### Cited Findings
- Speaker-independent macro-F1 (segment level) rises from frozen to fine-tuned in two of three wav2vec2 variants: linear 0.528→0.562 and SLS 0.637→0.738. The third falls: XLSR-eGeMAPS 0.710→0.544. With speaker overlap, fine-tuned linear jumps to 0.976 F1 (frozen 0.765), and speaker-ID accuracy stays at 91–96% — [Sun et al. 2026](https://arxiv.org/html/2604.14354v1)
- In the same study, adversarial speaker removal (DANN) helps fine-tuned linear (0.562→0.602) but hurts SLS (0.738→0.659) — [Sun et al. 2026](https://arxiv.org/html/2604.14354v1)
- A Sci. Rep. 2024 study reports DAIC-WOZ F1 79% when the large wav2vec2 is fully fine-tuned with a self-attention head, versus 74.72% without attention; full-layer fine-tuning was best in its ablation. It was not verified whether the split is speaker-disjoint — [Sci Rep 2024](https://www.nature.com/articles/s41598-024-60278-1)
- Emotion-fine-tuned SSL (frozen afterwards): WavLM_AER block 10 gets F1-avg 0.720 vs 0.700 for WavLM_PT block 8, and W2V2_AER gets 0.640 vs 0.627 for W2V2_PT. ASR fine-tuning helps shallow blocks but not the best block (W2V2_ASR best 0.639 at block 6) — [Wu et al. 2023](https://arxiv.org/html/2305.12263)
- HuBERT superb-ER vs plain HuBERT, speaker-level test macro-F1:
  - E-DAIC: large-er 0.670 vs large 0.570; base-er-L9 0.573 vs base-L9 0.611.
  - DAIC-WOZ: base-L9 0.553 vs base-er-L9 0.487.
  - The authors note DAIC-WOZ is the only dataset where the non-ER speech model won — [Gómez-Zaragozá et al. 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf)
- The strongest recent dev-set results with SSL (HAREN-CTC, 0.81 dev) keep WavLM-Large **frozen** and learn only layer weighting and aggregation — [Li et al. 2025](https://arxiv.org/html/2510.08593)

### Inferences
- With about 270 speakers, end-to-end fine-tuning risks learning speaker identity. The best honest gain reported (+0.10 F1, SLS) is at segment level on a single 38-speaker test set and has not been shown at session level under repeated CV. A low-cost alternative to full fine-tuning is a learnable weighted layer sum (SLS) over frozen layers plus a small head trained inside nested CV.
- The project already uses hubert-large-superb-ER, so it already captures most of the "emotion-FT" gain the literature reports.

### Gaps
- No study was found that compares frozen vs fine-tuned SSL at **session level** with repeated speaker-disjoint CV on DAIC-WOZ + E-DAIC combined.
- Details of the Sci. Rep. 2024 wav2vec2 transfer study (authors, speaker split) could not be fetched; the PMC page was bot-blocked.

## Q3. Which pooling/aggregation works best (segment voting, attention, LSTM over segments)? Layer selection?

### Takeaway
Intermediate SSL layers are consistently best: blocks 8–10 of 12 in base models, and block 9 beats the final block in HuBERT-base. Utterance-level mean pooling followed by a sequence model (Transformer or GRU over utterances/turns) underlies the strongest SSL results. Segment-level training with majority vote or probability averaging is the standard for CNN/raw-audio models, and there is no head-to-head evidence that attention beats voting under honest evaluation.

### Cited Findings
- Layer profiles by block (F1-avg, DAIC-WOZ dev):
  - W2V2_PT: 0.531 (block 2), 0.597 (6), 0.627 (8), 0.519 (12) — peaks at block 8, then drops sharply.
  - HuBERT_PT: peaks at block 10 (0.667).
  - WavLM_PT: peaks at block 8 (0.700).
  - Shallow blocks carry acoustic information with limited depression signal — [Wu et al. 2023](https://arxiv.org/html/2305.12263)
- HuBERT-base block 9 outperformed the final block for speech in both DAIC and Spanish data — [Gómez-Zaragozá et al. 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf)
- Wu et al. pipeline: average temporal pooling to one vector per utterance, then two 128-d Transformer encoder layers over the dialogue's sequence of utterance vectors (0.3M params). Sub-dialogue augmentation was essential: F1-avg 0.451 at M+=100 vs 0.679 at M+=1000 — [Wu et al. 2023](https://arxiv.org/html/2305.12263)
- Gómez-Zaragozá pipeline: per-turn mean-pooled HuBERT, GRU over windows of 20 turns (hop 10), majority vote over windows (ties → depressed) and over 5 CV-fold models — [Gómez-Zaragozá et al. 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf)
- HAREN-CTC: soft grouping of all 24 WavLM-Large layers into shallow/deep subspaces with cross-attention; subject score = confidence-weighted average of the 20 longest utterances; random 10 s crops during training — [Li et al. 2025](https://arxiv.org/html/2510.08593)
- Wang et al.: 3.84 s raw-audio segments, balanced sampling, 5-model averaging; speaker-level F1 — [Wang et al. 2023](https://arxiv.org/html/2306.01861)
- Dumpala et al.: segment-level LSTM/MK-CNN with majority voting over segment decisions per file (16-segment context for DAIC-WOZ); LSTM matched or beat MK-CNN — [Sci Rep 2023 (via search snippet)](https://www.nature.com/articles/s41598-023-35184-7)
- A dilated CNN-LSTM paper argues rule-based voting is hard to tune and proposes an LSTM over segment-level predictions for session-level classification — [arXiv 2104.04195](https://arxiv.org/pdf/2104.04195)
- Learnable weighted layer sum (SLS) beat a single-layer linear probe under speaker independence: frozen 0.637 vs 0.528; fine-tuned 0.738 vs 0.562 — [Sun et al. 2026](https://arxiv.org/html/2604.14354v1)
- Maji et al. (Interspeech 2024) compare pooling methods layer by layer and find WavLM features generalise better than HuBERT — [ISCA 2024](https://www.isca-archive.org/interspeech_2024/maji24_interspeech.pdf)

### Inferences
- The project's choices (wav2vec2-base L6, HuBERT-large L19/20 of 24, mean pooling) agree with the "intermediate layer" finding. Two untested, literature-backed options remain:
  - A WavLM-Base+ block-8 / WavLM-Large mid-layer feature, the best single SSL in Wu et al.
  - A learnable or CV-selected weighted layer sum.
- The project's finding that stats pooling did not help agrees with the field: gains come from modelling across utterances (sequence over turns), not from richer within-utterance statistics. The most direct untried analogue would be utterance-level embeddings followed by a small GRU/Transformer, or a robust aggregate such as the mean of per-utterance ridge scores. With N=272 this needs heavy augmentation (sub-dialogue sampling), and its benefit under nested CV is unproven.

### Gaps
- No head-to-head study was found of attention pooling vs majority vote vs mean of segment probabilities under repeated speaker-disjoint CV.
- No systematic chunk-length study for SSL features on DAIC was found. Reported lengths are 3.84 s (Wang), 10 s crops (HAREN), and whole utterances/turns (Wu, Gómez-Zaragozá).
- Gender-dependent models: no DAIC paper reporting gender-split model gains was verified. Gómez-Zaragozá find female speakers score 0.05–0.1 F1 lower on E-DAIC/DAIC+ — [Gómez-Zaragozá et al. 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf). Bailey & Plumbley 2021 on gender bias is [M].

## Q4. Participant-only speech and response-level segmentation

### Takeaway
Almost all modern audio work uses participant-only speech: E-DAIC releases only the participant side, and DAIC-WOZ papers cut participant turns using transcript timestamps. Response- or turn-level segmentation followed by utterance pooling is the basis of the best SSL systems. Audio appears less sensitive than text to interview content, which makes it less exposed to the interviewer-prompt shortcut that inflates text results.

### Cited Findings
- Wang et al. used only patient audio from DAIC-WOZ — [Wang et al. 2023](https://arxiv.org/html/2306.01861)
- Gómez-Zaragozá et al. segmented sessions into turn-level clips using transcript timestamps. They corrected known DAIC timestamp errors and added 0.5 s to each end time. VAD trimmed trailing silence — [Gómez-Zaragozá et al. 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf)
- Sun et al. used participant speech grouped into segments of 5 consecutive utterances (6,545 segments) — [Sun et al. 2026](https://arxiv.org/html/2604.14354v1)
- E-DAIC provides only participant-side transcripts. Ishikawa & Duke had to re-transcribe the full dialogue with faster-whisper large-v3 + pyannote diarisation to get question context — [Ishikawa & Duke 2026](https://arxiv.org/pdf/2605.23977)
- On paired same-participant slices of E-DAIC, symptom-dense minus symptom-light probability shift is +0.422 for text and −0.004 for audio. The text-minus-audio gap is positive in 5/5 seeds (p=0.0002). Text performance is tied to explicit symptom content, while audio is content-insensitive — [Ishikawa & Duke 2026](https://arxiv.org/pdf/2605.23977)
- In zero-shot external transfer of a reproduced ANDROIDS pipeline, audio was the most portable branch (CMDC audio macro-F1 0.647 / AUROC 0.768 vs text 0.400 / 0.288) — [Ishikawa & Duke 2026](https://arxiv.org/pdf/2605.23977)
- Using interviewer utterances in supervised text classifiers is listed as a leakage/shortcut risk in DAIC — [Ishikawa & Duke 2026](https://arxiv.org/pdf/2605.23977)

### Inferences
- Because audio is content-flat, voice-only AUC ~0.76 is plausibly a "cleaner" signal than equivalent text AUC. That supports reporting voice-only as a headline result.
- Response-level (per-answer) pooling, as the project already does after VAD, is standard. Question-aligned modelling (pooling per question type) is the main unexplored segmentation idea.

### Gaps
- No paper was found that directly ablates participant-only vs mixed-speaker audio on DAIC-WOZ with SSL features.
- No verified number was found for question-specific (answer-to-specific-prompt) audio modelling on DAIC-WOZ/E-DAIC.
