# Multimodal (audio + text + visual) depression detection on DAIC-WOZ / E-DAIC: fusion, ablations, challenge winners

Scope note: about 20 tool calls. Numbers come from primary papers where I could open them, otherwise from comparison tables in later papers (marked "as cited by"). Several arXiv PDFs could not be text-extracted in this environment, so some per-modality numbers are missing and are listed under Gaps. Any number I remembered but could not verify is in Gaps, not in Cited Findings.

Project context used for the inferences: binary PHQ-8>=10; frozen SSL audio + emotion-RoBERTa on Whisper text; logistic stacker on CDF-calibrated scores; N=272 (DAIC-WOZ + E-DAIC Avatar); speaker-disjoint repeated nested CV; AUC 0.764, accuracy 74.3% (majority 68.8%).

## Q1. Dense literature table and per-modality ablations (audio vs text vs visual vs combinations)

### Takeaway
Text is consistently the strongest single modality and visual is the weakest on E-DAIC. In the one clean, seed-averaged, full 7-way ablation found (QuestMF, E-DAIC 2025), video alone scores CCC -0.05 to -0.07, adding video to text gives +0.02 to +0.04 CCC, and adding it to text+audio gives about +0.02 CCC. Those gains are about the size of the seed standard deviation (about 0.02). On DAIC-WOZ, papers report larger gains from visual features (for example F1 0.720 to 0.789 when adding video to A+T in a 2025 multimodal-LLM paper), but only on the small official dev/test splits (35 and 47 speakers), so they are noisy.

### Cited Findings

**Dense table.** "Test" means the official challenge test set. "Dev" means the development set. Regression metrics are on PHQ-8 0-24. Classification uses PHQ-8>=10.

| # | Paper (year, venue) | Data | Modalities | Fusion type | Split | Reported metrics | Modality contribution / ablation | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | Valstar et al., AVEC 2016 overview (ACM AVEC'16) | DAIC-WOZ | A, V (+ transcripts released) | Baseline linear SVM (SGD), scikit-learn | official dev/test | Metric = F1 averaged over depressed/not-depressed | Baseline only (per-modality numbers not extracted) | [arXiv 1605.01600](https://arxiv.org/pdf/1605.01600) |
| 2 | Williamson et al. 2016 (AVEC'16) "Detecting depression using vocal, facial and semantic communication cues" | DAIC-WOZ | A, V (facial AUs), T | Gaussian Staircase model, decision/score fusion | dev/test | Best hand-crafted-feature model F1 0.57 (as cited by a later paper) | Uses all three; per-modality numbers not verified | [ResearchGate](https://www.researchgate.net/publication/309127631_Detecting_Depression_using_Vocal_Facial_and_Semantic_Communication_Cues); figure as cited in [Mental-Perceiver arXiv 2408.12088](https://arxiv.org/html/2408.12088) |
| 3 | Yang et al. 2016 "Decision tree based depression classification from audio, video and language information" (AVEC'16) | DAIC-WOZ | A, V, T | Decision tree over multimodal features (gender-aware design in the original, not verified) | test | DCC test average F1 0.724 vs a quoted baseline of 0.700 | n/a in snippet | [ResearchGate](https://www.researchgate.net/publication/309380604_Decision_Tree_Based_Depression_Classification_from_Audio_Video_and_Language_Information) |
| 4 | Yang, Sahli, Jiang et al. 2017 "Hybrid depression classification and estimation from audio, video and text" (AVEC'17) | DAIC-WOZ | A, V, T | Hybrid: classification first, then regression; multimodal fusion | dev & test | Dev RMSE 3.088 / MAE 2.477; Test RMSE 5.400 / MAE 4.359 (AVEC'17 baseline test 7.05 / 5.66) | Per-modality not extracted | [ResearchGate](https://www.researchgate.net/publication/320545362_Hybrid_Depression_Classification_and_Estimation_from_Audio_Video_and_Text_Information); baseline from [PMC10308283](https://pmc.ncbi.nlm.nih.gov/articles/PMC10308283/) |
| 5 | Stepanov et al. 2017/2018 "Depression severity estimation from multiple modalities" (AVEC'17; arXiv) | DAIC-WOZ | A, V, T (speech turn features) | Feature- and decision-level comparison | test | Best on test: Turn Features from transcripts, MAE 4.11 / RMSE 4.94; authors say this won AVEC'17 DSC | Text/turn features beat audio and video | [arXiv 1711.06095](https://arxiv.org/abs/1711.06095) |
| 6 | Gong & Poellabauer 2017 "Topic modeling based multi-modal depression detection" (AVEC'17) | DAIC-WOZ | A, V, T | Topic-level (question-wise) feature modelling plus feature selection | test | ResearchGate tags it "Depression Challenge Winner" (numbers not verified) | Topic modelling of interview questions | [ResearchGate](https://www.researchgate.net/publication/320544890_Topic_Modeling_Based_Multi-modal_Depression_Detection) |
| 7 | Al Hanai, Ghassemi, Glass 2018 (Interspeech) | DAIC-WOZ (142 subjects) | A, T | Sequence LSTM fusion | dev | F1: audio 0.63, text 0.67, A+T 0.77 (as cited by a 2026 paper) | Fusion gives +0.10 F1 over the best single modality | [ISCA](https://www.isca-archive.org/interspeech_2018/alhanai18_interspeech.html); [PDF](https://sls.csail.mit.edu/publications/2018/Alhanai_Interspeech-2018.pdf); numbers as cited in [MFDCL, Algorithms 2026](https://doi.org/10.3390/a19060452) |
| 8 | Qureshi, Hasanuzzaman, Saha, Dias 2019 (arXiv) "Verbal and non-verbal signals of depression" | DAIC-WOZ | A, T, V | Attention-based fusion | dev | Says -7.17% RMSE and -8.08% MAE vs prior state of the art | Ablation in full paper (not extracted) | [arXiv 1904.07656](https://arxiv.org/abs/1904.07656) |
| 9 | Ringeval et al. 2019, AVEC 2019 overview (ACM AVEC'19) | E-DAIC (275 sessions; 163/56/56) | A, V (released features), T | Baseline per-feature-set regressors plus fusion | test | Baseline test RMSE 6.37, CCC 0.111 | — | [arXiv 1907.11510](https://arxiv.org/pdf/1907.11510); values as cited in [Tank et al. 2024](https://arxiv.org/html/2407.06125v2) and [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 10 | Ray et al. 2019 "Multi-level attention network using text, audio and video" (AVEC'19) | E-DAIC | T, A, V | Multi-level attention (intra- and inter-modality), Bi-LSTM | conflicting (dev-only per one list) | CCC 0.67, RMSE 4.37 (Tank) or 4.73 (Mandal), MAE 4.02; abstract says 17.52% RMSE gain over baseline | The 0.67 CCC is attributed to the **text** modality; video and audio were weaker | [arXiv 1909.01417](https://arxiv.org/abs/1909.01417); [Tank 2024](https://arxiv.org/html/2407.06125v2); [Mandal 2025](https://arxiv.org/html/2503.20496) |
| 11 | Yin et al. 2019 "Multi-modal hierarchical RNN" (AVEC'19) | E-DAIC | A, V, T | Hierarchical RNN fusion | val & test | Val RMSE 4.94 / CCC 0.402; Test RMSE 5.50 / CCC 0.442 | — | [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 12 | Fan et al. 2019 "Multi-scale temporal dilated CNNs" (AVEC'19) | E-DAIC | multimodal | Dilated CNN per modality plus fusion | test | CCC 0.430, RMSE 5.91 | — | [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 13 | Makiuchi et al. 2019 "BERT-CNN and gated CNN" (AVEC'19) | E-DAIC | T (BERT-CNN) + A (gated CNN) | Late/representation fusion | test | Test CCC 0.403 (+0.283 over baseline) | Audio+text only (no visual) | [AVEC'19 TOC](http://www.sigmm.org/opentoc/AVEC2019-TOC) |
| 14 | Kaya, Salah et al. 2019 "Predicting depression and emotions in the cross-roads of cultures, para-linguistics and non-linguistics" (AVEC'19) | E-DAIC | ASR-derived linguistic + word-duration features | Feature fusion | test | Test CCC 0.344 | Linguistic ASR features alone are competitive | [author proof](https://webspace.science.uu.nl/~salah006/avec19authorproof.pdf) |
| 15 | "Multimodal framework for state-of-mind assessment with sentiment pre-classification" 2019 (AVEC'19) | E-DAIC | multimodal | Sentiment pre-classification then fusion | unclear | CCC 0.531 | — | [cyZhu98 list](https://github.com/cyZhu98/depression_papers) |
| 16 | Sun et al. 2021/22 TensorFormer | E-DAIC | A, V, T | Tensor-based transformer | test (as cited) | RMSE 4.31, CCC 0.491 | — | as cited in [Tank 2024](https://arxiv.org/html/2407.06125v2) |
| 17 | Sun et al. 2022 CubeMLP (ACM MM) | E-DAIC | T, A, V | MLP-based multimodal mixing | test (as cited) | MAE 4.37, CCC 0.583 | — | as cited in [Mandal 2025](https://arxiv.org/html/2503.20496) |
| 18 | Zhao & Wang 2022 | E-DAIC | T, A, V | Cross-attention plus SA-GAN augmentation | as cited | RMSE 4.14, MAE 3.56 | — | as cited in [Mandal 2025](https://arxiv.org/html/2503.20496) |
| 19 | Van Steijn et al. 2022 | E-DAIC | T, A (hand-crafted) | **Early fusion (concatenation)** | as cited | CCC 0.62, RMSE 6.06 | No visual, simple fusion, still competitive | as cited in [Mandal 2025](https://arxiv.org/html/2503.20496) |
| 20 | Li et al. FPT | E-DAIC | A, V | Transformer | as cited | RMSE 4.80, MAE 4.58 | — | as cited in [Tank 2024](https://arxiv.org/html/2407.06125v2) |
| 21 | Teng et al. | E-DAIC | A, V, T | — | as cited | MAE 5.21, CCC 0.466 | — | as cited in [Tank 2024](https://arxiv.org/html/2407.06125v2) |
| 22 | Sadeghi et al. | E-DAIC | T | LLM features | as cited | RMSE 5.36, MAE 4.26 | Text only | as cited in [Tank 2024](https://arxiv.org/html/2407.06125v2) |
| 23 | Yuan et al. 2024 MMFF | E-DAIC | T, A, V | High-order multimodal factor fusion | as cited | CCC 0.676, RMSE 4.91, MAE 3.98 | — | as cited in [Mandal 2025](https://arxiv.org/html/2503.20496) |
| 24 | Wei et al. 2022 "Multi-modal depression estimation based on sub-attentional fusion" (ECCV-W) | DAIC-WOZ | A, V, T | Sub-attention fusion | — | numbers not extracted | — | [arXiv 2207.06180](https://arxiv.org/pdf/2207.06180); [Springer](https://link.springer.com/chapter/10.1007/978-3-031-25075-0_42) |
| 25 | Guo et al. 2022 TOAT, topic-attentive transformer | DAIC-WOZ | A, T | Topic-attentive transformer | dev / test | Dev F1 0.741 (A+T); Test F1 0.647 (audio), as cited | — | [arXiv 2206.13256](https://arxiv.org/pdf/2206.13256); numbers as cited in [Zhao 2025](https://arxiv.org/html/2511.19877v2) |
| 26 | Burdisso et al. 2024 "DAIC-WOZ: on the validity of using the therapist's prompts" | DAIC-WOZ | T | — | — | Models using interviewer prompts reach F1 about 0.90 from text alone by exploiting prompt position (shortcut) | Participant-only models spread evidence across the interview | [arXiv 2404.14463](https://arxiv.org/pdf/2404.14463) |
| 27 | Tank et al. 2024 "Depression detection using LLMs on textual and audio-visual modalities" (arXiv, IIIT-Delhi) | E-DAIC 163/56/56 | T (Whisper-v3 ASR → GPT-4/Llama-3), A (MFCC, Whisper), V (49 OpenFace: pose, gaze, 17 AUs) | A+V concatenation → BiLSTM-attention | test (T), val (A/V) | GPT-4 2-shot text: RMSE 3.975 / MAE 3.161 / CCC 0.781. Val: audio BiLSTM RMSE 5.39, video BiLSTM 6.45, A+V fused 6.51 (or 6.72; the paper is internally inconsistent) | **Adding video to audio made RMSE worse**; A/V classifiers collapsed to predicting "healthy" | [arXiv 2407.06125v2](https://arxiv.org/html/2407.06125v2) |
| 28 | Mandal, Atzil-Slonim, Solorio, Gurevych 2025 QuestMF (arXiv; code "clpsych2025") | E-DAIC 163/56/56 (one dev and one test session dropped for missing video) | T (sentence-transformer), A (openSMILE LLDs), V (ResNet) | Turn-level cross-attention; one model per PHQ-8 item; ordinal ImbOLL loss | test, 3 seeds | Best CCC 0.685 (T+A+V) | **Full 7-way ablation (CCC, QuestMF):** T 0.615, A 0.273, V -0.052, T+A 0.643, T+V 0.659, A+V 0.159, T+A+V 0.662 (seed std about 0.02). Single-score regression: T 0.591, A 0.212, V -0.067, T+A 0.607, T+V 0.610, A+V 0.058, T+A+V 0.602 | [arXiv 2503.20496](https://arxiv.org/html/2503.20496) |
| 29 | Zhao, Shen, ..., Ge 2025 "It hears, it sees too: multi-modal LLM ... integrating visual understanding into audio LMs" (arXiv 2511.19877) | DAIC-WOZ 107/35/47 | A, T, V (pre-extracted landmarks + AUs; raw video not available) | Element-wise addition of timestamp-aligned visual and audio embeddings in Qwen2-Audio | dev & test | Test F1 0.825 (A+T+V) | **Dev F1:** T 0.564, V 0.668, A+T 0.720, T+V 0.643, A+T+V 0.789 (0.844 with multi-scale inference). Visual adds +0.069 F1 over A+T | [arXiv html](https://arxiv.org/html/2511.19877v2) |
| 30 | Prior DAIC-WOZ rows cited by Zhao 2025 | DAIC-WOZ | various | various | dev / test | Dev F1: RoBERTa T 0.602, Llama2-13B T 0.636, Whisper-v3 A 0.694, WavLM A 0.720, LSTM A+T 0.770, C-CNN A+T+V 0.769, ConvBiLSTM A+T+V about 0.70. Test F1: GloVe-CNN T 0.68, EmoAudioNet A 0.66, HiQuE A+T+V 0.79, MultiDepNet A+T 0.785 | Frozen SSL audio (WavLM) is as strong as text on dev | [Zhao 2025](https://arxiv.org/html/2511.19877v2) |
| 31 | MFDCL 2026 "Modality feature decomposition and contrastive learning" (MDPI Algorithms) | DAIC-WOZ | A, T, V | Decomposition (shared/private) plus contrastive learning | — | P 0.92 / R 0.90 / F1 0.91 / Acc 0.94 | Removing decomposition and contrastive modules: F1 0.91 → 0.85 (naive concatenation) | [doi 10.3390/a19060452](https://doi.org/10.3390/a19060452) |
| 32 | Gómez-Zaragoza et al. 2025 (Interspeech) "Speech and text foundation models for depression detection" | DAIC-WOZ | A, T | **Early fusion** of emotion-tuned foundation-model embeddings was best | — | Prior DAIC-WOZ A/T models typically F1 0.5–0.7 | Performance significantly lower for female speakers | [ISCA PDF](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf) |
| 33 | "Automatic depression detection via learning and fusing features from visual cues" 2022 | DAIC-WOZ | V only | Visual feature fusion | dev | Comparison table (garbled) suggests 2D landmarks + head pose about 0.857 F1 vs CNN on AUs + gaze + pose 0.609 (different methods) | Visual-only | [arXiv 2203.00304](https://arxiv.org/pdf/2203.00304) |
| 34 | 3D facial landmark pseudo-images + CNN-LSTM 2025 (MDPI BioMedInformatics) | DAIC-WOZ, E-DAIC | V only | — | — | Macro F1 0.74 on DAIC-WOZ | Landmark geometry alone carries signal | [doi 10.3390/biomedinformatics6010008](https://doi.org/10.3390/biomedinformatics6010008) |
| 35 | Flores et al. 2022 "Temporal facial features for depression screening" (UbiComp/ISWC adj.) | DAIC-WOZ | V (gaze, landmarks, AUs over 15 questions) | per-question time series | — | Eye gaze reached about F1 0.81 for one question (fragmentary snippet; verify) | Question-specific visual signal | [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/3544793.3563424) |
| 36 | Shen et al. 2022 EATD corpus + GRU/BiLSTM (ICASSP) | EATD + DAIC-WOZ | A, T | GRU/BiLSTM fusion | — | numbers not extracted | — | [arXiv 2202.08210](https://arxiv.org/pdf/2202.08210) |
| 37 | Female-only DAIC-WOZ model 2026 (F1000Research) | DAIC-WOZ female subset | T, A, V | deep multimodal | — | Accuracy 88% (authors call results "nuanced") | Gender-specific | [F1000Research 15-1002](https://f1000research.com/articles/15-1002) |

**Other ablation and protocol facts**
- A 2025/26 preprint argues that many DAIC-WOZ fusion gains are validated only in cross-validation and never confirmed on unseen participants. A teacher-student model reports test F1 0.991, which other authors attribute to leakage — [ResearchGate: adaptive gating vs naive concatenation](https://www.researchgate.net/publication/415639022_Multimodal_Depression_Detection_from_Audio_Visual_and_Textual_Cues_Evaluation_of_Adaptive_Gating_versus_Naive_Concatenation)
- A non-peer-reviewed GitHub project (BERT + Librosa + OpenFace, random-forest late fusion) attributes about 51% of predictions to audio, 29% to text and 20% to visual — [GitHub hossmurad](https://github.com/hossmurad/daic-woz-multimodal-depression-detection)

**Feature availability (visual)**
- DAIC-WOZ does not release raw video. Visual work uses pre-extracted OpenFace-style facial features (landmarks, AUs, gaze, head pose) — [Zhao 2025](https://arxiv.org/html/2511.19877v2); [emergentmind DAIC-WOZ summary](https://www.emergentmind.com/topics/distress-analysis-interview-corpus-wizard-of-oz-daic-woz)
- E-DAIC distributes OpenFace AU, gaze and pose features (Tank et al. used 49: pose, gaze, 17 AUs) and ResNet deep visual features (used by QuestMF) — [Tank 2024](https://arxiv.org/html/2407.06125v2); [Mandal 2025](https://arxiv.org/html/2503.20496)
- At least one E-DAIC dev session and one test session have incomplete video features — [Mandal 2025](https://arxiv.org/html/2503.20496)

### Inferences
- E-DAIC visual features (ResNet in particular) are near-useless alone (negative CCC). Their small positive contribution only appears when stacked on top of text. That pattern is consistent with a weak, partly redundant signal plus regularisation noise, not a strong independent modality.
- DAIC-WOZ visual gains (for example +0.07 F1) come from 35- or 47-speaker evaluation sets, where one or two flipped depressed speakers move F1 by about 0.05. They are not reliable estimates of the gain under our 272-speaker repeated nested CV.
- For our pipeline, a realistic expectation from adding a calibrated visual (OpenFace-AU/gaze/pose functionals) base model to the stacker is an AUC change between about 0 and +0.03. That gain could be undetectable given our CI. It is worth testing as a pre-registered single experiment, not a search.

### Gaps
- I could not extract the AVEC 2019 overview tables (per-feature-set baselines for BoAW, DeepSpectrum, OpenFace, ResNet, VGG, BoVW on dev/test). The full release list (VGG-16, BoVW) is from background knowledge and was not verified here.
- Ray et al. 2019 per-modality numbers, and whether CCC 0.67 is dev or test, are disputed: one list marks the paper dev-only, while Tank et al. call it the "AVEC 2019 winner".
- Unverified recollections (excluded from the table): Pampouchidou et al. 2016 (AVEC'16, gender-dependent fusion of high/low-level features); Samareh et al. 2018 (gender-specific multimodal); Lam et al. 2019 ICASSP (topic-based augmentation, high F1); Rohanian et al. 2019 Interspeech (word-level multimodal fusion with gating); Toto et al. 2021 AudiBERT; Ma et al. 2016 DepAudioNet; Dinkel et al. 2019 text-only; Song, Shen & Valstar 2018 FG (hand-crafted visual statistics). Verify these before citing.
- AVEC 2016 per-modality baseline F1 (audio vs video vs fusion) was not retrieved.

## Q2. AVEC 2016 / 2017 / 2019 winners and their methods

### Takeaway
The winners were dominated by **text/linguistic features**, not visual. AVEC 2017 was won by Stepanov et al. using transcript turn features (test RMSE 4.94 / MAE 4.11), per their own claim. Yang et al.'s hybrid A+V+T system scored test RMSE 5.40. AVEC 2019's top result (Ray et al., multi-level attention over T+A+V) reached CCC 0.67, mostly from text, against a baseline CCC of 0.111. Other AVEC 2019 entries reached test CCC 0.40–0.44.

### Cited Findings
- AVEC 2016 DCC: binary PHQ-8 label, scored by average F1 over both classes. Baseline was a linear SVM with SGD in scikit-learn — [AVEC 2016 overview](https://arxiv.org/pdf/1605.01600)
- The AVEC'16 proceedings have a "Sub-Challenge Winners" section, but the snippet did not name the winners — [AVEC'16 TOC](http://www.sigmm.org/opentoc/AVEC2016-TOC)
- AVEC 2017 DSC: Stepanov et al. claim the win with transcript turn features (test MAE 4.11, RMSE 4.94) — [arXiv 1711.06095](https://arxiv.org/abs/1711.06095)
- AVEC 2017: Yang et al. hybrid A+V+T scored test RMSE 5.400 / MAE 4.359 vs baseline 7.05 / 5.66 — [ResearchGate](https://www.researchgate.net/publication/320545362_Hybrid_Depression_Classification_and_Estimation_from_Audio_Video_and_Text_Information); [PMC10308283](https://pmc.ncbi.nlm.nih.gov/articles/PMC10308283/)
- Conflict: Gong & Poellabauer 2017 is labelled "Depression Challenge Winner" on ResearchGate — [ResearchGate](https://www.researchgate.net/publication/320544890_Topic_Modeling_Based_Multi-modal_Depression_Detection)
- AVEC 2019 DDS: E-DAIC has 275 sessions (163/56/56), including autonomous-avatar interviews. Baseline test RMSE 6.37, CCC 0.111 — [Tank 2024](https://arxiv.org/html/2407.06125v2); [cyZhu98](https://github.com/cyZhu98/depression_papers); [overview](https://arxiv.org/pdf/1907.11510)
- AVEC 2019: Ray et al. (multi-level attention, Bi-LSTM, T+A+V) reported CCC 0.67, RMSE 4.37, MAE 4.02 on the textual modality, described as the AVEC 2019 winner — [Tank 2024](https://arxiv.org/html/2407.06125v2); [arXiv 1909.01417](https://arxiv.org/abs/1909.01417)
- Other AVEC 2019 test results: Yin et al. hierarchical RNN CCC 0.442 (RMSE 5.50); Fan et al. dilated CNN CCC 0.430 (RMSE 5.91); Makiuchi et al. BERT-CNN + gated-CNN (A+T only) CCC 0.403; Kaya/Salah ASR-linguistic CCC 0.344 — [cyZhu98](https://github.com/cyZhu98/depression_papers); [AVEC'19 TOC](http://www.sigmm.org/opentoc/AVEC2019-TOC); [Salah proof](https://webspace.science.uu.nl/~salah006/avec19authorproof.pdf)

### Inferences
- In every challenge year, the lexical channel carried most of the signal. Top systems were designed around text, with audio and video as auxiliaries. This supports our text-first design.
- Ray et al.'s CCC 0.67 versus the other teams' 0.40–0.44 is a large gap. With the dev/test ambiguity, treat it cautiously.

### Gaps
- Official AVEC 2016/2017/2019 ranking tables were not retrieved (PDF extraction failed), so "winner" is based on author claims and secondary citations.

## Q3. Fusion strategies that work at N about 200–300

### Takeaway
On E-DAIC/DAIC-WOZ, simple fusion is competitive with elaborate architectures. Early-fusion concatenation of hand-crafted T+A (Van Steijn: CCC 0.62) is within about 0.05 CCC of attention, tensor and factor-fusion models (0.58–0.69). Seed variance (std about 0.02 CCC) and split noise are comparable to the architecture differences. Turn- or question-level alignment (QuestMF, topic modelling) gives the most consistent extra gain. For a 272-speaker binary task, late fusion of calibrated per-modality scores (our approach) is the lowest-variance option. Adding a visual base model to the stacker is the natural, cheap extension.

### Cited Findings
- Early fusion (concatenation) of hand-crafted text+audio: CCC 0.62 on E-DAIC — as cited in [Mandal 2025](https://arxiv.org/html/2503.20496)
- Cross-attention, MLP-mixing, tensor and factor fusion on E-DAIC span CCC 0.49 (TensorFormer) to 0.676 (MMFF) — [Mandal 2025](https://arxiv.org/html/2503.20496); [Tank 2024](https://arxiv.org/html/2407.06125v2)
- Question-wise modelling (8 per-item models) raised CCC over single-score regression in every combination (for example T+A+V 0.602 → 0.662). Seed std was about 0.01–0.03 — [Mandal 2025](https://arxiv.org/html/2503.20496)
- Early fusion of emotion-tuned foundation-model embeddings was the best A+T approach on DAIC-WOZ (Interspeech 2025) — [Gómez-Zaragoza 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf)
- A+V concatenation into a BiLSTM underperformed audio alone on E-DAIC val (RMSE 6.51 vs 5.39), and A/V classifiers collapsed to the majority class — [Tank 2024](https://arxiv.org/html/2407.06125v2)
- Fine-tuning large models was limited by small sample size (Llama-3 fine-tune produced malformed outputs) — [Tank 2024](https://arxiv.org/html/2407.06125v2)
- Concerns that DAIC-WOZ fusion gains are cross-validation artefacts or shortcut learning; near-ceiling F1 0.99 attributed to leakage — [ResearchGate preprint](https://www.researchgate.net/publication/415639022_Multimodal_Depression_Detection_from_Audio_Visual_and_Textual_Cues_Evaluation_of_Adaptive_Gating_versus_Naive_Concatenation); [Burdisso 2024](https://arxiv.org/pdf/2404.14463)
- Contrastive/decomposition fusion claims F1 0.91 vs 0.85 for naive concatenation on DAIC-WOZ. This is unclear protocol and above the about-0.9 level that critics associate with shortcuts — [MFDCL 2026](https://doi.org/10.3390/a19060452)

### Inferences
- Learned joint-attention fusion needs aligned sequences and many parameters. At 272 speakers it mostly adds variance. Score-level stacking with few parameters (one weight per modality) is the defensible choice, and it handles missing visual sessions gracefully.
- The most transferable idea is **question/topic alignment**: computing features per interview prompt or topic. This means alignment only, not using the prompt text, to avoid the Burdisso shortcut.
- Any reported F1 >0.85 on DAIC-WOZ should be treated as suspect unless the paper is speaker-disjoint and participant-only.

### Gaps
- No study found compares late-score stacking against attention fusion under repeated speaker-disjoint CV on the combined N about 270 set. That is the gap our project could fill.

## Q4. Gender-specific or domain-specific models

### Takeaway
DAIC-WOZ has a documented gender bias that can inflate reported performance. Recent work finds lower performance for female speakers. Mitigation is usually debiasing (re-distribution, causal/counterfactual methods) rather than separate gender models. I found no rigorous side-by-side male-vs-female-model comparison in this search. Domain-specific modelling (DAIC-WOZ human-piloted WoZ vs E-DAIC autonomous avatar) is barely addressed.

### Cited Findings
- Bailey & Plumbley (EUSIPCO 2021): gender bias in DAIC-WOZ audio can inflate performance. Mitigated via data re-distribution and raw audio. Preprocessing code removes known DAIC-WOZ errors (pre-interview chatter, interruptions) — [arXiv 2010.15120](https://arxiv.org/abs/2010.15120v3); [GitHub daic_woz_process](https://github.com/adbailey1/daic_woz_process)
- Interspeech 2025: performance significantly lower for female speakers; males roughly match overall test score — [Gómez-Zaragoza 2025](https://www.isca-archive.org/interspeech_2025/gomezzaragoza25_interspeech.pdf)
- IJCAI 2024 referee-guided fusion: treats gender as a confounder in multimodal fusion and proposes causal debiasing on DAIC-WOZ and other data — [IJCAI 2024](https://www.ijcai.org/proceedings/2024/0799.pdf)
- Counterfactual-inference gender debiasing on DAIC-WOZ (2025) — [arXiv 2512.01834](https://arxiv.org/html/2512.01834v2)
- Female-only multimodal DAIC-WOZ model: 88% accuracy, with authors noting nuanced results and limited data — [F1000Research](https://f1000research.com/articles/15-1002)
- Cross-context transfer of depressive facial patterns (German RCT ↔ E-DAIC) studied in 2026 — [arXiv 2609.05543](https://arxiv.org/pdf/2609.05543)
- DAIC-WOZ's virtual interviewer uses a fixed set of 191 prompts; interviewer effects are a recognised bias source — [LREC 2026 / arXiv 2603.24651](https://arxiv.org/pdf/2603.24651)

### Inferences
- Splitting N=272 by gender halves the depressed-class count (about 85 → about 40 per gender). That would almost certainly raise variance more than it reduces bias. Report per-gender AUC and add gender as a stacker covariate or stratification variable instead.
- The DAIC-WOZ vs E-DAIC-Avatar shift (human-piloted vs autonomous agent, and possibly different OpenFace versions for visual features) argues for a corpus indicator in the stacker, or per-corpus calibration before stacking, especially for visual features.

### Gaps
- No verified study of separate gender-specific models with numbers. The recalled examples (Pampouchidou 2016, Samareh 2018) are not verified.
- Whether DAIC-WOZ (CLNF/OpenFace v1 era) and E-DAIC (OpenFace 2) visual features are directly comparable was not verified. Check the E-DAIC documentation before pooling visual features across the two corpora.
