# Automatic Depression Detection Using Voice Cues

**B.Tech Final Year Capstone Project**

---

## 📌 Project Overview
This repository contains the production pipeline for detecting depression using vocal cues from human speech recordings. The system processes raw interview sessions (such as the **DAIC-WOZ** benchmark dataset), isolates participant speech, cleans acoustic noise, extracts dual-branch clinical and deep speech features, and feeds them into a **BiLSTM + Self-Attention Neural Network** to predict depression probability, continuous PHQ-8 severity scores, and clinical Explainable AI (XAI) attention heatmaps.

---

## 🛠 Project Structure

```
app/
├── README.md                           # Project documentation & execution guide
├── requirements.txt                    # Project dependencies
├── remove_ellie.py                     # Entry point for transcript-driven speaker isolation
├── voice.py                            # Entry point for audio waveform & biomarker inspection
├── pipeline.py                         # Master End-to-End Execution Pipeline
└── src/
    ├── preprocessing/
    │   ├── speaker_isolation.py        # Exact timestamp splicing & 5ms cross-fading
    │   └── audio_cleaner.py            # Bandpass filter, spectral gating, RMS normalization
    ├── features/
    │   ├── egemaps_extractor.py        # eGeMAPS 88 prosodic features + temporal pause metrics
    │   └── wav2vec_extractor.py        # Deep Wav2Vec2 / WavLM contextual speech embeddings
    ├── models/
    │   ├── dataset.py                  # PyTorch 3s sliding window dataset (50% overlap)
    │   ├── depression_model.py         # Dual-Branch BiLSTM + Self-Attention Network
    │   └── train.py                    # Multi-task BCE + MSE loss training loop
    └── explainability/
        └── visualizer.py               # Timeline attention heatmaps & biomarker visual profile
```

---

## 🚀 Key Technical Enhancements Over Previous Approach

1. **Deterministic Speaker Isolation**:
   - Replaced non-deterministic speaker diarization (`pyannote`) with exact transcript timestamp parsing (`*_TRANSCRIPT.csv`).
   - Slices participant speech and stitches clips with 5ms linear cross-fading.
   - Eliminates dead-silence zero padding (`clean_audio = 0.0`), preventing spectral step artifacts.

2. **Dual-Branch Feature Representation**:
   - **Branch A (Interpretable Clinical Bio-markers)**: 88 eGeMAPS features (Pitch F0 variance, Formants F1-F3, Jitter, Shimmer, HNR) + Temporal Pause-to-Speech Ratios.
   - **Branch B (Contextual Deep Features)**: 768-dimensional Wav2Vec 2.0 / WavLM representations per 3-second sliding window.

3. **Temporal Attention & Explainable AI (XAI)**:
   - Self-attention mechanism computes frame attention weights $\alpha_t$ across the interview timeline.
   - Generates interactive timeline heatmaps highlighting exact minutes/seconds of the interview where depressive vocal markers were strongest.

---

## 💻 Quick Start & Execution

### 1. Install Dependencies
```bash
cd app
pip install -r requirements.txt
```

### 2. Isolate Participant Speech (Remove Ellie)
```bash
python remove_ellie.py /path/to/300_AUDIO.wav /path/to/300_TRANSCRIPT.csv /path/to/output/300_PATIENT_ONLY.wav
```

### 3. Inspect Audio Waveform & Vocal Biomarkers
```bash
python voice.py /path/to/output/300_PATIENT_ONLY.wav
```

### 4. Run Full End-to-End Pipeline
```bash
python pipeline.py --audio /path/to/300_AUDIO.wav --transcript /path/to/300_TRANSCRIPT.csv --output_dir ./output
```

---

## 📊 Output Diagnostic Artifacts
When running `pipeline.py`, the following diagnostic outputs are automatically created in `./output`:
- `300_PARTICIPANT_CLEAN.wav`: Preprocessed, denoised, participant-only audio file.
- `300_attention_heatmap.png`: Timeline heatmap highlighting key depressive speech moments.
- `300_biomarkers.png`: Bar plots of pause ratio, mean pause duration, and hesitation frequency.
- `300_diagnostic_report.txt`: Clinical screening summary report containing depression risk probability (%) and predicted PHQ-8 score.
