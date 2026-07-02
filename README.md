# Depression Detection using Vocal Cues (Capstone PW26_CBH_04)

An end-to-end, modular machine learning data pipeline and interactive screening dashboard to detect clinical depression from paralinguistic biomarkers in voice recordings. 

This project is built based on the architecture described in **Capstone Review Phase-2 (Ms. Chandravva Hebbi)**, which leverages speaker-isolated audio transcripts, advanced spectral filtering, and sequence modeling with deep attention mechanisms.

---

## 🛠️ Pipeline Architecture & Workflow

```
[Raw Audio + Transcript]
          │
          ▼
 1. SPEAKER ISOLATION ──► Discards Virtual Interviewer ("Ellie") sections
          │
          ▼
 2. SIGNAL PREPROCESSING ──► Resampling (16kHz), Bandpass Filter (50-8000Hz),
          │                  Denoising (Spectral Noise Gating), RMS Normalization
          ▼
 3. MICRO-SEGMENTATION ──► Dynamic 2.0s overlapping frames (1.0s hop)
          │
          ▼
 4. FEATURE EXTRACTION ──► Extracts 33-dim acoustic vectors per frame
          │                (MFCCs, Delta MFCCs, Loudness, Pitch)
          ▼
 5. DUAL-PATH INFERENCE
       ├──► Frame Sequences ──► LSTM + Attention Neural Network ──► Temporal Attention Weights
       └──► Statistical Means ──► SVM & XGBoost Baselines
          │
          ▼
 6. EXPLAINABLE SCREENING ──► High-salience overlay heatmaps + Clinical report
```

---

## 📁 Repository Structure

```
depression-detection-pipeline/
├── app.py                     # Gorgeous Streamlit Web UI & Diagnostic Dashboard
├── requirements.txt           # Python package dependencies
├── src/
│   ├── __init__.py            # Module initializer
│   ├── preprocessing.py       # Audio cleaning, noise reduction, and speaker isolation
│   ├── features.py            # Librosa acoustic & prosodic feature extractor
│   ├── models.py              # PyTorch LSTM-Attention, SVM, and XGBoost models
│   ├── train.py               # Model training loop & cross-validation scripts
│   ├── explainability.py      # Attention waveform overlays & SHAP interpretability
│   └── pipeline.py            # End-to-end inference orchestrator
├── tests/
│   └── test_pipeline.py       # Unit tests for preprocessing & feature extraction
└── models/                    # Directory where trained weights are stored (.pth, .pkl)
```

---

## 🚀 Installation & Setup

Ensure Python 3.10+ is installed on your local machine.

1. **Clone the repository** (or navigate to the workspace directory):
   ```bash
   cd /Users/vivannaik/.gemini/antigravity-ide/scratch/depression-detection-pipeline
   ```

2. **Set up a Virtual Environment & Activate it**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Pipeline Usage

### 1. Training & Calibration
If you do not have pre-trained model weights, you can train and calibrate the models (LSTM with Attention, SVM, and XGBoost) using the simulated training script:
```bash
python -m src.train
```
*This generates synthetic sessions mimicking depressed (flattened intonation, low dynamic energy) and non-depressed vocal behaviors to compile initial calibrated weights under `models/`.*

### 2. Launching the Screening Dashboard
To start the premium interactive web app:
```bash
streamlit run app.py
```
*Open the local URL displayed in your terminal (usually `http://localhost:8501`) to access the dashboard.*

### 3. Running Unit Tests
Validate that the feature extraction dimensions and audio filter components function correctly:
```bash
pytest tests/
```

---

## 🔬 Scientific & Clinical Basis

This system evaluates specific vocal biomarkers backed by clinical psychiatric research:
- **Prosodic Flattening ($F_0$ / Pitch & Energy)**: Depressed subjects frequently show psychomotor retardation, leading to reduced volume variability (lower loudness standard deviation) and monotonous delivery. *(Cummins et al., 2015; Williamson et al., 2014)*.
- **Vocal Tract Muscles (MFCCs & Spectral Flux)**: Timbre shifts capture voice quality changes (breathiness, tension) caused by changes in articulation and vocal muscle coordination. *(Alghowinem et al., 2013)*.
- **Bahdanau-style LSTM Attention**: Rather than treating an entire interview session equally, the attention layer allows the model to learn long-range sequence dependencies and highlight key emotional periods (such as long pauses or flat speech).
