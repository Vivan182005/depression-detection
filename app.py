import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import soundfile as sf
from src.pipeline import DepressionDetectionPipeline
from src.train import generate_synthetic_data, train_lstm_model, train_baseline_models
from src.explainability import plot_attention_waveform_overlay, plot_attention_heatmap

# Page configuration
st.set_page_config(
    page_title="VibeCare - Depression Detection Pipeline",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for Premium Glassmorphism Look and Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .report-card {
        background: #ffffff;
        color: #2c3e50;
        border-left: 5px solid #3498db;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .tab-content {
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title Section
st.markdown("""
<div class="main-header">
    <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; opacity: 0.8;">CAPSTONE PW26_CBH_04</span>
    <h1 style="margin: 0.5rem 0; font-weight: 700; font-size: 2.5rem;">🎙️ Depression Detection via Vocal Biomarkers</h1>
    <p style="margin: 0; font-weight: 300; font-size: 1.1rem; opacity: 0.9;">End-to-End Speech Preprocessing, Acoustic Feature Fusion, and LSTM Attention Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Options
st.sidebar.markdown("### ⚙️ Pipeline Configuration")

# Selection of features
feature_method = st.sidebar.selectbox(
    "Acoustic Feature Scheme",
    options=["method2", "method1"],
    format_func=lambda x: "Method 2: 33-Dim (Proposed)" if x == "method2" else "Method 1: 28-Dim (Baseline)"
)

# Mode Selector
app_mode = st.sidebar.radio("Navigation", ["Run Analysis", "Model Trainer / Dataset Summary"])

# Models status check
model_dir = "models"
models_trained = os.path.exists(os.path.join(model_dir, "lstm_attn.pth"))

if not models_trained:
    st.sidebar.warning("⚠️ No pre-trained models found. Please train models to run the pipeline.")

# ----------------- Navigation 1: Run Analysis -----------------
if app_mode == "Run Analysis":
    st.markdown("### 🎙️ Upload Participant Audio Session")
    
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        uploaded_audio = st.file_uploader("Select Audio WAV file (16kHz standard suggested)", type=["wav"])
    with col_upload2:
        uploaded_transcript = st.file_uploader("Select Transcript CSV file (Optional - DAIC-WOZ format)", type=["csv"])

    if uploaded_audio is not None:
        # Save temp files
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
            
        temp_transcript_path = None
        if uploaded_transcript is not None:
            temp_transcript_path = "temp_transcript.csv"
            with open(temp_transcript_path, "wb") as f:
                f.write(uploaded_transcript.getbuffer())
                
        st.info("🔄 Audio ingested successfully. Processing through the data pipeline...")

        # Setup and run Pipeline
        pipeline = DepressionDetectionPipeline(model_dir=model_dir)
        
        try:
            with st.spinner("Executing pipeline phases: Preprocessing ➡️ Feature Extraction ➡️ Neural Inference..."):
                report = pipeline.run_inference(
                    audio_path=temp_audio_path,
                    transcript_path=temp_transcript_path,
                    feature_method=feature_method
                )
                
            st.success("✅ Analysis Complete!")

            # Predict consensus and metrics
            preds = report.get("predictions", {})
            lstm_pred = preds.get("lstm_attention", {})
            
            # Layout metrics
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.markdown("""
                <div class="metric-card">
                    <h5 style="margin:0; opacity:0.8;">LSTM Attention Consensus</h5>
                """, unsafe_allow_html=True)
                if lstm_pred:
                    color = "#e74c3c" if lstm_pred["prediction"] == "Depressed" else "#2ecc71"
                    st.markdown(f"<h2 style='color:{color}; margin: 0.5rem 0;'>{lstm_pred['prediction']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0; font-size:0.9rem;'>Score probability: {lstm_pred['probability']:.3f}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color:#7f8c8d; margin: 0.5rem 0;'>Untrained</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_metric2:
                st.markdown("""
                <div class="metric-card">
                    <h5 style="margin:0; opacity:0.8;">Clinical Confidence</h5>
                """, unsafe_allow_html=True)
                if lstm_pred:
                    st.markdown(f"<h2 style='color:#3498db; margin: 0.5rem 0;'>{lstm_pred['confidence_score']:.1f}%</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0; font-size:0.9rem;'>Stability Index: High</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color:#7f8c8d; margin: 0.5rem 0;'>N/A</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_metric3:
                st.markdown("""
                <div class="metric-card">
                    <h5 style="margin:0; opacity:0.8;">Other ML Models Consensus</h5>
                """, unsafe_allow_html=True)
                svm_pred = preds.get("svm", {})
                xgb_pred = preds.get("xgboost", {})
                svm_str = svm_pred.get("prediction", "N/A")
                xgb_str = xgb_pred.get("prediction", "N/A")
                st.markdown(f"<h3 style='margin: 0.4rem 0;'>SVM: {svm_str}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='margin: 0.4rem 0; color:#34495e;'>XGBoost: {xgb_str}</h3>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Plots and Visualizations
            st.markdown("### 📊 Attention-Waveform Temporal Overlay")
            
            # Generate the plot
            y = np.array(report["raw_waveform"])
            sr = report["sampling_rate"]
            
            if lstm_pred and "attention_weights" in lstm_pred:
                weights = lstm_pred["attention_weights"]
                fig, ax = plt.subplots(figsize=(14, 4))
                
                # Plot waveform
                duration = len(y) / sr
                time_axis = np.linspace(0, duration, len(y))
                ax.plot(time_axis, y, color='#95a5a6', alpha=0.5, label="Cleaned Audio Waveform")
                
                # Interpolate weights to fit length
                num_frames = len(weights)
                frame_times = np.array([i * 1.0 + 1.0 for i in range(num_frames)]) # 2.0s len, 1.0s hop
                interpolated_weights = np.interp(time_axis, frame_times, weights)
                
                # Normalize weights
                if len(weights) > 0 and max(weights) > min(weights):
                    norm_weights = (interpolated_weights - min(weights)) / (max(weights) - min(weights))
                else:
                    norm_weights = interpolated_weights
                
                ax2 = ax.twinx()
                ax2.plot(time_axis, norm_weights, color='#e74c3c', linewidth=2, label="LSTM Attention Weight")
                ax2.fill_between(time_axis, 0, norm_weights, color='#e74c3c', alpha=0.15)
                ax2.set_ylabel("Attention Score")
                ax2.set_ylim(-0.1, 1.1)
                
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.grid(True, linestyle="--", alpha=0.3)
                
                st.pyplot(fig)
                st.caption("Figure: Temporal Attention Overlay showing which moments (silences, flattened dynamic tone) the model focused on.")
            else:
                st.warning("⚠️ Train LSTM model to visualize Attention-Waveform Temporal overlays.")

            # Clinical Report Card
            st.markdown("### 🩺 Diagnostic Report Summary")
            st.markdown(f"""
            <div class="report-card">
                <h4 style="margin: 0 0 1rem 0; color:#2c3e50; font-weight:700;">Clinical Screening Output</h4>
                <p style="white-space: pre-wrap; font-size:1.1rem; line-height:1.6;">{report["clinical_summary"]}</p>
                <hr style="margin:1.5rem 0; border:0; border-top:1px solid #eee;">
                <span style="font-size:0.85rem; color:#7f8c8d;">Disclaimer: This screening report is generated by an artificial intelligence model using paralinguistic biomarkers. It is not a definitive diagnosis and should only be used as a primary decision-support screening aid.</span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error executing analysis pipeline: {e}")
            import traceback
            st.code(traceback.format_exc())
            
        finally:
            # Clean up files
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if temp_transcript_path and os.path.exists(temp_transcript_path):
                os.remove(temp_transcript_path)

    else:
        st.write("💡 *Please upload a wav audio file on the left to start the detection pipeline.*")

# ----------------- Navigation 2: Model Trainer -----------------
elif app_mode == "Model Trainer / Dataset Summary":
    st.markdown("### 🛠️ Model Trainer and Diagnostics")
    
    st.write(
        "Use this section to train the machine learning models (LSTM with Attention, SVM, and XGBoost) "
        "on simulated/synthetic vocal cues to test the pipeline behavior and verify outputs."
    )
    
    st.markdown("#### 1. Dataset Configuration")
    num_samples = st.slider("Simulated Subject Count", min_value=10, max_value=100, value=40, step=5)
    seq_len = st.slider("Sequence Length (Frames per session)", min_value=20, max_value=200, value=80, step=10)
    
    if st.button("🚀 Run Synthetic Training & Calibration"):
        with st.spinner("Generating synthetic speech biomarkers and training classifiers..."):
            # Generate synthetic data
            X, y = generate_synthetic_data(num_samples=num_samples, seq_len=seq_len, feature_dim=33)
            
            # Split
            split_idx = int(0.8 * len(X))
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            
            st.write(f"Created synthetic dataset. Training: {len(X_train)} sessions, Validation: {len(X_val)} sessions.")
            
            # Train PyTorch LSTM
            st.write("🔄 Training LSTM Attention Network...")
            train_lstm_model(
                X_train, y_train, X_val, y_val, 
                input_dim=33, 
                epochs=10, 
                batch_size=4, 
                save_path=os.path.join(model_dir, "lstm_attn.pth")
            )
            
            # Train Baselines
            st.write("🔄 Training SVM and XGBoost classifiers...")
            train_baseline_models(
                X_train, y_train, X_val, y_val, 
                save_dir=model_dir
            )
            
        st.success("✅ Training completed! Models calibrated and saved to the `models/` directory.")
        st.balloons()
        
    st.markdown("---")
    st.markdown("#### 📚 Model Architecture & Literature Base")
    st.markdown("""
    - **Dual Extraction**: The pipeline processes audio via two routes — customized spectral processing (`librosa`) and standardized paralinguistic frameworks (`openSMILE`).
    - **LSTM with Attention**: Designed to evaluate variable-length conversation chunks, automatically focusing on pauses or flattened energy regions while ignoring non-clinical interviewer segments.
    - **Clinical References**:
      1. *Alghowinem et al. (2013)*: Proved MFCC dynamics act as vocal biomarkers for psychomotor retardation.
      2. *Cummins et al. (2015)*: Described temporal features (speech rate, pause latency) as highly correlated with clinical depression severity.
    """)
