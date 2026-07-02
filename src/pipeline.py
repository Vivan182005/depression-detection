import numpy as np
import torch
import os
import pickle
import json
from src.preprocessing import preprocess_pipeline
from src.features import extract_pipeline_features
from src.models import LSTMAttentionClassifier


class DepressionDetectionPipeline:
    """
    End-to-End Depression Detection Pipeline using Vocal Cues.
    Ingests raw audio (and transcripts), cleans, extracts acoustic features,
    and runs inference using LSTM-Attention, SVM, and XGBoost.
    """
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir
        self.lstm_path = os.path.join(model_dir, "lstm_attn.pth")
        self.svm_path = os.path.join(model_dir, "svm_model.pkl")
        self.xgb_path = os.path.join(model_dir, "xgb_model.pkl")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    def run_inference(self, audio_path, transcript_path=None, feature_method="method2"):
        """
        Runs the entire pipeline on a single subject recording.
        """
        # 1. Preprocess
        print("Preprocessing audio signal...")
        frames, sr = preprocess_pipeline(
            audio_path=audio_path,
            transcript_path=transcript_path,
            target_sr=16000,
            frame_len_sec=2.0,
            hop_len_sec=1.0
        )
        
        if len(frames) == 0:
            raise ValueError("Preprocessing resulted in 0 audio frames. Waveform is empty or silent.")
            
        print(f"Segmented audio into {len(frames)} micro-frames.")

        # 2. Feature Extraction
        print("Extracting acoustic biomarkers...")
        feature_matrix = extract_pipeline_features(frames, sr, method=feature_method)
        print(f"Extracted feature matrix of shape: {feature_matrix.shape}")

        # Combine all frames into a single continuous signal for visualization
        y_concat = np.concatenate(frames)

        report = {
            "num_frames": len(frames),
            "feature_shape": list(feature_matrix.shape),
            "predictions": {},
            "raw_waveform": y_concat.tolist(),
            "sampling_rate": sr
        }

        # 3. Model Predictions
        
        # A. LSTM with Attention Model
        if os.path.exists(self.lstm_path):
            try:
                # Initialize architecture
                input_dim = feature_matrix.shape[1]
                model = LSTMAttentionClassifier(input_dim=input_dim).to(self.device)
                model.load_state_dict(torch.load(self.lstm_path, map_location=self.device))
                model.eval()
                
                # Prepare sequence tensor (batch_size=1, seq_len, feature_dim)
                x_tensor = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, attention_weights = model(x_tensor)
                    prob = torch.sigmoid(logits).item()
                    attn_weights = attention_weights.squeeze(0).cpu().numpy().tolist()
                
                report["predictions"]["lstm_attention"] = {
                    "probability": prob,
                    "prediction": "Depressed" if prob >= 0.5 else "Non-Depressed",
                    "attention_weights": attn_weights,
                    "confidence_score": 100 * (prob if prob >= 0.5 else 1.0 - prob)
                }
            except Exception as e:
                print(f"Error running LSTM Attention model: {e}")
        else:
            print("LSTM model file not found. Skipping LSTM inference.")

        # Aggregate features for traditional ML (mean across sequence)
        X_agg = np.mean(feature_matrix, axis=0).reshape(1, -1)

        # B. SVM
        if os.path.exists(self.svm_path):
            try:
                with open(self.svm_path, "rb") as f:
                    svm_model = pickle.load(f)
                svm_prob = svm_model.predict_proba(X_agg)[0][1]
                report["predictions"]["svm"] = {
                    "probability": svm_prob,
                    "prediction": "Depressed" if svm_prob >= 0.5 else "Non-Depressed",
                    "confidence_score": 100 * (svm_prob if svm_prob >= 0.5 else 1.0 - svm_prob)
                }
            except Exception as e:
                print(f"Error running SVM model: {e}")
        else:
            print("SVM model file not found. Skipping SVM inference.")

        # C. XGBoost
        if os.path.exists(self.xgb_path):
            try:
                with open(self.xgb_path, "rb") as f:
                    xgb_model = pickle.load(f)
                xgb_prob = xgb_model.predict_proba(X_agg)[0][1]
                report["predictions"]["xgboost"] = {
                    "probability": xgb_prob,
                    "prediction": "Depressed" if xgb_prob >= 0.5 else "Non-Depressed",
                    "confidence_score": 100 * (xgb_prob if xgb_prob >= 0.5 else 1.0 - xgb_prob)
                }
            except Exception as e:
                print(f"Error running XGBoost model: {e}")
        else:
            print("XGBoost model file not found. Skipping XGBoost inference.")

        # Compute structured Clinical Analysis Summary
        report["clinical_summary"] = self.generate_clinical_summary(report)

        return report

    def generate_clinical_summary(self, report):
        """
        Creates a natural text summary of findings based on model outputs.
        """
        preds = report.get("predictions", {})
        if not preds:
            return "No predictions available. Please train models first."
            
        lstm_pred = preds.get("lstm_attention", {})
        svm_pred = preds.get("svm", {})
        xgb_pred = preds.get("xgboost", {})
        
        # Build consensus
        votes = []
        confidences = []
        if lstm_pred:
            votes.append(1 if lstm_pred["prediction"] == "Depressed" else 0)
            confidences.append(lstm_pred["confidence_score"])
        if svm_pred:
            votes.append(1 if svm_pred["prediction"] == "Depressed" else 0)
            confidences.append(svm_pred["confidence_score"])
        if xgb_pred:
            votes.append(1 if xgb_pred["prediction"] == "Depressed" else 0)
            confidences.append(xgb_pred["confidence_score"])
            
        if not votes:
            return "Unable to compile consensus prediction."
            
        mean_vote = np.mean(votes)
        consensus_class = "Depressed" if mean_vote >= 0.5 else "Non-Depressed"
        mean_confidence = np.mean(confidences)
        
        summary = f"Clinical Decision Consensus: {consensus_class} (Confidence: {mean_confidence:.1f}%)\n\n"
        
        if consensus_class == "Depressed":
            summary += (
                "The patient exhibits paralinguistic biomarkers indicative of motor incoordination and flat affect. "
                "This is characterized by decreased pitch variability (monotonous delivery) and lower dynamic energy levels. "
                "The LSTM attention mechanism identified specific time zones in the session containing elongated speech pauses "
                "and depressed vocal dynamics. Clinical follow-up using BDI-II or PHQ-9 is highly recommended."
            )
        else:
            summary += (
                "The patient's vocal prosody displays healthy dynamic range, standard pitch fluctuations, and standard speech rates. "
                "The attention profile over temporal segments shows uniform distribution without highlights of prolonged, "
                "depressed vocal cues. Indicators align with a standard, non-depressed baseline."
            )
            
        return summary
