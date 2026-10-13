"""
Production Inference Script: infer_depression.py
Clinical Diagnostic Engine for Depression Severity Detection.
Integrates:
- Corrected Multi-Format BERT Text Extractor (DAIC-WOZ & E-DAIC 2019)
- OpenSMILE eGeMAPS 88d + 66d Temporal Acoustic Biomarkers
- Wav2Vec 2.0 Pretrained Frozen Audio Representations
- Candidate 2A Multi-Modal MLP (Calibrated with T = 1.4542)
- Optional High-Accuracy Ensemble Blend (Candidate 2A + Candidate 3A)

Usage:
  # Clean User Mode (Default for Demonstration)
  python3 app/infer_depression.py --audio path/to/audio.wav
  python3 app/infer_depression.py --pid 624

  # Developer Debug Mode (Full Diagnostic Breakdown)
  python3 app/infer_depression.py --pid 624 --debug
"""

import os
import sys
import glob
import re
import argparse
import contextlib
import warnings

# Suppress external library warnings and telemetry
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.egemaps_extractor import EgemapsExtractor
from src.features.wav2vec_extractor import Wav2VecExtractor
from src.features.bert_extractor import BertExtractor
from run_preprocessing import extract_features as extract_33d_features
from src.models.calibrated_model import TemperatureScaler
from label_mapper import OFFICIAL_PHQ8_LABELS, phq8_to_severity_class

# Clean user-facing clinical severity names
SEVERITY_DISPLAY_NAMES = [
    "Minimal/Mild Depression",
    "Moderate Depression",
    "Severe Depression"
]

SEVERITY_CATEGORIES = [
    "Minimal / Mild (PHQ-8: 0 - 9)",
    "Moderate Depression (PHQ-8: 10 - 14)",
    "Severe Depression (PHQ-8: 15 - 24)"
]


@contextlib.contextmanager
def suppress_internal_logs(enabled=True):
    """Context manager to suppress low-level stdout/stderr from libraries and C extensions."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


class StandardTriModalMLP(nn.Module):
    def __init__(self, in_dim=1690, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_class = nn.Linear(64, num_classes)
        self.head_phq = nn.Linear(64, 1)
        self.head_bin = nn.Linear(64, 1)

    def forward(self, batch):
        x = batch['x_fused'] if isinstance(batch, dict) else batch
        feat = self.net(x)
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        bin_logit = self.head_bin(feat).squeeze(-1)
        return {
            'severity_logits': logits,
            'severity_probs': F.softmax(logits, dim=-1),
            'phq8_score': phq,
            'binary_logits': bin_logit,
            'binary_probs': torch.sigmoid(bin_logit)
        }


def resolve_audio_and_transcript(audio_path=None, pid=None):
    """Resolves input audio and transcript paths with clear error messages."""
    if pid is not None:
        pattern_audio = os.path.join(BASE_DIR, "unseen", "**", f"*{pid}*.wav")
        audio_matches = glob.glob(pattern_audio, recursive=True)
        if not audio_matches:
            pattern_audio2 = os.path.join(BASE_DIR, "**", f"*{pid}*.wav")
            audio_matches = [m for m in glob.glob(pattern_audio2, recursive=True) if ".venv" not in m]

        if not audio_matches:
            raise FileNotFoundError(f"Could not locate audio file for Participant ID {pid}")
        audio_path = audio_matches[0]

    if not audio_path:
        raise ValueError("No audio path or participant ID specified. Use --audio <path> or --pid <id>.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

    fn = os.path.basename(audio_path)
    pid_match = re.search(r'\d+', fn)
    extracted_pid = int(pid_match.group()) if pid_match else None

    # Search for matching transcript
    transcript_path = None
    if extracted_pid is not None:
        candidates = [
            os.path.join(BASE_DIR, "unseen", "transcripts", f"{extracted_pid}_TRANSCRIPT.csv"),
            os.path.join(BASE_DIR, "unseen", "transcripts", "moderate", f"{extracted_pid}_TRANSCRIPT.csv"),
            os.path.join(BASE_DIR, "unseen", "transcripts", "mild", f"{extracted_pid}_TRANSCRIPT.csv"),
            os.path.join(BASE_DIR, "unseen", "transcripts", "minimal", f"{extracted_pid}_TRANSCRIPT.csv"),
            os.path.join(BASE_DIR, "unseen", "transcripts", "severe", f"{extracted_pid}_TRANSCRIPT.csv"),
            os.path.join(BASE_DIR, "unseen", "moderate", f"{extracted_pid}_TRANSCRIPT.csv"),
            os.path.join(BASE_DIR, "transcripts", f"{extracted_pid}_TRANSCRIPT.csv")
        ]
        for c in candidates:
            if os.path.exists(c):
                transcript_path = c
                break

    return audio_path, transcript_path, extracted_pid


def load_production_pipeline(use_ensemble=False):
    """Loads Candidate 2A and feature scalers from canonical artifacts/final_model/."""
    canonical_ckpt = os.path.join(BASE_DIR, "artifacts", "final_model", "model_checkpoint.pth")
    canonical_scaler = os.path.join(BASE_DIR, "artifacts", "final_model", "feature_scalers.joblib")
    fallback_ckpt = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1", "checkpoints", "cand2a_standard_mlp_mild_balanced.pth")
    fallback_scaler = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1", "checkpoints", "cand2a_feature_scalers.joblib")

    ckpt_2a_path = canonical_ckpt if os.path.exists(canonical_ckpt) else fallback_ckpt
    scaler_path = canonical_scaler if os.path.exists(canonical_scaler) else fallback_scaler

    if not os.path.exists(ckpt_2a_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_2a_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Feature scalers not found: {scaler_path}")

    scalers = joblib.load(scaler_path)

    # Load Model 2A
    ckpt_2a = torch.load(ckpt_2a_path, map_location='cpu', weights_only=False)
    m2a = StandardTriModalMLP(in_dim=1690, hidden_dim=128, dropout=0.3)
    m2a.load_state_dict(ckpt_2a['model_state_dict'])
    m2a.eval()
    cal_2a = TemperatureScaler(m2a)
    cal_2a.temperature.data.fill_(ckpt_2a.get('optimal_temperature', 1.4542))
    cal_2a.eval()

    cal_3a = None
    if use_ensemble:
        ckpt_3a_path = os.path.join(BASE_DIR, "experiments", "capstone_accuracy_v2", "checkpoints", "cand3a_standard_mlp_alpha_015.pth")
        if os.path.exists(ckpt_3a_path):
            ckpt_3a = torch.load(ckpt_3a_path, map_location='cpu', weights_only=False)
            m3a = StandardTriModalMLP(in_dim=1690, hidden_dim=128, dropout=0.3)
            m3a.load_state_dict(ckpt_3a['model_state_dict'])
            m3a.eval()
            cal_3a = TemperatureScaler(m3a)
            cal_3a.temperature.data.fill_(ckpt_3a.get('optimal_temperature', 1.3047))
            cal_3a.eval()

    return cal_2a, cal_3a, scalers


def extract_session_features(audio_path, transcript_path, egemaps_ext, wav2vec_ext, bert_ext, debug=False):
    """Extracts multimodal features using the training-time continuous protocol."""
    if debug:
        print(f"  [1/3] Extracting acoustic biomarkers (88d eGeMAPS + 66d temporal stats)...")

    data_33d = extract_33d_features(audio_path)
    audio_signal = data_33d['audio']
    temporal_frames = data_33d['temporal_features']

    e88, _ = egemaps_ext.extract(audio_signal)
    t_mean = np.mean(temporal_frames, axis=0) if len(temporal_frames) > 0 else np.zeros(33, dtype=np.float32)
    t_std = np.std(temporal_frames, axis=0) if len(temporal_frames) > 0 else np.zeros(33, dtype=np.float32)
    ac_154d = np.hstack([e88, t_mean, t_std]).astype(np.float32)

    if debug:
        print(f"  [2/3] Extracting pretrained Wav2Vec 2.0 representations (768d)...")
    w768 = wav2vec_ext.extract_embedding(audio_signal, sr=16000).astype(np.float32)

    if debug:
        print(f"  [3/3] Extracting pretrained BERT text representations (768d)...")
    if transcript_path and os.path.exists(transcript_path):
        text, word_count = BertExtractor.extract_participant_text(transcript_path)
        b768 = bert_ext.extract_from_text(text).astype(np.float32)
        if debug:
            print(f"        Processed {word_count} participant words from {os.path.basename(transcript_path)}")
    else:
        if debug:
            print(f"        Warning: No transcript provided or found. Falling back to zero-text embedding.")
        b768 = np.zeros(768, dtype=np.float32)
        word_count = 0

    return ac_154d, w768, b768, word_count


def predict(audio_path=None, transcript_path=None, pid=None, use_ensemble=False, debug=False):
    """Runs end-to-end clinical depression severity prediction."""
    real_audio, real_transcript, extracted_pid = resolve_audio_and_transcript(audio_path, pid)

    if debug:
        print("==========================================================================")
        print("      DEPRESSION DETECTION CLINICAL INFERENCE ENGINE (DEBUG MODE)         ")
        print("==========================================================================")
        print(f" Audio File       : {real_audio}")
        print(f" Transcript File  : {real_transcript if real_transcript else 'NONE'}")
        print(f" Participant ID   : {extracted_pid if extracted_pid else 'UNKNOWN'}")
        mode_str = "Ensemble Blend (50% Cand 2A + 50% Cand 3A)" if use_ensemble else "Candidate 2A (Production Mild-Balanced MLP)"
        print(f" Inference Mode   : {mode_str}")
        print("--------------------------------------------------------------------------")

    # Load extractors with stdout suppression when not in debug mode
    with suppress_internal_logs(enabled=not debug):
        egemaps_ext = EgemapsExtractor(sr=16000)
        wav2vec_ext = Wav2VecExtractor(device="cpu")
        bert_ext = BertExtractor(device="cpu")

        ac_154, w2v_768, bert_768, word_count = extract_session_features(
            real_audio, real_transcript, egemaps_ext, wav2vec_ext, bert_ext, debug=debug
        )

        cal_2a, cal_3a, scalers = load_production_pipeline(use_ensemble=use_ensemble)

        # Scale features using training-only fitted standardizers
        ac_scaled = scalers['scaler_ac'].transform(ac_154.reshape(1, -1))
        w2v_scaled = scalers['scaler_w2v'].transform(w2v_768.reshape(1, -1))
        b_scaled = scalers['scaler_bert'].transform(bert_768.reshape(1, -1))
        fused_scaled = np.hstack([ac_scaled, w2v_scaled, b_scaled]).astype(np.float32)

        # Forward pass Candidate 2A
        with torch.no_grad():
            out_2a = cal_2a({'x_fused': torch.tensor(fused_scaled, dtype=torch.float32)})
            probs_2a = out_2a['severity_probs'].numpy()[0]
            phq_2a = float(out_2a['phq8_score'].item())

        if use_ensemble and cal_3a is not None:
            with torch.no_grad():
                out_3a = cal_3a({'x_fused': torch.tensor(fused_scaled, dtype=torch.float32)})
                probs_3a = out_3a['severity_probs'].numpy()[0]
                phq_3a = float(out_3a['phq8_score'].item())
            probs = 0.5 * probs_2a + 0.5 * probs_3a
            phq_pred = 0.5 * phq_2a + 0.5 * phq_3a
        else:
            probs = probs_2a
            phq_pred = phq_2a

    pred_class_id = int(np.argmax(probs))
    display_severity = SEVERITY_DISPLAY_NAMES[pred_class_id]
    confidence_pct = probs[pred_class_id] * 100.0
    raw_phq_clipped = max(0.0, min(24.0, phq_pred))

    # Unified Expected PHQ-8 Score derived from calibrated probability distribution
    CLASS_CONDITIONAL_MEANS = np.array([3.24096, 11.05882, 18.25000], dtype=np.float32)
    expected_phq = float(np.sum(probs * CLASS_CONDITIONAL_MEANS))
    expected_phq_clipped = max(0.0, min(24.0, expected_phq))
    is_depressed = bool(expected_phq_clipped >= 10.0 or pred_class_id >= 1)

    if not debug:
        # Clean Single-Result Output for Demonstration
        print("==========================================================================")
        print(f"Predicted severity: {display_severity}")
        print("--------------------------------------------------------------------------")
        print("Disclaimer: This output is an experimental screening estimate, not a clinical diagnosis.")
        print("==========================================================================")
    else:
        # Full Developer Diagnostic Suite
        print("\n==========================================================================")
        print("                      CLINICAL INFERENCE RESULT (DEBUG)                   ")
        print("==========================================================================")
        print(f" Predicted Severity Class : {SEVERITY_CATEGORIES[pred_class_id]}")
        print(f" Primary Classification   : Predicted severity: {display_severity}")
        print(f" Screening Classification : {'CLINICAL DEPRESSION INDICATED' if is_depressed else 'MINIMAL / NON-DEPRESSED'}")
        print(f" Expected PHQ-8 Score     : {expected_phq_clipped:.2f} / 24.0 (Consistent with Calibrated Probs)")
        print(f" Raw Auxiliary PHQ-8 Head : {raw_phq_clipped:.2f} / 24.0 (Auxiliary MSE linear output)")
        print(f" Model Confidence         : {confidence_pct:.2f}%")
        print("--------------------------------------------------------------------------")
        print(" Calibrated Probability Distribution:")
        print(f"   [0] Minimal/Mild (0-9)   : {probs[0]*100:.2f}%")
        print(f"   [1] Moderate (10-14)     : {probs[1]*100:.2f}%")
        print(f"   [2] Severe (15-24)       : {probs[2]*100:.2f}%")
        print("--------------------------------------------------------------------------")
        print(" Disclaimer: This output is an experimental screening estimate, not a clinical diagnosis.")
        print("==========================================================================")

        if extracted_pid is not None and extracted_pid in OFFICIAL_PHQ8_LABELS:
            gt_phq = OFFICIAL_PHQ8_LABELS[extracted_pid]
            gt_class = 0 if gt_phq <= 9.0 else (1 if gt_phq <= 14.0 else 2)
            print(f" Clinical Ground Truth Reference:")
            print(f"   Actual PHQ-8 Score        : {gt_phq:.1f}")
            print(f"   Actual Category           : {SEVERITY_CATEGORIES[gt_class]}")
            print(f"   Classification Match      : {'CORRECT' if pred_class_id == gt_class else 'MISCLASSIFIED'}")
            print(f"   Expected PHQ-8 Error      : {abs(expected_phq_clipped - gt_phq):.2f} pts")
            print(f"   Raw Regression Head Error : {abs(raw_phq_clipped - gt_phq):.2f} pts")
            print("==========================================================================")

    return {
        'participant_id': extracted_pid,
        'predicted_class_id': pred_class_id,
        'predicted_severity': display_severity,
        'expected_phq8': expected_phq_clipped,
        'raw_phq8': raw_phq_clipped,
        'probabilities': probs.tolist(),
        'is_depressed': is_depressed,
        'confidence_pct': confidence_pct
    }


def main():
    parser = argparse.ArgumentParser(
        description="Explainable Multi-Modal Speech & Language Depression Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio .wav file")
    parser.add_argument("--transcript", type=str, default=None, help="Path to input transcript .csv file")
    parser.add_argument("--pid", type=int, default=None, help="Participant ID (e.g. 624 or 300)")
    parser.add_argument("--ensemble", action="store_true", help="Use optional 50/50 blend of Candidate 2A and 3A")
    parser.add_argument("--debug", action="store_true", help="Show developer diagnostic metrics, probabilities, and logs")
    args = parser.parse_args()

    try:
        if args.audio is None and args.pid is None:
            # Default demo sample on PID 624 (severe clinical depression in unseen AI cohort)
            predict(pid=624, use_ensemble=args.ensemble, debug=args.debug)
        else:
            predict(audio_path=args.audio, transcript_path=args.transcript, pid=args.pid, use_ensemble=args.ensemble, debug=args.debug)
    except FileNotFoundError as e:
        print(f"\n[ERROR] File Not Found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Invalid Input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.debug:
            raise
        print(f"\n[ERROR] Inference Failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
