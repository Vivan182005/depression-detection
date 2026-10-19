"""
Module: experiments/research_v2/extractors_v2.py
Description: Canonical Feature Extractors v2 (Zero-Fallback Architecture).
Strictly enforces:
1. Wav2Vec2: facebook/wav2vec2-base-960h, 30s chunk-pooled mean embeddings.
   Raises RuntimeError immediately on any load/inference failure (ZERO random fallback).
2. Acoustic: Genuine OpenSMILE eGeMAPSv02 functionals (88d) + temporal 33d mean (33d) + std (33d) -> 154d.
   Raises RuntimeError on any failure. Asserts dims 62-87 are non-zero.
3. BERT: Frozen bert-base-uncased pooled [CLS] representations across sliding windows (510 tokens, stride 128).
   Normalized text representation (removes DAIC tags, lowercase, punctuation stripped).
   Raises ValueError on empty text (ZERO zero-vector returns).
4. Strict audio loading: 16 kHz mono, no peak normalisation, no inserted silence gaps.
"""

import os
import sys
import re
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

try:
    import opensmile
except ImportError:
    opensmile = None

from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoModel, AutoTokenizer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def normalize_text_for_bert(text: str) -> str:
    """
    Clinical text normalizer to eliminate transcription formatting discrepancies:
    - Removes DAIC transcript tags (<laughter>, <sigh>, [unin], xxx)
    - Lowercases all characters
    - Strips punctuation and excessive whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[[^\]]+\]', ' ', text)
    text = re.sub(r'\bxxx\b', ' ', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^\w\s']", ' ', text)
    text = ' '.join(text.split())
    return text

def load_canonical_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads audio file in canonical format:
    - Native dynamic range: NO peak normalisation to -1 dBFS.
    - Continuous speech: NO inserted artificial silence gaps.
    - Mono 16 kHz resampled via librosa.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    if len(y) == 0:
        raise ValueError(f"Loaded audio file is empty: {audio_path}")
    return y.astype(np.float32), sr

class Wav2Vec2ExtractorV2:
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)
            
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load Wav2Vec2 model '{model_name}': {e}. No fallback allowed.")

    def extract(self, y: np.ndarray, sr: int = 16000, chunk_sec: int = 30) -> np.ndarray:
        if y is None or len(y) == 0:
            raise ValueError("Empty audio signal passed to Wav2Vec2ExtractorV2.")
            
        chunk_size = chunk_sec * sr
        if len(y) <= chunk_size:
            chunks = [y]
        else:
            chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size) if len(y[i:i+chunk_size]) >= sr]
            if not chunks:
                chunks = [y]

        vecs = []
        try:
            with torch.no_grad():
                for c in chunks:
                    inputs = self.processor(c, sampling_rate=sr, return_tensors="pt", padding=True).to(self.device)
                    outputs = self.model(**inputs)
                    mean_pooled = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0).cpu().numpy()
                    vecs.append(mean_pooled)
        except Exception as e:
            raise RuntimeError(f"Wav2Vec2 inference failed: {e}. No fallback permitted.")

        if len(vecs) == 0:
            raise RuntimeError("Wav2Vec2 produced 0 chunk embeddings.")
            
        full_vec = np.mean(vecs, axis=0).astype(np.float32)
        if not np.isfinite(full_vec).all():
            raise RuntimeError("Wav2Vec2 produced non-finite embedding values.")
        if full_vec.shape != (768,):
            raise RuntimeError(f"Expected (768,) Wav2Vec2 vector, got {full_vec.shape}")
        return full_vec


# ---------------------------------------------------------
# Acoustic Robust Feature Subset (Task A2)
# Drops 19 amplitude/timing-dependent features from the 154-d vector:
# - 17 eGeMAPSv02 features (indices 0..87):
#     10-19: loudness_sma3_* (10 features)
#     81: loudnessPeaksPerSec
#     82: VoicedSegmentsPerSec
#     83: MeanVoicedSegmentLengthSec
#     84: StddevVoicedSegmentLengthSec
#     85: MeanUnvoicedSegmentLength
#     86: StddevUnvoicedSegmentLength
#     87: equivalentSoundLevel_dBp
# - 2 temporal features (indices 88..153):
#     120: temporal RMS mean
#     153: temporal RMS std
# Total remaining: 135 features (acoustic_robust_135d).
# ---------------------------------------------------------
DROPPED_ACOUSTIC_INDICES = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 81, 82, 83, 84, 85, 86, 87, 120, 153]
ROBUST_ACOUSTIC_MASK = np.array([i not in DROPPED_ACOUSTIC_INDICES for i in range(154)], dtype=bool)

def get_robust_acoustic_135d(ac_154: np.ndarray) -> np.ndarray:
    """Extracts the 135-d amplitude/timing-invariant robust acoustic feature subset from a 154-d vector."""
    if ac_154.shape != (154,):
        raise ValueError(f'Expected (154,) acoustic vector, got {ac_154.shape}')
    return ac_154[ROBUST_ACOUSTIC_MASK].astype(np.float32)

class AcousticExtractorV2:
    def __init__(self, sr: int = 16000):
        self.sr = sr
        if opensmile is None:
            raise RuntimeError("opensmile package is not available. AcousticExtractorV2 requires opensmile with eGeMAPSv02.")
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenSMILE eGeMAPSv02: {e}")

    def extract_egemaps_88d(self, y: np.ndarray) -> np.ndarray:
        try:
            df = self.smile.process_signal(y, self.sr)
            feat = df.values.flatten().astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"OpenSMILE extraction failed: {e}. No fallback permitted.")
            
        if len(feat) != 88:
            raise RuntimeError(f"Expected 88 eGeMAPS features, got {len(feat)}")
        if not np.isfinite(feat).all():
            raise RuntimeError("OpenSMILE produced non-finite feature values.")
        if np.all(feat[62:88] == 0):
            raise RuntimeError("CRITICAL: Dims 62-87 of eGeMAPS are identically zero! OpenSMILE failed.")
        return feat

    def extract_temporal_33d(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=16, hop_length=256, n_fft=512).T
            delta_mfcc = librosa.feature.delta(mfcc, axis=0)
            rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256).T
            temporal_frames = np.hstack([mfcc, delta_mfcc, rms])
            
            t_mean = np.mean(temporal_frames, axis=0).astype(np.float32)
            t_std = np.std(temporal_frames, axis=0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Temporal feature extraction failed: {e}")
            
        return t_mean, t_std

    def extract(self, y: np.ndarray) -> np.ndarray:
        egemaps_88 = self.extract_egemaps_88d(y)
        t_mean, t_std = self.extract_temporal_33d(y)
        ac_154 = np.hstack([egemaps_88, t_mean, t_std]).astype(np.float32)
        if ac_154.shape != (154,):
            raise RuntimeError(f"Expected (154,) acoustic vector, got {ac_154.shape}")
        return ac_154

    def extract_robust(self, y: np.ndarray) -> np.ndarray:
        ac_154 = self.extract(y)
        return get_robust_acoustic_135d(ac_154)

class BertExtractorV2:
    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model '{model_name}': {e}")

    def extract_from_text(self, text: str, max_length: int = 512, stride: int = 128) -> np.ndarray:
        norm_text = normalize_text_for_bert(text)
        if not norm_text or len(norm_text.strip()) == 0:
            raise ValueError("BERT text extraction failed: normalized text is empty. Zero vectors are prohibited.")
            
        tokens = self.tokenizer.encode(norm_text, add_special_tokens=False, verbose=False)
        if len(tokens) == 0:
            raise ValueError("BERT text extraction failed: tokenization produced 0 tokens.")
            
        window_size = max_length - 2
        chunks = []
        if len(tokens) <= window_size:
            chunks.append(tokens)
        else:
            step = window_size - stride
            for i in range(0, len(tokens), step):
                chunk = tokens[i : i + window_size]
                if len(chunk) > 0:
                    chunks.append(chunk)
                if i + window_size >= len(tokens):
                    break

        cls_embeddings = []
        try:
            with torch.no_grad():
                for chunk in chunks:
                    input_ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
                    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                    attention_mask = torch.ones_like(input_tensor)
                    outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
                    cls_rep = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                    cls_embeddings.append(cls_rep)
        except Exception as e:
            raise RuntimeError(f"BERT inference failed: {e}")

        if len(cls_embeddings) == 0:
            raise RuntimeError("BERT produced 0 chunk representations.")
            
        pooled_vec = np.mean(cls_embeddings, axis=0).astype(np.float32)
        expected = (self.model.config.hidden_size,)
        if pooled_vec.shape != expected:
            raise RuntimeError(f"Expected {expected} encoder vector, got {pooled_vec.shape}")
        return pooled_vec

    def extract_from_transcript_file(self, transcript_path: str, domain: str = 'woz') -> Tuple[np.ndarray, int, str]:
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
            
        try:
            df = pd.read_csv(transcript_path, sep='\t')
            if len(df.columns) <= 1:
                df = pd.read_csv(transcript_path, sep=',')
        except Exception:
            df = pd.read_csv(transcript_path, sep=None, engine='python')

        df.columns = [c.strip().lower() for c in df.columns]
        
        speaker_col = None
        for c in df.columns:
            if 'speaker' in c:
                speaker_col = c
                break
                
        text_col = None
        for c in df.columns:
            if c in ['value', 'text', 'transcript', 'utterance']:
                text_col = c
                break
        if text_col is None:
            if len(df.columns) >= 3:
                text_col = df.columns[2]
            else:
                text_col = df.columns[-1]

        if speaker_col is not None:
            mask = df[speaker_col].astype(str).str.strip().str.lower() == 'participant'
            part_df = df[mask]
        else:
            part_df = df

        utterances = part_df[text_col].dropna().astype(str).tolist()
        raw_text = ' '.join(utterances)
        norm_text = normalize_text_for_bert(raw_text)
        if not norm_text:
            raise ValueError(f"Transcript {transcript_path} has zero participant text.")
            
        vec = self.extract_from_text(norm_text)
        word_count = len(norm_text.split())
        return vec, word_count, norm_text
