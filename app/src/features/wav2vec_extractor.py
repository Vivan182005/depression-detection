"""
Module: wav2vec_extractor.py
Description: Deep Self-Supervised Speech Embedding Extractor using Wav2Vec 2.0 / WavLM.
Extracts 768-dimensional contextual frame embeddings from raw audio chunks.
"""

import numpy as np

try:
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class Wav2VecExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        self.device = device or ("cuda" if HAS_TRANSFORMERS and torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        if HAS_TRANSFORMERS:
            try:
                print(f"[INFO] Loading pre-trained Wav2Vec2 model: {model_name}...")
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
                self.model.eval()
                self.has_model = True
            except Exception as e:
                print(f"[WARNING] Could not download/load Wav2Vec2 model '{model_name}': {e}. Using fallback vector generator.")
                self.has_model = False
        else:
            self.has_model = False

    def extract_embedding(self, audio_chunk, sr=16000):
        """
        Extracts 768-dimensional mean-pooled Wav2Vec2 representation for a 3-second audio chunk.

        Parameters:
            audio_chunk (np.ndarray): 1D float array of raw audio samples
            sr (int): Sample rate (16000 Hz)

        Returns:
            np.ndarray: 768-dimensional vector
        """
        if not self.has_model:
            # Deterministic fallback vector generator
            np.random.seed(int(np.sum(np.abs(audio_chunk[:100])) * 10000) % 4294967295)
            dummy_vec = np.random.randn(768).astype(np.float32)
            energy = np.sqrt(np.mean(audio_chunk**2) + 1e-9)
            return dummy_vec * (energy + 0.1)

        with torch.no_grad():
            inputs = self.processor(
                audio_chunk, sampling_rate=sr, return_tensors="pt", padding=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # Last hidden state shape: (batch_size, sequence_length, 768)
            hidden_states = outputs.last_hidden_state
            
            # Mean pool across time dimension -> (768,)
            mean_pooled = torch.mean(hidden_states, dim=1).squeeze(0).cpu().numpy()
            return mean_pooled.astype(np.float32)
