"""
Module: bert_extractor.py
Description: Participant-only speech text extraction and frozen BERT [CLS] embedding extractor.
Ensures strict speaker filtering (discards interviewer/Ellie speech) to prevent shortcut learning.
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class BertExtractor:
    """
    Extracts 768-dim textual representations from participant transcripts using frozen BERT.
    """

    def __init__(self, model_name="bert-base-uncased", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Strictly freeze BERT weights — do not fine-tune on small clinical dataset
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @staticmethod
    def extract_participant_text(transcript_path):
        """
        Parses DAIC-WOZ transcript CSV/TSV, filters strictly for participant utterances,
        and returns concatenated text and word count.
        """
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        # DAIC-WOZ transcripts are tab-separated
        try:
            df = pd.read_csv(transcript_path, sep='\t')
            if len(df.columns) <= 1:
                df = pd.read_csv(transcript_path, sep=',')
        except Exception:
            df = pd.read_csv(transcript_path, sep=None, engine='python')

        # Identify speaker and value columns
        speaker_col = None
        value_col = None
        for col in df.columns:
            c_low = col.strip().lower()
            if 'speaker' in c_low:
                speaker_col = col
            elif 'value' in c_low or 'text' in c_low or 'transcript' in c_low:
                value_col = col

        if speaker_col is None or value_col is None:
            if len(df.columns) >= 4:
                speaker_col = df.columns[2]
                value_col = df.columns[3]
            else:
                raise ValueError(f"Could not identify speaker/value columns in {transcript_path}: {df.columns}")

        # Filter strictly for Participant (case-insensitive)
        mask = df[speaker_col].astype(str).str.strip().str.lower() == 'participant'
        participant_df = df[mask]

        utterances = participant_df[value_col].dropna().astype(str).tolist()
        cleaned_utterances = []
        for u in utterances:
            u_clean = u.strip()
            # Remove filler formatting tags like <laughter>, <sigh> if present, but keep natural words
            u_clean = re.sub(r'<[^>]+>', ' ', u_clean)
            u_clean = ' '.join(u_clean.split())
            if u_clean:
                cleaned_utterances.append(u_clean)

        concatenated_text = " ".join(cleaned_utterances)
        words = concatenated_text.split()
        word_count = len(words)

        return concatenated_text, word_count

    def extract_from_text(self, text, max_length=512, stride=128):
        """
        Processes text through frozen BERT using a sliding window for long sessions,
        returning a 768-dim pooled [CLS] embedding (mean across chunk [CLS] tokens).
        """
        if not text or len(text.strip()) == 0:
            return np.zeros(768, dtype=np.float32)

        # Tokenize entire text into IDs
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            return np.zeros(768, dtype=np.float32)

        # Chunk tokens into windows of max_length - 2 (leaving room for [CLS] and [SEP])
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
        with torch.no_grad():
            for chunk in chunks:
                input_ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                attention_mask = torch.ones_like(input_tensor)

                outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
                # Pooled [CLS] token embedding at index 0 of last hidden state
                cls_rep = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                cls_embeddings.append(cls_rep)

        if len(cls_embeddings) == 0:
            return np.zeros(768, dtype=np.float32)

        pooled_vector = np.mean(cls_embeddings, axis=0).astype(np.float32)
        return pooled_vector

    def extract_from_transcript(self, transcript_path):
        """
        Full pipeline: extracts participant text from file and returns 768-dim embedding and word count.
        """
        text, word_count = self.extract_participant_text(transcript_path)
        feat_768d = self.extract_from_text(text)
        return feat_768d, word_count, text
