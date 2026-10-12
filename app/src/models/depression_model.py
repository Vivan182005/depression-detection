"""
Module: depression_model.py
Description: Production Tri-Modal Multimodal MLP Architecture for Depression Detection.
Fuses Acoustic Biomarkers (154d), Pretrained Frozen Wav2Vec 2.0 (768d), and Pretrained Frozen BERT (768d).
Optimized for small clinical cohorts (~183 sessions) with LayerNorm regularization and multi-task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_TORCH = True


class CompactTriModalMLP(nn.Module):
    """
    Compact Tri-Modal Multimodal MLP Classifier (Primary Model B):
    - Input: Fused 1690-dim vector (154d Acoustic + 768d Wav2Vec2 + 768d BERT)
    - Layers: Linear(1690, 128) -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> LayerNorm -> ReLU -> Dropout(0.3)
    - Heads:
      1. 3-Class Categorical Severity Softmax (Minimal/Mild, Moderate, Severe)
      2. Continuous PHQ-8 Regression Head
      3. Binary Clinical Head (PHQ-8 >= 10)
    """

    def __init__(
        self,
        in_dim=1690,
        hidden_dim=128,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head_class = nn.Linear(64, num_classes)
        self.head_phq = nn.Linear(64, 1)
        self.head_bin = nn.Linear(64, 1)

    def forward(self, egemaps, temporal=None, wav2vec=None, text_bert=None):
        # Support both forward(x_fused) and forward(egemaps, temporal, wav2vec, text_bert)
        if temporal is None and wav2vec is None and text_bert is None:
            x = egemaps
        else:
            if temporal.dim() == 3:
                t_mean = temporal.mean(dim=1)
                t_std = temporal.std(dim=1)
                ac = torch.cat([egemaps, t_mean, t_std], dim=-1)
            elif temporal.dim() == 2:
                ac = torch.cat([egemaps, temporal], dim=-1)
            else:
                ac = egemaps
            
            if wav2vec is None:
                wav2vec = torch.zeros((ac.shape[0], 768), device=ac.device, dtype=ac.dtype)
            if text_bert is None:
                text_bert = torch.zeros((ac.shape[0], 768), device=ac.device, dtype=ac.dtype)

            x = torch.cat([ac, wav2vec, text_bert], dim=-1)

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


# Alias TriModalDepressionNet to the production CompactTriModalMLP
TriModalDepressionNet = CompactTriModalMLP
DualBranchDepressionNet = CompactTriModalMLP
