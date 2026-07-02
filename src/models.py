import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
except (ImportError, Exception):
    XGBClassifier = None
import numpy as np


class LSTMAttentionClassifier(nn.Module):
    """
    LSTM model with a Bahdanau-style Attention mechanism for sequential speech features.
    Outputs:
        logits: Depression prediction score before sigmoid.
        attention_weights: Weights assigned to each frame in the sequence.
    """
    def __init__(self, input_dim=33, hidden_dim=64, lstm_layers=1, bidirectional=True):
        super(LSTMAttentionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention layer components
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention_w = nn.Linear(lstm_out_dim, lstm_out_dim)
        self.attention_v = nn.Linear(lstm_out_dim, 1, bias=False)
        
        # Classification layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        # LSTM output shape: (batch_size, seq_len, lstm_out_dim)
        lstm_out, _ = self.lstm(x)
        
        # Bahdanau Attention
        # Score = v^T * tanh(W_h * h_t)
        attn_scores = torch.tanh(self.attention_w(lstm_out))  # (batch_size, seq_len, lstm_out_dim)
        attn_scores = self.attention_v(attn_scores)  # (batch_size, seq_len, 1)
        
        # Softmax over sequence length
        attention_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Context vector: sum(alpha_t * h_t)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, lstm_out_dim)
        
        # Classification
        logits = self.fc(context_vector)  # (batch_size, 1)
        
        return logits, attention_weights.squeeze(-1)


# Helper functions to build machine learning baselines
def build_svm_classifier(C=1.0, kernel='rbf', probability=True):
    """Returns a scikit-learn Support Vector Machine classifier."""
    return SVC(C=C, kernel=kernel, probability=probability, random_state=42)


def build_xgboost_classifier(n_estimators=100, max_depth=5, learning_rate=0.1):
    """Returns an XGBoost classifier, or None if unavailable."""
    if XGBClassifier is None:
        print("Warning: XGBClassifier is unavailable (likely due to missing OpenMP library).")
        return None
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        eval_metric='logloss'
    )
