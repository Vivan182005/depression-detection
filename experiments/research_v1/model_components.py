import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

BASE_DIR = '/Users/vivannaik/Desktop/CAPSTONE'
EXP_DIR = os.path.join(BASE_DIR, 'experiments', 'research_v1')
sys.path.insert(0, EXP_DIR)

from data_loader import get_development_dataset, get_stratified_group_folds
from metrics_utils import compute_all_metrics

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SimpleDataset(Dataset):
    def __init__(self, X, y_cls, y_phq, pids, sids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_phq = torch.tensor(y_phq, dtype=torch.float32)
        self.y_bin = torch.tensor((y_phq >= 10.0).astype(float), dtype=torch.float32)
        self.pids = pids
        self.sids = sids
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'class_id': self.y_cls[idx],
            'phq8_score': self.y_phq[idx],
            'binary_label': self.y_bin[idx],
            'participant_id': self.pids[idx],
            'session_id': self.sids[idx]
        }

class StandardMLP(nn.Module):
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
        
    def forward(self, x):
        feat = self.net(x)
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        bin_logit = self.head_bin(feat).squeeze(-1)
        return {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'phq': phq,
            'bin_logit': bin_logit,
            'bin_probs': torch.sigmoid(bin_logit)
        }

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal.mean()

# MAGNet Architecture
class MAGNetMLP(nn.Module):
    def __init__(self, ac_dim=154, w2v_dim=768, bert_dim=768, num_classes=3, dropout=0.3):
        super().__init__()
        # Audio pathway: combines acoustic (154d) and w2v (768d)
        self.audio_proj = nn.Sequential(
            nn.Linear(ac_dim + w2v_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Text pathway: BERT (768d)
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        # Residual fusion and classification
        self.fusion_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_class = nn.Linear(64, num_classes)
        self.head_phq = nn.Linear(64, 1)
        self.head_bin = nn.Linear(64, 1)
        
    def forward(self, x):
        x_ac = x[:, :154]
        x_w2v = x[:, 154:922]
        x_bert = x[:, 922:]
        
        h_audio = self.audio_proj(torch.cat([x_ac, x_w2v], dim=-1))
        h_text = self.text_proj(x_bert)
        
        gates = self.gate_net(torch.cat([h_audio, h_text], dim=-1)) # (B, 2)
        g_audio = gates[:, 0:1]
        g_text = gates[:, 1:2]
        
        h_fused = g_audio * h_audio + g_text * h_text # (B, 128)
        feat = self.fusion_fc(h_fused)
        
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        bin_logit = self.head_bin(feat).squeeze(-1)
        return {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'phq': phq,
            'bin_logit': bin_logit,
            'bin_probs': torch.sigmoid(bin_logit)
        }

def train_nn_model(model_cls, in_dim, X_train, y_tr_cls, y_tr_phq, pids_tr, sids_tr,
                   X_val, y_val_cls, y_val_phq, pids_val, sids_val,
                   loss_type='mild_balanced', focal_gamma=1.5, epochs=40, lr=1e-3, weight_decay=1e-3, seed=42):
    set_seed(seed)
    model = model_cls(in_dim=in_dim) if model_cls == StandardMLP else model_cls()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Class weights computed strictly from train fold
    counts = np.bincount(y_tr_cls, minlength=3).astype(float)
    if loss_type == 'unweighted':
        w = torch.ones(3, dtype=torch.float32)
        crit_cls = nn.CrossEntropyLoss(weight=w)
    elif loss_type == 'inverse':
        w = torch.tensor(len(y_tr_cls) / (3.0 * counts), dtype=torch.float32)
        crit_cls = nn.CrossEntropyLoss(weight=w)
    elif loss_type == 'mild_balanced':
        raw_w = 1.0 / np.sqrt(counts)
        w = torch.tensor(raw_w / np.mean(raw_w), dtype=torch.float32)
        crit_cls = nn.CrossEntropyLoss(weight=w)
    elif loss_type == 'focal':
        raw_w = 1.0 / np.sqrt(counts)
        w = torch.tensor(raw_w / np.mean(raw_w), dtype=torch.float32)
        crit_cls = FocalLoss(alpha=w, gamma=focal_gamma)
    elif loss_type == 'pure_regression':
        crit_cls = None
        
    crit_mse = nn.MSELoss()
    crit_bce = nn.BCEWithLogitsLoss()
    
    ds_train = SimpleDataset(X_train, y_tr_cls, y_tr_phq, pids_tr, sids_tr)
    loader_train = DataLoader(ds_train, batch_size=16, shuffle=True)
    
    best_macro_f1 = -1.0
    best_state = None
    best_preds = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader_train:
            optimizer.zero_grad()
            out = model(batch['x'])
            if loss_type == 'pure_regression':
                loss = crit_mse(out['phq'], batch['phq8_score'])
            else:
                l_ce = crit_cls(out['logits'], batch['class_id'])
                l_mse = crit_mse(out['phq'], batch['phq8_score'])
                l_bce = crit_bce(out['bin_logit'], batch['binary_label'])
                loss = l_ce + 0.1 * l_mse + 0.2 * l_bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            x_v = torch.tensor(X_val, dtype=torch.float32)
            out_v = model(x_v)
            if loss_type == 'pure_regression':
                pred_phq = out_v['phq'].numpy()
                pred_cls = np.array([0 if p <= 9.0 else (1 if p <= 14.0 else 2) for p in pred_phq])
                from sklearn.metrics import f1_score
                val_macro = f1_score(y_val_cls, pred_cls, average='macro', zero_division=0)
                probs_v = np.zeros((len(pred_cls), 3))
                for i, c in enumerate(pred_cls): probs_v[i, c] = 1.0
            else:
                probs_v = out_v['probs'].numpy()
                pred_cls = np.argmax(probs_v, axis=1)
                pred_phq = out_v['phq'].numpy()
                from sklearn.metrics import f1_score
                val_macro = f1_score(y_val_cls, pred_cls, average='macro', zero_division=0)
                
            if val_macro > best_macro_f1:
                best_macro_f1 = val_macro
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_preds = {
                    'pred_cls': pred_cls,
                    'probs': probs_v,
                    'pred_phq': pred_phq
                }
                
    if best_preds is None:
        model.eval()
        with torch.no_grad():
            out_v = model(torch.tensor(X_val, dtype=torch.float32))
            best_preds = {
                'pred_cls': np.argmax(out_v['probs'].numpy(), axis=1),
                'probs': out_v['probs'].numpy(),
                'pred_phq': out_v['phq'].numpy()
            }
            
    return best_preds, best_state

print('Script loaded successfully')
