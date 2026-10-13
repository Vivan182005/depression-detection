"""
Module: experiments/research_v2/models.py
Description: Neural architectures, training loops, in-fold conditional means,
and inner-validation temperature calibration for Phase 1.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from typing import Dict, Any, Tuple

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DepressionTorchDataset(Dataset):
    def __init__(self, X, y_cls, y_phq, pids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_phq = torch.tensor(y_phq, dtype=torch.float32)
        self.y_bin = torch.tensor((y_cls >= 1).astype(int), dtype=torch.float32)
        self.pids = pids
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'class_id': self.y_cls[idx],
            'phq8_score': self.y_phq[idx],
            'binary_label': self.y_bin[idx]
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
            'bin_logit': bin_logit
        }

class MAGNetMLP(nn.Module):
    def __init__(self, ac_dim=154, w2v_dim=768, bert_dim=768, num_classes=3, dropout=0.3):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(ac_dim + w2v_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
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
        
        gates = self.gate_net(torch.cat([h_audio, h_text], dim=-1))
        g_audio = gates[:, 0:1]
        g_text = gates[:, 1:2]
        
        h_fused = g_audio * h_audio + g_text * h_text
        feat = self.fusion_fc(h_fused)
        
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        bin_logit = self.head_bin(feat).squeeze(-1)
        return {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'phq': phq,
            'bin_logit': bin_logit
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

def fit_temperature_on_inner_val(logits: torch.Tensor, labels: torch.Tensor, seed: int = 42) -> float:
    """
    Fits temperature parameter T in [0.1, 5.0] on inner-validation logits using L-BFGS.
    """
    set_seed(seed)
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50)
    
    def eval_fn():
        optimizer.zero_grad()
        with torch.no_grad():
            temperature.clamp_(min=0.1, max=5.0)
        loss = crit(logits / temperature, labels)
        loss.backward()
        return loss
        
    try:
        optimizer.step(eval_fn)
        with torch.no_grad():
            temperature.clamp_(min=0.1, max=5.0)
        return float(temperature.item())
    except Exception:
        return 1.0

def train_nn_epoch_loop(model, loader, optimizer, loss_type, counts, focal_gamma=1.5, epochs=40, use_jitter=False, seed=42):
    counts_f = counts.astype(float)
    if loss_type == 'unweighted':
        crit_cls = nn.CrossEntropyLoss()
    elif loss_type == 'inverse':
        w = torch.tensor(np.sum(counts_f) / (3.0 * counts_f), dtype=torch.float32)
        crit_cls = nn.CrossEntropyLoss(weight=w)
    elif loss_type == 'mild_balanced':
        raw_w = 1.0 / np.sqrt(counts_f)
        w = torch.tensor(raw_w / np.mean(raw_w), dtype=torch.float32)
        crit_cls = nn.CrossEntropyLoss(weight=w)
    elif loss_type == 'focal':
        raw_w = 1.0 / np.sqrt(counts_f)
        w = torch.tensor(raw_w / np.mean(raw_w), dtype=torch.float32)
        crit_cls = FocalLoss(alpha=w, gamma=focal_gamma)
    elif loss_type == 'pure_regression':
        crit_cls = None
        
    crit_mse = nn.MSELoss()
    crit_bce = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    rng = np.random.RandomState(seed)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            x_b = batch['x'].clone()
            if use_jitter:
                # Controlled Gaussian jitter on acoustic features (first 154 dimensions)
                noise = torch.tensor(rng.normal(0.0, 0.02, size=(x_b.shape[0], min(154, x_b.shape[1]))), dtype=torch.float32)
                x_b[:, :noise.shape[1]] += noise
                
            optimizer.zero_grad()
            out = model(x_b)
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

def run_train_and_val_fold(
    model_cls,
    in_dim: int,
    X_tr: np.ndarray,
    y_tr_cls: np.ndarray,
    y_tr_phq: np.ndarray,
    pids_tr: np.ndarray,
    X_val: np.ndarray,
    y_val_cls: np.ndarray,
    y_val_phq: np.ndarray,
    pids_val: np.ndarray,
    loss_type: str = 'mild_balanced',
    focal_gamma: float = 1.5,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    use_jitter: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes one cross-validation fold:
    1. Computes in-fold class conditional means mu_train.
    2. Fits temperature T on an inner-validation split of the training fold.
    3. Trains final fold model on full training fold.
    4. Evaluates on outer validation fold.
    """
    set_seed(seed)
    
    # 1. In-fold class conditional means from training fold only
    mu_fold = np.zeros(3, dtype=np.float32)
    for c in range(3):
        mask = (y_tr_cls == c)
        mu_fold[c] = np.mean(y_tr_phq[mask]) if np.sum(mask) > 0 else np.mean(y_tr_phq)
        
    counts_tr = np.bincount(y_tr_cls, minlength=3)
    
    # 2. Inner-validation split of training fold to calibrate temperature T
    inner_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    inner_tr_idx, inner_val_idx = next(inner_sgkf.split(X_tr, y_tr_cls, groups=pids_tr))
    
    inner_model = model_cls(in_dim=in_dim) if model_cls == StandardMLP else model_cls()
    inner_opt = torch.optim.AdamW(inner_model.parameters(), lr=lr, weight_decay=weight_decay)
    inner_ds = DepressionTorchDataset(X_tr[inner_tr_idx], y_tr_cls[inner_tr_idx], y_tr_phq[inner_tr_idx], pids_tr[inner_tr_idx])
    inner_loader = DataLoader(inner_ds, batch_size=16, shuffle=True)
    inner_counts = np.bincount(y_tr_cls[inner_tr_idx], minlength=3)
    
    train_nn_epoch_loop(
        inner_model, inner_loader, inner_opt, loss_type, inner_counts,
        focal_gamma=focal_gamma, epochs=epochs, use_jitter=use_jitter, seed=seed
    )
    
    inner_model.eval()
    with torch.no_grad():
        x_in_val = torch.tensor(X_tr[inner_val_idx], dtype=torch.float32)
        in_val_out = inner_model(x_in_val)
        in_val_logits = in_val_out['logits']
        in_val_labels = torch.tensor(y_tr_cls[inner_val_idx], dtype=torch.long)
        
    calibrated_T = fit_temperature_on_inner_val(in_val_logits, in_val_labels, seed=seed)
    
    # 3. Train final fold model on full training fold
    final_model = model_cls(in_dim=in_dim) if model_cls == StandardMLP else model_cls()
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    full_tr_ds = DepressionTorchDataset(X_tr, y_tr_cls, y_tr_phq, pids_tr)
    full_tr_loader = DataLoader(full_tr_ds, batch_size=16, shuffle=True)
    
    train_nn_epoch_loop(
        final_model, full_tr_loader, final_opt, loss_type, counts_tr,
        focal_gamma=focal_gamma, epochs=epochs, use_jitter=use_jitter, seed=seed + 1
    )
    
    # 4. Predict on outer validation fold
    final_model.eval()
    with torch.no_grad():
        x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_out = final_model(x_val_tensor)
        raw_logits = val_out['logits'].cpu().numpy()
        raw_phq = val_out['phq'].cpu().numpy()
        
    calibrated_logits = raw_logits / calibrated_T
    # Stable softmax
    exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    if loss_type == 'pure_regression':
        # Ordinal derivation: tune cutoffs on training predictions
        with torch.no_grad():
            x_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
            tr_pred_phq = final_model(x_tr_tensor)['phq'].cpu().numpy()
            
        best_t1, best_t2 = 9.0, 14.0
        best_tr_mf1 = -1.0
        from sklearn.metrics import f1_score
        for t1 in np.linspace(4.0, 11.0, 15):
            for t2 in np.linspace(t1 + 1.0, 18.0, 15):
                tr_c = np.where(tr_pred_phq <= t1, 0, np.where(tr_pred_phq <= t2, 1, 2))
                mf1 = f1_score(y_tr_cls, tr_c, average='macro', zero_division=0)
                if mf1 > best_tr_mf1:
                    best_tr_mf1 = mf1
                    best_t1, best_t2 = t1, t2
                    
        pred_cls = np.where(raw_phq <= best_t1, 0, np.where(raw_phq <= best_t2, 1, 2))
        probs = np.zeros((len(pred_cls), 3), dtype=np.float32)
        for i, c in enumerate(pred_cls):
            probs[i, c] = 1.0
    else:
        pred_cls = np.argmax(probs, axis=1)
        
    expected_phq = np.sum(probs * mu_fold, axis=1)
    
    return {
        'logits': raw_logits,
        'calibrated_logits': calibrated_logits,
        'probs': probs,
        'pred_cls': pred_cls,
        'expected_phq': expected_phq,
        'raw_phq': raw_phq,
        'calibrated_T': calibrated_T,
        'mu_fold': mu_fold
    }
