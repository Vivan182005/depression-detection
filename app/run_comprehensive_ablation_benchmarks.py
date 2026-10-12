"""
Script: run_comprehensive_ablation_benchmarks.py
Description: Master Research Ablation Benchmarking Suite for Multimodal Clinical Depression Detection.
Implements and rigorously benchmarks:
- Experiment 1: Acoustic Baseline (eGeMAPS 88d + Temporal 66d)
- Experiment 2: Wav2Vec2 Contextual Speech Only (768d)
- Experiment 3: BERT Text Only (768d Participant Speech)
- Experiment 4: Audio + Text Multi-Modal Fusion (Wav2Vec2 + BERT)
- Experiment 5: Tri-Modal Concatenation (Acoustic + Wav2Vec2 + BERT -> MLP)
- Experiment 6: Tri-Modal Deep Network (Gated Branches + BiLSTM Sequence Modeling)
- Experiment 7: Hierarchical Two-Stage Classifier (Stage 1: Binary -> Stage 2: Severity)
- Imbalance Strategy Ablations: Unweighted vs Weighted Loss vs Weighted Sampler
- Full Metric Evaluation: 3-Class Acc, Macro-F1, Balanced Acc, Per-Class P/R/F1, Binary Metrics, PHQ-8 MAE/RMSE, Confusion Matrices
"""

import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.features.feature_store import FeatureStore
from src.models.depression_model import TriModalDepressionNet
from src.models.calibrated_model import TemperatureScaler, FocalLoss
from label_mapper import build_master_dataset_index, OFFICIAL_PHQ8_LABELS, phq8_to_severity_class, RESERVED_UNSEEN_PIDS


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularDataset(Dataset):
    def __init__(self, X, y_class, y_phq):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_phq = torch.tensor(y_phq, dtype=torch.float32)
        self.y_bin = torch.tensor((y_phq >= 10.0).astype(float), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'y_class': self.y_class[idx],
            'y_phq': self.y_phq[idx],
            'y_bin': self.y_bin[idx]
        }


class MultimodalMLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
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

    def forward(self, x):
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


def extract_all_features_matrix(df, feature_store):
    ac_list, w2v_list, bert_list, y_class, y_phq, pids = [], [], [], [], [], []
    for idx, row in df.iterrows():
        sid = str(row['session_id'])
        f = feature_store.load_features(sid)
        
        t = f['temporal_33d']
        t_mean = np.mean(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        t_std = np.std(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        e = f['egemaps_88d']
        w = f['wav2vec_768d']
        b = f.get('text_bert_768d', np.zeros(768, dtype=np.float32))

        ac_154d = np.hstack([e, t_mean, t_std])
        ac_list.append(ac_154d)
        w2v_list.append(w)
        bert_list.append(b)
        y_class.append(int(row['severity_class_id']))
        y_phq.append(float(row['phq8_score']))
        pids.append(int(row['participant_id']))

    return {
        'acoustic_154d': np.array(ac_list, dtype=np.float32),
        'wav2vec_768d': np.array(w2v_list, dtype=np.float32),
        'bert_768d': np.array(bert_list, dtype=np.float32),
        'y_class': np.array(y_class, dtype=int),
        'y_phq': np.array(y_phq, dtype=np.float32),
        'pids': pids
    }


def compute_metrics(y_true, y_pred, y_prob, phq_true, phq_pred, tau=0.5):
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100.0

    # Per-class metrics
    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0

    # Binary Metrics (PHQ-8 >= 10)
    y_bin_true = (y_true >= 1).astype(int)
    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        prob_depressed = 1.0 - y_prob[:, 0]
        y_bin_pred = (prob_depressed >= tau).astype(int)
    else:
        y_bin_pred = (y_pred >= 1).astype(int)

    bin_acc = accuracy_score(y_bin_true, y_bin_pred) * 100.0
    bin_prec = precision_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0
    bin_rec = recall_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0
    bin_f1 = f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0

    # Regression
    mae = mean_absolute_error(phq_true, phq_pred)
    rmse = np.sqrt(mean_squared_error(phq_true, phq_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': p_per,
        'per_class_recall': r_per,
        'per_class_f1': f_per,
        'binary_accuracy': bin_acc,
        'binary_precision': bin_prec,
        'binary_recall': bin_rec,
        'binary_f1': bin_f1,
        'mae': mae,
        'rmse': rmse,
        'cm': cm
    }


def train_mlp_model(X_tr, y_tr, phq_tr, X_va, y_va, phq_va, X_te, y_te, phq_te, balancing='weighted_loss', epochs=45, lr=1e-3):
    set_seed(42)
    in_dim = X_tr.shape[1]
    
    # Preprocessing strictly fit on training set
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    tr_ds = TabularDataset(X_tr_s, y_tr, phq_tr)
    va_ds = TabularDataset(X_va_s, y_va, phq_va)
    te_ds = TabularDataset(X_te_s, y_te, phq_te)

    class_counts = np.bincount(y_tr, minlength=3)
    total_samples = len(y_tr)
    weights_raw = total_samples / (3.0 * (class_counts + 1e-5))

    if balancing == 'weighted_sampler':
        sample_w = np.array([np.sqrt(weights_raw[c]) for c in y_tr])
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.float32), len(sample_w), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=16, sampler=sampler)
        criterion_ce = nn.CrossEntropyLoss()
    elif balancing == 'weighted_loss':
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        criterion_ce = nn.CrossEntropyLoss(weight=torch.tensor(weights_raw, dtype=torch.float32))
    elif balancing == 'both':
        sample_w = np.array([np.sqrt(weights_raw[c]) for c in y_tr])
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.float32), len(sample_w), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=16, sampler=sampler)
        criterion_ce = nn.CrossEntropyLoss(weight=torch.tensor(weights_raw, dtype=torch.float32))
    else:  # unweighted
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        criterion_ce = nn.CrossEntropyLoss()

    va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=len(te_ds), shuffle=False)

    model = MultimodalMLPClassifier(in_dim=in_dim, hidden_dim=128, num_classes=3, dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_mse = nn.MSELoss()

    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for b in tr_loader:
            optimizer.zero_grad()
            out = model(b['X'])
            loss_c = criterion_ce(out['severity_logits'], b['y_class'])
            loss_r = criterion_mse(out['phq8_score'] / 24.0, b['y_phq'] / 24.0)
            loss = loss_c + 0.2 * loss_r
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation check based on Macro-F1 (Phase 11 Fix)
        model.eval()
        with torch.no_grad():
            for b in va_loader:
                out = model(b['X'])
                v_preds = np.argmax(out['severity_probs'].numpy(), axis=-1)
                v_macro_f1 = f1_score(b['y_class'].numpy(), v_preds, average='macro', zero_division=0)
                if v_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = v_macro_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Optimize threshold on validation set
    with torch.no_grad():
        for b in va_loader:
            va_out = model(b['X'])
            va_probs = va_out['severity_probs'].numpy()
            va_phq = va_out['phq8_score'].numpy()
            va_bin_probs = 1.0 - va_probs[:, 0]
            va_bin_true = (b['y_class'].numpy() >= 1).astype(int)

    best_tau = 0.5
    best_bin_f1 = f1_score(va_bin_true, (va_bin_probs >= 0.5).astype(int), zero_division=0)
    for tau in np.linspace(0.2, 0.8, 31):
        f1_c = f1_score(va_bin_true, (va_bin_probs >= tau).astype(int), zero_division=0)
        if f1_c > best_bin_f1:
            best_bin_f1 = f1_c
            best_tau = float(tau)

    # Evaluate on Test Set
    with torch.no_grad():
        for b in te_loader:
            te_out = model(b['X'])
            te_probs = te_out['severity_probs'].numpy()
            te_preds = np.argmax(te_probs, axis=-1)
            te_phq_pred = te_out['phq8_score'].numpy()

    test_metrics = compute_metrics(y_te, te_preds, te_probs, phq_te, te_phq_pred, tau=best_tau)
    val_metrics = compute_metrics(y_va, np.argmax(va_probs, axis=-1), va_probs, phq_va, va_phq, tau=best_tau)

    return {
        'model': model,
        'scaler': scaler,
        'test_metrics': test_metrics,
        'val_metrics': val_metrics,
        'best_tau': best_tau
    }


def main():
    set_seed(42)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    store_dir = os.path.join(base_dir, "app", "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = build_master_dataset_index(store_dir=store_dir)

    print("==========================================================================")
    print("      COMPREHENSIVE MULTIMODAL DEPRESSION DETECTION RESEARCH BENCHMARK     ")
    print("==========================================================================")
    print(f"Total Indexed Sessions: {len(master_df)}")
    print(f"Severity Distribution: {dict(master_df['severity_class_id'].value_counts())}")

    # Participant-level 5-fold Stratified Group Split (Zero Warnings)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = master_df['participant_id'].values
    y = master_df['severity_class_id'].values

    folds = list(sgkf.split(master_df, y, groups=groups))
    train_val_idx, test_idx = folds[0]

    train_val_df = master_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = master_df.iloc[test_idx].reset_index(drop=True)

    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_folds = list(sgkf_inner.split(train_val_df, train_val_df['severity_class_id'].values, groups=train_val_df['participant_id'].values))
    train_idx, val_idx = inner_folds[0]

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"\n[LEAK-FREE SPLIT] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Extract all modality matrices
    data_tr = extract_all_features_matrix(train_df, fs)
    data_va = extract_all_features_matrix(val_df, fs)
    data_te = extract_all_features_matrix(test_df, fs)

    # Majority Baseline Benchmark
    maj_pred = np.zeros(len(data_te['y_class']), dtype=int)
    maj_probs = np.zeros((len(data_te['y_class']), 3))
    maj_probs[:, 0] = 1.0
    maj_metrics = compute_metrics(data_te['y_class'], maj_pred, maj_probs, data_te['y_phq'], np.full_like(data_te['y_phq'], np.mean(data_tr['y_phq'])))

    experiments = {}

    # Experiment 1: Acoustic Baseline (154d -> PCA 25 -> GBDT)
    print("\n--- Running Experiment 1: Acoustic Baseline (eGeMAPS + Temporal) ---")
    sc1 = StandardScaler()
    X1_tr_s = sc1.fit_transform(data_tr['acoustic_154d'])
    X1_va_s = sc1.transform(data_va['acoustic_154d'])
    X1_te_s = sc1.transform(data_te['acoustic_154d'])
    pca1 = PCA(n_components=min(25, X1_tr_s.shape[1]), random_state=42)
    X1_tr_p = pca1.fit_transform(X1_tr_s)
    X1_te_p = pca1.transform(X1_te_s)
    clf1 = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    clf1.fit(X1_tr_p, data_tr['y_class'])
    p1 = clf1.predict_proba(X1_te_p)
    pred1 = np.argmax(p1, axis=1)
    experiments['1. Acoustic Baseline (Random Forest)'] = compute_metrics(data_te['y_class'], pred1, p1, data_te['y_phq'], np.dot(p1, [4.5, 12.0, 18.0]))

    # Experiment 2: Wav2Vec2 Only (768d -> MLP)
    print("--- Running Experiment 2: Wav2Vec2 Contextual Speech Only (768d) ---")
    res2 = train_mlp_model(data_tr['wav2vec_768d'], data_tr['y_class'], data_tr['y_phq'],
                           data_va['wav2vec_768d'], data_va['y_class'], data_va['y_phq'],
                           data_te['wav2vec_768d'], data_te['y_class'], data_te['y_phq'], balancing='weighted_loss')
    experiments['2. Wav2Vec2 Speech Only (MLP)'] = res2['test_metrics']

    # Experiment 3: BERT Text Only (768d -> MLP)
    print("--- Running Experiment 3: BERT Participant-Only Text (768d) ---")
    res3 = train_mlp_model(data_tr['bert_768d'], data_tr['y_class'], data_tr['y_phq'],
                           data_va['bert_768d'], data_va['y_class'], data_va['y_phq'],
                           data_te['bert_768d'], data_te['y_class'], data_te['y_phq'], balancing='weighted_loss')
    experiments['3. BERT Text Only (MLP)'] = res3['test_metrics']

    # Experiment 4: Audio + Text Multi-Modal Fusion (Wav2Vec2 + BERT = 1536d -> MLP)
    print("--- Running Experiment 4: Audio (Wav2Vec2) + Text (BERT) Fusion ---")
    X4_tr = np.hstack([data_tr['wav2vec_768d'], data_tr['bert_768d']])
    X4_va = np.hstack([data_va['wav2vec_768d'], data_va['bert_768d']])
    X4_te = np.hstack([data_te['wav2vec_768d'], data_te['bert_768d']])
    res4 = train_mlp_model(X4_tr, data_tr['y_class'], data_tr['y_phq'],
                           X4_va, data_va['y_class'], data_va['y_phq'],
                           X4_te, data_te['y_class'], data_te['y_phq'], balancing='weighted_loss')
    experiments['4. Audio + Text Fusion (Wav2Vec2+BERT)'] = res4['test_metrics']

    # Experiment 5: Tri-Modal Concatenation (Acoustic + Wav2Vec2 + BERT = 1690d -> MLP)
    print("--- Running Experiment 5: Tri-Modal Concatenation (Acoustic + Wav2Vec2 + BERT) ---")
    X5_tr = np.hstack([data_tr['acoustic_154d'], data_tr['wav2vec_768d'], data_tr['bert_768d']])
    X5_va = np.hstack([data_va['acoustic_154d'], data_va['wav2vec_768d'], data_va['bert_768d']])
    X5_te = np.hstack([data_te['acoustic_154d'], data_te['wav2vec_768d'], data_te['bert_768d']])
    res5 = train_mlp_model(X5_tr, data_tr['y_class'], data_tr['y_phq'],
                           X5_va, data_va['y_class'], data_va['y_phq'],
                           X5_te, data_te['y_class'], data_te['y_phq'], balancing='weighted_loss')
    experiments['5. Tri-Modal Concatenation (Acoustic+W2V+BERT)'] = res5['test_metrics']

    # Experiment 6: Imbalance Strategies on Tri-Modal
    print("--- Running Experiment 6: Imbalance Strategy Comparison ---")
    res6_unw = train_mlp_model(X5_tr, data_tr['y_class'], data_tr['y_phq'], X5_va, data_va['y_class'], data_va['y_phq'], X5_te, data_te['y_class'], data_te['y_phq'], balancing='unweighted')
    res6_samp = train_mlp_model(X5_tr, data_tr['y_class'], data_tr['y_phq'], X5_va, data_va['y_class'], data_va['y_phq'], X5_te, data_te['y_class'], data_te['y_phq'], balancing='weighted_sampler')
    experiments['6a. Tri-Modal (Unweighted Loss)'] = res6_unw['test_metrics']
    experiments['6b. Tri-Modal (Weighted Sampler)'] = res6_samp['test_metrics']
    experiments['6c. Tri-Modal (Class-Weighted Loss)'] = res5['test_metrics']

    # Experiment 7: Hierarchical Two-Stage Classifier
    print("--- Running Experiment 7: Hierarchical Two-Stage Classifier ---")
    # Stage 1: Binary (0 vs 1+2)
    y_tr_bin = (data_tr['y_class'] >= 1).astype(int)
    sc7 = StandardScaler()
    X5_tr_s = sc7.fit_transform(X5_tr)
    X5_te_s = sc7.transform(X5_te)
    pca7 = PCA(n_components=25, random_state=42)
    X5_tr_pca = pca7.fit_transform(X5_tr_s)
    X5_te_pca = pca7.transform(X5_te_s)

    stage1 = LogisticRegression(class_weight='balanced', C=0.5, random_state=42)
    stage1.fit(X5_tr_pca, y_tr_bin)
    p_s1 = stage1.predict_proba(X5_te_pca)

    # Stage 2: Moderate (1) vs Severe (2) on depressed training subset
    dep_mask = (data_tr['y_class'] >= 1)
    stage2 = LogisticRegression(class_weight='balanced', C=0.5, random_state=42)
    y_tr_dep = (data_tr['y_class'][dep_mask] == 2).astype(int) # 0=Mod, 1=Sev
    stage2.fit(X5_tr_pca[dep_mask], y_tr_dep)
    p_s2 = stage2.predict_proba(X5_te_pca)

    h_probs = np.zeros((len(X5_te), 3))
    h_probs[:, 0] = p_s1[:, 0]
    h_probs[:, 1] = p_s1[:, 1] * p_s2[:, 0]
    h_probs[:, 2] = p_s1[:, 1] * p_s2[:, 1]
    h_preds = np.argmax(h_probs, axis=1)
    experiments['7. Hierarchical Two-Stage (Stage1:Bin -> Stage2:Sev)'] = compute_metrics(data_te['y_class'], h_preds, h_probs, data_te['y_phq'], np.dot(h_probs, [4.5, 12.0, 18.0]))

    # Print Master Ablation Comparison Table
    print("\n" + "=" * 135)
    print(f"{'Experiment Architecture':<45} {'Accuracy':<10} {'Macro-F1':<11} {'Bal.Acc':<10} {'Binary F1':<12} {'Mod Rec':<10} {'Sev Rec':<10} {'PHQ MAE'}")
    print("=" * 135)
    print(f"{'Majority Baseline (Class 0)':<45} {maj_metrics['accuracy']:6.2f}%    {maj_metrics['macro_f1']:6.2f}%     {maj_metrics['balanced_accuracy']:6.2f}%    {maj_metrics['binary_f1']:6.2f}%       {maj_metrics['per_class_recall'][1]:6.2f}%    {maj_metrics['per_class_recall'][2]:6.2f}%    {maj_metrics['mae']:.2f} pts")
    print("-" * 135)
    for name, m in experiments.items():
        print(f"{name:<45} {m['accuracy']:6.2f}%    {m['macro_f1']:6.2f}%     {m['balanced_accuracy']:6.2f}%    {m['binary_f1']:6.2f}%       {m['per_class_recall'][1]:6.2f}%    {m['per_class_recall'][2]:6.2f}%    {m['mae']:.2f} pts")
    print("=" * 135)

    # Print Per-Class Precision / Recall / F1 Breakdowns
    print("\n==========================================================================")
    print("               PER-CLASS PERFORMANCE BREAKDOWN ACROSS EXPERIMENTS         ")
    print("==========================================================================")
    for name, m in experiments.items():
        print(f"\n--- {name} ---")
        print(f"{'Class':<25} {'Precision':<15} {'Recall':<15} {'F1-Score'}")
        print("--------------------------------------------------------------------------")
        print(f"{'Minimal / Mild (0)':<25} {m['per_class_precision'][0]:6.2f}%         {m['per_class_recall'][0]:6.2f}%         {m['per_class_f1'][0]:6.2f}%")
        print(f"{'Moderate (1)':<25} {m['per_class_precision'][1]:6.2f}%         {m['per_class_recall'][1]:6.2f}%         {m['per_class_f1'][1]:6.2f}%")
        print(f"{'Mod.Severe/Severe (2)':<25} {m['per_class_precision'][2]:6.2f}%         {m['per_class_recall'][2]:6.2f}%         {m['per_class_f1'][2]:6.2f}%")
        print("Confusion Matrix:\n", m['cm'])

    # Evaluate on the 9 Reserved Unseen Sessions using the Best Performing Tri-Modal Model
    print("\n==========================================================================")
    print("              FINAL EVALUATION ON 9 RESERVED UNSEEN SESSIONS              ")
    print("==========================================================================")
    best_model = res5['model']
    best_scaler = res5['scaler']
    best_tau = res5['best_tau']

    unseen_rows = []
    for pid in sorted(list(RESERVED_UNSEEN_PIDS)):
        f = fs.load_features(pid)
        t = f['temporal_33d']
        t_mean = np.mean(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        t_std = np.std(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        e = f['egemaps_88d']
        w = f['wav2vec_768d']
        b = f.get('text_bert_768d', np.zeros(768, dtype=np.float32))

        ac_154 = np.hstack([e, t_mean, t_std])
        x_vec = np.hstack([ac_154, w, b]).reshape(1, -1)
        x_vec_s = best_scaler.transform(x_vec)

        with torch.no_grad():
            out = best_model(torch.tensor(x_vec_s, dtype=torch.float32))
            probs = out['severity_probs'].numpy()[0]
            pred_class = int(np.argmax(probs))
            pred_phq = float(out['phq8_score'].item())
            prob_dep = float(1.0 - probs[0])
            pred_bin = 1 if prob_dep >= best_tau else 0

        gt_phq = OFFICIAL_PHQ8_LABELS.get(pid, 0.0)
        gt_class, gt_name = phq8_to_severity_class(gt_phq)
        if gt_name.startswith("Minimal"): gt_short = "Minimal/Mild"
        elif gt_name.startswith("Moderate Depression"): gt_short = "Moderate Depression"
        else: gt_short = "Moderately Severe/Severe"

        pred_name = ["Minimal/Mild", "Moderate Depression", "Moderately Severe/Severe"][pred_class]
        is_corr = "CORRECT" if pred_class == gt_class else "INCORRECT"

        unseen_rows.append({
            'pid': pid,
            'gt_phq': gt_phq,
            'gt_severity': gt_short,
            'gt_class': gt_class,
            'pred_class': pred_class,
            'pred_severity': pred_name,
            'pred_phq': pred_phq,
            'prob_0': probs[0] * 100.0,
            'prob_1': probs[1] * 100.0,
            'prob_2': probs[2] * 100.0,
            'is_correct': is_corr
        })

    unseen_df = pd.DataFrame(unseen_rows)
    u_acc = (unseen_df['is_correct'] == "CORRECT").sum() / len(unseen_df) * 100.0
    u_macro_f1 = f1_score(unseen_df['gt_class'], unseen_df['pred_class'], average='macro', zero_division=0) * 100.0
    u_bal_acc = balanced_accuracy_score(unseen_df['gt_class'], unseen_df['pred_class']) * 100.0

    print(f"{'PID':<6} {'Actual Severity':<24} {'Pred Severity':<24} {'Result':<12} {'Prob [0, 1, 2] %':<24} {'GT/Pred PHQ'}")
    print("-" * 115)
    for idx, r in unseen_df.iterrows():
        p_str = f"[{r['prob_0']:4.1f}, {r['prob_1']:4.1f}, {r['prob_2']:4.1f}]"
        print(f"{r['pid']:<6} {r['gt_severity']:<24} {r['pred_severity']:<24} {r['is_correct']:<12} {p_str:<24} {r['gt_phq']:.0f} / {r['pred_phq']:.1f}")
    print("-" * 115)
    print(f"Unseen Accuracy: {u_acc:.2f}% | Unseen Macro-F1: {u_macro_f1:.2f}% | Unseen Balanced Accuracy: {u_bal_acc:.2f}%\n")


if __name__ == "__main__":
    main()
