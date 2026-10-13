"""
Comprehensive Diagnostic and Modality Ablation Suite (Phase 5).
Benchmarks:
0. Majority Class Baseline
1. Acoustic Only (154d)
2. Wav2Vec2 Only (768d)
3. BERT Only (768d)
4. Acoustic + Wav2Vec2 (922d)
5. Acoustic + BERT (922d)
6. Wav2Vec2 + BERT (1536d)
7. Full Tri-Modal Baseline (1690d, Legacy Labels)
8. Full Tri-Modal (1690d, Verified Ground-Truth Labels)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP_DIR = os.path.join(BASE_DIR, "app")
EXP_DIR = os.path.join(BASE_DIR, "experiments", "capstone_improvement_v1")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, "src"))

from src.features.feature_store import FeatureStore
from src.models.train_leak_free import split_joint_stratified


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
            'x': self.X[idx],
            'class_id': self.y_class[idx],
            'phq8_score': self.y_phq[idx],
            'binary_label': self.y_bin[idx]
        }


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_classes=3, dropout=0.3):
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

    def forward(self, x):
        feat = self.net(x)
        logits = self.head_class(feat)
        phq = self.head_phq(feat).squeeze(-1)
        return {
            'severity_logits': logits,
            'severity_probs': F.softmax(logits, dim=-1),
            'phq8_score': phq
        }


def extract_features_matrix(df, feature_store, use_clean_labels=False):
    # Load true labels if requested
    if use_clean_labels:
        clean_df = pd.read_csv(os.path.join(BASE_DIR, "new_audio_csv", "detailed_lables.csv"))
        label_map = dict(zip(clean_df['Participant'], clean_df['Depression_severity']))
    else:
        label_map = None

    ac_list, w2v_list, bert_list, y_class, y_phq, pids = [], [], [], [], [], []

    for idx, row in df.iterrows():
        sid = str(row['session_id'])
        pid = int(row['participant_id'])
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

        if use_clean_labels and pid in label_map:
            phq = float(label_map[pid])
            cls_id = 0 if phq <= 9 else (1 if phq <= 14 else 2)
        else:
            phq = float(row['phq8_score'])
            cls_id = int(row['severity_class_id'])

        y_class.append(cls_id)
        y_phq.append(phq)
        pids.append(pid)

    return {
        'acoustic_154d': np.array(ac_list, dtype=np.float32),
        'wav2vec_768d': np.array(w2v_list, dtype=np.float32),
        'bert_768d': np.array(bert_list, dtype=np.float32),
        'y_class': np.array(y_class, dtype=int),
        'y_phq': np.array(y_phq, dtype=np.float32),
        'pids': pids
    }


def compute_metrics(y_true, y_pred, y_prob, phq_true, phq_pred):
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0)
    p_per = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    r_per = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0
    f_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0) * 100.0

    y_bin_true = (y_true >= 1).astype(int)
    prob_depressed = 1.0 - y_prob[:, 0]
    y_bin_pred = (prob_depressed >= 0.5).astype(int)
    bin_f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0) * 100.0)

    mae = float(mean_absolute_error(phq_true, phq_pred))
    rmse = float(np.sqrt(mean_squared_error(phq_true, phq_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'binary_f1': bin_f1,
        'mae': mae,
        'rmse': rmse,
        'per_class_recall': [float(x) for x in r_per],
        'per_class_f1': [float(x) for x in f_per],
        'confusion_matrix': cm
    }


def train_eval_mlp(X_tr, y_tr, phq_tr, X_va, y_va, phq_va, X_te, y_te, phq_te, epochs=35, lr=1e-3, seed=42):
    set_seed(seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    tr_ds = TabularDataset(X_tr_s, y_tr, phq_tr)
    va_ds = TabularDataset(X_va_s, y_va, phq_va)
    te_ds = TabularDataset(X_te_s, y_te, phq_te)

    tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=len(va_ds), shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=len(te_ds), shuffle=False)

    counts = np.bincount(y_tr, minlength=3)
    weights = len(y_tr) / (3.0 * (counts + 1e-5))
    criterion_ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    criterion_mse = nn.MSELoss()

    model = SimpleMLP(in_dim=X_tr.shape[1], hidden_dim=128, num_classes=3, dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tr_loader:
            optimizer.zero_grad()
            out = model(batch['x'])
            l_ce = criterion_ce(out['severity_logits'], batch['class_id'])
            l_mse = criterion_mse(out['phq8_score'], batch['phq8_score'])
            loss = l_ce + 0.1 * l_mse
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in va_loader:
                out = model(batch['x'])
                v_prob = out['severity_probs'].numpy()
                v_pred = np.argmax(v_prob, axis=-1)
                v_f1 = f1_score(batch['class_id'].numpy(), v_pred, average='macro', zero_division=0)
                if v_f1 > best_val_f1:
                    best_val_f1 = v_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        for batch in te_loader:
            out = model(batch['x'])
            probs = out['severity_probs'].numpy()
            preds = np.argmax(probs, axis=-1)
            phq_pred = out['phq8_score'].numpy()
            y_true = batch['class_id'].numpy()
            phq_true = batch['phq8_score'].numpy()

    return compute_metrics(y_true, preds, probs, phq_true, phq_pred)


def main():
    print("==========================================================================")
    print("            PHASE 5: MODALITY ABLATIONS & DIAGNOSTICS SUITE               ")
    print("==========================================================================")

    store_dir = os.path.join(APP_DIR, "feature_store")
    fs = FeatureStore(store_dir=store_dir)
    master_df = pd.read_csv(os.path.join(store_dir, "master_dataset_index.csv"))
    train_df, val_df, test_df = split_joint_stratified(master_df, verbose=False)

    print("\n--- Extracting Features for Legacy vs Clean Ground-Truth ---")
    data_legacy_tr = extract_features_matrix(train_df, fs, use_clean_labels=False)
    data_legacy_va = extract_features_matrix(val_df, fs, use_clean_labels=False)
    data_legacy_te = extract_features_matrix(test_df, fs, use_clean_labels=False)

    data_clean_tr = extract_features_matrix(train_df, fs, use_clean_labels=True)
    data_clean_va = extract_features_matrix(val_df, fs, use_clean_labels=True)
    data_clean_te = extract_features_matrix(test_df, fs, use_clean_labels=True)

    experiments = []

    # 0. Majority Baseline
    y_te = data_legacy_te['y_class']
    maj_c = int(pd.Series(data_legacy_tr['y_class']).mode()[0])
    maj_preds = np.full_like(y_te, maj_c)
    maj_prob = np.zeros((len(y_te), 3))
    maj_prob[:, maj_c] = 1.0
    phq_mean_tr = np.mean(data_legacy_tr['y_phq'])
    maj_phq_pred = np.full_like(data_legacy_te['y_phq'], phq_mean_tr)
    m0 = compute_metrics(y_te, maj_preds, maj_prob, data_legacy_te['y_phq'], maj_phq_pred)
    m0['experiment'] = '0. Majority Class Baseline'
    experiments.append(m0)

    # 1. Acoustic Only (154d)
    m1 = train_eval_mlp(
        data_legacy_tr['acoustic_154d'], data_legacy_tr['y_class'], data_legacy_tr['y_phq'],
        data_legacy_va['acoustic_154d'], data_legacy_va['y_class'], data_legacy_va['y_phq'],
        data_legacy_te['acoustic_154d'], data_legacy_te['y_class'], data_legacy_te['y_phq']
    )
    m1['experiment'] = '1. Acoustic Only (154d)'
    experiments.append(m1)

    # 2. Wav2Vec2 Only (768d)
    m2 = train_eval_mlp(
        data_legacy_tr['wav2vec_768d'], data_legacy_tr['y_class'], data_legacy_tr['y_phq'],
        data_legacy_va['wav2vec_768d'], data_legacy_va['y_class'], data_legacy_va['y_phq'],
        data_legacy_te['wav2vec_768d'], data_legacy_te['y_class'], data_legacy_te['y_phq']
    )
    m2['experiment'] = '2. Wav2Vec2 Only (768d)'
    experiments.append(m2)

    # 3. BERT Only (768d)
    m3 = train_eval_mlp(
        data_legacy_tr['bert_768d'], data_legacy_tr['y_class'], data_legacy_tr['y_phq'],
        data_legacy_va['bert_768d'], data_legacy_va['y_class'], data_legacy_va['y_phq'],
        data_legacy_te['bert_768d'], data_legacy_te['y_class'], data_legacy_te['y_phq']
    )
    m3['experiment'] = '3. BERT Only (768d)'
    experiments.append(m3)

    # 4. Acoustic + Wav2Vec2 (922d)
    X_ac_w2v_tr = np.hstack([data_legacy_tr['acoustic_154d'], data_legacy_tr['wav2vec_768d']])
    X_ac_w2v_va = np.hstack([data_legacy_va['acoustic_154d'], data_legacy_va['wav2vec_768d']])
    X_ac_w2v_te = np.hstack([data_legacy_te['acoustic_154d'], data_legacy_te['wav2vec_768d']])
    m4 = train_eval_mlp(
        X_ac_w2v_tr, data_legacy_tr['y_class'], data_legacy_tr['y_phq'],
        X_ac_w2v_va, data_legacy_va['y_class'], data_legacy_va['y_phq'],
        X_ac_w2v_te, data_legacy_te['y_class'], data_legacy_te['y_phq']
    )
    m4['experiment'] = '4. Acoustic + Wav2Vec2 (922d)'
    experiments.append(m4)

    # 5. Acoustic + BERT (922d)
    X_ac_bert_tr = np.hstack([data_legacy_tr['acoustic_154d'], data_legacy_tr['bert_768d']])
    X_ac_bert_va = np.hstack([data_legacy_va['acoustic_154d'], data_legacy_va['bert_768d']])
    X_ac_bert_te = np.hstack([data_legacy_te['acoustic_154d'], data_legacy_te['bert_768d']])
    m5 = train_eval_mlp(
        X_ac_bert_tr, data_legacy_tr['y_class'], data_legacy_tr['y_phq'],
        X_ac_bert_va, data_legacy_va['y_class'], data_legacy_va['y_phq'],
        X_ac_bert_te, data_legacy_te['y_class'], data_legacy_te['y_phq']
    )
    m5['experiment'] = '5. Acoustic + BERT (922d)'
    experiments.append(m5)

    # 6. Wav2Vec2 + BERT (1536d)
    X_w2v_bert_tr = np.hstack([data_legacy_tr['wav2vec_768d'], data_legacy_tr['bert_768d']])
    X_w2v_bert_va = np.hstack([data_legacy_va['wav2vec_768d'], data_legacy_va['bert_768d']])
    X_w2v_bert_te = np.hstack([data_legacy_te['wav2vec_768d'], data_legacy_te['bert_768d']])
    m6 = train_eval_mlp(
        X_w2v_bert_tr, data_legacy_tr['y_class'], data_legacy_tr['y_phq'],
        X_w2v_bert_va, data_legacy_va['y_class'], data_legacy_va['y_phq'],
        X_w2v_bert_te, data_legacy_te['y_class'], data_legacy_te['y_phq']
    )
    m6['experiment'] = '6. Wav2Vec2 + BERT (1536d)'
    experiments.append(m6)

    # 7. Full Tri-Modal (1690d, Legacy Labels)
    X_tri_tr = np.hstack([data_legacy_tr['acoustic_154d'], data_legacy_tr['wav2vec_768d'], data_legacy_tr['bert_768d']])
    X_tri_va = np.hstack([data_legacy_va['acoustic_154d'], data_legacy_va['wav2vec_768d'], data_legacy_va['bert_768d']])
    X_tri_te = np.hstack([data_legacy_te['acoustic_154d'], data_legacy_te['wav2vec_768d'], data_legacy_te['bert_768d']])
    m7 = train_eval_mlp(
        X_tri_tr, data_legacy_tr['y_class'], data_legacy_tr['y_phq'],
        X_tri_va, data_legacy_va['y_class'], data_legacy_va['y_phq'],
        X_tri_te, data_legacy_te['y_class'], data_legacy_te['y_phq']
    )
    m7['experiment'] = '7. Tri-Modal Baseline (1690d, Legacy Labels)'
    experiments.append(m7)

    # 8. Full Tri-Modal (1690d, Clean Verified Ground-Truth Labels)
    X_tri_c_tr = np.hstack([data_clean_tr['acoustic_154d'], data_clean_tr['wav2vec_768d'], data_clean_tr['bert_768d']])
    X_tri_c_va = np.hstack([data_clean_va['acoustic_154d'], data_clean_va['wav2vec_768d'], data_clean_va['bert_768d']])
    X_tri_c_te = np.hstack([data_clean_te['acoustic_154d'], data_clean_te['wav2vec_768d'], data_clean_te['bert_768d']])
    m8 = train_eval_mlp(
        X_tri_c_tr, data_clean_tr['y_class'], data_clean_tr['y_phq'],
        X_tri_c_va, data_clean_va['y_class'], data_clean_va['y_phq'],
        X_tri_c_te, data_clean_te['y_class'], data_clean_te['y_phq']
    )
    m8['experiment'] = '8. Tri-Modal (1690d, Clean Verified Ground-Truth)'
    experiments.append(m8)

    print("\n" + "="*115)
    print(f"{'Experiment Architecture':45s} | {'Accuracy':9s} | {'Macro-F1':9s} | {'Bal-Acc':9s} | {'Bin-F1':9s} | {'PHQ MAE':9s}")
    print("="*115)
    for exp in experiments:
        print(f"{exp['experiment']:45s} | {exp['accuracy']:6.2f}%   | {exp['macro_f1']:6.2f}%   | {exp['balanced_accuracy']:6.2f}%   | {exp['binary_f1']:6.2f}%   | {exp['mae']:6.2f} pts")
    print("="*115)

    # Save results table
    out_csv = os.path.join(EXP_DIR, "modality_ablation_results.csv")
    df_res = pd.DataFrame(experiments)
    df_res.to_csv(out_csv, index=False)
    print(f"\nAblation metrics saved to: {out_csv}")


if __name__ == "__main__":
    main()
