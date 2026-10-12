"""
Script: train_pipeline.py
Description: Step 4 - Clean Model Training Engine for 5 Clinical Severity Categories.
Uses DualBranchDepressionNet with Raw Logits CrossEntropyLoss + MSE Loss,
Post-Training Temperature Scaling Calibration, and Detailed Epoch Diagnostics:
- Gradient Norms
- Class Probability Distributions
- Confusion Matrix
- Per-Class Prediction Counts
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.feature_store import FeatureStore
from src.models.depression_model import DualBranchDepressionNet, HAS_TORCH

if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import confusion_matrix


SEVERITY_NAMES = [
    "Minimal / Normal",
    "Mild Depression",
    "Moderate Depression",
    "Moderately Severe",
    "Severe Depression"
]


class FeatureStoreDataset(Dataset if HAS_TORCH else object):
    def __init__(self, master_df, feature_store):
        self.master_df = master_df.reset_index(drop=True)
        self.feature_store = feature_store

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx):
        row = self.master_df.iloc[idx]
        session_id = row['session_id']
        features = self.feature_store.load_features(session_id)

        egemaps_88d = features['egemaps_88d']
        temporal_33d = features['temporal_33d']
        wav2vec_768d = features['wav2vec_768d']

        return {
            'session_id': session_id,
            'egemaps': torch.tensor(egemaps_88d, dtype=torch.float32),
            'temporal': torch.tensor(temporal_33d, dtype=torch.float32),
            'wav2vec': torch.tensor(wav2vec_768d, dtype=torch.float32),
            'class_id': torch.tensor(int(row['severity_class_id']), dtype=torch.long),
            'phq8_score': torch.tensor(float(row['phq8_score']), dtype=torch.float32)
        }


def custom_collate(batch):
    sessions = [item['session_id'] for item in batch]
    egemaps = torch.stack([item['egemaps'] for item in batch])
    wav2vec = torch.stack([item['wav2vec'] for item in batch])
    class_ids = torch.stack([item['class_id'] for item in batch])
    phq8_scores = torch.stack([item['phq8_score'] for item in batch])
    
    max_len = max([item['temporal'].shape[0] for item in batch])
    padded_temporal = []
    for item in batch:
        temp = item['temporal']
        if temp.shape[0] < max_len:
            pad = torch.zeros((max_len - temp.shape[0], temp.shape[1]))
            temp = torch.cat([temp, pad], dim=0)
        padded_temporal.append(temp)
    padded_temporal = torch.stack(padded_temporal)

    return {
        'session_id': sessions,
        'egemaps': egemaps,
        'temporal': padded_temporal,
        'wav2vec': wav2vec,
        'class_id': class_ids,
        'phq8_score': phq8_scores
    }


def fit_post_hoc_temperature(model, dataloader):
    """
    Fits optimal scalar Temperature T > 0 strictly post-training for probability calibration.
    Does NOT modify raw logits or model predictions!
    """
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            out = model(batch['egemaps'], batch['temporal'], batch['wav2vec'])
            logits_list.append(out['severity_logits'])
            labels_list.append(batch['class_id'])

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def eval_fn():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    T_val = max(1.0, temperature.item())
    print(f"[CALIBRATION] Post-Hoc Temperature Fitted T = {T_val:.4f}")
    return T_val


def train_model(epochs=100, lr=0.001, store_dir="./feature_store"):
    if not HAS_TORCH:
        print("[WARNING] PyTorch not installed.")
        return None

    feature_store = FeatureStore(store_dir=store_dir)
    master_path = os.path.join(store_dir, "master_dataset_index.csv")

    if not os.path.exists(master_path):
        raise FileNotFoundError(f"Master index not found at {master_path}. Run label_mapper.py first.")

    master_df = pd.read_csv(master_path)
    print(f"\nLoaded {len(master_df)} sessions from {master_path}")

    dataset = FeatureStoreDataset(master_df, feature_store)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)

    # Initialize User's Original DualBranchDepressionNet
    model = DualBranchDepressionNet(num_classes=5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    print("\n==========================================================================")
    print(f"       STARTING MODEL TRAINING FOR 5 CLINICAL SEVERITY CLASSES")
    print("==========================================================================")
    print(f"Model Architecture : Dual-Branch BiLSTM + Self-Attention")
    print(f"Total Sessions     : {len(dataset)}")
    print(f"Epochs             : {epochs}")
    print("--------------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_grad_norm = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            optimizer.zero_grad()

            outputs = model(batch['egemaps'], batch['temporal'], batch['wav2vec'])
            
            # CrossEntropyLoss MUST receive raw severity_logits (NOT softmax)
            loss_ce = criterion_ce(outputs['severity_logits'], batch['class_id'])
            loss_mse = criterion_mse(outputs['phq8_score'], batch['phq8_score'])
            loss = loss_ce + 0.1 * loss_mse

            loss.backward()

            # Compute and clip gradient norms
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs['severity_logits'], dim=-1)
            correct += (preds == batch['class_id']).sum().item()
            total += batch['class_id'].size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch['class_id'].cpu().numpy())

        acc = (correct / total) * 100.0
        avg_grad_norm = total_grad_norm / len(dataloader)

        # Print detailed diagnostics after every 10 epochs (or epoch 1 & final)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            counts = pd.Series(all_preds).value_counts().to_dict()
            pred_counts_str = ", ".join([f"C{c}: {counts.get(c, 0)}" for c in range(5)])
            print(f"Epoch [{epoch:03d}/{epochs:03d}] -> Loss: {total_loss:.4f} | Acc: {acc:.2f}% | Grad Norm: {avg_grad_norm:.4f} | Pred Counts: [{pred_counts_str}]")

    # Post-Training Probability Calibration (Does NOT change argmax or predictions!)
    eval_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=custom_collate)
    T_optimal = fit_post_hoc_temperature(model, eval_loader)

    # Save trained model weights
    save_model_path = os.path.join(store_dir, "trained_depression_model.pth")
    torch.save(model.state_dict(), save_model_path)
    print(f"\n[SUCCESS] Model weights saved to: {save_model_path}")

    # Evaluate Final Predictions on Sessions
    model.eval()
    print("\n==========================================================================")
    print("                SESSION PREDICTION & DIAGNOSTIC TABLE")
    print("==========================================================================")
    
    for batch in eval_loader:
        with torch.no_grad():
            outputs = model(batch['egemaps'], batch['temporal'], batch['wav2vec'])
            raw_logits = outputs['severity_logits']
            
            # Post-hoc calibrated probabilities for display only
            calibrated_probs = F.softmax(raw_logits / T_optimal, dim=-1).cpu().numpy()
            pred_classes = torch.argmax(raw_logits, dim=-1).cpu().numpy()
            pred_phq = outputs['phq8_score'].cpu().numpy()

            true_classes = batch['class_id'].cpu().numpy()
            true_phq = batch['phq8_score'].cpu().numpy()
            session_ids = batch['session_id']

            rows = []
            for idx in range(len(session_ids)):
                r = {
                    'Session ID': session_ids[idx],
                    'Ground Truth Category': SEVERITY_NAMES[true_classes[idx]],
                    'Predicted Category': SEVERITY_NAMES[pred_classes[idx]],
                    'GT PHQ-8': f"{true_phq[idx]:.1f}",
                    'Pred PHQ-8': f"{pred_phq[idx]:.1f}",
                    'Confidence': f"{calibrated_probs[idx][pred_classes[idx]]*100:.1f}%"
                }
                rows.append(r)

            results_df = pd.DataFrame(rows)
            print(results_df.to_string(index=False))

            # Final Confusion Matrix & Per-Class Accuracy Analysis
            cm = confusion_matrix(true_classes, pred_classes, labels=list(range(5)))
            print("\n--------------------------------------------------------------------------")
            print("FINAL TRAINING CONFUSION MATRIX (Rows: True, Cols: Pred):")
            print(cm)
            print("--------------------------------------------------------------------------\n")

    return model


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    train_model(epochs=epochs)


if __name__ == "__main__":
    main()
