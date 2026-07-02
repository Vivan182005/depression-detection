import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from src.models import LSTMAttentionClassifier, build_svm_classifier, build_xgboost_classifier


class SpeechSequenceDataset(Dataset):
    """
    Dataset representing sequences of speech features.
    X: list of numpy arrays, each of shape (seq_len, feature_dim)
    y: list of binary targets (0: non-depressed, 1: depressed)
    """
    def __init__(self, X, y):
        self.X = [torch.tensor(item, dtype=torch.float32) for item in X]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        
        # Calculate accuracy
        preds = torch.sigmoid(logits) >= 0.5
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        
    return total_loss / total, correct / total


def evaluate_lstm(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(logits) >= 0.5
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    return total_loss / total, correct / total


def train_lstm_model(X_train, y_train, X_val, y_val, input_dim=33, epochs=30, batch_size=4, lr=0.001, save_path="models/lstm_attn.pth"):
    """
    Trains the PyTorch LSTM + Attention model and saves it.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    train_dataset = SpeechSequenceDataset(X_train, y_train)
    val_dataset = SpeechSequenceDataset(X_val, y_val)
    
    # Use custom collate to pad variable length sequences if needed. 
    # For simplicity, if we assume they are already padded or matching size:
    def pad_collate(batch):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        max_len = max(x_lens)
        
        # Pad with zeros
        padded_xx = []
        for x in xx:
            if len(x) < max_len:
                padding = torch.zeros(max_len - len(x), x.shape[1])
                padded_xx.append(torch.cat([x, padding], dim=0))
            else:
                padded_xx.append(x)
                
        return torch.stack(padded_xx), torch.stack(yy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    
    model = LSTMAttentionClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    print(f"Training LSTM Model on {device}...")
    for epoch in range(epochs):
        train_loss, train_acc = train_lstm_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_lstm(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}")
            
    print(f"Training complete. Best model saved to: {save_path}")
    return model


def train_baseline_models(X_train, y_train, X_val, y_val, save_dir="models/"):
    """
    Aggregates frame features (by taking mean/std across frames) to train SVM and XGBoost.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Flatten/Aggregate sequential features for traditional ML (mean across sequence)
    X_train_agg = np.array([np.mean(seq, axis=0) for seq in X_train])
    X_val_agg = np.array([np.mean(seq, axis=0) for seq in X_val])
    
    # SVM
    svm_model = build_svm_classifier()
    svm_model.fit(X_train_agg, y_train)
    svm_acc = svm_model.score(X_val_agg, y_val)
    with open(os.path.join(save_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm_model, f)
    print(f"SVM Validation Accuracy: {svm_acc:.4f} (Saved to {save_dir}svm_model.pkl)")
    
    # XGBoost
    xgb_model = build_xgboost_classifier()
    if xgb_model is not None:
        xgb_model.fit(X_train_agg, y_train)
        xgb_acc = xgb_model.score(X_val_agg, y_val)
        with open(os.path.join(save_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
        print(f"XGBoost Validation Accuracy: {xgb_acc:.4f} (Saved to {save_dir}xgb_model.pkl)")
    else:
        print("Skipping XGBoost training as model is unavailable on this platform.")
    
    return svm_model, xgb_model


def generate_synthetic_data(num_samples=20, seq_len=100, feature_dim=33):
    """
    Generates dummy/synthetic sequential data to test pipeline/training.
    Depressed samples (class 1) are given slightly lower pitch variance and energy profiles.
    """
    X = []
    y = []
    
    for i in range(num_samples):
        # 50% depressed, 50% non-depressed
        label = 1 if i < num_samples // 2 else 0
        
        # Create sequence of features (frames)
        # Baseline normal features
        seq = np.random.randn(seq_len, feature_dim) * 0.5
        
        if label == 1:
            # Shift features representing flattened prosody (e.g. lower loudness/energy variance)
            seq[:, -1] = seq[:, -1] * 0.3 - 0.5  # lower energy
            
        X.append(seq)
        y.append(label)
        
    # Shuffle
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    X = [X[idx] for idx in indices]
    y = [y[idx] for idx in indices]
    
    return X, y


if __name__ == "__main__":
    # Self-test code: train on synthetic data
    print("Generating synthetic data for training demo...")
    X, y = generate_synthetic_data(num_samples=30, seq_len=80, feature_dim=33)
    
    # Split
    split = 24
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples...")
    train_lstm_model(X_train, y_train, X_val, y_val, input_dim=33, epochs=10)
    train_baseline_models(X_train, y_train, X_val, y_val)
