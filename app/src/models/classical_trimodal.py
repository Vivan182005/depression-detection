"""
Module: classical_trimodal.py
Description: Model A: Simple Fused Classical Tri-Modal Model with Dedicated Binary Classifier (Fix Prompt 16).
Concatenates pooled audio representations (824-dim: eGeMAPS 88d + Temporal mean/std 66d + Wav2Vec2 768d)
with BERT text embeddings (768-dim) into a 1,592-dimensional vector.
Applies PCA (25 components) -> Gradient Boosted Trees (3-Class) + Dedicated Logistic Regression (Binary).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, confusion_matrix


def build_trimodal_pooled_feature_vector(feats):
    """
    Extracts fixed-length 1,592d pooled feature vector from session feature dictionary:
    - egemaps_88d: 88d
    - temporal_33d: mean (33d) + std (33d) = 66d
    - wav2vec_768d: 768d
    - text_bert_768d: 768d
    Total: 88 + 66 + 768 + 768 = 1,592 dims
    """
    egemaps = feats.get('egemaps_88d', np.zeros(88, dtype=np.float32))
    t_33d = feats.get('temporal_33d', np.array([]))
    wav2vec = feats.get('wav2vec_768d', np.zeros(768, dtype=np.float32))
    text_bert = feats.get('text_bert_768d', np.zeros(768, dtype=np.float32))

    if len(t_33d) > 0:
        t_mean = np.mean(t_33d, axis=0)
        t_std = np.std(t_33d, axis=0)
    else:
        t_mean = np.zeros(33, dtype=np.float32)
        t_std = np.zeros(33, dtype=np.float32)

    pooled_vector = np.hstack([egemaps, t_mean, t_std, wav2vec, text_bert])
    return pooled_vector.astype(np.float32)


# Backward compatibility alias
build_pooled_feature_vector = build_trimodal_pooled_feature_vector


class TriModalPCAGradientBoostingPipeline:
    """
    Model A Pipeline:
    - StandardScaler -> PCA(25)
    - 3-Class Head: GradientBoostingClassifier
    - Dedicated Binary Head: LogisticRegression(class_weight='balanced', C=0.5)
    """

    def __init__(self, n_components=25, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=random_state
        )
        self.binary_clf = LogisticRegression(
            class_weight='balanced',
            C=0.5,
            random_state=random_state,
            max_iter=1000
        )

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        n_comp = min(self.n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
        if n_comp != self.pca.n_components:
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Fit 3-class classifier
        self.clf.fit(X_pca, y)

        # Fit dedicated binary classifier directly on depressed (>=1) vs non-depressed (0)
        y_binary = (y >= 1).astype(int)
        self.binary_clf.fit(X_pca, y_binary)
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.clf.predict_proba(X_pca)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.clf.predict(X_pca)

    def predict_binary(self, X):
        """Returns dedicated binary classification decision (0=Minimal/Mild, 1=Depressed)."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.binary_clf.predict(X_pca)

    def predict_binary_proba(self, X):
        """Returns dedicated binary classification probability [p(not depressed), p(depressed)]."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.binary_clf.predict_proba(X_pca)


def train_and_evaluate_model_a(train_df, val_df, test_df, feature_store, store_dir="./feature_store"):
    """
    Trains Model A (3-class GBDT + Dedicated Binary Logistic Regression),
    evaluates on test_df, and saves pipeline checkpoint.
    """
    def extract_matrix(df):
        X, y_class, y_phq = [], [], []
        for idx, row in df.iterrows():
            feats = feature_store.load_features(row['session_id'])
            vec = build_trimodal_pooled_feature_vector(feats)
            X.append(vec)
            y_class.append(int(row['severity_class_id']))
            y_phq.append(float(row['phq8_score']))
        return np.array(X), np.array(y_class), np.array(y_phq)

    X_train, y_train, phq_train = extract_matrix(train_df)
    X_val, y_val, phq_val = extract_matrix(val_df)
    X_test, y_test, phq_test = extract_matrix(test_df)

    pipeline = TriModalPCAGradientBoostingPipeline(n_components=25, random_state=42)
    pipeline.fit(X_train, y_train)

    # 3-Class holdout metrics
    test_probs = pipeline.predict_proba(X_test)
    test_preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, test_preds) * 100.0
    macro_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0) * 100.0
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2])

    # Dedicated Binary Holdout Metrics (Fix Prompt 16)
    y_test_binary_true = (y_test >= 1).astype(int)
    y_test_binary_pred = pipeline.predict_binary(X_test)
    binary_acc = accuracy_score(y_test_binary_true, y_test_binary_pred) * 100.0
    binary_f1 = f1_score(y_test_binary_true, y_test_binary_pred, zero_division=0) * 100.0

    class_centers = np.array([4.5, 12.0, 18.0])
    pred_phqs = np.dot(test_probs, class_centers)
    mae = float(np.mean(np.abs(pred_phqs - phq_test)))

    model_path = os.path.join(store_dir, "best_trimodal_model_a.joblib")
    joblib.dump(pipeline, model_path)

    return {
        'model_name': 'Model A (PCA 25 + Gradient Boosting)',
        'accuracy': acc,
        'macro_f1': macro_f1,
        'binary_accuracy': binary_acc,
        'binary_f1': binary_f1,
        'mae': mae,
        'confusion_matrix': cm,
        'test_probs': test_probs,
        'test_preds': test_preds,
        'y_test': y_test,
        'model_path': model_path
    }
