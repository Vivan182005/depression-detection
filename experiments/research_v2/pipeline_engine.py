"""
Shared Analysis Engine: experiments/research_v2/pipeline_engine.py
Phase 6A Pre-Registered 19-Variant Evaluation Engine.
Provides a SINGLE function run_pipeline(...) used for BOTH observed statistics and permutation nulls.
Strictly enforced:
- Zero real label inspection in exploratory/tuning loops.
- In-fold preprocessing (StandardScaler + winsorisation fitted strictly on outer train folds).
- Inner 4-fold CV with scoring='roc_auc' for binary PHQ >= 10.
- 5 selection seeds (0, 1, 2, 3, 4) averaged for participant-level OOF probabilities.
- Participant-grouped splits (zero participant leakage).
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids
from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine


# ==============================================================================
# 1. Neural Segment-Level Model Architecture (V12, V13)
# ==============================================================================

class SegmentMLP(nn.Module):
    """
    Fixed-architecture small 2-layer MLP for segment-level scoring.
    Trained for strictly fixed 40 epochs (zero early stopping against validation data).
    """
    def __init__(self, in_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_and_eval_segment_mlp(
    train_segments_list: List[np.ndarray],
    train_labels: np.ndarray,
    val_segments_list: List[np.ndarray],
    in_features: int,
    seed: int,
    epochs: int = 40,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Trains SegmentMLP on training participant segments and evaluates on validation participants.
    Aggregates validation segment predictions via arithmetic mean to 1 score per participant.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Gather training segments and broadcast participant labels
    X_train_list, y_train_list = [], []
    for segs, label in zip(train_segments_list, train_labels):
        if segs is not None and len(segs) > 0:
            X_train_list.append(segs)
            y_train_list.append(np.full(len(segs), label, dtype=np.float32))

    if not X_train_list:
        return np.full(len(val_segments_list), 0.5, dtype=np.float32)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # In-fold median imputation (prereg v1.1 sec 5.2): segment perturbation features are
    # NaN by design on turns that fail the validity thresholds.
    X_train = np.where(np.isfinite(X_train), X_train, np.nan)
    with np.errstate(all='ignore'):
        seg_medians = np.nan_to_num(np.nanmedian(X_train, axis=0), nan=0.0)

    def impute(a):
        a = np.where(np.isfinite(a), a, np.nan)
        return np.where(np.isnan(a), seg_medians, a)

    X_train = impute(X_train)

    # In-fold scaler fitted strictly on training segments
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Initialize model
    model = SegmentMLP(in_features=in_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    model.train()
    batch_size = 64
    n_samples = len(X_train_t)

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for b_start in range(0, n_samples, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            b_x = X_train_t[b_idx]
            b_y = y_train_t[b_idx]

            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

    # Inference on validation participants
    model.eval()
    val_preds = []
    with torch.no_grad():
        for segs in val_segments_list:
            if segs is None or len(segs) == 0:
                val_preds.append(0.5)
                continue
            segs_scaled = scaler.transform(impute(segs))
            segs_t = torch.tensor(segs_scaled, dtype=torch.float32).to(device)
            logits = model(segs_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.append(float(np.mean(probs)))

    return np.array(val_preds, dtype=np.float32)


# ==============================================================================
# 2. In-Fold Preprocessing
# ==============================================================================

def fit_transform_in_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    apply_winsor: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits StandardScaler and optional winsorisation strictly on training fold.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)

    if apply_winsor:
        p01 = np.percentile(X_tr, 1, axis=0)
        p99 = np.percentile(X_tr, 99, axis=0)
        X_tr = np.clip(X_tr, p01, p99)
        X_va = np.clip(X_va, p01, p99)

    # Impute any residual non-finite values using train medians
    train_medians = np.nanmedian(X_tr, axis=0)
    train_medians = np.nan_to_num(train_medians, nan=0.0)

    for col in range(X_tr.shape[1]):
        mask_tr = ~np.isfinite(X_tr[:, col])
        if np.any(mask_tr):
            X_tr[mask_tr, col] = train_medians[col]
        mask_va = ~np.isfinite(X_va[:, col])
        if np.any(mask_va):
            X_va[mask_va, col] = train_medians[col]

    return X_tr.astype(np.float32), X_va.astype(np.float32)


# ==============================================================================
# 3. Model Builder & Hyperparameter Grid Dispatch
# ==============================================================================

def get_variant_model_and_grid(
    variant_id: str,
    seed: int
) -> Tuple[Any, Dict[str, List[Any]], bool]:
    """
    Returns (base_estimator, param_grid, apply_winsor) for the given variant.
    """
    v = variant_id.upper()

    if v == 'V1':
        # eGeMAPS 135d + Elastic-Net Logistic Regression
        model = LogisticRegression(
            solver='saga',
            max_iter=1000,
            tol=1e-3,
            random_state=seed
        )
        grid = {
            'C': [0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
        apply_winsor = True

    elif v in ['V2', 'V4', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V14']:
        # L2 Logistic Regression (C in [0.001, 0.01, 0.1, 1.0, 10.0])
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=seed
        )
        grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
        apply_winsor = (v in ['V2', 'V4'])

    elif v in ['V3', 'V5']:
        # HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=seed)
        grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'min_samples_leaf': [5, 10]
        }
        apply_winsor = False

    elif v == 'V15':
        # Length metadata control + LR
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=seed
        )
        grid = {'C': [0.01, 0.1, 1.0, 10.0]}
        apply_winsor = False

    elif v in ['V16', 'V17']:
        # Gender-only or Domain-only control
        model = LogisticRegression(
            solver='lbfgs',
            C=1.0,
            max_iter=1000,
            random_state=seed
        )
        grid = {}  # Fixed C=1.0
        apply_winsor = False

    else:
        raise ValueError(f"Unknown tabular variant ID: {variant_id}")

    return model, grid, apply_winsor


# ==============================================================================
# 4. Single Unified Analysis Function: run_pipeline
# ==============================================================================

def run_pipeline(
    X: Union[np.ndarray, List[np.ndarray], Dict[str, Any], Tuple[Any, ...]],
    y: np.ndarray,
    groups: np.ndarray,
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    variant_id: str = 'V2',
    inner_cv_splits: int = 4,
    outer_cv_splits: int = 5,
    strata: Optional[np.ndarray] = None,
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    ONE shared pipeline execution function used for BOTH observed statistics and permutation nulls.

    Parameters:
    -----------
    X: Feature input:
       - 2D array (N, D) for standard tabular variants (V1-V11, V14-V17).
       - List of 2D segment arrays for segment models (V12, V13).
       - Dict or Tuple of feature matrices for multimodal stacking (V18, V19).
    y: Binary labels (PHQ >= 10), length N.
    groups: Participant IDs, length N.
    seeds: Outer selection seeds (default: 0, 1, 2, 3, 4).
    variant_id: One of 'V1' through 'V19'.
    inner_cv_splits: Splits for inner hyperparameter tuning (default: 4).
    outer_cv_splits: Splits for outer validation (default: 5).
    strata: Optional strata for StratifiedGroupKFold (defaults to y).
    device: 'cpu' or 'mps' for neural models.

    Returns:
    --------
    Dict containing:
    - 'auc': Participant-level ROC-AUC on 5-seed averaged OOF probabilities.
    - 'oof_probs': Averaged OOF predicted probabilities (length N).
    - 'seed_aucs': List of AUCs for each selection seed.
    - 'variant_id': The executed variant identifier.
    """
    variant_id = variant_id.upper()
    n_participants = len(groups)
    assert len(y) == n_participants, f"Mismatched X/y: len(y)={len(y)}, N={n_participants}"

    # Track out-of-fold predicted probabilities across seeds
    seed_oof_probs = np.zeros((len(seeds), n_participants), dtype=np.float32)
    seed_aucs = []

    for seed_idx, seed in enumerate(seeds):
        # 5-fold Stratified Group K-Fold outer split
        outer_cv = StratifiedGroupKFold(n_splits=outer_cv_splits, shuffle=True, random_state=seed)
        fold_probs = np.zeros(n_participants, dtype=np.float32)

        strat_target = strata if strata is not None else y

        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(np.zeros(n_participants), strat_target, groups=groups)):
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]

            # ------------------------------------------------------------------
            # Case A: Segment-Level Neural Models (V12, V13)
            # ------------------------------------------------------------------
            if variant_id in ['V12', 'V13']:
                assert isinstance(X, list), f"{variant_id} requires a list of participant segment arrays"
                train_segs = [X[i] for i in train_idx]
                val_segs = [X[i] for i in val_idx]
                in_feat = 55 if variant_id == 'V12' else 768

                p_val = train_and_eval_segment_mlp(
                    train_segments_list=train_segs,
                    train_labels=y_train,
                    val_segments_list=val_segs,
                    in_features=in_feat,
                    seed=seed * 100 + fold,
                    epochs=40,
                    lr=1e-3,
                    device=device
                )
                fold_probs[val_idx] = p_val

            # ------------------------------------------------------------------
            # Case B: Nested Stacking V2 + V14 (V18)
            # ------------------------------------------------------------------
            elif variant_id == 'V18':
                assert isinstance(X, dict), "V18 requires X as dict with keys 'v2' and 'v14'"
                X_v2 = X['v2']
                X_v14 = X['v14']

                # Preprocess both modalities in-fold
                X_tr_v2, X_va_v2 = fit_transform_in_fold(X_v2[train_idx], X_v2[val_idx], apply_winsor=True)
                X_tr_v14, X_va_v14 = fit_transform_in_fold(X_v14[train_idx], X_v14[val_idx], apply_winsor=False)

                # Inner CV on unique training participants
                n_inner = min(inner_cv_splits, sum(y_train == 1), sum(y_train == 0))
                n_inner = max(2, n_inner)
                inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=fold)
                inner_oof_v2 = np.zeros(len(train_idx), dtype=np.float32)
                inner_oof_v14 = np.zeros(len(train_idx), dtype=np.float32)

                for in_tr, in_va in inner_cv.split(X_tr_v2, y_train):
                    # Model V2 inner fit
                    m_v2, g_v2, _ = get_variant_model_and_grid('V2', seed=fold)
                    grid_v2 = GridSearchCV(m_v2, g_v2, cv=n_inner, scoring='roc_auc', n_jobs=1)
                    grid_v2.fit(X_tr_v2[in_tr], y_train[in_tr])
                    inner_oof_v2[in_va] = grid_v2.predict_proba(X_tr_v2[in_va])[:, 1]

                    # Model V14 inner fit
                    m_v14, g_v14, _ = get_variant_model_and_grid('V14', seed=fold)
                    grid_v14 = GridSearchCV(m_v14, g_v14, cv=n_inner, scoring='roc_auc', n_jobs=1)
                    grid_v14.fit(X_tr_v14[in_tr], y_train[in_tr])
                    inner_oof_v14[in_va] = grid_v14.predict_proba(X_tr_v14[in_va])[:, 1]

                # Meta-learner fitted on inner-OOF predictions
                X_meta_train = np.column_stack([inner_oof_v2, inner_oof_v14])
                meta_clf = LogisticRegression(C=1.0, solver='lbfgs', random_state=fold)
                meta_clf.fit(X_meta_train, y_train)

                # Refit full V2 and V14 on full outer train fold
                grid_v2_full = GridSearchCV(m_v2, g_v2, cv=n_inner, scoring='roc_auc', n_jobs=1)
                grid_v2_full.fit(X_tr_v2, y_train)
                outer_p_v2 = grid_v2_full.predict_proba(X_va_v2)[:, 1]

                grid_v14_full = GridSearchCV(m_v14, g_v14, cv=n_inner, scoring='roc_auc', n_jobs=1)
                grid_v14_full.fit(X_tr_v14, y_train)
                outer_p_v14 = grid_v14_full.predict_proba(X_va_v14)[:, 1]

                # Predict outer validation via meta-learner
                X_meta_val = np.column_stack([outer_p_v2, outer_p_v14])
                fold_probs[val_idx] = meta_clf.predict_proba(X_meta_val)[:, 1]

            # ------------------------------------------------------------------
            # Case C: Nested Stacking of Inner-CV Selected Best Audio + V14 (V19)
            # ------------------------------------------------------------------
            elif variant_id == 'V19':
                assert isinstance(X, dict), "V19 requires X as dict with candidate audio features and 'v14'"
                X_v14 = X['v14']
                X_tr_v14, X_va_v14 = fit_transform_in_fold(X_v14[train_idx], X_v14[val_idx], apply_winsor=False)

                n_inner = min(inner_cv_splits, sum(y_train == 1), sum(y_train == 0))
                n_inner = max(2, n_inner)
                inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=fold)

                best_audio_id = None
                best_inner_auc = -1.0
                best_audio_dim = 99999

                audio_candidates = [k for k in X.keys() if k.upper() != 'V14' and k.upper().startswith('V')]
                audio_candidates.sort()

                for cand in audio_candidates:
                    X_cand = X[cand]
                    if isinstance(X_cand, list):
                        continue
                    X_tr_c, _ = fit_transform_in_fold(X_cand[train_idx], X_cand[val_idx], apply_winsor=(cand.upper() in ('V2', 'V4')))
                    m_c, g_c, _ = get_variant_model_and_grid(cand, seed=fold)
                    grid_c = GridSearchCV(m_c, g_c, cv=inner_cv, scoring='roc_auc', n_jobs=1)
                    grid_c.fit(X_tr_c, y_train)
                    c_auc = grid_c.best_score_
                    c_dim = X_cand.shape[1]

                    if c_auc > best_inner_auc or (abs(c_auc - best_inner_auc) < 1e-5 and c_dim < best_audio_dim):
                        best_inner_auc = c_auc
                        best_audio_id = cand
                        best_audio_dim = c_dim

                if best_audio_id is None:
                    best_audio_id = 'V2'

                # Execute stacking with selected best audio variant
                X_best = X[best_audio_id]
                X_tr_b, X_va_b = fit_transform_in_fold(X_best[train_idx], X_best[val_idx], apply_winsor=(best_audio_id.upper() in ('V2', 'V4')))

                inner_oof_b = np.zeros(len(train_idx), dtype=np.float32)
                inner_oof_v14 = np.zeros(len(train_idx), dtype=np.float32)

                for in_tr, in_va in inner_cv.split(X_tr_b, y_train):
                    m_b, g_b, _ = get_variant_model_and_grid(best_audio_id, seed=fold)
                    grid_b = GridSearchCV(m_b, g_b, cv=n_inner, scoring='roc_auc', n_jobs=1)
                    grid_b.fit(X_tr_b[in_tr], y_train[in_tr])
                    inner_oof_b[in_va] = grid_b.predict_proba(X_tr_b[in_va])[:, 1]

                    m_v14, g_v14, _ = get_variant_model_and_grid('V14', seed=fold)
                    grid_v14 = GridSearchCV(m_v14, g_v14, cv=n_inner, scoring='roc_auc', n_jobs=1)
                    grid_v14.fit(X_tr_v14[in_tr], y_train[in_tr])
                    inner_oof_v14[in_va] = grid_v14.predict_proba(X_tr_v14[in_va])[:, 1]

                X_meta_train = np.column_stack([inner_oof_b, inner_oof_v14])
                meta_clf = LogisticRegression(C=1.0, solver='lbfgs', random_state=fold)
                meta_clf.fit(X_meta_train, y_train)

                grid_b_full = GridSearchCV(m_b, g_b, cv=n_inner, scoring='roc_auc', n_jobs=1)
                grid_b_full.fit(X_tr_b, y_train)
                outer_p_b = grid_b_full.predict_proba(X_va_b)[:, 1]

                grid_v14_full = GridSearchCV(m_v14, g_v14, cv=n_inner, scoring='roc_auc', n_jobs=1)
                grid_v14_full.fit(X_tr_v14, y_train)
                outer_p_v14 = grid_v14_full.predict_proba(X_va_v14)[:, 1]

                X_meta_val = np.column_stack([outer_p_b, outer_p_v14])
                fold_probs[val_idx] = meta_clf.predict_proba(X_meta_val)[:, 1]

            # ------------------------------------------------------------------
            # Case D: Standard Tabular Variants (V1-V11, V14-V17)
            # ------------------------------------------------------------------
            else:
                assert isinstance(X, np.ndarray), f"{variant_id} requires X as 2D numpy array"
                base_model, param_grid, apply_winsor = get_variant_model_and_grid(variant_id, seed=seed)

                X_tr, X_va = fit_transform_in_fold(
                    X[train_idx],
                    X[val_idx],
                    apply_winsor=apply_winsor
                )

                if param_grid:
                    n_inner = min(inner_cv_splits, sum(y_train == 1), sum(y_train == 0))
                    n_inner = max(2, n_inner)
                    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=fold)
                    grid = GridSearchCV(
                        base_model,
                        param_grid,
                        cv=inner_cv,
                        scoring='roc_auc',
                        n_jobs=1
                    )
                    grid.fit(X_tr, y_train)
                    p_val = grid.predict_proba(X_va)[:, 1]
                else:
                    base_model.fit(X_tr, y_train)
                    p_val = base_model.predict_proba(X_va)[:, 1]

                fold_probs[val_idx] = p_val

        # Record seed OOF probabilities
        seed_oof_probs[seed_idx] = fold_probs
        # Calculate seed AUC
        s_auc = roc_auc_score(y, fold_probs)
        seed_aucs.append(float(s_auc))

    # Pre-registered participant-level decision vector: arithmetic mean over seeds
    aggregated_oof_probs = seed_oof_probs.mean(axis=0)
    final_auc = float(roc_auc_score(y, aggregated_oof_probs))

    return {
        'variant_id': variant_id,
        'auc': final_auc,
        'oof_probs': aggregated_oof_probs,
        'seed_aucs': seed_aucs,
        'n_participants': n_participants
    }
