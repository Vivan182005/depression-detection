"""
Data Loader: experiments/research_v2/data_loader.py
Strictly quarantined data loader for the 239 active development sessions.
Provides label-blind feature matrices and stratified label permutation machinery.
Sealed cohort of 33 is strictly barred.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine

FEATURE_STORE_V3_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3')
FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
MANIFEST_PATH = os.path.join(FEATURE_STORE_V3_DIR, 'manifest_v3.csv')
LABELS_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'labels_official.csv')


def permute_labels_stratified(df: pd.DataFrame, seed: int) -> np.ndarray:
    """
    Permutes clinical binary labels strictly WITHIN domain x gender strata.
    Preserves stratum marginal base rates. Shuffling is NEVER within class.
    """
    rng = np.random.RandomState(seed)
    y_perm = np.zeros(len(df), dtype=int)
    
    strata = df.groupby(['domain', 'gender']).groups
    for stratum_key, idxs in strata.items():
        sub_y = df.loc[idxs, 'binary_label'].values.copy()
        rng.shuffle(sub_y)
        y_perm[df.index.get_indexer(idxs)] = sub_y
        
    return y_perm


def load_cohort_info(permute: bool = False, permute_seed: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads development cohort information for all 239 eligible sessions.
    Strictly asserts zero sealed participant inclusion.

    Returns:
    --------
    cohort_df: DataFrame with metadata (participant_id, domain, gender).
    y: Binary labels (PHQ >= 10). If permute=True, permuted within domain x gender.
    groups: Participant IDs (array of length 239).
    strata: Stratum identifiers for stratified CV.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    assert len(manifest) == 239, f"Expected 239 manifest rows, got {len(manifest)}"
    assert_no_sealed_ids(manifest['participant_id'].values)

    labels_df = pd.read_csv(LABELS_PATH)
    cohort_df = labels_df[labels_df['participant_id'].isin(manifest['participant_id'])].copy()
    cohort_df = cohort_df.sort_values('participant_id').reset_index(drop=True)

    assert len(cohort_df) == 239, f"Expected 239 merged cohort participants, got {len(cohort_df)}"
    assert_no_sealed_ids(cohort_df['participant_id'].values)

    for pid in cohort_df['participant_id'].values:
        assert_sealed_quarantine(int(pid))

    # labels_official.csv stores gender as 'male'/'female'; downstream code
    # (V16 control, subgroup endpoints) expects female=0, male=1.
    gender_map = {'female': 0, 'male': 1}
    unknown = set(cohort_df['gender'].unique()) - set(gender_map)
    if unknown:
        raise ValueError(f"Unexpected gender values in dev cohort: {unknown}")
    cohort_df['gender'] = cohort_df['gender'].map(gender_map).astype(int)
    cohort_df['phq_score'] = cohort_df['phq8_score'].astype(float)

    groups = cohort_df['participant_id'].values
    strata = (cohort_df['domain'] + "_" + cohort_df['gender'].astype(str)).values

    if permute:
        assert permute_seed is not None, "permute_seed required when permute=True"
        y = permute_labels_stratified(cohort_df, seed=permute_seed)
    else:
        y = cohort_df['binary_label'].values.astype(int)

    return cohort_df, y, groups, strata


def load_variant_features(
    variant_id: str,
    cohort_df: pd.DataFrame
) -> Union[np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Loads exact cached feature matrices for the specified variant across all 239 participants.
    """
    v = variant_id.upper()
    pids = cohort_df['participant_id'].values
    n = len(pids)

    # --------------------------------------------------------------------------
    # Tabular Audio Variants (V1-V11)
    # --------------------------------------------------------------------------
    if v == 'V1':
        # eGeMAPS 135d
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['egemaps_135d'])
        return np.array(feats, dtype=np.float32)

    elif v in ['V2', 'V3']:
        # Handcrafted summary (330d)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['handcrafted_summary'])
        return np.array(feats, dtype=np.float32)

    elif v in ['V4', 'V5']:
        # Handcrafted summary CMVN (330d)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['handcrafted_summary_cmvn'])
        return np.array(feats, dtype=np.float32)

    elif v == 'V6':
        # Wav2Vec2 Layer 4 (768d)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['w2v2_layer_4_mean'])
        return np.array(feats, dtype=np.float32)

    elif v == 'V7':
        # Wav2Vec2 Layer 8 (768d)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['w2v2_layer_8_mean'])
        return np.array(feats, dtype=np.float32)

    elif v == 'V8':
        # Wav2Vec2 Layer 12 (768d) - zero fallback
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            if 'w2v2_layer_12_mean' not in d:
                raise KeyError(f"Requested V8 (Wav2Vec2 L12) but 'w2v2_layer_12_mean' is missing for PID {pid}. Silent fallback prohibited.")
            feats.append(d['w2v2_layer_12_mean'])
        return np.array(feats, dtype=np.float32)

    elif v == 'V9':
        # WavLM Layer 4 (768d)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['wavlm_layer_4_mean'])
        return np.array(feats, dtype=np.float32)

    elif v == 'V10':
        # WavLM Layer 8 (768d, Confirmatory H2)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['wavlm_layer_8_mean'])
        return np.array(feats, dtype=np.float32)

    elif v == 'V11':
        # WavLM Layer 12 (768d)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            feats.append(d['wavlm_layer_12_mean'])
        return np.array(feats, dtype=np.float32)

    # --------------------------------------------------------------------------
    # Segment-Level Neural Models (V12, V13)
    # --------------------------------------------------------------------------
    elif v == 'V12':
        # Handcrafted segment list (N arrays, each Ni x 55)
        segs_list = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            segs_list.append(d['handcrafted_segments'])
        return segs_list

    elif v == 'V13':
        # WavLM Layer 8 segment list (N arrays, each Ni x 768)
        segs_list = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            segs_list.append(d['wavlm_layer_8_segments'])
        return segs_list

    # --------------------------------------------------------------------------
    # Text Benchmark (V14)
    # --------------------------------------------------------------------------
    elif v == 'V14':
        # BERT-T2 768d from feature_store_v2 (strictly true T2 Whisper text, zero fallback)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V2_DIR, f"{pid}_v2.npz"))
            if 'bert_t2_768d' not in d:
                raise KeyError(f"Requested V14 (BERT-T2) but 'bert_t2_768d' is missing for PID {pid}. Silent fallback prohibited.")
            feats.append(d['bert_t2_768d'])
        return np.array(feats, dtype=np.float32)

    # --------------------------------------------------------------------------
    # Metadata Controls (V15, V16, V17)
    # --------------------------------------------------------------------------
    elif v == 'V15':
        # Length metadata (total duration, speech duration, n segments, mean segment duration)
        feats = []
        for pid in pids:
            d = np.load(os.path.join(FEATURE_STORE_V3_DIR, f"{pid}_features.npz"))
            tot_dur = float(d['total_duration_s'])
            sp_dur = float(d['speech_duration_s'])
            n_segs = max(1, int(d['n_vad_segments']))
            mean_seg = sp_dur / n_segs
            feats.append([tot_dur, sp_dur, float(n_segs), mean_seg])
        return np.array(feats, dtype=np.float32)

    elif v == 'V16':
        # Gender-only control (shape Nx1)
        return cohort_df['gender'].values.astype(np.float32).reshape(-1, 1)

    elif v == 'V17':
        # Domain-only control (WoZ=1, Avatar=0, shape Nx1)
        is_woz = (cohort_df['domain'] == 'woz').astype(np.float32).values.reshape(-1, 1)
        return is_woz

    # --------------------------------------------------------------------------
    # Multimodal Stacking Blenders (V18, V19)
    # --------------------------------------------------------------------------
    elif v == 'V18':
        X_v2 = load_variant_features('V2', cohort_df)
        X_v14 = load_variant_features('V14', cohort_df)
        return {'v2': X_v2, 'v14': X_v14}

    elif v == 'V19':
        # Loads candidate audio variants and V14
        data_dict = {'v14': load_variant_features('V14', cohort_df)}
        for c in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']:
            data_dict[c.lower()] = load_variant_features(c, cohort_df)
        return data_dict

    else:
        raise ValueError(f"Unknown variant ID: {variant_id}")
