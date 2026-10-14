import sys, os
_REPO_ROOT = os.path.abspath(".")
if _REPO_ROOT not in sys.path: sys.path.insert(0, _REPO_ROOT)
"""
Script: experiments/research_v2/generate_official_labels.py
Description: Generates the authoritative, audit-verified master clinical label file
experiments/research_v2/labels_official.csv from detailed_lables.csv and the official
train/dev/test split CSVs.
Guarantees:
1. sum(PHQ8_1..8) == PHQ_Score for all 275 participants.
2. Official split gender mapped and auditable.
3. Class definitions: 0 (0-9), 1 (10-14), 2 (15-24).
4. Strictly NO default-0.0, NO fabricated values.
"""

import os
import sys
import pandas as pd
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def generate_official_labels():
    split_paths = [
        ('dev', os.path.join(_REPO_ROOT, 'new_audio_csv', 'dev_split.csv')),
        ('train', os.path.join(_REPO_ROOT, 'new_audio_csv', 'train_split.csv')),
        ('test', os.path.join(_REPO_ROOT, 'new_audio_csv', 'test_split.csv'))
    ]
    
    df_splits_list = []
    for split_name, path in split_paths:
        df_s = pd.read_csv(path)
        df_s['split'] = split_name
        df_splits_list.append(df_s)
    df_splits = pd.concat(df_splits_list, ignore_index=True)
    
    det_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'detailed_lables.csv')
    df_det = pd.read_csv(det_path)
    
    phq_cols = [c for c in df_det.columns if c.startswith('PHQ8_') and c != 'PHQ8_Score']
    assert len(phq_cols) == 8, f"Expected 8 PHQ items, found {len(phq_cols)}"
    df_det['phq8_sum'] = df_det[phq_cols].sum(axis=1)
    
    # Assert exact match between detailed_lables and split files
    merged = pd.merge(df_splits, df_det, left_on='Participant_ID', right_on='Participant')
    assert len(merged) == 275, f"Expected 275 merged participants, found {len(merged)}"
    
    phq_diffs = (merged['PHQ_Score'] != merged['phq8_sum']).sum()
    assert phq_diffs == 0, f"Found {phq_diffs} PHQ mismatches between splits and detailed_lables!"
    
    from experiments.research_v2.sealed_guard import SEALED_PIDS
    
    rows = []
    for idx, r in merged.iterrows():
        pid = int(r['Participant_ID'])
        phq = float(r['PHQ_Score'])
        cls_id = 0 if phq <= 9.0 else (1 if phq <= 14.0 else 2)
        
        # Gender from official challenge splits
        split_gender = str(r['Gender']).strip().lower()
        det_gender = str(r['gender']).strip().lower()
        # Official challenge split gender is canonical
        canonical_gender = split_gender
        
        age = float(r['age']) if not pd.isna(r['age']) else np.nan
        domain = 'woz' if pid < 600 else 'avatar'
        is_sealed = bool(pid in SEALED_PIDS)
        
        rows.append({
            'participant_id': pid,
            'phq8_score': phq,
            'class_id': cls_id,
            'gender': canonical_gender,
            'gender_detailed_labels': det_gender,
            'age': age,
            'split': r['split_x'],
            'domain': domain,
            'is_sealed': is_sealed,
            'binary_label': 1 if phq >= 10.0 else 0
        })
        
    df_out = pd.DataFrame(rows).sort_values('participant_id').reset_index(drop=True)
    out_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'labels_official.csv')
    df_out.to_csv(out_path, index=False)
    
    print(f"Successfully generated {out_path} with {len(df_out)} participants.")
    print("Class distribution across all 275 participants:")
    print(df_out['class_id'].value_counts().sort_index())
    print("\nNon-sealed development cohort class distribution (N=240):")
    dev_mask = ~df_out['is_sealed']
    print(df_out[dev_mask]['class_id'].value_counts().sort_index())
    print("\nWoZ development class distribution (N=179):")
    woz_dev_mask = dev_mask & (df_out['domain'] == 'woz')
    print(df_out[woz_dev_mask]['class_id'].value_counts().sort_index())
    print("\nAvatar development class distribution (N=61):")
    avatar_dev_mask = dev_mask & (df_out['domain'] == 'avatar')
    print(df_out[avatar_dev_mask]['class_id'].value_counts().sort_index())
    return df_out

if __name__ == '__main__':
    generate_official_labels()
