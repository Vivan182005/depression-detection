"""
Script: experiments/research_v2/generate_folds_combined.py
Description: Generates reproducible StratifiedGroupKFold splits for the Combined Development Cohort
(N=240 unique participants: 179 WoZ + 61 Avatar, 244 sessions).
Stratifies simultaneously on (domain, depression_class) to guarantee balanced representation.
Groups strictly on participant_id. Zero access to sealed cohort.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids
from experiments.research_v2.dataset import load_development_dataset

def generate_combined_folds(seeds=[0, 1, 2, 3, 4], n_splits=5):
    manifest_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'expanded_dataset_manifest.csv')
    df_manifest = pd.read_csv(manifest_path)
    
    # 1. WoZ dev sessions
    master_path = os.path.join(_REPO_ROOT, 'app', 'feature_store', 'master_dataset_index.csv')
    df_woz = pd.read_csv(master_path)
    clean_labels_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'detailed_lables.csv')
    clean_df = pd.read_csv(clean_labels_path)
    from experiments.research_v2.sealed_guard import SEALED_PIDS
    clean_df = clean_df[~clean_df['Participant'].isin(SEALED_PIDS)]
    clean_map = dict(zip(clean_df['Participant'], clean_df['Depression_severity']))
    
    woz_rows = []
    for idx, row in df_woz.iterrows():
        pid = int(row['participant_id'])
        sid = str(row['session_id'])
        phq = float(clean_map.get(pid, row['phq8_score']))
        cls_id = 0 if phq <= 9.0 else (1 if phq <= 14.0 else 2)
        woz_rows.append({
            'session_id': sid,
            'participant_id': pid,
            'domain': 'woz',
            'phq8_score': phq,
            'class_id': cls_id,
            'strat_key': f'woz_{cls_id}'
        })
        
    # 2. Avatar dev sessions
    avatar_target = df_manifest[df_manifest['cohort_role'] == 'avatar_candidate_expansion']
    avatar_rows = []
    for idx, row in avatar_target.iterrows():
        pid = int(row['participant_id'])
        sid = str(pid)
        phq = float(row['phq8_score'])
        cls_id = int(row['class_id'])
        avatar_rows.append({
            'session_id': sid,
            'participant_id': pid,
            'domain': 'avatar',
            'phq8_score': phq,
            'class_id': cls_id,
            'strat_key': f'avatar_{cls_id}'
        })
        
    df_comb = pd.DataFrame(woz_rows + avatar_rows)
    assert_no_sealed_ids(df_comb['participant_id'].values, context='generate_combined_folds')
    
    print(f'Total combined sessions: {len(df_comb)}')
    print(f'Total unique participants: {df_comb["participant_id"].nunique()}')
    print('Stratification key distribution:')
    print(df_comb['strat_key'].value_counts())
    
    folds_by_seed = {}
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds_list = []
        
        # Array of indices
        X_dummy = np.zeros(len(df_comb))
        y_strat = df_comb['strat_key'].values
        groups = df_comb['participant_id'].values
        
        for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X_dummy, y_strat, groups=groups)):
            # Verify no group leakage
            tr_pids = set(df_comb.iloc[tr_idx]['participant_id'])
            val_pids = set(df_comb.iloc[val_idx]['participant_id'])
            assert len(tr_pids.intersection(val_pids)) == 0, f'Group leakage in seed {seed} fold {fold_idx}!'
            
            folds_list.append({
                'fold': fold_idx,
                'train_sessions': df_comb.iloc[tr_idx]['session_id'].tolist(),
                'val_sessions': df_comb.iloc[val_idx]['session_id'].tolist(),
                'train_pids': sorted(list(tr_pids)),
                'val_pids': sorted(list(val_pids)),
                'train_count': len(tr_idx),
                'val_count': len(val_idx),
                'val_woz_count': int((df_comb.iloc[val_idx]['domain'] == 'woz').sum()),
                'val_avatar_count': int((df_comb.iloc[val_idx]['domain'] == 'avatar').sum())
            })
        folds_by_seed[str(seed)] = folds_list
        
    out_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'folds_combined.json')
    with open(out_path, 'w') as f:
        json.dump(folds_by_seed, f, indent=2)
    print(f'Successfully wrote combined folds to {out_path}')

if __name__ == '__main__':
    generate_combined_folds()
