import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = '/Users/vivannaik/Desktop/CAPSTONE'
APP_DIR = os.path.join(BASE_DIR, 'app')
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(APP_DIR, 'src'))

from src.features.feature_store import FeatureStore

def get_development_dataset(clean_labels=True):
    master_path = os.path.join(APP_DIR, 'feature_store', 'master_dataset_index.csv')
    df = pd.read_csv(master_path)
    
    clean_labels_df = pd.read_csv(os.path.join(BASE_DIR, 'new_audio_csv', 'detailed_lables.csv'))
    clean_map = dict(zip(clean_labels_df['Participant'], clean_labels_df['Depression_severity']))
    
    fs = FeatureStore(os.path.join(APP_DIR, 'feature_store'))
    
    ac_list, w2v_list, bert_list = [], [], []
    y_class, y_phq, pids, sids, genders = [], [], [], [], []
    
    for idx, row in df.iterrows():
        sid = str(row['session_id'])
        pid = int(row['participant_id'])
        f = fs.load_features(sid)
        
        t = f['temporal_33d']
        t_mean = np.mean(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        t_std = np.std(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        e = f['egemaps_88d']
        w = f['wav2vec_768d']
        b = f.get('text_bert_768d', np.zeros(768, dtype=np.float32))
        
        ac_154 = np.hstack([e, t_mean, t_std]).astype(np.float32)
        ac_list.append(ac_154)
        w2v_list.append(w.astype(np.float32))
        bert_list.append(b.astype(np.float32))
        
        if clean_labels and pid in clean_map:
            phq = float(clean_map[pid])
        else:
            phq = float(row['phq8_score'])
            
        c_id = 0 if phq <= 9.0 else (1 if phq <= 14.0 else 2)
        y_class.append(c_id)
        y_phq.append(phq)
        pids.append(pid)
        sids.append(sid)
        genders.append(int(row['gender']) if 'gender' in row and not pd.isna(row['gender']) else 0)
        
    X_ac = np.array(ac_list, dtype=np.float32)
    X_w2v = np.array(w2v_list, dtype=np.float32)
    X_bert = np.array(bert_list, dtype=np.float32)
    y_class = np.array(y_class, dtype=np.int64)
    y_phq = np.array(y_phq, dtype=np.float32)
    pids = np.array(pids, dtype=np.int64)
    sids = np.array(sids)
    genders = np.array(genders, dtype=np.int64)
    
    return {
        'X_ac': X_ac,
        'X_w2v': X_w2v,
        'X_bert': X_bert,
        'y_class': y_class,
        'y_phq': y_phq,
        'pids': pids,
        'sids': sids,
        'genders': genders
    }

def get_stratified_group_folds(data, n_splits=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    # Stratify by class, group by participant_id
    for train_idx, val_idx in sgkf.split(data['X_ac'], data['y_class'], groups=data['pids']):
        folds.append((train_idx, val_idx))
    return folds

if __name__ == '__main__':
    data = get_development_dataset()
    print(f'Loaded dataset: X_ac={data["X_ac"].shape}, X_w2v={data["X_w2v"].shape}, X_bert={data["X_bert"].shape}')
    print(f'Total sessions: {len(data["pids"])}, unique participants: {len(np.unique(data["pids"]))}')
    print('Class counts:', np.bincount(data['y_class']))
    folds = get_stratified_group_folds(data)
    for i, (tr, val) in enumerate(folds):
        tr_pids = set(data['pids'][tr])
        val_pids = set(data['pids'][val])
        overlap = tr_pids.intersection(val_pids)
        assert len(overlap) == 0, f'Leakage in fold {i}!'
        print(f'Fold {i+1}: Train={len(tr)} sessions ({len(tr_pids)} PIDs), Val={len(val)} sessions ({len(val_pids)} PIDs), Overlap={len(overlap)}')
