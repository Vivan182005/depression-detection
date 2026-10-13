"""
Module: experiments/research_v2/dataset.py
Description: Data loading module for Phase 1 research experiments.
Strictly quarantines sealed cohort (N=33) and loads all development features + metadata.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids

def load_development_dataset() -> Dict[str, Any]:
    master_path = os.path.join(_REPO_ROOT, 'app', 'feature_store', 'master_dataset_index.csv')
    clean_labels_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'detailed_lables.csv')
    daicwoz_dir = os.path.join(_REPO_ROOT, 'Clean_audio', 'daicwoz')
    fs_dir = os.path.join(_REPO_ROOT, 'app', 'feature_store')
    
    df = pd.read_csv(master_path)
    clean_df = pd.read_csv(clean_labels_path)
    clean_map = dict(zip(clean_df['Participant'], clean_df['Depression_severity']))
    
    # Enforce quarantine
    assert_no_sealed_ids(df['participant_id'].values, context='load_development_dataset')
    
    ac_list = []
    w2v_list = []
    bert_list = []
    meta_list = []
    
    y_class = []
    y_phq = []
    y_bin = []
    pids = []
    sids = []
    
    for idx, row in df.iterrows():
        sid = str(row['session_id'])
        pid = int(row['participant_id'])
        
        # Load pre-extracted feature archive (.npz)
        npz_file = os.path.join(fs_dir, f'{sid}_features.npz')
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f'Missing feature file for session {sid} at {npz_file}')
        
        data = np.load(npz_file, allow_pickle=True)
        t = data['temporal_33d']
        t_mean = np.mean(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        t_std = np.std(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
        e = data['egemaps_88d'].astype(np.float32)
        w = data['wav2vec_768d'].astype(np.float32)
        b = data.get('text_bert_768d', np.zeros(768, dtype=np.float32)).astype(np.float32)
        
        ac_154 = np.hstack([e, t_mean, t_std]).astype(np.float32)
        ac_list.append(ac_154)
        w2v_list.append(w)
        bert_list.append(b)
        
        # Ground truth labels
        phq = float(clean_map.get(pid, row['phq8_score']))
        cls_id = 0 if phq <= 9.0 else (1 if phq <= 14.0 else 2)
        bin_label = 1 if phq >= 10.0 else 0
        
        y_class.append(cls_id)
        y_phq.append(phq)
        y_bin.append(bin_label)
        pids.append(pid)
        sids.append(sid)
        
        # Metadata / shortcut features from transcript & audio frames
        t_path = os.path.join(daicwoz_dir, f'{pid}_TRANSCRIPT.csv')
        num_frames = float(row.get('num_frames', len(t)))
        audio_dur = num_frames * 256.0 / 16000.0  # seconds
        
        if os.path.exists(t_path):
            try:
                tdf = pd.read_csv(t_path, sep='	')
                mask = tdf['speaker'].astype(str).str.strip().str.lower() == 'participant'
                part_tdf = tdf[mask]
                turn_count = float(len(part_tdf))
                spk_durations = part_tdf['stop_time'].astype(float) - part_tdf['start_time'].astype(float)
                spk_dur = float(spk_durations.sum())
                words = float(part_tdf['value'].dropna().astype(str).apply(lambda s: len(s.split())).sum())
                speech_rate = words / max(1.0, spk_dur)
                sess_time = float(tdf['stop_time'].max()) if len(tdf) > 0 else audio_dur
                speaking_ratio = spk_dur / max(1.0, sess_time)
                mean_turn_dur = spk_dur / max(1.0, turn_count)
            except Exception:
                turn_count = 0.0
                spk_dur = 0.0
                words = 0.0
                speech_rate = 0.0
                speaking_ratio = 0.0
                sess_time = audio_dur
                mean_turn_dur = 0.0
        else:
            turn_count = 0.0
            spk_dur = 0.0
            words = 0.0
            speech_rate = 0.0
            speaking_ratio = 0.0
            sess_time = audio_dur
            mean_turn_dur = 0.0
            
        meta_vec = np.array([
            turn_count,
            spk_dur,
            words,
            speech_rate,
            speaking_ratio,
            sess_time,
            mean_turn_dur
        ], dtype=np.float32)
        meta_list.append(meta_vec)
        
    X_ac = np.array(ac_list, dtype=np.float32)
    X_w2v = np.array(w2v_list, dtype=np.float32)
    X_bert = np.array(bert_list, dtype=np.float32)
    X_tri = np.hstack([X_ac, X_w2v, X_bert]).astype(np.float32)
    X_meta = np.array(meta_list, dtype=np.float32)
    
    y_class = np.array(y_class, dtype=np.int64)
    y_phq = np.array(y_phq, dtype=np.float32)
    y_bin = np.array(y_bin, dtype=np.int64)
    pids = np.array(pids, dtype=np.int64)
    sids = np.array(sids)
    
    return {
        'X_ac': X_ac,
        'X_w2v': X_w2v,
        'X_bert': X_bert,
        'X_tri': X_tri,
        'X_meta': X_meta,
        'y_class': y_class,
        'y_phq': y_phq,
        'y_bin': y_bin,
        'pids': pids,
        'sids': sids
    }
