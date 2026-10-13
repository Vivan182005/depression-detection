"""
Module: experiments/research_v2/dataset.py
Description: Data loading module for Phase 1, Phase 2, and Phase 3 research experiments.
Strictly quarantines sealed cohort (N=33) and loads development features + metadata
for WoZ-only (N=179 participants, 183 sessions), Avatar-only (N=61 participants, 61 sessions),
or Combined development cohorts (N=240 participants, 244 sessions).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids, SEALED_PIDS

def load_development_dataset(cohort: str = 'woz_only') -> Dict[str, Any]:
    """
    Loads development dataset with strict isolation of sealed PIDs.
    cohort can be 'woz_only', 'avatar_only', or 'combined'.
    """
    if cohort not in ['woz_only', 'avatar_only', 'combined']:
        raise ValueError(f"cohort must be 'woz_only', 'avatar_only', or 'combined', got {cohort}")

    manifest_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'expanded_dataset_manifest.csv')
    df_manifest = pd.read_csv(manifest_path)
    gender_map = dict(zip(df_manifest['participant_id'], df_manifest['gender']))
    age_map = dict(zip(df_manifest['participant_id'], df_manifest['age']))
    
    # 1. WoZ Dev Data Loading
    woz_data = None
    if cohort in ['woz_only', 'combined']:
        master_path = os.path.join(_REPO_ROOT, 'app', 'feature_store', 'master_dataset_index.csv')
        clean_labels_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'detailed_lables.csv')
        daicwoz_dir = os.path.join(_REPO_ROOT, 'Clean_audio', 'daicwoz')
        fs_dir = os.path.join(_REPO_ROOT, 'app', 'feature_store')
        
        df = pd.read_csv(master_path)
        clean_df = pd.read_csv(clean_labels_path)
        clean_df = clean_df[~clean_df['Participant'].isin(SEALED_PIDS)]
        clean_map = dict(zip(clean_df['Participant'], clean_df['Depression_severity']))
        
        assert_no_sealed_ids(df['participant_id'].values, context='load_development_dataset_woz')
        
        ac_list, w2v_list, bert_list, meta_list = [], [], [], []
        y_class, y_phq, y_bin = [], [], []
        pids, sids, domains, genders, ages = [], [], [], [], []
        
        for idx, row in df.iterrows():
            sid = str(row['session_id'])
            pid = int(row['participant_id'])
            
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
            
            phq = float(clean_map.get(pid, row['phq8_score']))
            cls_id = 0 if phq <= 9.0 else (1 if phq <= 14.0 else 2)
            bin_label = 1 if phq >= 10.0 else 0
            
            y_class.append(cls_id)
            y_phq.append(phq)
            y_bin.append(bin_label)
            pids.append(pid)
            sids.append(sid)
            domains.append('woz')
            genders.append(gender_map.get(pid, 'unknown'))
            ages.append(age_map.get(pid, np.nan))
            
            # Transcript metadata
            t_path = os.path.join(daicwoz_dir, f'{pid}_TRANSCRIPT.csv')
            num_frames = float(row.get('num_frames', len(t)))
            audio_dur = num_frames * 256.0 / 16000.0
            
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
                    turn_count, spk_dur, words, speech_rate, speaking_ratio, sess_time, mean_turn_dur = 0.0, 0.0, 0.0, 0.0, 0.0, audio_dur, 0.0
            else:
                turn_count, spk_dur, words, speech_rate, speaking_ratio, sess_time, mean_turn_dur = 0.0, 0.0, 0.0, 0.0, 0.0, audio_dur, 0.0
                
            meta_vec = np.array([turn_count, spk_dur, words, speech_rate, speaking_ratio, sess_time, mean_turn_dur], dtype=np.float32)
            meta_list.append(meta_vec)
            
        woz_data = {
            'X_ac': np.array(ac_list, dtype=np.float32),
            'X_w2v': np.array(w2v_list, dtype=np.float32),
            'X_bert': np.array(bert_list, dtype=np.float32),
            'X_tri': np.hstack([np.array(ac_list, dtype=np.float32), np.array(w2v_list, dtype=np.float32), np.array(bert_list, dtype=np.float32)]).astype(np.float32),
            'X_meta': np.array(meta_list, dtype=np.float32),
            'y_class': np.array(y_class, dtype=np.int64),
            'y_phq': np.array(y_phq, dtype=np.float32),
            'y_bin': np.array(y_bin, dtype=np.int64),
            'pids': np.array(pids, dtype=np.int64),
            'sids': np.array(sids),
            'domains': np.array(domains),
            'genders': np.array(genders),
            'ages': np.array(ages, dtype=np.float32)
        }

    # 2. Avatar Dev Data Loading
    avatar_data = None
    if cohort in ['avatar_only', 'combined']:
        avatar_cache_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_cache_avatar_dev')
        avatar_target = df_manifest[df_manifest['cohort_role'] == 'avatar_candidate_expansion']
        assert_no_sealed_ids(avatar_target['participant_id'].values, context='load_development_dataset_avatar')
        
        ac_list, w2v_list, bert_list, meta_list = [], [], [], []
        y_class, y_phq, y_bin = [], [], []
        pids, sids, domains, genders, ages = [], [], [], [], []
        
        for idx, row in avatar_target.iterrows():
            pid = int(row['participant_id'])
            sid = str(pid)
            npz_file = os.path.join(avatar_cache_dir, f'{pid}_features.npz')
            if not os.path.exists(npz_file):
                raise FileNotFoundError(f'Missing feature file for Avatar PID {pid} at {npz_file}')
                
            data = np.load(npz_file, allow_pickle=True)
            t = data['temporal_33d']
            t_mean = np.mean(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
            t_std = np.std(t, axis=0) if len(t) > 0 else np.zeros(33, dtype=np.float32)
            e = data['egemaps_88d'].astype(np.float32)
            w = data['wav2vec_768d'].astype(np.float32)
            b = data['text_bert_768d'].astype(np.float32)
            
            ac_154 = np.hstack([e, t_mean, t_std]).astype(np.float32)
            ac_list.append(ac_154)
            w2v_list.append(w)
            bert_list.append(b)
            
            phq = float(row['phq8_score'])
            cls_id = int(row['class_id'])
            bin_label = 1 if phq >= 10.0 else 0
            
            y_class.append(cls_id)
            y_phq.append(phq)
            y_bin.append(bin_label)
            pids.append(pid)
            sids.append(sid)
            domains.append('avatar')
            genders.append(row['gender'])
            ages.append(float(row['age']) if not pd.isna(row['age']) else np.nan)
            
            audio_dur = float(row.get('audio_duration_sec', len(t) * 256.0 / 16000.0))
            meta_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, audio_dur, 0.0], dtype=np.float32)
            meta_list.append(meta_vec)
            
        avatar_data = {
            'X_ac': np.array(ac_list, dtype=np.float32),
            'X_w2v': np.array(w2v_list, dtype=np.float32),
            'X_bert': np.array(bert_list, dtype=np.float32),
            'X_tri': np.hstack([np.array(ac_list, dtype=np.float32), np.array(w2v_list, dtype=np.float32), np.array(bert_list, dtype=np.float32)]).astype(np.float32),
            'X_meta': np.array(meta_list, dtype=np.float32),
            'y_class': np.array(y_class, dtype=np.int64),
            'y_phq': np.array(y_phq, dtype=np.float32),
            'y_bin': np.array(y_bin, dtype=np.int64),
            'pids': np.array(pids, dtype=np.int64),
            'sids': np.array(sids),
            'domains': np.array(domains),
            'genders': np.array(genders),
            'ages': np.array(ages, dtype=np.float32)
        }

    if cohort == 'woz_only':
        return woz_data
    elif cohort == 'avatar_only':
        return avatar_data
    else:
        woz_pids_set = set(woz_data['pids'])
        avatar_pids_set = set(avatar_data['pids'])
        overlap = woz_pids_set.intersection(avatar_pids_set)
        if len(overlap) > 0:
            raise RuntimeError(f'Overlap detected between WoZ and Avatar cohorts: {overlap}')
            
        return {
            'X_ac': np.vstack([woz_data['X_ac'], avatar_data['X_ac']]),
            'X_w2v': np.vstack([woz_data['X_w2v'], avatar_data['X_w2v']]),
            'X_bert': np.vstack([woz_data['X_bert'], avatar_data['X_bert']]),
            'X_tri': np.vstack([woz_data['X_tri'], avatar_data['X_tri']]),
            'X_meta': np.vstack([woz_data['X_meta'], avatar_data['X_meta']]),
            'y_class': np.concatenate([woz_data['y_class'], avatar_data['y_class']]),
            'y_phq': np.concatenate([woz_data['y_phq'], avatar_data['y_phq']]),
            'y_bin': np.concatenate([woz_data['y_bin'], avatar_data['y_bin']]),
            'pids': np.concatenate([woz_data['pids'], avatar_data['pids']]),
            'sids': np.concatenate([woz_data['sids'], avatar_data['sids']]),
            'domains': np.concatenate([woz_data['domains'], avatar_data['domains']]),
            'genders': np.concatenate([woz_data['genders'], avatar_data['genders']]),
            'ages': np.concatenate([woz_data['ages'], avatar_data['ages']])
        }
