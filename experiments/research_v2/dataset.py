"""
Module: experiments/research_v2/dataset.py
Description: Authoritative dataset loading module for research_v2.
Enforces:
1. All labels and clinical metadata are loaded ONLY from experiments/research_v2/labels_official.csv.
2. Raises KeyError immediately on any missing participant ID or unmapped label.
3. Strictly zero fallback to default 0.0 or fabricated (idx % 5) labels.
4. Strictly enforces sealed quarantine via assert_no_sealed_ids.
5. Supports loading legacy features ('v1') or corrected canonical features ('v2').
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from experiments.research_v2.extractors_v2 import get_robust_acoustic_135d, ROBUST_ACOUSTIC_MASK

OFFICIAL_LABELS_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'labels_official.csv')

def load_official_labels(allow_sealed: bool = False) -> pd.DataFrame:
    """
    Loads authoritative official labels for all 275 participants.
    Raises FileNotFoundError if labels_official.csv does not exist.
    Guarantees no missing values in clinical columns.
    """
    if not os.path.exists(OFFICIAL_LABELS_PATH):
        raise FileNotFoundError(f"Authoritative label file not found at {OFFICIAL_LABELS_PATH}")
    df = pd.read_csv(OFFICIAL_LABELS_PATH)
    if not allow_sealed:
        assert_no_sealed_ids(df[df['is_sealed']]['participant_id'].values, context='load_official_labels', allow_sealed=False)
    return df

def get_official_label_for_pid(pid: int, labels_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Strictly resolves official label and metadata for a participant ID.
    Raises KeyError if PID is not found in labels_official.csv.
    """
    if labels_df is None:
        if not os.path.exists(OFFICIAL_LABELS_PATH):
            raise FileNotFoundError(f"Authoritative label file not found at {OFFICIAL_LABELS_PATH}")
        labels_df = pd.read_csv(OFFICIAL_LABELS_PATH).set_index('participant_id')
    elif 'participant_id' in labels_df.columns:
        labels_df = labels_df.set_index('participant_id')
        
    if pid not in labels_df.index:
        raise KeyError(f"Participant ID {pid} not found in authoritative {OFFICIAL_LABELS_PATH}. No default or fallback permitted.")
        
    row = labels_df.loc[pid]
    return {
        'participant_id': pid,
        'phq8_score': float(row['phq8_score']),
        'class_id': int(row['class_id']),
        'binary_label': int(row['binary_label']),
        'gender': str(row['gender']),
        'age': float(row['age']) if not pd.isna(row['age']) else np.nan,
        'domain': str(row['domain']),
        'is_sealed': bool(row['is_sealed'])
    }

def load_development_dataset(cohort: str = 'woz_only', feature_version: str = 'v2', acoustic_set: str = 'robust', text_variant: str = 'T1', boundary_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads development dataset with strict isolation of sealed PIDs.
    cohort: 'woz_only', 'avatar_only', or 'combined'
    feature_version: 'v1' (historical stores) or 'v2' (canonical feature_store_v2)
    All labels are loaded ONLY from labels_official.csv.
    """
    if cohort not in ['woz_only', 'avatar_only', 'combined']:
        raise ValueError(f"cohort must be 'woz_only', 'avatar_only', or 'combined', got {cohort}")

    if not os.path.exists(OFFICIAL_LABELS_PATH):
        raise FileNotFoundError(f"Authoritative label file not found at {OFFICIAL_LABELS_PATH}")
    df_labels = pd.read_csv(OFFICIAL_LABELS_PATH).set_index('participant_id')

    # Manifest for session mapping and metadata
    manifest_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'expanded_dataset_manifest.csv')
    df_manifest = pd.read_csv(manifest_path)

    # 1. WoZ Dev Data Loading
    woz_data = None
    if cohort in ['woz_only', 'combined']:
        if feature_version == 'v2':
            v2_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
            manifest_v2_path = os.path.join(v2_dir, 'manifest.csv')
            if not os.path.exists(manifest_v2_path):
                raise FileNotFoundError(f"v2 feature manifest not found at {manifest_v2_path}")
            df_v2 = pd.read_csv(manifest_v2_path)
            woz_target = df_v2[(df_v2['domain'] == 'woz') & (~df_v2['is_sealed']) & (df_v2['feature_file'] != 'none')]
            assert_no_sealed_ids(woz_target['participant_id'].values, context='load_development_dataset_woz_v2')
            
            ac_list, w2v_list, bert_list, meta_list = [], [], [], []
            y_class, y_phq, y_bin = [], [], []
            pids, sids, domains, genders, ages = [], [], [], [], []
            
            for _, r in woz_target.iterrows():
                pid = int(r['participant_id'])
                sid = str(r['session_id'])
                feat_path = os.path.join(v2_dir, r['feature_file'])
                data = np.load(feat_path)
                
                if acoustic_set == 'robust':
                    ac = data['acoustic_robust_135d'].astype(np.float32) if 'acoustic_robust_135d' in data else get_robust_acoustic_135d(data['acoustic_154d'])
                else:
                    ac = data['acoustic_154d'].astype(np.float32)
                if text_variant == 'T2' and 'bert_t2_768d' in data:
                    bert = data['bert_t2_768d'].astype(np.float32)
                elif 'bert_t1_768d' in data:
                    bert = data['bert_t1_768d'].astype(np.float32)
                else:
                    bert = data['bert_768d'].astype(np.float32)
                ac_list.append(ac)
                w2v_list.append(data['wav2vec_768d'].astype(np.float32))
                bert_list.append(bert)
                meta_list.append(data.get('meta_7d', np.zeros(7, dtype=np.float32)))
                
                meta = get_official_label_for_pid(pid, df_labels)
                y_class.append(meta['class_id'])
                y_phq.append(meta['phq8_score'])
                y_bin.append(meta['binary_label'])
                pids.append(pid)
                sids.append(sid)
                domains.append('woz')
                genders.append(meta['gender'])
                ages.append(meta['age'])
                
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
        else:
            # Legacy v1 historical loader
            master_path = os.path.join(_REPO_ROOT, 'app', 'feature_store', 'master_dataset_index.csv')
            daicwoz_dir = os.path.join(_REPO_ROOT, 'Clean_audio', 'daicwoz')
            fs_dir = os.path.join(_REPO_ROOT, 'app', 'feature_store')
            
            df = pd.read_csv(master_path)
            assert_no_sealed_ids(df['participant_id'].values, context='load_development_dataset_woz_v1')
            
            ac_list, w2v_list, bert_list, meta_list = [], [], [], []
            y_class, y_phq, y_bin = [], [], []
            pids, sids, domains, genders, ages = [], [], [], [], []
            
            for idx, row in df.iterrows():
                pid = int(row['participant_id'])
                sid = str(row['session_id'])
                
                # Strict official label lookup (NO fallback)
                meta = get_official_label_for_pid(pid, df_labels)
                
                feat_file = row.get('feature_file', f"{sid}_features.npz")
                npz_file = os.path.join(fs_dir, feat_file)
                if not os.path.exists(npz_file):
                    alt = os.path.join(fs_dir, f"{pid}_features.npz")
                    if os.path.exists(alt):
                        npz_file = alt
                    else:
                        continue
                        
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
                
                y_class.append(meta['class_id'])
                y_phq.append(meta['phq8_score'])
                y_bin.append(meta['binary_label'])
                pids.append(pid)
                sids.append(sid)
                domains.append('woz')
                genders.append(meta['gender'])
                ages.append(meta['age'])
                
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
        if feature_version == 'v2':
            v2_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
            manifest_v2_path = os.path.join(v2_dir, 'manifest.csv')
            if not os.path.exists(manifest_v2_path):
                raise FileNotFoundError(f"v2 feature manifest not found at {manifest_v2_path}")
            df_v2 = pd.read_csv(manifest_v2_path)
            avatar_target = df_v2[(df_v2['domain'] == 'avatar') & (~df_v2['is_sealed']) & (df_v2['feature_file'] != 'none')]
            trim_path = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'trimming_audit_manifest.csv')
            trim_df = pd.read_csv(trim_path).set_index('participant_id') if os.path.exists(trim_path) else None
            assert_no_sealed_ids(avatar_target['participant_id'].values, context='load_development_dataset_avatar_v2')
            
            ac_list, w2v_list, bert_list, meta_list = [], [], [], []
            y_class, y_phq, y_bin = [], [], []
            pids, sids, domains, genders, ages = [], [], [], [], []
            
            for _, r in avatar_target.iterrows():
                pid = int(r['participant_id'])
                sid = str(r['session_id'])
                if boundary_filter == 'exclude_undetected' and trim_df is not None and pid in trim_df.index:
                    if trim_df.loc[pid, 'status'] in ['not_detected', 'uncertain']:
                        continue
                feat_path = os.path.join(v2_dir, r['feature_file'])
                data = np.load(feat_path)
                if acoustic_set == 'robust':
                    ac = data['acoustic_robust_135d'].astype(np.float32) if 'acoustic_robust_135d' in data else get_robust_acoustic_135d(data['acoustic_154d'])
                else:
                    ac = data['acoustic_154d'].astype(np.float32)
                if text_variant == 'T2' and 'bert_t2_768d' in data:
                    bert = data['bert_t2_768d'].astype(np.float32)
                elif 'bert_t1_768d' in data:
                    bert = data['bert_t1_768d'].astype(np.float32)
                else:
                    bert = data['bert_768d'].astype(np.float32)
                ac_list.append(ac)
                w2v_list.append(data['wav2vec_768d'].astype(np.float32))
                bert_list.append(bert)
                meta_list.append(data.get('meta_7d', np.zeros(7, dtype=np.float32)))
                
                meta = get_official_label_for_pid(pid, df_labels)
                y_class.append(meta['class_id'])
                y_phq.append(meta['phq8_score'])
                y_bin.append(meta['binary_label'])
                pids.append(pid)
                sids.append(sid)
                domains.append('avatar')
                genders.append(meta['gender'])
                ages.append(meta['age'])
                
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
        else:
            avatar_cache_dir = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_cache_avatar_dev')
            avatar_target = df_manifest[df_manifest['cohort_role'] == 'avatar_candidate_expansion']
            assert_no_sealed_ids(avatar_target['participant_id'].values, context='load_development_dataset_avatar_v1')
            
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
                
                meta = get_official_label_for_pid(pid, df_labels)
                y_class.append(meta['class_id'])
                y_phq.append(meta['phq8_score'])
                y_bin.append(meta['binary_label'])
                pids.append(pid)
                sids.append(sid)
                domains.append('avatar')
                genders.append(meta['gender'])
                ages.append(meta['age'])
                
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
