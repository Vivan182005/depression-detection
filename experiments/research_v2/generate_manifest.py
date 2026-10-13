"""
Script: experiments/research_v2/generate_manifest.py
Description: Generates the comprehensive participant-level manifest for all 275 catalogued
participants in DAIC-WOZ and E-DAIC. Strictly enforces sealed quarantine boundary.
"""

import os
import re
import json
import pandas as pd
import soundfile as sf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def generate_participant_manifest():
    print("Generating comprehensive participant manifest...")
    labels_path = os.path.join(_REPO_ROOT, 'new_audio_csv', 'detailed_lables.csv')
    df_labels = pd.read_csv(labels_path)
    
    with open(os.path.join(_REPO_ROOT, 'sealed_ids.json')) as f:
        sealed_pids = set(json.load(f)['sealed_pids'])
        
    master_dev_df = pd.read_csv(os.path.join(_REPO_ROOT, 'app', 'feature_store', 'master_dataset_index.csv'))
    dev_store_pids = set(master_dev_df['participant_id'].tolist())
    
    # Locate all audio files
    audio_dict = {}
    for root, dirs, files in os.walk(_REPO_ROOT):
        if '.git' in root: continue
        for f in files:
            if f.endswith('.wav'):
                p = os.path.join(root, f)
                m = re.search(r'([0-9]{3})', f)
                if m:
                    pid = int(m.group(1))
                    if pid not in audio_dict:
                        audio_dict[pid] = []
                    audio_dict[pid].append(p)
                    
    # Locate all existing transcripts
    transcript_dict = {}
    for root, dirs, files in os.walk(_REPO_ROOT):
        if '.git' in root: continue
        for f in files:
            if f.endswith('.csv') and ('transcript' in f.lower() or 'TRANSCRIPT' in f):
                p = os.path.join(root, f)
                m = re.search(r'([0-9]{3})', f)
                if m:
                    pid = int(m.group(1))
                    if pid not in transcript_dict:
                        transcript_dict[pid] = []
                    transcript_dict[pid].append(p)
                    
    records = []
    for idx, row in df_labels.iterrows():
        pid = int(row['Participant'])
        
        # Domain and release provenance
        if 300 <= pid < 500:
            domain = 'human_woz'
            release = 'DAIC-WOZ_2016'
        elif 600 <= pid < 700:
            domain = 'autonomous_avatar'
            release = 'E-DAIC_AVEC_2019'
        elif 700 <= pid < 800:
            domain = 'autonomous_avatar'
            release = 'E-DAIC_AVEC_2019'
        else:
            domain = 'unknown'
            release = 'unknown'
            
        phq = float(row['Depression_severity']) if pd.notna(row['Depression_severity']) else None
        cls_id = int(0 if phq <= 9.0 else (1 if phq <= 14.0 else 2)) if phq is not None else None
        
        gender = str(row['gender']).lower() if pd.notna(row['gender']) else 'unknown'
        age = float(row['age']) if pd.notna(row['age']) else None
        
        is_sealed = pid in sealed_pids
        audio_files = audio_dict.get(pid, [])
        audio_avail = len(audio_files) > 0
        
        dur = None
        rel_audio = None
        if audio_avail:
            chosen_audio = None
            for af in audio_files:
                if 'unseen' in af and is_sealed:
                    chosen_audio = af; break
                elif 'NEW_AUDIO_DAIC' in af and not is_sealed:
                    chosen_audio = af; break
                elif 'Clean_audio/NEW_AUDIO' in af:
                    chosen_audio = af; break
            if not chosen_audio:
                chosen_audio = audio_files[0]
            try:
                info = sf.info(chosen_audio)
                dur = round(info.duration, 2)
            except Exception:
                pass
            rel_audio = os.path.relpath(chosen_audio, _REPO_ROOT)
            
        trans_files = transcript_dict.get(pid, [])
        trans_avail = len(trans_files) > 0
        rel_trans = None
        trans_source = 'none'
        if trans_avail:
            chosen_trans = trans_files[0]
            rel_trans = os.path.relpath(chosen_trans, _REPO_ROOT)
            if 'Clean_audio/daicwoz' in chosen_trans:
                trans_source = 'official_daicwoz_annotated'
            elif 'unseen/transcripts' in chosen_trans:
                trans_source = 'legacy_asr_sealed_csv'
            elif 'NEW_AUDIO_DAIC' in chosen_trans:
                trans_source = 'legacy_asr_csv'
            else:
                trans_source = 'other'
                
        # Cohort assignment
        if is_sealed:
            role = 'sealed_quarantine'
            exclusion = 'quarantined_sealed_cohort_evaluated_phase6_only'
        elif pid == 306:
            role = 'dev_active_woz'
            exclusion = 'features_cached_audio_missing_on_disk'
        elif not audio_avail:
            role = 'excluded_missing_data'
            exclusion = 'audio_file_missing_on_disk'
        elif pid in dev_store_pids:
            role = 'dev_active_woz'
            exclusion = 'none'
        elif domain == 'autonomous_avatar':
            role = 'avatar_candidate_expansion'
            exclusion = 'none'
        else:
            role = 'excluded_unspecified'
            exclusion = 'not_in_dev_store'
            
        records.append({
            'participant_id': pid,
            'domain': domain,
            'cohort_role': role,
            'session_count': 1,
            'dataset_release': release,
            'phq8_score': phq,
            'class_id': cls_id,
            'gender': gender,
            'age': age,
            'audio_available': audio_avail,
            'audio_duration_sec': dur,
            'audio_path': rel_audio,
            'transcript_available': trans_avail,
            'transcript_path': rel_trans,
            'transcript_source': trans_source,
            'is_sealed': is_sealed,
            'exclusion_reason': exclusion
        })
        
    df_out = pd.DataFrame(records)
    out_csv = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'expanded_dataset_manifest.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Manifest successfully generated with {len(df_out)} participants -> {out_csv}")
    return df_out

if __name__ == '__main__':
    generate_participant_manifest()
