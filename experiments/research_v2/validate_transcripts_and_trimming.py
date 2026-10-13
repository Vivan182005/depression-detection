"""
Script: experiments/research_v2/validate_transcripts_and_trimming.py
Description: Audits Whisper transcripts for technician setup dialogue, Ellie intro phrases,
and conversational start boundaries. Version-separates trimmed transcripts from raw transcripts.
Flags ambiguous sessions with 'uncertain' or 'not_detected' to defer participant-only extraction.
"""

import os
import re
import pandas as pd
import glob

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev')
TRIMMED_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev_trimmed')
AUDIT_CSV = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'trimming_audit_manifest.csv')

SETUP_PATTERNS = [
    re.compile(r"virtual human", re.IGNORECASE),
    re.compile(r"fly in here", re.IGNORECASE),
    re.compile(r"start her for you", re.IGNORECASE),
    re.compile(r"she looks like", re.IGNORECASE),
    re.compile(r"play the sims", re.IGNORECASE),
    re.compile(r"shrink her", re.IGNORECASE),
    re.compile(r"frozen or something", re.IGNORECASE),
    re.compile(r"mortal combat", re.IGNORECASE),
    re.compile(r"goodbye that means I'm done", re.IGNORECASE),
    re.compile(r"press this button", re.IGNORECASE),
    re.compile(r"there she is", re.IGNORECASE),
    re.compile(r"is that my phone", re.IGNORECASE),
    re.compile(r"turn it off", re.IGNORECASE),
    re.compile(r"adjust the mic", re.IGNORECASE)
]

ELLIE_GREETINGS = [
    re.compile(r"(hi,? I'm ellie)", re.IGNORECASE),
    re.compile(r"(thanks for coming in today)", re.IGNORECASE),
    re.compile(r"(safe and secure environment)", re.IGNORECASE),
    re.compile(r"(think of me as a friend)", re.IGNORECASE)
]

CLINICAL_RESPONSES = [
    re.compile(r"(doing fine|doing good|doing alright|I'm fine|I am fine|pretty good)", re.IGNORECASE),
    re.compile(r"(originally from|born in|raised in)", re.IGNORECASE),
    re.compile(r"(living situation|my family|roommates)", re.IGNORECASE),
    re.compile(r"(in the military|in construction|in the army|navy)", re.IGNORECASE),
    re.compile(r"(the weather|the people|not much|not really)", re.IGNORECASE)
]

def audit_session(pid, raw_csv_path):
    df_raw = pd.read_csv(raw_csv_path)
    if len(df_raw) == 0:
        return {
            'participant_id': pid,
            'status': 'not_detected',
            'interview_start_sec': 0.0,
            'detection_method': 'empty_transcript',
            'detected_phrase': '',
            'setup_dialogue_detected': False,
            'confidence': 'not_applicable',
            'defer_participant_only_extraction': True,
            'notes': 'Raw transcript file is empty.'
        }
        
    setup_found = False
    setup_phrases = []
    setup_end_time = 0.0
    setup_last_chunk_idx = -1
    
    # 1. Scan first 5 chunks or first 90 seconds for setup phrases
    for idx, row in df_raw.iterrows():
        s_time = row['start_time'] if pd.notna(row['start_time']) else 0.0
        if s_time > 90.0:
            break
        txt = str(row['text'])
        for pat in SETUP_PATTERNS:
            m = pat.search(txt)
            if m:
                setup_found = True
                setup_phrases.append(m.group(0))
                setup_end_time = row['stop_time'] if pd.notna(row['stop_time']) else s_time
                setup_last_chunk_idx = idx

    # 2. Check for Ellie greeting
    ellie_found = False
    ellie_phrase = ''
    ellie_time = 0.0
    ellie_chunk_idx = -1
    for idx, row in df_raw.iterrows():
        txt = str(row['text'])
        for pat in ELLIE_GREETINGS:
            m = pat.search(txt)
            if m:
                ellie_found = True
                ellie_phrase = m.group(0)
                ellie_time = row['start_time'] if pd.notna(row['start_time']) else 0.0
                ellie_chunk_idx = idx
                break
        if ellie_found:
            break

    # 3. Classify decision
    if setup_found:
        # Search for first clinical response AFTER setup
        transition_found = False
        start_ts = setup_end_time
        for idx in range(setup_last_chunk_idx + 1, min(len(df_raw), setup_last_chunk_idx + 4)):
            row = df_raw.iloc[idx]
            txt = str(row['text'])
            for pat in CLINICAL_RESPONSES:
                if pat.search(txt):
                    transition_found = True
                    start_ts = row['start_time'] if pd.notna(row['start_time']) else setup_end_time
                    break
            if transition_found:
                break
                
        if transition_found:
            return {
                'participant_id': pid,
                'status': 'accepted_setup_trimmed',
                'interview_start_sec': round(start_ts, 2),
                'detection_method': 'setup_dialogue_transition',
                'detected_phrase': '; '.join(set(setup_phrases)),
                'setup_dialogue_detected': True,
                'confidence': 'rule_based_deterministic',
                'defer_participant_only_extraction': False,
                'notes': f'Setup dialogue detected and trimmed up to {start_ts:.1f}s.'
            }
        else:
            return {
                'participant_id': pid,
                'status': 'uncertain',
                'interview_start_sec': round(setup_end_time, 2),
                'detection_method': 'setup_detected_transition_ambiguous',
                'detected_phrase': '; '.join(set(setup_phrases)),
                'setup_dialogue_detected': True,
                'confidence': 'uncertain',
                'defer_participant_only_extraction': True,
                'notes': 'Setup dialogue detected but interview transition boundary is ambiguous. Defer participant-only extraction.'
            }
            
    elif ellie_found:
        return {
            'participant_id': pid,
            'status': 'accepted_setup_trimmed',
            'interview_start_sec': round(ellie_time, 2),
            'detection_method': 'ellie_greeting_matched',
            'detected_phrase': ellie_phrase,
            'setup_dialogue_detected': False,
            'confidence': 'rule_based_deterministic',
            'defer_participant_only_extraction': False,
            'notes': f'Standard Ellie greeting matched at {ellie_time:.1f}s.'
        }
        
    else:
        # Check if participant answers standard clinical prompt immediately (at chunk 0)
        c0_txt = str(df_raw.iloc[0]['text']) if len(df_raw) > 0 else ''
        clinical_match = any(pat.search(c0_txt) for pat in CLINICAL_RESPONSES)
        if clinical_match:
            return {
                'participant_id': pid,
                'status': 'accepted_immediate_start',
                'interview_start_sec': 0.0,
                'detection_method': 'immediate_clinical_response_0s',
                'detected_phrase': '',
                'setup_dialogue_detected': False,
                'confidence': 'rule_based_deterministic',
                'defer_participant_only_extraction': False,
                'notes': 'No setup detected; participant provides immediate clinical responses from 0.0s.'
            }
        else:
            return {
                'participant_id': pid,
                'status': 'not_detected',
                'interview_start_sec': 0.0,
                'detection_method': 'no_greeting_no_setup_found',
                'detected_phrase': '',
                'setup_dialogue_detected': False,
                'confidence': 'uncalibrated',
                'defer_participant_only_extraction': True,
                'notes': 'Neither setup dialogue nor standard greeting detected; conversational start is uncertain.'
            }

def run_trimming_audit():
    os.makedirs(TRIMMED_DIR, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, '*_TRANSCRIPT_WHISPER.csv')))
    print(f'Auditing {len(raw_files)} raw Whisper transcripts in {RAW_DIR}...')
    
    records = []
    for p in raw_files:
        fname = os.path.basename(p)
        pid = int(re.match(r'([0-9]{3})', fname).group(1))
        audit = audit_session(pid, p)
        records.append(audit)
        
        # If accepted, produce versioned trimmed transcript
        if audit['status'] in ['accepted_setup_trimmed', 'accepted_immediate_start']:
            df_raw = pd.read_csv(p)
            start_ts = audit['interview_start_sec']
            df_trimmed = df_raw[df_raw['start_time'] >= start_ts].copy() if start_ts > 0 else df_raw.copy()
            trimmed_csv = os.path.join(TRIMMED_DIR, f'{pid}_TRANSCRIPT_WHISPER_TRIMMED.csv')
            df_trimmed.to_csv(trimmed_csv, index=False)
            
    df_audit = pd.DataFrame(records)
    df_audit.to_csv(AUDIT_CSV, index=False)
    print(f'Saved trimming audit manifest ({len(df_audit)} sessions) to {AUDIT_CSV}')
    print('=== Status Breakdown ===')
    print(df_audit['status'].value_counts())
    print('=== Defer Participant-Only Extraction Flag ===')
    print(df_audit['defer_participant_only_extraction'].value_counts())
    return df_audit

if __name__ == '__main__':
    run_trimming_audit()