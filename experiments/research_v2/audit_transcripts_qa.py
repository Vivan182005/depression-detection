"""
Script: experiments/research_v2/audit_transcripts_qa.py
Step S3: Label-Blind Transcript QA Audit with Pre-Stated Stop Thresholds.
Evaluates:
- Empty / near-empty transcripts (<20 words).
- Repetition loops (any 10-gram repeated >= 5 times).
- Word Error Rate (WER) of Whisper-tiny vs manual transcripts for all WoZ dev sessions.
- Text Domain-AUC on BERT-T2 embeddings (evaluating domain separation).
- Word counts by domain.
- Pre-stated stop gates:
  * Stop if >5% of sessions (>11 sessions) are empty or loop-affected.
  * Stop if median WoZ WER > 0.60.
Zero clinical labels accessed.
"""
import os
import sys
import time
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from test_label_blind_extraction import assert_sealed_quarantine, assert_label_blind

WOZ_WHISPER_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'woz_whisper')
AVATAR_WHISPER_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_avatar_dev')
WOZ_MANUAL_DIR = os.path.join(_REPO_ROOT, 'Clean_audio', 'daicwoz')
FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
FEATURE_STORE_V3_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3')
MANIFEST_PATH = os.path.join(FEATURE_STORE_V3_DIR, 'manifest_v3.csv')
QA_REPORT_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcript_qa_report.json')


def normalize_words(text: str) -> List[str]:
    """Cleans text and tokenizes into lowercase alphanumeric words."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return [w for w in t.split() if w]


def check_repetition_loops(words: List[str], n: int = 10, threshold: int = 5) -> bool:
    """Checks if any n-gram appears >= threshold times."""
    if len(words) < n * threshold:
        return False
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= threshold:
            return True
    return False


def compute_wer_dp(ref: List[str], hyp: List[str]) -> float:
    """Computes Word Error Rate using space-optimized 2-row dynamic programming."""
    r_len, h_len = len(ref), len(hyp)
    if r_len == 0:
        return float(h_len > 0)
    prev = np.arange(h_len + 1, dtype=np.int32)
    curr = np.zeros(h_len + 1, dtype=np.int32)

    for i in range(1, r_len + 1):
        curr[0] = i
        r_w = ref[i - 1]
        for j in range(1, h_len + 1):
            if r_w == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev[:] = curr[:]

    return float(curr[h_len]) / float(r_len)


def load_woz_manual_participant_text(pid: int) -> str:
    """Loads official WoZ transcript for participant utterances only."""
    p_csv = os.path.join(WOZ_MANUAL_DIR, f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(p_csv):
        return ""
    try:
        df = pd.read_csv(p_csv, sep='\t')
        if len(df.columns) <= 1:
            df = pd.read_csv(p_csv, sep=',')
    except Exception:
        df = pd.read_csv(p_csv, sep=None, engine='python')

    df.columns = [c.strip().lower() for c in df.columns]
    spk_col = next((c for c in df.columns if 'speaker' in c), None)
    txt_col = next((c for c in df.columns if c in ['value', 'text', 'transcript', 'utterance']), None)
    if txt_col is None:
        txt_col = df.columns[2] if len(df.columns) >= 3 else df.columns[-1]

    if spk_col is not None:
        part_df = df[df[spk_col].astype(str).str.strip().str.lower() == 'participant']
    else:
        part_df = df

    return ' '.join(part_df[txt_col].dropna().astype(str).tolist())


def main():
    print("======================================================================")
    print("Step S3: Label-Blind Transcript QA Audit & Pre-Stated Stop Rules")
    print("======================================================================")

    manifest = pd.read_csv(MANIFEST_PATH)
    assert len(manifest) == 239, f"Expected 239 manifest rows, got {len(manifest)}"
    assert_no_sealed_ids(manifest['participant_id'].values)

    records = []
    woz_wers = []
    empty_sessions = []
    loop_sessions = []

    words_by_domain = {'woz': [], 'avatar': []}

    for idx, row in manifest.iterrows():
        pid = int(row['participant_id'])
        domain = row['domain']

        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': str(domain)})

        if domain == 'woz':
            trans_path = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        else:
            trans_path = os.path.join(AVATAR_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")

        if not os.path.exists(trans_path):
            raise FileNotFoundError(f"Transcript missing for PID {pid}: {trans_path}")

        df_t = pd.read_csv(trans_path)
        full_text = ' '.join(df_t['text'].dropna().astype(str).tolist())
        words = normalize_words(full_text)
        n_words = len(words)
        words_by_domain[domain].append(n_words)

        is_near_empty = (n_words < 20)
        is_loop = check_repetition_loops(words, n=10, threshold=5)

        if is_near_empty:
            empty_sessions.append(pid)
        if is_loop:
            loop_sessions.append(pid)

        # WER on WoZ sessions vs manual reference
        wer_val = None
        if domain == 'woz':
            ref_text = load_woz_manual_participant_text(pid)
            ref_words = normalize_words(ref_text)
            if ref_words:
                wer_val = compute_wer_dp(ref_words, words)
                woz_wers.append(wer_val)

        records.append({
            'participant_id': pid,
            'domain': domain,
            'word_count': n_words,
            'is_near_empty': is_near_empty,
            'is_loop': is_loop,
            'wer_vs_manual': round(wer_val, 4) if wer_val is not None else None
        })

    # Summary Statistics
    total_sessions = len(manifest)
    n_empty = len(empty_sessions)
    n_loops = len(loop_sessions)
    pct_empty = (n_empty / total_sessions) * 100.0
    pct_loops = (n_loops / total_sessions) * 100.0

    median_wer = float(np.median(woz_wers)) if woz_wers else None
    mean_wer = float(np.mean(woz_wers)) if woz_wers else None
    q25_wer = float(np.percentile(woz_wers, 25)) if woz_wers else None
    q75_wer = float(np.percentile(woz_wers, 75)) if woz_wers else None

    # Text Domain Separation on BERT-T2
    print("\nEvaluating Text Domain Separation on BERT-T2...")
    X_t2 = []
    y_domain = []
    for idx, row in manifest.iterrows():
        pid = int(row['participant_id'])
        npz_p = os.path.join(FEATURE_STORE_V2_DIR, f"{pid}_v2.npz")
        d = np.load(npz_p)
        assert 'bert_t2_768d' in d, f"Missing bert_t2_768d for PID {pid}"
        X_t2.append(d['bert_t2_768d'])
        y_domain.append(1 if row['domain'] == 'woz' else 0)

    X_t2 = np.array(X_t2, dtype=np.float32)
    y_domain = np.array(y_domain, dtype=int)

    # 5-fold CV to evaluate domain-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    domain_oof = np.zeros(len(y_domain), dtype=np.float32)
    for tr, va in cv.split(X_t2, y_domain):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_t2[tr], y_domain[tr])
        domain_oof[va] = clf.predict_proba(X_t2[va])[:, 1]
    domain_auc_t2 = float(roc_auc_score(y_domain, domain_oof))

    # Pre-Stated Stop Rules Verification
    stop_triggered = False
    stop_reasons = []

    if pct_empty > 5.0:
        stop_triggered = True
        stop_reasons.append(f"Near-empty sessions exceeded 5% ({pct_empty:.1f}%, {n_empty}/{total_sessions})")

    if pct_loops > 5.0:
        stop_triggered = True
        stop_reasons.append(f"Repetition loops exceeded 5% ({pct_loops:.1f}%, {n_loops}/{total_sessions})")

    if median_wer is not None and median_wer > 0.60:
        stop_triggered = True
        stop_reasons.append(f"Median WoZ WER exceeded 0.60 ({median_wer:.4f})")

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_sessions': total_sessions,
        'near_empty_sessions': {'count': n_empty, 'percentage': pct_empty, 'pids': empty_sessions},
        'repetition_loop_sessions': {'count': n_loops, 'percentage': pct_loops, 'pids': loop_sessions},
        'word_counts': {
            'woz': {
                'mean': float(np.mean(words_by_domain['woz'])),
                'median': float(np.median(words_by_domain['woz'])),
                'min': int(np.min(words_by_domain['woz'])),
                'max': int(np.max(words_by_domain['woz']))
            },
            'avatar': {
                'mean': float(np.mean(words_by_domain['avatar'])),
                'median': float(np.median(words_by_domain['avatar'])),
                'min': int(np.min(words_by_domain['avatar'])),
                'max': int(np.max(words_by_domain['avatar']))
            }
        },
        'woz_wer_vs_manual': {
            'count': len(woz_wers),
            'median': median_wer,
            'mean': mean_wer,
            'q25': q25_wer,
            'q75': q75_wer
        },
        'text_domain_auc_t2': domain_auc_t2,
        'stop_rules': {
            'stop_triggered': stop_triggered,
            'reasons': stop_reasons,
            'thresholds': {
                'max_empty_pct': 5.0,
                'max_loop_pct': 5.0,
                'max_median_wer': 0.60
            }
        }
    }

    with open(QA_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n======================================================================")
    print("TRANSCRIPT QA RESULTS SUMMARY")
    print("======================================================================")
    print(f"Total Sessions: {total_sessions} (WoZ: {len(words_by_domain['woz'])}, Avatar: {len(words_by_domain['avatar'])})")
    print(f"Near-Empty Transcripts (<20 words): {n_empty} ({pct_empty:.2f}%) [Threshold <= 5.0%]")
    print(f"Repetition Loops (10-gram >= 5x): {n_loops} ({pct_loops:.2f}%) [Threshold <= 5.0%]")
    if median_wer is not None:
        print(f"WoZ Whisper WER vs Manual Reference: Median = {median_wer:.4f} (IQR: [{q25_wer:.4f}, {q75_wer:.4f}]) [Threshold <= 0.60]")
    print(f"Text Domain Separation AUC (BERT-T2): {domain_auc_t2:.4f} (vs T1: 0.9994)")
    print(f"Word Counts (WoZ): Mean={report['word_counts']['woz']['mean']:.1f}, Median={report['word_counts']['woz']['median']:.1f}")
    print(f"Word Counts (Avatar): Mean={report['word_counts']['avatar']['mean']:.1f}, Median={report['word_counts']['avatar']['median']:.1f}")
    print(f"--> STOP GATE VERDICT: {'STOP TRIGGERED' if stop_triggered else 'PASSED (CONTINUE AUTHORIZED)'}")
    if stop_reasons:
        print(f"    Reasons: {stop_reasons}")
    print("======================================================================")

if __name__ == '__main__':
    main()
