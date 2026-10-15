"""
Script: experiments/research_v2/audit_s3_vad_and_update_t2.py
Amendment v1.3c: label-blind S3 re-audit of VAD-chunked Whisper-tiny transcripts,
then (only if every pre-stated v1.3b gate passes) rebuild bert_t2_768d for all 239
active development sessions.

Gates (unchanged from Amendment v1.3b sec 4):
  - near-empty (< 20 words) sessions <= 5%
  - repetition loops (any 10-gram repeated >= 5 times) <= 5%
  - median WoZ WER vs manual participant transcript <= 0.60
Text-domain AUC on the rebuilt bert_t2_768d is reported descriptively.

The previous full-length transcripts' embeddings are archived under
bert_t2_fulllen_768d (not used by any variant).
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids
from test_label_blind_extraction import assert_sealed_quarantine, assert_label_blind
from audit_s3_extended import (normalize_words, compute_wer_dp, check_repetition_loops,
                               load_woz_manual_participant_text)

VAD_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcripts_v2', 'vad_chunked')
FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
FEATURE_STORE_V3_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v3')
MANIFEST_V3_PATH = os.path.join(FEATURE_STORE_V3_DIR, 'manifest_v3.csv')
REPORT_OUT = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'transcript_qa_v1_3c_report.json')


def domain_auc(X, y_domain):
    oof = np.zeros(len(y_domain))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_domain):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X[tr], y_domain[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return float(roc_auc_score(y_domain, oof))


def main():
    manifest = pd.read_csv(MANIFEST_V3_PATH)
    assert len(manifest) == 239
    assert_no_sealed_ids(manifest['participant_id'].values)

    rows, texts = [], {}
    for r in manifest.itertuples():
        pid, domain = int(r.participant_id), str(r.domain)
        assert_sealed_quarantine(pid)
        assert_label_blind({'participant_id': pid, 'domain': domain})
        f = os.path.join(VAD_DIR, f'{pid}_TRANSCRIPT_WHISPER_VAD.csv')
        assert os.path.exists(f), f'Missing VAD transcript for PID {pid}'
        t = pd.read_csv(f).iloc[0]
        text = '' if pd.isna(t['text']) else str(t['text'])
        raw = '' if pd.isna(t['text_raw']) else str(t['text_raw'])
        texts[pid] = text
        words = normalize_words(text)
        row = {
            'participant_id': pid, 'domain': domain, 'n_words': len(words),
            'near_empty': len(words) < 20,
            'loop_clean': check_repetition_loops(words, n=10, threshold=5),
            'loop_raw': check_repetition_loops(normalize_words(raw), n=10, threshold=5),
            'n_chunks_redecoded': int(t['n_chunks_redecoded']),
            'words_removed_by_filter': int(t['n_words_raw']) - int(t['n_words_clean']),
        }
        if domain == 'woz':
            ref = normalize_words(load_woz_manual_participant_text(pid))
            row['wer_vs_manual'] = compute_wer_dp(ref, words) if ref else np.nan
        rows.append(row)

    qa = pd.DataFrame(rows)
    n = len(qa)
    pct_empty = 100.0 * qa['near_empty'].sum() / n
    pct_loop = 100.0 * qa['loop_clean'].sum() / n
    pct_loop_raw = 100.0 * qa['loop_raw'].sum() / n
    wer = qa.loc[qa['domain'] == 'woz', 'wer_vs_manual'].dropna()
    med_wer = float(wer.median())

    reasons = []
    if pct_empty > 5.0:
        reasons.append(f'near-empty {pct_empty:.2f}% > 5%')
    if pct_loop > 5.0:
        reasons.append(f'repetition loops {pct_loop:.2f}% > 5%')
    if med_wer > 0.60:
        reasons.append(f'median WoZ WER {med_wer:.3f} > 0.60')

    print(f'Near-empty: {pct_empty:.2f}% | loops after filter: {pct_loop:.2f}% '
          f'(before filter: {pct_loop_raw:.2f}%) | median WoZ WER: {med_wer:.4f} '
          f'(IQR {wer.quantile(.25):.4f}-{wer.quantile(.75):.4f})')
    for dom in ['woz', 'avatar']:
        sub = qa[qa['domain'] == dom]
        print(f'  {dom}: n={len(sub)} loops_raw={int(sub.loop_raw.sum())} loops_clean={int(sub.loop_clean.sum())} '
              f'median words={sub.n_words.median():.0f} redecoded chunks={int(sub.n_chunks_redecoded.sum())} '
              f'words removed by filter={int(sub.words_removed_by_filter.sum())}')

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'amendment': 'v1.3c',
        'transcription': 'whisper-tiny, Silero-VAD chunks <= 28 s, per-segment re-decode on loop, '
                         'n-gram repeat collapse (>3 consecutive -> 3)',
        'gates': {
            'near_empty_pct': pct_empty,
            'repetition_loop_pct_after_filter': pct_loop,
            'repetition_loop_pct_before_filter': pct_loop_raw,
            'woz_wer_median': med_wer,
            'woz_wer_iqr': [float(wer.quantile(.25)), float(wer.quantile(.75))],
        },
        'stop_reasons': reasons,
        'passed': not reasons,
    }
    qa.to_csv(REPORT_OUT.replace('_report.json', '_per_session.csv'), index=False)

    if reasons:
        report['bert_t2_rebuilt'] = False
        json.dump(report, open(REPORT_OUT, 'w'), indent=2)
        print('STOP: gate failed ->', reasons)
        sys.exit(3)

    from extractors_v2 import BertExtractorV2
    bert = BertExtractorV2(device='cpu')
    X_new, X_old, y_dom = [], [], []
    for r in manifest.itertuples():
        pid = int(r.participant_id)
        emb = bert.extract_from_text(texts[pid])
        assert emb.shape == (768,) and np.isfinite(emb).all()
        p2 = os.path.join(FEATURE_STORE_V2_DIR, f'{pid}_v2.npz')
        d2 = dict(np.load(p2))
        if 'bert_t2_fulllen_768d' not in d2:
            d2['bert_t2_fulllen_768d'] = d2['bert_t2_768d']
        X_old.append(d2['bert_t2_fulllen_768d'])
        d2['bert_t2_768d'] = emb.astype(np.float32)
        np.savez_compressed(p2 + '_tmp.npz', **d2)
        os.replace(p2 + '_tmp.npz', p2)
        p3 = os.path.join(FEATURE_STORE_V3_DIR, f'{pid}_features.npz')
        if os.path.exists(p3):
            d3 = dict(np.load(p3, allow_pickle=True))
            d3['bert_t2_768d'] = emb.astype(np.float32)
            np.savez_compressed(p3 + '_tmp.npz', **d3)
            os.replace(p3 + '_tmp.npz', p3)
        X_new.append(emb)
        y_dom.append(1 if r.domain == 'woz' else 0)

    y_dom = np.array(y_dom)
    report['text_domain_auc_t2_vad'] = domain_auc(np.array(X_new, dtype=np.float32), y_dom)
    report['text_domain_auc_t2_fulllen'] = domain_auc(np.array(X_old, dtype=np.float32), y_dom)
    report['bert_t2_rebuilt'] = True
    manifest['text_source_t2'] = 'whisper_tiny_local_vad_chunked_v1_3c'
    manifest.to_csv(MANIFEST_V3_PATH, index=False)
    json.dump(report, open(REPORT_OUT, 'w'), indent=2)
    print(f"Rebuilt bert_t2_768d for 239 sessions. Text-domain AUC: VAD={report['text_domain_auc_t2_vad']:.4f} "
          f"vs full-length={report['text_domain_auc_t2_fulllen']:.4f}")
    print('ALL S3 GATES PASSED -> E1 authorized')


if __name__ == '__main__':
    main()
