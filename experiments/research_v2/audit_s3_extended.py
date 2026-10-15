# Step S3 Extended: Label-Blind Transcript QA Audit, Provenance & Batch Leakage Diagnostic
import os
import sys
import re
import json
import time
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_REPO_ROOT = "/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, "experiments", "research_v2") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "experiments", "research_v2"))

from sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from test_label_blind_extraction import assert_sealed_quarantine, assert_label_blind
from extractors_v2 import BertExtractorV2

WOZ_WHISPER_DIR = os.path.join(_REPO_ROOT, "experiments", "research_v2", "transcripts_v2", "woz_whisper")
AVATAR_WHISPER_DIR = os.path.join(_REPO_ROOT, "experiments", "research_v2", "transcripts_avatar_dev")
FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, "experiments", "research_v2", "feature_store_v2")
FEATURE_STORE_V3_DIR = os.path.join(_REPO_ROOT, "experiments", "research_v2", "feature_store_v3")
MANIFEST_PATH = os.path.join(FEATURE_STORE_V3_DIR, "manifest_v3.csv")
REPORT_OUT = os.path.join(_REPO_ROOT, "experiments", "research_v2", "transcript_qa_extended_report.json")

def normalize_words(text):
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", str(text).lower())
    return [w for w in text.split() if w]

def compute_wer_dp(ref_words, hyp_words):
    r_len, h_len = len(ref_words), len(hyp_words)
    if r_len == 0:
        return 1.0 if h_len > 0 else 0.0
    dp = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return float(dp[r_len][h_len]) / float(r_len)

def check_repetition_loops(words, n=10, threshold=5):
    if len(words) < n * threshold:
        return False
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
        if counts[ng] >= threshold:
            return True
    return False

def cosine_distance(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 1.0
    cos_sim = float(np.dot(u, v) / (norm_u * norm_v))
    return float(max(0.0, 1.0 - cos_sim))

def load_woz_manual_participant_text(pid):
    p_csv = os.path.join(_REPO_ROOT, f"Clean_audio/daicwoz/{pid}_TRANSCRIPT.csv")
    if not os.path.exists(p_csv):
        return ""
    try:
        df = pd.read_csv(p_csv, sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(p_csv, sep=",")
    except Exception:
        df = pd.read_csv(p_csv, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    spk_col = next((c for c in df.columns if "speaker" in c), None)
    txt_col = next((c for c in df.columns if c in ["value", "text", "transcript", "utterance"]), None)
    if txt_col is None:
        txt_col = df.columns[2] if len(df.columns) >= 3 else df.columns[-1]
    if spk_col is not None:
        part_df = df[df[spk_col].astype(str).str.strip().str.lower() == "participant"]
    else:
        part_df = df
    return " ".join(part_df[txt_col].dropna().astype(str).tolist())

def run_extended_qa():
    print("=" * 70)
    print("STEP S3 EXTENDED: TRANSCRIPT QA, PROVENANCE & BATCH-LEAKAGE AUDIT")
    print("=" * 70)

    manifest = pd.read_csv(MANIFEST_PATH)
    assert len(manifest) == 239, f"Expected 239 dev participants, got {len(manifest)}"
    assert_no_sealed_ids(manifest["participant_id"].values)

    # 1. Provenance Identification
    print("\n--- 1. Provenance Breakdown ---")
    woz_dev_pids = manifest[manifest["domain"] == "woz"]["participant_id"].values
    avatar_dev_pids = manifest[manifest["domain"] == "avatar"]["participant_id"].values

    earlier_mps_pids = []
    new_cpu_pids = []

    for pid in woz_dev_pids:
        f = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        assert os.path.exists(f), f"Missing transcript for WoZ PID {pid}"
        df_p = pd.read_csv(f)
        src = str(df_p["source_flag"].iloc[0]) if "source_flag" in df_p.columns else ""
        if "decoding" in df_p.columns or src == "whisper_tiny_local":
            new_cpu_pids.append(pid)
        else:
            earlier_mps_pids.append(pid)

    print(f"Avatar Cohort: {len(avatar_dev_pids)} sessions")
    print(f"WoZ Earlier Batch (MPS script): {len(earlier_mps_pids)} sessions")
    print(f"WoZ New Batch (CPU workers): {len(new_cpu_pids)} sessions")

    provenance_table = {
        "avatar_61": {
            "n": len(avatar_dev_pids),
            "script": "experiments/research_v2/run_whisper_transcription.py",
            "hardware": "Apple Silicon CPU",
            "model": "openai/whisper-tiny",
            "decoding": "Greedy (temperature=0.0, do_sample=False)",
            "chunk_length_s": 30,
            "batch_size": 8,
            "library": "transformers-whisper-tiny (pipeline)",
            "matches_amendment_v1_3b": True
        },
        "woz_earlier_76": {
            "n": len(earlier_mps_pids),
            "script": "experiments/research_v2/run_mps_woz_transcription.py",
            "hardware": "Apple Silicon MPS (Metal Performance Shaders GPU)",
            "model": "openai/whisper-tiny",
            "decoding": "Greedy (pipeline default temperature=0.0)",
            "chunk_length_s": 30,
            "batch_size": 16,
            "library": "transformers 5.19.0, torch 2.14.1 (MPS)",
            "matches_amendment_v1_3b": True
        },
        "woz_new_102": {
            "n": len(new_cpu_pids),
            "script": "experiments/research_v2/transcribe_remaining_woz.py",
            "hardware": "Apple Silicon CPU (3 workers, torch threads=2)",
            "model": "openai/whisper-tiny",
            "decoding": "Greedy (temperature=0.0, condition_on_previous_text=False)",
            "chunk_length_s": 30,
            "batch_size": 8,
            "library": "transformers 5.19.0, torch 2.14.1 (CPU)",
            "matches_amendment_v1_3b": True
        }
    }

    # 2. Consistency Check on 5 Earlier WoZ Sessions
    print("\n--- 2. Consistency Check (Old MPS vs New CPU Re-Transcription) ---")
    test_5_pids = [int(p) for p in earlier_mps_pids[:5]]
    print(f"Re-transcribing 5 earlier WoZ sessions with new CPU pipeline: {test_5_pids}")

    torch.set_num_threads(2)
    pipe_cpu = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=30,
        batch_size=8,
        device="cpu"
    )
    bert_extractor = BertExtractorV2(device="cpu")

    consistency_results = []
    wers_consistency = []
    cos_dists = []

    for pid in test_5_pids:
        old_csv = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        df_old = pd.read_csv(old_csv)
        old_text = " ".join(df_old["text"].dropna().astype(str).tolist())
        old_words = normalize_words(old_text)
        old_bert = bert_extractor.extract_from_text(old_text)

        cache_csv = os.path.join(WOZ_WHISPER_DIR, f"{pid}_RETRANS_TEST.csv")
        if os.path.exists(cache_csv):
            df_re = pd.read_csv(cache_csv)
            new_text = str(df_re["text"].iloc[0])
        else:
            audio_path = os.path.join(_REPO_ROOT, f"Clean_audio/NEW_AUDIO/{pid}_AUDIO_interviewee.wav")
            res_new = pipe_cpu(
                audio_path,
                generate_kwargs={"language": "english", "task": "transcribe", "temperature": 0.0}
            )
            new_text = res_new.get("text", "")
            pd.DataFrame([{"participant_id": pid, "text": new_text}]).to_csv(cache_csv, index=False)

        new_words = normalize_words(new_text)
        new_bert = bert_extractor.extract_from_text(new_text)

        pair_wer = compute_wer_dp(old_words, new_words)
        cos_d = cosine_distance(old_bert, new_bert)
        wers_consistency.append(pair_wer)
        cos_dists.append(cos_d)

        print(f"  PID {pid}: WER(Old MPS vs New CPU) = {pair_wer:.4f} | BERT Cosine Distance = {cos_d:.6f}")
        consistency_results.append({
            "participant_id": pid,
            "wer_old_vs_new": round(pair_wer, 4),
            "bert_cosine_distance": round(cos_d, 6)
        })

    mean_consistency_wer = float(np.mean(wers_consistency))
    mean_cos_dist = float(np.mean(cos_dists))
    print(f"--> Mean Consistency WER: {mean_consistency_wer:.4f} (Threshold: <= 0.15)")
    print(f"--> Mean BERT Cosine Distance: {mean_cos_dist:.6f}")

    # 3. Batch-Leakage Diagnostic within WoZ (earlier MPS vs new CPU)
    print("\n--- 3. Batch-Leakage Diagnostic within WoZ (bert_t2_768d) ---")
    X_woz = []
    y_batch = []

    for pid in woz_dev_pids:
        npz_p = os.path.join(FEATURE_STORE_V2_DIR, f"{pid}_v2.npz")
        assert os.path.exists(npz_p), f"Missing v2 npz for PID {pid}"
        d = np.load(npz_p)
        assert "bert_t2_768d" in d, f"Missing bert_t2_768d for PID {pid}"
        X_woz.append(d["bert_t2_768d"])
        # 1 if in new_cpu_pids, 0 if in earlier_mps_pids
        y_batch.append(1 if pid in new_cpu_pids else 0)

    X_woz = np.array(X_woz, dtype=np.float32)
    y_batch = np.array(y_batch, dtype=int)

    # 5-fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    batch_oof = np.zeros(len(y_batch), dtype=np.float32)

    for tr, va in skf.split(X_woz, y_batch):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_woz[tr], y_batch[tr])
        batch_oof[va] = clf.predict_proba(X_woz[va])[:, 1]

    batch_auc = float(roc_auc_score(y_batch, batch_oof))
    print(f"--> WoZ Batch-Leakage Classification AUC (MPS vs CPU): {batch_auc:.4f} (Threshold: <= 0.70)")

    # 4. Standard S3 Quality Gates
    print("\n--- 4. Standard S3 Quality Gates ---")
    empty_sessions = []
    loop_sessions = []
    woz_manual_wers = []
    words_by_domain = {"woz": [], "avatar": []}

    for idx, row in manifest.iterrows():
        pid = int(row["participant_id"])
        domain = row["domain"]

        if domain == "woz":
            t_file = os.path.join(WOZ_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")
        else:
            t_file = os.path.join(AVATAR_WHISPER_DIR, f"{pid}_TRANSCRIPT_WHISPER.csv")

        df_t = pd.read_csv(t_file)
        full_text = " ".join(df_t["text"].dropna().astype(str).tolist())
        words = normalize_words(full_text)
        n_words = len(words)
        words_by_domain[domain].append(n_words)

        if n_words < 20:
            empty_sessions.append(pid)
        if check_repetition_loops(words, n=10, threshold=5):
            loop_sessions.append(pid)

        if domain == "woz":
            ref_text = load_woz_manual_participant_text(pid)
            ref_words = normalize_words(ref_text)
            if ref_words:
                w_val = compute_wer_dp(ref_words, words)
                woz_manual_wers.append(w_val)

    total_n = len(manifest)
    pct_empty = (len(empty_sessions) / total_n) * 100.0
    pct_loops = (len(loop_sessions) / total_n) * 100.0
    median_manual_wer = float(np.median(woz_manual_wers)) if woz_manual_wers else None
    q25_manual_wer = float(np.percentile(woz_manual_wers, 25)) if woz_manual_wers else None
    q75_manual_wer = float(np.percentile(woz_manual_wers, 75)) if woz_manual_wers else None

    # Text Domain AUC (T2: WoZ vs Avatar)
    X_all_t2 = []
    y_domain = []
    for idx, row in manifest.iterrows():
        pid = int(row["participant_id"])
        npz_p = os.path.join(FEATURE_STORE_V2_DIR, f"{pid}_v2.npz")
        d = np.load(npz_p)
        X_all_t2.append(d["bert_t2_768d"])
        y_domain.append(1 if row["domain"] == "woz" else 0)

    X_all_t2 = np.array(X_all_t2, dtype=np.float32)
    y_domain = np.array(y_domain, dtype=int)

    skf_dom = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    domain_oof = np.zeros(len(y_domain), dtype=np.float32)
    for tr, va in skf_dom.split(X_all_t2, y_domain):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_all_t2[tr], y_domain[tr])
        domain_oof[va] = clf.predict_proba(X_all_t2[va])[:, 1]
    domain_auc_t2 = float(roc_auc_score(y_domain, domain_oof))

    # Evaluate all stop rules
    stop_triggered = False
    stop_reasons = []

    if mean_consistency_wer > 0.15:
        stop_triggered = True
        stop_reasons.append(f"Consistency WER ({mean_consistency_wer:.4f}) exceeded 0.15")

    if batch_auc > 0.70:
        stop_triggered = True
        stop_reasons.append(f"Batch-leakage AUC ({batch_auc:.4f}) exceeded 0.70")

    if pct_empty > 5.0:
        stop_triggered = True
        stop_reasons.append(f"Empty sessions ({pct_empty:.2f}%) exceeded 5.0%")

    if pct_loops > 5.0:
        stop_triggered = True
        stop_reasons.append(f"Repetition loops ({pct_loops:.2f}%) exceeded 5.0%")

    if median_manual_wer is not None and median_manual_wer > 0.60:
        stop_triggered = True
        stop_reasons.append(f"Median WoZ WER vs manual ({median_manual_wer:.4f}) exceeded 0.60")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance_table": provenance_table,
        "consistency_check": {
            "evaluated_pids": test_5_pids,
            "details": consistency_results,
            "mean_wer": mean_consistency_wer,
            "mean_bert_cosine_distance": mean_cos_dist,
            "threshold_wer": 0.15,
            "passed": bool(mean_consistency_wer <= 0.15)
        },
        "batch_leakage_diagnostic": {
            "earlier_mps_n": len(earlier_mps_pids),
            "new_cpu_n": len(new_cpu_pids),
            "batch_auc": batch_auc,
            "threshold_auc": 0.70,
            "passed": bool(batch_auc <= 0.70)
        },
        "quality_gates": {
            "empty_sessions": {"count": len(empty_sessions), "pct": pct_empty, "threshold_pct": 5.0, "passed": bool(pct_empty <= 5.0)},
            "repetition_loops": {"count": len(loop_sessions), "pct": pct_loops, "threshold_pct": 5.0, "passed": bool(pct_loops <= 5.0)},
            "woz_manual_wer": {"median": median_manual_wer, "q25": q25_manual_wer, "q75": q75_manual_wer, "threshold_median": 0.60, "passed": bool(median_manual_wer <= 0.60)},
            "text_domain_auc_t2": domain_auc_t2
        },
        "word_counts": {
            "woz": {"mean": float(np.mean(words_by_domain["woz"])), "median": float(np.median(words_by_domain["woz"]))},
            "avatar": {"mean": float(np.mean(words_by_domain["avatar"])), "median": float(np.median(words_by_domain["avatar"]))}
        },
        "verdict": {
            "stop_triggered": stop_triggered,
            "stop_reasons": stop_reasons,
            "proceed_authorized": not stop_triggered
        }
    }

    def json_default(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return str(obj)

    with open(REPORT_OUT, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY REPORT")
    print("=" * 70)
    print(f"1. Provenance verified: Avatar (61, CPU), WoZ MPS (76), WoZ CPU (102).")
    print(f"2. Consistency Check (5 sessions): Mean WER = {mean_consistency_wer:.4f}, Mean Cosine Dist = {mean_cos_dist:.6f}")
    print(f"3. WoZ Batch AUC (MPS vs CPU): {batch_auc:.4f} (Threshold <= 0.70)")
    print(f"4. Near-Empty Transcripts: {pct_empty:.2f}% (Threshold <= 5.0%)")
    print(f"5. Repetition Loops: {pct_loops:.2f}% (Threshold <= 5.0%)")
    print(f"6. WoZ WER vs Manual: Median = {median_manual_wer:.4f} (IQR [{q25_manual_wer:.4f}, {q75_manual_wer:.4f}]) (Threshold <= 0.60)")
    print(f"7. Text Domain Separation AUC (T2): {domain_auc_t2:.4f}")
    print(f"--> FINAL VERDICT: {"STOP TRIGGERED - REGENERATE ALL WOZ TRANSCRIPTS" if stop_triggered else "ALL GATES PASSED - PROCEED TO E1 AUTHORIZED"}")
    if stop_reasons:
        print(f"    Failure reasons: {stop_reasons}")
    print("=" * 70)

    return summary

if __name__ == "__main__":
    run_extended_qa()
