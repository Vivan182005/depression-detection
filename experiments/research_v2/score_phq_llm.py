"""
Script: experiments/research_v2/score_phq_llm.py
Phase 9 (prereg_phase9_llm_phq.md): zero-shot PHQ-8 item scores from a local 4-bit Mistral-7B (MLX) on the v1.3c
participant-only Whisper transcripts. For each item the score is E[s] = sum k p(k) over k in 0..3 from the next-token
distribution, decoding greedily. Label-free. Output: feature_store_9/llm_phq.csv (git-ignored), resumable.
  .venv_llm/bin/python score_phq_llm.py [--pids 300,301] [--limit N]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

RD = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(RD))
OUT_DIR = os.path.join(RD, 'feature_store_9')
OUT = os.path.join(OUT_DIR, 'llm_phq.csv')
MODEL = 'mlx-community/Mistral-7B-Instruct-v0.3-4bit'
ITEMS = ['interest', 'depressed', 'sleep', 'tired', 'appetite', 'failure', 'concentration', 'psychomotor']

PROMPT = """You are an experienced clinical psychologist. Below are only the participant's answers (the interviewer's questions are not included) from a recorded semi-structured interview about their life, mood, sleep and wellbeing. The text is an automatic speech-recognition transcript and may contain errors.

Estimate how the participant would answer each PHQ-8 item about the last two weeks, on the standard scale 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day. Base each rating on what they say and how they say it; if there is no evidence for an item, choose the most likely value.

PHQ-8 items:
interest: little interest or pleasure in doing things
depressed: feeling down, depressed, or hopeless
sleep: trouble falling or staying asleep, or sleeping too much
tired: feeling tired or having little energy
appetite: poor appetite or overeating
failure: feeling bad about yourself, or that you are a failure or have let yourself or your family down
concentration: trouble concentrating on things
psychomotor: moving or speaking noticeably slowly, or being fidgety or restless

Transcript:
\"\"\"{text}\"\"\"

Answer with only a JSON object with these 8 keys and integer values 0-3, in this order: {keys}."""


def load_transcripts():
    SEALED_STORE = os.path.join(RD, 'feature_store_sealed')     # same as run_sealed_evaluation.STORE (avoids torch import)
    T = {}
    vdir = os.path.join(RD, 'transcripts_v2', 'vad_chunked')
    for f in os.listdir(vdir):
        T[int(f.split('_')[0])] = str(pd.read_csv(os.path.join(vdir, f)).iloc[0]['text'])
    for p in json.load(open(os.path.join(ROOT, 'sealed_ids.json')))['sealed_pids']:
        T[int(p)] = str(np.load(os.path.join(SEALED_STORE, f'{p}_sealed.npz'))['transcript_clean'])
    return T


def score_one(model, tok, text, digit_ids, max_tokens=120):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    msgs = [{'role': 'user', 'content': PROMPT.format(text=text.strip(), keys=', '.join(ITEMS))}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    prompt = mx.array(prompt if isinstance(prompt, list) else tok.encode(prompt))
    out, scores, gen = '', {}, []
    for (t, logprobs), _ in zip(generate_step(prompt, model, sampler=lambda lp: mx.argmax(lp, axis=-1)), range(max_tokens)):
        t = int(t)
        if t == tok.eos_token_id:
            break
        gen.append(t)
        new = tok.decode(gen)
        piece = new[len(out):]
        if t in digit_ids:
            key = next((k for k in ITEMS if k not in scores and out.rstrip().endswith(f'"{k}":')), None)
            if key:
                lp = np.array(logprobs[digit_ids].astype(mx.float32))
                p = np.exp(lp - lp.max()); p /= p.sum()
                scores[key] = float(p @ np.arange(4))
        out = new
        if len(scores) == len(ITEMS) and '}' in piece:
            break
    return scores, out, int(prompt.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pids', default='')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    from mlx_lm import load
    model, tok = load(MODEL)
    digit_ids = []
    for d in '0123':
        ids = tok.encode(d, add_special_tokens=False)
        digit_ids.append(ids[-1])
    assert len(set(digit_ids)) == 4, digit_ids
    T = load_transcripts()
    pids = sorted(T) if not a.pids else [int(p) for p in a.pids.split(',')]
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set(pd.read_csv(OUT)['participant_id']) if os.path.exists(OUT) else set()
    todo = [p for p in pids if p not in done][: a.limit or None]
    print(f'{len(T)} transcripts, {len(done)} already scored, {len(todo)} to do; digit ids {digit_ids}', flush=True)
    for i, p in enumerate(todo):
        t0 = time.time()
        s, raw, n_tok = score_one(model, tok, T[p], digit_ids)
        row = {'participant_id': p, **{k: s.get(k, np.nan) for k in ITEMS}, 'n_items_parsed': len(s), 'prompt_tokens': n_tok,
               'raw': raw.replace('\n', ' ')}
        pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        print(f'[{i + 1}/{len(todo)}] PID {p} tokens={n_tok} items={len(s)} sum={np.nansum(list(s.values())):.2f} '
              f'in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
