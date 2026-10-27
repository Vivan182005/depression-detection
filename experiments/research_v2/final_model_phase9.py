"""
Module: experiments/research_v2/final_model_phase9.py
Prediction function for the Phase 9 final model F1 (models/final_phase9_f1.joblib): three voice ridge members, one
emotion-RoBERTa text ridge member and one ridge on the 8 zero-shot LLM PHQ-8 item scores, CDF-calibrated and stacked
by logistic regression. The members are Phase 7 ConfigModel objects (reg, no per-domain scaling).
"""
import numpy as np
from final_model_phase7 import ConfigModel  # noqa: F401  (needed to unpickle the bundle)


def predict_bundle_f1(bundle, feats, domains):
    """feats: dict base_name -> (n, d) array, including 'llm_items' (n, 8) in score_phq_llm.ITEMS order.
    Returns (probability, flag_max_accuracy, flag_youden)."""
    v = np.mean([m.score(feats[m.base], domains) for m in bundle['voice_members']], axis=0)
    t = bundle['text_member'].score(feats[bundle['text_member'].base], domains)
    l = bundle['llm_member'].score(feats['llm_items'], domains)
    p = bundle['meta'].predict_proba(np.column_stack([v, t, l]))[:, 1]
    return p, (p >= bundle['thresholds']['max_acc']).astype(int), (p >= bundle['thresholds']['youden']).astype(int)
