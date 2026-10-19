"""
Module: experiments/research_v2/final_model_phase7.py
Model classes for the Phase 7 final model (models/final_phase7_p4.joblib) and its prediction function.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


class ConfigModel:
    """One (feature base, model kind, per-domain scaling) config fitted on the full pool."""

    def __init__(self, base, kind, per_domain, hp, X, y, phq, dom, oof):
        self.base, self.kind, self.per_domain, self.hp = base, kind, per_domain, hp
        self.stats = {}
        groups = sorted(set(dom)) if per_domain else [None]
        Xs = X.astype(np.float64).copy()
        for g in groups:
            m = (dom == g) if g is not None else np.ones(len(dom), bool)
            mu, sd = Xs[m].mean(0), Xs[m].std(0) + 1e-8
            self.stats[g] = (mu, sd)
            Xs[m] = (Xs[m] - mu) / sd
        self.model = (LogisticRegression(C=hp, max_iter=3000).fit(Xs, y) if kind == 'cls'
                      else Ridge(alpha=hp).fit(Xs, phq))
        self.cdf_ref = np.sort(oof)   # full-pool OOF scores define the [0, 1] calibration

    def score(self, X, dom):
        Xs = X.astype(np.float64).copy()
        for i, d in enumerate(dom):
            mu, sd = self.stats[d if self.per_domain else None]
            Xs[i] = (Xs[i] - mu) / sd
        raw = self.model.predict_proba(Xs)[:, 1] if self.kind == 'cls' else self.model.predict(Xs)
        return np.searchsorted(self.cdf_ref, raw, side='right') / len(self.cdf_ref)



def predict_bundle(bundle, feats, domains):
    """feats: dict base_name -> (n, d) array; domains: list of 'woz'/'avatar'.
    Returns (probability, flag_max_accuracy, flag_youden)."""
    v = np.mean([m.score(feats[m.base], domains) for m in bundle['voice_members']], axis=0)
    t = bundle['text_member'].score(feats[bundle['text_member'].base], domains)
    p = bundle['meta'].predict_proba(np.column_stack([v, t]))[:, 1]
    return p, (p >= bundle['thresholds']['max_acc']).astype(int), (p >= bundle['thresholds']['youden']).astype(int)
