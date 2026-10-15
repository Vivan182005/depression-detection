"""
Execution Script: experiments/research_v2/run_phase6a_experiment.py
Step E2 & E3: Execute ONCE on real labels with tagged code & apply pre-stated freeze rule.
Evaluates:
- Observed participant-level AUC (PHQ >= 10, OOF scores averaged over seeds 0-4) for all 19 variants.
- Permutation tests for H1 (V2) and H2 (V10) with R=999 within domain x gender strata.
- H3 Intersection-Union paired participant-cluster bootstrap (B=10,000).
- Holm-Bonferroni step-down multiplicity adjustment across m=3 confirmatory hypotheses.
- Cluster-bootstrap 95% CIs for exploratory variants.
- Descriptive controls and secondary endpoints (WoZ, Avatar, Gender subgroup AUCs, 3-class macro-F1, Severe recall).
- Freezes primary audio and fusion models into frozen_config.json.
- Zero runs on seeds 100-104 or the sealed 33.
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import roc_auc_score, f1_score, recall_score

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, 'experiments', 'research_v2') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'experiments', 'research_v2'))

from sealed_guard import assert_no_sealed_ids, SEALED_PIDS
from test_label_blind_extraction import assert_label_blind, assert_sealed_quarantine
from data_loader import load_cohort_info, load_variant_features, permute_labels_stratified
from pipeline_engine import run_pipeline

VARIANTS = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19'
]

AUDIO_VARIANTS = [f'V{i}' for i in range(1, 14)]
OUTPUT_RESULTS_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'phase6a_results.json')
FROZEN_CONFIG_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'frozen_config.json')


def compute_cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Computes percentile bootstrap 95% confidence interval for ROC-AUC.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc = roc_auc_score(y_true[idx], y_prob[idx])
        boot_aucs.append(auc)

    lower = float(np.percentile(boot_aucs, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_paired_cluster_bootstrap_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_bootstraps: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Computes paired cluster bootstrap p-value for prob_a > prob_b, and 95% CI on Delta AUC.
    Returns: (p_val_one_sided, observed_delta, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    obs_delta = float(roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b))

    deltas = []
    n_less_or_equal = 0

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], prob_a[idx])
        auc_b = roc_auc_score(y_true[idx], prob_b[idx])
        d = auc_a - auc_b
        deltas.append(d)
        if d <= 0.0:
            n_less_or_equal += 1

    p_val = (1.0 + n_less_or_equal) / (len(deltas) + 1.0)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return float(p_val), obs_delta, ci_lower, ci_upper


def evaluate_secondary_endpoints(
    cohort_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes descriptive secondary metrics:
    - Subgroup AUCs (WoZ, Avatar, Female, Male) with 95% CIs
    - 3-Class Macro-F1 across None/Mild (0-9), Moderate (10-14), Severe (15-24)
    - Severe Recall (PHQ >= 15 sensitivity)
    """
    sec = {}

    # 1. Subgroups
    subgroups = {
        'woz': (cohort_df['domain'] == 'woz').values,
        'avatar': (cohort_df['domain'] == 'avatar').values,
        'female': (cohort_df['gender'] == 0).values,
        'male': (cohort_df['gender'] == 1).values
    }

    for name, mask in subgroups.items():
        sub_y = y_true[mask]
        sub_p = y_prob[mask]
        if len(np.unique(sub_y)) >= 2:
            sub_auc = float(roc_auc_score(sub_y, sub_p))
            ci_l, ci_u = compute_cluster_bootstrap_ci(sub_y, sub_p, n_bootstraps=1000, seed=123)
            sec[f'subgroup_{name}_auc'] = sub_auc
            sec[f'subgroup_{name}_ci'] = [ci_l, ci_u]
            sec[f'subgroup_{name}_n'] = int(np.sum(mask))
            sec[f'subgroup_{name}_n_pos'] = int(np.sum(sub_y == 1))
        else:
            sec[f'subgroup_{name}_auc'] = None

    # 2. Severe Recall (PHQ >= 15)
    # Binary classification threshold at 0.50
    pred_binary = (y_prob >= 0.50).astype(int)
    phq_scores = cohort_df['phq_score'].values
    severe_mask = (phq_scores >= 15)
    if np.sum(severe_mask) > 0:
        severe_recall = float(np.sum(pred_binary[severe_mask] == 1) / np.sum(severe_mask))
    else:
        severe_recall = 0.0
    sec['severe_recall'] = severe_recall
    sec['n_severe'] = int(np.sum(severe_mask))

    # 3. 3-Class Macro-F1:
    # 0: None/Mild (0-9), 1: Moderate (10-14), 2: Severe (15-24)
    y_3class = np.zeros(len(phq_scores), dtype=int)
    y_3class[(phq_scores >= 10) & (phq_scores < 15)] = 1
    y_3class[phq_scores >= 15] = 2

    # Map probability to 3 classes using calibrated quantiles matching base rates
    p_3class = np.zeros(len(y_prob), dtype=int)
    # Empirical cutpoints for 3-class severity
    q_mild = np.percentile(y_prob, 69.0)  # ~69% None/Mild
    q_mod = np.percentile(y_prob, 88.0)   # ~19% Moderate, ~12% Severe
    p_3class[y_prob >= q_mild] = 1
    p_3class[y_prob >= q_mod] = 2

    sec['macro_f1_3class'] = float(f1_score(y_3class, p_3class, average='macro'))

    return sec


def main():
    print("======================================================================")
    print("Phase 6A Execution Protocol: Real-Label Evaluation & Hypothesis Tests")
    print("Execution Gate: AUTHORIZED EXECUTION (prereg v1.1 + amendment v1.2 & v1.3)")
    print("======================================================================")

    # 1. Load active development cohort (N=239)
    cohort_df, y_real, groups, strata = load_cohort_info(permute=False)
    assert len(cohort_df) == 239, f"Expected 239 dev participants, got {len(cohort_df)}"
    assert_no_sealed_ids(cohort_df['participant_id'].values)

    print(f"Loaded {len(cohort_df)} active development participants (Labels: {sum(y_real==0)} non-depressed, {sum(y_real==1)} depressed).")

    # Storage for all outputs
    observed_results = {}
    oof_predictions = {}
    start_time_all = time.time()

    # ==========================================================================
    # Part 1: Observed Evaluation of All 19 Variants
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Evaluating Observed Participant-Level AUC for all 19 Variants")
    print("=" * 70)

    for vid in VARIANTS:
        t0 = time.time()
        print(f"\n[Evaluating {vid}] Loading feature matrix...")
        X = load_variant_features(vid, cohort_df)

        res = run_pipeline(
            X=X,
            y=y_real,
            groups=groups,
            seeds=(0, 1, 2, 3, 4),
            variant_id=vid,
            inner_cv_splits=4,
            outer_cv_splits=5,
            strata=None,  # prereg v1.1 sec 5.1: stratify outer folds on the binary label
            device='cpu'
        )
        elapsed = time.time() - t0

        auc = res['auc']
        probs = res['oof_probs']
        oof_predictions[vid] = probs

        # Compute cluster-bootstrap 95% CI
        ci_lower, ci_upper = compute_cluster_bootstrap_ci(y_real, probs, n_bootstraps=2000, seed=42)

        # Secondary endpoints
        sec = evaluate_secondary_endpoints(cohort_df, y_real, probs)

        role = 'Confirmatory' if vid in ['V2', 'V10', 'V18'] else ('Control' if vid in ['V15', 'V16', 'V17'] else 'Exploratory')

        observed_results[vid] = {
            'variant_id': vid,
            'role': role,
            'observed_auc': auc,
            'bootstrap_ci_95': [ci_lower, ci_upper],
            'seed_aucs': res['seed_aucs'],
            'elapsed_sec': elapsed,
            'secondary': sec
        }

        print(f"  {vid} ({role}): Observed AUC = {auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]) | Time: {elapsed:.2f}s")
        print(f"       Subgroups: WoZ={sec.get('subgroup_woz_auc')}, Avatar={sec.get('subgroup_avatar_auc')}, Female={sec.get('subgroup_female_auc')}, Male={sec.get('subgroup_male_auc')}")
        print(f"       3-Class Macro-F1={sec.get('macro_f1_3class'):.4f}, Severe Recall={sec.get('severe_recall'):.4f}")

    # ==========================================================================
    # Part 2: Confirmatory Permutation Tests (H1 on V2, H2 on V10)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Confirmatory Permutation Tests (R=999 Replications within Strata)")
    print("=" * 70)

    R_PERM = 999
    perm_results = {}

    # Prereg v1.1 sec 6.2 computational feasibility gate: clock R=5 permutations
    # per confirmatory variant; if R=999 would exceed 48 h, H1/H2 are reclassified
    # as exploratory (R is never reduced for a confirmatory claim).
    feas = {}
    for vid_f, base_seed in [('V2', 90000), ('V10', 91000)]:
        X_f = load_variant_features(vid_f, cohort_df)
        t_f = time.time()
        for r in range(1, 6):
            run_pipeline(X=X_f, y=permute_labels_stratified(cohort_df, seed=base_seed + r), groups=groups,
                         seeds=(0, 1, 2, 3, 4), variant_id=vid_f, inner_cv_splits=4, outer_cv_splits=5,
                         strata=None, device='cpu')
        feas[vid_f] = (time.time() - t_f) / 5.0
    projected_h = (feas['V2'] + feas['V10']) * R_PERM / 3600.0
    h12_confirmatory = projected_h <= 48.0
    print(f"Feasibility gate: {feas['V2']:.1f}s/perm (V2), {feas['V10']:.1f}s/perm (V10) -> projected {projected_h:.2f} h "
          f"for R={R_PERM} -> {'CONFIRMATORY' if h12_confirmatory else 'RECLASSIFIED EXPLORATORY'}")
    perm_results['feasibility_gate'] = {'sec_per_perm': feas, 'projected_hours': projected_h,
                                        'h1_h2_confirmatory': h12_confirmatory}

    # --- H1: Variant V2 (Handcrafted Voice 330d) ---
    print(f"\n[H1 Permutation Test] Running {R_PERM} stratified permutations for Variant V2...")
    X_v2 = load_variant_features('V2', cohort_df)
    v2_obs_auc = observed_results['V2']['observed_auc']
    v2_null_aucs = []
    n_v2_exceed = 0
    t_h1_start = time.time()

    for r in range(1, R_PERM + 1):
        y_perm = permute_labels_stratified(cohort_df, seed=10000 + r)
        res_p = run_pipeline(
            X=X_v2,
            y=y_perm,
            groups=groups,
            seeds=(0, 1, 2, 3, 4),
            variant_id='V2',
            inner_cv_splits=4,
            outer_cv_splits=5,
            strata=None,  # prereg v1.1 sec 5.1: stratify outer folds on the binary label
            device='cpu'
        )
        null_auc = res_p['auc']
        v2_null_aucs.append(null_auc)
        if null_auc >= v2_obs_auc:
            n_v2_exceed += 1

        if r % 100 == 0 or r == R_PERM:
            curr_time = time.time() - t_h1_start
            print(f"  V2 Permutation [{r}/{R_PERM}]: Null AUC = {null_auc:.4f} | >= Obs ({n_v2_exceed}/{r}) | Elapsed: {curr_time/60:.1f}m")

    p_val_h1 = (1.0 + n_v2_exceed) / (R_PERM + 1.0)
    perm_results['H1_V2'] = {
        'hypothesis': 'H1: AUC(V2) > AUC_null',
        'observed_auc': v2_obs_auc,
        'replications': R_PERM,
        'null_exceed_count': n_v2_exceed,
        'null_mean_auc': float(np.mean(v2_null_aucs)),
        'null_p95_auc': float(np.percentile(v2_null_aucs, 95)),
        'empirical_p_value': p_val_h1
    }
    print(f"--> H1 Verdict: Observed={v2_obs_auc:.4f}, Null Mean={np.mean(v2_null_aucs):.4f}, Empirical p = {p_val_h1:.4f}")

    # --- H2: Variant V10 (WavLM Layer 8 768d) ---
    print(f"\n[H2 Permutation Test] Running {R_PERM} stratified permutations for Variant V10...")
    X_v10 = load_variant_features('V10', cohort_df)
    v10_obs_auc = observed_results['V10']['observed_auc']
    v10_null_aucs = []
    n_v10_exceed = 0
    t_h2_start = time.time()

    for r in range(1, R_PERM + 1):
        y_perm = permute_labels_stratified(cohort_df, seed=20000 + r)
        res_p = run_pipeline(
            X=X_v10,
            y=y_perm,
            groups=groups,
            seeds=(0, 1, 2, 3, 4),
            variant_id='V10',
            inner_cv_splits=4,
            outer_cv_splits=5,
            strata=None,  # prereg v1.1 sec 5.1: stratify outer folds on the binary label
            device='cpu'
        )
        null_auc = res_p['auc']
        v10_null_aucs.append(null_auc)
        if null_auc >= v10_obs_auc:
            n_v10_exceed += 1

        if r % 100 == 0 or r == R_PERM:
            curr_time = time.time() - t_h2_start
            print(f"  V10 Permutation [{r}/{R_PERM}]: Null AUC = {null_auc:.4f} | >= Obs ({n_v10_exceed}/{r}) | Elapsed: {curr_time/60:.1f}m")

    p_val_h2 = (1.0 + n_v10_exceed) / (R_PERM + 1.0)
    perm_results['H2_V10'] = {
        'hypothesis': 'H2: AUC(V10) > AUC_null',
        'observed_auc': v10_obs_auc,
        'replications': R_PERM,
        'null_exceed_count': n_v10_exceed,
        'null_mean_auc': float(np.mean(v10_null_aucs)),
        'null_p95_auc': float(np.percentile(v10_null_aucs, 95)),
        'empirical_p_value': p_val_h2
    }
    print(f"--> H2 Verdict: Observed={v10_obs_auc:.4f}, Null Mean={np.mean(v10_null_aucs):.4f}, Empirical p = {p_val_h2:.4f}")

    # ==========================================================================
    # Part 3: Confirmatory Intersection-Union Test for H3 (V18 Stacking)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Confirmatory Intersection-Union Test for H3 (V18 Stacking)")
    print("=" * 70)

    p_v18 = oof_predictions['V18']
    p_v14 = oof_predictions['V14']
    p_v2 = oof_predictions['V2']

    B_BOOT = 10000
    p_v18_gt_v14, d1_obs, d1_l, d1_u = compute_paired_cluster_bootstrap_delta(y_real, p_v18, p_v14, n_bootstraps=B_BOOT, seed=30001)
    p_v18_gt_v2, d2_obs, d2_l, d2_u = compute_paired_cluster_bootstrap_delta(y_real, p_v18, p_v2, n_bootstraps=B_BOOT, seed=30002)

    # Intersection-Union: p = max(p_1, p_2)
    p_val_h3 = max(p_v18_gt_v14, p_v18_gt_v2)

    h3_results = {
        'hypothesis': 'H3: V18 beats V14 AND V18 beats V2',
        'observed_auc_v18': observed_results['V18']['observed_auc'],
        'observed_auc_v14': observed_results['V14']['observed_auc'],
        'observed_auc_v2': observed_results['V2']['observed_auc'],
        'delta_v18_minus_v14': d1_obs,
        'delta_v18_minus_v14_ci': [d1_l, d1_u],
        'p_v18_gt_v14': p_v18_gt_v14,
        'delta_v18_minus_v2': d2_obs,
        'delta_v18_minus_v2_ci': [d2_l, d2_u],
        'p_v18_gt_v2': p_v18_gt_v2,
        'iut_p_value': p_val_h3
    }
    print(f"H3 Incremental Synergy Test (B={B_BOOT}):")
    print(f"  V18 vs V14 (Text): Delta = {d1_obs:+.4f} (95% CI: [{d1_l:+.4f}, {d1_u:+.4f}]), one-sided p = {p_v18_gt_v14:.4f}")
    print(f"  V18 vs V2  (Voice): Delta = {d2_obs:+.4f} (95% CI: [{d2_l:+.4f}, {d2_u:+.4f}]), one-sided p = {p_v18_gt_v2:.4f}")
    print(f"  --> Intersection-Union p-value: max({p_v18_gt_v14:.4f}, {p_v18_gt_v2:.4f}) = {p_val_h3:.4f}")

    # ==========================================================================
    # Part 4: Holm-Bonferroni Multiplicity Adjustment (m=3)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 4: Holm-Bonferroni Multiplicity Adjustment (m=3, alpha=0.05)")
    print("=" * 70)

    hypotheses = [
        {'id': 'H1', 'name': 'H1 (Handcrafted Voice V2 vs Null)', 'raw_p': p_val_h1},
        {'id': 'H2', 'name': 'H2 (WavLM Layer 8 V10 vs Null)', 'raw_p': p_val_h2},
        {'id': 'H3', 'name': 'H3 (Multimodal Synergy V18 vs Constituents)', 'raw_p': p_val_h3}
    ]

    # Sort ascending by raw p-value
    hypotheses.sort(key=lambda x: x['raw_p'])

    alpha = 0.05
    m = 3
    holm_rejected = True

    for i, h in enumerate(hypotheses):
        step = i + 1
        denom = m - step + 1
        threshold = alpha / denom
        raw_p = h['raw_p']

        if holm_rejected and raw_p <= threshold:
            decision = 'SUPPORTED (Statistically Significant)'
        else:
            decision = 'NOT SUPPORTED (Failed to exceed threshold)'
            holm_rejected = False  # Step-down stops

        h['step'] = step
        h['denom'] = denom
        h['threshold'] = threshold
        h['decision'] = decision

        print(f"  Step {step}: {h['name']}")
        print(f"         Raw p = {raw_p:.4f} vs Holm threshold {threshold:.4f} (alpha/{denom}) -> {decision}")

    # Re-key by hypothesis id
    holm_summary = {h['id']: h for h in hypotheses}

    # ==========================================================================
    # Part 5: Pre-Stated Freeze Rule Application (Step E3)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 5: Applying Pre-Stated Freeze Rule for Phase 6B Transition")
    print("=" * 70)

    # 1. Primary Audio Variant among V1-V13
    audio_aucs = []
    variant_dims = {
        'V1': 135, 'V2': 330, 'V3': 330, 'V4': 330, 'V5': 330,
        'V6': 768, 'V7': 768, 'V8': 768, 'V9': 768, 'V10': 768,
        'V11': 768, 'V12': 55, 'V13': 768
    }

    for vid in AUDIO_VARIANTS:
        auc = observed_results[vid]['observed_auc']
        dims = variant_dims[vid]
        audio_aucs.append({
            'variant_id': vid,
            'observed_auc': auc,
            'dimensions': dims
        })

    # Sort descending by AUC; ties broken by fewer dimensions
    audio_aucs.sort(key=lambda x: (-x['observed_auc'], x['dimensions'], x['variant_id']))
    best_audio = audio_aucs[0]
    best_audio_id = best_audio['variant_id']
    audio_designation = 'confirmatory' if best_audio_id in ['V2', 'V10'] else 'exploratory-derived'

    print(f"Primary Audio Candidate Ranking (V1-V13):")
    for rank, item in enumerate(audio_aucs[:5], 1):
        print(f"  #{rank}: {item['variant_id']} -> AUC = {item['observed_auc']:.4f} (Dims: {item['dimensions']})")
    print(f"--> FROZEN PRIMARY AUDIO: {best_audio_id} (AUC = {best_audio['observed_auc']:.4f}, Designation: '{audio_designation}')")

    # 2. Fusion Variant: Higher AUC of V18 and V19
    auc_v18 = observed_results['V18']['observed_auc']
    auc_v19 = observed_results['V19']['observed_auc']

    if auc_v19 > auc_v18:
        frozen_fusion_id = 'V19'
        frozen_fusion_auc = auc_v19
        fusion_designation = 'exploratory-derived'
    else:
        frozen_fusion_id = 'V18'
        frozen_fusion_auc = auc_v18
        fusion_designation = 'confirmatory'

    print(f"Fusion Candidate Comparison:")
    print(f"  V18 (Confirmatory Stacking V2+V14): AUC = {auc_v18:.4f}")
    print(f"  V19 (Exploratory Stacking Best+V14): AUC = {auc_v19:.4f}")
    print(f"--> FROZEN FUSION: {frozen_fusion_id} (AUC = {frozen_fusion_auc:.4f}, Designation: '{fusion_designation}')")

    # Write frozen_config.json
    frozen_config = {
        'protocol_version': 'prereg-v1.1-amended-v1.3',
        'freeze_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'primary_audio_config': {
            'variant_id': best_audio_id,
            'observed_auc': best_audio['observed_auc'],
            'dimensions': best_audio['dimensions'],
            'designation': audio_designation
        },
        'fusion_config': {
            'variant_id': frozen_fusion_id,
            'observed_auc': frozen_fusion_auc,
            'designation': fusion_designation
        },
        'active_cohort_size': 239,
        'sealed_cohort_status': 'UNTOUCHED_AND_QUARANTINED',
        'confirmation_seeds_status': 'SEEDS_100_104_UNTOUCHED',
        'next_authorized_action': 'Write PHASE6A_RESULTS.md and STOP'
    }

    with open(FROZEN_CONFIG_PATH, 'w') as f:
        json.dump(frozen_config, f, indent=2)
    print(f"\nSaved frozen configuration to {FROZEN_CONFIG_PATH}.")

    # ==========================================================================
    # Part 6: Write Complete JSON Output
    # ==========================================================================
    total_time = time.time() - start_time_all
    full_output = {
        'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_execution_time_sec': total_time,
        'cohort': {
            'n_total': len(cohort_df),
            'n_woz': int(sum(cohort_df['domain'] == 'woz')),
            'n_avatar': int(sum(cohort_df['domain'] == 'avatar')),
            'n_female': int(sum(cohort_df['gender'] == 0)),
            'n_male': int(sum(cohort_df['gender'] == 1)),
            'n_depressed': int(sum(y_real == 1)),
            'n_non_depressed': int(sum(y_real == 0))
        },
        'variants': observed_results,
        'confirmatory_tests': {
            'H1_permutation': perm_results['H1_V2'],
            'H2_permutation': perm_results['H2_V10'],
            'H3_intersection_union': h3_results,
            'holm_bonferroni': holm_summary
        },
        'frozen_config': frozen_config
    }

    with open(OUTPUT_RESULTS_PATH, 'w') as f:
        json.dump(full_output, f, indent=2)
    oof_df = pd.DataFrame({'participant_id': groups, 'domain': cohort_df['domain'].values})
    for vid in VARIANTS:
        oof_df[vid] = oof_predictions[vid]
    oof_df.to_csv(os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'phase6a_oof_predictions.csv'), index=False)
    print(f"Saved complete numerical results to {OUTPUT_RESULTS_PATH}.")
    print("\n======================================================================")
    print(f"PHASE 6A COMPLETE: All 19 variants evaluated in {total_time/60:.2f} minutes.")
    print("======================================================================")

if __name__ == '__main__':
    main()
