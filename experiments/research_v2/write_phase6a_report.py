"""
Script: experiments/research_v2/write_phase6a_report.py
Renders phase6a_results.json into reports/research_v2/PHASE6A_RESULTS.md.
Reports ALL 19 variants (prereg v1.1 sec 6.3 mandatory reporting), including null results.
"""
import os
import json

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE'
RES = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'phase6a_results.json')
OUT = os.path.join(_REPO_ROOT, 'reports', 'research_v2', 'PHASE6A_RESULTS.md')

DESCR = {
    'V1': 'eGeMAPS 135d speech-only + elastic-net LR', 'V2': 'Hand-crafted VAD-aggregated 330d + L2 LR',
    'V3': 'Hand-crafted 330d + HistGBDT', 'V4': 'Hand-crafted CMVN 330d + L2 LR',
    'V5': 'Hand-crafted CMVN 330d + HistGBDT', 'V6': 'wav2vec2 L4 mean + L2 LR', 'V7': 'wav2vec2 L8 mean + L2 LR',
    'V8': 'wav2vec2 L12 mean + L2 LR', 'V9': 'WavLM L4 mean + L2 LR', 'V10': 'WavLM L8 mean + L2 LR',
    'V11': 'WavLM L12 mean + L2 LR', 'V12': 'Segment hand-crafted 55d + MLP', 'V13': 'Segment WavLM L8 + MLP',
    'V14': 'BERT on Whisper-tiny (VAD-chunked) text + L2 LR', 'V15': 'Length-only control',
    'V16': 'Gender-only control', 'V17': 'Domain-only control', 'V18': 'Stack V2 + V14',
    'V19': 'Stack inner-CV best audio + V14',
}


def fmt(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def main():
    r = json.load(open(RES))
    c = r['cohort']
    L = []
    L.append('# Phase 6A Results: Voice-Primary Depression Detection (PHQ-8 >= 10)\n')
    L.append(f"Executed {r['execution_timestamp']} | total runtime {r['total_execution_time_sec'] / 60:.1f} min | "
             f"protocol prereg v1.1 + amendments v1.2, v1.3, v1.3b, v1.3c\n")
    L.append(f"Cohort: N={c['n_total']} dev participants ({c['n_woz']} WoZ, {c['n_avatar']} Avatar; "
             f"{c['n_female']} F / {c['n_male']} M; {c['n_depressed']} depressed / {c['n_non_depressed']} not). "
             'Sealed 33 and seeds 100-104 untouched.\n')
    L.append('Primary endpoint: participant-level ROC-AUC of out-of-fold probabilities averaged over seeds 0-4 '
             '(5 x 5-fold stratified CV, all tuning in inner 4-fold CV). 95% CI = participant bootstrap (B=2000).\n')

    L.append('## 1. All 19 variants\n')
    L.append('| Variant | Role | Description | AUC | 95% CI | WoZ AUC | Avatar AUC | Female AUC | Male AUC | 3-class macro-F1 | Severe recall |')
    L.append('|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|')
    for vid, v in r['variants'].items():
        s = v['secondary']
        ci = v['bootstrap_ci_95']
        L.append(f"| {vid} | {v['role']} | {DESCR.get(vid, '')} | **{v['observed_auc']:.3f}** | "
                 f"[{ci[0]:.3f}, {ci[1]:.3f}] | {fmt(s.get('subgroup_woz_auc'))} | {fmt(s.get('subgroup_avatar_auc'))} | "
                 f"{fmt(s.get('subgroup_female_auc'))} | {fmt(s.get('subgroup_male_auc'))} | "
                 f"{fmt(s.get('macro_f1_3class'))} | {fmt(s.get('severe_recall'))} |")
    L.append('')

    ct = r['confirmatory_tests']
    L.append('## 2. Confirmatory family (Holm, m = 3, alpha = 0.05)\n')
    fg = ct.get('feasibility_gate')
    if fg:
        L.append(f"Feasibility gate: projected {fg['projected_hours']:.2f} h for R=999 -> "
                 f"H1/H2 {'confirmatory' if fg['h1_h2_confirmatory'] else 'reclassified exploratory'}.\n")
    for key, name in [('H1_permutation', 'H1: V2 hand-crafted voice vs stratified permutation null'),
                      ('H2_permutation', 'H2: V10 WavLM-L8 vs stratified permutation null')]:
        h = ct[key]
        L.append(f"- **{name}**: observed AUC {h['observed_auc']:.3f}; null mean {h['null_mean_auc']:.3f}, "
                 f"null 95th pct {h['null_p95_auc']:.3f}; empirical p = {h['empirical_p_value']:.4f} (R={h['replications']})")
    h3 = ct['H3_intersection_union']
    L.append(f"- **H3: V18 beats V14 AND V2**: dAUC(V18-V14) = {h3['delta_v18_minus_v14']:+.3f} "
             f"[{h3['delta_v18_minus_v14_ci'][0]:+.3f}, {h3['delta_v18_minus_v14_ci'][1]:+.3f}], p={h3['p_v18_gt_v14']:.4f}; "
             f"dAUC(V18-V2) = {h3['delta_v18_minus_v2']:+.3f} "
             f"[{h3['delta_v18_minus_v2_ci'][0]:+.3f}, {h3['delta_v18_minus_v2_ci'][1]:+.3f}], p={h3['p_v18_gt_v2']:.4f}; "
             f"IUT p = {h3['iut_p_value']:.4f}\n")
    L.append('| Step | Hypothesis | raw p | Holm threshold | Decision |')
    L.append('|--:|:--|--:|--:|:--|')
    for hid in sorted(ct['holm_bonferroni'], key=lambda k: ct['holm_bonferroni'][k]['step']):
        h = ct['holm_bonferroni'][hid]
        L.append(f"| {h['step']} | {h['name']} | {h['raw_p']:.4f} | {h['threshold']:.4f} | {h['decision']} |")
    L.append('')

    fc = r['frozen_config']
    L.append('## 3. Pre-stated freeze rule (v1.3 sec 4)\n')
    L.append(f"- Primary audio: **{fc['primary_audio_config']['variant_id']}** "
             f"(AUC {fc['primary_audio_config']['observed_auc']:.3f}, {fc['primary_audio_config']['designation']})")
    L.append(f"- Fusion: **{fc['fusion_config']['variant_id']}** "
             f"(AUC {fc['fusion_config']['observed_auc']:.3f}, {fc['fusion_config']['designation']})")
    L.append('- Freezing authorises no run on seeds 100-104 or the sealed 33 (next step requires explicit sign-off).\n')

    L.append('## 4. Leakage / confound audit triggers (v1.1 sec 6.3)\n')
    flags = [f"{vid} (AUC {v['observed_auc']:.3f})" for vid, v in r['variants'].items()
             if (vid in [f'V{i}' for i in range(1, 14)] and v['observed_auc'] > 0.75)
             or (vid in ['V18', 'V19'] and v['observed_auc'] > 0.85)]
    L.append('- Variants above the audit thresholds: ' + (', '.join(flags) if flags else 'none') + '\n')
    ctrl = max(r['variants'][k]['observed_auc'] for k in ['V15', 'V16', 'V17'])
    L.append(f"- Best metadata control AUC: {ctrl:.3f}. Any variant whose CI overlaps this value has not shown "
             'signal beyond length/gender/domain.\n')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
