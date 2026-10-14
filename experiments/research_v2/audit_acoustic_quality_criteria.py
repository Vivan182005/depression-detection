"""
Script: experiments/research_v2/audit_acoustic_quality_criteria.py
Audits acoustic quality criteria, missingness, and invalid-estimate rates across
all 239 development participants in feature_store_v2.
"""

import os
import sys
import numpy as np
import pandas as pd

_REPO_ROOT = '/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.research_v2.sealed_guard import SEALED_PIDS, assert_no_sealed_ids

FEATURE_STORE_V2_DIR = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'feature_store_v2')
MANIFEST_PATH = os.path.join(FEATURE_STORE_V2_DIR, 'manifest.csv')
OFFICIAL_LABELS_PATH = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'labels_official.csv')

def main():
    df_manifest = pd.read_csv(MANIFEST_PATH)
    dev_df = df_manifest[(~df_manifest['is_sealed']) & (df_manifest['feature_file'] != 'none')]
    assert_no_sealed_ids(dev_df['participant_id'].values)
    
    print(f"Auditing acoustic quality across {len(dev_df)} dev sessions ({len(dev_df[dev_df['domain']=='woz'])} WoZ, {len(dev_df[dev_df['domain']=='avatar'])} Avatar)...")
    
    results = []
    
    for _, row in dev_df.iterrows():
        pid = int(row['participant_id'])
        dom = row['domain']
        feat_path = os.path.join(FEATURE_STORE_V2_DIR, row['feature_file'])
        
        data = np.load(feat_path)
        ac_154 = data['acoustic_154d']
        
        # Exact eGeMAPSv02 feature indices:
        # Index 0: F0semitoneFrom27.5Hz_sma3nz_amean
        # Index 30: jitterLocal_sma3nz_amean
        # Index 32: shimmerLocaldB_sma3nz_amean
        # Index 34: HNRdBACF_sma3nz_amean
        # Index 40: F1frequency_sma3nz_amean
        # Index 46: F2frequency_sma3nz_amean
        # Index 52: F3frequency_sma3nz_amean
        
        f0_semitone = ac_154[0]
        f0_hz = 27.5 * (2.0 ** (f0_semitone / 12.0))
        jitter = ac_154[30]
        shimmer = ac_154[32]
        hnr = ac_154[34]
        f1 = ac_154[40]
        f2 = ac_154[46]
        f3 = ac_154[52]
        
        f0_valid = (60.0 <= f0_hz <= 400.0) and not np.isnan(f0_hz)
        jitter_valid = (0.0 <= jitter <= 0.12) and not np.isnan(jitter)
        shimmer_valid = (0.0 <= shimmer <= 3.5) and not np.isnan(shimmer)
        hnr_valid = (0.0 <= hnr <= 30.0) and not np.isnan(hnr)
        formants_valid = (200.0 <= f1 <= 1200.0) and (600.0 <= f2 <= 3000.0) and (1500.0 <= f3 <= 4500.0) and (f1 < f2 < f3)
        all_valid = f0_valid and jitter_valid and shimmer_valid and hnr_valid and formants_valid
        
        n_zeros = (ac_154 == 0.0).sum()
        n_nans = np.isnan(ac_154).sum()
        
        results.append({
            'participant_id': pid,
            'domain': dom,
            'f0_hz': round(f0_hz, 1),
            'f0_valid': f0_valid,
            'jitter': round(jitter, 4),
            'jitter_valid': jitter_valid,
            'shimmer_db': round(shimmer, 3),
            'shimmer_valid': shimmer_valid,
            'hnr_db': round(hnr, 2),
            'hnr_valid': hnr_valid,
            'f1_hz': round(f1, 1),
            'f2_hz': round(f2, 1),
            'f3_hz': round(f3, 1),
            'formants_valid': formants_valid,
            'all_prosody_valid': all_valid,
            'n_zero_features': n_zeros,
            'n_nan_features': n_nans
        })
        
    df_out = pd.DataFrame(results)
    out_csv = os.path.join(_REPO_ROOT, 'experiments', 'research_v2', 'acoustic_quality_audit.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Saved acoustic quality audit to {out_csv}")
    
    print("\n========================================================")
    print("ACOUSTIC QUALITY & MISSINGNESS AUDIT SUMMARY")
    print("========================================================")
    for dom in ['woz', 'avatar']:
        sub = df_out[df_out['domain'] == dom]
        print(f"\nCohort Domain: {dom.upper()} (N = {len(sub)})")
        print(f"  F0 Validity Rate         : {sub['f0_valid'].mean()*100:.1f}% ({sub['f0_valid'].sum()}/{len(sub)})")
        print(f"  Jitter Validity Rate     : {sub['jitter_valid'].mean()*100:.1f}% ({sub['jitter_valid'].sum()}/{len(sub)})")
        print(f"  Shimmer Validity Rate    : {sub['shimmer_valid'].mean()*100:.1f}% ({sub['shimmer_valid'].sum()}/{len(sub)})")
        print(f"  HNR Validity Rate        : {sub['hnr_valid'].mean()*100:.1f}% ({sub['hnr_valid'].sum()}/{len(sub)})")
        print(f"  Formants F1<F2<F3 Valid  : {sub['formants_valid'].mean()*100:.1f}% ({sub['formants_valid'].sum()}/{len(sub)})")
        print(f"  Complete Prosody Valid   : {sub['all_prosody_valid'].mean()*100:.1f}% ({sub['all_prosody_valid'].sum()}/{len(sub)})")
        print(f"  Average Zero Features    : {sub['n_zero_features'].mean():.1f} / 154")
        print(f"  Average NaN Features     : {sub['n_nan_features'].mean():.1f} / 154")
    print("========================================================\n")

if __name__ == '__main__':
    main()
