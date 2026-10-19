"""
show_sealed_predictions.py: per-person predictions for the 33 sealed (held-out) participants.

    .venv_opensmile/bin/python show_sealed_predictions.py              # print the table + save CSV
    .venv_opensmile/bin/python show_sealed_predictions.py --threshold youden

Columns
  participant_id   unique DAIC-WOZ / E-DAIC session ID (the dataset's own identifier)
  true_phq8 / truth  actual PHQ-8 score and label (DEPRESSED = PHQ-8 >= 10)
  V7 / V19         one-shot sealed test (models trained on the 239 development people only,
                   predictions locked before labels were read)
  Phase7           improved model, cross-validated on all 272 (each person predicted by a model
                   that did not train on them); flagged with the final model's threshold
Output CSV: reports/research_v2/sealed_33_predictions.csv (git-ignored: it contains dataset labels)
"""
import os
import json
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(ROOT, 'experiments', 'research_v2')
OUT = os.path.join(ROOT, 'reports', 'research_v2', 'sealed_33_predictions.csv')


def label(x):
    return 'DEPRESSED' if x else 'not'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', choices=['max_acc', 'youden'], default='max_acc',
                    help='Phase 7 operating point (V7/V19 use their locked sealed-test thresholds)')
    a = ap.parse_args()

    pred = pd.read_csv(os.path.join(RD, 'sealed_predictions.csv'))
    lab = pd.read_csv(os.path.join(RD, 'labels_official.csv')).set_index('participant_id')
    p7 = pd.read_csv(os.path.join(RD, 'phase7_oof_predictions.csv')).set_index('participant_id')
    thr = json.load(open(os.path.join(ROOT, 'models', 'final_phase7_p4.json')))['thresholds'][a.threshold]
    sealed = json.load(open(os.path.join(RD, 'sealed_results.json')))

    d = pred.join(lab[['phq8_score', 'gender', 'binary_label']], on='participant_id').join(p7[['P4_fusion']], on='participant_id')
    d = d.rename(columns={'phq8_score': 'true_phq8', 'binary_label': 'true_label', 'prob_audio': 'V7_score',
                          'pred_audio': 'V7_pred', 'prob_fusion': 'V19_score', 'pred_fusion': 'V19_pred',
                          'P4_fusion': 'Phase7_score'})
    d['Phase7_pred'] = (d['Phase7_score'] >= thr).astype(int)
    for m in ('V7', 'V19', 'Phase7'):
        d[f'{m}_correct'] = (d[f'{m}_pred'] == d['true_label'])
    d = d.sort_values(['true_label', 'true_phq8'], ascending=[False, False]).reset_index(drop=True)

    cols = ['participant_id', 'domain', 'gender', 'true_phq8', 'true_label', 'V7_score', 'V7_pred', 'V7_correct',
            'V19_score', 'V19_pred', 'V19_correct', 'Phase7_score', 'Phase7_pred', 'Phase7_correct']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d[cols].to_csv(OUT, index=False)

    ok = lambda b: 'correct' if b else 'WRONG'
    print(f"\n{'#':>2} {'ID':>4} {'setup':<6} {'sex':<6} {'PHQ-8':>5}  {'TRUTH':<9} | {'V7 voice':<22} | {'V19 voice+text':<22} | {'Phase 7 (CV)':<22}")
    print('-' * 112)
    for i, r in d.iterrows():
        print(f"{i + 1:>2} {r.participant_id:>4} {r.domain:<6} {r.gender:<6} {int(r.true_phq8):>5}  {label(r.true_label):<9} | "
              f"{r.V7_score:.2f} {label(r.V7_pred):<9} {ok(r.V7_correct):<7} | "
              f"{r.V19_score:.2f} {label(r.V19_pred):<9} {ok(r.V19_correct):<7} | "
              f"{r.Phase7_score:.2f} {label(r.Phase7_pred):<9} {ok(r.Phase7_correct):<7}")
    print('-' * 112)
    n, dep = len(d), int(d.true_label.sum())
    for m, name in (('V7', 'V7 voice (sealed test)'), ('V19', 'V19 voice + text (sealed test)'), ('Phase7', f'Phase 7 model (CV, {a.threshold})')):
        tp = int(((d[f'{m}_pred'] == 1) & (d.true_label == 1)).sum())
        tn = int(((d[f'{m}_pred'] == 0) & (d.true_label == 0)).sum())
        print(f'{name:<34} {int(d[f"{m}_correct"].sum())}/{n} correct ({d[f"{m}_correct"].mean():.1%}) | '
              f'depressed caught {tp}/{dep} | not-depressed cleared {tn}/{n - dep}')
    print(f"\nSealed-test AUC: V7 {sealed['audio_model']['auc']:.3f}, V19 {sealed['fusion_model']['auc']:.3f} | "
          f"predictions SHA-256 {sealed['predictions_sha256'][:16]}... (locked before labels were read)")
    print(f'Saved: {os.path.relpath(OUT, ROOT)}')


if __name__ == '__main__':
    main()
