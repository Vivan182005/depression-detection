"""
run_project.py: one command to reproduce the project's results.

    .venv_opensmile/bin/python run_project.py            # quick (~5 min): tests + Phase 7 model + final results
    .venv_opensmile/bin/python run_project.py --phase6a  # also re-run the pre-registered Phase 6A study (~2 h)
    .venv_opensmile/bin/python run_project.py --report   # only print the results already on disk

Uses the cached feature stores (extraction from raw audio takes several hours; see the
extract_* scripts in experiments/research_v2). The one-shot sealed evaluation is never re-run:
its lock file makes it refuse, and its saved result is printed instead.
"""
import os
import sys
import json
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(ROOT, 'experiments', 'research_v2')
PY = sys.executable
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}


def step(title, args, log):
    print(f'\n=== {title} ===', flush=True)
    t0 = time.time()
    with open(os.path.join(RD, log), 'w') as fh:
        r = subprocess.run([PY, '-W', 'ignore', *args], cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    print(f'    {"OK" if r.returncode == 0 else "FAILED"} in {time.time() - t0:.0f}s (log: experiments/research_v2/{log})', flush=True)
    if r.returncode != 0:
        sys.exit(f'Stopped: {title} failed. See the log above.')


def report():
    p7 = json.load(open(os.path.join(RD, 'phase7_results.json')))
    s = json.load(open(os.path.join(RD, 'sealed_results.json')))
    p6 = json.load(open(os.path.join(RD, 'phase6a_results.json')))
    fm = json.load(open(os.path.join(ROOT, 'models', 'final_phase7_p4.json')))
    line = '-' * 92
    print(f'\n{line}\nRESULTS: voice-primary depression detection (PHQ-8 >= 10)\n{line}')
    print('\n1) Pre-registered Phase 6A (N=239, 5x5 CV): confirmatory tests')
    for h in ('H1', 'H2', 'H3'):
        x = p6['confirmatory_tests']['holm_bonferroni'][h]
        print(f'   {x["name"]:<48} p = {x["raw_p"]:.4f}  -> {x["decision"]}')
    print(f'   Frozen: V7 voice AUC {p6["variants"]["V7"]["observed_auc"]:.3f}, V19 voice+text AUC {p6["variants"]["V19"]["observed_auc"]:.3f}')
    print(f'\n2) One-shot sealed test (N={s["n"]} never-seen people, {s["n_depressed"]} depressed)')
    for k, name in (('audio_model', 'V7 voice only'), ('fusion_model', 'V19 voice + text')):
        x = s[k]
        print(f'   {name:<18} AUC {x["auc"]:.3f} [{x["auc_ci95"][0]:.2f}, {x["auc_ci95"][1]:.2f}]  acc {x["accuracy"]:.1%}  '
              f'bal-acc {x["balanced_accuracy"]:.1%}  caught {x["confusion"]["tp"]}/{x["confusion"]["tp"] + x["confusion"]["fn"]}  p={x["auc_perm_p"]:.3f}')
    print(f'\n3) Phase 7 improved model (N={p7["n"]}, repeated nested CV; majority-class accuracy {p7["majority_class_accuracy"]:.1%})')
    print(f'   {"procedure":<34}{"AUC":>7}{"acc":>9}{"bal-acc":>9}{"sens":>7}{"spec":>7}   (max-accuracy threshold)')
    for k, x in p7['procedures'].items():
        m = x['operating_points']['max_acc']
        print(f'   {k:<34}{x["auc"]:>7.3f}{m["accuracy"]:>9.1%}{m["balanced_accuracy"]:>9.1%}{m["sensitivity"]:>7.0%}{m["specificity"]:>7.0%}')
    h = p7['procedures']['P4_fusion']
    y = h['operating_points']['youden']
    print(f'   P4 at balanced (Youden) threshold: acc {y["accuracy"]:.1%}, bal-acc {y["balanced_accuracy"]:.1%}, '
          f'sens {y["sensitivity"]:.0%}, spec {y["specificity"]:.0%}')
    print('   Bias check (P4 AUC): ' + ', '.join(f'{g} {v["auc"]:.3f}' for g, v in h['subgroups_max_acc_per_domain'].items()))
    print(f'\n4) Final saved model: models/final_phase7_p4.joblib')
    print(f'   voice = {[v[0] for v in fm["voice"]]}, text = {fm["text"][0]}, trained on {fm["trained_on"]}')
    print(f'   expected AUC on new people = {fm["phase7_nested_cv_auc"]:.3f} (nested CV), thresholds {fm["thresholds"]}')
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase6a', action='store_true', help='also re-run the pre-registered Phase 6A study (~2 h)')
    ap.add_argument('--report', action='store_true', help='only print results already on disk')
    a = ap.parse_args()
    if not a.report:
        step('Unit tests (fallbacks, pipeline engine, sealed guard)',
             ['-m', 'pytest', '-q', 'experiments/research_v2/test_silent_fallbacks.py',
              'experiments/research_v2/test_pipeline_engine.py', 'experiments/research_v2/test_sealed_guard.py'], 'run_tests.log')
        if a.phase6a:
            step('Phase 6A pre-registered 19-variant study + permutation tests (~2 h)',
                 ['experiments/research_v2/run_phase6a_experiment.py'], 'phase6a_run.log')
        step('Phase 7 nested cross-validation on all 272 participants', ['experiments/research_v2/run_phase7_experiments.py'], 'phase7_run.log')
        step('Train the final model on all 272 participants', ['experiments/research_v2/train_final_model_phase7.py'], 'train_final_model.log')
    report()


if __name__ == '__main__':
    main()
