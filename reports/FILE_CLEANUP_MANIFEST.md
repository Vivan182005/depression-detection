# Repository File Cleanup & Classification Manifest

**Project**: Explainable Multi-Modal Speech & Language Depression Detection System  
**Repository**: `/Users/vivannaik/Desktop/CAPSTONE`  
**Date**: October 2026  
**Status**: Pre-Demonstration Verified  

---

## 1. Classification of Repository Files

### Category A: Essential Production & Demonstration Assets (Retained in Place)
These files constitute the runnable application, inference engine, frozen model package, and verified evaluation artifacts:

| File / Directory | Description | Status |
| :--- | :--- | :---: |
| `artifacts/final_model/model_checkpoint.pth` | Canonical frozen weights for Candidate 2A (`32968e...`) | **Preserved** |
| `artifacts/final_model/feature_scalers.joblib` | Canonical training-only fitted standardizers (`9ae631...`) | **Preserved** |
| `artifacts/final_model/model_config.json` | Neural network architecture and hyperparameter specification | **Preserved** |
| `artifacts/final_model/label_mapping.json` | PHQ-8 severity thresholds and training conditional means $\mu$ | **Preserved** |
| `artifacts/final_model/calibration_params.json` | Temperature scaling parameters ($T=1.4542$) | **Preserved** |
| `artifacts/final_model/model_manifest.json` | Complete provenance, SHA-256 checksums, and metrics bundle | **Preserved** |
| `artifacts/final_model/unseen_feature_cache_repaired.npz` | Multimodal features for all 33 unseen participants | **Preserved** |
| `artifacts/final_model/unseen_predictions.csv` | Full prediction rows and probability distributions | **Preserved** |
| `artifacts/final_model/unseen_evaluation_report.json` | Complete multi-metric evaluation report on unseen cohort | **Preserved** |
| `app/infer_depression.py` | Production CLI inference engine (clean single-severity output) | **Preserved** |
| `app/evaluate_final_unseen.py` | Frozen unseen evaluation benchmark script | **Preserved** |
| `app/src/features/*` | Acoustic, Wav2Vec 2.0, and BERT feature extraction modules | **Preserved** |
| `app/src/models/*` | PyTorch architectures, temperature calibration, and dataset utils | **Preserved** |
| `app/label_mapper.py` | Clinical ground-truth label mapper | **Preserved** |
| `app/run_preprocessing.py` | Temporal acoustic feature preprocessing | **Preserved** |
| `app/feature_store/master_dataset_index.csv` | Index of all 183 training/val/test sessions | **Preserved** |
| `new_audio_csv/detailed_lables.csv` | Authoritative clinical ground-truth scores | **Preserved** |
| `unseen/` | Audio files and transcripts for all 33 holdout sessions | **Preserved** |
| `reports/FINAL_DEMO_AUDIT.md` | Pre-demonstration technical audit | **Preserved** |
| `FINAL_MODEL_REPORT.md` | Final frozen clinical milestone report | **Preserved** |
| `README.md` | Repository overview documentation | **Preserved** |

---

### Category B: Retained Historical Experiments & Evidence (Preserved in `experiments/`)
These files contain previous iterations, model candidates, and ablation benchmarks. Retained for scientific provenance:

| Directory / File | Description | Status |
| :--- | :--- | :---: |
| `experiments/capstone_improvement_v1/` | Initial apples-to-apples validation and Candidate 2 training logs | **Preserved** |
| `experiments/capstone_improvement_v1/checkpoints/` | Historical checkpoints (`cand2_standard_mlp_balanced.pth`, etc.) | **Preserved** |
| `experiments/capstone_accuracy_v2/` | Targeted ensemble sweep, Candidate 3A/3C benchmarks, and 5-fold CV | **Preserved** |
| `experiments/capstone_accuracy_v2/checkpoints/` | Candidate 3A and 3C checkpoints | **Preserved** |
| `ACCURACY_IMPROVEMENT_REPORT.md` | Comprehensive architectural sweep report | **Preserved** |
| `THEORY/` | Background literature and theoretical research notes | **Preserved** |

---

### Category C: Disposable Cache Files (Safely Cleaned)
Temporary compiler artifacts that do not contain code or data:

- `experiments/*/__pycache__/` (Cleaned)
- `app/*/__pycache__/` (Cleaned)
- `app/src/*/__pycache__/` (Cleaned)

---

## 2. Integrity & Non-Destruction Verification

- **Zero Datasets Deleted**: Raw audio and transcripts in `unseen/`, `NEW_AUDIO_DAIC/`, `Clean_audio/`, and `new_audio_csv/` are 100% preserved.
- **Zero Checkpoints Overwritten**: The primary `cand2a_standard_mlp_mild_balanced.pth` and secondary checkpoints remain unmodified.
- **Zero Git Modifications**: The working tree remains uncommitted and untracked.
