# Repository Cleanup & File Classification Manifest

**Project**: Explainable Multi-Modal Speech & Language Depression Detection System  
**Repository**: `/Users/vivannaik/Desktop/CAPSTONE`  
**Date**: October 2026  
**Status**: Milestone Frozen & Validated  

---

## 1. Classification of Repository Assets

### Category A: Essential Production & Evaluation Assets (Retained in Place)
These files constitute the core production runtime, inference pipeline, and frozen evaluation:

| Path | Purpose |
| :--- | :--- |
| `artifacts/final_model/model_checkpoint.pth` | Canonical frozen PyTorch weights for Candidate 2A (`32968e...`) |
| `artifacts/final_model/feature_scalers.joblib` | Canonical training-only fitted standardizers (`9ae631...`) |
| `artifacts/final_model/model_config.json` | Complete neural architecture specification and hyperparameters |
| `artifacts/final_model/label_mapping.json` | PHQ-8 score definitions, clinical boundaries, and training conditional means $\mu$ |
| `artifacts/final_model/calibration_params.json` | Post-hoc temperature calibration parameter ($T=1.4542$) |
| `artifacts/final_model/model_manifest.json` | Complete provenance, SHA-256 hashes, Git commit, and metrics bundle |
| `artifacts/final_model/unseen_feature_cache_repaired.npz` | Repaired multimodal feature cache for all 33 unseen participants |
| `artifacts/final_model/unseen_predictions.csv` | Full predictions and probability distributions for the unseen cohort |
| `artifacts/final_model/unseen_evaluation_report.json` | Complete multi-metric evaluation report for the unseen cohort |
| `app/infer_depression.py` | Standalone reproducible CLI inference engine for raw audio and participant IDs |
| `app/evaluate_final_unseen.py` | Frozen unseen evaluation benchmark script |
| `app/src/features/*` | Acoustic, Wav2Vec 2.0, and BERT feature extraction modules |
| `app/src/models/*` | Architecture classes, calibration modules, and leak-free split utilities |
| `app/label_mapper.py` | Official PHQ-8 ground-truth score dictionary and category mapper |
| `app/run_preprocessing.py` | 33-dimensional temporal acoustic feature preprocessing |
| `app/feature_store/master_dataset_index.csv` | Complete index of all 183 training, validation, and test sessions |
| `new_audio_csv/detailed_lables.csv` | Authoritative clinical assessment scores and metadata |
| `unseen/` | Audio files and transcripts for all 33 reserved unseen sessions |
| `FINAL_MODEL_REPORT.md` | Authoritative final clinical report and handover document |
| `README.md` | Primary project overview documentation |

---

### Category B: Retained Historical Experiments & Evidence (Preserved in `experiments/`)
These files contain previous iterations, model candidates, and sweep benchmarks. They are preserved intact for research provenance:

| Path | Purpose |
| :--- | :--- |
| `experiments/capstone_improvement_v1/` | Initial apples-to-apples validation, Candidate 2, and Candidate 2A training logs |
| `experiments/capstone_improvement_v1/checkpoints/` | Historical checkpoints (`cand2_standard_mlp_balanced.pth`, etc.) |
| `experiments/capstone_accuracy_v2/` | Targeted ensemble sweep, Candidate 3A, Candidate 3C, and 5-fold CV benchmarks |
| `experiments/capstone_accuracy_v2/checkpoints/` | Candidate 3A and 3C checkpoints |
| `ACCURACY_IMPROVEMENT_REPORT.md` | Comprehensive architectural sweep report |
| `THEORY/` | Background literature and theoretical research notes |

---

### Category C: Disposable Cache Files (Safely Cleaned)
Temporary compiler caches that do not affect code logic and can be regenerated on demand:

- `experiments/capstone_improvement_v1/__pycache__/`
- `experiments/capstone_accuracy_v2/__pycache__/`
- `app/src/features/__pycache__/`
- `app/src/models/__pycache__/`

---

## 2. Integrity & Non-Destruction Guarantee

- **Zero Datasets Deleted**: All raw audio, transcripts, and label CSVs in `unseen/`, `new_audio_csv/`, `Clean_audio/`, and `NEW_AUDIO_DAIC/` are 100% preserved.
- **Zero Checkpoints Overwritten**: The original `cand2a_standard_mlp_mild_balanced.pth` and `cand2_standard_mlp_balanced.pth` files remain unmodified.
- **Zero Git Modifications**: The repository state has not been committed, reset, or pushed.
