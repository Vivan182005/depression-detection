# Research Copy Verification Report

**Project**: Explainable Multi-Modal Speech & Language Depression Detection System  
**Original Repository**: `/Users/vivannaik/Desktop/CAPSTONE`  
**Research Copy Path**: `/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`  
**Date**: October 13, 2026  
**Auditor**: Senior Machine Learning & Repository Maintenance Engineer  
**Status**: VERIFIED & AUDITED — PHASE 1 COMPLETE  

---

## 1. Executive Summary & Storage Verification

Prior to copying, disk space and repository dimensions were audited:
- **Available Disk Space**: 68 GiB free on `/System/Volumes/Data`.
- **Original Repository Size**: 9.2 GiB (including `.venv`), 8.032 GiB (excluding ephemeral `.venv` and compiler bytecode caches).
- **Copy Mechanism**: Safe preservation copy via `rsync -a` preserving all timestamps, symbolic links, file permissions, hidden files (`.git`), directory structures, and large audio/checkpoint artifacts.
- **Intentional Exclusions**:
  1. `.venv/` (1.1 GiB machine-local virtual environment directory containing environment symlinks; Python 3.14 system/brew packages are used directly).
  2. `__pycache__/` and `*.pyc` (re-generable bytecode caches).
  3. `.DS_Store` (macOS filesystem metadata).

---

## 2. File Inventory Comparison

| Partition / Category | Original Files | Copied Files | Variance | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Non-Ephemeral Files** | **1,204** | **1,204** | **0** | **Exact Match** |
| **Total Non-Ephemeral Volume** | **8.032 GB** | **8.032 GB** | **0.000 GB** | **Exact Match** |
| `Clean_audio/` (Raw DAIC audio) | 189 files | 189 files | 0 | **Preserved** |
| `NEW_AUDIO_DAIC/` (Processed audio) | 189 files | 189 files | 0 | **Preserved** |
| `unseen/` (Reserved holdout cohort) | 68 files | 68 files | 0 | **Preserved** |
| `app/feature_store/` (Multimodal features) | 4 files | 4 files | 0 | **Preserved** |
| `artifacts/final_model/` (Frozen checkpoint/scalers) | 8 files | 8 files | 0 | **Preserved** |
| `experiments/` (Historical models/checkpoints) | 33 files | 33 files | 0 | **Preserved** |
| `.git/` (Complete commit history) | Complete | Complete | 0 | **Preserved** |

---

## 3. Cryptographic Checksum (SHA-256) Verification

All key model weights, feature standardizers, configuration schemas, ground-truth label maps, inference engines, and evaluation caches were verified using SHA-256:

| Artifact Path | SHA-256 Checksum (Original & Copy) | Status |
| :--- | :--- | :---: |
| `artifacts/final_model/model_checkpoint.pth` | `32968e86d2dbaf31019eb01e3f6dde7bda87fb8236b9861b8e8bcb7a52f355cd` | **MATCH** |
| `artifacts/final_model/feature_scalers.joblib` | `9ae631cd1750a078505e7c3defa0575b06f2dc799941fe4e9dfd15d1f25cf39b` | **MATCH** |
| `artifacts/final_model/model_config.json` | `cdfbb5f0c2011e4f05ec95eeb091757545ff691516090855cea04f0aecfd1328` | **MATCH** |
| `artifacts/final_model/label_mapping.json` | `17cb11a620d740a277164229a0e7fe76066d5347052eb3e0d387724bc2af6167` | **MATCH** |
| `artifacts/final_model/calibration_params.json` | `ccb542213372217eb1d05b918de088d951e7b77c0e0c89dce9e75ba4baef459f` | **MATCH** |
| `artifacts/final_model/unseen_feature_cache_repaired.npz` | `36f4c919b739caba1c7090e8453166007cc5962577782976c4901d949359a3e3` | **MATCH** |
| `artifacts/final_model/unseen_predictions.csv` | `8cfa8bbd20ad0d7681ba40f4b3f4b1968554fab04334149491a388bb9954a091` | **MATCH** |
| `artifacts/final_model/unseen_evaluation_report.json` | `3de6ff556b293d8c6367077b41ddf57b8b292e223e29fc41eafd4341f00b8b1e` | **MATCH** |
| `experiments/capstone_improvement_v1/checkpoints/cand2a_standard_mlp_mild_balanced.pth` | `32968e86d2dbaf31019eb01e3f6dde7bda87fb8236b9861b8e8bcb7a52f355cd` | **MATCH** |
| `experiments/capstone_improvement_v1/checkpoints/cand2a_feature_scalers.joblib` | `9ae631cd1750a078505e7c3defa0575b06f2dc799941fe4e9dfd15d1f25cf39b` | **MATCH** |
| `app/infer_depression.py` | `b735ea67a20d8ac8e012c7dd9aedf20928d0b28a6d7d6cfc319f9d94e6f30012` | **MATCH** |
| `app/evaluate_final_unseen.py` | `859a3ed7fa27ef164bc75dc7110ca4ff5952f0928b78f6e3cc9320189df1c5b5` | **MATCH** |
| `app/src/features/bert_extractor.py` | `2a5418f0602123961c7ccdffe882147d11addb011774ff56a956ec9d125c7691` | **MATCH** |
| `app/src/features/wav2vec_extractor.py` | `0382f22ffbe3139f2e46ca8114a86ed043f2878578c4a951de79ad95b7532d3e` | **MATCH** |
| `app/src/features/egemaps_extractor.py` | `93159afc547f784f46c7bdd992236900bc08be3025c931d5b75fe152cc9a7945` | **MATCH** |
| `app/src/models/calibrated_model.py` | `41be5a0db8a16f418ba3c97559afaa09317d71667ee156b1183280ddfd666141` | **MATCH** |
| `app/label_mapper.py` | `d2dd10d4654e7cfd26d4bb3b1f9a32d0f2de4372e60899abd640812c59100011` | **MATCH** |

---

## 4. Git State & Branch Isolation

- **Original Directory (`/Users/vivannaik/Desktop/CAPSTONE`)**:
  - Branch: `main`
  - Head Commit: `9fd65d8631507ac0c8c6290e30dbad612bf9b65f`
  - Status: Strictly untouched, uncommitted working tree preserved.
- **Research Copy (`/Users/vivannaik/Desktop/CAPSTONE_RESEARCH_COPY_2026-10-13`)**:
  - Branch: `research-upgrade-2026-10-13` (created and checked out).
  - Base Commit: `9fd65d8631507ac0c8c6290e30dbad612bf9b65f`.
  - Status: Isolated research environment. All subsequent operations are confined to this path.

---

## 5. Verification Conclusion

Phase 1 verification is completely successful. Zero file or hash discrepancies exist. The original CAPSTONE directory remains completely untouched.
