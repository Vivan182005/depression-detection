# Execution Prompt — Depression Detection Capstone (hand this to your coding LLM)

Paste everything below into Claude Code (or whichever LLM is executing the
project) as a single instruction. It assumes the existing codebase described
in the project architecture doc (`DualBranchDepressionNet`,
`wav2vec_extractor.py`, `egemaps_extractor.py`, `calibrated_model.py`,
`audio_cleaner.py`) already exists and is being audited/hardened, not
rebuilt from scratch.

---

## ROLE

You are acting as a senior ML engineer reviewing and hardening an existing
depression-severity-estimation pipeline (DAIC-WOZ, audio-only, dual-branch
neural network: eGeMAPS + MFCC acoustic features fused with Wav2Vec2
embeddings, BiLSTM + self-attention, dual classification/regression heads).
This is a final-year capstone project and needs to be methodologically
defensible, not just high-scoring.

## NON-NEGOTIABLE CONSTRAINTS

1. **Do not artificially cap or clip any reported metric.** Never hard-code
   an accuracy ceiling. Instead: if validation or test accuracy exceeds
   ~85%, stop and run the leakage-audit checklist below before reporting
   the number. On DAIC-WOZ, published reproducibility studies found that
   most audio-only models scoring above that range were not learning
   depression-specific signal — they were exploiting subject leakage or
   interviewer-prompt shortcuts. A number in the 60–78% accuracy / 3–5
   point MAE range, properly validated, is a credible and citable result
   for this dataset — do not treat it as a failure to be optimized away.
2. **Never let the model access interviewer speech content or metadata**
   at any stage of feature extraction, training, or inference. Confirm the
   VAD-based isolation in `audio_cleaner.py` is actually removing
   interviewer audio, not just down-weighting it — write a quick spot-check
   (e.g. render 5 cleaned files back to audio and manually listen, or
   diff VAD segment boundaries against the transcript's interviewer/
   participant turn labels if available).
3. **Zero participant overlap between train/val/test.** This is already
   the stated design — verify it programmatically (assert on participant
   ID sets, not just trust the split logic) and print the assertion result
   in the training log.

## TASK 1 — Data Splitting

Given a few hundred labeled audio sessions:

### 1a. Reserve a held-out "final sanity check" set FIRST — before any other split

Before doing anything else with the data (before train/val/test, before any
CV fold assignment, before feature-store indexing is used for splitting
logic), carve out a small, fixed set of sessions that will **never be used
in training, validation, testing, hyperparameter tuning, or cross-
validation in any fold** — not even the "test" fold of TASK 1b.

- Pick **1–2 sessions from each of the 5 PHQ-8 severity classes**
  (Minimal/Normal, Mild, Moderate, Moderately Severe, Severe) — so roughly
  5–10 sessions total, depending on how many are available for the rarer
  classes.
- Within that, prefer picking at least one male and one female session per
  class where the data allows it, so the final check also covers the
  gender axis, not just severity.
- Physically move (or copy, if disk space allows) the chosen audio files
  into a separate folder named `unseen/` at the project root, with
  subfolders per severity class (e.g. `unseen/severe/`, `unseen/mild/`,
  etc.) so the true label is encoded in the folder path itself — no JSON
  manifest, no separate metadata file, nothing extra to maintain. The
  filenames themselves (or a short `README.md`/`.txt` inside `unseen/`
  listing filename → true PHQ-8 score, if a manifest-free label reference
  is needed) are enough.
- All data-loading and split-generation code must explicitly skip anything
  under `unseen/` — point the loader at the remaining raw-audio directory
  only, so `unseen/` is structurally outside the pipeline rather than
  filtered out by ID matching.
- Every downstream split (train/val/test AND every CV fold) must assert
  that none of its participant IDs match a file present in `unseen/` —
  fail loudly if they do.
- This set is not used for any metric that goes into the results report in
  TASK 5 — it exists purely so you can run `predict_single_audio.py` on
  known, human-labeled examples afterward and eyeball whether the model's
  output makes sense on cases it has truly never touched at any stage.
  Include the model's prediction vs. true label for each reserved session
  as a small table in the final deliverable, separate from the main
  results table.

### 1b. Split the remaining sessions

- Split at the **participant level**, stratified **jointly by PHQ-8
  severity bin AND self-reported gender** (not severity alone) — DAIC-WOZ
  has a documented gender confound where models learn a spurious
  gender→depression correlation if gender ends up unevenly distributed
  across splits.
- Use **70% train / 15% validation / 15% test** of the *remaining* sessions
  (after removing the reserved set above) as the primary holdout, but
  because a few hundred sessions yields a test fold of only ~30–45 sessions
  (too small for a stable point estimate), **also run 5-fold stratified
  group cross-validation** (same joint stratification) and report mean ±
  std across folds for every headline metric, alongside the single-holdout
  numbers. Report both — don't pick whichever looks better.
- `StandardScaler` (and any other fold-dependent normalization) must be
  fit only on the training fold of each split/fold — already correct per
  the existing pipeline; keep it that way and assert no scaler is fit on
  val/test data anywhere in the code.
- Before finalizing splits, verify every session actually has a ground-
  truth PHQ-8 label — do not silently include or drop unlabeled files.

## TASK 2 — Architecture Review

- The current "Gated Feature Fusion" layer is a plain concatenation with
  no learned gate. Either:
  (a) implement an actual gate — e.g. a small `Linear -> Sigmoid` per
      modality producing a scalar or per-dimension weight applied before
      concatenation — or
  (b) rename the layer/docs to "Feature Concatenation" so the
      architecture description matches what's implemented.
  Pick (a) if there's time budget; it's a defensible, cheap addition and
  gives you an ablation to report (gated vs. ungated fusion).
- Keep the rest of the architecture as-is (Wav2Vec2 branch, eGeMAPS/MFCC
  branch, 2-layer BiLSTM, self-attention, dual heads) — it matches current
  published practice for this task and does not need to be replaced.
- Add class-weighted (or focal) loss for the classification head to
  address severity-class imbalance (the "Severe" class will be rare in a
  few-hundred-session dataset). Keep the existing 0.1-weighted MSE
  auxiliary regression loss.

## TASK 3 — Bias Auditing (required, not optional)

- After every training run, compute and report **F1 separately for male
  and female subgroups**, plus a disparate-impact-style ratio
  (positive/depressed prediction rate for one gender ÷ the other). A ratio
  far from 1.0 indicates the model is using gender as a shortcut.
- Report **macro-F1 alongside weighted-F1** for the 5-class severity task
  — weighted F1 can look fine while the model is simply never predicting
  the rare classes.
- If a significant gender gap appears, try (in this order, report results
  for each): (1) confirming stratified split is correct, (2) class+gender-
  aware loss reweighting, (3) sub-sampling the majority gender/class
  combination. Report the accuracy/fairness trade-off honestly — a fairer
  model does not have to mean a worse one, but if it does, say so.

## TASK 4 — Calibration

- Keep the existing fix: `CrossEntropyLoss` trains on raw unscaled logits;
  `TemperatureScaler` is fit strictly post-hoc on validation-fold logits
  after training is fully frozen, and only rescales displayed probabilities
  — it must never change argmax/predicted class.
- Report Expected Calibration Error (ECE) before and after scaling, per
  fold if using cross-validation.

## TASK 5 — Evaluation & Reporting

Produce a results table with:
- Accuracy, macro-F1, weighted-F1, confusion matrix (5-class)
- PHQ-8 MAE, RMSE (regression head)
- Per-gender F1 and disparate-impact ratio
- ECE pre/post calibration
- Mean ± std of all of the above across the 5 CV folds, plus the single
  70/15/15 holdout numbers, clearly labeled as two separate protocols
- A short leakage-audit note confirming: no participant overlap, no
  interviewer-audio leakage, scaler fit only on train

## TASK 6 — Deliverable

Output:
1. Updated/annotated code for the fusion layer, loss function, and
   evaluation script.
2. A markdown results report following the table structure in Task 5.
3. A one-paragraph plain-language summary comparing your results to the
   published baselines: LSTM-audio-only (~66% accuracy), MFCC-RNN
   (~76% accuracy, RMSE 0.4), and noting where your fusion approach lands
   relative to those.
4. A prediction-vs-truth table for every file in `unseen/` — true severity
   class/PHQ-8 score (from the folder path or the `unseen/README.md`
   reference) vs. the model's predicted class/score and calibrated
   confidence — this is your live demo evidence for the final
   presentation/viva, so make sure it's easy to screenshot.

Do not overstate results. If aggregate accuracy exceeds ~85% at any point,
treat that as a bug/leakage signal to investigate before writing it into
the report, not as a milestone to celebrate.
