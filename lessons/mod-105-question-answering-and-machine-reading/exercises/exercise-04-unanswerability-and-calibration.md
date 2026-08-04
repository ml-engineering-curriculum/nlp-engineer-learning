# exercise-04: Unanswerability and Calibration

**Estimated effort:** 2 hours

## Objective

Train an extractive QA model on SQuAD 2.0, tune the null-score abstention threshold on a held-out split (not the full dev set), and evaluate the calibration properties of the resulting reader — including the risk-coverage curve and per-slice abstention behaviour. Show that a global scalar threshold is a *starting point*, not the end, by fitting a simple calibrator on features that improves the trade-off.

## Prerequisites

- Chapters [02](../02-extractive-qa-and-the-squad-formulation.md), [04](../04-em-and-f1-the-squad-evaluation-protocol.md), [09](../09-unanswerability-and-abstention-with-squad-2.md).
- `transformers`, `datasets`, `evaluate`, `torch`, `scikit-learn` (for the calibrator and risk-coverage curves).

## Dataset

Use **SQuAD 2.0** as the primary dataset:

```python
from datasets import load_dataset
raw = load_dataset("rajpurkar/squad_v2")
```

Approximately 50 % of dev questions are unanswerable by construction.

## Problem statement

### Part A — Train a SQuAD-2.0-aware reader

Fine-tune `microsoft/deberta-v3-base` (or `roberta-base`) on SQuAD 2.0 with the chapter 09 recipe:

- Unanswerable questions labelled with `start = end = 0` (CLS position).
- Otherwise the standard extractive-QA recipe from exercise-01.
- `max_seq_length = 384`, `doc_stride = 128`, LR `3e-5`, 2 epochs.

Save the checkpoint.

### Part B — Score the dev set

For every dev question:

- Compute `best_non_null_score` (highest scoring valid span across all chunks).
- Compute `null_score` (`start_logits[CLS] + end_logits[CLS]` in the chunk with the highest CLS-CLS sum).
- Compute `score_diff = best_non_null_score - null_score`.

Also record:

- Question length in tokens.
- Context length in tokens (total, across chunks).
- Best-span length in tokens.
- Question's first interrogative word.
- Gold answerability.

Save all of the above to a Parquet/CSV file for the later parts.

### Part C — Threshold sweep on a split

Stratify-split the SQuAD 2.0 dev set into `dev-tune` (2/3) and `dev-holdout` (1/3) by answerability.

On `dev-tune`, sweep `threshold` over `linspace(-10, 10, 201)`. For each value compute SQuAD 2.0 F1 and EM. Pick the F1-maximising threshold.

Evaluate that fixed threshold on `dev-holdout`. Also report `evaluate.load("squad_v2")`'s `best_f1_thresh` on the *full* dev set — the number you would have reported without the split — and compare. Comment on the inflation.

### Part D — Slice report

At the chosen threshold, report on `dev-holdout`:

- Overall SQuAD 2.0 F1 and EM.
- Answerable-F1 (F1 on answerable questions with non-empty predictions).
- Unanswerable-accuracy (fraction of unanswerable questions correctly abstained).
- Precision on non-empty predictions.
- Coverage (fraction of questions answered).
- Per-question-type answerable-F1 and unanswerable-accuracy.

Discuss any question type where the model is unusually well- or ill-calibrated.

### Part E — Risk-coverage curve

Sweep `threshold` from very abstention-loving to very answer-loving. At each threshold, plot:

- x-axis: coverage.
- y-axis: error rate on the *answered* subset (1 − precision on non-empty predictions).

Report the area under the risk-coverage curve (AURC — lower is better). Add a horizontal line at the error rate of the model at coverage = 1.

### Part F — A calibrator on features

Fit a logistic regression on `dev-tune` predicting *is the model's answer correct?* using features:

- `score_diff` (from Part B).
- `null_score`.
- `best_non_null_score`.
- Question length in tokens.
- Best-span length in tokens.
- One-hot of the first interrogative word.

At inference, replace the "threshold on `score_diff`" rule with "predict when the calibrator says P(correct) > τ", sweeping τ on `dev-tune` and evaluating on `dev-holdout`.

Report the resulting SQuAD 2.0 F1, precision on non-empty, coverage, and AURC. Compare to Part D.

### Part G — Write-up

A 400–600 word `README.md` covering:

- Dataset and training recipe.
- The inflation between `dev-tune`-tuned threshold and full-dev `best_f1_thresh`.
- Full slice report from Part D.
- Risk-coverage plot and AURC.
- Calibrator vs. global-threshold comparison from Part F.
- One example of a question where the calibrator abstained and the global threshold did not (or vice versa), with the raw scores.
- One thing you would try next (e.g., a verifier LLM, temperature scaling, per-slice thresholds).

## Starter guidance

- SQuAD 2.0 tokenisation and post-processing are almost identical to SQuAD 1.1; the only meaningful difference is the null-score comparison at decode time. The HF `run_qa.py` example supports both via a `--version_2_with_negative` flag — read the `postprocess_qa_predictions` function once end-to-end.
- Recording `score_diff` per question requires a small change to the post-processor. Do this once and cache to disk — Parts C, D, E, F all reuse it.
- Use `sklearn.linear_model.LogisticRegression` for the calibrator; it will train in seconds. Do not over-engineer.
- For the AURC computation, sort questions by descending confidence and accumulate errors as you lower the coverage threshold. Any monotone confidence signal works.
- On split-based evaluation: seed your stratified split (`random_state=42`) so the numbers are reproducible.

## Acceptance criteria

- [ ] SQuAD 2.0 fine-tuning script (`train_squad2.py`) reproduces the Part A recipe.
- [ ] Per-question score CSV/Parquet from Part B is saved and used downstream.
- [ ] Threshold sweep on `dev-tune` with F1 curve; chosen threshold applied to `dev-holdout`; inflation vs. full-dev `best_f1_thresh` reported.
- [ ] Slice report (answerable F1, unanswerable accuracy, precision on non-empty, coverage, per-question-type) at the chosen threshold.
- [ ] Risk-coverage plot with AURC value.
- [ ] Feature-based calibrator (Part F) trained and compared against the global threshold on the same metrics.
- [ ] 400–600 word write-up with one worked example of calibrator-vs-threshold disagreement.

## Stretch goals

- **Temperature scaling.** Fit a scalar $T$ on `dev-tune` to divide the start and end logits before softmax. Rerun Parts D and E with the temperature-scaled logits.
- **Per-question-type thresholds.** Fit a separate threshold per interrogative word on `dev-tune`. Compare to the global threshold on `dev-holdout`.
- **Ensemble null-scores.** Train two more seeds and average the null-scores across the three models. Does the AURC improve?
- **Verifier model.** Train (or use an off-the-shelf) NLI model as a verifier: it reads `(question, context, predicted_answer)` and predicts `entailment`. Threshold on entailment probability. How does its AURC compare to the calibrator in Part F?
- **Adversarial evaluation.** Evaluate on AdversarialQA (Bartolo et al., ["Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension"](https://arxiv.org/abs/2002.00293), *TACL 2020*) using the same threshold. How badly does the calibration deteriorate on adversarial questions?
