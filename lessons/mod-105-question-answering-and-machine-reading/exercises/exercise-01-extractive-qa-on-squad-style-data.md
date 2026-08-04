# exercise-01: Extractive QA on SQuAD-style Data

**Estimated effort:** 3 hours

## Objective

Train an extractive-QA model on SQuAD 1.1 end-to-end, get the sliding-window preprocessing and offset mapping right, decode with the top-*k* span aggregator, and report EM and F1 with the official SQuAD normaliser — sliced by question type and answer length. This is the "workhorse recipe" you should be able to reproduce from scratch in an afternoon whenever a new extractive-QA dataset lands.

## Prerequisites

- Chapters [01](../01-question-answering-and-the-machine-reading-landscape.md), [02](../02-extractive-qa-and-the-squad-formulation.md), [03](../03-training-and-decoding-extractive-qa-models.md), [04](../04-em-and-f1-the-squad-evaluation-protocol.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `torch`.
- A GPU strongly recommended for `-base` encoders. CPU + a `-small` or `distil*` variant is workable but slow.

## Dataset

Use SQuAD 1.1 as the primary dataset:

```python
from datasets import load_dataset
raw = load_dataset("rajpurkar/squad")
```

Optional: also run one of the following for a comparison in Part F.

- **SQuADShifts** — Miller et al., ["The Effect of Natural Distribution Shift on Question Answering Models"](https://arxiv.org/abs/2004.14444), *ICML 2020*. `datasets.load_dataset("squadshifts", "new_wiki")`.
- **XQuAD** (a non-English target language) — Artetxe, Ruder & Yogatama, ["On the Cross-lingual Transferability of Monolingual Representations"](https://arxiv.org/abs/1910.11856), *ACL 2020*. `datasets.load_dataset("google/xquad", "xquad.es")`.

## Problem statement

### Part A — Preprocessing audit

Before training, build the sliding-window preprocessor from chapter 02 and print 10 tokenised training examples. For each example show:

- The tokenised sequence with subword pieces.
- `sequence_ids` marking question vs. context vs. special tokens.
- The `start_positions` and `end_positions`.
- The substring `context[offset_map[start_pos][0] : offset_map[end_pos][1]]`.

Assert that the sliced substring equals the gold `answers["text"][0]` for every example. Print any that fail.

Ship the audit as `preprocess_audit.py`. It must exit non-zero if any assertion fails.

### Part B — Train the workhorse model

Fine-tune `microsoft/deberta-v3-base` (or `roberta-base` if you want the reference recipe) with the settings from chapter 03:

- `AutoModelForQuestionAnswering`
- `max_seq_length = 384`, `doc_stride = 128`
- LR `3e-5`, effective batch `32`, `num_train_epochs = 2`, `warmup_ratio = 0.1`, `weight_decay = 0.01`
- `fp16 = True` (or `bf16 = True` on Hopper)
- `load_best_model_at_end = True`, `metric_for_best_model = "f1"`

Save training logs to disk.

### Part C — Decoding

Implement (or reuse from the HF example) `postprocess_qa_predictions`:

- Top-*k* start and end indices with `k = 20`.
- Filter by `sequence_ids == 1`, `end >= start`, `end - start + 1 <= 30`.
- Merge candidates across all chunks belonging to the same question.
- Slice the predicted answer string from the *original* context via `offset_mapping`.

Verify by picking 20 random dev questions and manually confirming that the predicted answer is a substring of the original context. Ship as `postprocess.py`.

### Part D — SQuAD evaluation

Compute dev-set EM and F1 with `evaluate.load("squad")`. Report:

- Overall EM and F1.
- 95 % bootstrap confidence intervals (1000 resamples over dev questions).
- Per-question-type F1, grouped by the first interrogative word (`who`, `what`, `when`, `where`, `why`, `how`, `other`).
- Per-answer-length-bucket F1 (1 token, 2–5, 6+).
- Per-context-length-bucket F1 (single-window vs. sliding-window).

Present as Markdown tables in the write-up.

### Part E — Failure analysis

Sample 20 dev questions where your model scores F1 = 0. Categorise each into:

- **Preprocessing bug** — the gold span was not aligned into the token space.
- **Post-processing bug** — the model logits favoured the right span but the top-*k* prune or filter missed it.
- **Reader failure** — the model actually preferred the wrong span.
- **Annotation ambiguity** — the gold answer is defensibly wrong or multiple answers are valid.

Report the counts and, for each category, one worked example with the tokenised input and the top-3 predicted spans.

### Part F — Distribution-shift or cross-lingual comparison (choose one)

Either:

- **F.1** Evaluate your SQuAD-1.1 model on SQuADShifts without any retraining. Report EM and F1 on `new_wiki`, `nyt`, `reddit`, and `amazon` subsets. Comment on the F1 drop and one hypothesis for its cause.
- **F.2** Evaluate on an XQuAD language other than English (with an XLM-R-based reader trained on English SQuAD). Report per-language F1. Comment on which language transfers best and why you think so.

### Part G — Write-up

A 400–600 word `README.md` covering:

- Dataset, encoder, and hyperparameters.
- Reported metrics tables from Part D.
- Failure category breakdown from Part E.
- Result of the Part F comparison, with one hypothesis about the drop and one thing you would try next.

## Starter guidance

- Use the HF `run_qa.py` example as a reference implementation, not as a copy-paste. Reading it once end-to-end is the best way to internalise the top-*k* decoder.
- The preprocess audit in Part A is not optional. Every extractive-QA bug I have ever debugged was either an alignment error or a top-*k* pruning error. Both are caught by the audit + Part E categorisation.
- If you use `roberta-base`, remember it has no `token_type_ids`. The HF tokeniser handles this — you do not need to manually strip the field, but do not rely on it downstream.
- Track truncation rate on the training set (`sequence_ids` fully consumed before context ends). If > 3 %, bump `max_seq_length` or `doc_stride`.
- `Trainer.evaluate()` will report the loss but not EM/F1 — you must plumb `compute_metrics` through a closure that has access to both the tokenised and raw dev sets.

## Acceptance criteria

- [ ] `preprocess_audit.py` prints 10 aligned examples and asserts the sliced substring equals the gold answer; exits non-zero if any fail.
- [ ] Training script (`train_qa.py`) reproduces the Part B recipe; run logs are saved.
- [ ] `postprocess.py` implements top-*k* decoding with the correct filters; 20 dev predictions manually verified to be substrings of the original context.
- [ ] `evaluate.load("squad")` overall EM and F1 reported with bootstrap CIs.
- [ ] Per-question-type, per-answer-length, and per-context-length F1 tables in the write-up.
- [ ] 20-example failure taxonomy with counts and one worked example per category.
- [ ] One distribution-shift or cross-lingual comparison completed (F.1 or F.2).
- [ ] 400–600 word write-up.

## Stretch goals

- **DeBERTa-v3-large.** Rerun with the large variant. Report EM/F1 delta and per-hour cost delta vs. `-base`.
- **Seed variance.** Train three seeds. Report mean ± std of dev F1.
- **BPE dropout.** Enable subword regularisation during training (via `tokenizer.enable_padding()` + a custom collator). Any generalisation gain?
- **Longer stride, shorter length.** Try `max_length = 256` with `stride = 128` and `max_length = 512` with `stride = 200`. Which combination has the best F1-per-training-hour trade-off?
- **Substitute the extractive-QA head with an SQuAD-style seq2seq template on FLAN-T5.** Report EM/F1 on the same dev set. Which model is better on `why` questions specifically?
