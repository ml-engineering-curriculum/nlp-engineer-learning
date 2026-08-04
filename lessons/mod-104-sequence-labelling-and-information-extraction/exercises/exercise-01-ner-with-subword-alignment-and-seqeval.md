# exercise-01: NER with Subword Alignment and seqeval

**Estimated effort:** 3 hours

## Objective

Train a transformer NER model end-to-end, prove that your subword-to-word alignment is correct, evaluate with `seqeval` under the strict CoNLL protocol, and report the results honestly with per-type F1 and seed variance. This exercise is the "workhorse" recipe you will reuse in every subsequent NER project — get it right once, paste it forever.

## Prerequisites

- Chapters [01](../01-why-sequence-labelling-and-information-extraction-still-matter.md), [02](../02-tagging-schemes-and-entity-level-evaluation-with-seqeval.md), [03](../03-subword-to-word-alignment-for-transformer-ner.md), [04](../04-token-classification-ner-the-workhorse-recipe.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `seqeval`, `torch`.
- A GPU strongly recommended for the `-base` encoder run; a CPU + `distil*` model is workable.

## Dataset

Pick one of the following BIO-tagged NER datasets. Use the same dataset for all parts.

- **CoNLL-2003 English** (Tjong Kim Sang & De Meulder, ["Introduction to the CoNLL-2003 Shared Task"](https://aclanthology.org/W03-0419/), *CoNLL 2003*). `datasets.load_dataset("conll2003")` — PER / ORG / LOC / MISC.
- **WNUT-17** (Derczynski et al., ["Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition"](https://aclanthology.org/W17-4418/), *WNUT 2017*). `datasets.load_dataset("wnut_17")` — social-media NER with rare entity types.
- **OntoNotes v5 (English)** via `datasets.load_dataset("tner/ontonotes5")` — 18 types, more realistic scale.

## Problem statement

### Part A — Alignment audit

Before training, print 20 aligned examples in a human-readable format:

```
[CLS]   ->  -100
Barack  ->  B-PER
##  Obama -> I-PER (was word 'Obama')
##  ...continuation subwords -> -100
```

Verify by hand that:

- Every `[CLS]` / `[SEP]` / `[PAD]` is `-100`.
- Every first-subword of an entity has the correct BIO tag.
- No continuation subword has a `B-*` or `I-*` tag (variant A) — or if you chose variant C from chapter 03, that every continuation subword has the correct `I-*` (never `B-*`).
- Round-trip: re-emit word-level tags from the aligned sequence and compare against the original `ner_tags`. 100 % match required on this sample.

Ship the audit script as `alignment_audit.py`.

### Part B — Train the workhorse model

Fine-tune `microsoft/deberta-v3-base` (or `roberta-base`, or `xlm-roberta-base` if you want the multilingual path) with the recipe from chapter 04:

- `AutoModelForTokenClassification`
- `DataCollatorForTokenClassification`
- LR `3e-5`, effective batch `16–32`, `num_train_epochs=3`, `warmup_ratio=0.06`, `weight_decay=0.01`, `fp16=True` if you have Ampere+.
- `load_best_model_at_end=True`, `metric_for_best_model="f1"`.

Use the first-subword-only alignment (variant A) from chapter 03. `-100` on continuations.

### Part C — Evaluate with strict seqeval

Compute test-set precision, recall, F1 with `seqeval` in **`mode="strict"`** and the correct `scheme` for your dataset:

- CoNLL-2003 → `scheme="IOB2"`.
- WNUT-17 → `scheme="IOB2"`.
- OntoNotes → `scheme="IOB2"`.

Report:

- **Overall precision / recall / F1** (micro).
- **Per-type F1** as a Markdown table with columns: `Type`, `Support`, `Precision`, `Recall`, `F1`.
- **Macro-F1** across types.

Also compute and report the **lenient-mode** F1 (`mode="default"`). Note the delta and explain in one sentence why lenient mode is higher.

### Part D — Seed variance

Repeat the training run with **at least three different seeds** (e.g., 42, 123, 2024). Report:

- Mean ± std of overall F1.
- Mean per-type F1.
- Which seed produced the best model.

If your best-vs-mean gap exceeds 1 F1, you are over-fitting to seed variance; call this out in the write-up.

### Part E — Inference and BIO repair

Load your best model and run inference on a held-out set (test split). For each document:

- Emit character-level entity spans via `pipeline("ner", aggregation_strategy="first")`.
- Compare against a **manual** decode path where you apply BIO repair (chapter 05): `O → I-X` becomes `O → B-X`; `I-X → I-Y` (`X ≠ Y`) becomes `I-X → B-Y`.

Report:

- The fraction of test predictions where the two decoding paths disagreed.
- Whether disagreement affected `seqeval` F1 (should be ≤ 0.1 F1).

### Part F — Write-up

A 300–500 word `README.md` covering:

- Which dataset, which encoder, which alignment variant.
- Reported metrics table (from Part C).
- Seed variance summary (from Part D).
- One concrete failure case: pick a false negative and a false positive from the confusion matrix, quote the original sentence, and explain why the model got it wrong (annotation ambiguity? tokenisation artefact? domain gap?).
- One thing you would try next.

## Starter guidance

- The `compute_metrics` snippet in chapter 02 is the canonical scaffold — start from there and adapt to your dataset's `label_names`.
- Do **not** skip the Part A audit. Alignment bugs pass training and often pass evaluation until they hit downstream.
- Use `Trainer(...).save_model(...)` at the end; save `id2label` and `label2id` so the `pipeline` API loads without warnings.
- Track truncation rate on the training set; if > 5 %, note it in the write-up and consider bumping `max_length`.
- For seed variance, launch three `Trainer.train()` calls with different `TrainingArguments(seed=...)`, `torch.manual_seed`, and `numpy.random.seed`. Do not reuse the same `output_dir` — checkpoint collisions will lose the best model.

## Acceptance criteria

- [ ] `alignment_audit.py` prints 20 aligned examples with subword → label mapping and passes the four checks in Part A.
- [ ] Training script (`train_ner.py`) reproduces the Part B recipe; run logs are saved.
- [ ] `seqeval` `mode="strict"` micro-F1 reported alongside per-type F1 (Markdown table) and macro-F1.
- [ ] Lenient-mode F1 reported for comparison, with one-sentence explanation of the delta.
- [ ] Three-seed mean ± std reported; best-seed model checkpoint retained.
- [ ] Inference script exercises both `pipeline("ner", aggregation_strategy="first")` and manual BIO-repair decoding; disagreement rate reported.
- [ ] 300–500 word write-up with metrics, a false-positive / false-negative analysis, and a "next step" idea.

## Stretch goals

- **Domain encoder.** Replace `deberta-v3-base` with a domain-adapted encoder (`BioClinicalBERT`, `SciBERT`, `LEGAL-BERT`) if your dataset has a matching domain. Rerun seed variance. Does the domain encoder help enough to justify the smaller pretraining corpus?
- **Alignment variant C.** Re-implement with the `B-X → I-X` continuation-subword conversion from chapter 03. Compare F1 to variant A. Any consistent gain?
- **CRF head.** Add a `pytorch-crf` layer (chapter 05). Compare F1 and per-token inference latency. Is the gain worth the cost on your dataset?
- **Uncased vs. cased.** Repeat with `bert-base-uncased`. How much does casing carry the signal for PER / ORG / LOC on your corpus?
- **Per-type learning curve.** Subsample training data at {10 %, 25 %, 50 %, 100 %} and plot per-type F1. Which types learn fastest? Which need the most data?
