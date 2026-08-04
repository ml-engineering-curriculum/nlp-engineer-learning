# exercise-01: Baseline Bake-off — Logreg, fastText, and Encoder

**Estimated effort:** 3 hours

## Objective

Train three classifiers on the same dataset — a TF-IDF + logistic regression baseline, a fastText model, and a fine-tuned encoder — and produce an apples-to-apples comparison across accuracy, macro-F1, inference latency, and model size. Recommend which one you would ship, and defend the recommendation with numbers.

The point is not to prove that the biggest model wins; it is to build the habit of climbing the ladder from chapter 01 in order and comparing rungs honestly.

## Prerequisites

- Chapters [01](../01-why-text-classification-is-still-the-workhorse-task.md), [02](../02-tfidf-logistic-regression-and-linear-svm-baselines.md), [03](../03-fasttext-linear-classifier-with-subword-embeddings.md), [04](../04-fine-tuning-encoders-bert-roberta-deberta-xlmr.md).
- Python 3.10+; `scikit-learn`, `fasttext` (`pip install fasttext`), `transformers`, `datasets`, `evaluate`, `torch`.
- A GPU is strongly recommended for the encoder run but not required (use `distilbert-base-uncased` or `distilroberta-base` if you are CPU-only).

## Dataset

Use one of the following, or a comparable public dataset with ≥ 3 classes and ≥ 10 000 labelled documents:

- **AG News** (Zhang, Zhao & LeCun, ["Character-level Convolutional Networks for Text Classification"](https://arxiv.org/abs/1509.01626), *NeurIPS 2015*) — 4 balanced news classes, 120 000 train / 7 600 test. Load via `datasets.load_dataset("ag_news")`.
- **DBpedia-14** — 14 balanced Wikipedia categories. `datasets.load_dataset("dbpedia_14")`.
- **Yelp Polarity** — binary sentiment. `datasets.load_dataset("yelp_polarity")`.
- **20 Newsgroups** — 20 classes, ~18 000 documents; more imbalanced and older but a common academic reference. `sklearn.datasets.fetch_20newsgroups`.

Pick one and stick with it for all three models. Do not switch datasets between models.

## Problem statement

### Part A — TF-IDF + logistic regression baseline

Build a scikit-learn `Pipeline` with `TfidfVectorizer` and `LogisticRegression`. Requirements:

- `TfidfVectorizer` with `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`.
- `LogisticRegression(solver="saga", class_weight="balanced")`.
- Sweep `C ∈ {0.1, 1.0, 10.0}` on a validation split (carve out from the training set) and pick the best by macro-F1.
- Report training time, per-class F1, macro-F1, weighted-F1, accuracy on the test set.
- Report inference throughput on the test set as documents/sec (batch predict, single thread).
- Persist the fitted pipeline with `joblib.dump`. Record disk size in MB.

### Part B — fastText

Format the dataset in fastText's `__label__<class> <text>` line format. Train a supervised model with these defaults:

```python
fasttext.train_supervised(
    input="train.txt",
    epoch=25, lr=0.5,
    wordNgrams=2, minn=3, maxn=6,
    dim=100, loss="softmax", thread=8,
)
```

Then:

- Sweep `lr ∈ {0.1, 0.5, 1.0}` and pick the best by macro-F1 on the validation set.
- Report the same metrics as Part A: per-class F1, macro-F1, weighted-F1, accuracy, training time, inference throughput.
- Save the model uncompressed (`.bin`) and quantised (`model.quantize(...)`, then `.ftz`). Report both file sizes and any accuracy delta from quantisation.

### Part C — Fine-tuned encoder

Fine-tune one of `distilroberta-base`, `roberta-base`, or `microsoft/deberta-v3-base` with `transformers.Trainer`. Requirements:

- Follow the "minimum viable fine-tune" recipe from chapter 04: `lr=2e-5`, `num_train_epochs=3`, `warmup_ratio=0.06`, `weight_decay=0.01`, mixed precision if available.
- Track validation macro-F1 per epoch; `load_best_model_at_end=True`, `metric_for_best_model="f1"`.
- Report at least **three seeds** and the mean ± std of macro-F1 (chapter 04 explains why).
- Report training time (total wall-clock across seeds), test macro-F1, per-class F1, accuracy.
- Report inference throughput on the test set at `batch_size=32` on both GPU and CPU (if available). Record the disk footprint of the saved model directory.

### Part D — The bake-off table

Produce a single Markdown table with rows for each of the three models and columns:

| Model | Macro-F1 | Weighted-F1 | Accuracy | Train time | Inference throughput (docs/sec) | Model size on disk |

Fill in every cell. Include mean ± std for the encoder row (from Part C's seed variance). Do not omit numbers because they are "obvious" — this table is the deliverable.

### Part E — The write-up

A 300–500 word `README.md` covering:

- Which model you would ship for this dataset, and why.
- What the deployment constraints would need to be for a different model to win instead.
- Which numbers surprised you (e.g., logreg within noise of fine-tuning on this dataset, or fastText being several times faster than expected).
- One specific improvement each model could get with more effort (better preprocessing for logreg, more epochs for fastText, hyperparameter sweep for the encoder).

## Starter guidance

- Use the same train/val/test split across all three models. If the dataset has only train/test, carve a 10 % validation set from train with a fixed seed and reuse it for every model's hyperparameter selection.
- Measure inference latency after warm-up (throw away the first 100 predictions before you start timing).
- For fair comparison, measure inference throughput on the same hardware for all three. If you must compare CPU-only logreg against GPU encoder, report both hardware settings and be explicit about what you are comparing.
- Do not tokenise the encoder inputs on-the-fly during inference timing — that penalises the encoder unfairly. Pre-tokenise, then time.
- Save class-weighted encoder training for exercise-03 if it feels tempting. Keep this bake-off vanilla.

## Acceptance criteria

- [ ] Three trained models: TF-IDF + logistic regression pipeline (`joblib`), fastText `.bin` and `.ftz`, fine-tuned encoder directory.
- [ ] Reproducible training scripts for each (`train_logreg.py`, `train_fasttext.py`, `train_encoder.py`).
- [ ] Bake-off table with all cells filled, including mean ± std over ≥ 3 seeds for the encoder.
- [ ] Per-class F1 reported for all three models — not just macro.
- [ ] Inference throughput reported on the same hardware for the CPU-friendly models, and on both CPU and GPU for the encoder.
- [ ] A short written recommendation (300–500 words) with the "which one to ship" call, explicit trade-offs, and one improvement per model.

## Stretch goals

- **Add a linear SVM** (`LinearSVC` with `CalibratedClassifierCV`) to the bake-off table. Does it beat logistic regression on your dataset?
- **Character-level fastText**. Set `minn=2, maxn=5` and disable word n-grams (`wordNgrams=1`). Does character-only improve on morphology-heavy inputs like reviews with typos or slang?
- **DistilBERT vs. RoBERTa vs. DeBERTa-v3.** Fine-tune all three at `base` size, compare macro-F1 vs. inference latency. Which is the Pareto frontier point for your use case?
- **Confusion-matrix analysis.** Pick the two worst-performing classes in the encoder's confusion matrix. Investigate a sample of misclassified examples. Are they labelling errors, taxonomy overlap, or genuine ambiguity?
- **Cost-weighted comparison.** Invent a plausible cost function for your dataset (e.g., a 10× higher cost for false positives on a specific class). Rerun the bake-off ranking under that cost. Does the winner change?
