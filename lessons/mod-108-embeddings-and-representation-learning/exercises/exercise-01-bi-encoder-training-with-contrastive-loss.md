# exercise-01: Bi-Encoder Training With Contrastive Loss

**Estimated effort:** 3 hours

## Objective

Fine-tune a sentence-transformers bi-encoder on a real query-passage dataset with `MultipleNegativesRankingLoss` (in-batch negatives / InfoNCE), then evaluate it against the untrained baseline on a held-out retrieval slice and on STS. This is the workhorse recipe from chapter 04 — every downstream exercise assumes you can reproduce it from scratch.

## Prerequisites

- Chapters [02](../02-bi-encoders-vs-cross-encoders.md), [03](../03-sentence-transformers-pooling-and-similarity.md), [04](../04-contrastive-objectives-for-retrieval-embeddings.md).
- Python 3.10+; `sentence-transformers>=3.0`, `datasets`, `torch`, `mteb`, `evaluate`, `scipy`, `numpy`.
- A GPU is strongly recommended. The recipe below fits comfortably on a single 16–24 GB card; CPU works but takes ~10× longer.

## Model and dataset

Pick one setup. The first is the recommended default for the acceptance criteria; the second is a stretch alternative.

- **Recommended.** Backbone `sentence-transformers/all-MiniLM-L6-v2` (22 M params, 384-dim, cosine-normalised). Training data: `sentence-transformers/msmarco-msmarco-distilbert-base-tas-b` triplets, or the simpler `sentence-transformers/gooaq` pair dataset (`load_dataset("sentence-transformers/gooaq")`). Both are `(anchor, positive)` pair datasets ready for MNR.
- **Alternative.** Backbone `microsoft/mpnet-base` warm-started (no sentence-transformers head yet — you will add mean pooling per chapter 03) with the same data.

Evaluate on the MTEB `NFCorpus` retrieval task (small, ~3 k docs, runs in minutes) plus `STSBenchmark` for a symmetric-similarity check.

## Problem statement

### Part A — Load and inspect the training data

1. Load a pair dataset — `gooaq` is the recommended default: `load_dataset("sentence-transformers/gooaq", split="train")`. It exposes `question` and `answer` columns.
2. Subsample to 100 000 pairs (`ds.shuffle(seed=42).select(range(100_000))`) so the exercise fits an afternoon.
3. Report: total pair count, mean/median token length per column (using the base model's tokenizer), and 10 random `(anchor, positive)` samples.
4. Deduplicate on both `(anchor, positive)` and on `positive` alone. Report before/after counts. Chapter 04 explains why exact-duplicate positives destroy MNR training.

Save inspection output to `data_stats.md`.

### Part B — Baseline evaluation

Before touching the model, evaluate `all-MiniLM-L6-v2` untouched on the two eval tasks:

- **NFCorpus** via `mteb.get_tasks(tasks=["NFCorpus"])`. Report nDCG@10, Recall@100.
- **STSBenchmark** via `mteb.get_tasks(tasks=["STSBenchmark"])`. Report Spearman ρ on cosine.

Wrap the model in an `mteb.Encoder` that applies whatever prefixes the backbone requires (MiniLM needs none; if you swap to E5 or BGE this is where the prefixes go — chapter 08). Cache the results to `baseline_report.md`.

### Part C — Train with MultipleNegativesRankingLoss

Fine-tune the bi-encoder with the chapter-04 recipe. Minimum requirements:

- `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` — pooling and normalisation come from the bundle; do not override.
- `losses.MultipleNegativesRankingLoss(model, scale=20.0, similarity_fct=util.cos_sim)`.
- Batch size `128` (or the largest you fit; log the effective negatives per query = `batch_size - 1`).
- LR `2e-5`, `warmup_ratio=0.1`, `weight_decay=0.01`, 1 epoch.
- `SentenceTransformerTrainer` (v3 API) with `SentenceTransformerTrainingArguments`; `fp16=True` or `bf16=True`.
- Log positive-cosine mean, negative-cosine mean, and the gap separately every 200 steps (per chapter 04's diagnostics section).

Save training logs and the final checkpoint. Wrap the whole pipeline in `train.py`.

### Part D — Cached MNR for a bigger effective batch

Repeat Part C but replace the loss with `losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32, scale=20.0)` and set the trainer batch size to `1024`. This uses GradCache (chapter 04) to give you 1023 effective negatives per query on the same GPU.

Compare the two runs' training curves (positive-cos, negative-cos, gap, loss) and their downstream NFCorpus + STS scores. Chapter 04 predicts a 2–5 point retrieval lift from bigger effective batches — report whether you observe it.

### Part E — Temperature sweep

Fix the recipe from Part D. Sweep `scale ∈ {5, 20, 50, 100}` (equivalently `τ ∈ {0.2, 0.05, 0.02, 0.01}`) and re-train each. Report NFCorpus nDCG@10 and STSBenchmark Spearman ρ for each. Chapter 04's prediction: the middle values win; extremes either underfit or saturate.

Present as a table in `report.md`.

### Part F — Diagnose one failure

Deliberately break the run in one of the ways chapter 04 lists (representation collapse or false-negative poisoning). Pick one:

- Set `scale=200` and LR `5e-4`. Observe representation collapse (positive-cos and negative-cos both approach 1.0).
- Do not deduplicate the training data; deliberately inject 500 near-duplicate positive pairs by copying random rows. Observe loss oscillation from false negatives colliding in-batch.

Log the failure signature and reproduce the diagnostic curves. This is a one-paragraph write-up in `report.md`.

### Part G — Write-up

400–600 word `report.md` covering:

- Dataset stats (Part A), including dedup counts.
- Baseline vs. trained metric table (NFCorpus + STS), Parts C and D.
- Temperature sweep table (Part E).
- Failure-mode reproduction (Part F).
- One sentence on what you would try next (bigger batch, different backbone, hard negatives — the last is exercise-02).

## Starter guidance

- **Use the v3 `SentenceTransformerTrainer`.** The older `model.fit(...)` API still works but the trainer gives you HF-style logging and eval hooks for free. Chapter 04.
- **Log positive- and negative-cosine separately.** `MultipleNegativesRankingLoss` is not `nn.CrossEntropyLoss` — its loss value alone hides both collapse modes. Chapter 04's diagnostics section.
- **`scale=20` is a multiplier on cosine, not a temperature.** `τ = 1 / scale`. If you switch backbones to something un-normalised, retune.
- **`CachedMultipleNegativesRankingLoss` needs `mini_batch_size` set.** Rule of thumb: `mini_batch_size` = whatever plain MNR fits on your GPU; the trainer batch size is what MNR-with-cache pretends to have.
- **NFCorpus is small on purpose.** For iteration speed, use it. Do *not* extrapolate a leaderboard from one small task; that is what exercise-03 is for.
- **`mteb.Encoder.encode` signature.** MTEB v1.10+ passes `task_name` and `prompt_name`. Even if MiniLM does not care, wire them through — you will need it for E5/BGE later.
- **Do not train past 1 epoch on this dataset at this scale.** Signs of memorisation on pair datasets are subtle; the exercise's numbers are calibrated for 1 epoch.

## Acceptance criteria

- [ ] `data_stats.md` reports pair counts, length distributions, and before/after dedup counts.
- [ ] `baseline_report.md` has NFCorpus nDCG@10, Recall@100, and STSBenchmark Spearman ρ for the untrained backbone.
- [ ] `train.py` fine-tunes with `MultipleNegativesRankingLoss` per Part C; logs positive-cos, negative-cos, gap, and loss every 200 steps.
- [ ] `train_cached.py` fine-tunes with `CachedMultipleNegativesRankingLoss` at effective batch 1024 per Part D.
- [ ] Comparison table of baseline vs. MNR vs. Cached-MNR on NFCorpus + STSBenchmark.
- [ ] Temperature sweep table over `scale ∈ {5, 20, 50, 100}` on the Cached-MNR run.
- [ ] Documented reproduction of one collapse or false-negative failure mode with the diagnostic curves.
- [ ] `report.md` (400–600 words) covers all of the above.

## Stretch goals

- **Backbone comparison.** Repeat Part D with `microsoft/mpnet-base` (bare backbone, add mean pooling + normalisation per chapter 03). Report the quality delta vs. `all-MiniLM-L6-v2` at matched compute.
- **Symmetric MNR.** Set `symmetric=True` on `CachedMultipleNegativesRankingLoss` and compare on STSBenchmark. Chapter 04 predicts a small symmetric lift.
- **Triplet loss head-to-head.** Reformat the top 20 % of pairs into explicit triplets (mine one random negative per anchor) and train with `TripletLoss(margin=0.3)`. Report the retrieval + STS delta vs. MNR at matched steps.
- **CoSENT on STS.** Fine-tune on the `sentence-transformers/stsb` dataset with `CoSENTLoss` for 3 epochs. Report STSBenchmark Spearman vs. the MNR-trained model — chapter 04 warns not to mix the losses across data types.
- **Push effective batch to 8192** with `CachedMultipleNegativesRankingLoss(mini_batch_size=64)` on the same GPU. Report the training-time cost and whether NFCorpus improves further past 1024.

## Deliverables

Ship as a directory with:

```
data_stats.md
baseline.py
baseline_report.md
train.py
train_cached.py
temperature_sweep.py
failure_repro.py
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
