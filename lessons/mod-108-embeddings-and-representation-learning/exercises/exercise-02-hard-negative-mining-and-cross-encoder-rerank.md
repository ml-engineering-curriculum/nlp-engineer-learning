# exercise-02: Hard-Negative Mining and Cross-Encoder Rerank

**Estimated effort:** 3 hours

## Objective

Build the modern two-stage retrieval training + serving pipeline end-to-end: mine hard negatives with a bi-encoder, denoise them with a cross-encoder, fine-tune the bi-encoder against the denoised negatives, and evaluate the combined `bi-encoder → cross-encoder` reranking pipeline against a bi-encoder-only baseline. This is the chapter-05 and chapter-07 recipe made concrete.

## Prerequisites

- Exercise-01 (a trained bi-encoder you can re-use, or a public MNR-trained model).
- Chapters [05](../05-hard-negative-mining-and-curriculum-training.md), [06](../06-curriculum-training-and-distillation.md), [07](../07-cross-encoder-reranking-and-colbert.md).
- Python 3.10+; `sentence-transformers>=3.0`, `datasets`, `torch`, `rank_bm25` (or `pyserini` for stretch), `mteb`, `faiss-cpu` or `faiss-gpu`, `numpy`.
- A GPU is strongly recommended.

## Model and dataset

- **Base bi-encoder.** Your exercise-01 checkpoint, or `sentence-transformers/all-MiniLM-L6-v2` as a fallback.
- **Cross-encoder for denoising and reranking.** `cross-encoder/ms-marco-MiniLM-L-6-v2` (cheap; ~5 ms/pair on GPU) *and* `BAAI/bge-reranker-base` (stronger; ~30 ms/pair). Compare both in Part E.
- **Training data.** MS MARCO passage-ranking triples (`sentence-transformers/msmarco-hard-negatives`) or reuse the `gooaq` pairs from exercise-01. The exercise assumes MS MARCO throughout; adjust if you swap.
- **Evaluation.** MTEB `NFCorpus` + `SciFact` (both small BEIR-style retrieval tasks) for the end-to-end two-stage pipeline. Report nDCG@10, Recall@100, and MRR@10.

## Problem statement

### Part A — Baseline: bi-encoder only

Establish the ceiling of a bi-encoder-only pipeline. On NFCorpus and SciFact, report:

- Recall@100 (this is the *ceiling* your reranker can operate against — chapter 07).
- nDCG@10.
- MRR@10.

Wrap with `mteb.Encoder` as in exercise-01 and cache to `baseline_biencoder.md`.

### Part B — Mine BM25 negatives

Implement the cheap baseline first. For each `(query, positive)` pair in a 20 000-pair training subset of MS MARCO:

1. Build a BM25 index over the MS MARCO passage collection (or over a 200 k random subset if the full 8.8 M is too large — record which). Use `rank_bm25.BM25Okapi` for the exercise; note in the write-up why you would use `pyserini` in production.
2. For each query, retrieve the top-100 BM25 candidates. Remove any that overlap the labelled positive (chapter 05 warns against pre-filtering positives before scoring).
3. Keep the top-7 remaining as hard negatives per query.
4. Sample 20 mined `(query, negative)` pairs and human-label them (yourself) as **true hard negative** / **false negative** / **trivial semantic negative**. Report the mix. Chapter 05's five negative types is your rubric.

Save the mined dataset as `hard_negs_bm25.jsonl` and the audit as `bm25_audit.md`.

### Part C — Mine bi-encoder negatives with `mine_hard_negatives`

Use the sentence-transformers first-class helper (chapter 05):

```python
from sentence_transformers.util import mine_hard_negatives

mined = mine_hard_negatives(
    dataset,
    model=bi_encoder,
    num_negatives=7,
    range_min=10,
    range_max=100,
    margin=0.0,
    use_faiss=True,
)
```

Repeat the 20-pair human-label audit from Part B on this set. Chapter 05 predicts: bi-encoder-mined negatives are *harder* than BM25 but also carry *more* false negatives when the bi-encoder is already decent.

Save as `hard_negs_biencoder.jsonl` and `biencoder_audit.md`.

### Part D — Cross-encoder denoising (RocketQA-style)

Add the denoise step from chapter 05. For each `(query, mined-negative)` pair from Part C:

1. Score it with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
2. Also score `(query, labelled-positive)`.
3. Drop mined negatives whose cross-encoder score is *within* `margin=0.1` of, or *above*, the positive's score — those are likely false negatives.
4. Keep the top-7 lowest-scoring survivors per query.

Repeat the human-label audit on 20 random denoised pairs. Chapter 05 predicts <5 % false-negative rate after denoising; report whether you observe it.

Save as `hard_negs_denoised.jsonl` and `denoise_audit.md`.

### Part E — Retrain the bi-encoder on hard negatives

For each of the three mined sets (BM25, bi-encoder, denoised) train a fresh bi-encoder from the Part A baseline:

- `MultipleNegativesRankingLoss` — the hard negative for each anchor becomes the "explicit" negative in the input triple `(anchor, positive, hard_negative)`. In-batch negatives are still active for free.
- Batch size 64 (larger if it fits), 1 epoch, LR `2e-5`, `warmup_ratio=0.1`.

Evaluate each retrained bi-encoder on NFCorpus + SciFact. Report nDCG@10, Recall@100, MRR@10 in a comparison table.

Chapter 05's prediction shape:
- Baseline (no hard negs) → BM25 mined → biencoder mined → cross-encoder denoised should be roughly monotonic, with denoising giving the largest lift.

### Part F — Two-stage pipeline: bi-encoder → cross-encoder rerank

Take the *denoised-negatives* bi-encoder from Part E. For each eval query:

1. Retrieve top-100 candidates from the bi-encoder.
2. Rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
3. Rerank the same top-100 with `BAAI/bge-reranker-base`.
4. Rerank with a variable `|C| ∈ {10, 20, 50, 100}` and record end-to-end wall-clock latency (query encoding + top-K retrieval + reranking) for each `|C|` on your hardware.

Report:

- nDCG@10 and MRR@10 for bi-encoder alone vs. `+ MiniLM rerank` vs. `+ bge-reranker-base rerank`, at `|C|=100`.
- nDCG@10 vs. wall-clock table for `|C| ∈ {10, 20, 50, 100}` (bge-reranker-base only).
- The **oracle-reranker** number for `|C|=100` (chapter 07): given the bi-encoder's top-100, what nDCG@10 would you get if the reranker were perfect? This is the ceiling of any reranker on this candidate list.

### Part G — Write-up

500–700 word `report.md` covering:

- Data and audit of the three mined sets (Parts B, C, D). Include the false-negative rate before and after denoising.
- Retrieval-quality table across the four bi-encoders (baseline + three mining sources), Part E.
- Two-stage pipeline table and the latency-vs-quality trade-off, Part F.
- Where the oracle-reranker headroom sits and what closes it: better reranker, more candidates, or a stronger first stage.
- One "what next" — likely candidates: distillation from the cross-encoder into the bi-encoder (chapter 06), or ColBERTv2 as a middle-road single-stage alternative (chapter 07).

## Starter guidance

- **Do the human-label audits.** The single cheapest bug catch in this whole module is a 5-minute audit of 20 mined negatives. Skipping it is how false-negative poisoning gets into production. Chapter 05.
- **`range_min=10` in `mine_hard_negatives` skips the top-10.** Those are the most likely false negatives, especially against a decent bi-encoder. Do not lower it below 5 without a denoise pass.
- **Cache cross-encoder scores.** The denoise pass is `queries × 100 × forward_pass`. Batch aggressively (`predict(pairs, batch_size=64, show_progress_bar=True)`) and persist to disk — if you need to redo the training run you do not want to rescore.
- **Match the training-negative distribution to the serving-first-stage.** Chapter 07 is emphatic on this. If you rerank a bi-encoder's top-100, the cross-encoder training should be on bi-encoder-top-100 candidates, not BM25 candidates.
- **Batch the reranker.** `bge-reranker-base` at ~30 ms/pair unbatched is 3 s per query at `|C|=100`. Batched at 32 pairs on GPU it drops to ~500 ms total. Chapter 07's latency-budget equation.
- **Wall-clock latency includes bi-encoder query encoding, ANN search, *and* reranking.** Report all three separately.
- **NFCorpus is small (3 633 documents).** Do not use it as a serving-latency benchmark; use it for quality only. For latency, either measure on your reranker in isolation with a fixed synthetic candidate list, or use SciFact's larger corpus.

## Acceptance criteria

- [ ] `baseline_biencoder.md` reports Recall@100, nDCG@10, MRR@10 on NFCorpus + SciFact for the untrained-for-hard-negatives bi-encoder.
- [ ] `hard_negs_bm25.jsonl`, `hard_negs_biencoder.jsonl`, `hard_negs_denoised.jsonl` are the three mined datasets.
- [ ] `bm25_audit.md`, `biencoder_audit.md`, `denoise_audit.md` each report a hand-labelled 20-pair mix by negative type.
- [ ] Retrained bi-encoders (one per mining source) exist as checkpoints; the comparison table in `report.md` shows quality across all four.
- [ ] Two-stage pipeline evaluation with MiniLM and bge-reranker-base rerankers on the denoised-negatives bi-encoder.
- [ ] Wall-clock latency table for `|C| ∈ {10, 20, 50, 100}`; oracle-reranker upper bound for `|C|=100`.
- [ ] `report.md` (500–700 words) covers audits, quality table, latency-quality trade-off, and one "what next".

## Stretch goals

- **Iterative re-mining (ANCE-style).** After Part E, re-mine hard negatives with the *retrained* bi-encoder, retrain, and repeat one more time. Chapter 05 predicts monotone quality gains for 2–3 iterations before false-negative saturation. Report when you see it saturate.
- **GPL on an unlabelled corpus.** Take Wikipedia paragraphs (or any label-free corpus of ~10 k passages), synthesise queries with `doc2query/msmarco-t5-base-v1`, run the full chapter-05 GPL loop, and evaluate on the same NFCorpus + SciFact panel. Chapter 09 uses this recipe end-to-end.
- **monoT5 as a reranker.** Add `castorini/monoT5-base-msmarco` to Part F's comparison. Report the quality/latency trade-off vs. `bge-reranker-base`. Chapter 07.
- **Cross-encoder training.** Instead of using an off-the-shelf reranker, fine-tune `cross-encoder/ms-marco-MiniLM-L-6-v2` yourself on the denoised negatives with `BCEWithLogitsLoss` per chapter 07. Evaluate the rerank stage before and after.
- **ColBERTv2 comparison.** Load `colbert-ir/colbertv2.0` via `RAGatouille` and score the same eval sets. Chapter 07 discusses when this middle road wins.

## Deliverables

Ship as a directory with:

```
baseline_biencoder.py + baseline_biencoder.md
mine_bm25.py         + bm25_audit.md
mine_biencoder.py    + biencoder_audit.md
denoise_ce.py        + denoise_audit.md
retrain.py                                # loops over the three mined sets
two_stage_eval.py    + latency_table.md
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
