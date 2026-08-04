# Contrastive Objectives for Retrieval Embeddings

## Motivation

The Sentence-BERT paper's contribution was less "a new architecture" and more "a training loss that makes the architecture behave." Every modern retrieval embedding — E5, BGE, GTE, Nomic, `text-embedding-3` — is trained with some variant of a *contrastive* objective: pull together the vectors of things that should be similar, push apart everything else. Get the loss, the batch composition, and the temperature right and a 22 M-parameter MiniLM catches within a few MTEB points of a 1 B-parameter unaligned model. Get them wrong and even a 1 B model produces embeddings that look uniformly close to everything.

This chapter is the loss zoo — MultipleNegativesRankingLoss (MNR / InfoNCE), triplet loss, CoSENT, and the STS-style regression losses — plus the dials that actually matter (batch size, temperature, in-batch vs. mined negatives). Chapter 05 handles what "mined negatives" means and how to get them right.

## The one-picture intuition

Every contrastive loss on a training example says the same thing:

$$
\text{make } \text{sim}(q, d^+) \gg \text{sim}(q, d_1^-), \; \text{sim}(q, d_2^-), \; \dots, \; \text{sim}(q, d_K^-).
$$

- $(q, d^+)$ is a positive pair — a query and a relevant document, or a sentence and its paraphrase.
- $\{d_k^-\}_{k=1..K}$ are negatives — documents that are *not* relevant to $q$.

The differences between losses come down to (a) how the "$\gg$" is measured, (b) whether the negatives are per-query or shared across the batch, and (c) whether "relevance" is binary or graded.

Get the positives right (chapter 03: what pairs your data has and what "similar" means) and the negatives right (chapter 05: mining) and the choice of loss matters much less than internet arguments suggest.

## InfoNCE / MultipleNegativesRankingLoss — the default

The dominant loss in modern retrieval embedding training is **InfoNCE** (van den Oord, Li & Vinyals, ["Representation Learning with Contrastive Predictive Coding"](https://arxiv.org/abs/1807.03748), 2018), which `sentence-transformers` ships as `MultipleNegativesRankingLoss` (MNR).

Given a batch of $N$ positive pairs $\{(q_i, d_i^+)\}_{i=1..N}$, MNR treats the other $N-1$ documents in the batch as negatives *for free*. The loss is a categorical cross-entropy over the batch:

$$
\mathcal{L}_\text{MNR} \;=\; -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(q_i, d_j) / \tau)}.
$$

- $\tau$ is the *temperature* — a scalar that scales the logits before softmax. In MNR it defaults to `scale=20`, which is a *multiplier*: $\text{sim}(q, d) \cdot 20$. Equivalent to $\tau = 1/20 = 0.05$.
- Symmetric or one-directional. Set `similarity_fct=cos_sim` with the default `scale=20` and you get the recipe most sentence-transformers were trained with. `symmetric=True` averages the loss in both directions ($q \to d$ and $d \to q$) — matters when you have query-passage asymmetry.

The API:

```python
from sentence_transformers import losses
loss = losses.MultipleNegativesRankingLoss(model, scale=20.0, similarity_fct="cos_sim")
```

Two properties of MNR are essential to internalise:

- **Bigger batches → harder negatives → better recall.** Every additional example in the batch is another negative for every query. The famous SimCSE and DPR results were with batch sizes of 512–8192 — much larger than a classification task would ever need.
- **False negatives destroy training.** If two rows in your batch happen to be paraphrases, MNR punishes the model for matching them, even though they *should* match. In-batch negative training assumes batch composition is IID and mutually unrelated. Chapter 05 covers the mitigation (deduplication + hard-negative mining with cross-encoder denoising).

## MegaBatchMarginLoss and the "bigger batches are all you need" era

Between 2020 and 2022, retrieval-embedding papers converged on a common finding: bigger effective batch sizes almost always help, up to the memory ceiling. The techniques:

- **`MegaBatchMarginLoss`** — pretend the batch is bigger by accumulating multiple micro-batches of embeddings without gradient, then computing contrastive loss over the full mega-batch and back-propagating only through the current micro-batch.
- **Cached MultipleNegativesRankingLoss** (`CachedMultipleNegativesRankingLoss`) — computes the encoding twice, once with no-grad to get the full similarity matrix and once with grad to back-propagate. Enables effective batch sizes of tens of thousands on a single GPU.
- **GradCache** (Gao et al., ["Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup"](https://arxiv.org/abs/2101.06983), 2021) — the general technique behind the cached variant. Recompute embeddings during back-prop instead of storing them all.

For any serious retrieval training run in 2025, `CachedMultipleNegativesRankingLoss` is the default. `MultipleNegativesRankingLoss` is the pedagogical version.

## Triplet loss — an older default worth knowing

**Triplet loss** (Schroff, Kalenichenko & Philbin, ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832), *CVPR 2015*) predates InfoNCE by a few years and still shows up in domain-adaptation and margin-tuning contexts:

$$
\mathcal{L}_\text{triplet}(a, p, n) = \max\bigl(0, \; \text{sim}(a, n) - \text{sim}(a, p) + \text{margin}\bigr).
$$

Given an *anchor* $a$, a *positive* $p$, and a *negative* $n$, the loss is zero if the positive is more similar than the negative by at least `margin`; otherwise it is the margin violation.

`sentence-transformers` API:

```python
loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.3)
```

Where you would still reach for it:

- **When you have explicit triplets** — an anchor plus a chosen hard negative — and want to teach the model that *specific* comparison. Common in domain adaptation where you have hand-mined negatives.
- **When batch size is genuinely constrained** and in-batch negatives are not viable.
- **For metric learning tasks outside retrieval** — e.g., near-duplicate detection where the "margin" is the operating threshold.

Otherwise, MNR outperforms triplet in most retrieval settings by ~2–4 points, mostly because the many-negatives softmax provides a stronger training signal than the single-margin comparison.

## STS regression: CoSENT and CosineSimilarityLoss

Retrieval is not the only training objective. When your data has *graded* similarity — human-labelled STS pairs on a 0–5 scale, click-through rates, ordinal relevance — the right loss is not contrastive at all. It is a *regression* loss on similarity scores.

Two live losses:

- **`CosineSimilarityLoss`** — the simplest form. MSE between predicted cosine similarity and the label:
  $$
  \mathcal{L} = \left(\text{cos}(\phi(x_1), \phi(x_2)) - y\right)^2
  $$
  Fast, stable, and directly optimised for STS-B-style benchmarks. This is what the original Sentence-BERT paper used for STS.

- **`CoSENTLoss`** — Suvorov's ["CoSENT"](https://kexue.fm/archives/8847) formulation, a scale-invariant ranking loss over cosine similarities. Given a batch of pairs with labels, it penalises *inversions*: any pair with a higher label whose cosine similarity is lower than a pair with a lower label. Empirically outperforms `CosineSimilarityLoss` on STS with less sensitivity to the batch's absolute similarity distribution, and is the current `sentence-transformers` recommendation for graded STS-style data.

```python
loss = losses.CoSENTLoss(model)
```

Do not use STS regression losses on retrieval data. They optimise for *calibrated* similarity, not ranking, and produce visibly worse retrieval quality (chapter 08 shows why on MTEB).

## The temperature knob and what happens when you get it wrong

Temperature $\tau$ (equivalently `scale = 1/τ` in `sentence-transformers`) is the single scalar that controls how sharp the softmax is over the batch.

- **Low $\tau$ (high scale, e.g. 20 → 100).** Sharp softmax. Only the top-1 negative contributes to the loss. Fast convergence, sensitive to false negatives.
- **High $\tau$ (low scale, e.g. 1 → 5).** Smooth softmax. Many negatives contribute. Slower convergence, more robust to noise.

The `sentence-transformers` MNR default of `scale=20` (i.e. $\tau = 0.05$) is calibrated for cosine similarity, which lives in $[-1, 1]$. If you switch to dot product on unnormalised vectors, you have to re-tune — dot products can grow arbitrarily large and `scale=20` will produce numerically saturated logits.

The DPR paper (Karpukhin et al., 2020) used $\tau = 1$ over dot products of un-normalised vectors and got away with it because the vector magnitudes were bounded by BERT-base initialisation. E5, BGE, and every modern normalised model use scales in the 20–100 range on cosine.

Practical rule: for any custom training run, sweep three temperatures (e.g. `scale ∈ {10, 20, 50}`) on a small evaluation set. The best is usually the middle, and the difference is usually 1–3 STS points.

## Batch size and effective negatives

For MNR, the effective number of negatives per query is `batch_size - 1`. This makes batch size, not epochs, the primary quality knob.

Empirical shape:

| Batch size | Effective negatives per query | Recall@100 (rough shape) |
|------------|-------------------------------|--------------------------|
| 32         | 31                            | Baseline                 |
| 128        | 127                           | +2–5 points              |
| 512        | 511                           | +3–8 points              |
| 4096       | 4095                          | +5–12 points, diminishing |
| 32768      | 32767                         | Saturates                |

Numbers are illustrative — exact magnitudes vary by dataset and backbone. The shape is the point: `batch_size=32` on a small GPU is a *bad* MNR configuration. Reach for `CachedMultipleNegativesRankingLoss` or `GradCache` to unlock 1000+ effective negatives on a single 24 GB GPU.

## Symmetric vs. asymmetric objectives

Retrieval is inherently asymmetric — a query looks very different from a passage. Two losses handle this differently:

- **MNR (one-directional).** For each anchor $q_i$, treat all other batch documents as negatives. Only $q \to d$ direction; nothing punishes documents that look like queries.
- **MNR-symmetric** (`symmetric=True` on `CachedMultipleNegativesRankingLoss`) — averages $q \to d$ and $d \to q$ losses. Encourages a symmetric embedding space.

For asymmetric retrieval (queries are short questions, documents are long passages), one-directional MNR is usually fine and matches how E5 was trained. For symmetric tasks (STS, paraphrase, deduplication), symmetric MNR is a small lift.

## Curriculum and multi-stage: previewed

Every strong published embedding model is trained in *stages*, not a single pass:

1. **Weakly supervised pretraining** on a large, noisy pair corpus (e.g. Common Crawl QA pairs, Reddit `title/body` pairs, S2ORC paper title/abstract). Batch size 1024+, in-batch negatives only, 1 epoch.
2. **Supervised fine-tuning** on cleaner labelled pairs (MS MARCO, NQ, TriviaQA, GooAQ, HotpotQA). Batch size 128–1024, mixed in-batch and mined hard negatives.
3. **(Optional) Distillation** from a stronger cross-encoder (chapter 06). Batch size 256, KL divergence or MSE against cross-encoder scores.
4. **(Optional) Task-specific / domain-specific fine-tune.** Batch size 64–256, target-domain supervised pairs, small LR.

That is the E5 recipe (Wang et al., 2022), roughly the BGE recipe (Xiao et al., 2023), and the shape most in-house domain-adapted embeddings should follow. Chapter 06 covers the distillation stage in depth; this chapter's exercise builds stage 2 end-to-end.

## Diagnosing a broken contrastive run

When contrastive training is not working, the failure modes are stereotyped:

- **All cosines converge to ~1.0.** Model has collapsed into a constant embedding (representation collapse). Cause: temperature too high, no negatives, or LR too aggressive. Fix: lower `scale`, verify batch contains negatives, drop LR.
- **All cosines are ~0.** Negative-space collapse — the model pushed everything to be uniformly dissimilar. Cause: too many false negatives or too aggressive a temperature. Fix: dedupe the training set, re-inspect mined negatives (chapter 05).
- **Loss oscillates.** Batch has too many false negatives (e.g. two paraphrases in the same batch). Fix: content dedupe before batching.
- **Loss goes down but STS/MTEB does not move.** Loss is on the wrong objective for the eval — e.g. training on retrieval MNR but evaluating on STS with `CosineSimilarityLoss`-expected scores. Fix: evaluate on a metric aligned with the loss.
- **Loss plateaus after ~500 steps.** Model saturated on easy in-batch negatives. Fix: add mined hard negatives (chapter 05).

Every one of those diagnostics is a lot faster if you log the mean and std of positive and negative cosines separately, alongside the loss. Do that from step 1.

## Chapter summary

- Every modern retrieval embedding is trained contrastively — pull positives together, push everything else apart.
- `MultipleNegativesRankingLoss` (a.k.a. InfoNCE) is the default. Its loss is a softmax over in-batch negatives; batch size is the primary quality knob.
- `CachedMultipleNegativesRankingLoss` / GradCache unlock effective batch sizes of thousands on a single GPU; use them for any serious training run.
- Triplet loss is a valid alternative when you have hand-picked triplets or genuinely small batches, but MNR outperforms in most retrieval settings.
- For *graded* similarity labels (STS), use `CoSENTLoss` or `CosineSimilarityLoss` — regression, not contrastive. Do not use them on retrieval data.
- Temperature (`scale`) controls softmax sharpness; the sentence-transformers default of `scale=20` is calibrated for cosine similarity and needs re-tuning if you switch to dot product.
- Bigger batches help, up to saturation; the effective number of negatives is roughly `batch_size - 1`.
- Real training pipelines are staged: weak-supervised pretrain, supervised fine-tune, (optional) distillation, (optional) domain fine-tune. Chapter 06.
- Diagnose representation collapse and false-negative failure modes by logging positive and negative cosines separately from step 1.
