# Hard-Negative Mining and Curriculum Training

## Motivation

If chapter 04 has one weakness, it is this: in-batch negatives are, on average, *easy*. When you throw 256 random passages into a batch, most of them are irrelevant enough that even an untrained encoder can rank the positive above them. The gradient signal from those "obvious" negatives dries up after a few hundred steps, and training plateaus — despite the model still being visibly bad on any query where the relevant passage is competing against topically similar ones.

The fix is *hard-negative mining*: find negatives that are close enough to the positive to be confusing, and train against those. Done well, hard negatives are worth a factor of 2–5× on retrieval quality; done badly (mining actual false negatives and training against them) they will silently degrade the model. This chapter is the mining playbook plus the curriculum that stitches multiple mining passes into a training loop.

## What "hard" means, precisely

A **hard negative** for a query $q$ with positive $d^+$ is a document $d^-$ that:

1. Is *not* relevant to $q$ under the ground-truth labelling, but
2. Is high on some *scoring function* — BM25, a warm-started bi-encoder, or a cross-encoder — such that a naive ranker would place it near or above $d^+$.

Hard is defined *by the scoring function*. A negative that BM25 ranks in position 3 for the query is "hard" for a BM25-based retriever. A negative that a fine-tuned E5 ranks at position 5 is "hard" for E5 but might have been position 500 for BM25. This is why mining is an *iterative* process — as your model improves, the definition of "hard" moves.

The **false negative** trap: a candidate that a scoring function ranks high but that is actually *also relevant* to $q$, just not labelled. Training against a real false negative teaches the model that a genuinely correct answer is wrong. On MS MARCO, papers estimate 30–50 % of BM25's top-100 negatives are actually plausible answers that annotators missed (Qu et al., ["RocketQA: An Optimized Training Approach to Dense Passage Retrieval"](https://arxiv.org/abs/2010.08191), *NAACL 2021*). Every hard-negative mining pipeline has to also have a *denoising* step. Section on that below.

## The mining hierarchy

Three mining sources you will use, roughly in order of increasing "hardness" and cost:

1. **BM25 hard negatives.** For each query, run BM25 against the corpus, take the top-K (say K=100), remove any that overlap the known positives, and treat the rest as negatives. Cheap, deterministic, no model required. This is the DPR baseline and still the right first pass for any new dataset.
2. **Model-mined hard negatives (ANCE / bi-encoder self-mining).** Train a bi-encoder on random or BM25 negatives for one epoch, then use *that* bi-encoder to re-mine negatives against the corpus. Repeat. Introduced as **ANCE** (Xiong et al., ["Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval"](https://arxiv.org/abs/2007.00808), *ICLR 2021*). The bi-encoder finds the specific things *it* gets wrong.
3. **Cross-encoder-mined negatives (RocketQA / GPL).** Use a strong cross-encoder to score all (query, candidate) pairs from stage 2, keep the ones the cross-encoder rates as *low* (true negatives) and discard the ones it rates as *high* (likely false negatives). The cross-encoder is your denoiser. This is the modern default for high-quality supervised retrieval training.

Each source can produce ~7–15 hard negatives per query. Common recipe: 1 positive + 7 hard negatives + N in-batch soft negatives per training example.

## The BM25 baseline recipe

`rank_bm25` is the standard Python implementation for a first pass:

```python
from rank_bm25 import BM25Okapi

corpus_tokens = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(corpus_tokens)

def mine_bm25_negatives(query, positive_ids, k=100, num_negatives=7):
    scores    = bm25.get_scores(query.lower().split())
    top_idx   = scores.argsort()[::-1][:k]
    negatives = [i for i in top_idx if i not in positive_ids][:num_negatives]
    return negatives
```

For any corpus larger than ~1 M documents, use ElasticSearch or Vespa BM25 in production. `pyserini` (Lin et al., 2021) wraps Anserini's Lucene BM25 for exactly this — production-scale BM25 with a Python interface — and is what most public retrieval papers use.

Common trap: **do not remove the positives from the BM25 candidate pool before ranking**. Remove them from the *mined negatives*. If you filter positives out before BM25 scoring, you distort the score distribution.

## ANCE and self-mining

**ANCE** (Xiong et al., 2021) introduced the iterative bi-encoder mining loop that is now the default:

1. Train a bi-encoder on BM25 (or random) negatives for one epoch.
2. Encode the entire corpus with the just-trained bi-encoder; build an ANN index.
3. For each training query, retrieve the top-K nearest neighbours from the index.
4. Remove any that overlap known positives. The rest are the new hard negatives.
5. Continue training on the new negatives.
6. Repeat 2–5 every few epochs.

The paper's finding: the negatives from a *just-trained* bi-encoder are meaningfully harder than BM25's, and re-mining every epoch produces a smooth quality curve that plain BM25 training saturates against.

Practical warnings:

- **Re-mining is expensive.** Each iteration requires re-encoding the whole corpus and rebuilding the ANN index. Budget for it.
- **Async re-mining.** In practice, most implementations use the *previous* iteration's mined negatives while the current iteration is training, and refresh once every N steps.
- **The negatives get *too* hard eventually.** After 3–4 iterations, you start mining false negatives. This is where cross-encoder denoising becomes essential.

`sentence-transformers` provides `mine_hard_negatives` as a first-class helper (added in v3.0):

```python
from sentence_transformers.util import mine_hard_negatives

mined = mine_hard_negatives(
    dataset,                              # HF Dataset with anchor + positive columns
    model,                                # bi-encoder to mine with
    num_negatives=7,
    range_min=10,                         # skip the top-10 (likely paraphrases / false negs)
    range_max=100,                        # look through the top-100
    margin=0.0,                           # score margin below the positive to accept
    use_faiss=True,
)
```

Two knobs you will care about:

- **`range_min`.** The top few candidates for any query are the most likely false negatives — they are what a decent model already ranks near the positive. Skipping the top 5–20 is a cheap denoise.
- **`margin`.** Only keep a candidate as a hard negative if its similarity to the query is at least `margin` below the positive's similarity. Removes candidates the model is already handling correctly.

## Cross-encoder denoising: the RocketQA / GPL loop

The 2020-2022 papers converged on a variation of the following loop, and it is the modern default:

1. **BM25 candidate mining.** Top-100 BM25 per query.
2. **Cross-encoder scoring.** Score every (query, BM25-candidate) pair with a strong cross-encoder. This is O(queries × 100 × pair-forward-pass) — expensive, but a one-shot preprocessing cost.
3. **Denoise.** For each query, drop candidates that the cross-encoder rates *above* the known positive. Those are the most likely false negatives.
4. **Rank the survivors.** Sort remaining candidates by cross-encoder score. Take the top ~7 as hard negatives.
5. **Train the bi-encoder** on those denoised hard negatives.
6. **(Optional) Re-mine after a few epochs** with the trained bi-encoder as the candidate source.

This is what **RocketQA** (Qu et al., 2021), **GPL** (Wang et al., ["GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval"](https://arxiv.org/abs/2112.07577), *NAACL 2022*), and every high-quality open bi-encoder training run implement.

`sentence-transformers` supports this with `mine_hard_negatives(..., cross_encoder=..., margin=...)`. Explicitly pass a strong reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2` at minimum; `BAAI/bge-reranker-large` for higher quality) and it will apply the denoising.

## GPL: negatives without labels

**Generative Pseudo Labeling (GPL)** (Wang et al., 2022) closes the loop for the domain-adaptation case where you have *no* labelled query-passage pairs at all — the situation you are always in when adapting to a new corpus.

The pipeline:

1. **Generate synthetic queries** from each passage with a query-generation model (`doc2query/msmarco-t5-base-v1` is the reference). One passage → several synthetic queries.
2. **Mine BM25 hard negatives** for each (synthetic-query, passage) pair.
3. **Cross-encoder score** every (synthetic-query, mined-negative) pair to get soft relevance scores.
4. **Train the bi-encoder** with `MarginMSELoss` — MSE between the bi-encoder's `score(q, d+) - score(q, d-)` and the cross-encoder's `score(q, d+) - score(q, d-)`.

The trained bi-encoder is now adapted to the corpus *without ever having seen a real labelled query*. Chapter 09 uses GPL as the reference domain-adaptation recipe.

## The five negative types you will actually encounter

Any hard-negative mining pipeline surfaces a mix of these; know which is which:

- **True hard negative.** Topically related but genuinely not the answer. Gold — this is what you want.
- **False negative (unlabelled positive).** Actually relevant, but not labelled as such. Poison. Mitigated by cross-encoder denoising or the top-K skip in mining.
- **Near-duplicate positive.** A copy or paraphrase of the labelled positive. Also a false negative. Mitigate with content-level deduplication before batching.
- **Trivial semantic negative.** Off-topic and easy to distinguish. Fine, but adds no gradient signal. Filter out by requiring `score(q, d-) > threshold`.
- **Adversarial / spam negative.** Passages that "look like" answers (long paragraphs of formal text) but are actually irrelevant. Occasionally useful for robustness training; usually filtered.

Log the mix on your mined negatives before every training run. A quick sanity check: pick 20 mined (query, negative) pairs at random and human-label them. If >20 % are false negatives, tighten the denoising.

## Curriculum training

The idea: don't train on the hardest negatives from day one. Ramp up hardness gradually — easy negatives when the model is untrained, harder ones as it improves.

Two practical implementations:

- **Iterative re-mining (ANCE-style).** The negatives naturally get harder as the bi-encoder they were mined with improves. This is curriculum learning in the loosest sense — no explicit stage boundaries, but the difficulty of the training data increases monotonically.
- **Explicit curriculum.** Stage the training data by difficulty. Train first on `(query, BM25-negatives-rank-50-100)`, then `(query, BM25-negatives-rank-20-50)`, then `(query, cross-encoder-denoised-BM25-rank-5-20)`. Common in domain-adaptation contexts where you have a small labelled set and want to make it stretch.

The `sentence-transformers` documentation's ["Training on Multiple Datasets"](https://www.sbert.net/docs/sentence_transformer/training_overview.html) pattern covers explicit-curriculum through the `MultiDatasetBatchSamplers` primitive — you can weight different data sources per stage.

## When to stop mining harder

A saturation heuristic: you have mined too hard when the fraction of *false* negatives in your mined set exceeds ~5 %. At that point, the training signal is dominated by mislabelled data and further mining hurts. Signals:

- **Recall@100 stops improving** across mining iterations while positive-cosine mean plateaus.
- **Positive-negative cosine gap narrows** — the model is being told "these are close but wrong," but they are actually close because they *are* right.
- **Human-labelled sample of mined negatives crosses ~5 % false rate.** The definitive check.

At that point, freeze the mining and either move to a distillation stage (chapter 06) or accept the current quality and evaluate.

## Chapter summary

- Hard negatives — negatives that a scoring function ranks *near* the positive — are the primary quality knob for retrieval-embedding training beyond the first few thousand steps.
- Mining hierarchy: BM25 (cheap baseline) → ANCE-style bi-encoder self-mining (better) → cross-encoder-denoised (best, avoids false negatives).
- False negatives — unlabelled positives that mining picks up — are the mining pipeline's main failure mode. Mitigate with a `range_min` top-K skip, a `margin` threshold, or cross-encoder denoising.
- GPL (generative pseudo-labelling) is the reference recipe when you have *no* labelled queries at all — synthesise queries from passages, mine with BM25, denoise with a cross-encoder, distil into a bi-encoder.
- `sentence-transformers.util.mine_hard_negatives` is the first-class helper. Read its arguments — `range_min`, `range_max`, `margin`, and `cross_encoder` are all doing real work.
- Curriculum training — ramping hardness — is natural in ANCE-style iterative mining and explicit in dataset-staged pipelines.
- Stop mining harder when >5 % of mined negatives are false negatives; further mining degrades the model.
- Always human-label 20 mined negatives before a new training run. It is the cheapest bug catch you will find in this whole module.
