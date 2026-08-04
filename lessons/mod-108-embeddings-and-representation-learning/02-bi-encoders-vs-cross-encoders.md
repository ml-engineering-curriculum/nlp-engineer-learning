# Bi-Encoders vs. Cross-Encoders

## Motivation

Almost every retrieval and semantic-matching decision you make will land on one axis: *do I need the query and the document to look at each other during scoring, or not?* Bi-encoders say no — they encode each side independently, once. Cross-encoders say yes — they run a fresh joint forward pass for every pair. The performance and cost consequences of that single decision are enormous, and they mostly explain the shape of every real retrieval stack you will build. This chapter is the architectural comparison; chapter 07 goes deep on cross-encoder rerankers and their late-interaction cousins.

## The bi-encoder

A bi-encoder (also: *dual encoder*, *two-tower model*, *Siamese network*) is two applications of the same encoder — one to the query, one to the document — followed by an independent pooling step, followed by a similarity function computed *after* the encoding is done.

```
              ┌── encoder ── pool ── q_vec  ─┐
   query  ──▶ │                              ├── sim(q_vec, d_vec) ── score
              └── encoder ── pool ── d_vec  ─┘   (cosine, dot, or L2)
   doc    ──▶
```

Two properties fall out and drive every downstream design choice:

- **The document embeddings are precomputable.** You encode your corpus once, offline. At query time you only encode the query and do an ANN lookup against the index. Total online cost per query is O(one encode + one ANN search) — independent of corpus size, in the ANN-limit sense.
- **No cross-attention between query and document.** The model never sees a query token attend to a document token, or vice versa. Every "interaction" happens *after* both sides have been reduced to a single vector.

Both properties are the reason bi-encoders scale to billions of documents. Both are also the reason they cap out below cross-encoders on ranking quality: the vector has to encode "everything that any query could ever need to know about this document" without knowing the query.

Reference recipe:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
docs  = ["Paris is the capital of France.", "The Eiffel Tower is in Paris."]
d_vec = model.encode(docs, normalize_embeddings=True)     # precompute once
q_vec = model.encode(["Where is the Eiffel Tower?"], normalize_embeddings=True)
scores = cos_sim(q_vec, d_vec)                             # 1 x N
```

`normalize_embeddings=True` matters: BGE, E5, and most modern bi-encoders are trained with cosine similarity, so you must L2-normalise before dot product to reproduce the training-time contract. Chapter 03.

## The cross-encoder

A cross-encoder puts the query and the document into the *same* input, runs a single transformer forward pass, and reads a scalar relevance score off the pooled representation.

```
   query ┐
         ├── [CLS] q [SEP] d [SEP] ── encoder ── pool ── linear ── score
   doc   ┘
```

The consequences:

- **Cross-attention on every layer.** Each query token attends to each document token, on every layer. That is where the ranking quality comes from — the model can look at the query while reading the document and vice versa.
- **No precomputation is possible.** The score depends on the *pair*. You cannot factor "encode(document)" out of the online path — every candidate has to be scored fresh against every query.
- **Latency scales with `candidates × pair_forward_pass`.** At ~50 ms per pair on a middling GPU for a base cross-encoder, 100 candidates is 5 s of pure GPU time. This is why cross-encoders only work as a *reranker*, not a first-stage retriever.

Reference recipe:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs    = [("Where is the Eiffel Tower?", d) for d in docs]
scores   = reranker.predict(pairs)                         # 1 forward pass per pair
```

## Two-stage retrieval: recall then precision

The dominant retrieval architecture in 2025 is the two-stage pipeline, and it is dominant precisely because it uses each architecture where it is strongest:

```
                            ┌── bi-encoder ANN top-k=100 ──┐
   query ──── bi-encoder ──▶│                              │
                            └── cross-encoder rerank top-10 ─▶ final
```

- **Stage 1 (bi-encoder):** high recall, low precision. Return the top ~100–200 candidates from a billion-document index in tens of milliseconds.
- **Stage 2 (cross-encoder):** high precision, low recall. Re-score those 100 candidates and return the top 10 in another ~200 ms.

Two commitments follow from this shape:

- **Optimise the bi-encoder for recall@100, not nDCG@10.** The cross-encoder will handle ordering; the bi-encoder's job is to make sure the right answer is *somewhere* in the top 100.
- **Optimise the cross-encoder for ranking on the bi-encoder's output distribution.** Train the reranker on hard negatives that the *bi-encoder* found (chapter 05), not on generic random negatives — otherwise it never learns to break ties that matter.

Both commitments show up in the exercises for this module.

## When you would use only a bi-encoder

- **Corpus > 10⁵ documents and latency < 200 ms.** No time for reranking.
- **Semantic dedup, clustering, kNN classification.** You need pairwise similarity between arbitrary items, not query-vs-corpus ranking. Precompute once; use forever.
- **Vector-index-based memory for an agent.** The write path is `encode(turn) → index`; the read path is `encode(query) → search`. There is no "candidate list" to rerank.
- **Zero-shot classification with label prototypes.** Encode label descriptions once; encode inputs at inference; assign by nearest neighbour.
- **Cross-lingual retrieval where the two sides are in different languages.** LaBSE and multilingual-E5 (chapter 10) are bi-encoders because that is what indexes.

## When you would add a cross-encoder

- **Precision-critical ranking.** Legal search, medical retrieval, product search where a wrong-top-1 has a real cost.
- **When you have a good bi-encoder recall but the top-1 is often #3–#10 in the list.** Symptom of "the right document is in the candidate set but not at the top" — exactly what reranking fixes.
- **When you can afford ~200 ms of extra latency per query** and your candidate list can be capped at 50–200 items.
- **When you have relevance-labelled query-document pairs** to train on. Cross-encoders are far more supervision-hungry per training example, but they benefit more per label than bi-encoders do.

If any of those are false, do not add a cross-encoder — you will pay a latency cost for no ranking benefit.

## The mathematical difference in one paragraph

Both architectures produce a score $s(q, d)$. In a bi-encoder,

$$
s_\text{bi}(q, d) = \phi_\theta(q) \cdot \phi_\theta(d)
$$

where $\phi_\theta$ is one encoder + pooling shared across sides. The score factorises as a dot product of two *independent* representations. In a cross-encoder,

$$
s_\text{cross}(q, d) = w^\top \psi_\theta([q ; d])
$$

where $\psi_\theta$ is a transformer that sees `[q ; d]` jointly and $w$ is a scoring head. The score is *not* a dot product; it is a function of the joint representation. That single algebraic difference — factorisable vs. not — is the entire tradeoff. If the score has to factor across query and document, it is a bi-encoder and you can precompute; if it cannot, it is a cross-encoder and you cannot.

## The ColBERT middle ground

**ColBERT** (Khattab & Zaharia, ["ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"](https://arxiv.org/abs/2004.12832), *SIGIR 2020*) and **ColBERTv2** (Santhanam et al., ["ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"](https://arxiv.org/abs/2112.01488), *NAACL 2022*) sit between the two. They encode each side to a *sequence* of per-token vectors (still independently — so the document tokens are precomputable) and score with

$$
s_\text{colbert}(q, d) = \sum_{i \in q} \max_{j \in d} \; \phi(q_i) \cdot \phi(d_j).
$$

This "late interaction" MaxSim lets each query token pick out its best-matching document token. The trade-offs:

- **Ranking quality between bi-encoder and cross-encoder** — often close to cross-encoder on MS MARCO, at a fraction of the online latency.
- **Index size 10–30× larger** than a plain bi-encoder — one vector per token, not per document.
- **More complex serving** — MaxSim over token vectors is not a straight ANN lookup, though PLAID (Santhanam et al., 2022) and ColBERTv2 introduced compression and centroid-based approximation to help.

You should recognise ColBERT as a third architecture, not as an "improved bi-encoder." Chapter 07 covers when it is the right choice.

## Where the training data differs

The same encoder backbone can be trained as either a bi-encoder or a cross-encoder — the architecture-level difference is where the query and document tokens meet, not what the weights are.

But the *training data* differs sharply:

| Data type                              | Bi-encoder | Cross-encoder |
|----------------------------------------|------------|---------------|
| Positive pairs `(q, d+)`                | Required   | Required      |
| Random negatives (in-batch)             | Sufficient for a first pass | Not enough — model saturates immediately |
| BM25-mined hard negatives               | Big lift (chapter 05) | Essential |
| Bi-encoder-mined hard negatives         | Second-pass lift | Essential — this is the reranker's job |
| Graded labels (0/1/2/3 relevance)       | Wastes signal — cannot express | Big lift — MSE or KL-div loss |
| Cross-encoder scores as soft targets    | The KD signal for the bi-encoder | N/A |

Chapter 06 covers the distillation loop where a strong cross-encoder produces soft targets that train a fast bi-encoder — the modern default for high-recall, low-latency retrieval.

## Failure modes and their symptoms

- **"Retrieval is perfect on the training set, terrible on production queries."** You likely trained a bi-encoder without hard negatives. Every training example was easy; the model never had to learn fine-grained ranking. Fix: chapter 05.
- **"Recall@100 is fine but nDCG@10 is bad."** Your bi-encoder is doing its job; the ranker on top of it is not. Add a cross-encoder or ColBERT reranker.
- **"The cross-encoder helps on some queries and hurts on others."** The cross-encoder was trained on a different candidate distribution than yours produces. Retrain the reranker on hard negatives your bi-encoder actually surfaces.
- **"Costs are unbounded because the cross-encoder is invoked on 200 candidates per query."** Cap the candidate list, cache reranked scores by query-fingerprint, or drop to ColBERT for lighter online cost.
- **"Embedding index quality drifts when we swap the encoder for a newer one."** Bi-encoder outputs are *not* interchangeable across models — even same-dimension outputs. Re-encode the corpus when you change the encoder.

## Chapter summary

- Bi-encoders factor the score as `sim(encode(q), encode(d))` — cheap online, indexable, and the only architecture that scales to billions of documents.
- Cross-encoders score `[q ; d]` jointly — far more accurate per pair, but must run a fresh forward pass for every pair and only work as a reranker.
- The canonical two-stage pipeline (bi-encoder recall → cross-encoder precision) is dominant because it uses each architecture where it wins.
- Optimise the bi-encoder for recall on your target-corpus queries; optimise the cross-encoder for ranking on the bi-encoder's output distribution.
- ColBERT and late-interaction models are a third architecture — bi-encoder-priced encoding with cross-encoder-adjacent ranking, at the cost of a much larger index.
- The mathematical distinction — factorisable vs. non-factorisable score — explains every downstream cost/quality trade-off.
- Bi-encoders and cross-encoders share encoder backbones; they diverge sharply in training-data requirements (see chapter 04–06).
