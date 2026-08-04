# Cross-Encoder Reranking and ColBERT

## Motivation

Chapter 02 introduced the cross-encoder as the accuracy-maximising second stage of a two-stage retrieval pipeline; chapters 04–06 covered the training of both the bi-encoder and the negatives the cross-encoder consumes. This chapter is the reranker's home: how to train one, how to serve one, when ColBERT-style late interaction is the better trade-off, and how to think about the latency budget the rest of your stack has to fit inside of.

Reranking is where most of the "wait, retrieval got much better on our real corpus" wins come from in production. A well-trained bi-encoder gets the right answer into the top 100; a well-trained cross-encoder gets it into the top 1 or top 3. That is the specific win, and the specific reason it's worth the ~200 ms per query.

## The reranker's job in one sentence

Given a first-stage candidate list, re-order it so that the most relevant items rise to the top. That's the job. Nothing else.

Concretely, the reranker is a scalar-scoring function $r(q, d)$ applied to each `(query, candidate)` pair independently, followed by a sort:

```python
scores = [reranker.score(query, d) for d in first_stage_candidates]
reranked = [d for _, d in sorted(zip(scores, first_stage_candidates), reverse=True)]
return reranked[:top_n_final]
```

Everything in this chapter is a variation on that skeleton. The `reranker.score` implementation varies (cross-encoder, ColBERT MaxSim, monoT5) and the "sorted, take top-N" step sometimes uses more elaborate fusion, but that is the core.

## Training the workhorse cross-encoder

The reference cross-encoder is a BERT/DeBERTa/ELECTRA encoder with a scalar regression head over the pooled `[CLS]` token, trained on `(query, document, relevance-label)` triples. The `sentence-transformers` API:

```python
from sentence_transformers import CrossEncoder, InputExample, losses
from torch.utils.data import DataLoader

model = CrossEncoder("microsoft/deberta-v3-base", num_labels=1)

train = [
    InputExample(texts=["what is the capital of France?", "Paris is the capital of France."], label=1.0),
    InputExample(texts=["what is the capital of France?", "Berlin has a population of 3.5M."], label=0.0),
    # ... hard negatives mined per chapter 05
]

model.fit(
    train_dataloader=DataLoader(train, shuffle=True, batch_size=32),
    epochs=1,
    warmup_steps=1000,
    optimizer_params={"lr": 2e-5},
    loss_fct=torch.nn.BCEWithLogitsLoss(),        # binary relevance
)
```

Three loss families, three training-data shapes:

- **Binary cross-entropy on `(q, d, 0 or 1)`.** The MS MARCO "shallow" setup. Cheapest data — you only need to know if a candidate is relevant or not. `num_labels=1`, `BCEWithLogitsLoss`.
- **Regression on `(q, d, graded relevance)`.** MSMARCO's `qrels` file gives 0/1/2/3 labels for TREC-DL. MSE or the MSE variant of `MarginMSELoss` from cross-encoder chapter 06. Preferred when you have graded labels.
- **Listwise KL divergence on `(q, [d_1, d_2, ...], teacher_scores)`.** The distillation setup — reproduce a teacher's ranking across candidates. This is how public state-of-the-art rerankers like `bge-reranker-v2` are trained (distilled from stronger models).

**Ranking, not classification, is the objective.** A cross-encoder that reports well-calibrated relevance probabilities is not necessarily a better *ranker* than one whose scores are miscalibrated but correctly ordered. Read the model card; if it says "trained with cross-entropy on binary labels," treat outputs as *logits for ordering*, not as probabilities.

## Which candidates to train on: the "hard negative" question, again

Chapter 05 covered mining. The specific concern for reranker training: your reranker will *only ever see candidates that made it through your first stage*. Training it on other candidates wastes the model's capacity. The rule:

- **Train the reranker on candidates your first-stage retriever actually surfaces.** Not on random negatives, not on BM25 negatives if you retrieve with a bi-encoder in production, not on training-set negatives that don't reflect the production candidate distribution.
- **Mine the negatives with the same retriever, same corpus, same tokenisation.** Anything else silently distorts the reranker's calibration.

`sentence-transformers.util.mine_hard_negatives` with your first-stage bi-encoder is again the right helper.

## The public cross-encoder shelf

Recognise these names and their trade-offs:

| Model                                             | Backbone         | Params | Latency (rough, 1 pair, GPU) | Notes                                     |
|---------------------------------------------------|------------------|--------|------------------------------|-------------------------------------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2`             | MiniLM           | 22 M   | ~5 ms                         | The "cheap default." Trained on MS MARCO. |
| `cross-encoder/ms-marco-MiniLM-L-12-v2`            | MiniLM           | 33 M   | ~10 ms                        | Slight quality lift.                       |
| `cross-encoder/ms-marco-electra-base`              | ELECTRA-base     | 110 M  | ~30 ms                        | Higher quality than MiniLM.                |
| `BAAI/bge-reranker-base` / `-large`                 | XLM-R / MPNet    | 278/560 M | ~60 / 100 ms               | Multilingual; strong on BEIR.              |
| `BAAI/bge-reranker-v2-m3`                          | XLM-R-based      | 568 M  | ~100 ms                       | Multilingual, current strong default.      |
| `mixedbread-ai/mxbai-rerank-large-v1`              | XLM-R-based      | 435 M  | ~80 ms                        | Strong on BEIR + open weights.             |
| `Alibaba-NLP/gte-multilingual-reranker-base`       | XLM-R-based      | 306 M  | ~60 ms                        | Multilingual reranker.                     |
| `castorini/monoT5-base-msmarco`                    | T5-base          | 220 M  | ~40 ms                        | Text-to-text; generates "true"/"false".    |
| `Cohere rerank-3` (hosted)                         | —                | —       | ~50 ms per pair (network)     | Best-in-class hosted reranker.             |
| `voyage-rerank-2` (hosted)                         | —                | —       | ~50 ms per pair (network)     | Alternative hosted reranker.               |

The rough decision: MiniLM if latency is critical, `bge-reranker-v2-m3` or `mxbai-rerank-large-v1` if quality matters and you can afford ~100 ms per pair. Batching 100 pairs on a GPU shaves that from `100 × 100 ms` down to `~500 ms` end-to-end for the whole rerank stage.

## monoT5 and the text-to-text reranker

**monoT5** (Nogueira, Jiang, Pradeep & Lin, ["Document Ranking with a Pretrained Sequence-to-Sequence Model"](https://arxiv.org/abs/2003.06713), *EMNLP 2020 Findings*) is a slightly different architecture worth naming: instead of a scalar head, use a T5-family encoder-decoder and prompt it to generate "true" or "false" given the pair. The score is the softmax probability of "true" at the first decoded token.

The template:

```
Query: {q} Document: {d} Relevant:
```

The model is trained to output `true` for relevant and `false` for irrelevant. At inference:

$$
r_\text{monoT5}(q, d) = \frac{p(\text{"true"})}{p(\text{"true"}) + p(\text{"false"})}.
$$

Where it shines: any setting where you already have a T5 in your stack, or where you want a reranker that is also "explainable" (you can prompt monoT5 for a rationale as a byproduct). Where it does not: latency-critical serving — the encoder-decoder pass is 2–3× slower than a same-size encoder.

## ColBERT: late-interaction as the middle road

**ColBERT** (Khattab & Zaharia, ["ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"](https://arxiv.org/abs/2004.12832), *SIGIR 2020*) and **ColBERTv2** (Santhanam et al., ["ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"](https://arxiv.org/abs/2112.01488), *NAACL 2022*) sit between bi-encoders and cross-encoders on both quality and cost.

The architecture:

- Encode both query and document as *sequences of per-token vectors* (typically 128-dim projections of BERT hidden states).
- Precompute the document token vectors offline, just like a bi-encoder.
- At query time, compute the score as *MaxSim*:

$$
s_\text{ColBERT}(q, d) = \sum_{i \in q} \max_{j \in d} \phi(q_i) \cdot \phi(d_j).
$$

Each query token picks the document token it likes best; the score is the sum of those per-query-token maxes.

Trade-offs:

- **Ranking quality:** typically within 1–3 nDCG@10 points of a cross-encoder on MS MARCO. Big lift over a plain bi-encoder.
- **Index size:** 10–30× larger than a bi-encoder. One vector per token, not per document. ColBERTv2 introduced centroid-based compression (PLAID) that recovers most of the size.
- **Serving:** MaxSim is not a native operation for typical ANN indexes. Requires PLAID or dedicated ColBERT-serving infrastructure (RAGatouille, colbert-live-search).

When to reach for it:

- **You need cross-encoder-adjacent quality at bi-encoder-adjacent latency**, and your corpus is small-to-medium (~1 M passages).
- **You are training on strictly limited compute** — ColBERTv2 can be fine-tuned on far less data than a bi-encoder + cross-encoder stack.
- **You want to skip the two-stage pipeline** entirely with a single-stage retriever.

When not:

- **Corpus > 100 M passages** — the token-level index size is prohibitive.
- **Your team doesn't want to run ColBERT-specific serving infra** — the operational cost outweighs the quality benefit.

Modern ColBERT tooling: `stanford-futuredata/ColBERT` for training; `Answer.AI/rerankers` and `mixedbread-ai/pylate` for lighter-weight serving; RAGatouille for the full "load a ColBERT and use it as a reranker" workflow.

## Late interaction beyond ColBERT

Several 2023–2024 models sit in similar territory:

- **`Alibaba-NLP/gte-multilingual-base` late-interaction mode** — the same GTE checkpoint can be used bi-encoder or late-interaction.
- **`BAAI/bge-m3`** — supports dense (bi-encoder), sparse (SPLADE-style), and multi-vector (ColBERT-style) representations from the *same model*, letting you pick the mode per corpus.
- **SPLADE** (Formal, Piwowarski & Clinchant, 2021) — a different middle road: sparse vectors that mimic BM25 but are learned, produced by a MLM head over the vocabulary. Not "late interaction" per se but sits in the same "between bi-encoder and cross-encoder" band.

You do not need to master all of these. You do need to know that "bi-encoder or cross-encoder" is not an exhaustive taxonomy in 2025.

## Latency budgeting for the reranker stage

The rerank stage's total latency is roughly:

$$
T_\text{rerank} \approx \underbrace{T_\text{encode}(q)}_{\sim 5\text{ms}} \;+\; \underbrace{|C| \cdot T_\text{pair}}_{\text{dominant}} \;+\; \underbrace{T_\text{sort}}_{\text{negligible}}
$$

where $|C|$ is the candidate-list size and $T_\text{pair}$ is a single (query, document) forward pass. With batching on a GPU, `|C| pairs` runs in roughly `|C| * T_pair / batch_size` — but you are memory-bound by document length, so the effective batch caps at 32–128 pairs on a mid-range GPU.

The two knobs to trade quality for latency:

- **Reduce `|C|`.** Rerank fewer candidates. Recall@|C| from the first stage matters here — if your bi-encoder gets the right answer into the top 20 reliably, rerank 20, not 100.
- **Cheaper reranker.** MiniLM is ~10× faster than `bge-reranker-large`. The quality drop is often less than the latency saving warrants.

Cache reranker scores by `(query-fingerprint, doc-id)` — the same query hitting the same corpus does not need to be re-scored — but note that fingerprinting queries loses out to any queries that differ by even a token, so cache hit rates are usually low outside of narrow domains (support chatbots hitting FAQ pages).

## Evaluating the rerank stage in isolation

You do not evaluate a reranker in a vacuum — you evaluate the *combined* two-stage pipeline. But you also want to separate "the bi-encoder is bad" from "the reranker is bad." The trick:

- **Recall@|C| of the first stage.** How often does the gold answer make it into the candidate list? This is a *ceiling* on the two-stage pipeline's precision. If recall is 60 %, no reranker can push precision above that.
- **nDCG@k, MAP, MRR of the reranker output vs. random shuffle.** How much does the reranker actually re-order things?
- **Oracle reranker on the same candidates.** What is the maximum achievable precision if the reranker were perfect? The gap between "oracle reranker" and "actual reranker" is your reranker's remaining headroom.

Chapter 08 puts this all together on MTEB.

## Common failure modes

- **Reranker helps on training-distribution queries, hurts on production queries.** Trained on the wrong candidate distribution — retrain on candidates from *your* first stage, not from BM25 or a public benchmark.
- **`|C| = 100` blows the latency budget.** Drop to `|C| = 20` and see if the recall drop matters. Usually it doesn't — the interesting candidates are in the top 20.
- **Reranker gives NaN or −inf scores on empty documents.** Filter empty and near-empty documents before scoring. This should not happen in a well-preprocessed corpus but often does.
- **Two-stage pipeline is worse than the bi-encoder alone.** Usually a symptom of a reranker trained on binary labels but a benchmark that cares about graded relevance. Train (or fine-tune) with graded labels or `MarginMSELoss`.
- **Reranker is slow *and* wrong.** You picked `bge-reranker-large` on CPU. Move to a GPU or drop to MiniLM.

## Chapter summary

- The reranker's job is to re-order a candidate list from the first stage. Its architecture is a cross-encoder — joint forward pass on `[q, d]`, scalar output.
- Train the reranker on candidates from *your* first-stage retriever, not from BM25 or a public dataset — the candidate distribution matters.
- Three loss families: binary cross-entropy (cheap), regression on graded labels (better), distillation with MarginMSE / KL (best).
- Public reranker shelf spans from `cross-encoder/ms-marco-MiniLM-L-6-v2` (~5 ms, cheap default) to `bge-reranker-v2-m3` (~100 ms, quality default).
- monoT5 is a text-to-text alternative — cross-encoder-adjacent quality with T5-native "true"/"false" scoring. Nice when you already have T5 in the stack.
- ColBERT / ColBERTv2 sit between bi-encoder and cross-encoder: late-interaction MaxSim over per-token vectors. Better ranking than bi-encoder, larger index. Reach for it on small-to-medium corpora when you want a single-stage pipeline.
- Latency = `|C| * T_pair` roughly. Cut `|C|` or switch to a smaller reranker as the primary knobs.
- Evaluate the two-stage pipeline as one system, but separately measure first-stage recall@|C| and reranker vs. oracle reranker to attribute quality wins and losses.
