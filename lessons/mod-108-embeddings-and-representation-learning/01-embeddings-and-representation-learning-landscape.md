# The Embeddings and Representation Learning Landscape

## Motivation

Almost every serving path you touch as an NLP engineer eventually reduces to `encode(text) -> vector` — semantic search, deduplication, clustering, near-duplicate detection, classification with a linear head, zero-shot retrieval, reranking, memory for an agent, features for a downstream model. The vector is the interface, and the model that produces it is the *embedding model*. This module owns the training, evaluation, adaptation, and serving of those models.

The field went through three sharp transitions worth naming, because each still ships in production somewhere:

1. **Static word embeddings** — word2vec (Mikolov et al., 2013), GloVe (Pennington et al., 2014), FastText (Bojanowski et al., 2017). One vector per token, learned from co-occurrence. Still the right answer for lexical similarity in a tight latency budget and for cold-start baselines in classification (mod-103).
2. **Contextual encoder representations** — ELMo, BERT, and the encoder side of every transformer since. Every occurrence of a token gets its own vector conditioned on context. But BERT's `[CLS]` and mean-pooled representations are famously bad at sentence similarity out of the box — the 2019 Sentence-BERT paper opens with exactly that observation.
3. **Contrastively trained sentence and document embeddings** — Sentence-BERT (Reimers & Gurevych, 2019), SimCSE, DPR, Contriever, E5, BGE, Nomic Embed, Cohere embed, OpenAI `text-embedding-3-*`. The modern default. Trained with contrastive objectives on pair or triplet data so that "semantically similar things are near each other, and everything else is far apart" is a property the model actually has.

This module lives inside (3). We assume you already know what a transformer encoder is (mod-101), how tokenisation and pooling work (mod-101, mod-103), and how to fine-tune a classifier (mod-103). We build from there to bi-encoders, cross-encoders, contrastive losses, hard-negative mining, curriculum-style training, MTEB, domain and multilingual adaptation, and the serving stack.

## What is an "embedding," precisely?

An embedding model $f_\theta$ is a function

$$
f_\theta : \text{text} \to \mathbb{R}^d
$$

that maps a variable-length text to a fixed-length vector, such that a chosen similarity function $\text{sim}(u, v)$ — usually cosine similarity or dot product — approximates a task-relevant notion of "closeness" between the underlying texts.

Three things are pinned by that definition and are worth surfacing before you go looking at model cards:

- **Fixed output dimension $d$.** Typical values: 384 (MiniLM), 768 (BERT-base, most `all-MPNet` models), 1024 (BGE-large, `text-embedding-3-large` reduced), 1536 (OpenAI `text-embedding-3-small` default), 3072 (`text-embedding-3-large` default). Matryoshka-trained models expose *nested* useful prefixes of the vector at multiple $d$'s (chapter 10).
- **The similarity function is a model contract.** Cosine similarity, dot product, and Euclidean distance are not interchangeable — every embedding model is trained with one in mind, and swapping it at query time silently degrades quality. Read the model card.
- **The notion of closeness is task-dependent.** "Semantic textual similarity" (paraphrase), "retrieval relevance" (query-passage), "topic clustering" (label overlap), and "classification" (linear-separability under a downstream head) are *different objectives* that need *different training data* and reward *different embeddings*. MTEB (chapter 8) exists precisely because a single "best embedding" does not.

## What you can build with an embedding index alone

Before the module goes deep on training and serving, keep this list of downstream uses in mind — every design decision downstream is in service of one of these:

| Use case                    | What the vector store returns                                            | Where it lives on the track                     |
|-----------------------------|--------------------------------------------------------------------------|-------------------------------------------------|
| Semantic search / retrieval | Top-*k* passages ranked by similarity to a query embedding               | This module (encoding + eval), `rag-engineer` (retrieval infra) |
| Reranking                   | Reordering an initial candidate list with a stronger scorer               | Chapter 07 (cross-encoders, ColBERT)             |
| Deduplication / near-dup    | Groups of near-identical texts within a corpus                            | mod-110 (data engineering), this module (encoder) |
| Clustering / topic discovery| Cluster labels or hierarchical structure over documents                   | This module (encoder + MTEB clustering tasks)    |
| Classification with kNN     | Nearest-neighbour labels                                                  | mod-103 (baseline), this module (encoder)        |
| Zero-shot / few-shot        | "This text is closest to which of these label descriptions?"              | mod-103 (setup), this module (encoder quality)   |
| Memory for agents / RAG     | Retrieved past chat turns, tool results, or documents                     | `rag-engineer` track, this module (encoder)      |
| Cross-lingual retrieval     | Passages in one language retrieved with a query in another                | Chapter 10 (multilingual encoders)               |

The unifying pattern: you produce vectors offline, index them, then compute similarity to a query vector online. Everything else — chunking, filtering, hybrid scoring, prompt assembly — is downstream, and much of it is owned by the `rag-engineer` track (chapter 11).

## The two-architecture split you will use every day

Modern embedding stacks have two architectures with two different jobs:

- **Bi-encoders** (dual encoders, two-tower models). Encode query and document *independently*, then score with `cosine(q, d)` or `dot(q, d)`. Cheap at query time — document vectors are precomputed and indexed — and the only architecture that scales to a corpus of billions of documents. This is the "embedding model" in every embedding API you have used.
- **Cross-encoders** (cross-attention rerankers). Encode `[query || document]` *jointly* through a transformer and predict a scalar relevance score. Far more accurate per pair, but requires a full forward pass at query time for *every pair* — so they only work as a reranker over a small candidate list produced by a bi-encoder or BM25.

The two-stage `bi-encoder → cross-encoder` pipeline — recall then precision — is the canonical serving shape and is what BEIR, MS MARCO leaderboards, and every serious retrieval stack use. Chapters 02 and 07 own the architectures; chapter 05 owns the training of both.

A hybrid worth naming up front: **ColBERT** (Khattab & Zaharia, 2020) and **ColBERTv2** (Santhanam et al., 2022) sit between the two — a bi-encoder that emits one vector per *token* instead of one per document, then scores with a "late interaction" MaxSim over the token vectors. More accurate than a plain bi-encoder, cheaper than a cross-encoder, but the index is 10–30× larger. Chapter 07.

## The MTEB axis: five task families

The **Massive Text Embedding Benchmark** (Muennighoff, Tazi, Magne & Reimers, ["MTEB: Massive Text Embedding Benchmark"](https://arxiv.org/abs/2210.07316), *EACL 2023*) is the modern reference evaluation suite. It matters not just because it produces a leaderboard, but because it enforces the discipline of evaluating an embedding on *all five things it might be asked to do*:

| MTEB family        | What is measured                                            | Metric family                     |
|--------------------|-------------------------------------------------------------|-----------------------------------|
| **Retrieval**      | Query → passage ranking on BEIR-like datasets               | nDCG@10, MAP, Recall@100          |
| **Reranking**      | Reranking a pre-retrieved candidate list                    | MAP, MRR                          |
| **STS**            | Semantic Textual Similarity — sentence pair correlation     | Spearman ρ on cosine similarity   |
| **Classification** | Linear-probe / logistic-regression accuracy on frozen embeddings | Accuracy, F1                 |
| **Clustering**     | Unsupervised clustering label recovery                      | V-measure                         |
| **Pair classification** | Binary "same or different" over pairs                  | AP, F1                            |
| **Bitext mining**  | Cross-lingual sentence-pair alignment                       | F1                                |

A model that wins on retrieval but loses on STS is not a paradox — it is a *design signal* that the training data was retrieval-heavy. Chapter 8 is the full protocol; the point here is that "which embedding is best" is not a single number.

## Model families you should recognise

The public embedding-model landscape as of 2025-Q3 breaks roughly into four groups. All are downstream trainings of a transformer encoder or encoder-decoder backbone.

- **General-purpose bi-encoders.** `sentence-transformers/all-MiniLM-L6-v2` (22 M params, 384-dim — the "cheap default"), `all-mpnet-base-v2` (110 M, 768-dim — the "quality default of 2021"), `BAAI/bge-large-en-v1.5` and `BAAI/bge-m3` (Xiao et al., ["C-Pack: Packaged Resources To Advance General Chinese Embedding"](https://arxiv.org/abs/2309.07597), 2023; and the multilingual `bge-m3` from Chen et al., 2024), `intfloat/e5-large-v2` and `intfloat/multilingual-e5-large` (Wang et al., ["Text Embeddings by Weakly-Supervised Contrastive Pre-training"](https://arxiv.org/abs/2212.03533), 2022), `nomic-ai/nomic-embed-text-v1.5` (Nussbaum et al., ["Nomic Embed: Training a Reproducible Long Context Text Embedder"](https://arxiv.org/abs/2402.01613), 2024).
- **Hosted / commercial embeddings.** OpenAI `text-embedding-3-small` / `-large`, Cohere `embed-english-v3` / `embed-multilingual-v3`, Voyage `voyage-3` / `voyage-3-large`, Google `text-embedding-004` / `gecko`, Anthropic (indirectly, via partner APIs). Same architecture family; you trade the ability to inspect and adapt weights for a maintained API.
- **Cross-encoders / rerankers.** `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/ms-marco-electra-base`, `BAAI/bge-reranker-large` and `bge-reranker-v2-m3`, `mixedbread-ai/mxbai-rerank-large-v1`, Cohere `rerank-3`. Chapter 07 in depth.
- **Domain-specialised.** `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical), `nlpaueb/legal-bert-base-uncased` (legal), `allenai/specter2` and `allenai/scincl` (scientific citation embeddings), `BAAI/bge-large-code` and CodeBERT-family (code search). Chapter 9.

You do not need to memorise the leaderboard. You do need to know that these families exist, roughly what they cost per token, and where to look for one that fits your domain.

## Where this module ends and `rag-engineer` begins

The single most common source of scope confusion in production RAG teams is: who owns the encoder, who owns the store, and who owns the ranking function? This track draws the line explicitly and chapter 11 makes it durable.

- **NLP engineer (this module) owns:** the encoder model — training, adapting, evaluating (MTEB), quantising, and shipping a serving path (chapter 10). The *embedding contract* — dimension, similarity function, normalisation, tokenisation, expected input length — is your API.
- **RAG engineer owns:** chunking strategy, the vector index (FAISS/HNSW/ScaNN/Qdrant/Weaviate), hybrid scoring with BM25, retrieval evaluation on the target corpus, prompt assembly, and the retrieval-augmented pipeline end-to-end.
- **Shared:** cross-encoder rerankers (this module trains them; RAG engineer wires them in), retrieval evaluation dataset construction (this module cares because MTEB, RAG engineer cares because BEIR-on-your-corpus), and the embedding-serving latency budget.

If someone on your team says "our RAG is bad," step one is always to answer "which stage" — retrieval recall (encoder or index), reranking precision (cross-encoder or model), or generation faithfulness (mod-105 / mod-106). This module owns the first stage's *encoder*.

## The five questions the rest of the module answers

Each chapter is anchored on one:

1. **Bi-encoder or cross-encoder?** Chapter 02.
2. **How do I train it — objective and negatives?** Chapters 03–06.
3. **Is it any good, and on which axis?** Chapter 08 (MTEB).
4. **Does it work on my domain / language?** Chapters 09–10.
5. **Can I serve it at production latency and cost?** Chapter 11.

## Chapter summary

- Embeddings are `text -> R^d` functions with a specified similarity function. The similarity function is part of the contract.
- Three-era history: static word vectors → contextual encoders → contrastively trained sentence/document embeddings. This module lives in the third era.
- Two dominant architectures: bi-encoders (independent, cheap, indexable) and cross-encoders (joint, accurate, only for reranking). ColBERT and late-interaction models sit in between.
- MTEB defines five task families — retrieval, reranking, STS, classification, clustering (plus pair-classification and bitext mining) — and there is no single "best" embedding across all of them.
- The public model landscape splits into general-purpose bi-encoders, hosted APIs, rerankers, and domain-specialised checkpoints. Recognise the families; do not memorise the leaderboard.
- Ownership boundary with the `rag-engineer` track is sharp: NLP engineer owns the encoder, contracts, and MTEB evaluation; RAG engineer owns the store, chunking, hybrid scoring, and pipeline.
