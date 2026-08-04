# The Embeddings / RAG-Engineer Boundary

## Motivation

Chapter 01 named the ownership boundary between this module and the `rag-engineer` track. This chapter makes it durable — the specific interfaces, the specific handoffs, the specific artefacts each side ships, and the specific failure modes that live in the gap between them. This is short and prescriptive by design: when a RAG stack has a quality problem, "which team owns this?" should not be a debate.

## The boundary in one paragraph

The NLP engineer owns the encoder: the *thing that turns text into a vector*. That includes training data curation, contrastive objective choice, hard-negative mining, MTEB evaluation, domain and language adaptation, cross-encoder reranker training, quantisation, and the serving-path throughput of the encoder itself. The RAG engineer owns everything else that has to happen for a retrieval-augmented system to work: chunking strategy, the ANN index and vector store, hybrid scoring with BM25, retrieval-eval on the target corpus, prompt assembly, context truncation, the response-generation pipeline. Shared surface: the interfaces between the two, the reranker (this module trains it, RAG wires it in), and the target-corpus retrieval evaluation set (both use it for different tests).

## The encoder contract

The single interface between the two teams is the *encoder contract*. This is what the NLP engineer publishes; this is what the RAG engineer consumes. It has exactly six fields:

1. **Model identifier and version.** `bge-large-en-v1.5-domain-adapted-2025-07-15@v3` — a name that pins model, adaptation, and version.
2. **Output dimension.** `768` (or the Matryoshka menu: `[64, 128, 256, 512, 768]`).
3. **Similarity function.** `cosine` (or `dot-product` if the model was trained that way).
4. **Precision.** `fp32` at emit; `int8` or `binary` if quantised. Include the calibration parameters if any.
5. **Task prefix.** The exact string prefix for queries and passages (or `null` for models that don't use them).
6. **Max input length.** In tokens, at the tokeniser level. The RAG engineer's chunker must respect this.

Publish it as a machine-readable file — YAML or JSON — that ships alongside the model weights:

```yaml
model: bge-large-en-domain-2025-07-15
version: v3
dim: 768
similarity: cosine
precision: fp32
query_prefix: "query: "
passage_prefix: "passage: "
max_seq_length: 512
tokenizer: BAAI/bge-large-en-v1.5
```

Every prod incident I have seen in this space traces back to one of these six fields drifting silently. Make them explicit.

## What each side ships

### NLP engineer ships

- **Trained encoder weights + tokeniser** on Hugging Face or a private model registry.
- **Encoder contract YAML** (above).
- **MTEB scores** for the shipped model on the relevant category — retrieval if that's the use case.
- **Domain evaluation** on an agreed-upon in-house set (~100 queries, human-labelled). Chapter 09.
- **Serving-path performance** — throughput, p50/p99 latency, VRAM footprint at target batch size. Chapter 11.
- **Cross-encoder reranker** (if trained) with matching contract.

### RAG engineer ships

- **Chunking strategy** for the target corpus — chunk size, overlap, splitting boundaries. Must respect the encoder's `max_seq_length`.
- **Vector store** — FAISS / HNSW / Qdrant / Weaviate / ScaNN / pgvector — with the appropriate distance metric matching the encoder contract.
- **Hybrid scoring** — BM25 + dense fusion (reciprocal-rank fusion or weighted linear).
- **Retrieval evaluation** on the target corpus — nDCG@10, MRR, Recall@k on a corpus-specific eval set.
- **Prompt assembly** — how retrieved passages are packed into a generation prompt.
- **End-to-end pipeline** — query → retrieval → rerank → assemble → generate → return.

### Shared

- **Target-corpus retrieval eval set.** Both teams use it. Both teams contribute to it.
- **Reranker.** NLP engineer trains it (chapter 07); RAG engineer decides `top-k` cut-off and integrates.
- **Latency budget.** Both teams have to fit inside it; both teams need to know each other's share.

## The three specific handoffs

Almost every "who owns this" argument in production falls under one of these:

**Handoff 1: the chunker's `max_seq_length`.** The chunker is a RAG-engineer artefact but its correctness depends on the encoder's `max_seq_length`. If the chunker over-chunks (100 tokens each on a 512-token encoder) it wastes recall. If it under-chunks (1024 tokens each on a 512-token encoder) it truncates content. The encoder contract fixes this: chunker MUST target `max_seq_length` minus a small buffer for prefix and special tokens.

**Handoff 2: the task prefix.** Almost every modern encoder needs one. The RAG engineer's retrieval client must apply it at both query time (`"query: "`) and index time (`"passage: "`). The encoder contract fixes this: prefixes are fields, and the retrieval client asserts they match the contract.

**Handoff 3: the reranker candidate list.** The reranker is trained on candidates drawn from a specific retriever distribution (chapter 07). If RAG engineer decides to swap the first-stage retriever (BM25-only → dense → hybrid), the reranker's calibration silently breaks. Ownership: any first-stage change requires re-evaluating the reranker on the new candidate distribution and, if quality drops, retraining. This is a joint decision.

## Where retrieval evaluation lives

Both teams need retrieval evaluation but for *different* reasons:

- **NLP engineer evaluates on MTEB** to validate that the encoder is generally competitive across the retrieval category. This is the *encoder's* evaluation.
- **RAG engineer evaluates on the target-corpus eval set** to validate that the *pipeline* (encoder + chunker + index + reranker) returns useful passages. This is the *system's* evaluation.

Both are necessary. MTEB is not sufficient — a strong-on-MTEB encoder can lose on your specific corpus for chunking or reranker reasons. The corpus eval is not sufficient either — it does not tell you whether the encoder itself is the bottleneck.

When retrieval quality is bad, the diagnostic order:

1. **Is recall@100 (bi-encoder only, no reranker) low on the corpus eval?** If yes, the encoder is the bottleneck. NLP-engineer problem: adapt or replace the encoder (chapter 09/10).
2. **Is recall@100 fine but nDCG@10 low?** If yes, the reranker is the bottleneck. NLP-engineer problem: train a better reranker or fix its candidate-distribution mismatch (chapter 07).
3. **Are both fine but end-to-end quality is bad?** RAG-engineer problem: chunking, prompt assembly, or generation.

Write this decision tree down in a shared place. Argue less at 2 a.m.

## Where the cross-encoder reranker really lives

The reranker is the most-fought-over artefact in the boundary because it looks like it belongs to both sides. Reality:

- **NLP engineer trains it and owns its weights**, because training a cross-encoder is a supervised-learning problem with hard-negative mining and MTEB-reranking eval — squarely inside this module.
- **RAG engineer wires it into the pipeline**, because deciding `top-k=100 → rerank → keep 10` is a retrieval-pipeline shape decision that depends on the corpus, the latency budget, and the downstream generation prompt.

The clean split: NLP engineer publishes a rerank model with a contract (`predict(query, doc) → score`); RAG engineer decides how many candidates to feed it.

## Failure modes that live specifically in the boundary

- **Encoder replaced, index not re-encoded.** New model's vectors are geometrically incompatible with old vectors in the index. Enforce: index is versioned by encoder-id + version.
- **Chunker updated to 1024 tokens; encoder still 512.** Silent truncation on every chunk. Enforce: chunker asserts its target size against the encoder contract's `max_seq_length`.
- **Task prefix drifts.** New retrieval client sends no prefix; old one sends `"query: "`. Silent quality drop. Enforce: retrieval client checks the exact prefix strings against the contract on startup.
- **First-stage retriever swapped from BM25 to dense.** Reranker was trained on BM25 negatives; now it sees dense negatives it never learned to break ties on. Silent quality regression. Enforce: any first-stage change triggers reranker re-evaluation.
- **Similarity function drifts.** Encoder shipped with `cosine`, index stored raw vectors with `dot-product` and no normalisation. Numbers come out; they are subtly wrong. Enforce: retrieval client normalises vectors to match the contract's declared similarity function.

## The team-shape question

Practical corollary: on small teams (1–3 people total), one person often wears both hats. That is fine. What is *not* fine is treating the encoder and the pipeline as one blob — even with one person, keep the artefacts separate: model weights + contract on one axis, pipeline code + retrieval eval on the other. When the team grows and the roles split, the boundary is already drawn.

On larger teams, both roles usually exist. The friction points are exactly the handoffs above.

## Chapter summary

- The encoder contract is the interface: model id, dim, similarity, precision, prefixes, max length. Ship it as YAML alongside the weights.
- NLP engineer owns the encoder and its evaluation; RAG engineer owns the store, chunker, hybrid scoring, and the pipeline.
- Cross-encoder reranker: NLP engineer trains, RAG engineer integrates.
- Retrieval-quality diagnostics run in a specific order: bi-encoder recall → reranker precision → pipeline. Attribute before arguing.
- The three concrete handoffs are chunker `max_seq_length`, task prefix parity, and reranker candidate distribution. Every one of them has to be enforced in code, not in convention.
- Failure modes in this boundary are silent — vectors and scores keep flowing while quality drops. Bind indices to encoder version; assert prefixes at startup; re-evaluate the reranker on every first-stage change.
- On small teams one person may wear both hats, but keep the artefacts separate so the boundary survives team growth.
