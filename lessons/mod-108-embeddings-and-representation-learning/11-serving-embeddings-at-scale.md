# Serving Embeddings at Scale

## Motivation

Every quality decision earlier in this module — backbone, pooling, hard-negative mining, MTEB score — matters only if the resulting encoder can hit the throughput and cost target of the system it lives inside. In practice, that target is savage: 50 k–1 M documents to embed at ingest, 10–1000 queries per second at read, sub-100 ms tail latency, and a serving budget that usually cannot justify a $10 k/month single-encoder GPU. This chapter is the last-mile playbook — batching, quantisation, dimension reduction (Matryoshka), specialised inference servers (TEI, ONNX, vLLM), and the observability that keeps a serving path honest.

The rag-engineer track owns the store, the ANN index, and the retrieval pipeline; this module owns the encoder's serving path. Chapter 12 (the module boundary) makes that split explicit.

## The four cost drivers

Any embedding-serving system's cost decomposes into four independent axes:

1. **Encoder forward-pass cost.** Params × input length × batch size. Reducible by quantisation, distillation, and specialised inference runtimes.
2. **Vector dimension × precision.** Storage cost = `N_docs × dim × bytes_per_float`. Reducible by Matryoshka truncation and int8 / binary quantisation.
3. **Search cost.** ANN index lookup — HNSW, IVF, PQ. This is mostly the RAG engineer's concern; the encoder's contract with the index (dim, precision) is yours.
4. **Network + serialisation.** Cost of moving vectors between encoder, store, and application. Batching helps; keeping vectors on GPU as long as possible helps more.

Everything below trades one of these against another.

## Batching: the free lift you probably haven't taken

Encoder GPUs are massively parallel. A batch of 32 sentences runs in roughly 1.5× the time of a single sentence — 21× throughput for 1.5× latency. This is the largest free performance lift in the entire stack, and it is routinely under-exploited.

Reference numbers (rough, MPNet-base on a mid-range GPU):

| Batch size | Latency (ms) | Throughput (sentences/sec) |
|------------|--------------|----------------------------|
| 1          | ~20          | ~50                        |
| 8          | ~25          | ~320                       |
| 32         | ~35          | ~900                       |
| 128        | ~60          | ~2100                      |
| 512        | ~180         | ~2800                      |

The knee of the curve is around batch 128 for base-size encoders. Beyond that, you saturate GPU memory-bandwidth and see diminishing returns.

Two batching regimes at serving time:

- **Ingest batching.** Batch aggressively — 256 or 512 — because ingest is throughput-bound, not latency-bound.
- **Query batching (dynamic batching).** Collect concurrent queries into a batch with a bounded wait (e.g. 10 ms). Trades a small tail-latency penalty for a 10× throughput lift. This is what specialised servers (TEI, Triton) do automatically.

`sentence-transformers`' `model.encode(batch_size=...)` handles the mechanic; production servers handle dynamic batching for concurrent queries.

## Precision: fp16, int8, binary, ubinary

Every embedding vector is a `dim × 32-bit-float` array by default. A billion documents at 768 dimensions is `1e9 × 768 × 4 = 3 TB` of vectors. Precision reduction is the biggest lever on storage cost.

Four precision modes, in order of aggressiveness:

- **fp16 (half precision).** 2× compression, near-zero quality loss. Requires an ANN index that supports fp16 (FAISS does). The default first move.
- **int8 (8-bit signed integer).** 4× compression. Small quality drop (<1 nDCG@10) when quantised well. Requires quantisation-aware calibration or a scale/offset per vector.
- **binary (`{-1, 1}` per bit).** 32× compression. Larger quality drop (~2–5 nDCG@10) for base models; sometimes negligible for well-trained large models. Similarity computed with Hamming distance.
- **ubinary / `{0, 1}` unsigned binary.** Same as binary, unsigned representation. Fastest Hamming implementations.

`sentence-transformers` v2.7+ supports quantisation at `encode` time:

```python
vectors_int8   = model.encode(texts, precision="int8", normalize_embeddings=True)
vectors_binary = model.encode(texts, precision="binary")
```

For the binary/ubinary path, the standard serving pattern is **binary quantisation for the ANN candidate set, then rescoring with fp32 or int8 on the top-K.** This gives you the storage / speed of binary for the recall stage and the precision of fp32 for the ranking stage. `sentence-transformers` `quantize_embeddings` + `semantic_search_hnsw` / `semantic_search_faiss` implement exactly this pattern.

A concrete "cost per million vectors stored" comparison (768-dim):

| Precision | Bytes/vec | GB per 1 M | Approx quality |
|-----------|-----------|------------|----------------|
| fp32      | 3072      | 3.07       | 100 %          |
| fp16      | 1536      | 1.54       | ~99.9 %        |
| int8      | 768       | 0.77       | ~99 %          |
| binary    | 96        | 0.10       | ~95–98 %       |

## Matryoshka Representation Learning

**Matryoshka Representation Learning** (Kusupati et al., ["Matryoshka Representation Learning"](https://arxiv.org/abs/2205.13147), *NeurIPS 2022*) trains a model so that *any prefix* of the output vector is still a useful embedding. `text-embedding-3-large` (3072-dim by default, but usable at 256/512/1024) and `nomic-embed-text-v1.5` (768/512/256/128/64) both ship this.

Practical shape: a Matryoshka model trained with target dimensions `[64, 128, 256, 512, 768]` can be truncated at inference time to any of those without retraining:

```python
v_full   = model.encode(text)                           # 768-dim
v_small  = v_full[:256]                                  # 256-dim, still useful
v_tiny   = v_full[:64]                                   # 64-dim, degraded but usable
```

This gives you a *runtime* quality/cost knob. Storage-critical corpora (hundreds of billions of documents) can use 64-dim; higher-quality-needed slices can use 512-dim. A single-model deployment, multiple corpus tiers.

Training a Matryoshka bi-encoder yourself is `MatryoshkaLoss` in `sentence-transformers` — wrap any base loss and provide the list of target dimensions:

```python
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
base = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, base, matryoshka_dims=[64, 128, 256, 512, 768])
```

Every training step computes the base loss at each target dimension and sums them. The model learns to make its first 64 dims maximally informative, its next 64 supplementary, and so on.

## Compressing further: quantisation + Matryoshka stacked

Matryoshka and binary quantisation compose. A `text-embedding-3-large` vector truncated to 512 dims and binary-quantised is `512 / 8 = 64 bytes` per document — a 384× reduction from the full 3072-dim fp32 vector. Quality trade-off: ~2–5 nDCG@10 vs. the full-precision full-dim version, depending on the corpus.

For any storage-dominated deployment, this stack (Matryoshka truncation → binary quantisation → binary ANN with fp32 rescore on top-K) is the modern default and worth a specific chapter of thought before committing to a giant flat float32 index.

## Serving runtimes

Once the model is trained, the encoder's *runtime* matters as much as the model itself. Options, in rough order of increasing operational complexity:

- **`sentence-transformers` on Python/PyTorch.** The default. Fine for research and low-QPS serving. Not fine at production QPS — the Python overhead per call is significant.
- **ONNX Runtime.** Export the encoder to ONNX; run with `onnxruntime` or `optimum` bindings. 1.5–3× throughput lift over PyTorch, cross-platform (CPU/GPU/Metal). The right first optimisation for CPU-bound serving.
- **[Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference).** Hugging Face's Rust-based inference server for embeddings and reranking models. Dynamic batching, HTTP/gRPC, fp16/int8 kernels. The reference server for GPU-heavy embedding serving.
- **[vLLM](https://docs.vllm.ai/) with embedding models** — vLLM supports embedding models via `LLM(task="embed")`. Not always the right pick for pure encoder-only models (TEI is more specialised) but useful when you already run vLLM for other traffic.
- **[Triton Inference Server](https://github.com/triton-inference-server/server).** NVIDIA's general-purpose server; supports ONNX, TensorRT, and dynamic batching. Bigger operational surface than TEI, more customisable.
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / TensorRT.** NVIDIA's optimising compiler for transformer inference. Highest peak throughput, tightest lock-in.
- **Hosted APIs.** OpenAI, Cohere, Voyage, Together. Zero operational cost, per-token pricing, no adaptation.

The 80/20 stack for a mid-size deployment is *TEI on a GPU with fp16 or int8, behind a thin HTTP wrapper, with dynamic batching enabled*. TEI's headline numbers (500 k+ tokens/sec on an A10) are reproducible in practice.

## Distillation for latency

Chapter 06 covered cross-encoder → bi-encoder distillation for quality. The same technique applies for *latency*: distil a big bi-encoder into a small one. This is how `all-MiniLM-L6-v2` (22 M params) ended up close to MPNet-base (110 M) in quality — it was distilled from the bigger model with MSE-on-embeddings.

Recipe:

```python
from sentence_transformers.losses import MSELoss
teacher = SentenceTransformer("BAAI/bge-large-en-v1.5")           # 335 M params
student = SentenceTransformer("microsoft/MiniLM-L6-H384-uncased")  # 22 M params

# For every text in the training set, precompute teacher_vec = teacher.encode(text)
# Train student with MSELoss(student_vec, teacher_vec)
```

Small quality loss, 10–30× throughput lift. The default move when you need to shave latency without switching architectures.

## The end-to-end serving-latency budget

A worked example for a RAG-adjacent stack, budgeted top-down:

| Stage                                | Budget (ms) | Notes                                  |
|--------------------------------------|-------------|----------------------------------------|
| Network / API roundtrip              | 20          | Client to LB to service                |
| Query preprocess (normalise, prefix) | 2           | Cheap                                  |
| Encoder forward (query)              | 15          | TEI, base encoder, fp16, batch of 1    |
| ANN search (top 100)                 | 20          | HNSW, ~10 M documents                  |
| Reranker forward (100 pairs)         | 100         | MiniLM cross-encoder, batched          |
| Post-process / assemble response     | 3           | Cheap                                  |
| **Total (p50)**                      | **160**     |                                        |
| **Total (p99)**                      | ~350        | Reranker tail dominates                |

Every line item is a knob. If the p99 tail is too high, the biggest wins are (a) drop reranker candidate list from 100 to 20 (chapter 07), (b) binary + Matryoshka on the ANN store, or (c) distil the reranker into a smaller student.

The point of writing out the budget: latency debugging is a lot faster with numbers per stage than without. Instrument every stage from day one.

## Observability contracts

The four numbers you always want visible for an embedding-serving path:

- **Encoder p50 / p95 / p99 latency**, split by input length bucket. Long inputs are systematically slower.
- **Effective batch size.** If dynamic batching is enabled but batches are always size 1, throughput is far below capacity.
- **Truncation rate.** Fraction of inputs that hit `max_seq_length`. Above 5 % is a red flag — you are losing content in the tail.
- **Query-vs-passage prefix parity.** Log the prefix used at query time and at index time. Prefix drift is silent and catastrophic; catch it in observability, not in an incident.

For quality, additionally:

- **Retrieval nDCG@10 (or MRR)** on a small in-house holdout set, computed nightly. Regressions from encoder swaps or index changes show up here first.
- **Cross-lingual Tatoeba F1** (if multilingual) on the same schedule. Same reason.

## Common serving-time regressions

- **Encoder swapped, index not re-encoded.** The new encoder produces incompatible vectors. Ingest and query must use the same encoder version. Vector index → encoder version is a bind you must enforce.
- **Query prefix drift.** Someone changed the retrieval client to send `"user query: "` instead of `"query: "` and the numbers moved silently. Guard by asserting on the exact prefix string at both ends.
- **Batch size dropped from 128 to 32 in prod.** A memory-pressure "fix" that killed throughput. Monitor batch-size histograms.
- **fp16 kernels swapped to fp32 by a runtime upgrade.** Latency doubled, quality unchanged. Pin your runtime version.
- **Binary index used with unrelated model's vectors.** Silent quality collapse. Bind index by model + precision + version.

## Chapter summary

- Cost decomposes into encoder forward pass, vector storage, ANN search, and serialisation. Each has its own knobs.
- Batching is the biggest free lift: 8× throughput for 1.5× latency, up to a saturation knee around batch 128.
- Precision ladder: fp32 → fp16 → int8 → binary. Binary + rescore is the modern default for storage-dominated deployments.
- Matryoshka embeddings expose runtime dimension truncation. Stacks with quantisation for 100–400× compression at small quality cost.
- Serving runtime matters. TEI is the specialist server for embeddings; ONNX + `optimum` is the CPU-friendly fallback; hosted APIs skip the operational cost.
- Distillation from a big bi-encoder to a small one is the latency-cutting move that mirrors chapter 06's quality-boosting move.
- Budget latency top-down. Track p50 / p95 / p99 per stage plus batch size, truncation rate, and prefix parity.
- Bind vector indices to encoder + precision + version. Prefix and version drift are the silent regressions this stack loses to most often.
