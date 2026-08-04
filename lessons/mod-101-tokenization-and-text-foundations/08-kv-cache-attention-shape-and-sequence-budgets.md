# KV-Cache, Attention Shape, and Sequence-Length Budgets

## Motivation

Chapter 07 described *what* the transformer families do. This chapter is about what they cost. Three concepts recur in every later module and every production incident:

1. **Attention shape** — memory and compute grow with the square of the effective sequence length.
2. **The KV cache** — the reason autoregressive generation is fast, and the reason large batches run out of GPU memory.
3. **Sequence-length budgets** — the tokenizer decides how much of your context window you actually get.

Get these three right and you can size, batch, and price NLP workloads without guessing. Get them wrong and you will over-provision GPUs while shipping products that OOM in the field.

## Attention shape and its consequences

Standard self-attention over a sequence of length `L`, hidden dimension `d`, with `h` heads of dimension `d_head = d / h`:

- **Projection compute (Q, K, V):** `O(L · d²)`.
- **Attention scores (Q @ Kᵀ):** produces an `L × L` matrix per head. Compute `O(L² · d)`, memory `O(h · L²)` if materialised.
- **Softmax + weighted V:** `O(L² · d)`.
- **Output projection:** `O(L · d²)`.

Total per-layer forward pass is `O(L · d² + L² · d)`. When `L << d`, the projection term dominates. When `L >> d`, the attention term dominates.

For a modern 7B model (`d = 4096`), the two terms cross at roughly `L ≈ 4k`. Every context beyond that is spent in quadratic territory. This is why long-context LLMs invest heavily in attention-shape optimisations: FlashAttention (Dao et al., NeurIPS 2022; Dao 2023), grouped-query and multi-query attention (Shazeer 2019; Ainslie et al., EMNLP 2023), sliding-window attention (Beltagy et al., 2020; Mistral), and various sparse / recurrent variants.

The tokenizer sits upstream of all of this. A tokenizer that doubles the number of tokens for the same document doubles `L`, and thus roughly quadruples attention cost in the long-context regime.

## The KV cache

### Why it exists

At inference, decoder-only models generate one token at a time. Naïvely, generating token `t+1` requires a full forward pass over tokens `1..t`, which is `O(t² · d)` per new token. Generating a 1000-token completion would cost `O(10⁹ · d)` — unusable at scale.

But because self-attention keys and values depend only on the *previous* tokens (which do not change), you can cache them. For each layer and each head:

```
K_cache: (B, h, L_max, d_head)     # keys for all previously-emitted tokens
V_cache: (B, h, L_max, d_head)     # values for all previously-emitted tokens
```

At each generation step, only the *new* token's `Q`, `K`, `V` need to be computed; the previously cached `K` and `V` are read directly. Per-step compute drops from `O(t² · d)` to `O(t · d)`. Over a full 1000-token generation, this is roughly a 1000× speedup.

Every production decoder-only inference engine implements this: PyTorch `generate()`, `vLLM`, `TensorRT-LLM`, `llama.cpp`, HuggingFace TGI, etc.

### Memory cost

The cache is the dominant memory consumer during long-context generation. Its size:

```
kv_cache_bytes  =  B · L_max · N_layers · 2 · N_kv_heads · d_head · sizeof(dtype)
```

Where the `2` is one factor of K and one of V. For a 7B model (`N_layers = 32`, `N_heads = 32`, `d_head = 128`) in FP16:

- 1 request, 4096 tokens, MHA (32 KV heads): `1 · 4096 · 32 · 2 · 32 · 128 · 2 B ≈ 2.15 GiB`.
- 32 requests, 4096 tokens, MHA: ~68 GiB — larger than the model weights themselves.

This is why **grouped-query attention (GQA)** and **multi-query attention (MQA)** are ubiquitous in production LLMs. LLaMA-2 70B, LLaMA-3, Mistral, and Qwen all use GQA with `N_kv_heads` in the range 4-8, which shrinks the cache by a factor of `N_heads / N_kv_heads`:

- LLaMA-2 70B has 64 query heads and 8 KV heads → 8× KV-cache reduction vs. MHA.

For cross-attention in encoder-decoder models, the KV cache holds the encoder's `K` and `V` — which are computed once per request and cached for the entire decoding loop. Decoder self-attention still grows per generated token.

### What the KV cache does not save

- **Encoder-only models** do not need a KV cache at all. They run a single forward pass.
- **Prefill (prompt processing)** for decoder-only models is still a full parallel forward pass over the prompt, filling the KV cache in one go. Prefill and decode have very different compute profiles — inference engines schedule them differently.
- **Attention over long prompts is still O(prompt_length²)** at prefill time. The KV cache only makes the *subsequent* decode cheaper.

### Practical implications

- **Concurrency ≠ throughput.** Doubling batch size doubles KV-cache memory and can push you into paging or OOM. Modern engines (vLLM's PagedAttention, Ainslie et al.'s reference implementations) treat the KV cache as a paged block store to squeeze more concurrent requests into fixed VRAM.
- **Long context is a memory story, not a compute one.** Beyond a few thousand tokens, KV-cache VRAM dominates and drives batch-size ceilings.
- **Model choice interacts with cache.** For the same parameter count, GQA models serve more concurrent requests than MHA models.

## Sequence-length budgets from the tokenizer's side

Two numbers you should always know for a production NLP system:

1. `L_max` — the model's advertised context window in **tokens**.
2. `tokens_per_unit` — how many tokens your tokenizer produces per character, word, or byte on your actual domain (from chapter 06's fertility metric).

The effective context window in *characters* is `L_max / tokens_per_char`. Sample calculations:

- English news with GPT-4-style byte-level BPE: ~4 chars/token → an 8k context window fits ~32k characters, ~5000-6000 words.
- Same content in Japanese with an English-tuned tokenizer: ~1 char/token → an 8k window fits ~8k characters.
- The same Japanese content with a SentencePiece Unigram tuned on Japanese: ~2-3 chars/token → 16-24k characters.

The tokenizer decision from chapters 02-06 is not just a quality decision — it is directly a *context-window* decision.

### Budgeting a real system

When designing a prompt template or a fine-tuning workflow, budget explicitly:

```
context_window = system_prompt + few_shot_examples + user_input + expected_output + safety_margin
```

Every term is in tokens, measured with the actual tokenizer of the target model. Track:

- The 99th-percentile length of each term across production traffic, not just the average.
- The margin needed for retries or re-ranking passes.
- The extra tokens introduced by chat templates (`<|im_start|>system\n...`).

### Truncation strategies

When inputs exceed budget, common approaches (with trade-offs):

- **Head truncation** — keep the last `L_max` tokens. Preserves recent context; loses framing.
- **Tail truncation** — keep the first `L_max` tokens. Preserves framing; loses conclusions.
- **Middle truncation** — keep both ends, drop the middle. Preserves both boundaries; loses body.
- **Chunk and retrieve** — split into chunks, embed, retrieve the most relevant ones. Adds pipeline complexity but scales far beyond `L_max`.
- **Summarise and prepend** — recursively summarise older context. Loses fidelity; used in agent memory systems.

None is universally right. Choose based on where the *task-relevant* signal sits in your inputs.

## Diagnostics for the field

When a production NLP system misbehaves, this triad of checks often finds the cause:

1. **Length distribution of production requests, in tokens.** If p99 is above the model's context, you have silent truncation happening somewhere. Instrument it.
2. **KV-cache memory per active request.** For a decoder-only model, this is `L · N_layers · 2 · N_kv_heads · d_head · sizeof(dtype)`. If your inference engine's reported max-concurrent-requests is far below what memory should allow, something (fragmentation, per-request overhead, absence of GQA / paged KV cache) is wasting cache slots.
3. **Fertility drift.** Compare fertility of production traffic against training-time fertility. If production text has drifted (new domain, new language mix, new source), your effective context window is smaller than you think.

## Chapter summary

- Attention compute and memory grow quadratically in effective sequence length; the tokenizer sets that length.
- The KV cache turns autoregressive decode from `O(L²)` per token to `O(L)`, at the cost of the largest memory allocation in most inference deployments.
- GQA and MQA reduce KV-cache memory by shrinking the number of KV heads and are the reason modern LLMs can serve high concurrency at long context.
- Sequence-length budgets are a joint property of the tokenizer's fertility and the model's context window; both must be measured on production data, not assumed.
- Chapter 01-08 together form the tokenization + transformer foundation the rest of the module (and the rest of this track) will build on. The exercises put each piece into your hands.
