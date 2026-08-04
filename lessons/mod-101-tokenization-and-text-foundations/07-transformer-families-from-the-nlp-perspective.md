# Transformer Families from the NLP Perspective

## Motivation

The transformer architecture from Vaswani et al., "Attention Is All You Need" (NeurIPS 2017), is one architecture — but three canonical *configurations* dominate NLP work today:

1. **Encoder-only** — BERT (Devlin et al., NAACL 2019), RoBERTa, DistilBERT, DeBERTa, ELECTRA.
2. **Encoder-decoder** — the original 2017 transformer, T5 (Raffel et al., JMLR 2020), BART (Lewis et al., ACL 2020), mBART, PEGASUS, NLLB.
3. **Decoder-only** — GPT (Radford et al., 2018 / 2019), GPT-3 (Brown et al., NeurIPS 2020), LLaMA, Mistral, Falcon, Qwen, and the rest of the modern open-weights LLM zoo.

Each family has different information flow, attention shape, and training objective. Getting the family right — or wrong — for your task is the single largest architectural decision after picking a tokenizer.

This chapter traces a forward pass through each family, notes the attention shape at every step, and gives a decision rubric for which to pick.

## The common core: a single transformer layer

Every family stacks `N` layers of the same primitives:

```
x  →  LayerNorm  →  MultiHeadAttention  →  add  →  LayerNorm  →  FeedForward (MLP)  →  add  →  x'
```

Modern variants shuffle these (pre-norm vs. post-norm) and swap components (GELU → SiLU, LayerNorm → RMSNorm, absolute → rotary or ALiBi positions), but the block signature is unchanged: `x ∈ ℝ^{B×L×d} → ℝ^{B×L×d}`.

The multi-head attention is the load-bearing operation. For each head:

```
Q, K, V  =  x W_Q, x W_K, x W_V              # shape (B, L, d_head)
scores   =  Q @ K.T / sqrt(d_head)           # shape (B, L, L)
scores   =  scores + attention_mask          # mask decides which positions can attend where
weights  =  softmax(scores, dim=-1)          # shape (B, L, L)
out      =  weights @ V                      # shape (B, L, d_head)
```

The **attention mask** is what makes the three families different.

## Encoder-only: bidirectional attention, per-token outputs

### Attention shape

The mask is all-ones (modulo padding). Every token sees every other token:

```
attn_mask[i, j] = 0  for all valid i, j
attn_mask[i, PAD] = -inf
```

### Forward pass

```
tokens                                     # (B, L)
    ↓
input_embed  +  positional_embed           # (B, L, d)
    ↓ × N
LayerNorm → MHA (bidirectional) → add
LayerNorm → MLP → add
    ↓
hidden states                              # (B, L, d)  — one vector per input token
    ↓
task head:
  - MLM: linear back to vocab, cross-entropy on masked positions (BERT pretraining)
  - Classification: pool ([CLS] or mean) → linear
  - Token-level (NER, span extraction): per-token linear → per-token label
```

### Pretraining objectives you will meet

- **Masked language modelling (MLM)** — BERT, RoBERTa. Mask 15% of tokens, predict them.
- **Replaced-token detection** — ELECTRA. Trains a discriminator that decides which tokens were replaced by a small generator; more sample-efficient than MLM.
- **Sentence-order prediction / next-sentence prediction** — BERT (NSP), ALBERT (SOP). Increasingly dropped.

### When to reach for encoder-only

- Classification (topic, sentiment, intent).
- Token labelling (NER, POS, span extraction, IOB tagging).
- Retrieval / bi-encoder embeddings (sentence-transformers models are BERT-family encoders).
- Any task where you want a dense representation of a fixed-length input.
- **Cheap inference** — no autoregressive loop, single forward pass.

### When to avoid

- Any task that produces free-form text longer than a few tokens. Encoders can be adapted (masked LM completion, template infilling) but it is not their natural shape.

## Encoder-decoder: bidirectional over source, causal over target, cross-attention between

### Attention shape

Three separate attention operations per layer of the decoder:

1. **Encoder self-attention.** Bidirectional. Shape as above.
2. **Decoder self-attention.** Causal — each target token can attend only to itself and earlier target tokens.
   ```
   attn_mask[i, j] = 0 if j <= i else -inf
   ```
3. **Cross-attention.** Queries from the decoder, keys/values from the encoder's final hidden states. Shape `(B, L_tgt, L_src)` per head.

### Forward pass (training, with teacher forcing)

```
src_tokens                                 # (B, L_src)
tgt_tokens (shifted right)                 # (B, L_tgt)

  encoder:
     src_embed  →  N × (self-attn bi + MLP)  →  enc_hidden   # (B, L_src, d)

  decoder:
     tgt_embed  →  N × (self-attn causal + cross-attn + MLP) →  dec_hidden  # (B, L_tgt, d)

  lm_head(dec_hidden)                      # (B, L_tgt, V)  → cross-entropy vs tgt_tokens
```

At inference, decoding is autoregressive: the encoder runs once, then the decoder emits one token at a time until an end-of-sequence token or a length cap.

### Pretraining objectives you will meet

- **Span corruption** — T5, mT5. Random spans of the source are replaced with sentinel tokens; the target is the concatenation of those spans.
- **Denoising autoencoder** — BART. Corrupt the input (masking, deletion, permutation) and predict the original.
- **Gap-sentence generation** — PEGASUS. Mask whole important sentences; predict them.

### When to reach for encoder-decoder

- **Sequence-to-sequence transformations** where the input and output are different modalities, languages, or lengths: machine translation, summarisation, question answering with a distinct answer format, style transfer, structured data-to-text.
- **Any task that reads a fixed conditioning input and writes a bounded output.** Cross-attention lets the decoder consult the encoded input at every step without cramming it into the growing context.
- **Multilingual or cross-lingual tasks** benefit from separate encoder / decoder tokenizers or shared multilingual tokenizers (NLLB, mBART).

### When to avoid

- Free-form open-ended generation with no distinct conditioning input. A decoder-only model is simpler and cheaper.
- Latency-critical single-shot classification. The two-stack architecture is overkill.

## Decoder-only: causal attention over a single stream

### Attention shape

Self-attention with a causal mask, same shape as the decoder self-attention above:

```
attn_mask[i, j] = 0 if j <= i else -inf
```

There is no encoder and no cross-attention.

### Forward pass

```
tokens                                     # (B, L)
    ↓
input_embed  +  positional / rotary / ALiBi bias   # (B, L, d)
    ↓ × N
LayerNorm → causal MHA → add
LayerNorm → MLP → add
    ↓
lm_head(hidden)                            # (B, L, V) — per-token next-token distribution
```

Training is next-token prediction across the entire sequence in parallel (teacher-forced); inference generates one token at a time, appending each step's output to the input.

### Pretraining objective

Next-token prediction (causal LM) on huge web-scale corpora. There is nothing else. This is the entire training story of GPT-2, GPT-3, LLaMA, Mistral, Falcon, and the rest.

### When to reach for decoder-only

- **Open-ended generation.** Chat, code, long-form writing.
- **Instruction-tuned assistants.** All modern chat LLMs are decoder-only, instruction-tuned + preference-optimised.
- **Few-shot / in-context learning.** Prompt with examples, no fine-tuning. This is a natural fit for decoder-only models trained at scale.
- **Any task you can rephrase as "continue this text"** — including many classical NLP tasks, at the cost of higher inference cost.

### When to avoid

- Cheap classification / NER / retrieval at high QPS — an encoder is 10-100× cheaper per query for the same task.
- Any task where a bidirectional encoding of the input genuinely helps (structural understanding, dense retrieval).

## Family decision rubric

Given a task, ask in this order:

1. **Do I need to generate free-form text?**
   - No → encoder-only (or a specialised head on an encoder).
   - Yes → step 2.
2. **Is there a distinct conditioning input that the output must faithfully transform (translation, summarisation, data-to-text)?**
   - Yes → encoder-decoder.
   - No → decoder-only.
3. **Do I need in-context learning (few-shot) or instruction-following without fine-tuning?**
   - Yes → decoder-only, large enough to have learned the capability.
   - No → the smaller of the two viable choices from step 2.
4. **What is my latency / cost budget?**
   - Encoder-only is cheapest; decoder-only with KV-cache is the next; encoder-decoder is heaviest per generated token.

The rubric is a starting point, not a mandate. Cross-check with a small experiment before committing large compute.

## Chapter summary

- All three families share the same transformer block; they differ in the attention mask (bidirectional vs. causal) and in whether cross-attention exists.
- Encoder-only: bidirectional, per-token outputs, best for classification / NER / retrieval.
- Encoder-decoder: bidirectional source, causal target, cross-attention between; best for translation and summarisation.
- Decoder-only: causal self-attention; best for open-ended generation and in-context learning.
- Chapter 08 zooms in on what the causal / cross-attention masks imply operationally — KV-cache, memory, and sequence-length budgets.
