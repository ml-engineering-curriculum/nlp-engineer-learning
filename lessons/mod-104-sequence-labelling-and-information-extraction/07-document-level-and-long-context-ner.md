# Document-Level and Long-Context NER

## Motivation

Chapters 04–06 assumed the input fits in the encoder's context window: a sentence, a headline, a 200-token paragraph. Real IE tasks routinely exceed that:

- **Clinical discharge summaries** — 2 000–8 000 tokens; diagnoses and medications cluster at the end.
- **Contracts and filings** — 10 000–50 000 tokens; parties are named once at the top, then referenced abstractly for pages.
- **Scientific articles** — 5 000–15 000 tokens; methods and results reference entities introduced in the abstract.
- **News wires with multi-article threads** — coreference across paragraphs that a sentence-level tagger cannot see.

Two problems compound:

1. **Positional-encoding limits.** BERT/RoBERTa/DeBERTa cap at 512 positions. Beyond that, the model has never seen positional embeddings and cannot process the input at all.
2. **Cross-sentence context.** *"The company reported $2B in Q4."* — is "the company" AAPL or GOOG? A sentence-only tagger can never know; a document-level one can.

This chapter covers the three families of solutions — sliding windows, long-context encoders, and coreference-augmented document-level models — and the practical trade-offs among them.

## Solution 1: sliding windows

The pragmatic default. Chunk the document into overlapping windows, tag each window independently, deduplicate span predictions across overlaps.

```python
tokenized = tokenizer(
    document_tokens,
    is_split_into_words=True,
    truncation=True,
    max_length=384,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding=False,
)
```

`return_overflowing_tokens=True` gives you multiple windowed chunks per document. Each chunk has its own `word_ids` and its own aligned labels; the stride overlaps consecutive chunks by `stride` tokens so entities that straddle the boundary appear intact in at least one chunk.

Post-inference, you deduplicate:

```python
from collections import defaultdict

def merge_span_predictions(preds_per_chunk):
    # preds_per_chunk: list of [(start_char, end_char, label, score), ...]
    best = {}
    for chunk_preds in preds_per_chunk:
        for start, end, label, score in chunk_preds:
            key = (start, end, label)
            if key not in best or best[key] < score:
                best[key] = score
    return sorted([(s, e, l, best[(s, e, l)]) for (s, e, l) in best], key=lambda x: x[0])
```

Sliding windows are:

- **Correct.** Every token gets tagged; nothing is dropped.
- **Cheap.** Same encoder, same recipe; ~1.3× training and inference cost for `stride = 0.33 * max_length`.
- **Blind to long-range context.** *"Apple"* in window 5 has no idea that window 1 introduced Apple Inc. as the subject.

For flat NER on domains where entity mentions are self-contained (medications, chemicals, ticker symbols), this is enough. For coreferent or context-dependent NER, it is not.

## Solution 2: long-context encoders

Longformer, BigBird, and their descendants extend the context window to 4 096 or 16 384 tokens by replacing full self-attention with sparse patterns (local + global). The full-attention cost of `O(n²)` becomes `O(n)` or `O(n log n)`.

The two references:

- **Beltagy, Peters & Cohan, ["Longformer: The Long-Document Transformer"](https://arxiv.org/abs/2004.05150), *arXiv 2020*.** Sliding-window local attention + global attention on task-specific tokens. Available as `allenai/longformer-base-4096`.
- **Zaheer et al., ["Big Bird: Transformers for Longer Sequences"](https://arxiv.org/abs/2007.14062), *NeurIPS 2020*.** Random + window + global attention. `google/bigbird-roberta-base`.

Trade-offs:

- **Longer context, real gains.** On document-level NER benchmarks (SciERC, CDR, ChemProt), Longformer/BigBird beat sentence-level baselines by 2–6 F1.
- **Compute cost.** 3–5× slower per token than a base transformer, and the sparse attention kernels have historically been finicky (`transformers` shipped better kernels in 2023 but check throughput on your hardware).
- **Fewer domain-adapted checkpoints.** `Clinical-Longformer` and `Longformer-scivocab` exist but the ecosystem is thinner than for base BERT.

For NER over ≤ 4 096 tokens with genuine cross-sentence dependencies, Longformer is often the right choice. For > 4 096, look at Solution 3 or consider hierarchical models.

## Solution 3: hierarchical and coreference-augmented models

Two research directions worth naming, both aimed at very long documents:

- **Hierarchical encoders.** Encode sentences with a base transformer, then encode sentence-level representations with a second transformer. Zhou et al., ["Hierarchical Attention Networks for Document Classification"](https://aclanthology.org/N16-1174/), *NAACL 2016*, established the pattern; long-doc NER variants apply it by adding a sentence-level context vector to each token representation.
- **Coreference-augmented NER.** Run coreference (chapter 12) first, tag entities within each coreference cluster jointly. Improves recall on abbreviations and pronouns that a sentence-level tagger cannot resolve. See Xu et al., ["A Neural Local Coherence Model for Text Quality Assessment"](https://aclanthology.org/D18-1464/), *EMNLP 2018*, and downstream document-level NER work.

Both are more complex than sliding windows or Longformer. Reach for them when the task requires them, not by default.

## Long-context sub-word alignment

Two subword-alignment concerns that only bite at document scale:

1. **Global attention tokens (Longformer).** For classification-like heads you set `global_attention_mask=1` on `[CLS]`. For token classification, most reference recipes set global attention on **entity boundary candidates** — but on very long documents that becomes memory-heavy. Standard practice: global attention on `[CLS]` and `[SEP]` only; local attention windows do the rest.
2. **Position-ID gaps in sliding windows.** When windows overlap, the *same* token gets different position IDs in different windows. The model handles this correctly, but if you cache activations across windows you must invalidate on overlap boundaries.

## Cross-window entity deduplication — the four strategies

When two overlapping windows both predict entities in their shared region, you need to pick one:

- **Max-score wins.** Keep the higher-softmax-score version. Default; works if the model is calibrated.
- **First-window wins.** Trust the earliest window. Slightly biases toward the beginning; simplest to reason about.
- **Voting.** Predict from every window that covers the span, majority-vote the label. Rarely worth the complexity.
- **Ensemble averaging.** Average logits across windows that cover the token, then decode once. The most principled; adds inference cost proportional to the overlap fraction.

Max-score wins is what most production pipelines run. Log the disagreement rate across windows; if > 5 %, something in your model or data is fragile.

## Document-level evaluation

Here `seqeval` still works, but you need to be careful:

- **Concatenate token / label sequences per document**, not per sentence, before scoring. If you evaluate per sentence, cross-sentence entities are split and scored twice.
- **Report both micro-F1 and macro-F1.** Long documents skew span counts; a single article with 50 entity mentions swamps 10 short articles.
- **Report a "unique-entity" F1** for document-level NER of the *set* of distinct entities mentioned in a document. A tagger that finds the same entity 5 times counts once for this metric. Match to whatever downstream cares about — a knowledge-graph populator cares about unique entities; a highlighting UI cares about every mention.

## Multilingual and CJK long-context concerns

- **XLM-R, mDeBERTa** cap at 512 positions like their monolingual siblings. `xlm-longformer` and `mLongformer` exist but are less mature.
- **CJK NER at document scale.** Chinese and Japanese entity boundaries are ambiguous at the character level; document-level context is often more informative than for space-separated languages, and long-context encoders pay off relatively more.

## When to give up on the tagger entirely

At document scales past 16 000 tokens, or when the schema is truly open-ended, decoder-LLM structured extraction (chapter 10) is often more practical than long-context tagging. The trade is:

- **Tagger + sliding window** — deterministic, calibrated, span-precise; expensive to bolt on cross-sentence context.
- **Long-context tagger (Longformer, BigBird)** — bounded context, still deterministic; ceiling around 4 096–16 384 tokens.
- **Decoder-LLM extraction** — arbitrary context (128 k+), handles document-level and cross-document structure, but harder to calibrate and slower per document.

For a compliance-monitoring pipeline reading 30-page contracts, the current pragmatic answer is often "Longformer + selective LLM escalation for long deals." No single tool wins.

## Failure modes specific to long-context NER

1. **Entity duplication across windows.** Same mention appears twice under different window predictions. Fix: deduplicate on `(start_char, end_char, label)` with score-max.
2. **Truncation invisibility.** `max_length=4096` still truncates. Log the fraction of documents that hit the cap; a 20-page contract will.
3. **Position drift.** Copying a token from a very long document into a downstream system with a stricter position budget corrupts offsets. Store character offsets, not token offsets, in your output schema.
4. **`O` dilution.** In a 4 000-token document with 30 entities, `O`-to-entity token ratio is 100:1. Longformer training with default loss is even more O-dominated than sentence-level NER. Class weighting occasionally helps here.
5. **Memory on the CRF.** If you kept the CRF from chapter 05, its `O(n × K²)` forward-backward is now noticeable. Drop the CRF for long-context, or apply it per sliding-window.

## Chapter summary

- Documents that exceed the encoder's 512-token positional cap are the rule, not the exception, in clinical, legal, and scientific IE.
- Sliding windows with stride-overlap and max-score deduplication are the pragmatic default: same encoder, ~1.3× cost, correct if entities are locally self-contained.
- Long-context encoders (Longformer, BigBird) buy 4 096–16 384-token context at 3–5× compute; useful when entities depend on cross-sentence context.
- Hierarchical and coreference-augmented models handle the very-long or coreference-heavy cases; use only when the task demands it.
- Decoder-LLM extraction (chapter 10) is a practical escape hatch at scales past what long-context tagging supports.
