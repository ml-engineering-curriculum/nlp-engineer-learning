# Long-Document Summarisation: Chunk-and-Fuse, Hierarchical, and Long Encoders

## Motivation

The recipe from chapter 03 assumed your document fits in a 512- or 1024-token window. Legal contracts, scientific papers, meeting transcripts, patient records, and government reports do not. GovReport documents (Huang et al., 2021) average ~9 k tokens; arXiv scientific papers average ~6 k; a two-hour meeting transcript can hit 20 k. Naively truncating to 1024 tokens loses the vast majority of the source and produces summaries that miss the point entirely — often literally, since scientific-paper conclusions live at the end.

This chapter covers the four strategies that scale abstractive summarisation past a 1024-token window. They are not mutually exclusive; production long-document summarisers usually stack two.

## Strategy 1: chunk-then-fuse (map-reduce)

The baseline. Split the document into windows of $L$ tokens with a small overlap, summarise each chunk independently with a stock encoder-decoder, then summarise the concatenation of chunk summaries.

```python
def chunk_then_fuse(document, chunk_summariser, fuse_summariser,
                    chunk_size=800, overlap=100):
    chunks   = sliding_chunks(document, size=chunk_size, overlap=overlap)
    partials = [chunk_summariser(c) for c in chunks]
    return fuse_summariser(" ".join(partials))
```

This is what LangChain calls "map-reduce" and LlamaIndex calls "compact then refine". It works surprisingly well because the two stages have different jobs:

- The chunk summariser is a fluent local summariser (BART-large-cnn or PEGASUS).
- The fuse summariser is a *shorter-input* summariser that consolidates chunk summaries into a coherent whole.

**Strengths.** Model-agnostic — any 1024-token summariser plugs in. Parallelisable across chunks. No architecture changes.

**Weaknesses.** Loses cross-chunk context: a claim on page 3 that is qualified on page 30 becomes an uncontextualised claim in the chunk summary. Redundancy in the fused stage — three chunks might each summarise the same recurring theme.

Two variants that address the weaknesses:

- **Refine (sequential).** Instead of fusing all partials at once, iterate through chunks: start with the first chunk's summary, then for each subsequent chunk feed both the running summary and the new chunk to the model and ask it to *refine* the summary. Better cross-chunk continuity; strictly sequential (no parallelism).
- **Rolling window with query.** Keep a running summary and slide the window; useful for streaming inputs (live meeting summarisation).

## Strategy 2: hierarchical (extract-then-abstract)

Insert an extractive stage between the source and the abstractive summariser:

1. Run an extractive summariser (BERTSumExt, TextRank, or an oracle labeller trained against a gold summary — chapter 02) to select the top-$k$ salient sentences from the full document.
2. Feed the extracted sentences (which fit in 1024 tokens) to a stock abstractive summariser.

```python
def hierarchical(document, extractor, abstractor,
                  extract_budget=768):
    sents  = sent_tokenise(document)
    picked = extractor(sents, budget=extract_budget)   # ordered subset
    return abstractor(" ".join(picked))
```

**Strengths.** The extractive stage can attend to the whole document at low cost (linear in sentence count). The abstractive stage sees only the salient content, so it does not need to be long-context. Very cheap to serve.

**Weaknesses.** Ceiling is bounded by the extractive stage — anything the extractor misses cannot appear in the summary. Requires a decent extractor; if you train one supervised, you need document-level oracle labels.

Hierarchical is the pattern behind PRIMERA (Xiao et al., 2022) for multi-document work and many production meeting summarisers.

## Strategy 3: long-context encoder-decoders

Replace the 1024-token encoder-decoder with one designed for long inputs. Two architectural families:

- **Sparse-attention encoder-decoders.** Local + global attention patterns; scale linearly in sequence length.
- **Hierarchical / pooling encoders.** Encode chunks locally, pool, and let the decoder cross-attend.

The checkpoints you should know:

| Model                                              | Context (train) | Notes                                                                                                          |
|----------------------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------------|
| `allenai/led-base-16384` / `-large-16384`          | 16 k            | Longformer-Encoder-Decoder — BART with Longformer sliding-window attention.                                    |
| `google/long-t5-tglobal-base` / `-large`           | 4 k–16 k        | LongT5 with transient-global attention. Text-to-text — plugs into the T5 recipe.                               |
| `google/pegasus-x-base` / `-large`                 | 16 k            | PEGASUS with block sparse attention. Phang, Zhao & Liu (2023).                                                 |
| `google/bigbird-pegasus-large-arxiv` / `-pubmed`   | 4 k             | PEGASUS with BigBird sparse attention; fine-tuned on scientific summarisation.                                 |

Practical default for long-doc summarisation: **LongT5-large-tglobal** at 4 k tokens for cost-sensitive; **PEGASUS-X-large** or **LED-large-16384** at 16 k when you need the full range.

**Strengths.** No pipeline. Cross-document context lives inside the attention layers. Often the strongest end-to-end quality when your documents fit in the model's train-time context.

**Weaknesses.** Memory scales linearly, but constants are large — a LED-large at 16 k tokens uses considerable GPU memory per example. Not all long checkpoints have been fine-tuned on your target domain; you may need to fine-tune. Beyond the train-time context, extrapolation is unreliable.

## Strategy 4: long-context decoder-only LLMs

The "just throw the doc in the prompt" fallback. GPT-4-class models, Claude 3, Llama 3.1 all accept context windows in the tens or hundreds of thousands of tokens.

**Strengths.** Zero fine-tuning. Handles unstructured document mixtures. Strong out-of-the-box quality on many tasks.

**Weaknesses.**

- **Cost.** You pay per input token per call.
- **Latency.** Prefill of a 100 k-token prompt takes many seconds even on top-tier hardware.
- **Lost in the middle** (Liu et al., 2024). Decoder-only models attend most strongly to the beginning and end of the context; facts buried in the middle are recalled poorly. Placement of the summary instruction matters.
- **Grounding.** No citation by default. You must instruct the model to quote source passages and validate that the quotes appear.

Even when a long-context LLM is technically viable, chunk-and-fuse or hierarchical with a smaller model often wins on cost, latency, and grounding. Benchmark before committing.

## Choosing a strategy

```
Is the document > your best long encoder's train-time context?
├─ YES → Strategy 1 (chunk-and-fuse) or Strategy 4 (long-context LLM).
└─ NO  → is the document > 1024 tokens?
         ├─ YES → Strategy 3 (long encoder) if you can fine-tune;
         │         Strategy 2 (hierarchical) if you cannot.
         └─ NO  → Chapter 03 recipe. Do not overbuild.

Do you need extractive citations for compliance / trust?
├─ YES → Force Strategy 2 (hierarchical extract-then-abstract).
└─ NO  → any of the above.
```

The most common production pattern in 2025 is *hybrid*: split the document into large chunks, run a long-context encoder-decoder or LLM on each, then fuse. Two of the four strategies stacked.

## Chain-of-density and iterative refinement

Adams et al., ["From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting"](https://arxiv.org/abs/2309.04269), *2023* introduced an inference-time trick worth knowing:

1. Generate an initial short summary.
2. Prompt the model to *rewrite* the summary at fixed length, adding one new entity.
3. Repeat 3–5 times.

The result is a "denser" summary — more entities per token — without changing the model or fine-tuning. Useful for iterative refinement over long documents where the first-pass summary skips over important entities. Requires a strong instruction-tuned decoder-only model; less relevant to a fine-tuned encoder-decoder pipeline.

## Recursive summarisation for very long inputs

For book-length inputs (100 k+ tokens), Wu et al., ["Recursively Summarizing Books with Human Feedback"](https://arxiv.org/abs/2109.10862), *2021* laid out the pattern:

1. Split the book into chapters.
2. Summarise each chapter with a stock summariser.
3. Concatenate chapter summaries and split *those* into windows; summarise each window.
4. Repeat until the concatenation fits in one window; produce the final summary.

The tree of summarisations is $O(\log_L n)$ deep for input length $n$ and window size $L$. Every level compresses by a factor of $\sim 10$–$20$. Faithfulness degrades at every level, so this is not a free strategy — you need a faithfulness gate (chapter 11) or human review at each level.

## Evaluation for long-document summarisation

Everything from chapter 10 (ROUGE, BERTScore) still applies, but three additional evaluations are important:

- **Coverage stratification.** For each summary, check whether the gold answer's supporting sentence(s) appear in the source region the model actually attended to. If your chunk-and-fuse summariser misses page 30's qualification of page 3's claim, this shows up as low coverage on late-document evidence.
- **Needle-in-a-haystack** (Kamradt, 2023). Inject a random fact at a random depth in a long distractor document and measure recall. Diagnoses lost-in-the-middle for LLM-based summarisers.
- **Depth-of-evidence F1.** Split your dev set by *where* in the source the gold-supporting sentences live (first quartile, middle two quartiles, last quartile). Report faithfulness and ROUGE per bucket. Middle-document collapse is common.

Long-document benchmarks worth knowing (also listed in `resources.md`):

- **arXiv / PubMed** — scientific papers with abstracts as targets.
- **GovReport** — US government reports; average ~9 k source tokens, ~500 target tokens.
- **BillSum** — US bills to summary.
- **BigPatent** — patent text to abstract.
- **QMSum** — query-focused meeting summarisation.
- **SCROLLS** — a benchmark suite of long-context QA and summarisation tasks.

## Chapter summary

- Long-document summarisation has four viable strategies: chunk-and-fuse (baseline), hierarchical extract-then-abstract, long encoder-decoders (LongT5, LED, PEGASUS-X), and long-context decoder-only LLMs.
- Chunk-and-fuse works with any 1024-token summariser and is trivially parallel; it loses cross-chunk context. Refine and rolling variants restore some of it at a latency cost.
- Hierarchical is bounded above by the extractive stage's ceiling but is cheap to serve and preserves citation.
- Long-context encoder-decoders remove the pipeline at the cost of memory and a narrower checkpoint zoo. LongT5, LED, PEGASUS-X are the workhorse choices.
- Long-context decoder-only LLMs work but pay in cost, latency, lost-in-the-middle, and lost grounding. Benchmark before committing.
- Evaluate with depth-of-evidence stratification and needle-in-a-haystack in addition to aggregate ROUGE — the aggregate hides middle-document collapse.
