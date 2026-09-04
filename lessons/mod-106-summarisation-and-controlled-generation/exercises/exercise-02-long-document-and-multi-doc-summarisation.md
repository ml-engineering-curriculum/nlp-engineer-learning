# exercise-02: Long-Document and Multi-Document Summarisation

**Estimated effort:** 3 hours

## Objective

Push past the 1024-token single-doc recipe from exercise-01 into two settings where the input does not fit the model's context window: a single very long document (a scientific paper, a bill, a government report) and a cluster of related documents that must be fused into one summary. Implement and compare **chunk-and-fuse**, **hierarchical extract-then-abstract**, and **long-context encoder-decoder** strategies from chapter 04; add **MMR-based dedup** and **skipped-source coverage** for the multi-doc side from chapter 05.

## Prerequisites

- Chapters [04](../04-long-document-summarisation.md) and [05](../05-multi-document-summarisation.md); exercise-01 recipe for a working single-doc summariser.
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `rouge_score`, `bert_score`, `sentence-transformers` (for MMR / clustering), `nltk` (`punkt` for sentence tokenisation).
- A GPU with ≥ 16 GB VRAM for LED / LongT5 / PEGASUS-X at 4 k–16 k tokens. CPU is only workable for the chunk-and-fuse baseline on `-base` checkpoints.

## Datasets

Pick one long-doc dataset and one multi-doc dataset:

**Long-document (pick one):**

- **GovReport** — Huang et al., ["Efficient Attentions for Long Document Summarization"](https://arxiv.org/abs/2104.02112), *NAACL 2021*. `datasets.load_dataset("ccdv/govreport-summarization")`. Mean source ~9 k tokens, mean target ~550. The workhorse long-doc benchmark.
- **arXiv** — Cohan et al., ["A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents"](https://arxiv.org/abs/1804.05685), *NAACL 2018*. `datasets.load_dataset("ccdv/arxiv-summarization")`. Scientific papers with abstracts as targets; ~6 k tokens source.
- **BillSum** — Kornilova & Eidelman, ["BillSum"](https://arxiv.org/abs/1910.00523), *EMNLP 2019 W/S*. `datasets.load_dataset("billsum")`. US congressional bills; ~1.5 k–5 k tokens source.

**Multi-document (pick one):**

- **Multi-News** — Fabbri et al., ["Multi-News"](https://arxiv.org/abs/1906.01749), *ACL 2019*. `datasets.load_dataset("alexfabbri/multi_news")`. 2–10 news articles per cluster, joined by `|||||`. The standard MDS starting point.
- **Multi-XScience** — Lu et al., ["Multi-XScience"](https://arxiv.org/abs/2010.14235), *EMNLP 2020*. Related-work section generation from a target abstract plus its cited references. Harder and more abstractive than Multi-News.

Recommendation: **GovReport + Multi-News.** GovReport exposes middle-document collapse fastest; Multi-News is well-instrumented and has strong published baselines.

## Problem statement

### Part A — Chunk-and-fuse (long-doc baseline)

Implement the map-reduce recipe from chapter 04 for your long-doc dataset:

```
chunks   = sliding_windows(source, size=800 tokens, overlap=100 tokens)
partials = [summariser(c) for c in chunks]      # BART-large-cnn or PEGASUS-XSum, zero-shot
final    = summariser(" ".join(partials))       # same model, second pass
```

Requirements:

- Use a **stock 1024-token summariser** (`facebook/bart-large-cnn` or `google/pegasus-xsum`) — no fine-tuning in this part.
- Enforce a per-chunk target length that is proportional to the chunk length: `chunk_target = max(30, source_target * (chunk_len / source_len))`.
- Also implement a **refine** variant (sequential: `running = summariser(chunk_0)`; then for each next chunk, `running = summariser(running + chunk_i)`).

Report on ≥ 300 dev examples: ROUGE-1/2/L, BERTScore-F1, wall-clock latency per example.

Save as `chunk_and_fuse.py`.

### Part B — Hierarchical extract-then-abstract

Implement the two-stage pipeline from chapter 04:

1. **Extract** the top-$k$ salient sentences with either an unsupervised extractor (LexRank via `sumy`) or a supervised one (BERTSumExt if pretrained weights are available; otherwise fall back to LexRank). Budget: 768 tokens of extracted content.
2. **Abstract** the extracted content with the same stock summariser from Part A.

Report the same metric panel on the same 300 dev examples so Parts A / B / C are directly comparable.

Save as `hierarchical.py`.

### Part C — Long-context encoder-decoder

Fine-tune (or evaluate zero-shot, if compute is tight) **one** of:

- `allenai/led-large-16384` — Longformer-Encoder-Decoder at 16 k tokens.
- `google/long-t5-tglobal-base` — LongT5 with transient-global attention at 4 k tokens.
- `google/pegasus-x-large` — PEGASUS-X at 16 k tokens.

Fine-tuning recipe (if you fine-tune):

- 3 epochs, effective batch 4 with gradient accumulation, LR `1e-4` (T5-family) or `3e-5` (BART/LED/PEGASUS-family).
- `max_source_length` = your chosen context; `max_target_length` = dataset-typical (`≈ 600` for GovReport, `≈ 250` for arXiv, `≈ 200` for BillSum).
- `bf16` (or `gradient_checkpointing=True` if OOM); `predict_with_generate=True`, `generation_num_beams=4`.
- For LED: set `global_attention_mask` on the first token (`<s>`) — this is the LED convention.

Report the same metric panel plus GPU memory used.

Save as `long_context.py`.

### Part D — Depth-of-evidence stratification (long-doc)

Aggregate ROUGE hides the middle-document collapse chapter 04 warns about. For 200 dev examples, do a **coverage stratification**:

- Approximate the "supporting region" of each gold summary sentence by finding its highest-ROUGE-L single source sentence.
- Bucket support locations into `[first_quartile, middle_two_quartiles, last_quartile]` of the source.
- For each of your three systems (A/B/C), report ROUGE-L broken down by support-location bucket.

Expected finding (from chapter 04): chunk-and-fuse loses coverage in the middle two quartiles more than the other two.

Present as a 3-row × 3-column Markdown table.

### Part E — Multi-doc pipeline

Pick your multi-doc dataset. Implement **two** MDS systems:

- **Extract-then-abstract with MMR dedup** (chapter 05 recipe).
  1. Tokenise every document into sentences; tag each `(doc_id, sentence)`.
  2. Embed sentences with `sentence-transformers/all-mpnet-base-v2`.
  3. Iteratively pick sentences by MMR (λ = 0.7) against the query "summary of the cluster" (or the cluster centroid embedding) until you hit a 1500-token budget.
  4. Fuse the picked sentences with BART-large-cnn.
- **PRIMERA** — `allenai/PRIMERA`. Concatenate documents with the `<doc-sep>` separator; run zero-shot or a short fine-tune (3 epochs).

Requirements:

- Track which documents each output sentence's picked sources came from (Part F depends on it).
- Cap total input at 4 k tokens (MMR pipeline) or 4096 tokens (PRIMERA-Multi-News default).

### Part F — MDS-specific evaluation

On ≥ 200 dev examples for each system in Part E, report:

- **ROUGE-1/2/L, BERTScore-F1** — the standard panel.
- **Redundancy** — mean pairwise ROUGE-L between *output* sentences. Lower is better.
- **Skipped-source rate** — for each cluster, the fraction of source documents that have zero output sentences supported by them (approximate "support" by max ROUGE-L ≥ 0.3 between an output sentence and any sentence of the source doc). Lower is better.

Present as a Markdown table alongside the long-doc results.

### Part G — Write-up

A 600–900 word `README.md` covering:

- The three long-doc systems and the MDS pipeline you built.
- The metric table with ROUGE/BERTScore, redundancy, skipped-source, and depth-of-evidence stratification.
- Which system won on each axis (quality, coverage, cost) and why you think so.
- One qualitative example per long-doc system: paste the source URL / ID, the gold summary, and the three system outputs side by side. Point out one concrete failure mode of each.
- One "what next" idea (e.g., stacking hierarchical + long-context, adding a faithfulness gate, replacing MMR with cluster-based dedup).

## Starter guidance

- **Token counting matters.** Use `AutoTokenizer` for the target model, not `str.split()`. A 2 k-word document can be 3 k or 5 k tokens depending on tokeniser.
- **`sliding_windows` is not `str.split()`.** Chunk at *sentence* boundaries whenever possible — mid-sentence cuts hurt both parts of the pipeline. `nltk.sent_tokenize`, then greedy pack sentences into windows.
- **LED needs `global_attention_mask`.** Setting global attention on `<s>` (the first token) is the LED default. Skipping this gives silently-worse quality.
- **PEGASUS-X and LongT5 want `bf16`, not `fp16`.** `fp16` occasionally NaNs on long inputs; `bf16` is the safer default on Ampere+ hardware.
- **MMR needs L2-normalised embeddings.** `sentence-transformers` already normalises by default; if you swap encoders, check.
- **PRIMERA `<doc-sep>`.** The literal string `"<doc-sep>"` is a real token in PRIMERA's vocabulary. Do not accidentally split it during tokenisation.
- **Multi-News uses `|||||` as its document separator.** Split on that first, then re-join with `<doc-sep>` for PRIMERA.
- **Refine is slow.** Sequential, no parallelism. Budget accordingly.

## Acceptance criteria

- [ ] `chunk_and_fuse.py` implements both map-reduce and refine variants; reports metrics on ≥ 300 dev examples with latency.
- [ ] `hierarchical.py` implements LexRank (or BERTSumExt) + abstract fusion; same metric panel.
- [ ] `long_context.py` runs a long-context encoder-decoder (LED, LongT5, or PEGASUS-X); same metric panel plus GPU memory.
- [ ] Depth-of-evidence stratification table (3 systems × 3 buckets) reported.
- [ ] Two multi-doc systems (MMR + fusion; PRIMERA) evaluated with ROUGE, BERTScore, redundancy, and skipped-source rate.
- [ ] 600–900 word write-up with the metric tables, one qualitative example per long-doc system, and one "what next" idea.

## Stretch goals

- **Chain-of-density on top of chunk-and-fuse.** Run 3 iterations of Adams et al.'s chain-of-density prompt against a strong instruction-tuned LLM (Llama-3.1-70B-instruct, Mistral-Large, or an API model). Compare density (entities per token) against your Part A / B / C outputs.
- **Recursive summarisation for book-length inputs.** Concatenate 5–10 GovReport documents into a synthetic 40 k-token super-document, then implement Wu et al.'s recursive tree from chapter 04. Measure how much quality degrades at each level.
- **Needle-in-a-haystack.** Inject a distinctive sentence at a random depth into a long document; check whether each of your three long-doc systems reproduces it. Report recall by depth quartile.
- **Contradiction handling in MDS.** Manually construct 20 Multi-News clusters where source articles disagree on a numeric or entity fact. Run both MDS systems; classify the output as "resolved with attribution", "silently picks one", or "contradicts itself".
- **Fine-grained citation for MDS.** Extend the MMR-pipeline output to include `(doc_id, sentence_id)` citations for every claim in the summary. Validate by NLI (chapter 11) that the citation entails the claim.
