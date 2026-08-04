# Long-Context QA: Sliding Window, Long Encoders, and Fusion-in-Decoder

## Motivation

The chapters so far assumed the context fits in a 384–1024 token window. Real documents — 30-page contracts, patient records, product manuals, research papers, transcripts of multi-hour meetings — do not. The question "what is the effective interest rate after the third rate adjustment?" might require reading paragraphs on pages 4, 17, and 32 of a mortgage document, and *no* single window captures all three.

This chapter surveys the four viable strategies for long-context QA, with an emphasis on the engineering trade-offs. All four have production deployments; the correct choice is dataset- and cost-dependent.

## Strategy 1: sliding-window chunking with span aggregation

The baseline. Chapter 02 introduced it as a preprocessing detail for extractive QA; here we lean on it as a *long-context strategy* in its own right.

- Split the document into overlapping chunks (`max_length - question_length`, `stride ≈ max_length / 3`).
- Run the extractive-QA reader on every chunk independently.
- Collect per-chunk candidate spans with their logit scores, keep the top-*k* per chunk (chapter 03).
- Rank candidates from all chunks by `start_logit + end_logit` and return the best.

**Strengths.** Works with any 512-token encoder. No architecture changes. Extractive citation is preserved. Parallelisable across chunks.

**Weaknesses.** Cannot combine evidence across chunks — the answer must live in a single chunk. A question whose answer requires stitching together evidence from pages 4 and 17 is unsolvable in this paradigm.

**When to use.** The answer is a single span in a specific location. Long contract QA, long-form log analysis, extractive citation over a manual.

## Strategy 2: long-context encoders

Replace the 512-token encoder with one designed to attend over thousands of tokens at once. The mechanism is either sparse attention (each token attends only to a local window plus a few global tokens) or approximate attention (low-rank / kernelised).

- **Longformer** (Beltagy, Peters & Cohan, ["Longformer: The Long-Document Transformer"](https://arxiv.org/abs/2004.05150), 2020). Sliding-window self-attention plus a small number of "global" attention positions. Context 4096; the `allenai/longformer-base-4096` checkpoint plugs into `AutoModelForQuestionAnswering` and drop-in-replaces `bert-base`.
- **BigBird** (Zaheer et al., ["Big Bird: Transformers for Longer Sequences"](https://arxiv.org/abs/2007.14062), *NeurIPS 2020*). Similar idea — local + global + random attention — with different sparsity patterns.
- **LongT5** (Guo et al., ["LongT5: Efficient Text-to-Text Transformer for Long Sequences"](https://arxiv.org/abs/2112.07916), *NAACL 2022 Findings*). Encoder-decoder with local + transient-global attention, trained up to 16 k tokens. Drop-in replacement for T5 in the abstractive recipe (chapter 05).

**Strengths.** No pipeline changes — the encoder just accepts a longer input. Cross-chunk reasoning happens implicitly inside the attention layers.

**Weaknesses.** Memory scales roughly linearly (vs. quadratic for full attention), but constants are large — a Longformer-large at 4096 tokens uses more memory per example than a BERT-large at 512. Not all pretrained checkpoints are comparably strong; you inherit the base model's ceiling.

**When to use.** The document is 2–16 k tokens and you want one model to read it end-to-end. Legal, biomedical, and long-form question answering.

## Strategy 3: Fusion-in-Decoder (FiD)

Fusion-in-Decoder (Izacard & Grave, ["Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"](https://arxiv.org/abs/2007.01282), *EACL 2021*) is the canonical retrieval-augmented reader architecture. The idea:

1. Retrieve the top-$N$ passages relevant to the question (retrieval belongs to the `rag-engineer` track; chapter 10).
2. Encode each `(question, passage_i)` pair *independently* with a T5 encoder — producing $N$ separate encoder-hidden-state sequences.
3. Concatenate all $N$ encoder outputs into a single long sequence.
4. Run a single T5 decoder over that concatenated sequence with cross-attention.

The decoder therefore sees all $N$ passages fused into one context, but the encoder never has to attend across passage boundaries — encoder cost stays $O(N \cdot L^2)$ where $L$ is a *single* passage length rather than $O((N \cdot L)^2)$.

**Strengths.** Combines evidence across many passages. Scales gracefully with retrieval depth. Battle-tested on Natural Questions and TriviaQA open-domain settings.

**Weaknesses.** Requires a retriever, so it is only useful when you have an open-domain or multi-document setting. Abstractive by construction — you lose extractive citation unless you add a separate citation head.

**When to use.** Open-domain QA over a corpus. Retrieval-augmented reading over many candidate passages per question.

## Strategy 4: long-context decoder-only LLMs

Modern decoder-only LLMs (GPT-4-class API models, Claude 3-family, Llama 3.1 128k) accept context windows in the tens or hundreds of thousands of tokens. The recipe is the closed-book prompt of chapter 06 with the document(s) inserted into the prompt.

**Strengths.** No fine-tuning. No pipeline. Handles unstructured document mixtures. Best out-of-the-box quality on many tasks.

**Weaknesses.**

- **Cost.** You pay per input token, every time.
- **Latency.** Prefill of a 100 k-token prompt takes many seconds even on top-tier hardware.
- **Lost-in-the-middle** (Liu et al., ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172), *TACL 2024*). Models attend most strongly to the beginning and end of the context; facts buried in the middle are recalled poorly. Placement of the question relative to the document matters.
- **Grounding.** No citation is produced by default. You must add explicit "quote the passage that supports your answer" instructions and validate the quote actually appears.

**When to use.** Rapid prototyping. When a corpus is too small to justify a retrieval pipeline. When the mix of documents varies enough that chunking heuristics keep breaking.

Even when long-context LLMs are viable, retrieval-augmented generation with a smaller context window often *beats* them on cost, latency, and grounding — this is a live area of debate and depends on the specific corpus. Benchmark before committing.

## Choosing a strategy: a decision tree

```
Is the answer always a single span in the source?
├─ YES → is the document > 4k tokens?
│         ├─ NO  → Strategy 1: sliding-window extractive.
│         └─ YES → Strategy 2: long encoder + extractive head.
└─ NO (answer requires paraphrase / aggregation / multi-doc)
    │
    Is the document collection open-ended (many docs per query)?
    ├─ YES → Strategy 3: FiD with retriever from the rag-engineer track.
    └─ NO (a single long document per query)
        │
        Can you afford full-context LLM prompting?
        ├─ YES → Strategy 4: long-context decoder-only LLM.
        └─ NO  → Strategy 2: long encoder-decoder (LongT5, PEGASUS).
```

None of this is set in stone. The most common production pattern in 2025 is *hybrid*: retrieve $N$ passages with a retrieval pipeline, feed them to a long-context LLM with an explicit citation instruction, and fall back to an extractive reader when the LLM abstains.

## Evaluating long-context QA specifically

Beyond the SQuAD-style metrics of chapter 04 and the paraphrase-tolerant metrics of chapter 05, long-context QA needs two additional evaluations:

- **Needle-in-a-haystack** (Kamradt, [Needle In A Haystack tests](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), 2023). Inject a random fact at a random depth in a long distractor document and measure recall. Diagnoses lost-in-the-middle and context-window utilisation.
- **Depth-of-evidence stratification.** Split the eval set by *where* in the document the gold evidence lives (start / middle / end). Report EM and F1 per bucket. A model whose F1 drops from 90 (start) to 40 (middle) has a lost-in-the-middle problem that the aggregate number hides.

Multi-document benchmarks worth knowing:

- **Qasper** (Dasigi et al., ["A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers"](https://arxiv.org/abs/2105.03011), *NAACL 2021*). QA over full scientific papers.
- **NarrativeQA** (Kočiský et al., ["The NarrativeQA Reading Comprehension Challenge"](https://arxiv.org/abs/1712.07040), *TACL 2018*). QA over full books and movie scripts, abstractive.
- **SCROLLS** (Shaham et al., ["SCROLLS: Standardized CompaRison Over Long Language Sequences"](https://arxiv.org/abs/2201.03533), *EMNLP 2022*). A benchmark suite of long-context QA and summarisation tasks.

## Chapter summary

- Four strategies handle long contexts: sliding-window extractive (baseline), long encoders (Longformer, LongT5), Fusion-in-Decoder (retriever + fused encoders + one decoder), and long-context decoder-only LLMs.
- Sliding-window works when the answer lives in a single chunk. Long encoders work when the whole document must be attended over. FiD scales to open-domain multi-passage QA. Long-context LLMs are the "just throw the doc in the prompt" fallback.
- Long-context evaluation needs needle-in-a-haystack and depth-of-evidence stratification on top of standard EM/F1 — aggregate numbers hide lost-in-the-middle regressions.
- Production systems increasingly use hybrids (retrieve + long-context LLM + extractive fallback). Benchmark on your data before committing to a single architecture.
