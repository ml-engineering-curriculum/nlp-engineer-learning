# Multi-Document Summarisation

## Motivation

Single-document summarisation compresses one input into one summary. Multi-document summarisation (MDS) compresses *many* inputs — typically related but not identical — into a single coherent summary. Every "here are the last 20 news articles about X" digest, every competitive-intelligence brief, every "summarise this incident from Slack, PagerDuty, and the postmortem doc" is an MDS problem.

MDS is not just single-doc summarisation with a bigger input. It has three failure modes single-doc systems never see, and its evaluation is genuinely harder.

## What makes MDS distinctive

The three MDS-specific failure modes:

1. **Redundancy.** If ten articles all say "the launch failed at 09:00 UTC", a naive concatenation produces a summary that says the same fact ten times.
2. **Contradiction.** Article A says the outage lasted 20 minutes, article B says 45 minutes. The summariser must either resolve the conflict, report both with attribution, or abstain.
3. **Attribution / source tracking.** Users of an MDS system usually want to know *which* source each claim comes from. Single-doc systems get citation for free (the source is the only source); MDS systems must track it.

A fourth issue is practical rather than conceptual: multi-document inputs quickly outgrow the context window. A "digest of today's 20 news articles" easily hits 30 k tokens. The long-context strategies from chapter 04 apply, but with the added constraint that the fusion step must be *aware* of source boundaries.

## Two architectural families

MDS systems generally follow one of two blueprints:

### 1. Extractive-then-abstractive (with dedup)

The canonical multi-doc pipeline:

1. **Cluster or select** salient sentences across the document set, with a dedup pass that suppresses near-duplicates. Standard techniques: sentence clustering (k-means on embeddings), MMR (Carbonell & Goldstein, 1998), or a coverage-aware extractive summariser like MatchSum (Zhong et al., 2020).
2. **Fuse** the selected sentences with a stock abstractive summariser (BART, PEGASUS, LongT5).
3. **Attribute** each output sentence back to the source sentence(s) it was fused from — either via an offline alignment step or via a citation-tuned decoder (chapter 09).

```python
def mds_extract_then_abstract(docs, extractor, dedup, abstractor,
                               budget_tokens=1500):
    all_sents = [(doc_id, s) for doc_id, doc in enumerate(docs)
                 for s in sent_tokenise(doc)]
    scored    = extractor(all_sents)              # (doc_id, sent, score)
    picked    = dedup(scored, budget=budget_tokens)  # MMR / clustering
    fused     = abstractor(" ".join(s for _, s, _ in picked))
    return fused, picked                          # keep picks for citation
```

**Strengths.** Model-agnostic. Handles arbitrary numbers of documents. Attribution is straightforward via the extractive stage. Redundancy is handled by dedup.

**Weaknesses.** Ceiling bounded by extractor. Fusion stage sees sentences out of local context, so it can produce jarring transitions.

### 2. Long-context multi-doc encoders

Pretrained encoder-decoders designed for multi-document input. The canonical example is **PRIMERA** (Xiao et al., ["PRIMERA: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization"](https://arxiv.org/abs/2110.08499), *ACL 2022*), which extends Longformer-Encoder-Decoder with a multi-doc pretraining objective: mask the "entity-pyramid" sentences across a related-document cluster and reconstruct them.

The key input trick: concatenate documents with a special `<doc-sep>` token between them, and use *global* attention on that separator token so the encoder can attend across document boundaries efficiently.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("allenai/PRIMERA")
model     = AutoModelForSeq2SeqLM.from_pretrained("allenai/PRIMERA")

DOC_SEP = "<doc-sep>"
inp = DOC_SEP.join(documents)
# PRIMERA's global-attention mask is applied to <doc-sep> positions
```

**Strengths.** No pipeline. Handles cross-document context implicitly.

**Weaknesses.** Requires the whole input to fit in the encoder's context (16 k for PRIMERA). No explicit source attribution — you need to add a citation head or run a post-hoc alignment step.

## Handling redundancy: MMR and clustering

Maximal Marginal Relevance (Carbonell & Goldstein, 1998) is the extractive-summarisation classic for balancing relevance and novelty:

$$
\text{MMR}(s) = \lambda \cdot \text{sim}(s, \text{query}) - (1-\lambda) \cdot \max_{s' \in S} \text{sim}(s, s')
$$

Iteratively select the sentence with the highest MMR score against the already-selected set $S$. $\lambda$ trades off relevance vs. novelty; $0.7$ is the traditional default.

MMR is trivially applicable to MDS: use "the concatenated set of all documents" as the pool, embed each sentence with any sentence encoder, and iterate. Ship it as your redundancy baseline.

Cluster-based dedup — k-means or agglomerative on sentence embeddings, take one representative per cluster — is a mild alternative. MMR usually wins because it does not require pre-choosing $k$.

## Handling contradiction: two acceptable strategies

There is no universal right answer for how a summariser should treat conflicting sources. Two acceptable strategies:

1. **Report and attribute.** "Sources reported outage durations of 20 minutes (article A) and 45 minutes (article B)." Requires the summariser to produce inline citations — see chapter 09 on structured outputs.
2. **Prefer the most-supported claim.** Cluster claims by semantic content, take the modal position. Reasonable for news digests where you trust majority signal; catastrophic for early-breaking stories.

Both strategies require an explicit *claim-extraction* step before fusion, either via a fact-decomposition prompt (FactScore-style, Min et al., 2023) or via a rule-based extractor for numeric / entity claims. The claim extraction is the hard part; the reconciliation is easy once the claims are aligned.

## Handling attribution: citations from the ground up

Users of MDS systems generally want to know which source each claim came from. Three levels of rigour:

- **Document-level citation.** Every output sentence is tagged with the source document(s) that contributed to it. Cheap. Implementable at fusion time by tracking which extracted sentences from which document ended up in the fusion input.
- **Sentence-level citation.** Every output sentence is tagged with the specific source sentence(s). Requires an alignment step — either at training time (constrained decoding on the specific source spans, see chapter 08) or at post-processing (NLI-based sentence alignment).
- **Fine-grained citation.** Every noun phrase or numeric claim gets its own citation. Requires either fine-tuning on citation-annotated data or a post-hoc claim-attribution pipeline.

Most production systems ship document-level. Sentence-level is achievable with careful engineering; fine-grained is a research problem.

## Datasets

- **Multi-News** (Fabbri et al., 2019). ~56 k news events, each with 2–10 source articles and a human-written summary that fuses them.
- **WikiSum** (Liu et al., 2018). Wikipedia articles as summaries of their cited references — an extreme long-input multi-doc benchmark.
- **DUC 2004 / TAC 2008** — classical MDS benchmarks with human summaries. Small but influential.
- **QMSum** (Zhong et al., 2021). Query-focused meeting summarisation — one dialogue split across many turns can be treated as multi-doc.
- **ArXiv-Multi and Multi-XScience** — scientific multi-doc summarisation.

For prototyping: Multi-News is the standard starting point. It is news-domain, so beware of LEAD-N-style extractive biases baked in.

## Evaluation for MDS

Standard ROUGE / BERTScore apply to the summary itself. Two MDS-specific evaluations:

- **Redundancy.** Measure the mean pairwise ROUGE-L between sentences of the *output* summary. Lower is better — a summary that repeats itself scores high here.
- **Source coverage.** For each source document, check whether at least one output sentence is supported by it (NLI entailment, chapter 11). "Skipped-source" rate — the fraction of source documents with zero output coverage — flags summaries that ignore whole sources.

If citations are part of your product, add:

- **Citation precision.** Of the citations the model emitted, how many actually support the associated claim? (Manual or LLM-as-judge.)
- **Citation recall.** Of the claims the model made, how many have a supporting citation? (Similar audit.)

## When to use MDS vs. single-doc

The boundary is fuzzier than it looks. Two heuristics:

- **If the documents are *the same* content in different phrasings** (e.g., ten news outlets covering the same story), MDS is the right formulation — you gain redundancy signal and lose nothing.
- **If the documents are *complementary* content** (e.g., a design doc, a code review, a postmortem), MDS is required — no single doc contains the answer.
- **If the documents are *one long document in disguise*** (e.g., chapters of a book), treat it as long-doc single summarisation (chapter 04). Do not pretend the chapter break matters if it does not.

## Chapter summary

- Multi-document summarisation adds three failure modes over single-doc: redundancy, contradiction, and source attribution. Every MDS system needs an answer for each.
- Two architectural blueprints dominate: extractive-then-abstractive with MMR/clustering dedup, and long-context multi-doc encoders like PRIMERA.
- Redundancy: MMR or cluster-based dedup at extraction time. Contradiction: report-and-attribute or prefer-majority; both require an explicit claim-extraction step. Attribution: pick a rigour level (document, sentence, fine-grained) and design the pipeline around it.
- Multi-News is the standard prototyping dataset; Multi-XScience and QMSum stress harder settings.
- Evaluate ROUGE plus redundancy (pairwise output ROUGE) and source coverage (skipped-source rate) — aggregate ROUGE hides both.
