# The QA / RAG Boundary

## Motivation

Every retrieval-augmented QA system is two systems bolted together: a *retriever* that finds candidate passages, and a *reader/generator* that turns those passages into an answer. The two components have different failure modes, different owners in most engineering orgs, and — in this curriculum — different tracks. This chapter draws the boundary explicitly so that neither side spends effort re-solving the other's problems, and so that the interface between them is designed rather than accidental.

The NLP-engineer role owns the reader/generator side (everything in chapters 02–09 of this module). The `rag-engineer` role owns the retriever side. Both roles co-own the *interface*: what the reader is handed, what it does with an empty retrieval, and how end-to-end metrics attribute failures. That interface is the topic of this chapter.

## What each side owns

**Retriever (rag-engineer track):**

- Corpus ingestion and chunking policy.
- Sparse indexing (BM25 / Elasticsearch), dense indexing (embedding models, FAISS / Qdrant / Weaviate / pgvector), and hybrid retrieval.
- Reranking (cross-encoder or LLM-based) between initial retrieval and reader hand-off.
- Query understanding: HyDE (Gao et al., ["Precise Zero-Shot Dense Retrieval without Relevance Labels"](https://arxiv.org/abs/2212.10496), *ACL 2023*), query rewriting, query decomposition for retrieval.
- Freshness (when to re-index) and access control (per-user document filters).
- Retrieval evaluation: recall\@k, MRR, nDCG, retrieval-precision, coverage.

**Reader / generator (NLP-engineer track — this module):**

- Reading strategy: extractive (chapter 03), abstractive (chapter 05), or closed-book (chapter 06).
- Prompt or head design.
- Long-context strategy: sliding-window, long encoder, FiD (chapter 07).
- Multi-hop reasoning inside the reader (chapter 08).
- Answer generation, formatting, and citation.
- Unanswerability, abstention, and calibration (chapter 09).
- Reader evaluation: EM, F1, faithfulness, LLM-as-judge, human rubric (chapter 11).

**Shared / interface:**

- Passage schema: fields, ordering, per-passage metadata (source URL, section title, timestamp).
- Behaviour on zero-hit retrieval.
- Behaviour on partial-hit retrieval (fewer than $k$ passages returned).
- Citation format: passage IDs, character offsets, both.
- End-to-end metrics and failure attribution.

## The passage schema — the contract

The single most under-designed piece of a real RAG system is the schema the retriever hands the reader. A minimal, robust contract:

```json
{
  "query": "original user question",
  "passages": [
    {
      "id": "doc-42/chunk-7",
      "text": "the passage content",
      "source": "https://example.com/doc-42",
      "section": "Section 3.2 · Interest Rate Adjustments",
      "score": 0.83,
      "timestamp": "2025-02-14T12:00:00Z"
    },
    ...
  ],
  "retrieval_metadata": {
    "recall_estimate": 0.71,
    "used_fallback": false
  }
}
```

Why each field matters:

- **`id`** — every citation and every failure log points to this. Stable across index rebuilds is worth engineering effort.
- **`text`** — the reader sees this and nothing else from the corpus. Passage-length policy (a rag-engineer concern) directly caps reader quality.
- **`source`, `section`** — the reader can surface these in the answer. Missing these forces the reader to invent citations, which is a hallucination vector.
- **`score`** — the reader uses this for confidence weighting and, in FiD, for passage ordering. Do not throw it away.
- **`timestamp`** — the reader can use this to answer freshness questions or to abstain on stale evidence.
- **`retrieval_metadata`** — end-to-end error attribution needs this. If retrieval flagged low confidence, the reader should abstain more aggressively.

Codify the schema. Version it. Break end-to-end tests on schema drift.

## The zero-hit case

If retrieval returns zero passages (or fewer than a configured minimum), what should the reader do?

Three viable policies:

1. **Abstain.** Return a canonical "I could not find an answer" response. Correct default for compliance-sensitive domains.
2. **Fall back to closed-book.** Ignore retrieval and answer from parametric knowledge. Correct only when the domain allows it *and* the reader is calibrated well enough not to hallucinate.
3. **Escalate.** Emit a signal ("no evidence") that a downstream orchestrator uses to try a different retriever, ask a clarifying question, or hand off to a human.

The choice belongs to the product spec, not the model. Whatever it is, ship it as an explicit code path — a silent fallback that appears identical to a successful RAG response in the logs is how "the model made up an answer" incidents happen.

## Failure attribution: which side failed?

When an end-to-end answer is wrong, the retrieval and reading failure modes need to be separated:

- **Retrieval failure.** The gold passage was not in the top-$k$. Fix on the retriever side (embedding model, chunking, reranker).
- **Reader failure with correct retrieval.** The gold passage *was* in the top-$k$, but the reader picked the wrong span, generated the wrong answer, or hallucinated. Fix on the reader side.
- **Reader failure with distractors.** The gold passage was present but so was a distractor and the reader picked the distractor. Ambiguous — often needs both a reranker and a stronger reader.
- **Correct answer, wrong evidence.** The reader gave the right final answer but cited the wrong passage. Fix on the reader (citation head, faithfulness prompting).
- **Retrieval + reader both wrong.** Fix retrieval first — a reader trained on distractor-heavy inputs learns the wrong signal.

The instrumentation to support this attribution is a joint responsibility. At minimum, every logged QA response should include the retrieved passage IDs, the gold passage IDs (if known from evaluation), and the model's chosen citation. Without those three fields you cannot decide which team should ship the fix.

## The reader's contract with the retriever

The reader owes the retriever a few concrete signals in return:

- **Which passages it actually used.** A citation like `passages=["doc-42/chunk-7"]` tells the retriever which retrievals were productive; over time this becomes weak supervision for a reranker.
- **When retrieval quality was insufficient.** An explicit "insufficient evidence" signal (per chapter 09) that the retriever can feed into query-rewriting or fallback strategies.
- **Latency budget consumption.** If the reader is slow, the retriever must know so it can trim $k$ or skip expensive reranking. Publish a target and a measurement.

Without these signals the retriever is optimising blind — chasing recall\@50 when the reader only uses the top 5, or shipping a reranker that adds 200 ms for no reader improvement.

## End-to-end metrics

Neither side's isolated metric is sufficient. The end-to-end system needs its own:

- **Answer F1 / EM at retrieval depth $k$** — the SQuAD-style metric of chapter 04 applied to the end-to-end pipeline. Report at multiple $k$ values.
- **Faithfulness rate** — fraction of answers entailed by the cited passages (chapter 05 metric).
- **Citation accuracy** — fraction of answers whose cited passage IDs actually contain the answer (a retriever + reader joint metric).
- **Abstention accuracy** — fraction of unanswerable questions correctly abstained (chapter 09 metric applied end-to-end).
- **RAGAS-style suite** (Es et al., ["RAGAS: Automated Evaluation of Retrieval Augmented Generation"](https://arxiv.org/abs/2309.15217), *EACL 2024 Demo*). Faithfulness, answer relevance, context precision, context recall — computed with LLM-as-judge. Read chapter 11 first before trusting the numbers.

Each metric belongs jointly to both teams, and each degradation should be routed to the responsible team via the attribution logic above.

## What we deliberately do *not* cover here

The following belong to the `rag-engineer` track and should be studied there:

- Chunking strategies (fixed-size, sentence-based, recursive, semantic).
- Embedding models (contrastive training, MTEB benchmarking, matryoshka embeddings).
- Vector-database selection and hybrid retrieval.
- Reranking with cross-encoders or LLMs.
- Query understanding (HyDE, decomposition, rewriting).
- Retrieval-specific evaluation and its debugging.

If your reader keeps getting handed poor passages, the fix is not in this module. Read the `rag-engineer` track's chapters on retrieval quality, and then come back here to tighten the reader once the input distribution is honest.

## Chapter summary

- The NLP-engineer track owns the reader/generator: reading strategy, decoding, calibration, evaluation. The `rag-engineer` track owns the retriever: indexing, retrieval quality, reranking.
- The passage schema is the contract between the two sides. Codify it, version it, and make schema drift a test failure.
- Design the zero-hit and partial-hit behaviour explicitly. Silent fallback from RAG to closed-book is the top source of hallucination incidents in production.
- Failure attribution requires joint logging of retrieved passage IDs, gold IDs (in eval), and reader citations. Without that instrumentation, neither team can defensibly ship fixes.
- End-to-end metrics (answer F1, faithfulness, citation accuracy, abstention accuracy, RAGAS-family) are jointly owned. Isolated retriever recall\@k and isolated reader F1 do not add up to a working product.
