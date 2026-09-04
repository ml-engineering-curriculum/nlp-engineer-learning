# Budget Allocation Across the NLP Stack

## Motivation

An NLP project's budget is not one number — it is an allocation across five buckets that trade off against each other: **data engineering**, **training**, **fine-tuning**, **retrieval**, and **managed-API usage**. Get the allocation wrong and the system either underperforms (starved data engineering, over-invested training compute) or overspends (managed-API burn on a task a $2 000 fine-tune would have solved). The blueprint (chapter 1) named the stages and the stances; this chapter attaches costs and forces the trade-offs into a document a finance partner can read.

The goal is not to minimise total cost — it is to make the allocation *deliberate*, with each trade-off documented, so that six months later a reader can answer "why did we spend 60 % on data and 10 % on training?" and the answer is a decision, not an accident.

## The five buckets

### Data engineering

Everything upstream of the model: sourcing, cleaning, labelling, deduplication, splitting, pipeline maintenance, dataset cards (mod-110 and chapter 5). Includes annotator time (in-house or vendor), tooling (Prodigy, Label Studio, Scale AI, Argilla), storage, and the ML engineer time spent iterating on data quality.

Common under-investment. The "we'll figure out the data later" project ships a model that hits 68 % where the same architecture on cleaner data hits 84 %. The Google *Rules of Machine Learning* (["Rules of Machine Learning: Best Practices for ML Engineering"](https://developers.google.com/machine-learning/guides/rules-of-ml), Rule 4) puts this bluntly: keep the first model simple and get the infrastructure right, because the data is where the wins are.

Cost drivers:

- **Labelling.** Complex NER at expert-quality rates (clinical, legal) commonly runs into the low tens of dollars per document; crowd-labelled sentiment on short text is orders of magnitude cheaper. Budget the annotator hours *before* committing to a supervised approach. <!-- needs-research: specific per-hour rates by domain vary widely by vendor and region — do not quote a single number without the sourcing engagement. -->
- **Corpus sourcing and licensing.** Web-scraped corpora are cheap and legally fraught; licensed corpora (LDC, ELRA, domain-specific paid datasets) are expensive and shipable. Chapter 4 covers domain-specific licensing.
- **Data platform.** Delta / Iceberg tables, feature stores, streaming ingestion — mostly one-time capital and steady-state maintenance. Amortised across projects if the platform team owns it.

Under-investment signals: labelling is a one-person side task; there is no dataset card; the eval set was extracted from the training corpus with `train_test_split` and no thought about temporal or entity leakage.

### Training

Pretraining a base model from scratch, or a heavy domain adaptive pretraining pass. Compute-dominated: many-thousand GPU-hours of A100 / H100 / equivalent.

In 2026, for standard NLP tasks, this bucket should almost always be zero. Base models are numerous, high-quality, and cheap to license or run. Justified only when:

- The domain corpus is large and genuinely off the base models' pretraining distribution (large-scale biomedical, code-only for a specific language, low-resource languages with a strong data advantage).
- The organisation has strategic reasons to own the base (sovereignty, IP, downstream research programme).
- There is a research team who will actually own the model long-term.

If the bucket is non-zero, the dominant variable is the compute cost. The Chinchilla scaling result (Hoffmann et al., ["Training Compute-Optimal Large Language Models"](https://arxiv.org/abs/2203.15556), *NeurIPS 2022*) is the reference for compute-vs-data trade-offs on pretraining; the practical corollary is that under-training a large model wastes both compute and data.

### Fine-tuning

Task adaptation on an open-weight base — full fine-tuning, LoRA / QLoRA, or instruction-tuning on a small task corpus. Compute-modest: single-to-few-GPU runs measured in hours to days, not weeks.

The cost profile is dominated by:

- **The base model size.** A 7B fine-tune is order-of-magnitude cheaper than a 70B; a base encoder fine-tune is order-of-magnitude cheaper than either.
- **The technique.** Full fine-tuning updates all weights; LoRA (Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), *ICLR 2022*) trains ~0.1–1 % of the parameters and produces a small adapter artifact; QLoRA (Dettmers et al., ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314), *NeurIPS 2023*) fine-tunes a quantised base and fits large models onto a single GPU.
- **The iteration count.** The first fine-tune is one run; a real project has 5–20 runs across hyperparameters, data mixes, and eval slices.

Under-investment signal: one fine-tune, no ablation, no proper eval; the artifact is picked because it was the only one that finished.

### Retrieval

The vector store, hybrid retrievers, chunking pipeline, embedding models, and reranking model that sit under a RAG stage (chapter 1). Ongoing compute + storage.

Cost drivers:

- **Embedding compute at ingest.** Embedding 10M chunks with a 1B-parameter model is a real budget item; caching is essential.
- **Embedding compute at query time.** Query embeddings are cheaper per call but scale with QPS.
- **Vector store.** Self-hosted (`FAISS`, `Milvus`, `Qdrant`, `Weaviate`, `pgvector`) trades off latency and cost against operational overhead; managed (Pinecone, MongoDB Atlas Vector, AWS OpenSearch) pushes ops to a vendor at a premium.
- **Reranker.** Optional but commonly a large single-cross-encoder call per query; adds latency and cost, buys precision.

Retrieval budgets scale with corpus size (ingest, storage) and with QPS (query embeddings, retrieval hops, reranking). A common trap: budget only the query-time cost and discover the ingest bill during the first backfill.

### Managed-API usage

Per-token / per-request cost against a hosted model (OpenAI, Anthropic, Vertex, Bedrock, Cohere, etc.). No infra spend; pure variable cost per call.

Cost drivers, all of which the blueprint must estimate:

- **QPS at steady state.** Requests per second at peak and average.
- **Tokens per request.** Prompt + retrieved context + output; for chat systems, the prompt tokens include conversation history and grow linearly.
- **Model tier.** Frontier tier vs. cost-optimised tier; a common pattern is to reserve the frontier tier for hard queries and route the rest to a smaller model.
- **Caching effectiveness.** Prompt caching (available on the frontier APIs — Anthropic, OpenAI, Google) can drop 60–90 % of the input token cost on cached prefixes if the prompt is designed for it (large static system prompt / retrieved-context prefix). Not free; requires prompt-engineering discipline. <!-- needs-research: per-provider cache hit thresholds and pricing vary; consult the provider's caching docs for current numbers. -->

Managed-API costs are the most visible line in the finance report and the hardest to control after launch — user behaviour drives spend directly. Rate limiting, per-user quotas, and tier-routing all belong in the blueprint before the first paid call.

## The trade-offs, made concrete

The five buckets are not independent. The characteristic trade-offs — the ones a good decision doc names explicitly:

- **Data engineering trades against model compute.** More labelled data means a smaller fine-tuned encoder can match a much larger zero-shot LLM. Every dollar spent on annotation is a dollar potentially not spent on runtime tokens forever.
- **Training trades against fine-tuning.** Pretraining a domain model up front means every subsequent task fine-tune starts from a better base; but the payoff is only there if you will fine-tune many tasks on the same domain.
- **Fine-tuning trades against managed-API.** A LoRA on a 7B open-weight model that gets shipped can eliminate the need for a large managed-API call on the same task; the amortisation curve depends on QPS. Below a QPS threshold, the managed API wins on total cost; above it, the fine-tune wins.
- **Retrieval trades against fine-tuning.** RAG grounds the model in fresh data without retraining; fine-tuning bakes knowledge into weights. For frequently-changing knowledge, RAG wins; for stable, task-specific reasoning, fine-tuning wins. Both have a role in most systems.
- **Managed-API trades against everything.** The API is the simplest to prototype with and the hardest to control at scale. Almost every mature NLP stack contains at least one managed-API stage and at least one owned stage; the mix is the trade-off.

Ex-post regret patterns worth naming:

- **The fine-tune-that-never-was.** Team ships on managed API to prototype fast; QPS scales; API bill is 5× the annual salary of an engineer who would have fine-tuned the encoder that replaces it. The correction is a fine-tune project six quarters late.
- **The over-trained base.** Team domain-pretrains a 13B on 200 GB of legal text; nine months later the base is superseded by an open-weight release with better legal performance and no one uses the trained artifact. The compute was sunk.
- **The starved data pipeline.** Team spends 80 % of budget on training runs; the eval set is dirty; every quality delta between runs is within the noise of eval-set noise. mod-111 chapter 7 (paired significance) is the diagnostic.
- **The RAG that indexes yesterday's data.** Team ships RAG; forgets the ingest budget; the index freezes at ingest-day-one; the model answers stale questions. The remediation is an ongoing ingest job that no one budgeted for.

## The trade-off document

The output of this chapter's work is a **budget allocation document** attached to the blueprint from chapter 1. It has three sections.

### Section 1: allocation table

For each blueprint stage, name the bucket and the estimated year-one cost. Concrete example (numbers illustrative, not benchmarks):

| Stage                         | Bucket                    | Year-one estimate | Rationale                                                           |
|-------------------------------|---------------------------|-------------------|---------------------------------------------------------------------|
| Corpus labelling (10k docs)   | Data engineering          | \$X               | Domain-specialist annotators; two-pass with adjudication            |
| Base model fine-tune          | Fine-tuning               | \$Y               | LoRA on `mistralai/Mistral-7B-Instruct-v0.2`; ~50 GPU-hours × 10 runs |
| Retrieval index               | Retrieval (ingest + storage) | \$Z            | 2M-chunk index, embedded once with `all-mpnet-base-v2`; monthly re-embed |
| Managed API fallback          | Managed-API               | \$W               | Frontier model for hard queries (~5 % of traffic)                   |
| Serving infra                 | (Infra — separate cost centre) | ...          | ...                                                                 |
| **Total**                     |                           | **\$T**           |                                                                     |

Every row cites its assumptions. "\$W depends on 10 QPS × 500 output tokens × \$X / 1k output tokens × 24 × 365 seconds/year × 5 % routing" — spelled out, so the assumption a reviewer wants to challenge is visible.

### Section 2: trade-offs considered

For each bucket, name the trade-off the team faced and the choice they made. Concrete:

- "We considered pretraining a domain base (~\$100k compute + 3 months) versus fine-tuning `Mistral-7B-Instruct` (~\$5k compute + 3 weeks). We chose the fine-tune because we only need one downstream task and the base has been legal-adapted enough for our data."
- "We considered self-hosting the retriever versus using Pinecone. We chose self-hosted `Qdrant` because our corpus is on-premise and the data-flow review would not clear Pinecone."
- "We considered eliminating the managed-API stage. We kept it as a fallback for the 5 % of queries the fine-tuned model rejects, because the alternative was a 10-point-quality drop on the hard tail."

Trade-offs on the paper trail are the difference between a considered budget and one someone will second-guess three months in.

### Section 3: sensitivity analysis

Which assumptions dominate the budget? A one-page sensitivity table:

| Assumption                        | Baseline | ±20 % swing impact  |
|-----------------------------------|----------|---------------------|
| Managed-API QPS at steady state   | 10       | ±\$50k/year         |
| Fine-tuning iteration count       | 10       | ±\$1k               |
| Retrieval corpus size at year-end | 2M       | ±\$20k              |

The point is to identify which variables are worth monitoring after launch (usually 1–2), and which are noise (the rest). Chapter 3 covers the release-gate metrics that watch the sensitivity-dominant variables in production.

## Cost-per-request as the operational unit

Once the allocation is set, the operational unit becomes **cost per request** or **cost per document processed**. Instrument it from day one — a Prometheus histogram of tokens-in / tokens-out per request, converted to dollars via a static price table, is enough to catch a runaway before the finance report does.

Rules of thumb the instrumentation catches:

- A sudden 3× cost-per-request spike is almost always a prompt bloat (context injection got longer) or a retrieval regression (too many chunks retrieved). Both are visible in the metric before they are visible on the bill.
- A gradual drift up is usually organic input growth (users ask longer questions) — an alert on p95 cost per request catches it.
- A sudden drop is often a caching effect (prompt caching became eligible) or, worse, a silent output truncation. Both are worth investigating; a "we saved money for no reason" story is usually a bug.

The FinOps Foundation's ["FinOps Framework"](https://www.finops.org/framework/) is the reference for the cloud-financial-operations discipline that this chapter borrows from; every mature managed-API stack ends up applying its patterns (chargeback, showback, unit economics) whether or not the team uses the terminology.

## Anti-patterns

Three failure modes worth naming so the exercise 2 decision doc avoids them.

- **The unpriced blueprint.** The blueprint (chapter 1) exists; no dollar numbers are attached. Six months later the CFO asks why the cloud bill tripled and nobody has an explanation, because there was no baseline. Fix: no blueprint ships without a first-pass budget, even if the numbers are rough.
- **The vibes-based build-vs-buy.** "We built it because we like building things." Not a rationale. If the fine-tune costs \$100k and the managed API would have cost \$5k, the build has to justify the delta. Fix: every build stance in the blueprint has an alternative-considered line.
- **The one-shot re-forecast.** The budget is estimated once at project kickoff and never revisited. Costs drift; assumptions shift; the number is stale within a quarter. Fix: a quarterly re-forecast against actuals, driven by the cost-per-request instrumentation above.

## Chapter summary

- NLP project budgets partition across five buckets — data engineering, training, fine-tuning, retrieval, managed-API — and the buckets trade off against each other. Get the allocation deliberate, not accidental.
- Data engineering is the most common under-investment; training is the most common over-investment (in 2026, for standard tasks, training-from-scratch should almost always be zero).
- Fine-tuning vs. managed-API is the QPS-amortisation trade-off: below a QPS threshold, the API wins on total cost; above it, the fine-tuned owned model wins. Every stack ends up with a mix.
- Retrieval budgets scale with corpus size (ingest) and QPS (queries); budget both, not just one.
- Managed-API costs are the most visible line and the hardest to control after launch; instrument cost-per-request from day one and route by tier.
- The chapter's artifact is a **budget allocation document** with (1) per-stage allocation, (2) trade-offs considered, (3) sensitivity analysis. Ships attached to the blueprint from chapter 1.
- Instrument cost-per-request in production; alert on p95; re-forecast quarterly. The FinOps Framework is the discipline reference; adopt as much as you have process for.
- Anti-patterns: the unpriced blueprint, the vibes-based build-vs-buy, the one-shot re-forecast. Exercise 2 walks the decision doc that avoids them.
