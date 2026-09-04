# exercise-02: Build vs. Buy vs. Fine-Tune vs. RAG Decision Doc

**Estimated effort:** 3 hours

## Objective

Author a **decision document** for a single, concrete NLP stage that compares four stances — **buy-managed-API**, **buy-open-weight-self-host**, **adapt-fine-tune**, and **RAG-augment an already-chosen model** — under a realistic budget envelope. Deliver a cost / quality / operational-cost analysis with a sensitivity table, a recommended stance with rationale, and a signed one-page executive summary a non-ML stakeholder can act on. The exercise operationalises chapter 2's budget-allocation discipline on a stage of the blueprint from exercise-01 (or a new one you scope).

## Prerequisites

- Chapters [01](../01-product-goal-to-nlp-blueprint.md) and [02](../02-budget-allocation-across-the-nlp-stack.md).
- Exercise-01 is not strictly required, but a completed blueprint gives you a concrete stage to run this exercise on.
- Spreadsheet or Jupyter notebook for the cost model — the arithmetic is elementary but the parameter sweep is easier interactive. `pandas` is enough; `numpy` for a sensitivity sweep is nice-to-have.
- Access to public pricing pages for whichever managed-API provider you choose (OpenAI, Anthropic, Google Vertex, AWS Bedrock, Cohere, Azure OpenAI). Prices change — cite the URL and the date you retrieved it, so a reader can reconcile the numbers.

## Problem statement

### Part A — Scope the stage

Pick **one** NLP stage from a real product or from an exercise-01 blueprint. Concrete, single-stage examples that work well:

- A customer-support-ticket classification stage: 12 queue labels, ~20 000 tickets/day, sub-200 ms p95 required.
- A contract-clause extraction stage: 15 clause types, ~500 contracts/week, no latency SLA.
- A doc-Q&A generation stage for internal knowledge base: ~10 QPS at business hours, 512-token answers, must ground on the KB.
- A financial-news-sentiment scoring stage: ~50 QPS during market hours, 3-way classification, needs to be defensibly reproducible.
- An email-summariser stage: ~5 QPS, 3–5 bullet outputs, 90 % English, must not hallucinate names or dates.

Write `scope.md`:

- The stage in one paragraph — inputs, outputs, upstream, downstream.
- Traffic estimate: peak QPS, daily volume, monthly volume, expected year-2 growth.
- Latency budget: p50, p95, p99.
- Quality bar: which metric matters, what floor is acceptable, what improvement is worth the switch.
- Data-flow constraints: PII / PHI / MNPI / confidential / public; residency; retention.
- Operational constraints: team headcount available to maintain, on-call rotation, deployment cadence.

### Part B — Enumerate the four stances

Produce `stances.md` — for each of the four stances, write a two-paragraph description:

- **Buy-managed-API.** Which provider, which model tier, which prompt shape. Rate-limits and quota assumptions. What the failure mode looks like (provider outage, price change, rate-limit exhaustion, model deprecation).
- **Buy-open-weight-self-host.** Which base model (real IDs: `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`, `microsoft/deberta-v3-base`, whatever fits), which serving stack (vLLM, TGI, Triton, TorchServe), which hardware (A10G, L4, A100, CPU-only for encoders). What the ops burden looks like — patching, GPU-availability, model updates.
- **Adapt-fine-tune.** Which open-weight base + which fine-tuning technique (full FT, LoRA, QLoRA); labelled-data requirement estimate; iteration count and per-iteration compute; deployed stack (same as self-host). What the retraining cadence is.
- **RAG-augment.** Which retriever (bi-encoder + optional cross-encoder reranker), which vector store, which base model (managed or open-weight); ingest-time cost model; query-time cost model.

For each stance, note explicitly: what **cannot** be answered without a small POC (e.g. "cannot confirm sub-200 ms p95 on `Mistral-7B-Instruct` with vLLM without benchmarking"). Do not paper over unknowns; they are the input to the sensitivity table below.

### Part C — Build the cost model

Produce `cost_model.py` (or `cost_model.ipynb` / `cost_model.xlsx` — whatever you can share). For each of the four stances, compute:

- **Year-one total cost** = one-time cost + ongoing monthly cost × 12.
    - One-time: annotation, fine-tuning compute, ingest compute for RAG, integration eng time.
    - Ongoing: managed-API tokens, GPU-hours, storage, vector-store, ops-eng time.
- **Cost per 1 000 requests** at (baseline QPS, 2× QPS, 5× QPS, 10× QPS). This is the load-scaling curve — the four stances scale very differently.
- **Break-even QPS** where the fine-tuned self-hosted stance costs the same as the managed-API stance. Below the break-even, managed API wins; above it, fine-tune wins.

Assumptions the model must expose as variables (not hard-code):

- Tokens in and tokens out per request. Assume real distributions; a "sentiment" task and a "summarisation" task differ by two orders of magnitude in output tokens.
- Managed-API per-token pricing (per provider, per tier), retrieved from the provider's public pricing page — cite URL and date.
- GPU-hour cost on your target cloud (on-demand, spot, and reserved variants where relevant); cite the provider's on-demand pricing page and date.
- Fine-tuning compute hours per run × runs per year.
- Retraining cadence for the fine-tuned stance (quarterly? monthly?).
- Vector-store cost (per-record and per-query if managed; per-GB storage + amortised compute if self-hosted).
- Human-annotation cost per label (order-of-magnitude estimate; if you do not have a real quote, cite an assumed range and mark it as an assumption).
- Ops-eng FTE cost (loaded cost per hour × hours per stance per month).

The point is **not** to produce a defensible-to-the-CFO number — it is to expose the parameter dependencies so a reviewer can challenge the ones they think are wrong.

### Part D — Sensitivity analysis

Produce `sensitivity.md` (with a table or a small plot). For each of the four stances, sweep the top 3 parameters that dominate cost (from the model in Part C) at −50 %, −20 %, baseline, +20 %, +50 %, +100 % and record how the year-one total moves and where the break-even QPS moves.

Interpret the result:

- Which stance is most sensitive to which parameter?
- Which parameter, if wrong by ±20 %, would flip the recommendation?
- Which parameters are worth investing in a small POC to nail down before committing?

### Part E — Quality × operational risk × cost matrix

Produce `matrix.md` — a small 3-column table per stance:

| Stance                | Expected quality (metric + floor) | Operational risk               | Year-one cost |
|-----------------------|-----------------------------------|--------------------------------|---------------|
| Buy-managed-API       | ...                               | vendor lock-in, price changes  | \$...         |
| Buy-open-weight       | ...                               | GPU capacity, patching, drift  | \$...         |
| Adapt-fine-tune       | ...                               | annotation quality, retraining | \$...         |
| RAG-augment           | ...                               | index staleness, ingest ops    | \$...         |

Each row has a short reasoning note (2–3 sentences) — not just the number, but why. Quality claims that cannot be substantiated with existing evidence are marked "assumed, POC required" and linked to a Part-F POC item.

### Part F — POC list

Produce `poc_list.md` — the small, cheap experiments that would sharpen the analysis before the decision. For each POC:

- The question it answers.
- The rough scope (compute, time, data required).
- The expected outcome and how it would change the recommendation.

Typical POCs for this exercise:

- "Fine-tune LoRA on `Mistral-7B-Instruct` with 500 labelled examples and measure F1 on our held-out set" — answers whether the fine-tune quality ceiling is real.
- "Benchmark p95 latency of the self-hosted stack on an L4 with vLLM at target QPS" — answers whether the operational option meets the SLA.
- "Retrieve top-10 KB chunks for 100 real queries with `all-mpnet-base-v2` + `bge-reranker-v2-m3` and hand-label recall@10" — answers whether the RAG retriever is good enough before spending on generation.

### Part G — The recommendation and executive summary

Produce `decision.md` — the recommendation itself:

- One paragraph naming the recommended stance and the top three reasons.
- The trade-offs accepted (what the recommendation gives up).
- The conditions under which the recommendation should be revisited (QPS crossing the break-even, price changes, quality-floor changes).
- Owner and next steps (starting with the POCs from Part F).

And `exec_summary.md` — a **one-page** version aimed at a non-ML stakeholder (product manager, engineering director, finance partner). Structure:

- The stage, in one sentence.
- The four stances, one sentence each, one number each.
- The recommendation, in one sentence, with the top reason.
- The three key risks and their mitigations.
- The POC plan and timeline.

The exec summary is the test — if you cannot fit the decision on one page without ML jargon, the analysis is not clear enough yet.

## Starter guidance

- **Public pricing pages, with a date.** Managed-API prices change quarterly-ish; GPU-hour spot prices change hourly. Every price in the model has a URL and a retrieval date. A reviewer six months later will need this to reconcile the model.
- **Do not confuse "cost per token" with "cost per request."** Requests have prompt tokens + retrieved context tokens + output tokens; a 4 000-token retrieved-context RAG stage costs more than a 200-token classification. Model at the *request* granularity.
- **Break-even QPS is the load-bearing number in this exercise.** If it is inside your operating range, both stances are legitimate and other factors decide. If it is far above your operating range, the answer is managed-API. If it is far below, the answer is self-host. This one number does most of the work.
- **Operational risk is real cost.** A stance that requires two on-call engineers and a fine-tuning pipeline is expensive even before the compute bill. Load the ops-eng FTE fraction into the cost model; a stance that "just works" via API often wins the total-cost picture even when the token bill is bigger.
- **Do not skip the "cannot answer without a POC" lines.** They are where the decision becomes honest. A doc that claims to know the quality of every stance without evidence is a bad doc.
- **Prompt caching matters.** Frontier managed APIs (Anthropic, OpenAI, Google) support prompt caching that can drop input-token cost 60–90 % on stable prefixes. If your prompt is designed for it, model the cache-hit rate as a variable in Part C's sensitivity sweep.
- **Do not model quality with false precision.** "Fine-tune F1 is 0.87" without a POC is invented. "Fine-tune F1 is estimated in [0.80, 0.90] based on comparable published tasks, POC required" is honest.

## Acceptance criteria

- [ ] `scope.md` names the stage, traffic, latency, quality bar, data-flow constraints, and operational constraints.
- [ ] `stances.md` describes each of the four stances with real model / provider names and per-stance failure modes.
- [ ] `cost_model.{py,ipynb,xlsx}` is parameterised — every input is a named variable with a cited source (URL + date) or an explicit assumption.
- [ ] Cost model computes year-one total, cost-per-1 000-requests at four load points, and the break-even QPS between fine-tune and managed-API.
- [ ] `sensitivity.md` sweeps the top 3 parameters per stance and identifies which parameter, if wrong by ±20 %, flips the recommendation.
- [ ] `matrix.md` populates the 4-row × 3-column table with reasoning per cell; quality claims not backed by evidence are marked "POC required."
- [ ] `poc_list.md` proposes at least three small experiments that would sharpen the analysis, each with a scope and expected outcome.
- [ ] `decision.md` recommends a stance, names the trade-off accepted, and gives the revisit conditions.
- [ ] `exec_summary.md` is one page, jargon-free, and includes the recommendation + top-3 risks + POC plan.

## Stretch goals

- **Two stages, two decisions.** Run the exercise on two contrasting stages from your exercise-01 blueprints. Different stances usually win for different stages; the second run demonstrates the rubric is not one-size-fits-all.
- **Prompt-caching sensitivity.** Add prompt-caching effectiveness (0 %, 30 %, 60 %, 90 % hit rate) as a fifth dimension of the sensitivity sweep. Managed-API stances can flip winners under high cache-hit rates.
- **Multi-provider comparison.** Model three managed-API providers (OpenAI, Anthropic, Google Vertex or Bedrock) side-by-side. The between-provider spread is often as large as the between-stance spread.
- **Include the "hybrid" stance.** Route easy requests to a cheap stance (self-hosted encoder classification) and only escalate the hard tail to a managed-API frontier model. Model the routing threshold and the cost mix; hybrid often wins the real-world total-cost race.
- **Real POC on one item.** Actually run one of the Part F POCs — a small LoRA on 100 examples, or a latency benchmark on a self-hosted model — and update the cost model with the measured number. The exercise ends up much sharper than a pure paper analysis.
- **Adversarial review.** Have a colleague read `decision.md` and score every claim on 1–5 confidence. Where they score < 3, rewrite. This is what a real design review does.

## Deliverables

```
scope.md
stances.md
cost_model.py             # or .ipynb / .xlsx
sensitivity.md
matrix.md
poc_list.md
decision.md
exec_summary.md
README.md                 # how to read the repo, and the choices you made
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
