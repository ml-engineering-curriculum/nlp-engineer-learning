# From Product Goal to NLP System Blueprint

## Motivation

An NLP system almost never fails at the model layer first. It fails at the framing layer: the team built a classifier when the product wanted extraction, chose a decoder-only when an encoder was the right primitive, trained a monolingual model when the traffic was 40 % multilingual, or built when they should have bought and now maintain a stack no one else will touch. Every one of those mistakes is invisible in a benchmark and expensive in production.

This chapter walks the framing work — the decisions you make *before* opening a notebook. It is the discipline of translating "we want to answer customer questions from our docs" into a blueprint that names the task type, the model family, the language coverage, and the build-vs-buy stance for each stage, with the reasoning trail attached. Everything else in the module — budget (chapter 2), release gates (chapter 3), domain governance (chapter 4), responsible-release artifacts (chapter 5) — depends on the blueprint being right.

## Start with the product goal, not the model

The trap is opening a Hugging Face model page first. The correct order:

1. **What decision or action does the system enable?** "Route this ticket to the right queue." "Redact PHI before a document leaves our VPC." "Give the sales rep a two-line summary of the last three calls." A goal without a downstream action is a research project, not a product.
2. **What is the input, and who produces it?** Free-text email, structured form + free-text comment, call transcript, PDF, code, chat message. Who wrote it — an internal employee, a paying customer, a scraped web page? Language(s), length distribution, character-set assumptions, quality range.
3. **What is the output, and who consumes it?** A label the routing service acts on, a highlighted span the reviewer clicks through, a JSON object the billing system ingests, a paragraph a human reads. Output shape *is* the API contract; get it explicit.
4. **What is the latency, throughput, and cost budget?** Sub-100 ms for a synchronous UI. Best-effort within a batch window for an overnight job. $10 per thousand documents to be defensible. Numbers, not adjectives.
5. **What are the failure-mode costs?** A false-positive PHI redaction is annoying; a false-negative is a HIPAA incident. A miscategorised sentiment tag is noise; a mistranslated dosage is a lawsuit. Symmetric-cost systems and asymmetric-cost systems ship differently.

Write these down as a one-page **product goal spec** before touching model selection. Every subsequent decision in this chapter refers back to it. Teams that skip this end up re-litigating scope during the release-gate discussion (chapter 3), which is the worst possible time.

## Task framing: classification vs. extraction vs. generation

The single most consequential framing decision is which of three shapes the task takes. They have different data requirements, different evaluation regimes, different model families, and different failure modes.

### Classification

The output is one of a fixed label set — one label, top-k, or multi-label. Sentiment, intent detection, topic routing, toxicity, spam, priority, language ID.

- **Data:** labelled examples per class; typically 500–5 000 per class for a fine-tuned encoder, or a few examples per class for zero-/few-shot with an instruction-tuned decoder.
- **Model family:** encoder-only Transformer (BERT / RoBERTa / DeBERTa / MPNet) is the workhorse; small LLMs at zero-shot work for coarse taxonomies but underperform fine-tuned encoders on any label set the model has not seen a lot of during pretraining.
- **Eval:** accuracy, precision/recall/F1 per class, macro-F1 for imbalanced sets, calibration (mod-111 chapter 2). Confusion matrix is the diagnostic surface.
- **Failure modes:** class imbalance masks per-class collapse; drift in label distribution (mod-112 chapter 6) silently degrades production; ambiguous labels produce inter-annotator disagreement floors.

Choose classification when the decision surface is genuinely discrete and finite. "Route to one of 8 queues" is classification. "Extract the account number" is not.

### Extraction

The output is one or more spans from the input — NER, key-value extraction from documents, span-level PII, event triggers, argument roles. mod-104 covers the mechanics.

- **Data:** span-annotated documents; more expensive to label than classification (10–100× per document); labels are typically IOB / BILOU or offset-based.
- **Model family:** encoder-only with a token-classification head (BERT-family); span-classification variants; increasingly LLMs with instruction prompts that emit structured spans, especially for schemas that change quickly.
- **Eval:** entity-level F1 (`seqeval`), partial-match F1, per-entity-type breakdown; exact-match on offsets, not on strings (chapter mod-111 chapter 2).
- **Failure modes:** nested entities (mod-104 chapter 2) break IOB; long-tail entity types are rarely learnable; character-vs-token offset mismatches (mod-112 chapter 1) silently corrupt downstream consumers.

Choose extraction when the answer *is* text from the input and downstream systems need offsets. Do not fake it as a classification over enumerated spans unless the span set is genuinely closed and small.

### Generation

The output is free-form text: summary, translation, answer, explanation, rewritten paragraph, generated JSON via a schema, generated code. mod-105/mod-106/mod-107 cover the mechanics.

- **Data:** paired input/output examples for fine-tuning; instructions + a small demonstration set for prompt engineering with a large decoder.
- **Model family:** encoder-decoder (T5 / BART / mT5 / mBART) for tasks with a clear source-target mapping (summarisation, translation, paraphrasing); decoder-only (Llama / Mistral / Gemma / GPT-family) for open-ended generation and instruction-following.
- **Eval:** the hard case — reference-based (BLEU, ROUGE, chrF, BERTScore, COMET) plus reference-free (faithfulness scoring, LLM-as-judge, human eval). mod-111 chapters 4 and 9 are the reference.
- **Failure modes:** hallucination, prompt injection, verbosity drift, calibration collapse, output-format violations for structured generation.

Choose generation when the answer is not enumerable and cannot be a span from the input. Do not reach for generation because it is more capable; it is also less predictable, less measurable, and more expensive to serve.

### Hybrids and pipelines

Many products are actually a pipeline of two or three of these — extraction feeds a classifier feeds a generator ("extract the entities, classify the intent, then draft the response"). mod-112 chapter 1 covers composition. The blueprint records the *shape* of each stage; do not blur the shapes together.

## Model-family choice: encoder / encoder-decoder / decoder-only

Given a task shape, the model-family choice narrows quickly, but the trade-offs are worth naming explicitly.

### Encoder-only (BERT-family)

Pretrained with masked language modelling; produces contextual token embeddings and can be topped with a classification / span / regression head.

- **Strengths:** small (base ~110M params, small ~66M), fast on CPU or a modest GPU, strong on classification and extraction after fine-tuning, deterministic outputs.
- **Weaknesses:** cannot generate text; poor for open-ended tasks; multilingual variants (mBERT, XLM-R) exist but underperform monolingual encoders on their native language.
- **Reach for it when:** the task is classification or extraction and you have labelled data. This is the default for production NLP; every "we shipped a transformer" story that runs on CPU under 100 ms is almost always an encoder.

### Encoder-decoder (seq2seq)

Encoder over the input, autoregressive decoder over the output; T5, BART, mBART, Flan-T5, NLLB. Pretrained with span corruption or denoising objectives.

- **Strengths:** the natural fit for source-target tasks — summarisation, translation, paraphrasing, structured text-to-text (T5-style "task-prefix" framing). Strong at faithfulness on grounded inputs.
- **Weaknesses:** larger than encoders; not the best fit for open-ended chat; instruction-tuned variants (Flan-T5) narrow the gap but the ecosystem has largely consolidated on decoder-only for chat.
- **Reach for it when:** the input and output are both texts with a clear mapping — MT, summarisation, text-to-SQL, controlled generation.

### Decoder-only (GPT-family)

Autoregressive LM over concatenated context; Llama, Mistral, Gemma, Qwen, DeepSeek open-weights; GPT, Claude, Gemini as managed APIs.

- **Strengths:** the most general shape; strong at zero-/few-shot prompting; the ecosystem is where the current instruction-tuning, tool-use, and multi-turn work happens; scales to the largest sizes.
- **Weaknesses:** larger and slower per token; non-deterministic without seed control; harder to audit; latency dominated by output length.
- **Reach for it when:** the task is generation, especially open-ended or instruction-following; when labelled data is scarce and you can lean on zero-/few-shot; when you already need chat multi-turn.

### The default choice, revisited

The 2024–2026 pattern is that many teams *default* to a decoder-only LLM (managed API or open-weight) for everything, including tasks an encoder would do faster and cheaper. That is a real trade-off — one API surface, no fine-tuning pipeline, faster prototyping — but it is the more expensive stance at scale. A useful blueprint rule: **for a stable, high-QPS, well-labelled classification or extraction task, fine-tune an encoder; for exploratory, low-QPS, poorly-labelled, or generative tasks, use a decoder LLM**. Chapter 2 makes the cost side concrete.

## Multilingual vs. monolingual

The language question is not "does our product speak English?" — it is "what languages actually appear in production traffic, in what proportions, and does model quality per language meet the product bar?"

Three postures, and how to choose:

- **Monolingual (target one language).** Train / fine-tune on that language; pick a monolingual encoder or decoder (e.g. `roberta-large` for English, `camembert-base` for French, `bert-base-japanese` for Japanese). Highest quality per parameter for that language. Reach for it when >95 % of traffic is one language and non-target inputs can be routed elsewhere or rejected.
- **Multilingual (one model, N languages).** Use a multilingual encoder (XLM-R, mDeBERTa, LaBSE) or seq2seq (mBART, NLLB, mT5). Lower per-language ceiling than a monolingual model but avoids N deployments; enables cross-lingual transfer for low-resource languages via zero-shot. Reach for it when traffic is genuinely multilingual and per-language traffic does not justify per-language models.
- **Language-routed (N monolingual models behind a language ID).** Language-ID stage routes to the right monolingual model. Highest total quality; highest operational overhead (N models to train, evaluate, monitor, roll back). Reach for it when a few high-traffic languages dominate and the per-language teams / evaluations exist.

Mismatches to watch:

- Ship a monolingual English model, then find 15 % of traffic is Spanish, then the model silently degrades on that slice. mod-107 chapter 3 covers cross-lingual zero-shot as a stopgap.
- Ship a multilingual model, then get complaints that the English quality is worse than the previous monolingual English model. Both are true; a per-language quality gate (chapter 3) is the mitigation.
- Ship language ID as a stage without tracking its confidence and downstream reject rate — short inputs misidentify at high rates (mod-107 chapter 3, fastText `lid.176` caveats).

The blueprint should record the target languages, the expected traffic split, and the posture — not just "supports multilingual."

## Build vs. buy vs. adapt: the architecture-layer decision

Every NLP system has stages, and each stage has a build-vs-buy stance. The naive framing "build our own model or use OpenAI" is under-resolved. The full grid, per stage:

- **Buy — managed API.** OpenAI, Anthropic, Google Vertex, Azure OpenAI, AWS Bedrock, Cohere, or a specialised API (Google Cloud Healthcare API for de-id, spaCy Prodigy for annotation, Twilio for STT). No infra; per-call cost; vendor lock-in on prompt, model version, and data-flow.
- **Buy — open-weight model, run yourself.** A `mistralai/Mistral-7B-Instruct` on your GPU cluster; a `microsoft/deberta-v3-base` for classification; a `nvidia/NV-Embed` for retrieval. Zero training cost; capex/opex on serving; vendor lock-in on the model card.
- **Adapt — fine-tune an open-weight base.** Take an open-weight base, LoRA / QLoRA / full fine-tune on your data. Costs training compute and data engineering; produces an artifact you own; quality typically dominates zero-shot on your task.
- **Build — train from scratch or heavily pre-train.** Almost never the right stance in 2026 for standard NLP tasks — the base models are too good and too cheap to reproduce. Justified for domain-scale corpora where existing base models genuinely fail (large-scale legal-code training, code-only pretraining for a specific language, biomedical from-scratch pretraining if the domain corpus is large enough to justify a `PubMedBERT`-style run).

Decision heuristics for a single stage:

- **Latency-sensitive + high QPS + well-labelled task → adapt or buy-open-weight.** A managed API's per-call latency and cost do not close.
- **Bursty, low-QPS, or exploratory → buy managed API.** The infra and operational overhead of self-serving is not amortised.
- **Regulated data that cannot leave the VPC → buy open-weight or adapt, never managed API without a BAA and data-flow review.** Chapter 4 walks the governance side per domain.
- **Task where a base model already beats your best fine-tune → buy managed API (large decoder) or buy open-weight (large decoder).** Do not fine-tune to prove you can.
- **Task where a small fine-tuned encoder beats every base model on your data → adapt.** The ceiling on a supervised task with labels is often 5–10 F1 above zero-shot LLM performance.

Every stage of the blueprint carries one of {buy-API, buy-open-weight, adapt, build}, with the reasoning. Chapter 2 attaches costs to each; exercise 2 walks the decision doc for a concrete stage.

## Retrieval-augmented, tool-using, and agentic stances

A rising fraction of NLP systems are not "one model," they are "one model plus a retriever" or "one model plus tools." These are architecture choices that belong in the blueprint.

- **RAG (retrieval-augmented generation).** Retrieve top-k passages from a vector store or hybrid retriever, condition the generator on the retrieved passages. Reach for it when the answer needs to be grounded in a private corpus, when the knowledge changes faster than you can retrain, or when you want to cite sources. Adds a retriever stage (mod-108 covers embeddings, mod-105 chapter 8 covers RAG proper). The trade-off is another moving part — a stale index degrades quality silently.
- **Tool use.** The model emits a function call; a tool (calculator, SQL, code interpreter, external API) runs and the result is fed back to the model. Reach for it when the answer requires computation the model cannot reliably do. Adds an executor stage with its own security surface.
- **Agentic loops.** Multi-step reasoning with tool calls in a loop, potentially planning and self-critique. Reach for it very sparingly — the extra steps compound latency, cost, and failure modes; the ROI over a well-designed prompt + tool is often negative for standard tasks.

The blueprint records whether the system is single-shot, RAG, tool-augmented, or agentic. Confusing single-shot with agentic is the fastest way to blow a budget forecast; the two have order-of-magnitude different cost profiles.

## The blueprint artifact

The output of this chapter's work is a short, structured document — not a model card (that lives in chapter 5), but the *design* document that precedes model selection. A working template:

```markdown
# System blueprint: <name>

## Product goal
- Decision or action enabled: ...
- User-visible surface: ...

## Task framing (per stage)
| Stage | Shape (cls/ext/gen) | Input | Output | Latency budget | Failure-mode cost |
|-------|---------------------|-------|--------|----------------|-------------------|
| ...   | ...                 | ...   | ...    | ...            | ...               |

## Model-family choice (per stage)
| Stage | Family (encoder / seq2seq / decoder) | Candidates | Rationale |
|-------|--------------------------------------|------------|-----------|

## Language coverage
- Target languages: ...
- Expected traffic split: ...
- Posture: monolingual / multilingual / language-routed
- Rationale: ...

## Build vs. buy (per stage)
| Stage | Stance (buy-API / buy-OW / adapt / build) | Rationale | Alternatives considered |

## Augmentation
- Single-shot / RAG / tool-use / agentic: ...
- Rationale: ...

## Explicit non-goals
- ...

## Open questions / decisions deferred to next review
- ...
```

Two properties matter:

- **Every choice has a rationale**, not just a name. A reader six months later can reconstruct why the team chose what they chose. This is the single biggest lever against re-litigating the same decisions during release review.
- **Non-goals and open questions are explicit.** Blueprints that only enumerate goals silently accumulate scope. An "explicit non-goals" section is the guardrail that lets you say "no, this system does not do X" during a stakeholder review without embarrassment.

Exercise 1 turns this template into a working rubric on two concrete product goals.

## Chapter summary

- The framing layer is where most NLP systems fail first. Get the product goal, task shape, model family, language posture, and build-vs-buy stance right *before* opening a notebook.
- Task shape drives everything downstream: classification (fixed label set, encoder-friendly), extraction (spans from input, encoder + token head), generation (free-form text, seq2seq or decoder). Pipelines are hybrids; each stage still has one shape.
- Model-family default: encoder for stable high-QPS classification/extraction with labels; encoder-decoder for source-target tasks; decoder-only for open-ended generation, instruction-following, and low-label prototyping. The 2024–2026 default-to-LLM stance is real, but not free at scale.
- Multilingual posture is a product question, not a research one. Record the target languages, expected traffic split, and posture (monolingual / multilingual / language-routed) — with the rationale.
- Build vs. buy is per stage, not per system. Grid = {buy-managed-API, buy-open-weight-self-host, adapt-fine-tune, build-from-scratch}. Build-from-scratch is almost never the right stance for standard NLP in 2026.
- Augmentation stance (single-shot / RAG / tool-use / agentic) belongs in the blueprint; confusing single-shot with agentic blows budget forecasts.
- The chapter's artifact is a **one-page blueprint** with rationales, non-goals, and open questions. It is the input to chapters 2 (budget), 3 (release gates), 4 (domain governance), and 5 (release artifacts).
