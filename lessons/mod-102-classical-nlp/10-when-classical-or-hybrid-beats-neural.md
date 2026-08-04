# When Classical (or Hybrid) NLP Beats Neural — and Why

## Motivation

Chapter 01 argued *that* classical NLP still ships. This chapter is about *when* and *how* to choose it, so that the decision on a new pipeline is not driven by tooling fashion but by measurable properties of the problem. The goal is a decision framework you can apply to any incoming request: "the product team wants to auto-classify support tickets" — should classical, neural, or hybrid win?

## The decision framework

For any new NLP component, walk through five questions. If most of the answers point toward classical, that is your default. If most point toward neural, that is your default. Anything in between is a hybrid, and the design work is choosing the seams.

### 1. Is the input space enumerable?

- **Enumerable** — the set of valid or interesting inputs is a grammar you could write down: phone numbers, dates, currency amounts, log lines, HTTP request lines, chemical SMILES, addresses in a known format. → Regex or FST wins.
- **Fuzzy** — free-form prose, opinions, questions, summaries. → Neural wins.
- **Semi-enumerable** — mostly-structured with fuzzy tails. → Hybrid: rules for the enumerable core, neural for the tail.

### 2. What is the latency and throughput budget?

- **≤1 ms per document, millions of QPS** (search query rewriting, ad-serving text signals, safety filters in-line with generation). → Regex, FST, gazetteer, or CRF; a transformer will not fit the budget.
- **≤50 ms per document, hundreds of QPS** (chat message classification, on-page annotation, e-commerce query understanding). → Small neural model (distilled encoder) or classical, choose by accuracy.
- **≤1 s per document, tens of QPS** (document summarisation, extraction with reasoning). → Neural, up to LLM-scale.

### 3. What is the cold-start data situation?

- **Zero labelled data, engineer time to write rules.** → Rules. A hand-written regex or a gazetteer NER with 200 entities ships in a day.
- **Hundreds to thousands of labelled examples.** → Classical statistical model (CRF, perceptron, small neural fine-tune). Enough data for a CRF to shine; too little for a stable neural fine-tune.
- **Tens of thousands+ labelled or self-supervised examples.** → Neural, fine-tuned or in-context.
- **No labels but a strong LLM.** → LLM zero-shot or few-shot; add classical guardrails on the output.

### 4. What is the auditability requirement?

- **Every decision must be justifiable to a human reviewer** (health, finance, legal, employment, moderation appeals). → Classical, or a hybrid where the classical layer is the *decisive* one.
- **Aggregate decisions are auditable, per-decision is not.** → Neural is acceptable if you can characterise error rates statistically.
- **Errors have low individual cost** (search ranking, recommendation candidate generation). → Neural, and log for retraining.

### 5. What is the update cadence?

- **Multiple times per day, small changes** (new brand names, new date formats, new attack strings). → Rules / gazetteers. Editing a config file wins over retraining a model every time.
- **Weekly to monthly retraining.** → Classical statistical or small neural — you can afford the retrain.
- **Quarterly or slower with a large training run.** → Large neural. The retrain is a project.

## Concrete case studies

The following are patterns you will meet in real work. Each names *which* question dominates and *why* the answer is what it is.

### Case 1: PII redaction in a data ingest pipeline

- **Enumerability:** SSNs, credit cards, US phone numbers are enumerable. Free-form names and addresses are not.
- **Latency:** millisecond-scale on multi-GB batches.
- **Cold-start:** near-zero labelled examples.
- **Audit:** high (compliance).
- **Cadence:** rules updated when new formats appear.

**Answer:** hybrid. Regex + gazetteers for enumerable classes (credit card + Luhn checksum, SSN pattern, phone regex per country); a small neural NER (or off-the-shelf `en_core_web_sm`) for names, organisations, and locations; a classical FST post-processor that guarantees the redaction is applied to complete tokens, not mid-token. Any tension between the neural NER and the regex is resolved in favour of the regex — the deterministic layer has final say.

### Case 2: Multilingual customer-message intent classification

- **Enumerability:** intents are enumerable (`refund`, `track`, `cancel`, `password_reset`, ...) but the *phrasings* are not.
- **Latency:** ~100 ms.
- **Cold-start:** hundreds of examples per intent per language.
- **Audit:** medium; per-decision explanations helpful.
- **Cadence:** weekly retrain acceptable.

**Answer:** hybrid. Language identification via fastText (chapter 09). Per-language pipeline: (a) classical `PhraseMatcher` for high-precision, deterministic phrases ("cancel my order"); (b) small neural intent classifier (fine-tuned multilingual encoder) for everything the matcher misses; (c) log confidence and disagreement between (a) and (b) for retraining.

### Case 3: TTS text normalisation for a smart speaker

- **Enumerability:** currency, dates, times, addresses, phone numbers, cardinals, ordinals, measures — dozens of semiotic classes, each with a well-defined grammar.
- **Latency:** ~10 ms end-to-end.
- **Cold-start:** decades of published grammars available.
- **Audit:** extreme — a customer hearing "one thousand and four dollars" instead of "ten cents" is a business incident.
- **Cadence:** monthly for grammar corrections.

**Answer:** classical (FST cascade), full stop. Neural TN systems exist and improve on rare cases, but every serious production system runs the neural output through an FST safety net — because the *correctness* metric is dominated by tail behaviour, and the FST guarantees the tail. This is where `nemo-text-processing` (chapter 03) lives.

### Case 4: Summarising a long legal contract

- **Enumerability:** none — the output is free prose.
- **Latency:** seconds to minutes acceptable.
- **Cold-start:** tens of thousands of contracts; but summarisation is a generative task.
- **Audit:** high — but the audit is post-hoc, and expert review is expected.
- **Cadence:** quarterly is fine.

**Answer:** neural (LLM), with classical guardrails. Use a decoder LLM to summarise. Post-process with regex to redact PII the LLM copied through, dependency-parse the output to verify each summary claim mentions an entity that appears in the source (via string or lemma match), and score the output with a smaller LM for surface fluency. The neural does the work; classical layers verify.

### Case 5: Anomaly detection over ETL logs

- **Enumerability:** the "language" of logs is structured; each pipeline has a known set of message templates.
- **Latency:** millisecond per line at ingest rate.
- **Cold-start:** vast unlabelled logs.
- **Audit:** low per-message; high in aggregate.
- **Cadence:** rules per new log template, models per week.

**Answer:** classical. Log template extraction (Drain, Spell — classical algorithms) + character-n-gram LM for scoring template arguments + threshold on perplexity. A neural LM here would be strictly worse: slower, more expensive, less interpretable, and no better at flagging log lines that violate the template distribution.

## Anti-patterns

Common mistakes seen in the field:

- **"Just use an LLM"** applied to enumerable-input problems. A transformer to extract dates from a filing is a fifty-thousand-times cost premium over a regex, and can hallucinate dates that never appeared in the source.
- **Adding a neural model without measuring the classical baseline.** Any neural component's shipping decision should include a "vs. spaCy Matcher rules" ablation. Sometimes the ablation is a rounding error; sometimes it is the entire signal.
- **Stripping stop-words and lower-casing before a neural model** out of muscle memory. Modern sub-word tokenizers handle case and function words natively; the classical preprocessing hurts.
- **A single Python `re` regex used as a WAF-style filter on untrusted input.** ReDoS territory (chapter 02). Use RE2 or an engine with a timeout.
- **One monolithic pipeline for every language.** Route with LID (chapter 09), then apply language-specific components; a shared "multilingual" pipeline typically underperforms both extremes.
- **A rule + model system where the rule and the model disagree silently.** Always log disagreements; either the rule is over-fitting, or the model is missing a signal the rule captures.

## Hybrid seam patterns

Three shapes worth adding to muscle memory:

### Pre-filter → neural

A cheap classical filter drops the input the neural does not need to see. Example: rule out messages shorter than 3 characters or that match a spam regex before sending to a moderation LLM. Saves 80% of the cost with no accuracy loss.

### Neural → post-verifier

A neural model produces a candidate; a classical layer verifies it against structural constraints. Example: NER model proposes a date span; a regex verifies it is a valid ISO-8601 date; failure triggers a fallback to a broader parse. Turns fuzzy outputs into strict schema.

### Parallel voting

A classical and a neural component both make a prediction; ensemble the two, log disagreements. Example: PhraseMatcher gazetteer NER + neural NER; take the union; deduplicate; log conflicting spans for human review. Recall improves without changing either component.

## The framing that keeps a pipeline maintainable

Every layer of a pipeline should have a written answer to: *what is this layer good at, what fails silently, and how do we detect that failure?*

- Classical layers: state the invariant they preserve (e.g., "this normaliser produces NFC-normalised, case-folded, whitespace-collapsed output"), and unit-test it.
- Neural layers: state the metric they optimise and the eval slice they are measured on; alert when the metric slips.
- Hybrid seams: state the tie-breaking rule when the layers disagree.

A pipeline built to this discipline stays legible after a year of ownership changes and model upgrades. A pipeline built by adding whatever component seemed cool at the time does not.

## Chapter summary

- Choose between classical, neural, and hybrid components using five properties of the problem: enumerability of input, latency budget, cold-start data, auditability, and update cadence.
- Case studies (PII, intent classification, TTS normalisation, contract summarisation, log anomaly) show how the answers combine into concrete pipeline shapes.
- Common anti-patterns: LLMs where a regex suffices, missing classical baselines, ReDoS-prone regex, and silent disagreements between rules and models.
- Three hybrid patterns — pre-filter, post-verifier, parallel-voting — cover most production seams.
- The maintainability discipline is unchanged from earlier engineering: every layer has a written invariant, a failure mode, and a monitor.
- This closes the module. The next module (`mod-103-text-classification`) picks up with the neural end of the spectrum, but the framework here should shape every subsequent design choice.
