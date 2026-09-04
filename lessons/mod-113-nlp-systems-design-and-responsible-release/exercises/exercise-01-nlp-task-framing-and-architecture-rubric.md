# exercise-01: NLP Task Framing and Architecture Rubric

**Estimated effort:** 3 hours

## Objective

Turn chapter 1's blueprint template into a **working rubric** by applying it to **two contrasting product goals** end-to-end. Each blueprint records the product goal, task framing per stage, model-family choice with rationale, language posture, build-vs-buy stance per stage, augmentation shape (single-shot / RAG / tool-use / agentic), explicit non-goals, and open questions. Deliver both blueprints in a single repo, plus a `rubric.md` that codifies the reusable decision heuristics extracted from having done the work twice — the rubric is the artifact you would hand a colleague starting their own blueprint.

## Prerequisites

- Chapter [01 — From product goal to NLP system blueprint](../01-product-goal-to-nlp-blueprint.md).
- No coding required. Familiarity with the model families surveyed in mod-101 through mod-108 is assumed; specifically the encoder / seq2seq / decoder distinction and the multilingual model landscape (mod-107).
- Any Markdown editor. Diagram tools optional (draw.io, Mermaid, Excalidraw); if you use one, embed the exported image or Mermaid source in the repo.

## Problem statement

### Part A — Pick two contrasting product goals

Pick **two** product goals from the list below (or propose your own, so long as the two are genuinely different in shape). The contrast is the point: applying the same rubric to two similar goals does not surface the trade-offs; applying it to two contrasting goals does.

Suggested goals (pick two):

1. **Support-ticket routing for a mid-sized SaaS company.** ~20 000 tickets/day; must route to one of 12 queues in < 200 ms p95; 90 % English, 8 % Spanish, 2 % other; regressions on the "billing" queue are especially costly.
2. **Contract-clause extraction for an in-house legal team.** ~500 contracts/week; must identify 15 canonical clause types with character-level spans; contracts are confidential; downstream reviewer needs highlighted PDFs.
3. **Doctor-note-to-ICD-10 code suggestion.** ~5 000 notes/day at a regional health system; suggested codes are reviewed by a human coder; PHI cannot leave the on-prem VPC; must integrate with the existing EHR via HL7 FHIR.
4. **Multi-language customer-facing chat assistant for a travel-booking site.** 24×7, ~50 QPS, needs to answer questions from the help centre corpus and hand off complex issues to a human; supports en / es / fr / de / ja / zh.
5. **Earnings-call insights product for institutional investors.** Ingest daily earnings-call transcripts (~200/day during peak seasons), extract structured events (guidance changes, executive commentary sentiment), publish to a subscriber-facing feed; publishing SLA is 15 minutes post-call-end.
6. **Scientific-literature triage for a pharma discovery team.** Nightly ingest of new PubMed and biorxiv entries; extract gene / disease / drug mentions with canonical IDs; produce a ranked list of "papers worth reading today" per researcher's tracked topics.

Write `products.md` with a one-paragraph statement of each chosen goal, why you find the two contrasting, and what the "user-visible action" is per system.

### Part B — Fill in each blueprint

For each product goal, produce a `blueprints/<slug>.md` that populates the chapter 1 template in full:

- **Product goal.** Decision or action enabled; user-visible surface; primary and secondary users; inputs (with expected volume, length, language mix, quality); outputs (schema, consumer, downstream action); latency / throughput / cost budget with numbers; failure-mode costs, symmetric or asymmetric.
- **Task framing (per stage).** Break the system into stages; per stage, name the shape (classification / extraction / generation) and justify. Do not blur shapes — a "chat assistant" is *not* one stage; it is retrieval + generation + safety filter, minimum.
- **Model-family choice (per stage).** Encoder / encoder-decoder / decoder-only; candidate model IDs (real ones — `microsoft/deberta-v3-base`, `google/flan-t5-large`, `mistralai/Mistral-7B-Instruct-v0.2`, `openai/gpt-4o-mini` etc.); one-paragraph rationale that references the task shape.
- **Language coverage.** Target languages, expected traffic split, posture (monolingual / multilingual / language-routed), rationale, per-language quality bar.
- **Build vs. buy (per stage).** buy-managed-API / buy-open-weight-self-host / adapt-fine-tune / build-from-scratch; rationale referencing the domain and QPS; at least one *alternative considered* per stage.
- **Augmentation.** Single-shot / RAG / tool-use / agentic; rationale referencing the knowledge-freshness and grounding requirements.
- **Non-goals.** At least three explicit non-goals — things this system is *not* trying to do. Boundary conditions the team can point to when scope creeps.
- **Open questions.** At least three questions that a design review would need to resolve before implementation.

Every non-trivial choice has a rationale of at least a sentence, not a name. Blueprints without rationales fail this exercise.

### Part C — Contrast the two blueprints

Write `contrast.md` — a one-page comparison that explicitly names where the two blueprints landed *differently* and why:

- Which stages are classification in one and generation in the other, and what forced that difference (the shape of the output, the label space size, latency budget)?
- Which stages are managed-API in one and open-weight-self-host in the other, and what forced that (regulated data, QPS, licensing, cost)?
- Which stage in one system does not exist at all in the other (a segmentation stage, a redaction stage, a citation-grounding stage), and what does its absence mean about the product?

Contrast makes the rubric — the differences are the decision points.

### Part D — Distil the rubric

Write `rubric.md` — the reusable, human-readable decision heuristics you now believe are correct after having done the work twice. Structure as a decision tree or a numbered checklist. Suggested sections:

- **Task framing.** Under what conditions do you pick classification, extraction, generation? Signal words / product-shape indicators for each.
- **Model family.** Under what conditions do you pick encoder, encoder-decoder, decoder-only? What is the default and when do you override it?
- **Language posture.** How do you decide monolingual vs. multilingual vs. language-routed given a traffic split?
- **Build vs. buy.** The 2×2 (task-shape × QPS) or 3×3 heuristics the exercise makes you defend.
- **Augmentation.** When does the answer need to be grounded, and by what mechanism?

The rubric is 1–2 pages, not a chapter — the goal is a working reference, not a textbook.

### Part E — Peer-review checklist

Write `review-checklist.md` — a set of yes/no questions a reviewer would ask when reading someone else's blueprint. Aim for 15–20 questions. Examples:

- Is the user-visible action stated in one sentence?
- Is every stage's task shape (cls / ext / gen) explicit?
- Does every build-vs-buy stance name at least one alternative considered?
- Are the non-goals explicit and enforceable?
- Is the language traffic split expressed as numbers, not adjectives?

The checklist is what a reviewer uses to catch the blueprint that skipped Part D's discipline.

## Starter guidance

- **Do not confuse a single system with a single stage.** Even the simplest chat assistant has at least three stages (retrieval / generation / safety); a routing system usually has two (classification / handoff). The blueprint must reflect the pipeline, not the marketing description.
- **Numbers, not adjectives.** "Fast" is not a latency budget; "sub-200 ms p95 for 20 000 requests/day" is. Push yourself to invent plausible numbers where you do not know real ones — the exercise is calibration, not literal accuracy.
- **The rationale is the exercise.** A blueprint that names "decoder-only, Mistral 7B" without reasoning is worth zero. A blueprint that names "encoder, DeBERTa-v3-base — task is classification into a stable 12-label set with 20k training examples, decoder-only would work but cost 10× more per request and not improve on the F1 achievable with a supervised head" is worth the exercise.
- **Non-goals are the safeguard against scope creep.** Force yourself to name at least three per blueprint. Reviewers a year later will thank you.
- **Use real model IDs.** `mistralai/Mistral-7B-Instruct-v0.2`, `sentence-transformers/all-mpnet-base-v2`, `google/flan-t5-large`, `dslim/bert-base-NER`, `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-6`, etc. The exercise is not "propose a model"; it is "propose a defensible choice from the model landscape as it actually exists."
- **Diagram if it helps you and skip if it doesn't.** A one-page block diagram of the pipeline is worth including for a reviewer; making it pretty is not.

## Acceptance criteria

- [ ] `products.md` states the two chosen product goals in a paragraph each with a clear user-visible action, and names why the two contrast.
- [ ] Two blueprints under `blueprints/` populate every field of chapter 1's template, with per-stage rationale (not just names).
- [ ] Each blueprint declares at least three explicit non-goals and at least three open questions.
- [ ] Every build-vs-buy stance names at least one alternative considered.
- [ ] Language coverage is expressed as a numeric traffic split with a posture and rationale.
- [ ] `contrast.md` names at least five points where the two blueprints landed differently and explains the driver.
- [ ] `rubric.md` fits on 1–2 pages, is decision-tree or checklist shaped, and covers task framing, model family, language posture, build-vs-buy, and augmentation.
- [ ] `review-checklist.md` has 15–20 yes/no questions a reviewer would apply to a peer's blueprint.

## Stretch goals

- **A third blueprint from your day job.** Apply the same template to a real product you or your team is building. Redact sensitive details but keep the shape. This is where the exercise pays off; the artificial-goal blueprints are practice for the real one.
- **Blueprint versioning.** Save `v1` of one blueprint. Deliberately change one input (say, "traffic scales 10×" or "regulated-data classification changes"). Author `v2` and diff — what stances changed, what stayed. The diff is the exercise's meta-learning: which decisions are brittle to input changes.
- **Rubric review with a colleague.** Trade rubrics with someone else who did the exercise. Where do you disagree? A one-page `rubric-diff.md` capturing the disagreements is the substantive product.
- **Mermaid diagrams.** For each blueprint, add a Mermaid `flowchart` showing the pipeline stages and their data types. A reviewer skimming the repo should see the shape in 10 seconds.
- **Adversarial reframing.** Take one of your two blueprints and rewrite it under the constraint "the entire system must be one managed-API call" and again as "the entire system must be self-hosted on one server." What breaks under each extreme? Adversarial reframing surfaces the load-bearing decisions.

## Deliverables

```
products.md
blueprints/
    <goal-1-slug>.md
    <goal-2-slug>.md
contrast.md
rubric.md
review-checklist.md
README.md                 # what you built, how to read it, and the choices you made
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
