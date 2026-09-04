# exercise-04: Domain NLP Patterns Survey

**Estimated effort:** 3 hours

## Objective

Produce a **domain survey artifact** for one of the four domains covered in chapter 4 — clinical, legal, financial, or scientific — that maps the canonical tasks, canonical stacks, canonical benchmarks, governance envelope, and a **worked example** of a production-ready NLP-system blueprint sensitive to the domain's constraints. Deliver: (1) `survey.md` covering the domain in the chapter-4 template plus depth; (2) a **domain data-flow diagram** naming where PII / PHI / MNPI / privileged / confidential data lives and where it *cannot* go; (3) a **domain-specific model + dataset card sketch** for a system on which the chosen domain's regulations bite; (4) a **compliance-question checklist** a governance partner from the domain would ask you.

## Prerequisites

- Chapter [04 — Domain NLP patterns and their data-governance constraints](../04-domain-nlp-patterns-and-governance.md) and [05 — Model cards, dataset cards, and the responsible-NLP release review](../05-model-cards-dataset-cards-and-responsible-release-review.md).
- Familiarity with encoder / seq2seq / decoder distinctions (mod-101–108).
- No coding required. Access to primary regulatory sources for whichever domain you choose:
    - **Clinical:** HIPAA guidance ([HHS de-identification methods](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/)), MIMIC / n2c2 DUAs, FDA SaMD, EU MDR.
    - **Legal:** ABA Formal Opinion 512 on generative AI, CUAD paper + repo, LexGLUE paper.
    - **Financial:** SEC / FINRA guidance, EU MAR, OFAC sanctions programs, FinBERT paper.
    - **Scientific:** PMC OA licensing, publisher terms, Retraction Watch database, UMLS licence.

Cite the primary source when you make a compliance claim.

## Problem statement

### Part A — Pick the domain and scope the survey

Pick **one** domain. Write `scope.md`:

- One paragraph naming the domain, the sub-area if the domain is broad (e.g. "clinical NLP, sub-area: automated ICD-10 coding from discharge summaries at a US hospital system"), and why that sub-area matters commercially or scientifically.
- One paragraph naming the audience — who is the survey *for*. An engineer starting a project in the domain? A tech-lead evaluating whether the org should enter the domain? A governance partner deciding whether to approve a proposed system?

Different audiences read the survey differently. Naming the audience keeps the depth calibrated.

### Part B — Author the survey

Produce `survey.md`, structured as:

1. **Canonical tasks.** The 3–6 tasks the domain is most often building. Per task: definition, canonical benchmark(s), canonical corpus(es), canonical metric(s). Include for each: is this a classification / extraction / generation task (chapter 1's framing), and why.
2. **Canonical stacks.** The pipelines and models the domain has consolidated on. Include:
    - Open-weight domain-adapted models (BioBERT / LegalBERT / FinBERT / SciBERT and successors).
    - Managed offerings (Amazon Comprehend Medical, Google Cloud Healthcare NLP, Azure Text Analytics for Health, John Snow Labs, Bloomberg / Refinitiv APIs, etc.), noting their compliance posture.
    - Domain-specific tooling (cTAKES, MedSpaCy, scispaCy, LexNLP, sec-edgar-tools) — the utilities most projects reach for.
3. **Corpora and their licences.** Enumerate the corpora a real project would train / evaluate on. Per corpus: source, size, licence, redistribution constraints, cost / paperwork required to obtain. If the domain has famous "closed" corpora (i2b2, LDC-licensed data, publisher subscription content), name them and their restrictions.
4. **Governance envelope.** The regulations, standards, and professional-responsibility frameworks. Per regulation: what it constrains, what triggers applicability, what the enforcement regime looks like. Cite the primary regulatory source (not a blog post).
5. **Failure modes with real accountability.** Named incidents, sanctioned firms, retracted claims, or well-known near-misses that a project in this domain would be foolish to ignore. Cite the source. If you cannot find a well-documented incident, name the *type* of failure and mark `<!-- needs-research: representative documented incident -->`.
6. **What "buy" looks like.** For each stage a project in this domain would build, which managed service is competitive, whether it clears the domain's data-flow review, and what the paperwork burden is.
7. **Signals that a project is not ready.** The red flags a domain expert would see in a system proposal — "the team is prototyping on public GPT-4o against real PHI", "the corpus is undocumented", "the model output flows to a clinician without a coder review step". A checklist of things a mature domain lead would say "no" to.

Aim for 4–6 pages of survey. Prioritise **substantive detail** over encyclopedic coverage — a survey that names one benchmark per task with the citation and one paragraph of characterisation beats a survey that lists ten benchmarks with no context.

### Part C — Data-flow diagram

Produce `data_flow.md` with either an embedded Mermaid diagram or a linked draw.io / Excalidraw export. The diagram must show, for a hypothetical production system in the domain:

- Where the raw data originates (EHR, contract-management-system, EDGAR filings, publisher API, ...).
- Every hop the data takes: ingest, storage, preprocessing, model inference, output storage, downstream consumers.
- The **data classification** at each hop (PHI / PII / MNPI / privileged / confidential / public), colour-coded or otherwise annotated.
- The **boundary** across which the data may not flow (organisational, jurisdictional, VPC-level).
- The **audit trail** — where every prediction is logged with provenance.

A supplementary paragraph describes what would happen if a component on the diagram tried to send data across the boundary — what the technical control is, what the compliance response is, what the incident looks like.

### Part D — Domain-sensitive model + dataset card sketches

Produce `MODEL_CARD.md` and `DATASET_CARD.md` for a hypothetical system in the domain — a de-identification model for clinical notes, a clause-extraction model for MSAs, a corporate-event-extraction model for 8-Ks, an adverse-event-extraction model for FAERS reports. Populate every field of chapter 5's model card and Gebru et al. dataset card templates, with domain-specific answers:

- The "intended use" and "out-of-scope use" reflect the domain's constraints (a clinical de-id model is *not* a legal-anonymisation model even though the output looks similar).
- The "training data" field cites a real domain corpus with its real licence.
- The "ethical considerations" field names the domain's specific risks (hallucinated ICD codes, hallucinated case cites, hallucinated ticker moves, hallucinated gene mentions).
- The "caveats and recommendations" field names the human-review step that keeps the model output safe in the domain.

You are not building the model; you are producing the card that would ship with it. The card is judged on whether a domain governance partner would sign it off, not on whether the numbers are real. Where a number would be measured on a real training run, use a placeholder and mark it clearly: `TBD: measured on eval-run-id ABC`.

### Part E — Compliance-question checklist

Produce `compliance_checklist.md` — 15–25 yes/no questions a governance partner in this domain would ask during a release review of the hypothetical system from Part D. Cluster by lens (privacy / security / legal / fairness — chapter 5's four partners). Each question is answerable "yes / no / mitigation" with a link back to the artifact where the evidence lives.

Sample question shapes (fill in for your domain):

- Privacy: "Are all training records covered by a DUA authorising the training use?" "Is the model retention policy documented, and does it allow deletion in response to a data-subject request?"
- Security: "Are model weights and eval outputs encrypted at rest with organisational keys?" "Is the serving stack in the same trust zone as the data source?"
- Legal: "Does the deployment context trigger regulatory classification (SaMD / MAR surveillance / ECOA)?" "Are hallucinated outputs a legally-cognisable harm in the deployment context?"
- Fairness: "Is the eval-slice matrix populated across the demographics the system will serve?" "Is there a documented human-review step for high-stakes outputs?"

The checklist is what a governance partner brings to a meeting; you build it in advance so the meeting is efficient.

### Part F — Reflection

Write `reflection.md` (1 page):

- One paragraph on what surprised you about the domain compared to your prior mental model.
- One paragraph on what the survey missed — the questions you tried to answer but could not, and what a next round of research would need.
- One paragraph on what a comparable survey in a *second* domain would need to add or drop. This is the meta-learning that makes the exercise generalisable.

## Starter guidance

- **Cite primary sources.** Regulatory guidance from the regulator; papers from arXiv / venue proceedings; corpora from the organisation that hosts them; incidents from court filings or first-party reports. Blog-post citations are acceptable only for practitioner-community references (e.g. Retraction Watch); they are not acceptable for regulatory or legal claims.
- **Use `<!-- needs-research: ... -->` liberally.** If you cannot verify a fact (a specific FDA classification, a named incident's outcome, a corpus's current licence terms), mark it — do not invent. The exercise is calibrated survey work, not confabulation.
- **The data-flow diagram is the single most valuable artifact.** A domain governance partner reads it first. Spend disproportionate time making it clear.
- **The model / dataset cards do not have to be real numbers.** Where you would run an eval and populate a number, write a placeholder + eval-run-id. The card is judged on *shape* and *fields covered*, not the specific value.
- **The compliance checklist doubles as your test of Part D.** Every question in the checklist should be answerable from Parts B / C / D. If a question does not have an answer, you missed a section — go back and add it.
- **Do not attempt two domains in one exercise.** The survey depth is the exercise; a shallow two-domain survey is worth less than a deep one-domain survey. The stretch goal below covers the second-domain case.

## Acceptance criteria

- [ ] `scope.md` names the domain, the sub-area, and the audience the survey is for.
- [ ] `survey.md` covers canonical tasks, canonical stacks, corpora + licences, governance envelope, failure modes with real accountability, buy-side options with compliance posture, and readiness red-flags — 4–6 pages, primary sources cited.
- [ ] `data_flow.md` includes a diagram showing data provenance, hops, classifications, boundaries, and the audit trail, with a supplementary paragraph on boundary-crossing responses.
- [ ] `MODEL_CARD.md` and `DATASET_CARD.md` populate every field from chapter 5's templates, with domain-specific answers; placeholders marked clearly where a real run would produce a number.
- [ ] `compliance_checklist.md` has 15–25 questions clustered by governance lens (privacy / security / legal / fairness), each answerable "yes / no / mitigation" against the artifacts in Parts C–D.
- [ ] `reflection.md` covers surprises, gaps, and what a second-domain survey would need to change.
- [ ] Primary sources cited, and every unverifiable claim marked `<!-- needs-research: ... -->` rather than invented.

## Stretch goals

- **Two-domain contrast.** After completing the first domain, do an abbreviated pass on a *second* domain — just the canonical tasks, canonical stacks, and governance envelope. Write `contrast.md` — what carries over unchanged, what is domain-specific, what surprised you about the second when you already had the first.
- **Compliance-partner interview.** If you have access to a compliance / privacy / legal partner in the chosen domain (even a friend-of-a-friend), send them Part E's checklist and get their reaction. Their answers are the ground-truth signal your survey is aimed at. Redact and summarise their feedback in `partner_review.md`.
- **Real corpus onboarding.** For the chosen domain, walk through the actual paperwork to obtain the reference corpus (MIMIC-IV credentialing, LDC DUA, CUAD download, PMC OA bulk download). Do not sign a DUA on the org's behalf without authorisation; the exercise is to enumerate the paperwork, not sign it. Attach the checklist as `corpus_onboarding.md`.
- **Historical-incident deep dive.** Pick one well-documented incident in the domain (Mata v. Avianca for legal; a specific SaMD FDA action for clinical; a named surveillance-system failure for financial; a well-known retracted paper affair for scientific). Author `incident.md` — what happened, what artifacts existed / were missing, what the exercise's compliance checklist would have caught. The incident is the acid test for the checklist.
- **A blueprint tuned to the domain.** Take one of your exercise-01 blueprints (or author a new one) and rewrite it under the constraints from this domain's governance envelope. `blueprint_domain.md` — what changed, what stances flipped, what non-goals became explicit.
- **A dataset-card chain diagram.** Sketch the DAG of dataset cards for a real training programme (pretrain → domain-adapt → task-finetune → eval) with each corpus's licence and DUA status noted. Chapter 5 named this graph as the shape you need; making it concrete is the exercise.

## Deliverables

```
scope.md
survey.md
data_flow.md
MODEL_CARD.md
DATASET_CARD.md
compliance_checklist.md
reflection.md
README.md                 # how to read the repo, choices made
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
