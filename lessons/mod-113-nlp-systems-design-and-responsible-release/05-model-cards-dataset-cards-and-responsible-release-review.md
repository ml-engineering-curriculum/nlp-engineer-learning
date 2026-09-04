# Model Cards, Dataset Cards, and the Responsible-NLP Release Review

## Motivation

An artifact ships with paperwork. The paperwork is not overhead — it is the interface the model presents to every reviewer, integrator, and future oncall who was not in the room when the model was trained. A model card, the dataset cards it depends on, and a responsible-release review are the three artifacts that translate the release-gate results (chapter 3), the budget trade-offs (chapter 2), and the domain-governance constraints (chapter 4) into a document a governance / risk / security partner can sign off on.

This chapter is the final NLP-engineer-side layer. Depth on the risk / compliance / policy side lives in upstream tracks (mod-114-and-beyond). The engineer's job is to write cards and reviews that satisfy those upstream tracks without requiring endless clarification calls — and that inherit their claims from the same automated release harness that produced the release-gate results.

## The three-artifact chain

- **Dataset card** — one per training / eval dataset used. Describes the data: what it is, where it came from, who created it, how it was collected, what licence applies, what biases and limitations are known. Ships with the dataset.
- **Model card** — one per released model. Describes the model: what it does, on what data it was trained, how it was evaluated, on what slices, with what known failure modes and out-of-scope uses. Ships with the model.
- **Responsible-release review** — one per production release. Aggregates the model card, the dataset cards, the release-gate results (chapter 3), the domain-governance analysis (chapter 4), and the sign-offs from the reviewing parties. Lives with the release ticket.

Each artifact is the input to the next. A model card that cannot cite a dataset card cannot be signed off; a release review that cannot cite a model card cannot pass governance. The chain is what makes claims *auditable*.

## The dataset card

The foundational reference is Gebru et al., ["Datasheets for Datasets"](https://arxiv.org/abs/1803.09010) (*Communications of the ACM 2021*). The Hugging Face dataset-card spec ([Dataset Cards documentation](https://huggingface.co/docs/hub/datasets-cards)) is the industry-adopted concrete template; the Google [Data Cards Playbook](https://sites.research.google/datacardsplaybook/) is a complementary practitioner guide.

### What a dataset card must answer

Structured by the questions from Gebru et al., adapted for NLP:

- **Motivation.** Why was the dataset created? Who funded it? What gap does it fill?
- **Composition.** What does each instance represent (document, sentence pair, span-annotated record)? How many instances? Are labels present, and how are they defined? Is there missing data or are examples redacted?
- **Collection process.** How was the data acquired (scraped, licensed, human-generated, machine-generated)? Over what time window? From what sources? Was consent obtained where relevant? Was human labour involved, under what conditions?
- **Preprocessing / cleaning / labelling.** What transformations were applied? What was filtered out and why? What was the labelling protocol? What was inter-annotator agreement (mod-111 chapter 9)?
- **Uses.** What tasks has the dataset been used for? What tasks should the dataset **not** be used for? Are there groups the dataset systematically under-represents?
- **Distribution.** How is the dataset licensed? Under what conditions can it be redistributed? Are there export-control or regulatory constraints?
- **Maintenance.** Who maintains the dataset? Is there a versioning policy? Where are corrections tracked?

The card is not aspirational — every field is a factual claim, and every claim is either supported by evidence in the pipeline (row counts, hash of the file, licence file) or is explicitly marked as "unknown" or "to be characterised." Empty fields are worse than "unknown" — they read as "nobody checked."

### Concrete NLP-specific fields worth adding

- **Language distribution.** Per-language row count with method of language ID (fastText, cld3, or manual).
- **Text-length distribution.** Character / token length p50, p95, p99; the eval consumer needs this to know if the dataset covers their length regime.
- **Label distribution.** Per-label counts; for imbalanced sets, mention the imbalance ratio and the sampling / weighting expected downstream.
- **PII assessment.** Was the dataset scanned for PII / PHI? By what tool, against what taxonomy, at what recall? For domain data, the assessment is a compliance artifact (chapter 4).
- **Temporal window.** What time range does the source data cover? Downstream: any deployment predicting on more recent data is out-of-distribution to some degree.
- **Known biases and demographic representation.** What demographic groups are over- / under-represented in the source population; what dialects, registers, or genres are missing; what the dataset was **not** designed to represent.

### The dataset-card chain

Real training corpora are almost always **composed** — pretraining base + domain adaptive pretraining corpus + task fine-tuning set + eval set. Each has its own card. The composed system's dataset card is a *directed graph* of cards:

```
[base_pretrain_card] ──► [domain_pretrain_card] ──► [task_finetune_card]
                                                          │
                                                          ▼
                                                    [eval_card]
```

The model card (below) references the leaves and the root of that graph, so a reviewer can trace any prediction back to the data lineage. This is the point of the chain — a system whose model card cites datasets whose provenance is untraceable does not survive a governance review.

## The model card

The foundational reference is Mitchell et al., ["Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993) (*FAT\* 2019*). Hugging Face model cards ([Model Cards documentation](https://huggingface.co/docs/hub/model-cards)) are the practical template. The [Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook) walks the fields.

### The mandatory fields

- **Model details.** Name, version, developed by, model type (encoder / seq2seq / decoder, parameter count), base model (if fine-tuned), languages, licence, citation.
- **Intended use.** Primary intended uses; primary intended users; out-of-scope uses (explicitly enumerated — the "do not use this for X" list is as important as the "use this for Y" list).
- **Training data.** Reference each training dataset's card by name and version. Never inline the dataset description; link to it.
- **Evaluation data.** Reference the eval dataset cards. Note whether the eval set is held-out from the training data, whether contamination has been checked (mod-111 chapter 8), and what the eval-set collection date is.
- **Metrics.** Which metrics were used (mod-111 chapter 1) and why. For each metric: the aggregate value and the paired-significance CI (mod-111 chapter 7). Not just "F1 = 0.87" — "F1 = 0.87, 95 % bootstrap CI [0.85, 0.89]".
- **Slice-wise results.** The per-slice matrix from the slice gate (chapter 3). This is the single most important field for a fairness / safety reviewer.
- **Ethical considerations.** Known risks — hallucination, disparate impact, dual-use, safety, privacy. What has been done to mitigate them. What remains open.
- **Caveats and recommendations.** Under what conditions the model behaves poorly; what the user should do (RAG grounding, human review, ensemble, ...).

### NLP-specific fields worth adding beyond the Mitchell et al. template

- **Model-family and architecture rationale.** From chapter 1's blueprint — why encoder / seq2seq / decoder, why this size.
- **Tokeniser.** Which tokeniser, which vocabulary, any special tokens added. Tokeniser mismatch is a silent bug (mod-112 chapter 4 and chapter 7); the model card is where you commit to the tokeniser version.
- **Latency and cost profile.** p50 / p95 / p99 under a reference load and hardware; cost per 1 000 tokens. Chapter 2's cost-per-request. A reviewer signing off on operational readiness needs these on the card.
- **Robustness / red-team results.** Performance on any internal or public adversarial / robustness suite; jailbreak / prompt-injection resistance for generative models. Publish the *methodology* and the *results*, even if the results are uncomfortable — a card that only shows favourable numbers is a card no reviewer trusts.
- **Environmental footprint.** Total training-run CO₂e estimated via a tool such as [MLCO2's Machine Learning Impact calculator](https://mlco2.github.io/impact/) or [CodeCarbon](https://codecarbon.io/). Not every organisation reports this, but a growing number of governance regimes ask for it.
- **Update / retirement policy.** How often the model is retrained; how long the current version is supported; when the org intends to deprecate it.
- **Contact.** Who owns the model. A card without an owner is a card no reviewer can escalate to.

### The automated model card

The best model cards are **generated** from the release harness, not typed by hand. Every field that can be automated should be — from the manifest (mod-112 chapter 5) for artifact metadata, from the eval harness for metrics and slice results, from the profiler for latency, from the dataset-card chain for training-data references. The free-text fields (intended use, out-of-scope, ethical considerations) are the human-authored part.

A pattern that works: a `MODEL_CARD.md.template` in the repo, with `{{jinja}}` placeholders that the release script populates from the manifest and eval outputs; the human-authored sections are separate `.md` files included via template partials. The rendered card ships with the artifact.

## The responsible-NLP release review

The release review is the meeting-and-document combo where governance / risk / security partners sign off on the release. The engineer's job is to make the review efficient — to walk in with a document that answers 90 % of the questions before they are asked.

### The review artifact structure

A working template:

```markdown
# Responsible release review: <system name> <version>

## System summary
- Blueprint (chapter 1): link
- Budget (chapter 2): link, notable trade-offs summarised in one paragraph
- Domain (chapter 4): named domain (clinical / legal / financial / scientific / general), applicable regulatory envelope

## Release-gate results (chapter 3)
- Offline gate: pass / fail per check
- Slice gate: pass / fail per declared slice, with any exceptions flagged
- Operational gate: latency / cost / runtime / interface parity
- Live gate: canary plan and observed results (if pre-review; forecast if pre-canary)

## Artifact chain
- Model card (linked): version, sha, generated date
- Dataset cards (linked, with graph): pretrain → domain-adapt → fine-tune → eval

## Governance-partner sign-offs
- Privacy: assessment, sign-off, open items
- Security: assessment, sign-off, open items
- Legal / compliance: assessment, sign-off, open items
- Fairness / responsible-AI: assessment, sign-off, open items
- Product owner: sign-off

## Risk register
- Enumerated known risks (from the model card's "ethical considerations" plus domain-specific — chapter 4)
- Per risk: severity, likelihood, mitigation in place, residual risk

## Rollout and rollback plan (chapter 3)
- Rollout shape, canary %, soak duration, live-gate thresholds
- Rollback shape, SLA, non-rollbackable changes explicitly listed
- Owners for each phase

## Post-launch commitments
- Monitoring surface (mod-112 chapter 6) with thresholds
- Retraining cadence
- Re-review trigger conditions (drift beyond X, incident, regulatory change)
```

Every section is a claim; every claim links to its source. A section marked "TBD" is a review-blocker.

### Governance-partner sign-offs: what they actually want

The four partners each read the artifact with a different lens.

- **Privacy** asks: what data does the model see at training and at inference; was there consent and lawful basis; can a data subject exercise deletion rights; is there PII in the model weights (memorisation risk); is the data-flow to any managed API cleared. The dataset cards, PII assessment, and blueprint's data-flow diagram are the artifacts they read.
- **Security** asks: what is the attack surface (prompt injection, model extraction, membership inference, adversarial examples); how are model weights and training data protected at rest and in transit; is the serving stack hardened; what is the incident-response plan. Mod-112 chapter 7 and the release-review's rollback plan are the artifacts they read.
- **Legal / compliance** asks: are all training-data licences compatible with the deployment; does the deployment trigger regulatory classification (SaMD / MDR, MAR surveillance, ECOA fair-lending); are outputs treated as advice or information; who is liable when the model is wrong. Chapter 4's domain governance envelope is the artifact they read.
- **Fairness / responsible-AI** asks: on what slices was the model evaluated; where does it under-perform; how are the failure modes disclosed to end users; is there human oversight where the decision is high-stakes; are there known dual-use concerns. The slice-gate matrix, the red-team results, and the "ethical considerations" section of the model card are the artifacts they read.

A release review that anticipates the four lenses reduces the meeting to a quick walk-through and a set of sign-offs. A release review that leaves the partners to reconstruct claims from the code produces a multi-week loop.

### The escalation and exception process

Not every release passes every check. Real exceptions are common — a slice regressed but the aggregate improved; a licence is ambiguous but the product need is real; a latency budget is exceeded but a feature was requested. The exception process is what turns those into documented decisions, not silent passes.

A working exception process:

- **Named exception.** "The Spanish slice regressed by 2 F1; the aggregate improved by 0.6 F1. We are accepting the Spanish regression because [rationale]. Owner: [name]. Follow-up: retrain with additional Spanish data by [date]."
- **Documented in the release review.** The exception is a section, not a footnote. Signed off by the partner whose lens the exception touches (fairness partner for the slice example).
- **Tracked to a mitigation deadline.** Every exception has a "when does this stop being an exception" date. Untracked exceptions accumulate into permanent debt.

## The Data Cards and Model Cards Playbook: what actually happens in practice

Two field-observation notes from teams that have adopted this discipline at scale:

- **The cards are only as good as the release harness feeds them.** Hand-typed model cards drift within one release cycle — a number gets stale, a slice gets renamed, an eval set gets swapped and the card still cites the old one. Automate everything that can be automated; the free-text sections should be short enough that keeping them fresh is trivial.
- **The review is only as good as the partners' access.** A responsible-AI reviewer who has to file a ticket to see the training data cannot review the training data. Partners need read-access to the dataset cards, the model card, the manifest, and the eval-set metrics — as first-class artifacts of the ML platform, not through a request queue.

References worth internalising: the [Google Data Cards Playbook](https://sites.research.google/datacardsplaybook/) and the [Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook) are the closest things to practitioner playbooks; the NIST AI Risk Management Framework ([NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework)) is the reference the US-side governance partners will cite. The EU AI Act ([Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)) is the reference for high-risk AI systems deployed into the EU market and is the framework EU compliance partners will map back to.

## Anti-patterns

Three worth naming so the exercise 3 review artifact avoids them:

- **The aspirational card.** Fields describe what the model *should* do rather than what it *does*. Reviewers stop trusting the card after the first mismatch. Fix: every claim is a factual statement supported by an artifact; "unknown" is preferable to "aspirational."
- **The one-time review.** A big review happens for the launch; every subsequent release ships without governance review because "it's the same model." Silent scope creep. Fix: define re-review triggers (chapter 3's rollback SLA + drift thresholds) and enforce them.
- **The review as gatekeeping theatre.** The review is a meeting where nobody reads the artifact and sign-off is a rubber-stamp. Fails at the first incident, when the sign-offs are asked to account for a decision. Fix: pre-review reading time is expected; empty fields are review blockers; the sign-off implies the signer read the section.

## Chapter summary

- The responsible-release artifact chain is **dataset card(s) → model card → release review**. Each artifact cites the previous; a break in the chain is a review blocker.
- **Dataset cards** answer the Gebru et al. datasheet questions (motivation, composition, collection, preprocessing, uses, distribution, maintenance) plus NLP-specific fields (language distribution, length distribution, label distribution, PII assessment, temporal window, known biases).
- **Model cards** answer the Mitchell et al. questions (details, intended use, training / eval data, metrics with significance, slice-wise results, ethical considerations, caveats) plus NLP-specific fields (family rationale, tokeniser, latency + cost, red-team results, environmental footprint, retirement policy, contact). Generate from the release harness; the free-text is the human-authored part.
- **Responsible-release review** = system summary + release-gate results + artifact chain + governance-partner sign-offs (privacy / security / legal / fairness) + risk register + rollout/rollback plan + post-launch commitments. Every claim links to its source; every empty field is a blocker.
- The four governance partners each read with a different lens; a review that anticipates the four lenses reduces the meeting to a walk-through.
- Exceptions are documented, signed by the partner whose lens they touch, and tracked to a mitigation deadline. Untracked exceptions accumulate into permanent debt.
- References: Gebru et al. Datasheets, Mitchell et al. Model Cards, HF Model / Dataset Card specs, Google Data Cards Playbook, NIST AI RMF, EU AI Act. Depth on the risk / policy side is handed off to upstream tracks; the engineer's contribution is the automated, honest, well-linked artifact chain.
- Anti-patterns to avoid: the aspirational card, the one-time review, the review as gatekeeping theatre.
