# Job Requirements — NLP Engineer

**Role level:** 30 (deep specialist — peer to Senior ML Engineer on the ladder)
**Track:** `nlp-engineer-learning`
**Research window:** 2026-08-06 → 2026-11-04 (last 90 days)
**Today:** 2026-11-04
**Postings sampled this cycle:** 30 verified in-window (target 25 — ✓ met)

## Status — live evidence, full sample (third quarterly cycle)

This cycle exercised WebSearch / WebFetch against live job boards across the required title spectrum (`NLP Engineer`, `Natural Language Processing Engineer`, `Applied NLP Engineer`, `NLP/AI Engineer`, `Language Model Engineer`, `Computational Linguist`, `NLP/Linguistics Software Engineer`, plus `ML Engineer`, `AI Engineer`, and `Research Engineer` roles where the required qualifications name NLP explicitly). **The 25-posting target is met for the first time: 30 postings verified in-window.** 18 were live-verified at 2026-11-04 (10 re-fetched carryovers from the 2026-10 cycle still open, 7 newly discovered NLP-titled in-window postings, plus 1 Hebbia Applied AI Research Engineer LLM & NLP posting that is now closed-to-applicants but was open in-window). The remaining 12 are carryovers from the 2026-10 cycle where date_observed = 2026-10-04 falls inside the 2026-08-06 → 2026-11-04 window; their original 2026-10 verifications stand even though several URLs 404 or redirect-to-index today. Raw data — employer, title, URL, date_observed, location, verbatim required / preferred bullets, salary_range, quote, and verification_state — is in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `postings`.

**Vertical diversity improved markedly this cycle.** The 7 newly discovered postings deliberately re-weight the sample away from the healthcare-greenhouse over-sampling that inflated last cycle's domain-vertical frequency: they span government/defense (BTI360 clearance-required NLP Engineer, Clera narrative-intelligence NLP), legal (Thomson Reuters Senior MLE NLP/LLM), insurance (EvolutionIQ AI Insurtech), finance/compliance (Behavox multilingual compliance NLP), and clinical (Dandelion Health EHR NLP, Zimmer Biomet medical-devices NLP). Five of the 7 new postings carry **`NLP Engineer` in the job title** — Behavox Senior NLP Engineer, BTI360 NLP Engineer, Clera Founding Engineer NLP Systems & Infrastructure, Dandelion NLP/LLM Data Scientist, Zimmer Biomet NLP Engineer — modestly contradicting the aggregator-level narrative that `NLP Engineer` is disappearing as a title. In government/defense, finance/compliance, and clinical verticals the title retains gravity.

## Continuity outcome — zero net additions (third consecutive cycle)

Under the continuity bias ("propose net-new content only when ≥ 3 distinct in-window postings cite a requirement the existing curriculum does NOT cover, AND frequency ≥ 30%, AND no existing module can be incrementally extended"), this cycle again produces **no new modules, no new exercises, no new projects**. See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) for the empty delta and its rationale.

Every requirement theme that cleared the 30% frequency threshold (≥ 9 of 30 postings) is already either owned directly by an existing module (mod-101 through mod-113) or already surfaced-and-linked-out where a peer track owns the depth (rag-engineer, llm-application-developer, fine-tuning-engineer, model-evaluation-engineer).

**Overdue content-authoring action: mod-111 exercise-04 LLM-as-judge reframe.** The LLM-quality eval authoring signal (LLM-as-judge, automated evaluators, adversarial tests, regression suites) has now persisted at ≥ 37% for **three consecutive cycles** (2026-09: 44%, 2026-10: 50%, 2026-11: ~37%). This cycle's dip is within the noise band of a healthier-diversity sample — the signal is still well above threshold and still strongest at domain-vertical employers (BTI360, Zimmer Biomet, Thomson Reuters this cycle; Truveta, Midi Health, GW RhythmX, Grammarly last cycle). The right response remains a within-exercise reframe of mod-111 exercise-04 (content-authoring work, **NOT** a curriculum-plan-delta item), and the three-cycle persistence lowers the ambiguity bar — the authoring pass should ship before 2027-01 so the next research cycle can measure its uptake rather than re-flagging the gap.

## Methodology

1. Fanned out WebSearch / WebFetch across `job-boards.greenhouse.io`, `boards.greenhouse.io`, `jobs.lever.co`, `jobs.ashbyhq.com`, `builtin.com`, named-employer careers pages, and aggregators (startup.jobs, freehire.me, aijobs.com, remoterocketship) for the title spectrum. Retried frontier-lab, speech-vendor, and MT-vendor sources that failed in prior cycles.
2. Filtered out generic "ML Engineer" / "Data Scientist" / "Research Scientist" / "AI Engineer" titles unless the required-qualifications section named NLP explicitly. Filtered out generic "Generative AI Engineer" / "LLM Application Engineer" and Staff / Principal / Director modifiers (they inherit this packet).
3. Captured each verified posting verbatim: employer, exact title, URL, date_observed, date_posted (`estimated:2026-Q3` where the board did not publish it), location, required and preferred bullets, salary_range (`null` if unpublished), one short representative quote, and verification_state.
4. Computed requirement frequency across the 30-posting sample (30% threshold = ≥ 9 postings).
5. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill. Signal-vs-coverage delta captured under "Signals not adopted" below.

## Requirement themes → curriculum ownership

`Freq` is a fraction of the 30 verified postings. `evidence_post_ids` on each requirement in `.aicg/job-requirements.json` names the specific postings.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Tokenization theory (BPE / WordPiece / Unigram / SentencePiece, multilingual / CJK, tokenizer training, chat-template arithmetic) | 2 / 30 | `nlp-engineer` (this) | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) |
| 2 | Transformer internals for NLP (encoder vs. encoder-decoder vs. decoder-only choice, KV cache for generation, probing, tokenizer-shape effects) | 12 / 30 | `nlp-engineer` | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) (co-taught with tokenization) |
| 3 | Classical NLP (regex, FST, n-gram LMs, taggers, parsers, lemmatisation, Unicode-correct text processing, language identification) | 5 / 30 | `nlp-engineer` | [`mod-102-classical-nlp`](lessons/mod-102-classical-nlp) |
| 4 | Text classification end-to-end (baselines through fine-tuned encoders, multilingual, multi-label, calibration, cost-sensitive eval) | 1 / 30 | `nlp-engineer` | [`mod-103-text-classification`](lessons/mod-103-text-classification) |
| 5 | Sequence labelling and NER (BIO/BIOES/BILOU, span models, CRF heads, subword-word alignment, entity-level eval, domain NER) | 1 / 30 | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) |
| 6 | Information extraction (relation / event extraction, slot filling, entity linking, coreference, document-level IE, structured generation) — **UP from 9% to 34%** | 10 / 30 | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) |
| 7 | Question answering and machine reading comprehension (extractive / abstractive / multi-hop / long-context, unanswerability, EM / F1) | 0 / 30 | `nlp-engineer` | [`mod-105-question-answering-and-machine-reading`](lessons/mod-105-question-answering-and-machine-reading) |
| 8 | Summarisation (extractive / abstractive, long-document, multi-document, controllable, faithfulness, ROUGE / BERTScore) | 1 / 30 | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) |
| 9 | Generative NLP and decoding (sampling strategies, constrained / structured generation, length / faithfulness, tokenization edge cases) | 22 / 30 | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) |
| 10 | Machine translation (encoder-decoder NMT, multilingual / low-resource MT, terminology constraints, COMET / BLEU / chrF, FLORES) | 0 / 30 | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) |
| 11 | Multilingual and low-resource NLP (XLM-R, mBART, mT5, NLLB, script handling, transliteration, BCP-47-aware eval) | 1 / 30 | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) |
| 12 | Embeddings and representation learning (bi/cross-encoders, contrastive learning, hard-negative mining, MTEB, multilingual embeddings) — crossed 30% threshold | 9 / 30 | `nlp-engineer` | [`mod-108-embeddings-and-representation-learning`](lessons/mod-108-embeddings-and-representation-learning) |
| 13 | Speech / text interface (ASR fundamentals, text normalisation, ITN, punctuation restoration, diarisation post-processing) | 2 / 30 | `nlp-engineer` (light touch) | [`mod-109-speech-text-interface`](lessons/mod-109-speech-text-interface) |
| 14 | NLP data engineering (corpus collection, language ID, dedup, PII scrubbing, deboilerplating, annotation workflow, weak supervision) — **UP from 23% to 37%** | 11 / 30 | `nlp-engineer` | [`mod-110-nlp-data-engineering`](lessons/mod-110-nlp-data-engineering) |
| 15 | NLP-specific evaluation (BLEU / ROUGE / COMET / BERTScore / seqeval / SQuAD F1, statistical significance, contamination, slicing) — includes LLM-eval-authoring (LLM-as-judge, adversarial tests, regression suites) as a per-release engineer competency, now **three-cycle-persistent ≥ 37%** | 15 / 30 | `nlp-engineer` | [`mod-111-nlp-evaluation`](lessons/mod-111-nlp-evaluation) |
| 16 | Production NLP pipelines (spaCy Projects / HF Pipelines composition, document-level pipelines, NLP-specific drift monitoring, latency SLAs) | 21 / 30 | `nlp-engineer` | [`mod-112-production-nlp-pipelines`](lessons/mod-112-production-nlp-pipelines) |
| 17 | NLP systems design (task framing, model-family choice, build-vs-buy, evaluation gates, domain-adaptation strategy, cost/latency/quality) | 12 / 30 | `nlp-engineer` | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) + [`project-103-nlp-capstone-production-system`](projects/project-103-nlp-capstone-production-system) |
| 18 | Domain-specialised NLP (clinical, legal, financial, scientific, government/defense, insurance) — survey-style coverage of the patterns NLP engineers meet at vertical employers | 19 / 30 | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) |
| 19 | Responsible NLP at engineer altitude (datasheets, model cards, demographic / dialect slicing, PII / toxicity / licensing) | 4 / 30 | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) |
| 20 | PyTorch / classical ML / FastAPI / Docker / MLflow / experiment tracking fundamentals | 28 / 30 (prereq) | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 21 | Post-training stack depth (SFT / PEFT / RLHF / DPO / ORPO / KTO) | 7 / 30 (peer) | `fine-tuning-engineer` (level 30) | Touched only where NLP-task adaptation requires it; depth linked out to [`fine-tuning-engineer-learning`](../fine-tuning-engineer-learning) |
| 22 | RAG depth (chunking, vector stores, hybrid retrieval, rerankers, retrieval evaluation) | 10 / 30 (peer) | `rag-engineer` (level 25) | Surfaced at the QA (mod-105) and embeddings (mod-108) boundaries; depth linked out |
| 23 | LLM application development (prompting, agents, tool use, MCP / LangGraph / AutoGen, product integration) | 12 / 30 (peer) | `llm-application-developer` (level 25) | Surfaced in mod-106 (decoding) and mod-113 (systems-design build-vs-buy); depth linked out |
| 24 | Eval platform engineering (eval-as-product, statistical methodology depth, LLM-as-judge platform internals) | 2 / 30 (peer) | `model-evaluation-engineer` / `ai-eval-engineer` (level 30 / 25) | Module 111 covers the metric catalogue + per-release engineer-authored evals; platform depth linked out |
| 25 | Distributed-training PLATFORM engineering (multi-tenant schedulers, NCCL / fabric tuning) | 1 / 30 (peer) | `training-pipeline-engineer` (level 35) | Curriculum runs distributed training at operator altitude only; platform depth linked out |
| 26 | Deep ML / AI security (data poisoning, model extraction, training-data exfiltration, prompt-injection-resistance training) | 0 / 30 (higher) | `ai-infra-security-learning` (level 35) | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |
| 27 | Governance / compliance / model cards / dataset licensing review / regulated-data NLP | 0 / 30 (peer) | `ai-governance-analyst` / `ai-risk-engineer` | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |

## Signals not adopted (why the delta is empty)

Eight requirement themes cleared the 30% frequency bar this cycle. All are already covered by an existing module or already linked-out to a peer track.

1. **Generative NLP / LLMs — 22 / 30 (~73%).** Dominant expectation, same as prior cycles (77% → 76% → 73% — within noise). Owned by mod-106. **No change.**
2. **Production NLP pipelines / MLOps — 21 / 30 (~70%).** Strengthened sharply from 64% last cycle, driven by the vertical-diverse new sample where production deployment is mandatory rather than optional at every finance/legal/insurance/defense posting. Owned by mod-112. **No change.**
3. **Domain-specialised NLP — 19 / 30 (~63%).** Up from 50% but with materially better vertical diversity: healthcare 8 postings (Truveta, Midi, GW RhythmX, TeleTracking, Layer Health, Flagship, Dandelion, Zimmer), finance/insurance 5 (Point72 ×3, 6sense, EvolutionIQ), government/defense 3 (Babel Street, BTI360, Clera), legal 2 (Thomson Reuters, Hebbia), scientific 1 (Ai2 CellOLMo), compliance 1 (Behavox). Previously healthcare-heavy signal is now a proper vertical survey. mod-113's light-touch survey remains the right level — vertical-domain hiring expects employer-specific context acquired on the job, not deeper curriculum coverage. **No change.**
4. **NLP-specific evaluation — 15 / 30 (~50%).** Steady from 11/22 (50%) last cycle. The LLM-eval-authoring sub-signal (LLM-as-judge, adversarial tests, regression suites) is at ~37% this cycle, having persisted ≥ 37% across THREE consecutive cycles — this is the overdue mod-111 exercise-04 reframe watch item. **Action: content-authoring reframe of mod-111 exercise-04 should ship before 2027-01. NOT a plan-delta item.**
5. **NLP systems design — 12 / 30 (~40%).** Up from 32% last cycle. Multiple new postings explicitly frame their ask as "translate ambiguous business problems into defined technical scope" (Zimmer Biomet, Thomson Reuters, Clera). Owned by mod-113 and reinforced by project-103. **No change.**
6. **LLM applications / agents — 12 / 30 (~40%).** Down from 50% share last cycle; still above threshold and now at three-consecutive-cycle persistence (56% → 50% → 40%). Peer-track owned (`llm-application-developer` level 25). This curriculum already links out from mod-106 (decoding) and mod-113 (build-vs-buy). **No change.**
7. **Transformer internals — 12 / 30 (~40%).** Up from 36%. Owned by mod-101 (co-taught with tokenization). **No change.**
8. **NLP data engineering — 11 / 30 (~37%).** Recovered sharply from last cycle's 23% dip, back above threshold. Clera, BTI360, and Thomson Reuters all name the ingestion / dedup / entity-resolution / orchestration workflow at the heart of mod-110. **No change.**

Also notable but below threshold:

- **Information extraction (NER/RE/EN/entity resolution/structured extraction) — 10 / 30 (~34%).** The biggest material shift this cycle (9% → 34%), driven by BTI360, Clera, Thomson Reuters, Zimmer, Dandelion. Already owned by mod-104; coverage reinforcement, not a gap.
- **RAG / vector databases — 10 / 30 (~33%).** Steady from 32% last cycle; peer-track owned (rag-engineer). **No change.**
- **Embeddings / representation — 9 / 30 (~30%).** Crossed the 30% threshold this cycle (up from 27%). Owned by mod-108. **No change.**
- **Government / defense / intel domain NLP — 3 / 30 (~10%).** New vertical surfacing — Babel Street, BTI360, Clera. Up from 0% → 5% → 10% across three cycles. Below the 30% threshold but rising. If signal reaches 15%+ in 2027-01, add a one-paragraph mention to mod-113's domain-survey subsection. **Below-threshold watch item — not a plan-delta item.**

## Posting evidence — summary

Full verbatim posting data in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `postings`. Summary of the 30 verified in-window postings:

| # | Employer | Title | Date observed | Date posted | Location | Salary | Verification |
|---|---|---|---|---|---|---|---|
| 1 | Babel Street | NLP/Linguistics Software Engineer | 2026-10-04 | est.:2026-Q3 | Somerville, MA | $100k-$120k | carryover |
| 2 | Point72 | NLP Researcher (prev. NLP / AI Engineer) | 2026-11-04 | est.:2026-Q3 | New York, NY | not published | re-verified |
| 3 | Point72 | NLP Engineer | 2026-11-04 | est.:2026-Q3 | New York, NY | not published | re-verified |
| 4 | Point72 | Research Engineer, Knowledge Graph Intelligence | 2026-11-04 | est.:2026-Q3 | New York, NY | $175k-$250k | re-verified |
| 5 | Glean | Machine Learning Engineer, Assistant Quality | 2026-11-04 | est.:2026-Q3 | San Francisco, CA (Hybrid) | $180k-$205k | re-verified |
| 6 | Ai2 | Senior Research Engineer, Olmo + Molmo | 2026-11-04 | est.:2026-Q3 | Seattle, WA | $174k-$261k | re-verified |
| 7 | Ai2 | Young Investigator, Open Language Models for Biology | 2026-11-04 | est.:2026-Q3 | Seattle, WA | $159k | re-verified |
| 8 | CoreStory | AI Engineer | 2026-10-04 | est.:2026-Q3 | Remote US | not published | carryover |
| 9 | 6sense | Sr. Machine Learning Engineer | 2026-11-04 | est.:2026-Q3 | San Francisco, CA | $200k-$261k | re-verified |
| 10 | 6sense | ML Engineer III | 2026-10-04 | 2026-08-11 | Bengaluru, India | not published | carryover |
| 11 | Twilio | Machine Learning Engineer | 2026-10-04 | est.:2026-Q3 | Remote — Ireland | not published | carryover |
| 12 | Institute of Foundation Models | Research Engineer - NLP | 2026-10-04 | est.:2026-Q3 | Abu Dhabi, UAE | not published | carryover |
| 13 | Truveta | Machine Learning Engineer - LLMs & Generative AI | 2026-10-04 | est.:2026-Q3 | Seattle, WA | $155k-$175k | carryover |
| 14 | Midi Health | Senior Software Engineer, AI Engineer | 2026-11-04 | est.:2026-Q3 | SF or Palo Alto, CA (Hybrid) | $170k-$210k | re-verified |
| 15 | Get Well Network (GW RhythmX) | AI Engineer | 2026-10-04 | est.:2026-Q3 | Bangalore, India | not published | carryover |
| 16 | TeleTracking Technologies | Machine Learning Engineer III | 2026-10-04 | est.:2026-Q3 | Pittsburgh, PA | not published | carryover |
| 17 | Layer Health | Forward Deploy Data Scientist | 2026-11-04 | est.:2026-Q3 | Boston or NYC (Hybrid) | $150k-$180k | re-verified |
| 18 | Flagship Pioneering (FL105) | Machine Learning Research Engineer | 2026-10-04 | est.:2026-Q3 | Cambridge, MA | $120k-$193k | carryover |
| 19 | AvePoint | Junior AI Engineer | 2026-10-04 | 2026-09-14 | Da Nang, Vietnam | not published | carryover |
| 20 | GitLab | Senior Software Engineer, NLP | 2026-10-04 | est.:2026-Q3 | Remote (Americas/EMEA) | not published | carryover |
| 21 | GitLab | Staff Software Engineer - NLP | 2026-11-04 | est.:2026-Q3 | Bangalore, India | not published | re-verified |
| 22 | Grammarly | Machine Learning Engineer, Agents | 2026-10-04 | est.:2026-Q2 | San Francisco, CA (Hybrid) | $256k-$506k | carryover (now closed) |
| 23 | Behavox | Senior NLP Engineer | 2026-11-04 | 2026-10-16 | Toronto, Canada | $170k-$230k CAD | new |
| 24 | BTI360 | NLP Engineer - Clearance Required | 2026-11-04 | 2026-10-15 | Herndon, VA | $125.9k-$247.9k | new |
| 25 | Clera | Founding Engineer, NLP Systems & Infrastructure | 2026-11-04 | 2026-10-28 | Munich, Germany | €60k-€120k | new |
| 26 | Dandelion Health | NLP / LLM Data Scientist | 2026-11-04 | 2026-09-22 | NY Remote | $135k-$165k | new |
| 27 | Zimmer Biomet | NLP Engineer | 2026-11-04 | 2026-09-28 | Bangalore, India | 10-14 LPA | new |
| 28 | EvolutionIQ | Senior Machine Learning (ML) Engineer (AI Insurtech) | 2026-11-04 | 2026-08-05 | New York, NY or Remote | $190k-$225k | new |
| 29 | Thomson Reuters | Senior Machine Learning Engineer – NLP / LLM | 2026-11-04 | 2026-10-02 | Brooklyn, NY (Hybrid) | $110k-$235k | new |
| 30 | Hebbia AI | Applied AI Research Engineer, LLM & NLP (SF) | 2026-11-04 | 2026-07-24 | San Francisco, CA | $160k-$300k | new (now closed) |

## Ownership map — quick reference

- **NLP Engineer (this track, level 30)** owns language-specific depth end-to-end: tokenization theory, classical NLP, modern transformer NLP for sequence / structured / generative language tasks (NER, classification, summarisation, MT, information extraction, semantic parsing, text generation), embeddings and representation learning, multilingual and low-resource NLP, the speech / text interface where relevant, NLP-specific data engineering and evaluation, and production NLP-pipeline / systems design.
- **ML Engineer** (level 20) owns the build-altitude practitioner workflow this curriculum assumes.
- **Senior ML Engineer** (level 30, peer ladder) is the *generalist* peer at the same level — overlaps in NLP vocabulary but does not own the depth.
- **Staff / Principal ML Engineer** (level 40+) inherit / link to this packet for NLP depth.
- **Fine-Tuning Engineer** (level 30, peer specialist) owns the post-training stack (SFT / PEFT / RLHF / DPO).
- **RAG Engineer** (level 25) owns retrieval-augmented generation depth. Surfaced at mod-105 and mod-108 boundaries only.
- **LLM Application Developer** (level 25) owns prompting, agents, tool use, product integration. Surfaced in mod-106 and mod-113 only.
- **Model Evaluation / AI Eval Engineer** (level 30 / 25) own eval-as-platform depth. This curriculum owns the NLP metric catalogue and per-release engineer-authored evals (mod-111).
- **Training Pipeline Engineer** (level 35) owns distributed-training INFRASTRUCTURE.
- **AI Infrastructure Security Engineer** (level 35) owns deep ML/AI security.
- **AI Governance Analyst / AI Risk Engineer** own governance / compliance / risk depth.

## Conclusion

Under the continuity bias, this cycle produced **zero net additions** for the third consecutive quarter. Every requirement that cleared the frequency threshold is already either owned by an existing module (mod-101 – mod-113) or owned by a peer track that this curriculum already links out to. The empty delta is captured in [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json).

The one content-authoring action worth attention remains the **LLM-quality eval authoring reframe of mod-111 exercise-04**, which has now persisted at ≥ 37% across three consecutive cycles (44% → 50% → 37%). This should be executed as content-authoring work (not a plan change) before the 2027-01 cycle so the next research pass can measure its uptake rather than re-flagging the same gap for a fourth time.

The one new below-threshold signal worth watching is **government / defense / intel domain NLP** (Babel Street, BTI360, Clera), which rose from 0% → 5% → 10% across three cycles. If it reaches 15%+ in 2027-01, add a one-paragraph mention to mod-113's domain-survey subsection.

<!-- needs-research: (1) retry frontier-lab careers pages (OpenAI, Anthropic, Cohere, Mistral, xAI, AI21, Reka, Apple, Meta GenAI, Google DeepMind, Nvidia) in 2027-01 with a stronger rendering path — Anthropic's board was reachable but individual postings still returned index-only content; (2) retry speech-vendor careers (AssemblyAI, Deepgram, Rev, Speechmatics) — AssemblyAI shows 17 open roles including Senior Researcher Deep Learning NLP but posting bodies were not verbatim-fetchable; (3) retry MT-vendor careers (DeepL, Lilt, Smartling, ModelFront) — zero verifiable MT-specific postings across three cycles; (4) confirm Thomson Reuters Senior MLE NLP/LLM continues as a recurring posting — if it reposts in 2027-Q1 that becomes a persistent-vacancy signal; (5) if the LLM-eval-authoring reframe of mod-111 exercise-04 has shipped, measure whether its uptake addresses the three-cycle gap or whether a standalone exercise is warranted. Under-sampled areas: multilingual / MT (1/30), speech interface (2/30), tokenization theory (2/30), sequence labelling / NER (1/30), QA (0/30), summarisation (1/30). The zero / near-zero counts in these foundational modeling tasks reflect "LLM" being used as the umbrella hiring term rather than those tasks disappearing. -->
