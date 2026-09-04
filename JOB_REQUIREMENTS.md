# Job Requirements — NLP Engineer

**Role level:** 30 (deep specialist — peer to Senior ML Engineer on the ladder)
**Track:** `nlp-engineer-learning`
**Research window:** 2026-06-06 → 2026-09-04 (last 90 days)
**Today:** 2026-09-04
**Postings sampled this cycle:** 18 verified in-window (target 25 — see Shortfall below)

## Status — live evidence, partial sample

This cycle exercised WebSearch / WebFetch against live job boards and verified 18 in-window postings across the required title spectrum (`NLP Engineer`, `Natural Language Processing Engineer`, `Applied NLP Engineer`, `NLP/AI Engineer`, `Computational Linguist`, `NLP/Linguistics Software Engineer`, `ML Engineer` roles where the required qualifications name NLP explicitly, and `Research Engineer` roles at open-language-model shops). Raw data — employer, title, URL, date_observed, location, verbatim required / preferred bullets, salary_range, quote — is in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `postings`.

**Shortfall vs. 25-posting target.** Frontier-lab careers pages (Apple, Anthropic, OpenAI, Cohere, Mistral, xAI, AI21, Reka) either 404'd on posting-level URLs, returned index-only content, or surfaced only meta / aggregator articles. Speech-vendor careers pages (AssemblyAI, Deepgram, Speechmatics) similarly returned index-only content. MT-vendor careers pages (DeepL, Lilt, Unbabel, ModelFront) were not scrapeable at posting granularity. Rather than fabricate URLs / dates / quotes to hit the numeric target, this cycle capped at what was actually verified. Next cycle should retry those sources — see `.aicg/job-requirements.json` `research_status.needs_research_note`.

## Continuity outcome — zero net additions

Under the continuity bias ("propose net-new content only when ≥ 3 distinct in-window postings cite a requirement the existing curriculum does NOT cover, AND frequency ≥ 30%, AND no existing module can be incrementally extended"), this cycle produces **no new modules, no new exercises, no new projects**. See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) for the empty delta and its rationale.

The three signals that cleared the 30% frequency bar (LLM / GenAI generation 72%, production ML / MLOps 61%, RAG 61%, LLM applications & agents 61%, transformers 33%, systems design 33%, classification 33%, data engineering 39%, evaluation 44%, domain-vertical 39%) are already covered by the existing 13 modules — either owned directly (mod-101 through mod-113) or already surfaced-and-linked-out where a peer track owns the depth (rag-engineer, llm-application-developer, fine-tuning-engineer, model-evaluation-engineer). See the **Signals not adopted** section below for the reasoning per signal.

## Methodology

1. Fanned out WebSearch / WebFetch across `job-boards.greenhouse.io`, `boards.greenhouse.io`, `jobs.lever.co`, `jobs.ashbyhq.com`, `builtin.com`, and named-employer careers pages for the title spectrum.
2. Filtered out generic "ML Engineer" / "Data Scientist" / "Research Scientist" titles unless the required-qualifications section named NLP explicitly. Filtered out "AI Engineer" / "Generative AI Engineer" / "LLM Application Engineer" and Staff / Principal / Director modifiers (they inherit this packet).
3. Captured each verified posting verbatim: employer, exact title, URL, date_observed, date_posted (`estimated:2026-MM` where the board does not publish it), location, required and preferred bullets, salary_range (`null` if unpublished), one short representative quote.
4. Computed requirement frequency across the 18-posting sample (30% threshold = ≥ 6 postings).
5. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill. Signal-vs-coverage delta captured under "Signals not adopted" below.

## Requirement themes → curriculum ownership

`Freq` is a fraction of the 18 verified postings. `evidence_post_ids` on each requirement in `.aicg/job-requirements.json` names the specific postings.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Tokenization theory (BPE / WordPiece / Unigram / SentencePiece, multilingual / CJK, tokenizer training, chat-template arithmetic) | 2 / 18 | `nlp-engineer` (this) | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) |
| 2 | Transformer internals for NLP (encoder vs. encoder-decoder vs. decoder-only choice, KV cache for generation, probing, tokenizer-shape effects) | 6 / 18 | `nlp-engineer` | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) (co-taught with tokenization) |
| 3 | Classical NLP (regex, FST, n-gram LMs, taggers, parsers, lemmatisation, Unicode-correct text processing, language identification) | 3 / 18 | `nlp-engineer` | [`mod-102-classical-nlp`](lessons/mod-102-classical-nlp) |
| 4 | Text classification end-to-end (baselines through fine-tuned encoders, multilingual, multi-label, calibration, cost-sensitive eval) | 6 / 18 | `nlp-engineer` | [`mod-103-text-classification`](lessons/mod-103-text-classification) |
| 5 | Sequence labelling and NER (BIO/BIOES/BILOU, span models, CRF heads, subword-word alignment, entity-level eval, domain NER) | 4 / 18 | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) |
| 6 | Information extraction (relation / event extraction, slot filling, entity linking, coreference, document-level IE, structured generation) | 3 / 18 | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) |
| 7 | Question answering and machine reading comprehension (extractive / abstractive / multi-hop / long-context, unanswerability, EM / F1) | 4 / 18 | `nlp-engineer` | [`mod-105-question-answering-and-machine-reading`](lessons/mod-105-question-answering-and-machine-reading) |
| 8 | Summarisation (extractive / abstractive, long-document, multi-document, controllable, faithfulness, ROUGE / BERTScore) | 1 / 18 | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) |
| 9 | Generative NLP and decoding (sampling strategies, constrained / structured generation, length / faithfulness, tokenization edge cases) | 13 / 18 | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) |
| 10 | Machine translation (encoder-decoder NMT, multilingual / low-resource MT, terminology constraints, COMET / BLEU / chrF, FLORES) | 2 / 18 | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) |
| 11 | Multilingual and low-resource NLP (XLM-R, mBART, mT5, NLLB, script handling, transliteration, BCP-47-aware eval) | 2 / 18 | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) |
| 12 | Embeddings and representation learning (bi/cross-encoders, contrastive learning, hard-negative mining, MTEB, multilingual embeddings) | 3 / 18 | `nlp-engineer` | [`mod-108-embeddings-and-representation-learning`](lessons/mod-108-embeddings-and-representation-learning) |
| 13 | Speech / text interface (ASR fundamentals, text normalisation, ITN, punctuation restoration, diarisation post-processing) | 2 / 18 | `nlp-engineer` (light touch) | [`mod-109-speech-text-interface`](lessons/mod-109-speech-text-interface) |
| 14 | NLP data engineering (corpus collection, language ID, dedup, PII scrubbing, deboilerplating, annotation workflow, weak supervision) | 7 / 18 | `nlp-engineer` | [`mod-110-nlp-data-engineering`](lessons/mod-110-nlp-data-engineering) |
| 15 | NLP-specific evaluation (BLEU / ROUGE / COMET / BERTScore / seqeval / SQuAD F1, statistical significance, contamination, slicing) — now including LLM-eval-authoring (LLM-as-judge, adversarial tests, regression suites) as a per-release engineer competency | 8 / 18 | `nlp-engineer` | [`mod-111-nlp-evaluation`](lessons/mod-111-nlp-evaluation) |
| 16 | Production NLP pipelines (spaCy Projects / HF Pipelines composition, document-level pipelines, NLP-specific drift monitoring, latency SLAs) | 11 / 18 | `nlp-engineer` | [`mod-112-production-nlp-pipelines`](lessons/mod-112-production-nlp-pipelines) |
| 17 | NLP systems design (task framing, model-family choice, build-vs-buy, evaluation gates, domain-adaptation strategy, cost/latency/quality) | 6 / 18 | `nlp-engineer` | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) + [`project-103-nlp-capstone-production-system`](projects/project-103-nlp-capstone-production-system) |
| 18 | Domain-specialised NLP (clinical, legal, financial, scientific) — survey-style coverage of the patterns NLP engineers meet at vertical employers | 7 / 18 | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) |
| 19 | Responsible NLP at engineer altitude (datasheets, model cards, demographic / dialect slicing, PII / toxicity / licensing) | 1 / 18 | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) |
| 20 | PyTorch / classical ML / FastAPI / Docker / MLflow / experiment tracking fundamentals | 16 / 18 (prereq) | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 21 | Post-training stack depth (SFT / PEFT / RLHF / DPO / ORPO / KTO) | 6 / 18 (peer) | `fine-tuning-engineer` (level 30) | Touched only where NLP-task adaptation requires it; depth linked out to [`fine-tuning-engineer-learning`](../fine-tuning-engineer-learning) |
| 22 | RAG depth (chunking, vector stores, hybrid retrieval, rerankers, retrieval evaluation) | 11 / 18 (peer) | `rag-engineer` (level 25) | Surfaced at the QA (mod-105) and embeddings (mod-108) boundaries; depth linked out |
| 23 | LLM application development (prompting, agents, tool use, MCP / LangGraph / AutoGen, product integration) | 11 / 18 (peer) | `llm-application-developer` (level 25) | Surfaced in mod-106 (decoding) and mod-113 (systems-design build-vs-buy); depth linked out |
| 24 | Eval platform engineering (eval-as-product, statistical methodology depth, LLM-as-judge platform internals) | 2 / 18 (peer) | `model-evaluation-engineer` / `ai-eval-engineer` (level 30 / 25) | Module 111 covers the metric catalogue + per-release engineer-authored evals; platform depth linked out |
| 25 | Distributed-training PLATFORM engineering (multi-tenant schedulers, NCCL / fabric tuning) | 2 / 18 (peer) | `training-pipeline-engineer` (level 35) | Curriculum runs distributed training at operator altitude only; platform depth linked out |
| 26 | Deep ML / AI security (data poisoning, model extraction, training-data exfiltration, prompt-injection-resistance training) | 0 / 18 (higher) | `ai-infra-security-learning` (level 35) | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |
| 27 | Governance / compliance / model cards / dataset licensing review / regulated-data NLP | 1 / 18 (peer) | `ai-governance-analyst` / `ai-risk-engineer` | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |

## Signals not adopted (why the delta is empty)

Four requirement themes cleared the 30% frequency bar but did NOT drive net-new content. The continuity bias rejects any theme that is (a) already covered, (b) covered by a peer track that we can link to, or (c) extensible within an existing module without a new one.

1. **Agentic frameworks and orchestration (MCP, LangGraph, LangChain, AutoGen, CrewAI, tool use, multi-agent workflows) — 10 / 18 (~56%).** Evidence: Ai2 Asta ("agentic frameworks (e.g., MCP)"), 6sense ("LangGraph, LangChain, or Amazon Bedrock"), Twilio ("agentic AI frameworks such as LangGraph, AutoGen, CrewAI"), Primer.ai ("LLMs and agentic systems"), Ford ("designing or deploying agentic AI systems"), Glean ("agent systems"), Grammarly ("Machine Learning Engineer, Agents"), Ai2 Olmo ("Agentic systems knowledge — tools, memory, and long-running workflows"), PitchBook ("LangSmith and LangGraph"). **Not adopted:** peer track `llm-application-developer` (level 25) owns agent depth and this curriculum already links out from mod-106 (decoding) and mod-113 (build-vs-buy). Adopting it here would duplicate the peer track and blur ownership. See [`llm-application-developer-learning`](../llm-application-developer-learning) for depth.
2. **RAG / vector databases / hybrid retrieval — 11 / 18 (~61%).** Evidence: Point72, 6sense, PitchBook, Twilio, Primer.ai, Grammarly, Glean, Ford, WEX, Ai2 Bio, AvePoint. **Not adopted:** peer track `rag-engineer` (level 25) owns retrieval depth and this curriculum already surfaces the boundary in mod-105 (QA reader / generator side) and mod-108 (embedding-model side). See [`rag-engineer-learning`](../rag-engineer-learning) for depth.
3. **Prompt engineering — 7 / 18 (~39%).** Evidence: 6sense, Twilio, Primer.ai (prompt & context engineering), Grammarly, Ford (prompt engineering, LLM evaluation), Glean, PitchBook. **Not adopted:** peer track `llm-application-developer` owns prompting depth. Same rationale as (1). Surfaced in mod-106 and mod-113.
4. **LLM-quality eval authoring (LLM-as-judge, automated evaluators, adversarial tests, regression suites) — 8 / 18 (~44%).** Evidence: Ford ("evaluation datasets, automated evaluators, adversarial tests, red-team scenarios, and regression test suites for generative AI"), Primer.ai ("defining evals to measure and improve quality"), Grammarly, Glean, PitchBook, Twilio (LLMOps/testing), JPMorgan (intrinsic and extrinsic metrics). **Not adopted:** covered incrementally by mod-111's existing scope (metric catalogue, contamination, human-evaluation-protocol design). The framing has shifted from "run the benchmark" to "author + maintain evals per release" but the underlying skills (metric selection, statistical significance, human-eval design) are the same and are taught in mod-111. If the signal persists next cycle we should reframe mod-111 exercise-04 ("human-evaluation-protocol-design") to explicitly cover LLM-as-judge and regression-suite authoring — but that is a within-exercise reframe, not a new item. Deep eval-platform work remains owned by `model-evaluation-engineer` (level 30).

Below-threshold signals that were also considered and skipped:
- **GCP Vertex AI / Amazon Bedrock managed-LLM platforms.** 2 / 18 (Ford, 6sense) = 11%. Below threshold; peer-track territory (`ml-engineer` / `mlops`).
- **Knowledge graphs.** 1 / 18 (Primer.ai). Below threshold.
- **Open-source contributions to spaCy / AllenNLP / transformers / langchain.** 3 / 18 (all Ai2). Cultural preference at one employer, not a general market shift.

## Posting evidence — summary

Full verbatim posting data in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `postings`. Summary of the 18 verified in-window postings:

| # | Employer | Title | Date | Location | Salary |
|---|---|---|---|---|---|
| 0 | Babel Street | NLP/Linguistics Software Engineer | estimated:2026-08 | Somerville, MA | $100k-$120k |
| 1 | Point72 | NLP Engineer | estimated:2026-08 | New York, NY | not published |
| 2 | Point72 | NLP / AI Engineer | estimated:2026-07 | New York, NY | not published |
| 3 | Ai2 | Senior Research Engineer, Olmo + Molmo | estimated:2026-07 | Seattle, WA | $174k-$261k |
| 4 | Ai2 | Research Engineer, FlexOlmo | estimated:2026-07 | Seattle, WA | $147k-$220k |
| 5 | Ai2 | Research Engineer, Asta | estimated:2026-07 | Seattle, WA | $129k-$193k |
| 6 | Ai2 | Young Investigator, Open Language Models for Biology | estimated:2026-08 | Seattle, WA | $160k |
| 7 | 6sense | Sr. Machine Learning Engineer | estimated:2026-08 | San Francisco, CA | $200k-$261k |
| 8 | PitchBook Data | Machine Learning Engineer | estimated:2026-08 | Seattle, WA | $125k-$180k + bonus |
| 9 | Twilio | Machine Learning Engineer | estimated:2026-08 | Remote — Ireland | not published |
| 10 | Primer.ai | Staff Machine Learning Engineer | 2026-08-04 | Pasadena / SF / DC | not published |
| 11 | Grammarly | Machine Learning Engineer, Agents | estimated:2026-07 | San Francisco, CA | $256k-$506k |
| 12 | Glean | Machine Learning Engineer, Assistant Quality | estimated:2026-08 | San Francisco, CA (Hybrid) | $180k-$205k |
| 13 | Ford Motor Company | Data Scientist NLP (AI) Engineer | estimated:2026-07 | Mexico (Remote) | not published |
| 14 | WEX Inc. | Mid-Level AI/ML/NLP Engineer | estimated:2026-07 | Remote (multi-city US) | $87k-$115k |
| 15 | Wispr Flow | Computational Linguist and Data Annotation Lead | estimated:2026-06 | San Francisco, CA | not published |
| 16 | J.P. Morgan | 2026 ML Center of Excellence (NLP) — Summer Associate | 2025-09-05 | New York, NY | $135k-$155k |
| 17 | AvePoint | Junior AI Engineer | 2026-08-13 | Hanoi, Vietnam | not published |

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

Under the continuity bias, this cycle produced **zero net additions** because every requirement that cleared the frequency threshold is already either owned by an existing module (mod-101 – mod-113) or owned by a peer track that this curriculum already links out to. The empty delta is captured in [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json).

The one signal worth watching next cycle is **LLM-quality eval authoring** (LLM-as-judge, adversarial / regression test suites) at 44% of the sample. If it holds or grows, the right response is to **reframe mod-111 exercise-04 to explicitly cover LLM-as-judge and regression-suite authoring** rather than add a new module — the underlying methodology (metric selection, statistical significance, human-eval-protocol design) is already there. That reframe is an authoring pass, not a curriculum-plan-delta item; it should be filed as content work, not a plan change.

<!-- needs-research: retry frontier-lab (OpenAI, Anthropic, Cohere, Mistral, xAI, AI21, Reka, Apple, Meta GenAI, Google DeepMind), speech-vendor (AssemblyAI, Deepgram, Speechmatics), and MT-vendor (DeepL, Lilt, Unbabel, ModelFront) careers pages next cycle to push the sample size ≥ 25 and give the 30% frequency threshold statistical grip. Under-sampled areas this cycle: multilingual / MT (2 / 18), speech interface (2 / 18), tokenization theory (2 / 18). -->
