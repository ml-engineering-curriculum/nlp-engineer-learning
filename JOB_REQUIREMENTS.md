# Job Requirements — NLP Engineer

**Role level:** 30 (deep specialist — peer to Senior ML Engineer on the ladder)
**Track:** `nlp-engineer-learning`
**Research window:** 2026-07-06 → 2026-10-04 (last 90 days)
**Today:** 2026-10-04
**Postings sampled this cycle:** 22 verified in-window (target 25 — see Shortfall below)

## Status — live evidence, partial sample (second quarterly cycle)

This cycle exercised WebSearch / WebFetch against live job boards and verified 22 in-window postings across the required title spectrum (`NLP Engineer`, `Natural Language Processing Engineer`, `Applied NLP Engineer`, `NLP/AI Engineer`, `Language Model Engineer`, `Computational Linguist`, `NLP/Linguistics Software Engineer`, `ML Engineer` and `Research Engineer` roles where the required qualifications name NLP explicitly). Raw data — employer, title, URL, date_observed, location, verbatim required / preferred bullets, salary_range, quote — is in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `postings`.

**Shortfall vs. 25-posting target.** As in the 2026-09 cycle, frontier-lab careers pages (OpenAI, Anthropic, Cohere, Mistral, xAI, AI21, Reka, Apple, Meta GenAI, Google DeepMind, Nvidia) returned 403 / 404 / index-only content and remained mostly unscrapeable at posting granularity. Anthropic's careers index was reachable this cycle but individual posting URLs still returned index-only content. Speech-vendor careers pages (AssemblyAI, Deepgram, Rev, Speechmatics) and MT-vendor careers pages (DeepL, Lilt, Smartling, ModelFront) likewise returned no verbatim-fetchable in-window NLP postings — Unbabel is now bankrupt per 2026-Q1 (TransPerfect acquisition). Two previously-verified URLs 404'd this cycle (Ai2 FlexOlmo `jobs/7819106`, Ai2 Asta `jobs/7899565`) — those postings appear to have been closed between cycles. Rather than fabricate URLs / dates / quotes to hit the numeric target, this cycle capped at what was actually verified. Next cycle should retry those sources and deliberately diversify away from the healthcare greenhouse over-sampling that this cycle exhibits — see `.aicg/job-requirements.json` `research_status.needs_research_note`.

## Continuity outcome — zero net additions (second cycle)

Under the continuity bias ("propose net-new content only when ≥ 3 distinct in-window postings cite a requirement the existing curriculum does NOT cover, AND frequency ≥ 30%, AND no existing module can be incrementally extended"), this cycle again produces **no new modules, no new exercises, no new projects**. See [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json) for the empty delta and its rationale.

Every requirement theme that cleared the 30% frequency threshold (≥ 7 of 22 postings) — generative NLP / LLMs (77%), production ML / MLOps (64%), domain-specialised NLP (50%, inflated by healthcare over-sampling), evaluation (50%), LLM applications / agents (50%), transformer internals (36%), RAG (32%), systems design (32%) — is already either owned directly by an existing module (mod-101 through mod-113) or already surfaced-and-linked-out where a peer track owns the depth (rag-engineer, llm-application-developer, fine-tuning-engineer, model-evaluation-engineer). See the **Signals not adopted** section below for the per-signal reasoning.

**Within-exercise reframe watch item.** The LLM-quality eval authoring signal (LLM-as-judge, adversarial tests, regression suites, automated evaluators) strengthened for the second consecutive cycle (44% → 50%). The right response remains a within-exercise reframe of mod-111 exercise-04, not a new curriculum item — this is content-authoring work, not a plan change. If the signal persists through 2027-01, re-evaluate whether a standalone exercise is warranted.

## Methodology

1. Fanned out WebSearch / WebFetch across `job-boards.greenhouse.io`, `boards.greenhouse.io`, `jobs.lever.co`, `jobs.ashbyhq.com`, `builtin.com`, and named-employer careers pages for the title spectrum. Retried frontier-lab, speech-vendor, and MT-vendor sources that failed in 2026-09.
2. Filtered out generic "ML Engineer" / "Data Scientist" / "Research Scientist" titles unless the required-qualifications section named NLP explicitly. Filtered out generic "AI Engineer" / "Generative AI Engineer" / "LLM Application Engineer" and Staff / Principal / Director modifiers (they inherit this packet).
3. Captured each verified posting verbatim: employer, exact title, URL, date_observed, date_posted (`estimated:2026-Q3` where the board did not publish it), location, required and preferred bullets, salary_range (`null` if unpublished), one short representative quote.
4. Computed requirement frequency across the 22-posting sample (30% threshold = ≥ 7 postings).
5. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill. Signal-vs-coverage delta captured under "Signals not adopted" below.

## Requirement themes → curriculum ownership

`Freq` is a fraction of the 22 verified postings. `evidence_post_ids` on each requirement in `.aicg/job-requirements.json` names the specific postings.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Tokenization theory (BPE / WordPiece / Unigram / SentencePiece, multilingual / CJK, tokenizer training, chat-template arithmetic) | 2 / 22 | `nlp-engineer` (this) | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) |
| 2 | Transformer internals for NLP (encoder vs. encoder-decoder vs. decoder-only choice, KV cache for generation, probing, tokenizer-shape effects) | 8 / 22 | `nlp-engineer` | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) (co-taught with tokenization) |
| 3 | Classical NLP (regex, FST, n-gram LMs, taggers, parsers, lemmatisation, Unicode-correct text processing, language identification) | 3 / 22 | `nlp-engineer` | [`mod-102-classical-nlp`](lessons/mod-102-classical-nlp) |
| 4 | Text classification end-to-end (baselines through fine-tuned encoders, multilingual, multi-label, calibration, cost-sensitive eval) | 1 / 22 | `nlp-engineer` | [`mod-103-text-classification`](lessons/mod-103-text-classification) |
| 5 | Sequence labelling and NER (BIO/BIOES/BILOU, span models, CRF heads, subword-word alignment, entity-level eval, domain NER) | 0 / 22 | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) |
| 6 | Information extraction (relation / event extraction, slot filling, entity linking, coreference, document-level IE, structured generation) | 2 / 22 | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) |
| 7 | Question answering and machine reading comprehension (extractive / abstractive / multi-hop / long-context, unanswerability, EM / F1) | 0 / 22 | `nlp-engineer` | [`mod-105-question-answering-and-machine-reading`](lessons/mod-105-question-answering-and-machine-reading) |
| 8 | Summarisation (extractive / abstractive, long-document, multi-document, controllable, faithfulness, ROUGE / BERTScore) | 0 / 22 | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) |
| 9 | Generative NLP and decoding (sampling strategies, constrained / structured generation, length / faithfulness, tokenization edge cases) | 17 / 22 | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) |
| 10 | Machine translation (encoder-decoder NMT, multilingual / low-resource MT, terminology constraints, COMET / BLEU / chrF, FLORES) | 0 / 22 | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) |
| 11 | Multilingual and low-resource NLP (XLM-R, mBART, mT5, NLLB, script handling, transliteration, BCP-47-aware eval) | 0 / 22 | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) |
| 12 | Embeddings and representation learning (bi/cross-encoders, contrastive learning, hard-negative mining, MTEB, multilingual embeddings) | 6 / 22 | `nlp-engineer` | [`mod-108-embeddings-and-representation-learning`](lessons/mod-108-embeddings-and-representation-learning) |
| 13 | Speech / text interface (ASR fundamentals, text normalisation, ITN, punctuation restoration, diarisation post-processing) | 2 / 22 | `nlp-engineer` (light touch) | [`mod-109-speech-text-interface`](lessons/mod-109-speech-text-interface) |
| 14 | NLP data engineering (corpus collection, language ID, dedup, PII scrubbing, deboilerplating, annotation workflow, weak supervision) | 5 / 22 | `nlp-engineer` | [`mod-110-nlp-data-engineering`](lessons/mod-110-nlp-data-engineering) |
| 15 | NLP-specific evaluation (BLEU / ROUGE / COMET / BERTScore / seqeval / SQuAD F1, statistical significance, contamination, slicing) — including LLM-eval-authoring (LLM-as-judge, adversarial tests, regression suites) as a per-release engineer competency | 11 / 22 | `nlp-engineer` | [`mod-111-nlp-evaluation`](lessons/mod-111-nlp-evaluation) |
| 16 | Production NLP pipelines (spaCy Projects / HF Pipelines composition, document-level pipelines, NLP-specific drift monitoring, latency SLAs) | 14 / 22 | `nlp-engineer` | [`mod-112-production-nlp-pipelines`](lessons/mod-112-production-nlp-pipelines) |
| 17 | NLP systems design (task framing, model-family choice, build-vs-buy, evaluation gates, domain-adaptation strategy, cost/latency/quality) | 7 / 22 | `nlp-engineer` | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) + [`project-103-nlp-capstone-production-system`](projects/project-103-nlp-capstone-production-system) |
| 18 | Domain-specialised NLP (clinical, legal, financial, scientific) — survey-style coverage of the patterns NLP engineers meet at vertical employers | 11 / 22 | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) |
| 19 | Responsible NLP at engineer altitude (datasheets, model cards, demographic / dialect slicing, PII / toxicity / licensing) | 3 / 22 | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) |
| 20 | PyTorch / classical ML / FastAPI / Docker / MLflow / experiment tracking fundamentals | 20 / 22 (prereq) | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 21 | Post-training stack depth (SFT / PEFT / RLHF / DPO / ORPO / KTO) | 6 / 22 (peer) | `fine-tuning-engineer` (level 30) | Touched only where NLP-task adaptation requires it; depth linked out to [`fine-tuning-engineer-learning`](../fine-tuning-engineer-learning) |
| 22 | RAG depth (chunking, vector stores, hybrid retrieval, rerankers, retrieval evaluation) | 7 / 22 (peer) | `rag-engineer` (level 25) | Surfaced at the QA (mod-105) and embeddings (mod-108) boundaries; depth linked out |
| 23 | LLM application development (prompting, agents, tool use, MCP / LangGraph / AutoGen, product integration) | 11 / 22 (peer) | `llm-application-developer` (level 25) | Surfaced in mod-106 (decoding) and mod-113 (systems-design build-vs-buy); depth linked out |
| 24 | Eval platform engineering (eval-as-product, statistical methodology depth, LLM-as-judge platform internals) | 2 / 22 (peer) | `model-evaluation-engineer` / `ai-eval-engineer` (level 30 / 25) | Module 111 covers the metric catalogue + per-release engineer-authored evals; platform depth linked out |
| 25 | Distributed-training PLATFORM engineering (multi-tenant schedulers, NCCL / fabric tuning) | 1 / 22 (peer) | `training-pipeline-engineer` (level 35) | Curriculum runs distributed training at operator altitude only; platform depth linked out |
| 26 | Deep ML / AI security (data poisoning, model extraction, training-data exfiltration, prompt-injection-resistance training) | 0 / 22 (higher) | `ai-infra-security-learning` (level 35) | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |
| 27 | Governance / compliance / model cards / dataset licensing review / regulated-data NLP | 0 / 22 (peer) | `ai-governance-analyst` / `ai-risk-engineer` | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |

## Signals not adopted (why the delta is empty)

Six requirement themes cleared the 30% frequency bar this cycle but did NOT drive net-new content. The continuity bias rejects any theme that is (a) already covered, (b) covered by a peer track that we can link to, or (c) extensible within an existing module without a new one.

1. **Agentic frameworks and orchestration (MCP, LangGraph, LangChain, AutoGen, CrewAI, tool use, multi-agent workflows) — 11 / 22 (~50%).** Steady from 10/18 (~56%) in 2026-09. Evidence: Glean "agent systems," Ai2 Olmo+Molmo "agentic systems knowledge — tools, memory, and long-running workflows," CoreStory "LangChain, LangGraph, tool calling, chat agent architectures," 6sense Sr. MLE / III "LangGraph, LangChain, Amazon Bedrock," Twilio "LangGraph, AutoGen, CrewAI," Midi Health "agentic systems or production RAG at scale," Flagship Pioneering "vLLM, LangGraph," GitLab Senior SWE NLP "agentic frameworks," GitLab Staff SWE NLP "Agentic AI Leadership," Grammarly "multi-agent AI platform." **Not adopted:** peer track `llm-application-developer` (level 25) owns agent depth. This curriculum already links out from mod-106 (decoding) and mod-113 (build-vs-buy). Adopting it here would duplicate the peer track and blur ownership. See [`llm-application-developer-learning`](../llm-application-developer-learning) for depth.
2. **LLM-quality eval authoring (automated evaluators, adversarial tests, regression suites, LLM-as-judge, red-team scenarios) — 11 / 22 (~50%).** Up from 8/18 (~44%). Evidence: Point72 KG "model evaluation / error analysis," Glean "evaluation frameworks," Ai2 Olmo+Molmo "evaluation, profiling, and monitoring," CoreStory implicit, 6sense (×2) "model evaluation, MLOps practices," Truveta "evaluation of generative systems," Midi Health "rigorous about evaluation and safety; allergic to vibes-only launches," GW RhythmX "BLEU, ROUGE, accuracy, recall, and human-in-the-loop validation," TeleTracking "model performance assessment, training multiple models, tuning, and A/B testing capability," Flagship "standard evaluation protocols," Grammarly "evaluating the performance of LLMs." **Not adopted:** covered incrementally by mod-111's existing scope (metric catalogue, contamination, human-evaluation-protocol design). The framing has shifted from "run the benchmark" to "author + maintain evals per release" but the underlying skills (metric selection, statistical significance, human-eval design) are the same and are taught in mod-111. Signal persisted for the second consecutive cycle — the right next move is a **within-exercise reframe of mod-111 exercise-04** to explicitly cover LLM-as-judge and regression-suite authoring. This is a content-authoring pass, not a curriculum-plan-delta item. Deep eval-platform work remains owned by `model-evaluation-engineer` (level 30).
3. **RAG / vector databases / hybrid retrieval — 7 / 22 (~32%).** Down from 11/18 (~61%). Evidence: Glean, CoreStory, 6sense (Sr. MLE + III), Truveta, Midi Health, GW RhythmX. **Not adopted:** peer track `rag-engineer` (level 25) owns retrieval depth and this curriculum already surfaces the boundary in mod-105 (QA reader / generator side) and mod-108 (embedding-model side). The sharp dip is a sample-composition artifact of healthcare-vertical over-sampling (clinical greenhouse boards describe pipelines without naming "RAG"), not a market shift. See [`rag-engineer-learning`](../rag-engineer-learning) for depth.
4. **Domain-specialised NLP (healthcare / clinical / EHR, finance, scientific, linguistics) — 11 / 22 (~50%).** Up from 7/18 (~39%). Evidence: six healthcare postings (Truveta, Midi Health, GW RhythmX, TeleTracking, Layer Health, Flagship Pioneering), three finance (Point72 ×3), one linguistics-adjacent (Babel Street), one scientific (Ai2 CellOLMo). **Not adopted:** mod-113 already covers domain-specialised NLP as a light-touch survey. Vertical-domain hiring expects employer-specific context acquired on the job, not deeper curriculum coverage at this track altitude. The uptick is partly a sampling artifact — healthcare greenhouse boards were disproportionately scrapeable this cycle. Flagged to re-sample a more diverse employer set next cycle.
5. **NLP systems design — 7 / 22 (~32%).** Steady from 6/18 (~33%). Already owned by mod-113 and project-103. No change.
6. **Transformer internals for NLP — 8 / 22 (~36%).** Up from 6/18 (~33%). Already owned by mod-101. No change.

Below-threshold signals that were also considered and skipped:

- **GCP Vertex AI / Amazon Bedrock managed-LLM platforms.** 4 / 22 (~18%) — up from 2/18 (~11%). Still below the 30% threshold; peer-track territory (`ml-engineer` / `mlops`).
- **Knowledge graphs.** 2 / 22 (Point72 KG, CoreStory). Below threshold.
- **Open-source contributions to spaCy / AllenNLP / transformers / langchain.** 1 / 22 (6sense III). Dropped from 3/18; cultural preference at specific employers, not a general market shift.
- **Linguistics / multilingual-speaker background.** 2 / 22 (Babel Street linguistics-centric role, Grammarly computational-linguist collaboration). Narrower than mod-107 would predict; retained in mod-107 for the vendor employers.

## Posting evidence — summary

Full verbatim posting data in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) `postings`. Summary of the 22 verified in-window postings:

| # | Employer | Title | Date | Location | Salary |
|---|---|---|---|---|---|
| 1 | Babel Street | NLP/Linguistics Software Engineer | estimated:2026-Q3 | Somerville, MA | $100k-$120k |
| 2 | Point72 | NLP / AI Engineer | estimated:2026-Q3 | New York, NY | not published |
| 3 | Point72 | NLP Engineer | estimated:2026-Q3 | New York, NY | not published |
| 4 | Point72 | Research Engineer, Knowledge Graph Intelligence | estimated:2026-Q3 | New York, NY | $175k-$250k |
| 5 | Glean | Machine Learning Engineer, Assistant Quality | estimated:2026-Q3 | San Francisco, CA (Hybrid) | $180k-$205k |
| 6 | Ai2 | Senior Research Engineer, Olmo + Molmo | estimated:2026-Q3 | Seattle, WA | $174k-$261k |
| 7 | Ai2 | Young Investigator, Open Language Models for Biology | estimated:2026-Q3 | Seattle, WA | $160k |
| 8 | CoreStory | AI Engineer | estimated:2026-Q3 | Remote US | not published |
| 9 | 6sense | Sr. Machine Learning Engineer | estimated:2026-Q3 | San Francisco, CA | $200k-$261k |
| 10 | 6sense | ML Engineer III | 2026-08-11 | Bengaluru, India | not published |
| 11 | Twilio | Machine Learning Engineer | estimated:2026-Q3 | Remote — Ireland | not published |
| 12 | Institute of Foundation Models | Research Engineer - NLP | estimated:2026-Q3 | Abu Dhabi, UAE | not published |
| 13 | Truveta | Machine Learning Engineer - LLMs & Generative AI | estimated:2026-Q3 | Seattle, WA | $155k-$175k |
| 14 | Midi Health | Senior Software Engineer, AI Engineer | estimated:2026-Q3 | SF or Palo Alto, CA (Hybrid) | $170k-$210k |
| 15 | Get Well Network (GW RhythmX) | AI Engineer | estimated:2026-Q3 | Bangalore, India | not published |
| 16 | TeleTracking Technologies | Machine Learning Engineer III | estimated:2026-Q3 | Pittsburgh, PA | not published |
| 17 | Layer Health | Forward Deploy Data Scientist | estimated:2026-Q3 | Boston or NYC (Hybrid) | $150k-$180k |
| 18 | Flagship Pioneering (FL105) | Machine Learning Research Engineer | estimated:2026-Q3 | Cambridge, MA | $120k-$193k |
| 19 | AvePoint | Junior AI Engineer | 2026-09-14 | Da Nang, Vietnam | not published |
| 20 | GitLab | Senior Software Engineer, NLP | estimated:2026-Q3 | Remote (Americas/EMEA) | not published |
| 21 | GitLab | Staff Software Engineer - NLP | estimated:2026-Q3 | Bangalore, India | not published |
| 22 | Grammarly | Machine Learning Engineer, Agents | estimated:2026-Q3 | San Francisco, CA (Hybrid) | $256k-$506k |

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

Under the continuity bias, this cycle produced **zero net additions** for the second consecutive quarter. Every requirement that cleared the frequency threshold is already either owned by an existing module (mod-101 – mod-113) or owned by a peer track that this curriculum already links out to. The empty delta is captured in [`.aicg/curriculum-plan-delta.json`](.aicg/curriculum-plan-delta.json).

The single signal worth content-authoring attention next pass is **LLM-quality eval authoring** (LLM-as-judge, adversarial / regression test suites) which has now persisted at ≥ 44% across two cycles (44% → 50%). The right response is to **reframe mod-111 exercise-04 to explicitly cover LLM-as-judge and regression-suite authoring** rather than add a new module — the underlying methodology (metric selection, statistical significance, human-eval-protocol design) is already there. That reframe is an authoring pass, not a curriculum-plan-delta item; it should be filed as content work, not a plan change. If the signal persists at ≥ 50% through the 2027-01 cycle, re-evaluate whether a standalone exercise is warranted.

<!-- needs-research: retry frontier-lab (OpenAI, Anthropic, Cohere, Mistral, xAI, AI21, Reka, Apple, Meta GenAI, Google DeepMind, Nvidia), speech-vendor (AssemblyAI, Deepgram, Rev, Speechmatics), and MT-vendor (DeepL, Lilt, Smartling, ModelFront) careers pages next cycle to push the sample size ≥ 25 and re-weight away from healthcare-greenhouse over-sampling. Under-sampled areas this cycle: multilingual / MT (0 / 22), speech interface (2 / 22), tokenization theory (2 / 22), sequence labelling / NER (0 / 22), QA (0 / 22), summarisation (0 / 22). The zero counts in several foundational modeling tasks reflect "LLM" being used as the umbrella hiring term rather than those tasks disappearing. -->
