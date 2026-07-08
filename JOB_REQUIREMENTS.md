# Job Requirements — NLP Engineer

**Role level:** 30 (deep specialist — peer to Senior ML Engineer on the ladder)
**Track:** `nlp-engineer-learning`
**Research window:** 2026-03-27 → 2026-06-25 (last 90 days)
**Today:** 2026-06-25

This file documents the requirements catalog used to seed the NLP Engineer curriculum. Raw normalized data lives in [`.aicg/job-requirements.json`](.aicg/job-requirements.json); the planned curriculum lives in [`.aicg/curriculum-plan.json`](.aicg/curriculum-plan.json) (authored in a follow-up phase).

## Status — bootstrap session, postings deferred

<!-- needs-research: collect >= 25 distinct in-window postings titled "NLP Engineer" / "Natural Language Processing Engineer" / "Senior NLP Engineer" / "Computational Linguist (Engineer)" / "Speech & Language Engineer" / "Language AI Engineer" (NOT Staff / Principal modifiers — they will inherit this packet — and NOT generic "ML Engineer" / "Data Scientist" / "Research Scientist" / "LLM Application Engineer" / "AI Engineer" / "Generative AI Engineer" titles, which are owned by peer tracks). Re-validate every requirement against the live evidence and demote any whose evidence stays empty. -->

This packet was authored in a bootstrap session **without an exercised WebSearch / WebFetch pass against live job boards**. The web tools returned `"Claude requested permissions ... not yet granted"` on the first attempts:

- `WebSearch` on `"NLP Engineer" job posting site:boards.greenhouse.io 2026`
- `WebSearch` on `"NLP Engineer" job site:jobs.lever.co 2026`
- `WebFetch` on `https://boards.greenhouse.io/anthropic`

Per the project rules (*"No fabricated URLs, employers, dates, or salary figures... If you cannot find >= 25 verifiable in-window postings... DO NOT INVENT. Write what you have, mark the shortfall honestly"*), the `postings` array in `.aicg/job-requirements.json` is intentionally empty — the curriculum will not claim to have analysed 25 live postings when none were fetched.

The autonomous research loop is expected to fill that gap on its next cycle. To keep the loop deterministic and prevent the requirements catalogue from being invented, this document grounds every requirement in **authoritative public references** that document what the role is hired against:

- **Canonical textbooks:** Jurafsky & Martin SLP3 (2024-25 draft), Eisenstein NLP.
- **Canonical libraries:** Hugging Face transformers / tokenizers / datasets / evaluate, spaCy and spaCy Projects, NLTK, Stanford CoreNLP, Stanza, sentence-transformers, fastText, Gensim, AllenNLP, Flair, fairseq, Marian NMT, seqeval.
- **Foundational papers:** "Attention Is All You Need" (Vaswani 2017), BERT (Devlin 2019), RoBERTa (Liu 2019), T5 (Raffel 2020), GPT-2 (Radford 2019), BPE (Sennrich 2016), SentencePiece (Kudo 2018), WordPiece (Wu 2016), mBART (Liu 2020), XLM-R (Conneau 2020), NLLB (Meta 2022), Whisper (Radford 2022), wav2vec 2.0 (Baevski 2020), MTEB (Muennighoff 2023), BLEU (Papineni 2002), ROUGE (Lin 2004), COMET (Rei 2020), BERTScore (Zhang 2020).
- **Canonical benchmarks:** SQuAD / SQuAD 2.0, GLUE / SuperGLUE, XTREME / XNLI, CoNLL-2003 NER, CoNLL-2012 coreference, Universal Dependencies, FLORES-101 / FLORES-200, MTEB.
- **Standards:** ISO 639, IETF BCP 47, Unicode UAX #29 (text segmentation), ICU.
- **Responsible-NLP framework:** ACL ethics policy, Datasheets for Datasets (Gebru 2021), Model Cards for Model Reporting (Mitchell 2019).

Every requirement below cites at least one such reference, and every requirement is shaped so that posting-frequency evidence can be added underneath it without restructure.

## Methodology

1. Sourced the canonical task domains for a deep-specialist NLP engineer from public references — see `authoritative_references` in `.aicg/job-requirements.json` (54 entries total covering textbooks, libraries, papers, benchmarks, and standards).
2. Mapped each task domain to (a) the role on our level ladder that should own it primarily and (b) the curriculum module that covers it.
3. Applied the **ownership rule**: assign coverage to the lowest-level role that genuinely requires the skill, with higher levels linking back rather than duplicating fundamentals. Defer down to `ml-engineer` (level 20) for ML / PyTorch / packaging fundamentals; defer sideways to `fine-tuning-engineer` (level 30, peer specialist) for the post-training stack, to `rag-engineer` (level 25) for retrieval-augmented generation depth, to `llm-application-developer` (level 25) for prompting / agent depth, to `model-evaluation-engineer` / `ai-eval-engineer` (level 25) for eval-as-platform depth, to `training-pipeline-engineer` (level 25) for distributed-training infrastructure depth, to `ai-infra-security-learning` (level 35) for deep ML/AI security, and to `ai-governance-analyst` / `ai-risk-engineer` for governance / risk depth.
4. Flagged everything that has not yet been validated against in-window postings so the next cycle can demote any requirement whose evidence stays empty.

## Requirement themes -> curriculum ownership

The table below lists each requirement theme, its planned owner per the level hierarchy, and the curriculum coverage path. **Freq** is intentionally blank for this cycle — it will be backfilled by the next research pass. Several requirements share a single curriculum module where the topics are naturally co-taught — the `curriculum-plan.json` consolidates 20 NLP-owned requirements into 13 modules.

| # | Theme | Freq | Owner role | Coverage |
|---|---|---|---|---|
| 1 | Tokenization theory (BPE / WordPiece / Unigram / SentencePiece, multilingual / CJK, tokenizer training, chat-template arithmetic) | <!-- needs-research --> | `nlp-engineer` (this) | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) |
| 2 | Transformer internals for NLP (encoder vs encoder-decoder vs decoder-only choice, KV cache for generation, probing, tokenizer-shape effects) | <!-- needs-research --> | `nlp-engineer` | [`mod-101-tokenization-and-text-foundations`](lessons/mod-101-tokenization-and-text-foundations) (co-taught with tokenization) |
| 3 | Classical NLP (regex, FST, n-gram LMs, taggers, parsers, lemmatisation, Unicode-correct text processing, language identification) | <!-- needs-research --> | `nlp-engineer` | [`mod-102-classical-nlp`](lessons/mod-102-classical-nlp) |
| 4 | Text classification end-to-end (baselines through fine-tuned encoders, multilingual, multi-label, calibration, cost-sensitive eval) | <!-- needs-research --> | `nlp-engineer` | [`mod-103-text-classification`](lessons/mod-103-text-classification) |
| 5 | Sequence labelling and NER (BIO/BIOES/BILOU, span models, CRF heads, subword-word alignment, entity-level eval, domain NER) | <!-- needs-research --> | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) |
| 6 | Information extraction (relation / event extraction, slot filling, entity linking, coreference, document-level IE, structured generation) | <!-- needs-research --> | `nlp-engineer` | [`mod-104-sequence-labelling-and-information-extraction`](lessons/mod-104-sequence-labelling-and-information-extraction) (co-taught with NER) |
| 7 | Question answering and machine reading comprehension (extractive / abstractive / multi-hop / long-context, unanswerability, EM / F1) | <!-- needs-research --> | `nlp-engineer` | [`mod-105-question-answering-and-machine-reading`](lessons/mod-105-question-answering-and-machine-reading) |
| 8 | Summarisation (extractive / abstractive, long-document, multi-document, controllable, faithfulness, ROUGE / BERTScore) | <!-- needs-research --> | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) |
| 9 | Generative NLP and decoding (sampling strategies, constrained / structured generation, length / faithfulness, tokenization edge cases) | <!-- needs-research --> | `nlp-engineer` | [`mod-106-summarisation-and-controlled-generation`](lessons/mod-106-summarisation-and-controlled-generation) (co-taught with summarisation — the most decoding-sensitive task) |
| 10 | Machine translation (encoder-decoder NMT, multilingual / low-resource MT, terminology constraints, COMET / BLEU / chrF, FLORES) | <!-- needs-research --> | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) |
| 11 | Multilingual and low-resource NLP (XLM-R, mBART, mT5, NLLB, script handling, transliteration, BCP-47-aware eval) | <!-- needs-research --> | `nlp-engineer` | [`mod-107-machine-translation-and-multilingual-nlp`](lessons/mod-107-machine-translation-and-multilingual-nlp) (co-taught with MT — shared pretraining data, evaluation, and script-handling concerns) |
| 12 | Embeddings and representation learning (bi/cross-encoders, contrastive learning, hard-negative mining, MTEB, multilingual embeddings) | <!-- needs-research --> | `nlp-engineer` | [`mod-108-embeddings-and-representation-learning`](lessons/mod-108-embeddings-and-representation-learning) |
| 13 | Speech / text interface (ASR fundamentals, text normalisation, ITN, punctuation restoration, diarisation post-processing) | <!-- needs-research --> | `nlp-engineer` (light touch) | [`mod-109-speech-text-interface`](lessons/mod-109-speech-text-interface) |
| 14 | NLP data engineering (corpus collection, language ID, dedup, PII scrubbing, deboilerplating, annotation workflow, weak supervision) | <!-- needs-research --> | `nlp-engineer` | [`mod-110-nlp-data-engineering`](lessons/mod-110-nlp-data-engineering) |
| 15 | NLP-specific evaluation (BLEU / ROUGE / COMET / BERTScore / seqeval / SQuAD F1, statistical significance, contamination, slicing) | <!-- needs-research --> | `nlp-engineer` | [`mod-111-nlp-evaluation`](lessons/mod-111-nlp-evaluation) |
| 16 | Production NLP pipelines (spaCy Projects / HF Pipelines composition, document-level pipelines, NLP-specific drift monitoring, latency SLAs) | <!-- needs-research --> | `nlp-engineer` | [`mod-112-production-nlp-pipelines`](lessons/mod-112-production-nlp-pipelines) |
| 17 | NLP systems design (task framing, model-family choice, build-vs-buy, evaluation gates, domain-adaptation strategy, cost/latency/quality) | <!-- needs-research --> | `nlp-engineer` | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) + [`project-103-nlp-capstone-production-system`](projects/project-103-nlp-capstone-production-system) |
| 18 | Domain-specialised NLP (clinical, legal, financial, scientific) — survey-style coverage of the patterns NLP engineers meet at vertical employers | <!-- needs-research --> | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) (co-taught with systems-design as the natural place for vertical-domain pattern survey) |
| 19 | Responsible NLP at engineer altitude (datasheets, model cards, demographic / dialect slicing, PII / toxicity / licensing) | <!-- needs-research --> | `nlp-engineer` (light touch) | [`mod-113-nlp-systems-design-and-responsible-release`](lessons/mod-113-nlp-systems-design-and-responsible-release) (co-taught with systems-design — release-level responsible-NLP package) |
| 20 | PyTorch / classical ML / FastAPI / Docker / MLflow / experiment tracking fundamentals | n/a — prerequisite | `ml-engineer` (level 20) | Listed in [`PREREQUISITES.md`](PREREQUISITES.md); not re-taught |
| 21 | Post-training stack depth (SFT / PEFT / RLHF / DPO / ORPO / KTO) | n/a — peer track | `fine-tuning-engineer` (level 30) | Touched only where NLP-task adaptation requires it; depth linked out |
| 22 | RAG depth (chunking, vector stores, hybrid retrieval, rerankers, retrieval evaluation) | n/a — peer track | `rag-engineer` (level 25) | Surfaced at the QA (mod-105) and embeddings (mod-108) boundaries; depth linked out |
| 23 | LLM application development (prompting, agents, tool use, product integration) | n/a — peer track | `llm-application-developer` (level 25) | Surfaced in mod-106 (decoding) and mod-113 (systems-design build-vs-buy); depth linked out |
| 24 | Eval platform engineering (eval-as-product, statistical methodology depth, LLM-as-judge platform internals) | n/a — peer track | `model-evaluation-engineer` / `ai-eval-engineer` (level 25) | Module 111 covers the metric catalogue; platform depth linked out |
| 25 | Distributed-training PLATFORM engineering (multi-tenant schedulers, NCCL / fabric tuning) | n/a — peer track | `training-pipeline-engineer` (level 25) | This curriculum runs distributed training at operator altitude only; platform depth linked out |
| 26 | Deep ML / AI security (data poisoning, model extraction, training-data exfiltration, prompt-injection-resistance training) | n/a — higher level | `ai-infra-security-learning` (level 35) | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |
| 27 | Governance / compliance / model cards / dataset licensing review / regulated-data NLP | n/a — peer track | `ai-governance-analyst` / `ai-risk-engineer` | Surfaced as awareness in mod-110 / mod-113; depth owned upstream |

## Posting evidence

<!-- needs-research: populate with the >= 25 in-window postings sampled next cycle. Use the same table shape as `ai-infra-agentic-ai-engineer-learning/JOB_REQUIREMENTS.md`. -->

No postings were sampled this cycle. See the **Status** section above for the reason. The next autonomous research cycle should fan out across:

- ATS roots: `job-boards.greenhouse.io`, `boards.greenhouse.io`, `jobs.lever.co`, `jobs.ashbyhq.com`.
- **Frontier labs:** OpenAI, Anthropic, Google DeepMind, Cohere, Mistral, AI21, xAI, Reka, Inflection, Character.AI.
- **Enterprise NLP shops:** Apple (Siri / Apple AI/ML), Amazon (Alexa AI / AGI), Meta (GenAI / FAIR), Microsoft (incl. Nuance), IBM Research, Bloomberg AI, Reuters / Refinitiv, LexisNexis, ServiceNow, Salesforce (Einstein / AI Research), Adobe (Sensei), Spotify, Spotify Research.
- **Search / conversational:** DuckDuckGo, Perplexity, You.com, Replika, Character.AI.
- **Translation / multilingual:** DeepL, Lilt, Unbabel, ModelFront.
- **Speech:** Speechmatics, AssemblyAI, Deepgram, Soniox.
- **Healthtech NLP:** Abridge, Suki, Augmedix, Nuance/Microsoft.
- **Legal / finance NLP:** Casetext, Harvey, Hebbia, Stripe, Plaid, Tradeshift.
- **Academic / non-profit:** Allen AI / AI2, EleutherAI, Hugging Face.

For each posting, capture employer, exact title, URL, `date_observed`, `date_posted` (or `estimated:2026-MM`), location, 5-10 verbatim requirement bullets, 2-6 preferred-qualification bullets, `salary_range` (verbatim string or `null` when unpublished), and one short representative quote. **Filter out**:

- Senior / Staff / Principal modifiers (they will inherit this packet).
- Generic "ML Engineer" / "Data Scientist" / "Research Scientist" titles (owned by peer tracks).
- "LLM Application Engineer" / "AI Engineer" / "Generative AI Engineer" titles (owned by `llm-application-developer`).
- "Conversational AI" titles where the work is plainly chatbot / prompt engineering without language-modeling depth.

## Ownership map — quick reference for next cycle

When backfilling postings, use this ownership decision to keep the curriculum from drifting into peer territory:

- **NLP Engineer (this track, level 30)** owns language-specific depth end-to-end: tokenization theory, classical NLP, modern transformer NLP for sequence / structured / generative language tasks (NER, classification, summarisation, MT, information extraction, semantic parsing, text generation), embeddings and representation learning, multilingual and low-resource NLP, the speech / text interface where relevant, NLP-specific data engineering and evaluation, and production NLP-pipeline / systems design.
- **ML Engineer** (level 20) owns the build-altitude practitioner workflow this curriculum assumes (Python / PyTorch fluency, FastAPI / Docker packaging, MLflow tracking, classical eval).
- **Senior ML Engineer** (level 30, peer ladder) is the *generalist* peer at the same level — overlaps in NLP vocabulary but does not own the depth.
- **Staff / Principal ML Engineer** (level 40+) inherit / link to this packet for NLP depth, but add architectural and cross-team scope.
- **Fine-Tuning Engineer** (level 30, peer specialist) owns the post-training stack (SFT / PEFT / RLHF / DPO). This curriculum *uses* fine-tuning to adapt models to NLP tasks but does not re-teach the full stack.
- **RAG Engineer** (level 25) owns retrieval-augmented generation depth. Surfaced only at the QA (mod-106) and embeddings (mod-109) boundaries.
- **LLM Application Developer** (level 25) owns prompting, agents, tool use, product integration. Surfaced only in mod-110 (decoding) and mod-117 (build-vs-buy).
- **Model Evaluation / AI Eval Engineer** (level 25) own eval-as-platform depth. This curriculum owns the NLP metric catalogue (mod-114).
- **Training Pipeline Engineer** (level 25) owns distributed-training INFRASTRUCTURE. This curriculum operates it as a user only.
- **AI Infrastructure Security Engineer** (level 35) owns deep ML/AI security.
- **AI Governance Analyst / AI Risk Engineer** own governance / compliance / risk depth.

## Conclusion

<!-- needs-research: re-run on the next autonomous cycle with web tools exercised against live job boards, populate `postings` in `.aicg/job-requirements.json`, backfill the Freq column above, and demote any requirement whose `evidence_post_ids` stays empty. -->

The requirements catalog in [`.aicg/job-requirements.json`](.aicg/job-requirements.json) is structured so that posting frequencies can be added underneath each requirement without restructure. The themes themselves are grounded in cited public references — the dominant open-source NLP stack, foundational papers, canonical benchmarks, and the textual / locale standards (Unicode, BCP 47, ISO 639) — not invented; they are the canonical task domains the role is hired against in industrial NLP teams and frontier labs.
