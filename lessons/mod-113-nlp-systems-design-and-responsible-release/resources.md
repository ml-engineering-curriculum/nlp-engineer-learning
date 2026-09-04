# Resources for mod-113 · NLP Systems Design, Domain Survey, and Responsible Release

Primary docs, standards, and papers. Prefer these over blog posts when digging deeper. Where a primary source exists, cite it; regulatory citations point to the regulator, not to a summary blog.

## Product-goal-to-blueprint and system design

- [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — the foundational practitioner reference for production ML system design; several rules translate directly to NLP task framing and simplicity-first defaults.
- [Google SRE workbook — Implementing SLOs](https://sre.google/workbook/implementing-slos/) — the SLA / SLO framing used in the release-gate chapter.
- [Hugging Face — Tasks](https://huggingface.co/tasks) — a catalogue of the standard NLP task shapes and the models that solve them; useful as a task-framing sanity check.
- [Hugging Face model / tokenizer / pipelines documentation](https://huggingface.co/docs/transformers/) — the reference for encoder / seq2seq / decoder-only families used throughout chapter 1.

## Model families

- [Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019](https://arxiv.org/abs/1810.04805) — the encoder-only reference.
- [Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), JMLR 2020](https://arxiv.org/abs/1910.10683) — the encoder-decoder text-to-text framing.
- [Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*, ACL 2020](https://arxiv.org/abs/1910.13461) — encoder-decoder for generation tasks.
- [Brown et al., *Language Models are Few-Shot Learners* (GPT-3), NeurIPS 2020](https://arxiv.org/abs/2005.14165) — decoder-only in-context learning.
- [Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, 2023](https://arxiv.org/abs/2307.09288) — open-weight decoder reference.
- [Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale* (XLM-R), ACL 2020](https://arxiv.org/abs/1911.02116) — the multilingual encoder reference.
- [Costa-jussà et al., *No Language Left Behind: Scaling Human-Centered Machine Translation* (NLLB), 2022](https://arxiv.org/abs/2207.04672) — the current-state multilingual seq2seq reference for MT.

## Budget allocation and cost modelling

- [Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), NeurIPS 2022](https://arxiv.org/abs/2203.15556) — pretraining compute-vs-data trade-offs; the reference for anyone considering the "training-from-scratch" bucket.
- [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022](https://arxiv.org/abs/2106.09685) — parameter-efficient fine-tuning.
- [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023](https://arxiv.org/abs/2305.14314) — the quantised-fine-tuning approach that fits large models on single GPUs.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft) — the library implementation of LoRA and other adapter methods.
- [FinOps Foundation — FinOps Framework](https://www.finops.org/framework/) — the discipline reference for managed-API cost governance; chapter 2 borrows the cost-per-request instrumentation pattern from here.
- Cloud provider pricing pages — cite URL + retrieval date in any cost model:
    - [OpenAI pricing](https://openai.com/api/pricing/)
    - [Anthropic pricing](https://www.anthropic.com/pricing)
    - [Google Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing)
    - [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
    - [Azure OpenAI pricing](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/)
    - [Cohere pricing](https://cohere.com/pricing)
- [Anthropic — Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching), [OpenAI — Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching), [Google Vertex — Context caching](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview) — caching mechanics that dominate managed-API cost forecasts.
- [MLCO2 — Machine Learning Impact calculator](https://mlco2.github.io/impact/) and [CodeCarbon](https://codecarbon.io/) — tools for estimating training compute emissions for the model-card environmental-footprint field.

## Release gates, candidate selection, and rollback

- [Google SRE book — Release engineering](https://sre.google/sre-book/release-engineering/) — the "how software gets to production" chapter that underpins chapter 3.
- [Google SRE workbook — Canarying releases](https://sre.google/workbook/canarying-releases/) — the canary-analysis reference.
- [Argo Rollouts documentation](https://argo-rollouts.readthedocs.io/) — progressive-delivery controller.
- [Flagger documentation](https://docs.flagger.app/) — progressive-delivery controller (Istio / Linkerd / others).
- [MLflow Model Registry documentation](https://mlflow.org/docs/latest/model-registry.html) — the "alias flip" rollback shape.
- [Semantic Versioning 2.0.0](https://semver.org/) — the versioning discipline release manifests inherit.
- [Google SRE book — Postmortem culture: Learning from failure](https://sre.google/sre-book/postmortem-culture/) — the postmortem reference, cross-cited from mod-112 ch. 7.
- Companion reading: mod-111 chapters [7 (paired significance)](../mod-111-nlp-evaluation/07-statistical-significance-for-nlp.md), [8 (contamination)](../mod-111-nlp-evaluation/08-contamination-detection-and-decontamination.md); mod-112 chapters [5 (packaging)](../mod-112-production-nlp-pipelines/05-packaging-and-reproducible-shipping.md), [6 (drift)](../mod-112-production-nlp-pipelines/06-nlp-drift-instrumentation.md), [7 (postmortem)](../mod-112-production-nlp-pipelines/07-postmortem-of-a-degraded-nlp-service.md).

## Domain NLP — clinical

- [US HHS — Guidance Regarding Methods for De-identification of Protected Health Information](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/) — the primary reference for HIPAA Safe Harbor and Expert Determination.
- [HHS — HIPAA for Professionals](https://www.hhs.gov/hipaa/for-professionals/index.html) — the overall HIPAA regulatory framework.
- [MIMIC-III / MIMIC-IV data and DUA](https://mimic.mit.edu/) — the reference credentialed critical-care corpus.
- [n2c2 NLP Research Data Sets (i2b2 successor)](https://www.i2b2.org/NLP/DataSets/) — the challenge-series clinical-NLP corpora.
- [Apache cTAKES](https://ctakes.apache.org/) — the long-standing UIMA-based clinical NLP pipeline.
- [scispaCy](https://allenai.github.io/scispacy/) — spaCy pipelines pretrained on biomedical text.
- [medspaCy](https://github.com/medspacy/medspacy) — clinical-note-specific spaCy tooling.
- [Gu et al., *Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing* (PubMedBERT), ACM TCHIT 2021](https://arxiv.org/abs/2007.15779) — the domain-adapted-pretraining reference.
- [Lee et al., *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*, Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) — the biomedical BERT reference.
- [FDA — Software as a Medical Device (SaMD)](https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd) — the US device-classification framework.
- [Regulation (EU) 2017/745 — Medical Device Regulation (MDR)](https://eur-lex.europa.eu/eli/reg/2017/745/oj) — the EU counterpart.
- [European Data Protection Board — Guidelines on processing of special-category data](https://edpb.europa.eu/) — GDPR Article 9 practical guidance.

## Domain NLP — legal

- [Hendrycks et al., *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021 D&B](https://arxiv.org/abs/2103.06268) — the reference clause-extraction benchmark.
- [CUAD project + data (Atticus Project)](https://www.atticusprojectai.org/cuad) — the corpus itself.
- [Chalkidis et al., *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL 2022](https://arxiv.org/abs/2110.00976) — the legal benchmark suite.
- [Chalkidis et al., *LEGAL-BERT: The Muppets straight out of Law School*, EMNLP 2020 Findings](https://arxiv.org/abs/2010.02559) — the domain-adapted encoder for English legal text.
- [Case.law / Caselaw Access Project (Harvard)](https://case.law/) — the open case-law corpus.
- [ABA — Formal Opinion 512, Generative Artificial Intelligence Tools (2024)](https://www.americanbar.org/groups/professional_responsibility/publications/aba_formal_opinions/) — the US professional-responsibility opinion on generative-AI use in legal practice.
- Mata v. Avianca, Inc., S.D.N.Y. 22-cv-1461 — the June 22, 2023 sanctions order (docket available via [CourtListener](https://www.courtlistener.com/docket/63107798/mata-v-avianca-inc/)) is the fabricated-citation incident that crystallised the professional-liability side of generative legal NLP.

## Domain NLP — financial

- [SEC EDGAR full-text search](https://efts.sec.gov/LATEST/search-index?q=&dateRange=custom) — programmatic access to US public-company filings.
- [Malo et al., *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts* (Financial PhraseBank), JASIST 2014](https://arxiv.org/abs/1307.5336) — the reference financial-sentiment corpus.
- [Yang et al., *FinBERT: A Pretrained Language Model for Financial Communications*, 2020](https://arxiv.org/abs/2006.08097) — the domain-adapted encoder.
- [Wu et al., *BloombergGPT: A Large Language Model for Finance*, 2023](https://arxiv.org/abs/2303.17564) — the domain-scale decoder case study.
- [SEC — Regulation Fair Disclosure (Reg FD)](https://www.sec.gov/rules/final/33-7881.htm) — the US non-selective-disclosure regulation.
- [Regulation (EU) 596/2014 — Market Abuse Regulation (MAR)](https://eur-lex.europa.eu/eli/reg/2014/596/oj) — the EU market-abuse framework.
- [FINRA — Rules and Guidance](https://www.finra.org/rules-guidance) — the US broker-dealer supervisory framework governing communication surveillance.
- [OFAC — Sanctions Programs and Country Information](https://ofac.treasury.gov/sanctions-programs-and-country-information) — the US sanctions regime driving KYC / AML NER.
- [FCA — Handbook](https://www.handbook.fca.org.uk/) — the UK regulator's rulebook.
- [ESMA — Market Integrity](https://www.esma.europa.eu/policy-activities/market-integrity) — the EU securities regulator's oversight remit.

## Domain NLP — scientific

- [PubMed Central Open Access Subset](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) — the redistributable biomedical full-text corpus.
- [PubMed E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25501/) — programmatic access to PubMed.
- [Semantic Scholar Open Research Corpus](https://www.semanticscholar.org/product/api) — the citation-graph substrate.
- [OpenAlex](https://openalex.org/) — the open scholarly-record dataset.
- [Neumann et al., *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing*, BioNLP 2019](https://arxiv.org/abs/1902.07669) — the scientific-NLP pipeline reference.
- [SciBERT](https://arxiv.org/abs/1903.10676) and [BioMed-RoBERTa](https://arxiv.org/abs/2004.10964) — the domain-adapted encoders for scientific text.
- [BC5CDR, BioCreative shared tasks](https://biocreative.bioinformatics.udel.edu/) — the reference biomedical IE benchmarks.
- [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html) — the biomedical terminology superset; concept-normalisation target.
- [Retraction Watch Database](http://retractiondatabase.org/) — the reference dataset for retraction status.
- [FAIR Data Principles](https://www.go-fair.org/fair-principles/) — the community norms for scientific data-sharing.

## Model cards, dataset cards, and responsible-release

- [Gebru et al., *Datasheets for Datasets*, Communications of the ACM 2021](https://arxiv.org/abs/1803.09010) — the foundational dataset-card reference.
- [Mitchell et al., *Model Cards for Model Reporting*, FAT\* 2019](https://arxiv.org/abs/1810.03993) — the foundational model-card reference.
- [Hugging Face — Model Cards documentation](https://huggingface.co/docs/hub/model-cards) — the practical model-card template used across the Hub.
- [Hugging Face — Dataset Cards documentation](https://huggingface.co/docs/hub/datasets-cards) — the dataset-card template counterpart.
- [Hugging Face — Model Card Guidebook](https://huggingface.co/docs/hub/en/model-card-guidebook) — the practitioner walk-through of model-card fields.
- [Google Research — Data Cards Playbook](https://sites.research.google/datacardsplaybook/) — the complementary Google practitioner guide.
- [NIST — AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework) — the US federal risk-management framework governance partners will map back to.
- [NIST AI RMF Playbook](https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook) — the actionable companion to the RMF.
- [Regulation (EU) 2024/1689 — Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — the EU AI Act.
- [OECD AI Principles](https://oecd.ai/en/ai-principles) — the multilateral norm-setting reference many national frameworks build on.
- [Partnership on AI — About ML](https://partnershiponai.org/workstream/about-ml/) — cross-industry model-documentation guidance.
- [ISO/IEC 42001:2023 — Artificial intelligence management systems](https://www.iso.org/standard/42001.html) — the emerging management-system standard governance partners increasingly cite.

## Cross-track references

- [Learning from Incidents in Software (learningfromincidents.io)](https://www.learningfromincidents.io/) — cross-company practitioner community on incident learning; applies to NLP releases unchanged.
- [Google — Machine Learning Test Score](https://research.google/pubs/pub46555/) — the "how tested is this ML system?" rubric that pairs with chapter 3's release gates.
- Companion modules in this track:
    - [mod-110 · NLP data engineering](../mod-110-nlp-data-engineering/README.md) — upstream of the dataset cards.
    - [mod-111 · NLP evaluation](../mod-111-nlp-evaluation/README.md) — upstream of the offline gate.
    - [mod-112 · Production NLP pipelines](../mod-112-production-nlp-pipelines/README.md) — upstream of the operational and live gates.
