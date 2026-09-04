# Domain NLP Patterns and Their Data-Governance Constraints

## Motivation

Every domain has its own NLP tradition — its canonical tasks, its labelled corpora, its measurement conventions, its regulators, and its non-negotiable data-handling constraints. An NLP engineer who walks into clinical or legal or financial or scientific NLP without knowing the patterns and constraints ships a system the domain will reject on grounds that were obvious to everyone else in the room.

This chapter surveys the four domain areas the NLP-engineer role most often has to work in — clinical, legal, financial, scientific — for the *patterns* (canonical tasks, canonical models, canonical benchmarks) and for the *governance* constraints (regulation, licensing, PII/PHI, audit trail) each imposes. The goal is not to make you a domain expert; it is to give you the vocabulary and the guardrails to have the first conversation with the domain owner without embarrassing the team.

Depth on the compliance / risk side is handed off upstream (mod-114-and-beyond in the parent curriculum; the ML risk track for the org). This chapter is the NLP-engineer-side survey.

## Clinical NLP: de-identification and coding

### The canonical tasks

Clinical NLP works on **clinical notes** (discharge summaries, progress notes, radiology reports, pathology reports), **discharge letters**, **claims data**, and **structured EHR fields with free-text remarks**. Two task families dominate:

- **De-identification (de-id).** Detect and remove or replace direct and indirect Protected Health Information (PHI) — names, dates, medical record numbers, ages over 89, locations, etc. Structured as span-level NER with a canonical label set. Anchored on the **HIPAA Safe Harbor** enumeration of 18 identifier categories (US Department of Health and Human Services, ["Guidance Regarding Methods for De-identification of PHI"](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/)).
- **Medical coding.** Map free-text clinical statements to structured terminologies — **ICD-10 / ICD-11** (diagnoses), **CPT / HCPCS** (procedures, billing), **SNOMED CT** (clinical terms), **RxNorm** (drugs), **LOINC** (labs). Structured as classification-over-huge-label-space, extraction-then-linking (mod-104 chapter 4), or generation-with-constrained-output. The i2b2 / n2c2 shared tasks (n2c2 is the successor; [n2c2 NLP research data sets](https://www.i2b2.org/NLP/DataSets/)) are the reference challenge series.

Adjacent tasks: **assertion detection** (is the mentioned condition present, absent, hypothetical, historical), **temporal reasoning**, **medication reconciliation**, **adverse-event extraction from spontaneous reports**, and **section segmentation** of the notes themselves.

### The canonical stacks

- **cTAKES** (Apache) — the long-standing clinical NLP pipeline, UIMA-based, ships with SNOMED / RxNorm dictionaries. [Apache cTAKES](https://ctakes.apache.org/).
- **MedSpaCy / scispaCy** — spaCy pipelines tuned for biomedical text; scispaCy ships models trained on PubMed abstracts and full text. [scispaCy](https://allenai.github.io/scispacy/), [medspaCy](https://github.com/medspacy/medspacy).
- **ClinicalBERT / BioBERT / PubMedBERT** — domain-adapted encoder-only models pretrained on clinical / biomedical corpora. PubMedBERT (Gu et al., ["Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing"](https://arxiv.org/abs/2007.15779), *ACM TCHIT 2021*) is the standard reference for domain-specific pretraining outperforming domain-adaptive fine-tuning on biomedical benchmarks.
- **Med-PaLM 2 / Med-Gemini / MedLM** and equivalents — clinical-tuned LLMs from Google. Commercial only; audit and access controls vary by region.
- **Amazon Comprehend Medical, Google Cloud Healthcare NLP API, Microsoft Azure Text Analytics for Health, John Snow Labs' Spark NLP for Healthcare** — managed offerings; the licensing terms determine whether they clear a HIPAA data-flow review.

### The governance envelope

The dominant constraint is **regulatory data handling**, not model quality.

- **HIPAA (US).** Protected Health Information cannot leave the covered entity's boundary except under a Business Associate Agreement (BAA) with the receiving party. Using a managed API without a BAA on PHI is a HIPAA violation. Cloud providers offer HIPAA-eligible services with BAAs — the eligibility list is per-service, not per-account, and changes over time. [HHS — HIPAA for Professionals](https://www.hhs.gov/hipaa/for-professionals/index.html).
- **GDPR (EU).** Special-category data (health) requires an explicit lawful basis (typically Article 9 conditions — explicit consent, healthcare-provision necessity, public-health mandate). Data-subject rights (access, deletion) apply; a model trained on such data inherits complicated deletion questions. [EDPB guidelines on processing of special-category data](https://edpb.europa.eu/).
- **Country-specific regimes.** UK NHS Digital Data Security and Protection Toolkit; Canada PHIPA; Australia Privacy Act + My Health Records Act; regional data-residency laws. Any pan-jurisdictional clinical NLP system inherits *the intersection* of the constraints, not the average.
- **Institutional Review Board (IRB) approvals.** Research use of clinical data at academic medical centers typically requires IRB approval; production use requires additional data-governance approvals (Data Use Agreement, Data Access Committee). [ClinicalTrials.gov policies](https://clinicaltrials.gov/) frame trial-context data.
- **De-identification is a specific legal act.** HIPAA Safe Harbor (removing the 18 identifier categories) or Expert Determination (a statistician certifies the residual re-identification risk is very small). Both are legal artifacts, not just NLP outputs — a "de-id model" whose output has not been certified against one of the two standards is not de-identified in the regulatory sense.

The NLP-engineer-side takeaways:

- **Data leaves your VPC only under a paper trail.** No prototyping on a managed API without a BAA. Even hashed / redacted data usually requires a data-flow review.
- **The training set is a compliance artifact.** MIMIC-III / MIMIC-IV (the widely-used de-identified critical-care datasets from MIT / BIDMC — [MIMIC](https://mimic.mit.edu/)) require a signed Data Use Agreement and completion of the CITI Data or Specimens Only Research course before download. Redistribution is forbidden.
- **The eval set is also a compliance artifact.** i2b2 / n2c2 challenge data is DUA-gated; challenge results published against it do not confer redistribution rights.
- **The model itself may be regulated.** In the EU, a clinical-decision-support model may be a Medical Device under the Medical Device Regulation (MDR / EU 2017/745); in the US, the FDA's Software as a Medical Device (SaMD) framework applies. Not every clinical NLP model triggers device classification, but the ones that produce recommendations acted on by clinicians often do. Loop in regulatory counsel early.

## Legal NLP: clause extraction and contract analytics

### The canonical tasks

Legal NLP works on **contracts** (MSAs, NDAs, SaaS ToS, employment agreements, financing agreements), **court opinions and pleadings**, **statutes and regulations**, **patents**, and **regulatory filings** that overlap with financial NLP below.

- **Clause extraction.** Detect and classify canonical clauses — indemnification, limitation of liability, non-compete, non-solicit, governing law, term & termination, IP assignment, change-of-control. Structured as span extraction with a clause taxonomy (mod-104 chapter 1). The benchmark is **CUAD** (Hendrycks et al., ["CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"](https://arxiv.org/abs/2103.06268), *NeurIPS 2021 D&B*), the Contract Understanding Atticus Dataset, ~13 000 labels across 41 clause types in 510 contracts.
- **Judgment prediction, statutory retrieval, citation graph analysis, case similarity search.** The **LexGLUE** benchmark (Chalkidis et al., ["LexGLUE: A Benchmark Dataset for Legal Language Understanding in English"](https://arxiv.org/abs/2110.00976), *ACL 2022*) aggregates several of these.
- **Redaction and privilege review.** Detect privileged content (attorney–client, work product), attorney-marked confidentiality, or personally identifiable information for discovery / disclosure workflows.
- **Contract generation and review assist.** LLM-driven drafting suggestions, redline proposals, playbook enforcement — the fastest-growing commercial application category, and the one with the most active hallucination-risk debate (below).

### The canonical stacks

- **LegalBERT** (Chalkidis et al., ["LEGAL-BERT: The Muppets straight out of Law School"](https://arxiv.org/abs/2010.02559), *EMNLP 2020 Findings*) — BERT pretrained on English legal corpora; the reference encoder for many legal NLP baselines.
- **CaseLaw Access Project embeddings** and other case-law-derived corpora (Harvard's CAP — [case.law](https://case.law/)) — the licence and terms of use frame what a research corpus can be redistributed as.
- **Prodigy / Label Studio for expert clause annotation.** Expert legal annotation is expensive; every mature stack has a heavy investment in in-house annotation tooling and a small pool of trained lawyers-as-annotators.
- **Vendor stacks** — Ironclad, Kira Systems, Evisort, ContractPodAi, LexisNexis, Thomson Reuters (Practical Law / CoCounsel / Westlaw). The buy option often dominates in this domain because the domain expertise embedded in the vendor's taxonomy is significant.

### The governance envelope

- **Attorney–client privilege.** Documents that carry privilege cannot be exposed to third-party services in ways that waive the privilege. Sending privileged contract text to a public LLM is a well-known concern; ABA opinions (see e.g. American Bar Association Standing Committee on Ethics and Professional Responsibility guidance on lawyers' use of AI — [ABA Formal Opinion 512, ["Generative Artificial Intelligence Tools"](https://www.americanbar.org/groups/professional_responsibility/publications/aba_formal_opinions/), 2024]) frame the professional-responsibility side.
- **Confidentiality and NDA obligations.** Contract text is usually confidential; the party analysing the contract may itself be bound by NDA. Managed-API stances mirror clinical: no BAA-equivalent, no send.
- **Hallucination risk with real accountability.** The *Mata v. Avianca* incident (S.D.N.Y. 2023 — attorneys sanctioned for filing a brief with fabricated citations generated by ChatGPT) crystallised the professional-liability side of generative legal NLP. Systems that surface citations must ground them; systems that draft text a lawyer will sign must make hallucinated authority impossible or auditable. RAG with strict citation-check gates (mod-105 chapter 8) is the standard mitigation.
- **Retention and audit.** For discovery-support systems, every model action on every document is potentially discoverable itself. The audit trail is a feature, not an operational nice-to-have.
- **Jurisdictional variance.** US federal vs. state, EU vs. UK vs. Switzerland; the "governing law" clause exists for a reason. A contract-analytics system that folds clauses from different jurisdictions into a single taxonomy has to reason about that variance somewhere.

The NLP-engineer-side takeaways:

- **The clause taxonomy is the product.** More than the model — the accepted-in-the-industry taxonomy (CUAD-style) is where the domain expertise sits.
- **Every citation must be groundable.** Never surface a case or statute reference that was not extracted from a real source; RAG plus a citation-existence check is the pattern.
- **Contract corpora do not distribute cleanly.** CUAD is CC-BY-4.0; most others are licensed. Do not assume redistribution rights on any legal corpus you did not create.

## Financial NLP: event extraction, sentiment, and disclosure analytics

### The canonical tasks

Financial NLP works on **earnings-call transcripts**, **SEC filings** (10-K, 10-Q, 8-K, S-1, proxy statements), **analyst research**, **news wires and social media** ("finance Twitter"), **regulatory correspondence**, **KYC / AML documents**, and **internal trading communications** (Slack, Bloomberg messages) subject to compliance review.

- **Event extraction.** Detect corporate events — mergers, acquisitions, executive changes, earnings surprises, product launches, dividend changes, insider transactions, regulatory actions. Structured as span extraction + argument roles (mod-104 chapter 3).
- **Sentiment / tone analysis.** Directional sentiment on news, earnings-call sentiment (delta between prepared remarks and Q&A), management tone in filings. The **Financial PhraseBank** (Malo et al., ["Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts"](https://arxiv.org/abs/1307.5336), *JASIST 2014*) is a widely-cited labelled sentiment dataset. **FinBERT** (Yang et al., ["FinBERT: A Pretrained Language Model for Financial Communications"](https://arxiv.org/abs/2006.08097), 2020) is the go-to domain-adapted encoder.
- **Named-entity recognition — companies, tickers, funds, people, currencies, dates.** Extended NER with financial types.
- **Disclosure / risk-factor analytics.** Extract and cluster the "Risk Factors" section (10-K Item 1A), track changes across filings, benchmark against peers. Structured as retrieval + summarisation + change-detection.
- **KYC / AML NER.** Party name, address, beneficial-ownership extraction with high recall requirements; SDN / sanctions-list matching is downstream.
- **Trading-communication surveillance.** Detect language patterns associated with market abuse (spoofing, insider trading, front-running); high false-positive tolerance is not an option because every alert is human-reviewed.

### The canonical stacks

- **FinBERT** (Yang et al. 2020, and earlier Araci variants) — the domain-adapted encoder baseline.
- **BloombergGPT** (Wu et al., ["BloombergGPT: A Large Language Model for Finance"](https://arxiv.org/abs/2303.17564), 2023) — a closed decoder-only model trained on financial data; interesting as a case study in domain-scale pretraining even if the model is not distributed.
- **Public filings via EDGAR.** SEC EDGAR is a free source of US public-company filings; the [EDGAR full-text search](https://efts.sec.gov/LATEST/search-index?q=&dateRange=custom) plus per-filing XBRL exports are the substrate for most public-filings NLP. EU equivalents include ESMA registers and each national regulator's filings.
- **Bloomberg Terminal, Refinitiv, S&P Capital IQ, FactSet** — commercial data providers whose terms usually restrict redistribution and downstream model training.
- **Financial-services vendor NLP.** Behavox, Digital Reasoning (surveillance), Kira Systems' financial variant, Ayasdi — the vendor list is long and the compliance-officer relationships matter more than model benchmarks.

### The governance envelope

- **Regulator oversight of surveillance.** FINRA in the US, FCA in the UK, ESMA / BaFin in the EU, MAS in Singapore, ASIC in Australia. Surveillance systems are supervisory obligations; false negatives are regulatory findings; false positives are review-cost overhead. Any tuning that reduces recall must be documented and defensible.
- **Insider trading and market-abuse regulation.** MAR (EU Market Abuse Regulation), SEC Rule 10b-5 and Regulation FD, FCA equivalents. Trading-communication surveillance operates in this legal context; model output is potential evidence.
- **KYC / AML: FinCEN (US), FATF recommendations, EU AMLD.** Sanctions screening against OFAC (US), UN, EU consolidated lists; false negatives can be criminal.
- **Fair-lending / disparate-impact analytics.** ECOA (US), FCA fair-treatment expectations. Anywhere NLP feeds into a credit / underwriting decision, the anti-discrimination framework applies; explainability of the NLP contribution matters.
- **MNPI (material non-public information).** Analyst-side systems that ingest research must not leak MNPI across information barriers. Data-flow controls determine what an NLP system can see and where its outputs can go.
- **Data licensing.** Bloomberg, Refinitiv, S&P, Moody's, FactSet — content is licensed for specific uses. Training a model on the licensed content and redistributing the trained model is almost never allowed. Read the terms.

The NLP-engineer-side takeaways:

- **Recall matters more than precision on surveillance.** Systems are tuned to over-flag and let human reviewers filter; changing that trade-off requires documented compliance approval.
- **Filings are public; providers' aggregated data is not.** EDGAR is fair game; the pretty-formatted-and-aggregated version from a paid provider is not.
- **Every prediction is potentially evidence.** Audit trail (which model, which version, which input, which output, which timestamp) is non-negotiable. Chapter 5's model card and mod-112 chapter 5's manifest discipline both feed into this.

## Scientific NLP: literature mining and structured information

### The canonical tasks

Scientific NLP works on **peer-reviewed articles** (PubMed / PMC, arXiv, biorxiv, chemrxiv), **preprints**, **grant applications**, **patents** (overlap with legal), **clinical-trial protocols**, and **scientific databases** (UniProt, PubChem, DrugBank, ChEMBL).

- **Biomedical NER and normalisation.** Genes, proteins, diseases, chemicals, cell lines, species — extracted as spans and linked to canonical IDs (UMLS CUIs, NCBI Gene, MeSH, UniProt accessions, ChEBI IDs). The **BC5CDR** (chemicals + diseases), **NCBI Disease**, **JNLPBA** (bio entities), and **BioCreative** shared tasks are the reference corpora.
- **Relation and event extraction.** Chemical-protein interaction, gene-disease association, adverse-drug-reaction, drug-drug interaction. BioCreative and the DDIExtraction corpora anchor the benchmarks (mod-104 chapter 3).
- **Systematic-review support and evidence synthesis.** Screen abstracts for eligibility, extract PICO (Population, Intervention, Comparator, Outcome) elements, summarise across studies. Cochrane's [EPPI-Reviewer and their guidance on machine learning for evidence synthesis](https://eppi.ioe.ac.uk/CMS/) is the practitioner reference.
- **Citation graph analytics.** Semantic Scholar's Open Research Corpus, OpenAlex, Crossref — programmatic citation-graph exploration; downstream to influence mapping, novelty detection, and reproducibility triage.
- **Chemistry- and materials-NLP.** SMILES normalisation, reaction extraction, materials-property extraction. Overlaps with dedicated chemistry ML but often begins as an IE problem on text.

### The canonical stacks

- **scispaCy** (Neumann et al., ["ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"](https://arxiv.org/abs/1902.07669), *BioNLP 2019*) — the spaCy-family pipeline for scientific text; ships models trained on PubMed corpora with UMLS entity linkers.
- **BioBERT, PubMedBERT, SciBERT, BioMedBERT, BioGPT** — domain-adapted encoder and decoder models.
- **Semantic Scholar API, PubMed E-utilities, Crossref, OpenAlex, PMC Open Access subset** — programmatic access to the scientific literature substrate.
- **OGER, Metamap, cTAKES** — legacy pipelines still in wide use for concept normalisation against UMLS.

### The governance envelope

Scientific NLP is the *least* regulated of the four but has its own distinct constraints:

- **Publisher licensing.** Full-text access differs from abstract access. PMC Open Access, arXiv, and biorxiv are broadly redistributable; Elsevier, Springer, Wiley, and other subscription publishers restrict what can be done with their content. Corpus-scale ingestion for training requires explicit licensing or falls under fair-use / research-exception arguments that vary by jurisdiction.
- **Bulk API limits.** PubMed E-utilities has rate limits; Semantic Scholar has explicit terms of use; Crossref has its own usage policies. Respect the terms; ignore them and the API cuts you off.
- **Research-integrity signals.** Retracted papers, corrigenda, and disputed findings need to be flagged in the pipeline; a scientific NLP system that surfaces retracted findings as canonical evidence is worse than one that finds nothing. Cite Retraction Watch and the Retraction Watch Database ([retractiondatabase.org](http://retractiondatabase.org/)) as data sources.
- **PII in scientific text.** Case reports and older articles sometimes contain patient details that should have been redacted. Downstream applications that surface such content have a privacy exposure even though the source is a "published paper."
- **Reproducibility norms.** Scientific pipelines that produce claims should be reproducible from the corpus; the community expects deposited code, data cards, and version pinning.

The NLP-engineer-side takeaways:

- **The corpus is diverse and the licences vary within it.** A system that hits PMC OA is legally distinct from a system that hits Elsevier.
- **Concept-normalisation IDs are the interoperability contract.** Genes-as-strings do not compose across papers; genes as NCBI Gene IDs do.
- **The Retraction Watch check is a Table-1 quality gate**, not an afterthought.

## Cross-cutting patterns

Some patterns recur across all four domains:

- **Domain-adapted pretraining beats generic pretraining at the model class.** PubMedBERT, LegalBERT, FinBERT, SciBERT all outperform their vanilla-BERT counterparts on their domain benchmarks. The pattern generalises: for a task in a specialised domain, a domain-adapted base is a cheaper win than a larger generic base.
- **Managed APIs need a compliance-approved instance.** HIPAA-eligible AWS Bedrock, Azure Government, Google Vertex regulated-industry variants — the offering exists but requires paperwork. The "just use OpenAI" default of general NLP does not survive a regulated-industry data-flow review.
- **The label taxonomy is the product.** In clinical (ICD / SNOMED), legal (CUAD-style clause set), financial (event taxonomy), and scientific (UMLS / MeSH / UniProt) domains, the codified taxonomy is worth more than the model. Investing in taxonomy quality outperforms investing in bigger models.
- **Audit trails are not optional.** Every regulated domain requires that predictions are attributable to (artifact, input, timestamp). mod-112 chapter 5's manifest discipline is a hard requirement in these domains, not a nice-to-have.
- **RAG with grounded citations is the safe default for generative use.** Hallucination-with-authority (fake case cites, fake ICD codes, fake tickers, fake gene names) is the specific failure mode these domains cannot tolerate.

## Anti-patterns

Three worth naming so the exercise 4 survey highlights them:

- **The "we'll just use ChatGPT" prototype on regulated data.** Data flows to a non-BAA endpoint; the compliance officer discovers it after launch; the incident is real. Fix: no prototyping on regulated data with unapproved endpoints, even in a sandbox.
- **The un-normalised entity system.** NER extracts "IBM" and "International Business Machines" and "IBM Corp" as three separate entities; downstream analytics double-count. Fix: linking to canonical IDs is a Table-1 requirement in every domain.
- **The evaluation on the public benchmark that shipped years ago.** LexGLUE / CUAD / n2c2 / BioCreative are useful for research; the production system's real distribution is its own, and the model needs an in-domain held-out set.

## Chapter summary

- Every domain has its own canonical tasks, canonical models, canonical benchmarks, and canonical governance envelope. Learn the vocabulary before designing the system.
- **Clinical NLP** = de-identification + medical coding on regulated data. HIPAA + GDPR + institutional approvals; managed APIs need BAAs; MIMIC / n2c2 / i2b2 are DUA-gated; SaMD / MDR classification may apply.
- **Legal NLP** = clause extraction, judgment prediction, contract review, privilege detection. Attorney–client privilege, ABA professional-responsibility opinions; hallucinated citations are a real professional-liability incident; CUAD / LexGLUE are the reference benchmarks.
- **Financial NLP** = event extraction, sentiment, disclosure analytics, surveillance. FINRA / FCA / SEC / MAR / OFAC; recall dominates on surveillance; filings are public but provider aggregations are licensed; audit trails are evidentiary.
- **Scientific NLP** = literature mining, biomedical NER, evidence synthesis, citation graph. Publisher licences vary; API rate limits are real; Retraction Watch is a quality gate; UMLS / MeSH / UniProt IDs are the interoperability contract.
- Cross-cutting: domain-adapted pretraining wins; managed APIs require compliance-approved instances; the taxonomy is the product; audit trails are non-negotiable; RAG-with-grounded-citations is the safe default for generative use.
- The chapter's artifact is a **domain patterns and governance survey** — one page per candidate domain summarising the canonical tasks, stacks, benchmarks, licensing constraints, and regulatory envelope. Exercise 4 walks the survey against a chosen domain.
