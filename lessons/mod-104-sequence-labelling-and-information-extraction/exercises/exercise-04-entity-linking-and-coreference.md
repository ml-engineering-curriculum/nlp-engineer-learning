# exercise-04: Entity Linking and Coreference

**Estimated effort:** 3 hours

## Objective

Ground extracted mentions to a real knowledge base and cluster mentions that refer to the same real-world entity across a document. You will build a candidate-generation + ranking + NIL-detection linker, evaluate it under both gold-mention and end-to-end conditions, then add a coreference pass and measure how much it lifts document-level entity recall. The result is the connective tissue between raw NER output and a knowledge graph.

## Prerequisites

- Exercise-01 (NER model). Ideally exercise-02 (document-level tagging).
- Chapters [11](../11-entity-linking-against-a-knowledge-base.md), [12](../12-coreference-resolution-at-document-scale.md).
- Python 3.10+; `transformers`, `datasets`, `sentence-transformers` or `faiss-cpu`.
- Depending on the domain: `scispacy` (biomedical), `blink` (Wikipedia), or a local KB dump.

## Datasets and knowledge bases

Pick **one dataset + KB pairing** and use it throughout.

### Wikipedia / Wikidata

- **AIDA-CoNLL** (Hoffart et al., ["Robust Disambiguation of Named Entities in Text"](https://aclanthology.org/D11-1072/), *EMNLP 2011*). Standard newswire EL benchmark; Wikipedia-linked. Gold mentions and gold links.
- **KILT** (Petroni et al., ["KILT: a Benchmark for Knowledge Intensive Language Tasks"](https://aclanthology.org/2021.naacl-main.200/), *NAACL 2021*). Multi-task Wikipedia benchmark; includes AIDA and other EL subsets.
- **KB:** a Wikipedia dump filtered to entities with English articles, or the pre-processed BLINK KB from <https://github.com/facebookresearch/BLINK>.

### Biomedical

- **MedMentions** (Mohan & Li, ["MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts"](https://arxiv.org/abs/1902.09476), *AKBC 2019*). PubMed abstracts linked to UMLS CUIs.
- **BC5CDR** (Li et al., ["BioCreative V CDR task corpus"](https://doi.org/10.1093/database/baw068), *Database 2016*). Chemical + disease mentions linked to MeSH.
- **KB:** UMLS Metathesaurus (requires a free UMLS licence from NLM) or scispaCy's pre-packaged entity linker index (<https://allenai.github.io/scispacy/>).

### Coreference (used in Part D — same document collection as above works)

- **OntoNotes 5.0 CoNLL-2012** (Pradhan et al., ["CoNLL-2012 Shared Task"](https://aclanthology.org/W12-4501/), *EMNLP-CoNLL 2012*). Standard coref benchmark.
- **PreCo** (Chen et al., ["PreCo: A Large-scale Dataset in Preschool Vocabulary for Coreference Resolution"](https://arxiv.org/abs/1810.09807), *EMNLP 2018*). Open-access alternative.
- **GAP** (Webster et al., ["Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns"](https://arxiv.org/abs/1810.05201), *TACL 2018*). Bias-focused subset.

You may reuse the document-level corpus from exercise-02 (SciERC, MedMentions, CDR) if it has coreference annotations, and evaluate only the two-tasks-that-are-annotated.

## Problem statement

### Part A — Alias-index candidate generation

Build a surface-form → KB-id index from your KB. Populate it with:

- Canonical entity names.
- All aliases from the KB (`skos:altLabel`, `alsoKnownAs`, MeSH `synonyms`, UMLS `str` values, Wikipedia anchor text, redirect pages).
- Lowercased and Unicode-normalised (NFKD) variants.

For each gold mention in the dev split, look up its normalised surface form. Report:

- **Alias-index recall@∞** — fraction of mentions whose gold KB-id appears anywhere in the returned candidate list.
- **Average candidate-list length** per mention.

Alias recall < 90 % means your index is missing common name variants — extend it with character-n-gram or edit-distance backoff before moving on.

### Part B — Dense-retrieval candidate generation

Build a bi-encoder candidate retriever:

- Encode each KB entity's description (first paragraph of its Wikipedia article; UMLS entity `definition` field) with a sentence-transformer (start with `sentence-transformers/all-MiniLM-L6-v2` for speed; upgrade to `BAAI/bge-small-en-v1.5` or `intfloat/e5-base` if you have time). Index in FAISS.
- Encode each mention with its context (mention + ±64 tokens on each side) with the same encoder.
- Retrieve the top-100 KB entities by cosine similarity.

Report:

- **Bi-encoder recall@1, recall@10, recall@100** — fraction of mentions whose gold KB-id is in the top-`k`.
- **Alias ∪ dense recall@100** — the combined shortlist your ranker will see.

The union should push recall above 95 % on the dev split. If not, either your KB is missing the entity (a genuine NIL) or your encoder is misaligned to the domain (try SapBERT (Liu et al., ["Self-Alignment Pretraining for Biomedical Entity Representations"](https://arxiv.org/abs/2010.11784), *NAACL 2021*) for biomedical text).

### Part C — Cross-encoder ranker + NIL detection

Fine-tune (or evaluate zero-shot) a cross-encoder ranker on the alias ∪ dense shortlist:

- Input per candidate: `[CLS] mention_left_context [MENTION] mention [/MENTION] mention_right_context [SEP] candidate_name : candidate_description [SEP]`.
- Output: a single scalar score per candidate.
- Loss (if fine-tuning): margin ranking loss with 1 positive + `k=15` in-batch negatives; the BLINK training recipe (Wu et al., ["Scalable Zero-shot Entity Linking with Dense Entity Retrieval"](https://aclanthology.org/2020.emnlp-main.519/), *EMNLP 2020*) is the reference.
- If you cannot fine-tune, evaluate an off-the-shelf cross-encoder such as `cross-encoder/nli-deberta-v3-base` as a zero-shot ranker — worse than a fine-tuned one, but usable.

**NIL detection.** Threshold the top-1 score. Choose the threshold on the dev split by maximising in-KB micro-F1 + NIL F1 (report the value). A calibrated threshold matters — see chapter 07 of mod-103.

Report on the test split:

- **In-KB accuracy** over mentions whose gold link is in the KB.
- **NIL precision, recall, F1** — separately.
- **Overall accuracy** (correct = right KB-id OR both predicted and gold are NIL).

### Part D — End-to-end EL

Run your NER model (exercise-01) on the test split, then feed its predicted mentions into the linker instead of gold mentions.

Report:

- End-to-end EL accuracy.
- **Gold-vs-end-to-end gap** — analogous to exercise-03. This is your NER's contribution to EL loss.
- Break down the gap: what fraction is boundary drift (span off by ± 1 token), what fraction is type-filter mismatch, what fraction is a missed mention?

### Part E — Coreference-augmented EL

Add a coreference pass with `fastcoref` (Otmazgin et al., ["F-COREF: Fast, Accurate and Easy to Use Coreference Resolution"](https://arxiv.org/abs/2209.04280), *AACL 2022*) or `maverick-coref` (Martinelli et al., ["Maverick: Efficient and Accurate Coreference Resolution Defying Recent Trends"](https://arxiv.org/abs/2407.21489), *ACL 2024*).

For each document:

1. Run coref to cluster mentions.
2. For each cluster, link the **longest / most-informative mention** once (the "canonical mention" — usually the first proper-noun mention).
3. Propagate that KB-id to every other mention in the cluster (including pronouns).

Report:

- End-to-end EL accuracy with coref propagation.
- Delta vs. Part D — how much did coref buy you?
- Break down the delta by mention type: pronouns, definite noun phrases, acronyms.

### Part F — Coreference metrics on the gold-annotated portion

If your dataset has coref annotations (OntoNotes, PreCo, SciERC), also evaluate your coref model directly:

- Compute **MUC-F1, B³-F1, CEAFφ4-F1** with `coval` (<https://github.com/ns-moosavi/coval>) or `perl scorer.pl` from the CoNLL-2012 release.
- Report the **CoNLL average** (arithmetic mean of the three).

If your dataset does not have coref annotations, sample 20 documents and hand-annotate 3–5 clusters per document as a spot-check. Report per-mention linkage accuracy on the spot-check.

### Part G — Bias probe

Run WinoBias (Zhao et al., ["Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods"](https://arxiv.org/abs/1804.06876), *NAACL 2018*) or WinoGender (Rudinger et al., ["Gender Bias in Coreference Resolution"](https://aclanthology.org/N18-2002/), *NAACL 2018*) on your coref model. Report:

- **Type-1 (pro-stereotype) accuracy** vs. **anti-stereotype accuracy**.
- The accuracy gap. A gap of 10+ points is a shipping blocker in a product where gendered-pronoun resolution matters (HR tools, biography processing, chatbots).

### Part H — Write-up

A 500–700 word `README.md` covering:

- Chosen dataset + KB.
- Alias-vs-dense-vs-union recall table.
- In-KB / NIL / overall accuracy table.
- End-to-end EL number and the gold-vs-end-to-end gap analysis.
- Coref F1 (CoNLL average) and the EL lift from coref propagation.
- WinoBias / WinoGender gap.
- One production concern (e.g., "the KB updates weekly; how do I keep the alias index fresh?" or "our documents are 30 pages long; coref VRAM is a blocker").

## Starter guidance

- **Normalise before both indexing and lookup.** Lowercasing + Unicode NFKD + strip diacritics — apply the same function to both sides. Case-sensitivity bugs in the alias index are the most common EL failure mode.
- **Store character offsets, not token offsets.** Coref clusters that anchor on token offsets will drift when you retokenise for the ranker.
- **Do not fine-tune the KB encoder if you re-encode the KB.** Re-encoding 6M Wikipedia entities takes hours. If you must retrain, use `sentence-transformers` `fit()` with a hard-negative mining schedule and cache embeddings.
- **NIL calibration.** Do not pick the NIL threshold on the test split. Pick it on dev; report test numbers at the dev-optimal threshold.
- **Coref singletons.** Neural coref emits many size-1 "clusters." Filter to size ≥ 2 before propagating KB-ids or you multiply your inference cost for no gain.
- **Cross-document consistency.** If your test set has the same mention string appearing in different documents, the ranker may link it to different KB-ids per document. If your product needs cross-document consistency, aggregate mentions first, link once, propagate — this is chapter 12's cross-document coref idea applied to EL.

## Acceptance criteria

- [ ] Alias index built; recall@∞ and average candidate-list length reported.
- [ ] Dense-retrieval index built; recall@1 / @10 / @100 and alias ∪ dense recall reported.
- [ ] Cross-encoder ranker (fine-tuned or zero-shot) evaluated; in-KB / NIL / overall accuracy reported at a dev-optimal threshold.
- [ ] End-to-end EL (Part D) run through the NER pipeline; gold-vs-end-to-end gap decomposed by error type.
- [ ] Coref-augmented EL (Part E) evaluated; delta broken down by mention type.
- [ ] Coref F1 reported on the gold-annotated portion — MUC / B³ / CEAF and the CoNLL average. (Or a spot-check if no coref gold exists.)
- [ ] WinoBias / WinoGender gap reported for the coref model.
- [ ] 500–700 word write-up.

## Stretch goals

- **GENRE / autoregressive linking.** Run `facebook/genre-linking-aidayago2` (De Cao et al., ["Autoregressive Entity Retrieval"](https://arxiv.org/abs/2010.00904), *ICLR 2021*) on AIDA and compare to your bi-encoder + cross-encoder pipeline.
- **LLM-as-linker.** For each mention, prompt a strong LLM with the mention, its context, and the top-K candidate descriptions; ask which is correct. Compare accuracy and cost.
- **Cross-lingual EL.** Use mGENRE (De Cao et al., ["Multilingual Autoregressive Entity Linking"](https://arxiv.org/abs/2103.12528), *TACL 2022*) or a multilingual bi-encoder on Mewsli-9 (<https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/mel>) and report per-language accuracy.
- **KB freshness.** Add 100 net-new entities to your KB (from a Wikidata weekly dump); re-run the alias index build without retraining the encoder; measure recall on those specific entities. This tests the "KB drift" failure mode from chapter 11.
- **Bag-of-Concepts F1.** For a biomedical corpus, compute document-level F1 over the *set* of unique CUIs mentioned. Compare against per-mention accuracy.
- **Cross-document coref.** Extend the coref pass across all documents in the test split (Cattan et al., ["Streamlining Cross-Document Coreference Resolution: Evaluation and Modeling"](https://arxiv.org/abs/2009.11032), *arXiv 2020*). Report the cross-doc linkage consistency lift.
