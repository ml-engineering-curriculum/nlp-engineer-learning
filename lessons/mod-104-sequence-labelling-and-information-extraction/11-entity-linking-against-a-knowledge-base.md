# Entity Linking Against a Knowledge Base

## Motivation

NER (chapters 04–06) tells you *"positions 12–18 name a person."* Entity linking (EL) tells you *which* person: `Q76` (Barack Obama, 44th US President) versus `Q22686` (Barack Obama Sr., his father). EL is what closes the loop between text and a knowledge base — Wikidata, UMLS (Unified Medical Language System), MeSH, ChEMBL, GLEIF, an internal product catalogue — turning a span into a KB identifier that downstream systems can join on.

Without EL, an NER pipeline produces highlights. With EL, it populates a knowledge graph, powers deduplication, drives faceted search, and enables cross-document reasoning ("did this drug show up in any 2024 papers, under any of its 12 synonyms?").

## The task, precisely

Given a text span `mention` (usually the output of a NER model) plus its context, output a KB identifier `k ∈ KB ∪ {NIL}`. `NIL` is the "no such entity in the KB" answer — a real thing (novel drug, misnamed company, private person) that must be recognised as unlinkable, not force-mapped to the wrong ID.

Two evaluation modes:

- **In-KB accuracy** — over mentions whose gold answer is in the KB, fraction correctly linked.
- **NIL detection F1** — precision and recall on identifying unlinkable mentions.

Report both. A system that hits 90 % in-KB accuracy while linking every NIL mention to the wrong ID is a bad EL system that looks good on the wrong metric.

## The three-stage pipeline

Every mainstream EL system decomposes into:

1. **Candidate generation** — from the KB's millions of entities, produce a shortlist of 5–100 candidates for this mention.
2. **Candidate ranking** — score each candidate against the mention's context; return top-1.
3. **NIL detection** — decide whether the top-1 candidate is confidently the right entity or the mention is unlinkable.

Each stage has its own optimisation regime; the pipeline is dominated by the stage with the loosest performance.

## Stage 1: candidate generation

The engineering-heavy stage. Two indexes are essential:

- **Alias index (surface-form dictionary).** Maps a normalised surface string to KB IDs that have ever been referred to by that string. Populated from KB `alsoKnownAs` / `alias` / `label` fields; augmented with anchor text from Wikipedia hyperlinks, MeSH synonyms, UMLS `strings`.
- **Dense retrieval index.** Encode both mentions (with context) and entity descriptions (from KB glosses) with a bi-encoder; nearest-neighbour lookup in a FAISS index.

Modern systems use both. The alias index guarantees recall on common name matches; the dense index catches paraphrases and abbreviations.

References:

- **BLINK** — Wu, Petroni, Josifoski, Riedel & Zettlemoyer, ["Scalable Zero-shot Entity Linking with Dense Entity Retrieval"](https://aclanthology.org/2020.emnlp-main.519/), *EMNLP 2020*. Bi-encoder for candidate generation + cross-encoder for ranking. Open weights on Wikipedia; the canonical modern baseline.
- **GENRE** — De Cao, Izacard, Riedel & Petroni, ["Autoregressive Entity Retrieval"](https://arxiv.org/abs/2010.00904), *ICLR 2021*. Generates the entity's canonical name character-by-character with constrained decoding over a KB trie. Elegant, effective on Wikipedia; harder to extend to KBs where the "canonical name" is not the natural sort key.
- **REL** — van Hulst et al., ["REL: An Entity Linker Standing on the Shoulders of Giants"](https://arxiv.org/abs/2006.01969), *SIGIR 2020*. Practical Wikipedia EL system with alias-based candidate generation.
- **scispaCy EntityLinker** — Neumann, King, Beltagy & Ammar, ["ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"](https://arxiv.org/abs/1902.07669), *BioNLP 2019*. UMLS-linked biomedical NER + EL. The default for biomedical projects.

## Stage 2: candidate ranking

Given ~50 candidates from stage 1, rank them by "which one does the mention's context actually refer to." Two dominant architectures:

### Cross-encoder ranker

For each candidate, encode `[CLS] mention_context [SEP] candidate_description [SEP]` with a transformer and score. Standard BLINK's stage-2 setup. Accurate but expensive: 50 candidates × cross-encoder forward passes per mention.

### Late-interaction ranker

Encode context and candidate separately (like a bi-encoder), but compute a token-level interaction score. Cheaper than cross-encoder, slightly less accurate. See ColBERT-style architectures adapted for EL.

### Type-aware ranking

If your NER model produces entity types, filter candidates to those with a matching KB type before ranking. A `PER`-tagged mention should not link to a `LOC` entity; the type filter cuts candidates 3–10× and reduces cross-encoder cost proportionally.

## Stage 3: NIL detection

Two common approaches:

### Threshold on top-1 score

If the top-1 candidate's score is below a threshold, emit `NIL`. Simple; requires a calibrated ranker (mod-103 chapter 07 covers calibration).

### Explicit NIL classifier

Train a binary classifier on the ranker's outputs (top-1 score, top-2 score, margin, mention type, mention length) that predicts NIL vs. link. More expressive; requires labelled NIL examples.

The classic NIL evaluation reference is the KBP-EDL shared tasks (2015–2017) — see Ji, Nothman & Hachey, ["Overview of TAC-KBP2015 Tri-lingual Entity Discovery and Linking"](https://tac.nist.gov/publications/2015/additional.papers/TAC2015.KBP_Tri-lingual_Entity_Discovery_and_Linking_overview.proceedings.pdf), *TAC 2015*, for the ontology.

## The zero-shot regime

Zero-shot EL (link to entities the ranker was not trained on — e.g., linking a 2026 organisation from a 2024-trained model) has become the dominant modern setting.

- **Bi-encoder + cross-encoder over KB descriptions** — the BLINK setup extends to zero-shot naturally: as long as new entities have descriptions in the KB, both the retrieval and ranking stages can score them.
- **GENRE-style constrained generation** — needs a fresh trie build over the KB but no retraining.
- **LLM-as-linker** — prompt an LLM with the mention, its context, and the top-K candidate descriptions; ask which is the right link (or NIL). Slower and more expensive, but strong on ambiguous / rare cases.

## Multilingual EL

Cross-lingual EL — link Spanish mentions to English Wikipedia, Chinese mentions to Wikidata — is a first-class task. XLM-R-based bi-encoders (multilingual BLINK, mGENRE) are the current baselines.

- **mGENRE** — De Cao et al., ["Multilingual Autoregressive Entity Linking"](https://arxiv.org/abs/2103.12528), *TACL 2022*. Multilingual GENRE.
- **XL-BEL** — Botha, Shan & Gillick, ["Entity Linking in 100 Languages"](https://arxiv.org/abs/2011.02690), *EMNLP 2020*. Google-scale multilingual EL.
- **Mewsli-9** — the standard evaluation benchmark for cross-lingual EL.

## Domain KBs

The KB choice dominates the whole system. Practical picks:

- **General knowledge:** Wikidata (Q-ids). Well-covered by open EL systems; the KB itself is under continuous public revision.
- **Biomedical:** UMLS Metathesaurus (CUIs). scispaCy is the reference stack; SapBERT (Liu et al., ["Self-Alignment Pretraining for Biomedical Entity Representations"](https://arxiv.org/abs/2010.11784), *NAACL 2021*) is a strong bi-encoder for candidate generation.
- **Chemistry:** ChEMBL, PubChem CIDs.
- **Genes/proteins:** UniProt accessions, HGNC symbols.
- **Legal entities:** GLEIF LEIs (Legal Entity Identifiers); CIK numbers for US public filers.
- **Product catalogues:** internal SKUs; usually needs an in-house index because the schema is proprietary.

Rule: use the smallest KB that covers your product. Wikidata is 100M+ entities; a domain KB restricted to your relevant slice (10K–1M entities) makes every stage of EL cheaper and more accurate.

## The end-to-end evaluation trap

If you evaluate EL on gold-mention inputs, you overestimate real performance. Two sources of error compound in production:

- **NER boundary errors.** Off-by-one span boundaries change the mention string, which changes the alias-index hit set, which changes the candidate list.
- **NER type errors.** A mention mis-typed as `LOC` when it should be `ORG` gets a wrong type filter applied.

Report **end-to-end EL** (NER + EL together) as the primary number when your product runs both.

## Evaluation metrics

- **Micro-accuracy over mentions** — the headline. Correct = top-1 candidate is the gold KB ID (or both are NIL).
- **In-KB precision, recall, F1.**
- **NIL precision, recall, F1** — separate from in-KB; a system that gets in-KB right but NILs wrong is a bad linker.
- **Bag-of-Concepts F1** (biomedical) — for document-level EL, compute F1 over the set of unique KB IDs mentioned in the document. Standard on MedMentions and BC5CDR.

## The library ecosystem

- **BLINK** — <https://github.com/facebookresearch/BLINK>. Wikipedia EL; the code base for the canonical modern baseline.
- **scispaCy `EntityLinker`** — <https://allenai.github.io/scispacy/>. UMLS EL.
- **GENRE** — <https://github.com/facebookresearch/GENRE>. Autoregressive EL.
- **spaCy `EntityLinker`** — pluggable pipeline component; usable with a custom KB.
- **REL** — <https://github.com/informagi/REL>. Wikipedia-only, mature.

## Failure modes

1. **Alias-index recall collapse on unseen surface forms.** Add character-n-gram, phonetic, and edit-distance backoff for common name variants.
2. **NIL class imbalance.** If your training data has 5 % NIL, your model will underpredict NIL. Oversample or use a NIL-focused loss.
3. **KB drift.** Your KB adds/removes entities weekly (Wikidata does); the ranker trained a month ago missed them. Retrain the alias index continuously and add new entities to the dense-retrieval index without retraining the encoder.
4. **Overconfident wrong links.** The top-1 score is high but the entity is wrong (common with acronyms — `AI` links to `Artificial Intelligence` when the context says `Amnesty International`). Log per-mention-type accuracy; watch acronyms specifically.
5. **Case sensitivity in alias index.** Normalise (lowercasing + Unicode NFKD) *before* both indexing and lookup.
6. **Coreference-blind linking.** *"Apple"* and *"the company"* both need to link to `Q312`. Run coreference (chapter 12) first, link the head of each cluster once, propagate.
7. **Cross-document consistency.** In a batch job over 10 000 documents, the same mention string may link differently in each. If your product needs cross-document consistency, aggregate mentions across documents before linking.

## Chapter summary

- Entity linking grounds a mention to a KB identifier; the KB choice (Wikidata, UMLS, GLEIF, product catalogue) dominates system design.
- The standard architecture is candidate generation (alias index + dense retrieval) → candidate ranking (cross-encoder or GENRE-style generation) → NIL detection.
- Report both in-KB accuracy and NIL F1; a system that hits one and misses the other is broken.
- BLINK, GENRE, and scispaCy are the canonical open references — study their code before implementing your own.
- End-to-end EL (NER + EL) is the number that matches production; gold-mention EL flatters the ranker unrealistically.
- Coreference-augmented and cross-document-consistent linking are the two biggest levers for improving realistic accuracy beyond isolated-mention scores.
