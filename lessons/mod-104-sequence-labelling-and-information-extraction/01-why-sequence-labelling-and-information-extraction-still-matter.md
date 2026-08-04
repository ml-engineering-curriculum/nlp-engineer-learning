# Why Sequence Labelling and Information Extraction Still Matter

## Motivation

Classification (mod-103) told you *what a document is about*. Information extraction (IE) asks the harder, more useful question: *what specific things are in this document, and how do they relate?* Every downstream product that has to reason about text — a search index, a knowledge graph, a compliance monitor, an EHR autocoder, a legal-diligence pipeline, an LLM tool-use system that needs structured arguments — needs someone to have first pulled the entities, relations, events, and coreference chains out of the raw text.

That "someone" is you, and the workhorse job is called sequence labelling or, more broadly, information extraction. This module builds the modern IE stack:

1. **Named-entity recognition (NER)** — mark every span of text that names a person, organisation, location, drug, gene, financial instrument, contract clause, whatever your ontology says.
2. **Nested and document-level NER** — handle entities inside entities (*Bank of America Merrill Lynch* contains an ORG that contains a LOC-style ORG) and entities that cross paragraphs.
3. **Relation extraction (RE)** — given a set of entities, decide which pairs stand in each ontology-relation (`works_at`, `interacts_with`, `subsidiary_of`).
4. **Event extraction (EE)** — trigger detection plus argument role labelling (WHO did WHAT to WHOM, WHEN, WHERE).
5. **Slot filling** — a dialogue-flavoured cousin of RE/EE, where a user utterance yields typed values for a fixed schema.
6. **Entity linking (EL)** — the extracted spans are grounded to a knowledge-base identifier (Wikidata Q-id, UMLS CUI, GLEIF LEI).
7. **Coreference resolution** — mentions that refer to the same real-world entity are clustered together, including pronouns.
8. **Structured generation from decoder LLMs** — a rising alternative to (1)–(6): prompt an LLM to emit JSON conforming to a schema, and skip the tagger entirely.

The academic literature draws hard lines between these tasks; a real pipeline usually stitches at least (1), (6), (7), and one of (3)/(4)/(5)/(8) together. This module treats them as one ladder — each rung feeds the next — and teaches the design decisions honest engineers argue about when they build it.

## The IE ladder — and the two ways to climb it

Two paradigms coexist in 2026, and neither has retired the other:

**Path A — tagger-first.** Encode the text with a transformer, put a token-classification head on top, decode spans, then run RE/EL/coref as downstream models. Deterministic latency, calibratable probabilities, small footprint, easy to evaluate span-by-span. This is what production IE at scale still looks like at Bloomberg, LinkedIn, Google Search snippets, most EHR pipelines, and every regulated domain where auditors need to see the score for span `[142:178]`.

**Path B — decoder-extraction.** Prompt a decoder LLM with a schema and let it emit JSON that captures every entity, relation, and event in one shot. Flexible, zero-shot to new schemas, handles nested and document-level structure without special heads — but slower, harder to calibrate, harder to evaluate at the span level, and dependent on constrained-decoding tooling to guarantee schema validity. Chapters 09 and 10 will show when this beats Path A and when it does not.

The two paths are complementary. The current industry pattern is:

- **Path A for the hot path.** Millisecond latency, deterministic recall, monitored per-type F1.
- **Path B for the tail.** New entity types, low-resource languages, one-off extraction jobs, rare schemas where nobody wants to hand-label 5 000 documents.
- **Path B to bootstrap Path A.** LLM emits proposed spans; humans review; a lightweight tagger is distilled from the reviewed set. Weak supervision (chapter 13) formalises this.

## Why this is harder than classification

Classification gives you one label per document; IE gives you a *variable-sized structured output*. That changes almost every design decision:

- **Evaluation is not accuracy.** Token-level accuracy on a BIO-tagged corpus is dominated by `O` and lies about performance. You need *entity-level* precision, recall, and F1 — a span counts only when both its boundary and its type match. Chapter 02 walks the `seqeval` toolkit and why the CoNLL evaluation script has been the standard for two decades.
- **Alignment matters.** Your annotations are on words (or characters); your model consumes subword tokens. Every training example needs a label-alignment step, and the four common recipes give measurably different F1. Chapter 03 is entirely about this.
- **The decoder has structure.** A raw softmax per token can emit `B-PER O I-PER` — an invalid tag sequence. CRFs, constrained Viterbi, and post-processing rules all exist because token-independent softmax leaks. Chapter 05 covers when the CRF still pays for its extra parameters.
- **The label space is graph-shaped, not flat.** Entities have a type hierarchy, relations are typed pairs of entities, events are typed triples-plus. You end up defending an *ontology*, not just a class list.
- **Real domains are entity-scarce.** In clinical, legal, and financial NER, gold spans cost $5–$50 apiece; you never have enough. Active learning (chapter 13) and weak supervision (chapter 13) are not nice-to-haves, they are what makes the project financeable.

## Where IE shows up in production, concretely

- **Search relevance and answer boxes.** Google's "featured snippet" pipeline, Bing entity cards, and every enterprise-search "answers" feature run entity taggers over the query and the corpus and match them.
- **Clinical documentation.** UMLS-linked NER over discharge summaries powers billing autocoders, adverse-event surveillance, and cohort selection for trial matching. See Stanford's Stanza biomedical models and cTAKES for the reference stacks.
- **Contract review.** LexNLP, Kira, and the diligence-tooling ecosystem tag clauses, parties, dates, dollar amounts, governing-law jurisdictions, then normalise those to a schema for reviewers.
- **Financial news + trade surveillance.** Ticker/company recognition, event extraction ("acquired", "announced dividend", "filed 8-K"), and entity linking to CIK / LEI power surveillance queues and event-driven trading feeds.
- **Product knowledge graphs.** Amazon, eBay, and every marketplace extract attributes ("brand: Sony, capacity: 128GB") from titles and descriptions into a schema for facet search.
- **Conversational assistants.** Alexa, Google Assistant, and enterprise chatbots run intent + slot models (a form of NER + classification) on every utterance. See FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual NLU Dataset"](https://arxiv.org/abs/2204.08582), *ACL 2023*, for the modern benchmark.
- **LLM tool-use.** When an agent calls `weather(city="Paris", date="2026-08-05")`, something had to extract `city` and `date` from a user turn. Structured extraction (chapter 09) is the same task under a new name.

None of these have been retired by "just call an LLM." They have been *augmented* by LLMs — usually as the labelling assistant in the loop.

## What we will not cover

- **Part-of-speech tagging, chunking, dependency parsing.** These are sequence-labelling tasks in the same family, but modern pipelines depend on off-the-shelf parsers (spaCy, Stanza, UDPipe) rather than training them per project. See mod-102 for the classical NLP grounding.
- **Semantic role labelling as a research task.** Chapter 09 treats SRL-flavoured event extraction; it does not survey the PropBank/FrameNet ecosystem.
- **Table extraction and layout-aware IE.** Real invoices, forms, and PDFs need vision-language models (LayoutLMv3, Donut, Textract). That is a mod-112 topic; here we assume text is already extracted.
- **Fine-grained IE evaluation ontologies.** OntoNotes types vs. FIGER vs. ACE are choices you inherit from a dataset; we spend more time on evaluation *methodology* than on choosing between competing type systems.

## Reading order for this module

- Chapter **02** builds the tagging-scheme vocabulary (IO / BIO / BIOES / BILOU) and the entity-level F1 you will report for the rest of the module.
- Chapter **03** covers the subword-to-word alignment tax that every transformer NER pays.
- Chapter **04** is the workhorse recipe: encoder + linear head, trained with `Trainer`, evaluated with `seqeval`.
- Chapter **05** adds structured decoders (CRF, constrained Viterbi) and answers the "does the CRF still help?" question honestly.
- Chapter **06** covers span-based NER, the current default for nested / discontinuous entities.
- Chapter **07** covers document-level and long-context IE — sliding windows, doc-level context, memory-augmented encoders.
- Chapter **08** builds relation extraction — pipeline, joint, marker-based.
- Chapter **09** covers event extraction and slot filling as one family.
- Chapter **10** introduces decoder-LLM structured extraction with schema validation and constrained decoding.
- Chapter **11** covers entity linking against a knowledge base (candidate generation, disambiguation, NIL detection).
- Chapter **12** covers coreference resolution at document scale.
- Chapter **13** closes with active learning and weak supervision — the two levers that make domain IE work when gold data is scarce.

By the end you should be able to walk into a new IE project, define its schema, pick the right split of Path A vs. Path B for each sub-task, evaluate honestly, and defend the design to a reviewer.

## Chapter summary

- Information extraction turns unstructured text into structured records — the layer between raw text and any downstream system that needs typed values.
- The task family spans NER, relation extraction, event extraction, slot filling, entity linking, and coreference resolution. Real pipelines stitch several together.
- Two paradigms coexist: tagger-first (Path A) for hot-path production and decoder-extraction (Path B) for flexibility, exploration, and low-resource cases.
- IE is harder to evaluate than classification (entity-level F1, not accuracy) and harder to preprocess (subword alignment is a real tax) — the next two chapters set up both.
- Domain IE (clinical, legal, financial) is data-scarce; active learning and weak supervision are load-bearing, not optional.
