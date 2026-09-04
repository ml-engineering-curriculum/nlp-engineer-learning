# mod-104 · Sequence Labelling and Information Extraction

Named-entity recognition, relation and event extraction, entity linking, coreference, and schema-driven structured extraction — the layer between raw text and any downstream system that needs typed records.

**Estimated effort:** 16 hours

## Learning objectives

- Train and evaluate NER with BIO / BIOES / BILOU tagging schemes, subword-to-word alignment, and entity-level F1 (seqeval).
- Compare span-based, CRF-headed, and decoder-extraction models for flat, nested, and document-level entity tasks.
- Build relation extraction, event extraction, and slot-filling models — including schema-driven structured-generation alternatives.
- Tackle entity linking against a knowledge base and coreference resolution at document scale.
- Operate active-learning and weak-supervision loops appropriate to domain-NER data scarcity (clinical, legal, finance).

## Chapters

1. [Why sequence labelling and information extraction still matter](01-why-sequence-labelling-and-information-extraction-still-matter.md)
2. [Tagging schemes and entity-level evaluation with seqeval](02-tagging-schemes-and-entity-level-evaluation-with-seqeval.md)
3. [Subword-to-word alignment for transformer NER](03-subword-to-word-alignment-for-transformer-ner.md)
4. [Token-classification NER: the workhorse recipe](04-token-classification-ner-the-workhorse-recipe.md)
5. [CRF heads and structured decoders](05-crf-heads-and-structured-decoders.md)
6. [Span-based NER for flat, nested, and discontinuous entities](06-span-based-ner-for-flat-nested-and-discontinuous-entities.md)
7. [Document-level and long-context NER](07-document-level-and-long-context-ner.md)
8. [Relation extraction: pipeline, joint, and marker-based](08-relation-extraction-pipeline-joint-and-marker-based.md)
9. [Event extraction and slot filling](09-event-extraction-and-slot-filling.md)
10. [Schema-driven structured extraction with decoder LLMs](10-schema-driven-structured-extraction-with-decoder-llms.md)
11. [Entity linking against a knowledge base](11-entity-linking-against-a-knowledge-base.md)
12. [Coreference resolution at document scale](12-coreference-resolution-at-document-scale.md)
13. [Active learning and weak supervision for domain IE](13-active-learning-and-weak-supervision-for-domain-ie.md)

## Exercises

- [exercise-01 · NER with subword alignment and seqeval](exercises/exercise-01-ner-with-subword-alignment-and-seqeval.md)
- [exercise-02 · Nested and document-level NER](exercises/exercise-02-nested-and-document-level-ner.md)
- [exercise-03 · Relation and event extraction](exercises/exercise-03-relation-and-event-extraction.md)
- [exercise-04 · Entity linking and coreference](exercises/exercise-04-entity-linking-and-coreference.md)
- [exercise-05 · Schema-driven structured extraction with LLMs](exercises/exercise-05-schema-driven-structured-extraction-with-llms.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
