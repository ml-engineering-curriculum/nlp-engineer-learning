# mod-113 · NLP Systems Design, Domain Survey, and Responsible Release

The capstone module for the NLP-engineer track. The chapters walk the discipline of translating a product goal into an NLP-system blueprint, allocating budget across the five buckets of an NLP stack, designing release gates and rollback plans specific to NLP, surveying domain-NLP patterns (clinical, legal, financial, scientific) and their governance envelopes, and authoring the model-card / dataset-card / responsible-release-review artifact chain that a governance / risk / security partner can actually sign off on. Prior modules taught the components; this module ties them together into a shippable, defensible system-design practice.

**Estimated effort:** 14 hours

## Learning objectives

- Walk a real product goal to an NLP-system blueprint: task framing (classification vs. extraction vs. generation), encoder / encoder-decoder / decoder-only choice, multilingual vs. monolingual, build-vs-buy.
- Allocate budget across data engineering, training, fine-tuning, retrieval, and managed-API usage with documented trade-offs.
- Design eval gates, release-candidate selection, and rollback strategy for NLP releases.
- Survey domain-NLP patterns (clinical de-id + coding, legal clause extraction, financial event extraction, scientific NLP) and the data-governance constraints each imposes.
- Author a model card, dataset-card chain, and responsible-NLP release review that satisfies governance / risk / security partners (depth handed off to upstream tracks).

## Chapters

1. [From product goal to NLP system blueprint](01-product-goal-to-nlp-blueprint.md)
2. [Budget allocation across the NLP stack](02-budget-allocation-across-the-nlp-stack.md)
3. [Release gates, candidate selection, and rollback for NLP](03-release-gates-and-rollback.md)
4. [Domain NLP patterns and their data-governance constraints](04-domain-nlp-patterns-and-governance.md)
5. [Model cards, dataset cards, and the responsible-NLP release review](05-model-cards-dataset-cards-and-responsible-release-review.md)

## Exercises

- [exercise-01 · NLP task framing and architecture rubric](exercises/exercise-01-nlp-task-framing-and-architecture-rubric.md)
- [exercise-02 · Build vs. buy vs. fine-tune vs. RAG decision doc](exercises/exercise-02-build-vs-buy-vs-fine-tune-vs-rag-decision-doc.md)
- [exercise-03 · Release gates and rollback design](exercises/exercise-03-release-gates-and-rollback-design.md)
- [exercise-04 · Domain-NLP patterns survey](exercises/exercise-04-domain-nlp-patterns-survey.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
