# mod-108 · Embeddings & Representation Learning for Text

Bi-encoders and cross-encoders trained with `sentence-transformers`, contrastive objectives and hard-negative mining for high-recall retrieval, MTEB evaluation across retrieval / reranking / STS / classification / clustering, domain adaptation (clinical / legal / scientific) and cross-lingual adaptation, and the serving stack — batching, quantisation, Matryoshka truncation, TEI — plus the durable ownership boundary with the `rag-engineer` track.

**Estimated effort:** 12 hours

## Learning objectives

- Train bi-encoder and cross-encoder sentence/document embedding models with `sentence-transformers`.
- Apply contrastive objectives, hard-negative mining, and curriculum-style training for high-recall retrieval embeddings.
- Evaluate embeddings on the Massive Text Embedding Benchmark (MTEB) across classification, clustering, retrieval, STS, and reranking.
- Adapt embeddings to a domain (clinical, legal, scientific) and to a target language family.
- Serve embeddings at scale — quantisation, batching, dimension-reduction trade-offs — and define the boundary with `rag-engineer` ownership.

## Chapters

1. [The embeddings and representation learning landscape](01-embeddings-and-representation-learning-landscape.md)
2. [Bi-encoders vs. cross-encoders](02-bi-encoders-vs-cross-encoders.md)
3. [`sentence-transformers`, pooling, and the similarity contract](03-sentence-transformers-pooling-and-similarity.md)
4. [Contrastive objectives for retrieval embeddings](04-contrastive-objectives-for-retrieval-embeddings.md)
5. [Hard-negative mining and curriculum training](05-hard-negative-mining-and-curriculum-training.md)
6. [Multi-stage training and cross-encoder distillation](06-curriculum-training-and-distillation.md)
7. [Cross-encoder reranking and ColBERT](07-cross-encoder-reranking-and-colbert.md)
8. [MTEB: the Massive Text Embedding Benchmark](08-mteb-evaluation-suite.md)
9. [Domain adaptation: clinical, legal, scientific](09-domain-adaptation-clinical-legal-scientific.md)
10. [Multilingual and cross-lingual embeddings](10-multilingual-and-cross-lingual-embeddings.md)
11. [Serving embeddings at scale](11-serving-embeddings-at-scale.md)
12. [The embeddings / rag-engineer boundary](12-the-embeddings-rag-boundary.md)

## Exercises

- [exercise-01 · Bi-encoder training with contrastive loss](exercises/exercise-01-bi-encoder-training-with-contrastive-loss.md)
- [exercise-02 · Hard-negative mining and cross-encoder rerank](exercises/exercise-02-hard-negative-mining-and-cross-encoder-rerank.md)
- [exercise-03 · MTEB evaluation suite](exercises/exercise-03-mteb-evaluation-suite.md)
- [exercise-04 · Domain and multilingual embeddings](exercises/exercise-04-domain-and-multilingual-embeddings.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
