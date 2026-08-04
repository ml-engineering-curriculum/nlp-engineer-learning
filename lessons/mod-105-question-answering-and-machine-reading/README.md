# mod-105 · Question Answering & Machine Reading Comprehension

Extractive, abstractive, and closed-book QA — plus the calibration, long-context, and human-evaluation questions every production reader has to answer.

**Estimated effort:** 12 hours

## Learning objectives

- Train extractive QA models on SQuAD-style data and evaluate with exact match / F1.
- Build abstractive and closed-book QA with encoder-decoder and decoder-only models.
- Handle long-context, multi-hop, and unanswerable questions correctly.
- Reason about the QA / RAG boundary: own the reader/generator side and link out to the `rag-engineer` track for retrieval depth.
- Design human-evaluation rubrics for QA where reference answers are insufficient.

## Chapters

1. [Question answering and the machine reading landscape](01-question-answering-and-the-machine-reading-landscape.md)
2. [Extractive QA and the SQuAD formulation](02-extractive-qa-and-the-squad-formulation.md)
3. [Training and decoding extractive QA models](03-training-and-decoding-extractive-qa-models.md)
4. [EM and F1: the SQuAD evaluation protocol](04-em-and-f1-the-squad-evaluation-protocol.md)
5. [Abstractive QA with encoder-decoder models](05-abstractive-qa-with-encoder-decoder-models.md)
6. [Closed-book QA with decoder-only LLMs](06-closed-book-qa-with-decoder-only-llms.md)
7. [Long-context QA: sliding window, long encoders, and Fusion-in-Decoder](07-long-context-qa-with-sliding-window-and-fid.md)
8. [Multi-hop QA and question decomposition](08-multi-hop-qa-and-question-decomposition.md)
9. [Unanswerability and abstention with SQuAD 2.0](09-unanswerability-and-abstention-with-squad-2.md)
10. [The QA / RAG boundary](10-the-qa-and-rag-boundary.md)
11. [Human evaluation rubrics for QA](11-human-evaluation-rubrics-for-qa.md)

## Exercises

- [exercise-01 · Extractive QA on SQuAD-style data](exercises/exercise-01-extractive-qa-on-squad-style-data.md)
- [exercise-02 · Abstractive QA with encoder-decoder](exercises/exercise-02-abstractive-qa-with-encoder-decoder.md)
- [exercise-03 · Long-context and multi-hop QA](exercises/exercise-03-long-context-and-multi-hop-qa.md)
- [exercise-04 · Unanswerability and calibration](exercises/exercise-04-unanswerability-and-calibration.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
