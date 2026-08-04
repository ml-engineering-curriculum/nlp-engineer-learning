# mod-106 · Summarisation & Controlled Generation

Encoder-decoder summarisation from BART/T5/PEGASUS/mT5, long-document and multi-document strategies, the decoding-strategy zoo (greedy → beam → nucleus → typical → contrastive), constrained/structured generation, and the faithfulness metrics that stop your summariser from confidently making things up.

**Estimated effort:** 16 hours

## Learning objectives

- Fine-tune encoder-decoder models (BART / T5 / PEGASUS / mT5) for extractive and abstractive summarisation.
- Handle long-document and multi-document summarisation with chunk-then-fuse and hierarchical strategies.
- Apply decoding strategies (greedy, beam, nucleus, typical, contrastive) and reason about their quality / diversity trade-offs.
- Implement constrained generation (regex / JSON-schema / context-free-grammar) for structured outputs.
- Diagnose and mitigate hallucination and faithfulness failures with ROUGE / BERTScore / faithfulness probes / human eval.

## Chapters

1. [The summarisation and controlled-generation landscape](01-summarisation-and-controlled-generation-landscape.md)
2. [Extractive summarisation baselines: LEAD-N, TextRank, and BERTSum](02-extractive-summarisation-baselines.md)
3. [Abstractive summarisation with BART, T5, PEGASUS, and mT5](03-abstractive-summarisation-with-bart-t5-pegasus-mt5.md)
4. [Long-document summarisation: chunk-and-fuse, hierarchical, and long encoders](04-long-document-summarisation.md)
5. [Multi-document summarisation](05-multi-document-summarisation.md)
6. [Decoding strategies: greedy, beam, nucleus, typical, contrastive](06-decoding-strategies.md)
7. [Generation controls: length, repetition, and calibration knobs](07-generation-controls-length-repetition-calibration.md)
8. [Constrained decoding: regex, JSON schema, and context-free grammars](08-constrained-decoding-regex-json-cfg.md)
9. [Structured output patterns for summarisation](09-structured-output-patterns.md)
10. [Reference-based evaluation: ROUGE, BERTScore, and their limits](10-reference-based-evaluation-rouge-bertscore.md)
11. [Faithfulness and hallucination diagnostics](11-faithfulness-and-hallucination-diagnostics.md)
12. [Mitigating hallucination in generation](12-mitigating-hallucination.md)
13. [Human evaluation for summarisation](13-human-evaluation-for-summarisation.md)

## Exercises

- [exercise-01 · Abstractive summarisation with BART or T5](exercises/exercise-01-abstractive-summarisation-with-bart-or-t5.md)
- [exercise-02 · Long-document and multi-doc summarisation](exercises/exercise-02-long-document-and-multi-doc-summarisation.md)
- [exercise-03 · Decoding strategies comparison](exercises/exercise-03-decoding-strategies-comparison.md)
- [exercise-04 · Constrained and structured generation](exercises/exercise-04-constrained-and-structured-generation.md)
- [exercise-05 · Faithfulness and hallucination diagnostics](exercises/exercise-05-faithfulness-and-hallucination-diagnostics.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
