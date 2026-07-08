# mod-106-summarisation-and-controlled-generation: Summarisation & Controlled Generation: Decoding Strategies, Constraints, Faithfulness

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 16 hours

## Learning objectives

- Fine-tune encoder-decoder models (BART / T5 / PEGASUS / mT5) for extractive and abstractive summarisation
- Handle long-document and multi-document summarisation with chunk-then-fuse and hierarchical strategies
- Apply decoding strategies (greedy, beam, nucleus, typical, contrastive) and reason about their quality / diversity trade-offs
- Implement constrained generation (regex / JSON-schema / context-free-grammar) for structured outputs
- Diagnose and mitigate hallucination and faithfulness failures with ROUGE / BERTScore / faithfulness probes / human eval

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
