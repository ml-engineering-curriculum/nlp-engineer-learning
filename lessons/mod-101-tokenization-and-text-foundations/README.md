# mod-101-tokenization-and-text-foundations: Tokenization & Text Foundations: Subword Algorithms, Unicode, and Transformer Internals from the NLP Perspective

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 14 hours

## Learning objectives

- Implement and reason about subword tokenization (BPE, WordPiece, Unigram LM, SentencePiece) and pick the right scheme per language family
- Apply Unicode-correct text processing (UAX #29 segmentation, ICU normalisation, BCP-47 language tags, CJK / Arabic / Indic script handling)
- Train a domain-adapted tokenizer with the Hugging Face tokenizers library and explain how its vocabulary shapes downstream task quality
- Trace a forward pass through encoder, encoder-decoder, and decoder-only transformer families and explain when each is the right NLP-modeling choice
- Reason about KV-cache behaviour, attention shape, and tokenizer-aware sequence-length budgets that recur in every later module

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
