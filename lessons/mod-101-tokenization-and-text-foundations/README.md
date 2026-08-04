# mod-101 · Tokenization & Text Foundations

Subword algorithms, Unicode-correct text handling, and transformer internals — from the NLP engineer's perspective.

**Estimated effort:** 14 hours

## Learning objectives

- Implement and reason about subword tokenization (BPE, WordPiece, Unigram LM, SentencePiece) and pick the right scheme per language family.
- Apply Unicode-correct text processing (UAX #29 segmentation, ICU normalisation, BCP-47 language tags, CJK / Arabic / Indic script handling).
- Train a domain-adapted tokenizer with the Hugging Face `tokenizers` library and explain how its vocabulary shapes downstream task quality.
- Trace a forward pass through encoder, encoder-decoder, and decoder-only transformer families and explain when each is the right NLP-modelling choice.
- Reason about KV-cache behaviour, attention shape, and tokenizer-aware sequence-length budgets that recur in every later module.

## Chapters

1. [Why tokenization shapes every later decision](01-why-tokenization-shapes-every-later-decision.md)
2. [Byte Pair Encoding and byte-level BPE](02-byte-pair-encoding-and-byte-level-bpe.md)
3. [WordPiece and Unigram LM](03-wordpiece-and-unigram-language-model.md)
4. [SentencePiece and language-family selection](04-sentencepiece-and-language-family-selection.md)
5. [Unicode-correct text processing](05-unicode-correct-text-processing.md)
6. [Training a domain-adapted tokenizer](06-training-a-domain-adapted-tokenizer.md)
7. [Transformer families from the NLP perspective](07-transformer-families-from-the-nlp-perspective.md)
8. [KV-cache, attention shape, and sequence budgets](08-kv-cache-attention-shape-and-sequence-budgets.md)

## Exercises

- [exercise-01 · BPE from scratch and against HF tokenizers](exercises/exercise-01-bpe-from-scratch-and-against-hf-tokenizers.md)
- [exercise-02 · SentencePiece Unigram vs. WordPiece experiment](exercises/exercise-02-sentencepiece-unigram-vs-wordpiece-experiment.md)
- [exercise-03 · Unicode segmentation and normalisation audit](exercises/exercise-03-unicode-segmentation-and-normalisation-audit.md)
- [exercise-04 · Domain-adapted tokenizer training](exercises/exercise-04-domain-adapted-tokenizer-training.md)
- [exercise-05 · Transformer family decision rubric](exercises/exercise-05-transformer-family-decision-rubric.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
