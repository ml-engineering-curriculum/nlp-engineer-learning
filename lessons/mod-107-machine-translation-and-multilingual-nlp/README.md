# mod-107 · Machine Translation & Multilingual NLP

Encoder-decoder NMT with Marian / mBART / NLLB / M2M-100, terminology-constrained decoding and domain adaptation for production translation, multilingual classifiers and taggers on XLM-R and mT5 with cross-lingual zero-shot transfer, script- and locale-aware handling for Arabic / CJK / Indic, and the evaluation stack — BLEU / chrF / COMET / BLEURT plus MQM- and DA-style human protocols — that lets you actually decide whether a system is better on FLORES-200.

**Estimated effort:** 16 hours

## Learning objectives

- Train and adapt encoder-decoder NMT (Marian / mBART / NLLB) for high- and low-resource language pairs.
- Apply terminology constraints, glossary injection, and domain adaptation for production MT.
- Build multilingual classifiers, taggers, and extractors with XLM-R and mT5 — including cross-lingual zero-shot transfer.
- Evaluate MT with COMET / BLEURT / chrF / BLEU and design human direct-assessment protocols.
- Reason about script handling (Arabic, CJK, Indic), transliteration, and BCP-47 locale-aware evaluation across FLORES-200.

## Chapters

1. [The machine translation and multilingual NLP landscape](01-machine-translation-and-multilingual-landscape.md)
2. [Encoder-decoder NMT architectures: Marian, mBART, NLLB, M2M-100](02-encoder-decoder-nmt-architectures.md)
3. [Fine-tuning NMT for high-resource language pairs](03-fine-tuning-nmt-for-high-resource-pairs.md)
4. [Low-resource NMT and back-translation](04-low-resource-nmt-and-back-translation.md)
5. [Domain adaptation for production MT](05-domain-adaptation-for-production-mt.md)
6. [Terminology constraints and glossary injection](06-terminology-constraints-and-glossary-injection.md)
7. [Multilingual encoders: XLM-R and mT5](07-multilingual-encoders-xlm-r-and-mt5.md)
8. [Cross-lingual transfer and zero-shot](08-cross-lingual-transfer-and-zero-shot.md)
9. [Scripts, transliteration, and BCP-47 locale](09-scripts-transliteration-and-locale.md)
10. [Automatic MT evaluation: BLEU, chrF, COMET, BLEURT](10-mt-automatic-evaluation-bleu-chrf-comet-bleurt.md)
11. [Human evaluation for MT: DA, SQM, and MQM](11-human-evaluation-for-mt.md)
12. [Multilingual benchmarks: FLORES-200 and friends](12-multilingual-benchmarks-flores-and-friends.md)

## Exercises

- [exercise-01 · Encoder-decoder NMT fine-tune](exercises/exercise-01-encoder-decoder-nmt-fine-tune.md)
- [exercise-02 · Terminology-constrained decoding](exercises/exercise-02-terminology-constrained-decoding.md)
- [exercise-03 · Cross-lingual zero-shot classification](exercises/exercise-03-cross-lingual-zero-shot-classification.md)
- [exercise-04 · COMET / BLEU / chrF MT evaluation](exercises/exercise-04-comet-bleu-chrf-mt-evaluation.md)
- [exercise-05 · Low-resource MT with back-translation](exercises/exercise-05-low-resource-mt-with-back-translation.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
