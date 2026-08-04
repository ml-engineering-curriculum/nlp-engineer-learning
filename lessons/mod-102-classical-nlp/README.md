# mod-102 · Classical NLP

Regex, finite-state transducers, n-gram LMs, statistical taggers and parsers, and the production text-normalisation and language-routing layers that sit around every serious neural NLP pipeline.

**Estimated effort:** 12 hours

## Learning objectives

- Build a production-grade text-normalisation pipeline (regex / FSTs, Unicode-aware sentence segmentation, lemmatisation, stop-word and casing policy).
- Train and evaluate n-gram language models and statistical taggers as baselines and fallbacks.
- Use spaCy / NLTK / Stanza / Stanford CoreNLP to compose dependency parsing, POS tagging, and rule-based matching pipelines.
- Identify languages and route text correctly with fastText / CLD3-style classifiers.
- Recognise when a rule-based or hybrid classical-NLP solution beats a neural one — and why that case still appears in industrial NLP.

## Chapters

1. [Why classical NLP still ships](01-why-classical-nlp-still-ships.md)
2. [Regular expressions as production grammar](02-regular-expressions-as-production-grammar.md)
3. [Finite-state transducers for text normalisation and morphology](03-finite-state-transducers-for-text.md)
4. [Sentence segmentation, lemmatisation, stop-words, and casing policy](04-sentence-segmentation-lemmatisation-and-casing.md)
5. [n-gram language models and perplexity](05-ngram-language-models-and-perplexity.md)
6. [Statistical taggers: HMMs, CRFs, and the averaged perceptron](06-statistical-taggers-hmm-crf-perceptron.md)
7. [Dependency and constituency parsers](07-parsers-dependency-and-constituency.md)
8. [Composing pipelines with spaCy, NLTK, Stanza, and Stanford CoreNLP](08-composing-pipelines-spacy-nltk-stanza-corenlp.md)
9. [Language identification and routing](09-language-identification-and-routing.md)
10. [When classical (or hybrid) beats neural — and why](10-when-classical-or-hybrid-beats-neural.md)

## Exercises

- [exercise-01 · Regex and FST normaliser](exercises/exercise-01-regex-and-fst-normaliser.md)
- [exercise-02 · N-gram language model and perplexity](exercises/exercise-02-ngram-language-model-and-perplexity.md)
- [exercise-03 · spaCy pipeline with rule-based components](exercises/exercise-03-spacy-pipeline-with-rule-based-components.md)
- [exercise-04 · Language identification and routing](exercises/exercise-04-language-identification-and-routing.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
