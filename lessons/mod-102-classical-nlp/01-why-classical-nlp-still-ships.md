# Why Classical NLP Still Ships

## Motivation

By 2026, most public NLP demos are decoder-only LLMs. Most production NLP is not. Somewhere between the user's request and the transformer, real systems still run regex, finite-state transducers, `langdetect` or CLD3, spaCy `Matcher` rules, a KenLM n-gram model rescoring an ASR hypothesis, and a Porter-family stemmer building a search index. These components are not embarrassing legacy — they are load-bearing. They ship because they satisfy constraints a neural model cannot: cost per document, tail-latency, auditability, offline execution, guaranteed reversibility, and zero training data.

This module is about that layer. It exists for two reasons:

1. **You will maintain it.** Any long-lived NLP codebase will contain a normalisation front door, a language router, a lemmatiser, or a rule matcher. Understanding these components lets you fix them, retire them, or defend them against a "let's just use an LLM" refactor that would regress precision, latency, or cost.
2. **You will design new hybrids.** The best production pipelines mix classical and neural: rules where rules are cheap and correct; a small neural model where fuzziness is required; an LLM only where the input truly demands generalisation. Choosing the seams requires knowing what each layer is good at.

If module 101 was the transformer boundary — text becoming IDs — module 102 is the *pre-boundary* pipeline: text becoming clean text, and the small statistical models that surround the neural core.

## The three eras and what they left behind

- **Symbolic (roughly 1950s–1990s).** Hand-written grammars, expert systems, morphological analysers, finite-state phonology. Left behind: regex, FSTs, WordNet, Penn Treebank annotation guidelines, ICU's rule-based transliterators.
- **Statistical (roughly 1990s–2015).** MLE, smoothing, HMMs/CRFs, PCFGs, Perceptron/SVM taggers, phrase-based SMT, n-gram LMs at web scale. Left behind: KenLM/SRILM, MaltParser/MSTParser, CRF++, `scikit-learn` text pipelines, feature engineering habits.
- **Neural (roughly 2013–present).** word2vec, seq2seq, transformers, foundation models. Left behind: the assumption that everything is a supervised or in-context problem — an assumption the earlier two eras still routinely undercut.

Classical components did not survive by inertia. They survived by having engineering properties that neural models do not automatically inherit.

## What classical NLP is genuinely better at

- **Determinism.** A regex or FST produces the same output every time, on every machine, offline, with zero warm-up. A neural model produces the same output *ish*, subject to floating-point non-determinism, quantisation, model version pinning, and prompt-cache boundaries. When your normaliser feeds a downstream ID system, deterministic wins.
- **Auditability.** A compliance reviewer can read a 40-line regex. Explaining why a 7 B model tagged a phone number as an address requires SHAP, integrated gradients, or a shrug. In regulated domains — health, finance, legal — the auditable component is the one that ships.
- **Latency and cost.** A `re2` match on a 4 KB document runs in microseconds. A byte-level BPE + transformer forward pass runs in milliseconds on a warm GPU and tens of milliseconds on CPU. For real-time paths (search query rewriting, ASR post-processing, log parsing), classical layers exist to keep the neural layer off the critical path.
- **Reversibility.** Text-normalisation FSTs are designed to be inverted: TN (spoken → written) and ITN (written → spoken) share grammars. Neural sequence-to-sequence models are one-way — you can train the inverse but not derive it.
- **Cold-start.** Classical models need zero labels for regex/FST rules and thousands (not millions) for a CRF tagger. New domains, low-resource languages, and internal jargon dictionaries stand up in a day.
- **Tight resource envelopes.** Edge devices, on-device search, embedded ASR, keyboards, autocomplete-in-a-textbox. Kilobytes of vocabulary and megabytes of KenLM beat a distilled encoder every time.

## What neural is genuinely better at

The counter-list matters, because "classical is fine" is as wrong as "neural is fine":

- **Long-range semantic composition** — coreference, discourse, multi-hop reasoning.
- **Zero-shot generalisation** to unseen phrasings, domains, or task descriptions.
- **Free-form generation** — summarisation, translation, question answering that requires writing a novel sentence.
- **Multi-modal or multi-task** interfaces where a single prompt controls behaviour.

Anything on this list, classical NLP handles only via ever-larger rule bases, and rule bases have a productivity ceiling.

## The hybrid pattern that wins

Most production NLP systems reach a version of this shape:

```
raw input
  │
  ▼
[1] Unicode + encoding decode              ── classical (must be lossless)
  │
  ▼
[2] Language identification & routing      ── classical (fastText / CLD3)
  │
  ▼
[3] Text normalisation                     ── classical (regex / FST)
  │
  ▼
[4] Sentence + token segmentation          ── classical (UAX #29 / ICU)
  │
  ▼
[5] Rule-based extraction (cheap wins)     ── classical (Matcher / TokensRegex)
  │
  ▼
[6] Neural model (task-specific)           ── neural (encoder / decoder LLM)
  │
  ▼
[7] Post-processing & guardrails           ── classical (regex, FST, schema)
  │
  ▼
downstream consumer
```

Layers 1-5 and 7 are what this module builds. The pattern is not a coincidence: it minimises the traffic that reaches the expensive neural layer, catches whatever a rule can catch before probability enters the picture, and constrains the neural output back to a machine-readable shape.

## Concrete places classical still wins in 2026

- **Text normalisation and ITN in ASR/TTS.** Google's Kestrel, and the open-source successor lineage `Sparrowhawk` → NVIDIA's `nemo-text-processing`, still ship as WFST grammars. Neural TN systems exist but are typically post-corrected by an FST safety net so `USD 1,204.30` and `twelve hundred and four dollars thirty cents` round-trip losslessly (see the [`nemo-text-processing` documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html) and the primary paper Zhang et al., "NeMo Inverse Text Normalization: From Development To Production", *Interspeech 2021*, [arXiv:2104.05055](https://arxiv.org/abs/2104.05055)).
- **Language identification.** Google's Compact Language Detector v3 (CLD3) and Meta's `fastText` LID are the workhorse routers for scraped corpora and browser translation gating. Both are single-digit MB and run in microseconds per document.
- **Search and retrieval preprocessing.** Elasticsearch/OpenSearch analyzers are chains of classical components — lowercasing, ICU folding, stemmers (`porter`, `snowball`), stopword filters, synonym maps. The retrieval quality of a hybrid BM25 + dense system depends on those analyzers being right.
- **ASR/TTS language modelling and re-scoring.** KenLM n-gram models rescore lattice hypotheses in production ASR pipelines because they are cheap enough to run in a beam.
- **Log parsing, protocol parsing, PII scrubbing, code linting for prompts.** These are grammar problems, not classification problems. Regex and PEG parsers do them faster and more predictably than any transformer.
- **Rule-based safety and guardrail layers.** Deterministic filters run *around* an LLM to catch policy violations that must be caught with probability 1 (e.g., blocking a specific competitor name, redacting a Social Security number pattern).

## How to read the rest of the module

- Chapters **02** and **03** cover the primitives — regular expressions and finite-state transducers — with attention to Unicode-correct behaviour, engine choice, and morphology.
- Chapter **04** turns those primitives into a normalisation front door with sentence segmentation, lemmatisation, and casing/stop-word policy.
- Chapters **05** and **06** cover the small statistical models that still hold their ground: n-gram language models and sequence taggers.
- Chapter **07** looks at classical parsers — dependency and constituency — as the substrate for rule-based extraction.
- Chapter **08** compares the four major toolkits (spaCy, NLTK, Stanza, Stanford CoreNLP) and shows how to compose their pieces into a maintainable pipeline.
- Chapter **09** covers language identification and per-language routing.
- Chapter **10** closes with a decision framework: when to reach for classical, when for neural, when for a deliberate hybrid.

## Chapter summary

- Classical NLP components ship in 2026 not for nostalgia but because they satisfy engineering constraints — determinism, latency, cost, auditability, cold-start, reversibility — that neural models do not automatically satisfy.
- The dominant production pattern is a hybrid: classical layers around a smaller, more expensive neural core.
- Understanding the classical stack is defensive (you will maintain it) and generative (you will design new hybrids that beat pure-neural baselines on the metrics that ship products).
- The rest of the module builds each layer of that stack in turn.
