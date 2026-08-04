# Language Identification and Routing

## Motivation

Every multilingual NLP pipeline eventually needs to answer: *"what language is this document written in?"* — and route it to the correct tokenizer, model, gazetteer, or downstream service. Get this wrong and English-language content is scored against a Chinese sentiment model, PII regex written for Latin script silently misses Cyrillic, and your bilingual users get monolingual results.

Language identification (LID) is one of the classic classical-NLP components that has *not* been replaced by an LLM in production. It is called far too often (at ingest time, on every chat message, on every web page) to afford a transformer call. The reigning tools — Meta's `fastText` LID and Google's Compact Language Detector v3 (CLD3) — are single-digit MB, run in microseconds on CPU, and cover 100+ languages.

## What the input actually looks like

Real-world language ID inputs cluster into a few shapes:

- **Long, monolingual, clean.** A Wikipedia paragraph or a news article. Any reasonable LID model gets these right.
- **Short, monolingual, noisy.** A tweet, a search query, a chat message. Under ~40 characters, accuracy drops sharply for every model. This is where most bugs happen.
- **Code-switched.** Two or more languages within one document ("I'm going to the boulangerie later"). Off-the-shelf LID reports the dominant language; genuine code-switching detection is a separate task.
- **Same-script family confusables.** Norwegian vs. Danish vs. Swedish. Spanish vs. Galician vs. Portuguese. Malay vs. Indonesian. Serbian (Cyrillic) vs. Bulgarian vs. Macedonian. Every LID model has known error modes here.
- **Same-language, different script.** Serbian in Cyrillic (`sr-Cyrl`) vs. Latin (`sr-Latn`). Hindi in Devanagari vs. transliterated Latin. Urdu in Perso-Arabic vs. Latin. This is a language *and* script identification problem, not language alone.

Design your LID layer around these shapes, not around a single expected-input assumption.

## The tools that actually ship

### fastText language identification (Meta / FAIR)

- Bag-of-character-n-grams over a linear softmax classifier. Trained on Wikipedia + Tatoeba across 176 languages (the `lid.176.bin` model; the smaller `lid.176.ftz` quantised variant is ~1 MB).
- Reference: Bojanowski, Grave, Joulin, Mikolov, ["Enriching Word Vectors with Subword Information"](https://arxiv.org/abs/1607.04606), *TACL 2017*; and the classifier architecture in Joulin et al., ["Bag of Tricks for Efficient Text Classification"](https://arxiv.org/abs/1607.01759), *EACL 2017*.
- Runs offline; predictions include a confidence.
- Model files: <https://fasttext.cc/docs/en/language-identification.html>.

```python
import fasttext
model = fasttext.load_model("lid.176.bin")
labels, probs = model.predict("Bonjour tout le monde", k=3)
# (('__label__fr', '__label__en', '__label__ca'), [0.98, 0.01, 0.003])
```

### CLD3 — Google's Compact Language Detector v3

- Character-n-gram neural network trained on web data, 107 languages.
- The engine inside Chromium's "translate this page?" prompt.
- Reference implementation: <https://github.com/google/cld3>. Python wrapper: `pycld3`, and its recent successor `gcld3` on PyPI.

### CLD2 (still in the wild)

- Older, purely rule + n-gram tables; used by many pre-2020 pipelines. Still worth knowing for legacy codebases (`pycld2`).

### `langid.py`

- Pure-Python, trained on 97 languages; a good pip-installable fallback when you cannot ship a C dependency.
- Reference: Lui & Baldwin, ["langid.py: An Off-the-shelf Language Identification Tool"](https://aclanthology.org/P12-3005.pdf), *ACL 2012*.

### `langdetect`

- Port of Nakatani Shuyo's Java `language-detection` (Bayesian classifier over character n-grams). Widely used, but the underlying algorithm is 15+ years old and it is nondeterministic by default (call `DetectorFactory.seed = 0` for reproducibility). Prefer fastText / CLD3 for new work.

### Character-n-gram LMs (the DIY option)

- One n-gram LM per language, scored on the input; highest per-character log-probability wins.
- Reference: Cavnar & Trenkle, "N-Gram-Based Text Categorization", *SDAIR 1994*.
- Useful when you need to add a domain-specific "language" that off-the-shelf models do not cover (e.g., "structured JSON logs" vs. "prose").

## When to use what

| Requirement                                          | Reach for                        |
|------------------------------------------------------|----------------------------------|
| Simple, offline, 100+ languages, tiny model          | fastText `lid.176.ftz` (1 MB)    |
| Web-page language ID with modest memory              | CLD3                             |
| Custom in-house "languages" (log formats, dialects)  | Character-n-gram LM per class    |
| Pure-Python, no native deps                          | `langid.py`                      |
| Legacy repo you must keep running                    | Whatever it already uses         |

Do **not** reach for a general LLM here. It is 3-6 orders of magnitude more expensive per call and does not improve over fastText for the common case.

## Script-based fast paths

A large fraction of LID work can be done without a model at all, by inspecting Unicode scripts:

- Any document with only Han + CJK punctuation: `zh` / `ja` / `ko` (script alone does not distinguish). Break down further with a small model.
- Any document with a majority of Devanagari: `hi` / `mr` / `ne` / `sa`.
- Any document with only Cyrillic: `ru` / `uk` / `bg` / `sr-Cyrl` / etc.
- Any document with only Arabic script: `ar` / `fa` / `ur` / `ku-Arab`.
- Any document with only Latin script + no diacritics + fewer than 20 words: use a model.

Python:

```python
import unicodedata
from collections import Counter

def script_histogram(text: str) -> dict[str, int]:
    counts = Counter()
    for ch in text:
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue
        # Take the first script-like word out of the character name.
        if " " in name:
            script = name.split(" ", 1)[0]
        else:
            script = ""
        counts[script] += 1
    return counts
```

More rigorously, use `unicodedata.category(ch)` and Unicode `Script` property (via PyICU: `icu.Char.getPropertyValueName(icu.UProperty.SCRIPT, ...)`). A script histogram is a cheap pre-filter that catches the confident cases and hands the ambiguous ones to a real model.

## Confidence and threshold policy

LID models emit a probability distribution. Two decisions you have to make:

- **Confidence threshold below which you treat the answer as `und` (undetermined).** Below this, do not route to a language-specific pipeline; use a script-based fallback or reject.
- **Delta threshold between top-1 and top-2.** If the model says `en=0.51, sv=0.49`, that is not a confident answer, even though top-1 exceeds 0.5.

Concrete pattern:

```python
def identify_language(model, text: str) -> str:
    labels, probs = model.predict(text.replace("\n", " "), k=3)
    if probs[0] < 0.65 or probs[0] - probs[1] < 0.15:
        return "und"
    return labels[0].replace("__label__", "")
```

Tune the thresholds against a held-out sample from *your* domain. General benchmarks are optimistic on your data.

## Short-text handling

Under ~40 characters, all common LID models suffer. Mitigations:

- **Aggregate context.** Batch messages from the same user, session, or thread; run LID on the concatenation.
- **Use language priors from user metadata.** UI language, browser `Accept-Language`, prior message history. Combine with the model output via Bayesian update.
- **Confidence-weight the fallback.** For low-confidence short text, fall back to a script-based or majority-user-language route.

## Routing after identification

Once a document has a canonical BCP-47 tag (chapter 05 of module 101), the router picks the pipeline:

```
en, en-US, en-GB      → English pipeline
es, es-419, es-MX     → Spanish pipeline (regional pipelines only if evaluation justifies it)
zh-Hans / zh-Hant     → separate; do not merge
ja                    → Japanese pipeline
ar                    → Arabic pipeline (with RTL awareness downstream)
und                   → generic multilingual pipeline (e.g., XLM-R for classification)
```

Design rules:

- **Route by BCP-47 tag, not by ISO 639-1.** Serbian without a script is under-specified.
- **Route by capability, not by identity.** If English and German both use the same pipeline (multilingual encoder + English NER + German NER + shared tokenizer), you have one route; do not proliferate routes without benchmarks.
- **Log every route decision** — input, LID output, threshold decision, chosen route. Post-hoc debugging of "why was this doc routed to Chinese?" requires exactly this log.

## Code-switching and mixed-language documents

Pure LID is document-level. For genuine multilingual ("code-switched") content, you need:

- **Token-level LID** — a sequence tagger that emits a language per token. Reference: LinCE benchmark (Aguilar et al., ["LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation"](https://aclanthology.org/2020.lrec-1.223/), *LREC 2020*).
- **Multilingual models downstream** — XLM-R, mBERT, mT5 — that can handle mixed input without routing.

Most production systems ignore the problem until a customer complaint arrives; then they add a "multilingual" route to a shared multilingual model. Choose the earlier moment deliberately.

## Evaluating an LID pipeline

- **Per-language recall and precision.** Aggregate accuracy hides low-recall on minority languages.
- **Confusion matrix** for confusables (Nordic languages, Iberian languages, South Slavic Cyrillic).
- **Short-text bucket** — evaluate <50 char, 50-200, 200+ separately.
- **Coverage** — the fraction of your traffic that gets routed at all (rather than falling to `und`). Too high a threshold makes coverage collapse.
- **Latency and memory.** LID should be a sub-millisecond, sub-10 MB step.

## Chapter summary

- LID is a cheap, offline, deterministic step that gates every multilingual NLP pipeline; fastText and CLD3 are the modern defaults.
- Script-based fast paths handle confident cases without a model; feed the ambiguous remainder to fastText/CLD3.
- Confidence + delta thresholds turn a probability into a routing decision; `und` is a legitimate output.
- Short text and code-switching are the two most common failure modes; mitigate with context aggregation, priors, and (for genuine code-switching) token-level LID or a multilingual downstream model.
- Store the BCP-47 tag on every document; route by tag, not by ISO code alone.
- Next chapter: the meta-question that runs through the whole module — when to choose classical, when neural, when hybrid.
