# Sentence Segmentation, Lemmatisation, Stop-Words, and Casing Policy

## Motivation

Every downstream NLP component eventually asks the same three questions of its input: "Where does a sentence end?", "Are `run`, `ran`, and `running` the same thing?", and "Should `The` and `the` be one thing or two?" The answers are policy decisions. Get them wrong and the same document indexed at 2 pm and 3 pm produces different embeddings, different NER spans, and different retrieval hits.

This chapter is about building the front-door pipeline that answers those questions consistently: sentence segmentation, lemmatisation, stop-word handling, and casing. It builds on the Unicode chapter of module 101 (UAX #29, NFC/NFKC, BCP-47) and turns it into a concrete normalisation policy.

## Sentence segmentation

### Why it is not `split(".")`

Naïve rules fail on:

- **Abbreviations:** `Dr. Smith`, `U.S.`, `e.g.`, `Fig. 3`.
- **Decimal numbers and versions:** `3.14`, `Python 3.10`.
- **URLs and file paths:** `example.com`, `/etc/hosts`.
- **Ellipses and non-ASCII terminators:** `…`, `。` (CJK full stop), `؟` (Arabic question mark), `።` (Ethiopic).
- **Quotes and parentheses:** `"…was true." Then he left.` vs. `"…was true. Then he left."`.
- **Missing terminators:** headers, list items, chat messages.

Anything that hard-codes `.`, `?`, `!` breaks on multilingual or noisy data.

### The three approaches

1. **Rule-based (UAX #29 + tailoring).**
   - ICU's `BreakIterator` with `getSentenceInstance(locale)` implements UAX #29 sentence boundary analysis and applies CLDR-tailored abbreviation lists per locale. This is the default in Java, .NET, and PyICU. For Python without ICU, `pysbd` (Pragmatic Segmenter port) is a reasonable fallback and ships language-specific rule sets.
   - **When to use:** structured, mostly-clean text; multilingual with locale metadata; when determinism and offline execution are hard requirements.
2. **Statistical (Punkt-style).**
   - NLTK's `PunktSentenceTokenizer` (Kiss & Strunk, "Unsupervised Multilingual Sentence Boundary Detection", *CL 2006*) learns abbreviations and sentence starters from unlabelled data. Ships pre-trained for English, German, Portuguese, Estonian, etc.
   - **When to use:** you have a corpus in a language without a good rule-based tailoring, or your domain (e.g. legal, biomedical) has abbreviations that off-the-shelf rules miss.
3. **Neural.**
   - spaCy's default `senter` and `parser` both produce sentence boundaries. Stanza uses a joint tokenise + sentence-split model. Both handle noisy text (chat, transcripts) better than rules but at 10-100× the cost.
   - **When to use:** noisy user-generated text; scripts without punctuation cues; when accuracy of downstream parsing depends on the split and you can afford the model.

Rule of thumb: start with ICU or spaCy, evaluate on 100 held-out documents from your domain, only train a statistical splitter if the recall/precision on that eval justifies the operational cost.

### A locale-aware Python example

```python
import icu  # PyICU

def sentences(text, locale="en_US"):
    it = icu.BreakIterator.createSentenceInstance(icu.Locale(locale))
    it.setText(text)
    start = it.first()
    for end in it:
        yield text[start:end].strip()
        start = end

for s in sentences("Dr. Smith arrived at 3.14 p.m. He said hi.", "en_US"):
    print(repr(s))
# 'Dr. Smith arrived at 3.14 p.m.'
# 'He said hi.'
```

The same call with `locale="ja"` correctly splits on `。` and treats CJK punctuation.

## Stemming vs. lemmatisation

Both reduce inflectional variants to a canonical form. They differ in ambition:

- **Stemming** — a fixed set of morphological rewrite rules (Porter, Snowball, Lancaster). Cheap, deterministic, language-specific, and often produces non-words (`running → run`, `fairly → fair`, `argument → argu`). Reference: Martin Porter, ["An algorithm for suffix stripping"](https://tartarus.org/martin/PorterStemmer/def.txt), *Program*, 14(3), 1980; the current Snowball family — Porter2 for English plus stemmers for 15+ other languages — lives at <https://snowballstem.org/>.
- **Lemmatisation** — reduces to the *dictionary form*. Needs a lexicon (WordNet for English, morphology FSTs for others) and usually a POS tag to disambiguate (`saw` → `see` verb vs. `saw` noun). Slower, more accurate, always a real word.

Which to use:

| Task                            | Prefer      | Why                                                     |
|---------------------------------|-------------|---------------------------------------------------------|
| Full-text search (BM25) recall  | Stem        | Aggressive normalisation, speed matters                 |
| NLU / IE that reads the lemma   | Lemmatise   | The lemma is user-visible; must be a real word          |
| Multi-lingual pipeline          | Lemmatise (or FST morphology) | Snowball covers many languages, but agglutinative languages need morphological FSTs |
| Downstream neural model         | Neither     | Sub-word tokenisation handles morphology; adding a stemmer is usually a regression |

For serious morphological analysis in agglutinative or templatic languages (Finnish, Turkish, Arabic, Hebrew), Snowball is too shallow. Use `HFST`-based analysers (see chapter 03) or Stanza, which trains language-specific lemmatisers on Universal Dependencies treebanks (<https://universaldependencies.org/>).

### Python: canonical calls

```python
# Snowball (multilingual, no POS needed)
from nltk.stem.snowball import SnowballStemmer
stem_en = SnowballStemmer("english")
stem_fi = SnowballStemmer("finnish")

# WordNet (English lemmatisation with POS)
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
lem.lemmatize("running", pos="v")   # 'run'

# spaCy (any language with a model)
import spacy
nlp = spacy.load("en_core_web_sm")
for tok in nlp("The children were running fastest"):
    print(tok.text, "->", tok.lemma_)
```

## Stop-words

A stop-word list is a policy statement: "these tokens contribute more noise than signal to my task." Two things go wrong with them in practice:

- **Wrong task fit.** Stop-words help sparse retrieval (BM25) and topic models (LDA) by removing high-frequency tokens with low IDF. They *hurt* dense-retrieval, generation, sentiment, and NER because those tasks either encode context implicitly or explicitly need function words. Do not add stop-words to a neural pipeline out of habit.
- **Wrong list.** Every library ships a different list. NLTK's English list has 179 words in v3.9. spaCy's has ~326. Sklearn's has 318. sklearn's list has known issues documented in its own docs ([sklearn text feature extraction guide](https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words)); no list is authoritative. Pick one, commit it to your repo, and treat changes as breaking.

Rules that keep this manageable:

- Store the stop-word list *in your repo*, not by name from a library. `stopwords = {"the", "a", ...}` in a config file. Do not `nltk.corpus.stopwords.words("english")` at inference; the corpus can update.
- Version the list. Adding or removing a word changes indices and should ship with a re-indexing plan.
- Never strip stop-words before a neural model unless you have benchmark evidence it helps that specific task.
- For non-English, prefer language-native lists (<https://github.com/stopwords-iso/stopwords-iso> aggregates community lists; audit before use).

## Casing policy

The classic trade-off: cased text preserves signal (`Apple` company vs. `apple` fruit; German nouns capitalised); lowercased text collapses variants (`iPhone` vs. `IPHONE` vs. `iphone`).

Options, in increasing sophistication:

1. **Preserve case.** The right default in 2026 for any pipeline that ends in a Unicode-correct sub-word tokenizer. GPT-family tokenizers see `The` and `the` as different tokens, and the model learns when the distinction matters.
2. **Lowercase (or, correctly, `casefold()`).** Reduces vocabulary in classical pipelines (BM25, sparse retrieval). Use `str.casefold()`, not `str.lower()` — it handles Turkish dotless-i (`İ ↔ i`), German `ß`, and Greek final-sigma correctly. Reference: [Python `str.casefold` documentation](https://docs.python.org/3/library/stdtypes.html#str.casefold).
3. **True-casing.** Restore proper casing from a noisy input (e.g., all-caps chat, ASR output). Approaches: dictionary lookup for known entities, statistical models trained on well-cased text, or a neural sequence tagger. Referenced widely in ASR post-processing.
4. **Case as a feature.** Keep the surface case in the input but expose flags: `is_upper`, `is_title`, `is_all_caps`, `contains_mixed_case` to a downstream tagger. spaCy already surfaces these on every `Token`.

Turkish and Azerbaijani are the perennial trap. `"İSTANBUL".lower()` on a POSIX locale gives `"i̇stanbul"` (a lowercase `i` with a combining dot above); `"i".upper()` gives `"I"` — the wrong letter. Always call `casefold()` and, when locale is known, use ICU's `UnicodeString.foldCase()` or Python 3.12+ `str.casefold()` with locale-aware upstream code. This is not academic — Turkish `İ` handling has broken production auth systems (Spotify's "Turkey `i` bug", widely blogged).

## Putting it together: the normalisation front door

A production classical front door for English + Spanish text might look like this:

```python
import unicodedata, re, icu
from nltk.stem.snowball import SnowballStemmer
STOPWORDS = frozenset(open("configs/stopwords_en.txt").read().split())
STEM = {"en": SnowballStemmer("english"), "es": SnowballStemmer("spanish")}

_WS = re.compile(r"\s+")

def normalise(text: str, lang: str = "en") -> list[str]:
    # 1. Encoding assumed decoded; reject undecodable upstream.
    # 2. Unicode normalise.
    text = unicodedata.normalize("NFKC", text)
    # 3. Case-fold (Unicode-correct; better than .lower() for Turkic/German).
    text = text.casefold()
    # 4. Whitespace collapse.
    text = _WS.sub(" ", text).strip()
    # 5. UAX #29 word segmentation via ICU.
    it = icu.BreakIterator.createWordInstance(icu.Locale(lang))
    it.setText(text)
    tokens, start = [], it.first()
    for end in it:
        piece = text[start:end]
        if piece.strip():  # skip whitespace tokens
            tokens.append(piece)
        start = end
    # 6. Stop-word removal.
    tokens = [t for t in tokens if t not in STOPWORDS]
    # 7. Stem.
    stem = STEM[lang]
    return [stem.stem(t) for t in tokens]
```

Notice what is *not* in the pipeline: no `lower()`, no `split()`, no ASCII-only regex, no hard-coded stop-word list. Every step names a policy decision that must be documented in the config file next to the code. The pipeline is deterministic, per-language, and reviewable.

## Versioning the front door

A single normalisation change (add a stop-word, switch to `casefold()`, upgrade Snowball) changes every token ID produced downstream. Rules that keep this from becoming an outage:

- **Version the normaliser** as its own artefact (`text-norm@v3.2.1`). Bump on any behaviour change.
- **Attach the normaliser version** to every persisted artefact — indexed document, cached embedding, training example. Reject reads across a version mismatch until the artefact is re-normalised.
- **Ship changes behind a re-indexing job.** Never change stop-words or stemmers in a hot path without a rebuild plan.

## Chapter summary

- Sentence segmentation is not `split(".")`. Use ICU / UAX #29 for structured multilingual text, Punkt-style statistical splitters for domain-adapted work, or a neural splitter for noisy user text.
- Stemming (Porter/Snowball) is cheap and rough; lemmatisation (WordNet, morphology FSTs, spaCy, Stanza) is accurate and expensive. Match the tool to the downstream consumer.
- Stop-word lists are policy artefacts: pin them to your repo, version them, never apply them to a neural pipeline without evidence.
- Casing decisions have Unicode traps (Turkish `İ`, German `ß`); use `casefold()`, not `lower()`, and consider case as a feature rather than something to erase.
- The front-door pipeline is a versioned artefact; every downstream cache and index must know which version produced it.
- The next chapter zooms out from single-token normalisation to sequence-level statistics with n-gram language models.
