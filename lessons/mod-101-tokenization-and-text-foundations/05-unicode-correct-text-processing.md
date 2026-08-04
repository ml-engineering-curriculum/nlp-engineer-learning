# Unicode-Correct Text Processing

## Motivation

Every tokenizer inherits the correctness of its Unicode pipeline. If you `str.lower()` a Turkish `İ`, drop combining marks from Devanagari, count code points as characters in an emoji-heavy dataset, or split Arabic RTL text on visual order, you will silently corrupt the training data. The model will then learn to reproduce the corruption.

This chapter covers the four standards and libraries that keep text processing honest: UAX #29 (segmentation), the Unicode normalisation forms (UAX #15), BCP-47 (language tags), and ICU (the reference implementation of all of the above). It then works through the script-specific issues that come up most often in production NLP: CJK, Arabic, and Indic.

## Code points, grapheme clusters, and why they differ

A **code point** is an integer in `[0, 0x10FFFF]` — an entry in the Unicode Character Database. A **user-perceived character** is often several code points glued together. The Unicode term is a **grapheme cluster**.

Examples:

- `é` can be one code point (U+00E9) or two (U+0065 U+0301, "e" + combining acute). Both look identical.
- `👨‍👩‍👧` (family) is a zero-width-joiner sequence of three emoji code points plus two ZWJ joiners — five code points, one grapheme.
- `क्षि` (Devanagari) is four code points (क + ्  + ष + ि) but one perceived character.

`len("👨‍👩‍👧")` in Python returns 5. `len("é" in NFD)` returns 2. Naïve character counts, character-level tokenizers, and truncation policies are all wrong on this input. UAX #29 defines the boundary rules that convert code point sequences into grapheme clusters, words, and sentences.

## UAX #29 — Unicode Text Segmentation

[UAX #29](https://www.unicode.org/reports/tr29/) specifies default algorithms for three levels of segmentation:

1. **Grapheme cluster boundaries** — one user-perceived character. Use this for cursor movement, string truncation, and character counting.
2. **Word boundaries** — script-aware word splitting. Handles apostrophes, hyphenated compounds, kana, Han, and mixed scripts.
3. **Sentence boundaries** — end-of-sentence detection.

The algorithms are *defaults*. They can be extended per language via CLDR data. In Python, the reference implementations are:

- [`grapheme`](https://pypi.org/project/grapheme/) or [`uniseg`](https://pypi.org/project/uniseg/) for grapheme iteration.
- [`icu` (PyICU)](https://gitlab.pyicu.org/main/pyicu) for the full UAX #29 + CLDR-tailored break iterators.

```python
import grapheme

s = "👨‍👩‍👧 café"
len(s)                          # 10  (code points)
grapheme.length(s)              # 6   (grapheme clusters, ignoring space)
list(grapheme.graphemes(s))     # ['👨‍👩‍👧', ' ', 'c', 'a', 'f', 'é']
```

Use grapheme-cluster iteration whenever you:

- truncate strings to fit a UI or a byte budget;
- count "characters" for user-facing metrics;
- build a character-level tokenizer or per-character embeddings.

## UAX #15 — Normalisation forms

[UAX #15](https://www.unicode.org/reports/tr15/) defines four normal forms:

- **NFC** — canonical composition. `é` (U+0065 U+0301) → `é` (U+00E9). Use for storage and web APIs (W3C recommends NFC).
- **NFD** — canonical decomposition. `é` → `e` + combining acute. Use when you want to work at the combining-mark level.
- **NFKC** — compatibility composition. Additionally maps compatibility variants: `ﬁ` (ligature) → `fi`, full-width Latin → half-width, superscript digits → regular digits. Use for search, tokenizer training, and any pipeline that should treat visually-equivalent variants identically.
- **NFKD** — compatibility decomposition. NFKC without recomposition.

Tokenizers usually apply NFC or NFKC. The choice matters:

- **NFC** preserves compatibility characters (full-width Latin, superscript digits) as distinct tokens. Useful for tasks where those variants are semantically meaningful (typography, mathematics).
- **NFKC** collapses variants. Reduces vocabulary size and improves recall on user-typed queries, but loses information — the token for `x²` and `x2` becomes the same.

For most modern SentencePiece pipelines, `nmt_nfkc` (NFKC plus some NMT-specific rules from the SentencePiece source) is the default and a safe choice for training data that has already been web-scraped.

Never assume input is normalised. Explicitly normalise at the pipeline's front door, even in "clean" datasets.

## BCP-47 — Language tags

[BCP-47](https://www.rfc-editor.org/info/bcp47) (RFC 5646) is the standard for identifying human languages, scripts, and regions in text metadata. A well-formed tag has up to five components:

```
language[-script][-region][-variant][-extension]
```

Examples:

- `en` — English
- `en-US` — American English
- `zh-Hans` — Chinese, Simplified script
- `zh-Hant-TW` — Chinese, Traditional script, Taiwan region
- `sr-Latn` — Serbian in Latin script (vs. `sr-Cyrl` for Cyrillic)
- `ar-EG` — Egyptian Arabic
- `hi-Deva`, `hi-Latn` — Hindi in Devanagari vs. transliterated Latin
- `und` — undetermined language (use when a language identifier failed)

Rules that matter:

- Tags are **case-insensitive**, but there is a canonical case (language lowercase, script title-case, region uppercase). Store canonical form.
- **Script is not implied by language**. Serbian, Chinese, Hindi, Uzbek, Azerbaijani, Kazakh all have multiple written scripts. If you route per language without a script tag, you will misroute.
- **Region is not a proxy for script**. `zh-CN` implies Simplified only by convention; explicit `zh-Hans-CN` is safer for tokenizer routing.

Store BCP-47 tags on every document from the moment it enters your pipeline. Language identification (fastText, CLD3, langid.py) is covered in module 102, but the tag format is set here.

## ICU — the reference implementation

[International Components for Unicode (ICU)](https://icu.unicode.org/) is the C/C++/Java library that implements essentially every Unicode standard: normalisation, segmentation, collation, transliteration, locale-aware case mapping, calendars, and formatting. When a Python library says "Unicode-correct", it usually means it wraps ICU.

Practical wrappers:

- **PyICU** — full ICU bindings.
- **icu4c** — C++, embedded in browsers, databases, and JVM.
- **Rust `icu4x`** — modular Rust implementation, used by Deno, ripgrep, and (as of writing) the newer Hugging Face `tokenizers` normalisers.

Two ICU features every NLP engineer should know:

- **`BreakIterator`** — the reference implementation of UAX #29. Configurable per locale (`en_US`, `th`, `ja`), and CLDR-tailored where the default rules are wrong.
- **`Transliterator`** — script-to-script conversion (`Any-Latin`, `Han-Latin`, `Katakana-Hiragana`), Unicode normalisation, and de-accenting via rule sets like `NFKC; [:Mn:] Remove; NFC`.

If you have a script- or locale-specific text problem, check ICU before you write code.

## Script-specific hazards

### CJK (Chinese, Japanese, Korean)

- **No inter-word whitespace** in Chinese/Japanese. Word segmentation requires either a segmenter (Jieba, MeCab, jieba-based, kkma) or a sub-word tokenizer that treats characters as tokens (SentencePiece with `character_coverage < 1.0` and `split_by_unicode_script=True`).
- **Han unification**: the same character can look slightly different in Simplified vs. Traditional Chinese vs. Japanese Kanji fonts. Store the language tag (`zh-Hans`, `zh-Hant`, `ja`) so rendering can pick the right font.
- **Full-width vs. half-width variants** (`Ａ` vs. `A`, `１` vs. `1`) — normalise with NFKC unless typography matters.

### Arabic (and Hebrew, Persian, Urdu, other RTL/joining scripts)

- **Bidirectional text (bidi).** Characters in Arabic runs display right-to-left, but they are stored left-to-right in logical order (as they were typed). Tokenizers should work on the **logical** stream, not the visual one. If you find yourself needing to reverse a string, stop and read [UAX #9](https://www.unicode.org/reports/tr9/).
- **Joining forms.** Arabic letters have up to four contextual shapes (isolated, initial, medial, final). These are typically presentation forms; a good source stream uses the abstract letters and lets the renderer handle shaping. Watch out for scraped PDFs, which sometimes embed presentation forms directly.
- **Diacritics (harakat).** Fully-vocalised text has short-vowel marks; most modern Arabic text does not. Decide whether your model should be diacritic-invariant (strip them in normalisation) or diacritic-sensitive (train on both).
- **Alef and hamza variants** (`ا`, `أ`, `إ`, `آ`) are often normalised to a canonical `ا` for search and classification. This loses information — do it only if downstream loss confirms it helps.

### Indic scripts (Devanagari, Bengali, Tamil, ...)

- **Virama and conjuncts.** The virama (U+094D in Devanagari, script-specific in others) suppresses the inherent vowel and forms conjuncts. `क` + `्` + `ष` renders as `क्ष`. UAX #29 grapheme clustering groups these correctly; code-point iteration does not.
- **Independent vs. dependent vowels.** Vowel signs like `ि` (Devanagari) attach to the preceding consonant. Handle them at grapheme granularity.
- **Multiple scripts per language.** Hindi is Devanagari and (in social media / romanised transliteration) Latin. Punjabi is Gurmukhi and Shahmukhi. Tag scripts explicitly.
- **ZWJ / ZWNJ** (U+200D, U+200C) — zero-width joiner / non-joiner — control conjunct formation. Do not strip them blindly; they carry orthographic meaning.

## A minimum-viable Unicode front door

Before any tokenizer sees text in a production pipeline, run at least:

1. **Decode confidently.** Assume UTF-8, but reject invalid sequences (`errors="replace"` masks bugs). Log invalid inputs.
2. **Normalise to NFC or NFKC** — decide once, apply everywhere.
3. **Detect and tag language + script** (BCP-47). Store on the record.
4. **Segment with UAX #29** (via ICU or a wrapper) if you need word or grapheme units for downstream logic.
5. **Log a small set of samples per script** so you can eyeball rendering, not just print bytes.

## Chapter summary

- Text is not a string of "characters". It is a stream of code points that group into grapheme clusters per UAX #29.
- Normalise (NFC or NFKC) at the front door of every pipeline; tokenizer training data must be normalised too.
- Tag every document with a canonical BCP-47 language + script code; language alone is not enough.
- ICU is the reference implementation for all of the above; PyICU / `icu4x` are the production wrappers to reach for.
- CJK, Arabic, and Indic each break at least one common Western-centric assumption; the hazards above are the ones most likely to bite you.
- Chapter 06 puts these ideas into practice by training a domain-adapted tokenizer and measuring what its vocabulary buys you.
