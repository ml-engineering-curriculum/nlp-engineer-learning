# exercise-03: Unicode Segmentation and Normalisation Audit

**Estimated effort:** 2 hours

## Objective

Audit a "clean" text pipeline for the Unicode hazards from chapter 05 and produce a written report that a review board could act on. The point is to convert Unicode standards from abstract references into a checklist you actually run before shipping any tokenizer.

## Prerequisites

- Chapter [05](../05-unicode-correct-text-processing.md) (UAX #29, UAX #15, BCP-47, ICU).
- Python 3.10+, one of `PyICU`, `grapheme`, or `uniseg`; the `unicodedata` stdlib module.
- Optional: [`langid`](https://pypi.org/project/langid/) or [`fasttext` language identification model](https://fasttext.cc/docs/en/language-identification.html) for the BCP-47 tagging step.

## Problem statement

You are handed a modestly-sized multilingual text corpus and asked to check whether it is safe to feed to a tokenizer trainer. Produce a Unicode audit report covering the four dimensions below.

### The corpus

Build (or receive) an audit corpus of ~50-200 sentences that deliberately includes:

- Latin text with combining marks (`é`, `ñ`, `ü` — some in NFC, some in NFD).
- One CJK sample (Chinese, Japanese, or Korean).
- One RTL sample (Arabic or Hebrew), including at least one diacritic (harakat).
- One Indic sample with a conjunct (e.g. `क्ष`, `क्त`, `স্ব`) and a ZWJ / ZWNJ character.
- At least one emoji sequence, ideally with a ZWJ family or skin-tone modifier (e.g. `👩‍💻`, `👨🏽‍🚀`).
- One string with a "ligature" compatibility character (`ﬁ`, `½`, `²`).
- One string with a hidden problem: a mixed-script confusable (`раураӏ` — Cyrillic pretending to be Latin), or a leading BOM, or a ZWNBSP (U+FEFF).

You can hand-author these into a single file; the exercise is about the analysis, not corpus scale.

### The audits

For each of the four dimensions, produce a table (or code + printout) that reports the finding per sentence.

#### 1. Encoding sanity

- Decode every input as UTF-8 with `errors="strict"`. Flag any input that fails.
- Report the byte length, code-point count, and grapheme-cluster count (via `grapheme.length` or ICU `BreakIterator`) per sentence. Show at least one sentence where the three numbers differ substantially and explain why.
- Detect and log any control characters (`Cc` category), BOMs, or bidi controls (`RLM`, `LRM`, `RLO`, `LRO`). These are usually bugs.

#### 2. Normalisation

- For each sentence, compute the four normal forms (`NFC`, `NFD`, `NFKC`, `NFKD`) with `unicodedata.normalize`.
- Report, per sentence, whether the sentence is already in each form, and where it differs, show a diff-friendly rendering (byte lengths, first differing code points).
- Recommend one normalisation form for the pipeline and justify the choice. Consider: web-form input, ligature preservation, Arabic tatweel, Han unification.

#### 3. Segmentation

- Use UAX #29 (via ICU `BreakIterator` or a wrapper) to segment each sentence at:
  - grapheme-cluster boundaries;
  - word boundaries;
  - sentence boundaries (compare to the input which is one sentence per line).
- Compare grapheme-cluster segmentation to `list(sentence)` (naïve code-point iteration) and show a case where they disagree — especially for the emoji sequence and the Devanagari conjunct.
- For the CJK sample, discuss what "word" means and how UAX #29 handles it by default.

#### 4. Language and script tagging

- Run a language identifier on each sentence. Log the raw output.
- Convert it to a canonical **BCP-47** tag (language + script + region where meaningful). Use the ISO 15924 script codes (`Latn`, `Cyrl`, `Hans`, `Hant`, `Arab`, `Deva`, `Beng`, `Zyyy` for undetermined).
- Explicitly flag any sentence where the language identifier's script guess is wrong (e.g. mixed-script confusable).

### The report

Produce a single markdown file (`REPORT.md`) with:

- One paragraph per audit dimension summarising findings.
- The tables from each audit inline.
- A **remediation checklist** — the concrete pipeline steps (normaliser, segmenter, language tagger, mixed-script guard) you would add before this corpus is safe to train on.

## Starter guidance

- If you have not installed ICU before, `pip install PyICU` on Linux typically requires the `libicu-dev` system package. `uniseg` is a pure-Python fallback for the segmentation part.
- The `unicodedata.name(char)` function is your friend for figuring out what a suspicious code point actually is. Print it beside the raw glyph.
- For the mixed-script confusable check, group by `unicodedata.script` (or ICU's `Script.getScript`) per grapheme cluster; sentences with more than one non-common script deserve a manual look.
- Use [`langcodes`](https://pypi.org/project/langcodes/) or ICU `ULocale` for BCP-47 canonicalisation — hand-rolling the case rules is a source of bugs.

## Acceptance criteria

- [ ] Audit corpus meets all diversity requirements (Latin+combining, CJK, RTL, Indic, emoji ZWJ, ligature, at least one hazard).
- [ ] Encoding sanity table exists and correctly reports byte / code-point / grapheme counts, with an explanation of at least one three-way disagreement.
- [ ] Normalisation table shows the four normal forms per sentence and a recommendation with justification.
- [ ] Segmentation output demonstrates a grapheme-cluster vs. code-point disagreement on both the emoji sequence and the Indic conjunct.
- [ ] Every sentence is tagged with a canonical BCP-47 tag (`language[-script][-region]`); mixed-script confusables are flagged.
- [ ] `REPORT.md` includes the remediation checklist and could plausibly be handed to a reviewer without further explanation.

## Stretch goals

- **Bidi correctness.** Apply the UAX #9 bidirectional algorithm to the Arabic / Hebrew sample using ICU's `Bidi` API. Show the logical order vs. the visual order and explain why the tokenizer must see the logical order.
- **Fingerprinting.** Compute a canonical fingerprint per sentence (NFKC + case-fold + collapse-whitespace + script-transliterate) and use it to find near-duplicates in a larger corpus of your choice.
- **Language-ID vs. script disagreement in the wild.** Run your audit on a real slice of a public multilingual corpus (a few thousand rows of OSCAR, mC4, etc.). Quantify the disagreement rate.
- **Custom transliteration.** Use ICU's `Transliterator` (rule `"NFD; [:Mn:] Remove; NFC"`) to strip diacritics from the Latin sample. Discuss when this is appropriate and when it is destructive.
