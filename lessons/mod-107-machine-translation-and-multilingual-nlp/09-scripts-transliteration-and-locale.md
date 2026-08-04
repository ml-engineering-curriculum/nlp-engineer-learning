# Scripts, Transliteration, and BCP-47 Locale

## Motivation

The multilingual model shipped fine on the demo; production started routing Arabic requests and half of them come back with reversed digits, mangled ligatures, or output in the wrong dialect of Chinese. This chapter walks the script-, normalisation-, and locale-handling issues you must budget for. None of it is glamorous. All of it will silently corrupt your outputs if you skip it — sometimes only for the users who write in a script other than Latin, which is exactly the tail your evaluation set is least likely to catch.

We cover the four script families that produce the most engineering surprises — **Arabic**, **CJK**, **Indic**, and (briefly) **Thai / Ethiopic / Vietnamese** — plus the standards you will lean on (Unicode normalisation, BCP-47 locale, ISO 15924 script codes) and the transliteration tools worth knowing.

## The standards you build on top of

Two standards do the heavy lifting; the rest are conveniences:

- **Unicode normalisation** — [Unicode Standard Annex #15](https://unicode.org/reports/tr15/). Any user-facing text should be normalised, usually to **NFC** (composed) form. The four normal forms:
  - `NFC` — canonical decomposition then canonical composition. The default output form; visually equivalent characters collapse.
  - `NFD` — canonical decomposition only. Useful for accent stripping, diacritic analysis.
  - `NFKC` — compatibility decomposition then composition. Collapses width variants (fullwidth digits → ASCII digits), ligatures, styled numerals. Aggressive; can lose information you care about.
  - `NFKD` — compatibility decomposition only.

  Rule of thumb: **normalise all training data and all inputs to `NFC`** at ingestion; use `NFKC` only when you specifically want to fold compatibility variants (search indexing, name matching).

- **BCP-47** — the IETF standard for language tags. Format is `language[-script][-region][-variant]`, e.g. `en-GB`, `zh-Hans-CN`, `sr-Latn-RS`, `az-Cyrl-AZ`. Two things to internalise:
  - **The script subtag is `Aaaa` (four letters, initial-cap)** from **ISO 15924**: `Latn`, `Cyrl`, `Arab`, `Hans` (Simplified Han), `Hant` (Traditional Han), `Deva` (Devanagari), `Ethi` (Ethiopic), `Thai`.
  - **The region subtag is `AA` (two letters, uppercase)** from **ISO 3166-1 alpha-2**. `GB` not `UK`; `MX` not `MEX`.
  - NLLB's language codes (`eng_Latn`, `zho_Hans`) are almost BCP-47 but use ISO 639-3 for the language part and an underscore separator. mBART's codes (`en_XX`) are entirely their own.

Reach for [`langcodes`](https://github.com/rspeer/langcodes) (Python) when you need to normalise, compare, or extract from BCP-47 tags. Do not parse tags manually — the grammar is surprisingly subtle.

## Arabic, Hebrew, and other bidi scripts

Arabic and Hebrew read **right-to-left**. Digits and Latin loanwords inside a bidi paragraph read left-to-right. The Unicode Bidirectional Algorithm (**UAX #9**) handles this; you almost never implement it, but you *must not fight it*.

Things that go wrong:

- **Storing bidi-reordered text.** Your database receives an Arabic string that the client rendered right-to-left; when you retrieve and re-render it, the letters are reversed. Fix: store *logical* order (the order the user typed characters), let the renderer handle visual order. Never call `str[::-1]` on Arabic text to "fix" the display.
- **Digit shape.** Arabic can use Western digits (`0-9`), Arabic-Indic digits (`٠-٩`, U+0660–), or Extended Arabic-Indic digits (`۰-۹`, U+06F0–, used in Persian/Urdu). NFKC folds them all to ASCII; NFC preserves. Decide per product whether you want the fold.
- **Ligatures and presentation forms.** Arabic letters have four positional forms (isolated, initial, medial, final). Unicode encodes them in the "Arabic Presentation Forms" blocks (U+FB50, U+FE70). If your source text uses presentation forms, normalise to the base characters — NFKC does this. Otherwise a search index will treat "presentation-form-of-alif" and "alif" as different tokens.
- **Diacritics (tashkeel).** Optional short-vowel marks — أَبْجَدِيَّة. Most NMT models drop them; if your product needs them (Quranic text, dictionaries), either preserve at both ends or use a diacritisation model (**Farasa**, **MADAMIRA**, **CAMeL Tools**) as a post-process.
- **Modern Standard Arabic vs. dialects.** MSA (`arb_Arab`) is the written standard; Egyptian (`ary_Arab`), Moroccan (`ary_Arab`), Levantine, Gulf, and others are spoken variants that appear in social media. NLLB-200 covers several formally; do not assume MSA-trained models handle dialect input well.

For MT specifically: **NLLB has separate codes per dialect** (`arb_Arab`, `acm_Arab` for Mesopotamian, `apc_Arab` for Levantine, etc.). Use them; do not pool all "Arabic" input into MSA.

## CJK: Chinese, Japanese, Korean

The three scripts that share the "CJK" umbrella are typographically different problems.

### Chinese

- **Simplified (`Hans`) vs. Traditional (`Hant`).** Two separate script standards; ~2 000 character-level differences. Simplified is used in mainland China and Singapore; Traditional in Taiwan, Hong Kong, Macau. **These are different NLLB targets.** `opencc` (or the [`OpenCC`](https://github.com/BYVoid/OpenCC) library) is the standard tool for converting between them.
- **Word segmentation.** Written Chinese has no space between words. Every downstream task that assumes tokens (chrF character-mode ok, BLEU word-mode broken) needs word segmentation. Standard tools: **`jieba`** (Python, fast, dictionary-based), **`pkuseg`**, **spaCy Chinese**. For evaluation, SacreBLEU's `--tokenize zh` flag handles it.
- **Punctuation.** Chinese uses full-width punctuation (`，。！？`) distinct from ASCII. NFKC folds fullwidth to ASCII; NFC preserves. Decide per product.

### Japanese

- **Three scripts, one language.** Hiragana (curvy phonetic), Katakana (angular phonetic, mostly loanwords), and Kanji (borrowed Han characters). All three appear in normal text.
- **Word segmentation.** Same problem as Chinese; **`MeCab`** with a UniDic or IPAdic dictionary is the reference. `fugashi` is the Python wrapper. SacreBLEU has `--tokenize ja-mecab`.
- **Kanji ambiguity.** Kanji can have multiple readings; some are context-dependent. Only matters if you are producing furigana / romaji.
- **Half-width Katakana** (U+FF60 block) exists for legacy reasons. NFKC folds to full-width Katakana; you almost always want this.

### Korean

- **Hangul syllables.** Composed of jamo (letter units) into syllabic blocks. Two representations in Unicode: **precomposed syllables** (U+AC00–D7AF; the normal storage form) and **jamo** (U+1100–). NFC folds jamo sequences to precomposed syllables. Always normalise Korean to NFC.
- **Word segmentation.** Space-separated (like English) but morphology is agglutinative. `khaiii`, `KoNLPy`, or `soynlp` for morphological analysis.

### Han unification

CJK characters are unified in Unicode — the same "code point" can render as slightly different glyphs in Simplified Chinese, Traditional Chinese, Japanese Kanji, and Korean Hanja fonts, and the "correct" glyph depends on the language tag of the surrounding text. Do not try to fight this; propagate the language tag through your rendering pipeline (HTML `lang` attribute, PDF language metadata) and let the font handle it.

## Indic scripts

Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Odia, Sinhala — each with its own Unicode block. Common issues:

- **Virama sequences.** Consonants combine into conjuncts via the virama (halant) mark. `क + ् + ष = क्ष`. NFC composes; NFD decomposes. Always NFC for storage.
- **Matra ordering.** Vowel signs (matras) attach around consonants. Historically some fonts store them in visual order (e.g. `i-matra` before the consonant even though phonetically it comes after). Unicode always stores in *phonetic* order — verify your source data does the same. Text mistakenly stored in visual order looks fine on-screen but sorts and tokenises wrong.
- **Zero-width joiners.** ZWJ (U+200D) and ZWNJ (U+200C) control conjunct rendering. Presence or absence changes meaning. Never strip them "for hygiene."
- **Nukta.** Some scripts modify base consonants with a subscript dot (`ड़` = `ड` + `़`). Two visually-identical characters can be nukta-composed or not. NFC composes; if you accept mixed input, normalise.
- **Multiple scripts per language.** Hindi is written in Devanagari; Urdu (mutually intelligible) in Perso-Arabic. Sanskrit historically has been written in many scripts. Punjabi has both Gurmukhi (Indian) and Shahmukhi (Pakistani). Always tag script explicitly in language codes.

Tooling: **[indic-nlp-library](https://github.com/anoopkunchukuttan/indic_nlp_library)** covers normalisation, tokenisation, and script conversion. **[AI4Bharat](https://ai4bharat.iitm.ac.in/)** publishes IndicTrans2 and multiple Indic-focused resources.

## Thai, Lao, Khmer, Burmese

Scripts with **no word boundaries** at all. Word segmentation is a research problem, not a config flag.

- **Thai.** `pythainlp` for tokenisation. SacreBLEU has `--tokenize ja-mecab` but nothing built-in for Thai — the community usually pre-tokenises with `pythainlp` before scoring.
- **Khmer, Lao, Burmese.** Similar. Look up per-language tokenisers; MT quality is often bottlenecked here.

## Transliteration

Converting between scripts for the same underlying language (Serbian Cyrillic ↔ Latin, Japanese Kanji → Romaji, Hindi Devanagari → Latin). Two families of technique:

- **Rule-based transliterators.** [`unidecode`](https://pypi.org/project/Unidecode/) for a lossy Latin transliteration of anything; [`transliterate`](https://pypi.org/project/transliterate/) for language-specific bijections; `openCC` for Traditional ↔ Simplified Chinese (technically not transliteration but conversion); `pytransliterate` for Indic.
- **Model-based transliterators.** For high-quality use cases (search across scripts, romanising names for a database key), fine-tune a small seq2seq (mBART-tiny or ByT5-small) on parallel transliteration pairs. **NEWS** (Named Entities Workshop) shared tasks and **Dakshina** dataset (Roark et al., ["Processing South Asian Languages Written in the Latin Script"](https://arxiv.org/abs/2007.01176), *LREC 2020*) are the standard sources.

Transliteration is a *lossy* operation for most script pairs — Devanagari to Latin loses vowel length and retroflex distinctions; Cyrillic to Latin loses some Slavic-specific sounds. Never round-trip through a lossy transliteration and expect the result to be the original.

## BCP-47 locale-aware evaluation

Two things you should have opinions about before shipping multilingual eval:

- **Match locale precision to your data.** If your training data has `es-MX` and `es-ES` labelled separately, keep them separate in evaluation. Averaging over "Spanish" masks Latin American vs. Iberian differences that users notice.
- **Evaluate on FLORES-200 with the *exact* NLLB script code.** FLORES treats `zho_Hans` and `zho_Hant` as separate targets. Report against both if your production ships to both regions.

For number, date, and currency formatting (which any product UI adjacent to MT will need): rely on **ICU** (International Components for Unicode) or the **CLDR** database. Do not roll your own — locale-aware formatting is a several-decade project and is finished.

## A defensive preprocessing pipeline

A skeleton you should adapt per language:

```python
import unicodedata

def normalise(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text)

def preprocess_source(text: str, lang_tag: str) -> str:
    text = normalise(text, "NFC")

    if lang_tag.startswith("ar") or lang_tag.startswith("fa"):
        # Preserve digit shape by default; strip tashkeel if the model was trained on undiacritised MSA.
        text = strip_tashkeel(text)  # only if training data was undiacritised

    if lang_tag.startswith("zh"):
        # Optional: force Simplified/Traditional based on target locale
        # text = opencc_convert(text, "s2t")   # or "t2s"
        pass

    # Do NOT lowercase for Turkish (dotted/dotless I) or Azerbaijani without locale-aware casing.
    if lang_tag.startswith("tr"):
        text = locale_aware_lower(text, "tr")

    return text
```

Two invariants worth asserting:

- **After preprocessing, the text still round-trips through `unicodedata.normalize("NFC", ...)` unchanged.** Any preprocessing step that produces non-NFC output is a bug — write a test.
- **The tokeniser produces at least one token per non-empty input.** Empty tokenisation almost always means an unsupported script and byte-fallback silently dropping.

## Case: Turkish and Azerbaijani dotted/dotless I

The classical example of locale-aware casing. Turkish and Azerbaijani have both `I` (dotless) and `İ` (dotted); the lowercase of `I` is `ı`, and the lowercase of `İ` is `i`. Python's default `str.lower()` is not locale-aware.

```python
"İstanbul".lower()           # 'i̇stanbul' — WRONG (adds a combining dot)
"İstanbul".casefold()        # 'i̇stanbul' — WRONG
# Correct:
from icu import Locale, UnicodeString
s = UnicodeString("İstanbul").toLower(Locale("tr_TR"))  # 'istanbul' ✓
```

Applies to any pipeline that lowercases before comparison (search index, deduplication key). If you do not use PyICU, at minimum branch on `tr` / `az` locales and apply the Turkish-specific casing rule.

## Common failure modes and their fixes

- **Arabic text is stored reversed.** The producer applied a bidi reordering. Store in logical order; render at display time.
- **NFKC dropped a script variant your product cares about.** Switch to NFC. NFKC is a fold; use it deliberately.
- **A German comma-decimal is parsed as a US thousands separator.** Never parse locale-formatted numbers with a fixed regex. Use ICU / CLDR-backed parsers.
- **Simplified-trained model gets Traditional Chinese input.** Convert with `opencc` before feeding, or fine-tune per script.
- **Vietnamese tone marks disappear in output.** Almost always a Unicode-composition bug — the source text was in NFC-precomposed form and something in the pipeline decomposed it to NFD and then dropped the combining marks.
- **Hebrew or Arabic search returns nothing on user query.** Presentation-form vs. base-form mismatch between index and query. Normalise both to NFKC before indexing.

## Chapter summary

- **Normalise to NFC at ingestion.** Use NFKC only when you specifically want compatibility folds (search index, deduplication).
- **BCP-47 is the locale standard.** Use ISO 15924 script subtags (`Latn`, `Hans`, `Deva`, `Arab`) and ISO 3166-1 alpha-2 region subtags. Reach for `langcodes` (Python) rather than parsing manually.
- **Arabic:** bidi ordering, digit shape, dialect codes (`arb_Arab`, `arz_Arab`, `apc_Arab`), diacritics.
- **CJK:** Simplified vs. Traditional (`opencc`); word segmentation (jieba / MeCab / KoNLPy); halfwidth/fullwidth; Han unification via language tags.
- **Indic:** virama and conjunct composition; matra ordering; ZWJ/ZWNJ preservation; nukta normalisation; script-per-language explicitness.
- **Thai / Khmer / Lao / Burmese:** no word boundaries — language-specific tokenisers required.
- **Transliteration:** rule-based (`unidecode`, `opencc`) for keys; model-based (Dakshina, NEWS) for user-facing.
- **Turkish/Azerbaijani dotted/dotless I** is the canonical locale-aware casing example; branch or use PyICU.
- Assert Unicode normalisation invariants in tests; script bugs are hard to catch by eye.
