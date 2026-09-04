# Text Normalisation for ASR

## Motivation

Two transcripts that a human would call identical can produce a Word Error Rate difference of 15 points against the same reference. `"Call 911"` vs. `"call nine one one"`; `"$25"` vs. `"twenty-five dollars"`; `"Dr. Smith"` vs. `"doctor smith"`. Normalisation is the layer that decides which of those forms count as *the same* when you score, when you train, and when you compare model A to model B.

This chapter is about the *forward* direction — mapping either side (reference and hypothesis) into a canonical spoken- or minimal-orthographic form so that the comparison is meaningful. Chapter 06 covers the *inverse* — going from that canonical form back to a written form a human would read.

## What normalisation is really for

Text normalisation for ASR shows up in three distinct workflows. Treat them separately or you will keep tripping over your own rules.

- **WER-scoring normalisation.** Applied to both the reference and the hypothesis before edit-distance is computed. Its job is to make sure orthographic differences that nobody would care about (case, punctuation, hyphenation, spelled-out vs. digit numbers) do not inflate the WER. This is what the Whisper English normaliser is doing when the Whisper paper reports numbers on LibriSpeech.
- **Training-data preparation.** Text-side normalisation of training transcripts before they are used to fine-tune an ASR or a language model. The goal is a consistent, decoder-friendly vocabulary. Overly aggressive normalisation here erodes downstream capability (a model that never saw digits will never emit `911`); under-normalising leaks inconsistencies that show up as WER regressions.
- **Pre-processing text for TTS-style tasks.** The text is going *back into audio* (TTS, ASR data augmentation via TTS, forced alignment with a synthetic reference). Every abbreviation and number must be expanded to its spoken form. This is essentially the ITN problem in reverse and is where the Kestrel / NeMo TN grammars live (chapter 06).

The reference for the first two is the Whisper normaliser; for the third, WFST grammars from NeMo or research-grade Kestrel. Do not use the WER-scoring normaliser to prep training data — it is *lossy on purpose* and will collapse distinctions your ASR needs.

## The Whisper reference English normaliser

The Whisper paper's evaluation setup is public and reasonable to copy wholesale for English WER work. Two files matter:

- [`whisper/normalizers/english.py`](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py) — the `EnglishTextNormalizer` class.
- [`whisper/normalizers/basic.py`](https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py) — the `BasicTextNormalizer` used for non-English languages.

`EnglishTextNormalizer` composes several steps in order. Read the source; the summary here is a reading guide, not a spec.

- **Regex-based cleanups.** Strip HTML-like markup, strip parenthetical asides (`(inaudible)`, `[laughter]`, `<unk>`), lowercase.
- **Spelling replacements.** UK → US equivalents from a shipped `english.json` map (`colour → color`, `metre → meter`). Also expands common contractions (`won't → will not`, `you're → you are`).
- **Number expansion.** Uses `EnglishNumberNormalizer` to expand `"one"` and `"1"` and `"1st"` all into a canonical form — usually the *digit* form after unit joins (`"twenty five dollars"` → `"25 dollars"`).
- **Punctuation and casing removal.** Almost all punctuation is stripped. Ampersand → `and`. Hyphens are turned into spaces. Multiple spaces collapse to one.
- **Non-Latin scripts left alone.** The English normaliser does not touch CJK or Cyrillic — it is English-specific.

Concretely:

```python
from whisper.normalizers import EnglishTextNormalizer

n = EnglishTextNormalizer()
n("Dr. Smith called 911 for $25 worth of aspirin.")
# → 'doctor smith called 911 for 25 dollars worth of aspirin'
n("Dr Smith called nine-one-one for twenty five dollars worth of aspirin.")
# → 'doctor smith called 911 for 25 dollars worth of aspirin'
```

Both forms collapse to the same string. That is the whole point.

`BasicTextNormalizer` is the generic fallback used for non-English work: it lowercases, strips punctuation from a Unicode-aware punctuation table, normalises whitespace, and optionally splits on words vs. characters. It is deliberately conservative because doing more per-language requires per-language number expansion, and Whisper does not ship those.

## When to write your own normaliser

You should write your own — or fork the Whisper reference and layer domain rules on top — when:

- **You are evaluating on a non-English language.** `BasicTextNormalizer` is a starting point; per-language number expansion, morphological reduction, and diacritic handling are yours.
- **You are working in a specialist domain.** Medical (drug names with locale differences: `paracetamol` vs. `acetaminophen`), legal (case citations: `123 F.3d 456`), financial (ticker symbols, currency codes), scientific (units, chemical formulas). These need shipped dictionaries and canonical forms.
- **The reference transcripts follow a house style you must respect.** Many broadcast corpora ship reference transcripts that treat `[MUSIC]` and `(overlapping speakers)` as first-class tokens; a WER normaliser needs to strip or preserve these consistently on both sides.
- **You care about locale-aware casing.** German capitalises all nouns; Turkish has dotless-i / dotted-i case pairs that trip generic `.lower()`. Use ICU (`PyICU`) or the CLDR data for anything beyond ASCII lowercase.

For anything beyond ASCII, `.lower()` in Python is *not* Unicode-correct for all locales. The classic examples: Turkish `İ`.lower() gives `i̇` (with combining dot), which then round-trips wrong; German `ß`.lower() gives `ß` but `.upper()` gives `SS`. Prefer `str.casefold()` for comparison, and `PyICU`'s locale-aware case mappings for anything that has to round-trip.

## Building a normalisation pipeline you can trust

Whether you fork Whisper or start from scratch, the pipeline shape is:

```
raw text
  │
  ▼
Unicode normalisation (NFKC or NFC) — collapse compatibility forms
  │
  ▼
markup stripping — HTML, angle-bracket special tokens, square-bracket labels
  │
  ▼
casing policy (lowercase for WER; leave alone for training data)
  │
  ▼
punctuation policy (strip for WER; keep for training data)
  │
  ▼
number and unit expansion / digitisation
  │
  ▼
domain dictionary substitutions (drug names, entities, house-style spellings)
  │
  ▼
whitespace collapse
  │
  ▼
canonical form
```

Two policy decisions determine most of the outcome:

- **NFKC vs. NFC.** NFKC collapses compatibility differences (`ﬁ` ligature → `fi`; full-width digits → half-width) which is almost always what you want at WER-scoring time. NFC is safer for training data where you might want to preserve typography. Pick one, apply it everywhere upstream *and* downstream, and document.
- **Digit vs. word form for numbers.** The Whisper normaliser converts to digit form (`"nine one one" → "911"`). This is a defensible default but not the only one; some evaluation corpora convert the other way (`"911" → "nine one one"`). What matters is that *the same rule applies to both sides* of the comparison.

## Testing normalisation

Normalisation is a rule-based artefact and deserves the same test hygiene as any other rule-based artefact. A minimal test suite:

- **Round-trip on the corpus.** Take 1 000 lines of your reference corpus; normalise; normalise again; assert idempotence. If the second pass changes anything, one of your rules is not a fixed point and will bite you later.
- **Golden pairs.** A file of `(raw, normalised)` pairs covering every rule you added. When you change a rule you know exactly which golden pairs move.
- **Adversarial pairs.** Manually-curated `(raw_A, raw_B)` pairs where `raw_A` and `raw_B` are semantically identical but orthographically different. Assert `normalise(raw_A) == normalise(raw_B)`. This is the *positive* test — that your normaliser actually collapses the distinctions you claim.
- **Non-collapse pairs.** `(raw_A, raw_B)` pairs that are *not* semantically identical. Assert `normalise(raw_A) != normalise(raw_B)`. Cheap protection against over-eager rules that also collapse real distinctions.

Golden and adversarial test files live in the repo. Add to them every time a real transcript surfaces a normalisation edge case. That test file is your normaliser's real spec.

## Normalisation and WER

WER, MER, WIL are all defined against a *sequence of tokens*. Normalisation determines what a token is. Two representative gotchas:

- **Hyphenation.** `state-of-the-art` is one token or four depending on where you strip hyphens. The Whisper normaliser treats hyphens as spaces (giving four tokens); some other conventions treat them as no-ops (giving one). Different WER numbers, same underlying transcript.
- **Number expansion timing.** If you compute WER before number expansion, `"911"` vs. `"nine one one"` is three insertions and one deletion (WER catastrophe). After number expansion, they are the same string. This is why the number-expansion step lives *inside* the normaliser, not after it.

Report the normaliser version alongside every WER number. `WER=8.4% (Whisper englishnormalizer v20240930)` is meaningful; `WER=8.4%` is unfalsifiable and unreproducible. Morris, Maier & Green ["From WER and RIL to MER and WIL"](https://www.researchgate.net/publication/221489635_From_WER_and_RIL_to_MER_and_WIL_improved_evaluation_measures_for_connected_speech_recognition) (Interspeech 2004) is the canonical reference for the metric family; the current standard Python implementation is `jiwer`.

`jiwer` has a built-in `Compose` transform pipeline you can point at your normaliser:

```python
import jiwer

transform = jiwer.Compose([
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

wer = jiwer.wer(references, hypotheses,
                truth_transform=transform, hypothesis_transform=transform)
```

For anything more complex, plug your own callable in as a transform — `jiwer` is fine with that.

## Training-data normalisation is a different problem

The same file cannot be your WER normaliser *and* your training-data normaliser. The two answer different questions:

- WER-scoring normaliser: "are these two transcripts equivalent for the reader's purpose?" — lossy on purpose.
- Training-data normaliser: "is this transcript clean enough that a model trained on it will emit a consistent style?" — preserves everything the model should learn.

A concrete example: for WER you collapse `"$25"` and `"twenty-five dollars"` to the same string. For training, you decide once whether the model should emit `"$25"` or `"twenty-five dollars"` (or `"twenty five dollars"`, or `"USD 25"`) and normalise all training transcripts to *that* form. Otherwise the model learns the mixture and outputs a random pick per file.

The output-format contract (chapter 02) drives this decision; you cannot pick a training normaliser without first fixing the schema you promise to consumers.

## Bringing it together with the pipeline

The pipeline in chapter 01 has a normalisation step between the ASR and the punctuation restoration. In practice, three related things live at that stage:

1. **Verbatim clean-up.** Strip Whisper's own artefacts (leading/trailing spaces on segments; occasional decoder-emitted `<|nospeech|>` markers on very quiet segments; repeated hallucinated phrases). This is a per-model quirks list; keep it small and versioned.
2. **Case + punctuation policy.** If Whisper produced cased and punctuated text you *trust*, keep it. If you plan to run the punctuation restoration model (chapter 07) yourself, strip Whisper's punctuation and casing first so you are not fighting two models at once.
3. **ITN.** Applied *after* punctuation, because ITN grammars are usually easier to write over well-punctuated text (chapter 06).

Text normalisation for the WER-scoring loop, however, lives *outside* the pipeline — it is a scoring-time artefact, not a serving-time artefact. Keep the two normalisers separate in code even if some rules happen to overlap.

## Chapter summary

- Three normalisation jobs travel under one name: WER scoring (lossy), training-data prep (consistent), and text-for-TTS (expansive / ITN reverse). Do not share code across them.
- Whisper ships `EnglishTextNormalizer` and `BasicTextNormalizer` — the WER-scoring reference. Fork it for domain or per-language work.
- Pipeline shape: Unicode → markup strip → case → punctuation → numbers → domain dictionary → whitespace. NFKC-vs-NFC and digit-vs-word are the two policy switches that determine most of the behaviour.
- Test normalisers like production code: idempotence, golden pairs, adversarial equivalences, non-collapse negatives.
- Report the normaliser and its version alongside every WER number — an unqualified WER is unreproducible.
- Training-data normalisation is a distinct artefact from WER-scoring normalisation; the training normaliser is chosen to match the output-format contract from chapter 02.
- WER-scoring normalisation lives outside the serving pipeline; verbatim clean-up + punctuation policy + ITN live inside it.
