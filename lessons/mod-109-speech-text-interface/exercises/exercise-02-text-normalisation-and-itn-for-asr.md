# exercise-02: Text Normalisation and ITN for ASR

**Estimated effort:** 2 hours

## Objective

Build both sides of the normalisation stack: a WER-scoring text normaliser (chapter 05) and an inverse text normalisation grammar layer (chapter 06). Show that the two are separate artefacts, that WER scoring changes materially with normalisation applied, and that ITN produces the written form your JSON schema promises.

You will exercise the two most common workflows end-to-end: reproducing WER numbers over a public corpus with a documented normaliser, and running an ITN pass over Whisper output for at least one language.

## Prerequisites

- Chapters [02](../02-asr-outputs-whisper-and-wav2vec.md), [05](../05-text-normalisation-for-asr.md), [06](../06-inverse-text-normalisation.md).
- Python 3.10+; `whisper` (for the reference normaliser), `jiwer`, `nemo_text_processing` (for ITN), `datasets`, `faster-whisper`. NeMo Text Processing needs `pynini`; follow the [install docs](https://github.com/NVIDIA/NeMo-text-processing).
- CPU is fine. NeMo Text Processing FST compilation is slow the first time; expect several minutes for the initial ITN grammar build.

## Data

- **WER-scoring corpus.** Use LibriSpeech `test-clean` or `test-other` via `openslr/librispeech_asr` on Hugging Face, or the smaller `hf-internal-testing/librispeech_asr_dummy`. LibriSpeech reference transcripts are in clean uppercase; the exercise is calibrated to that.
- **ITN corpus.** Take 100 short clips from Common Voice English (`common_voice_16_1`, `en`) *and* transcribe your own audio-recorded reading of the ITN sanity phrases in the ITN test suite below. Common Voice references are already close to written form, so they will not exercise ITN as hard as spoken-form audio does.

Alternatively, if audio compute is a concern, skip Whisper entirely for the ITN part and feed hand-crafted `spoken-form-only` transcript strings directly into the ITN pipeline; the exercise will still hit the acceptance criteria.

## Problem statement

### Part A — WER-scoring normalisation

1. Transcribe LibriSpeech `test-clean` (a subset of ~200 utterances is fine) with `faster-whisper` `medium.en`. Save `hypotheses.txt` (one line per utterance, in original file order) and the reference transcripts in `references.txt`.
2. Compute WER *without* normalisation using `jiwer.wer(refs, hyps)`. Record `wer_raw`.
3. Compute WER *with* the Whisper `EnglishTextNormalizer` applied to both sides. Record `wer_normalised`.
4. Compute WER with a *hand-rolled* normaliser you build (see Part B below). Record `wer_custom`.
5. Report the three numbers side-by-side in `wer_report.md`, and one paragraph explaining why they differ. Reference the specific normaliser rules that account for the biggest gaps.

### Part B — Build a small custom normaliser

Author a `custom_normaliser.py` that composes:

- Unicode NFKC normalisation.
- Lowercase.
- Contraction expansion (`won't` → `will not`) for the top 20 English contractions.
- Punctuation removal (retain apostrophes inside words if you want a UK-style variant; strip otherwise — pick and document).
- Number expansion: convert digit forms to word forms (`"25"` → `"twenty five"`). Use the `num2words` library or hand-roll it for cardinals 0–99.
- Whitespace collapse.

Wrap it with a `jiwer.Compose` transform, plug it into the WER computation, and record the result in `wer_report.md`.

### Part C — Idempotence and adversarial equivalence tests

Write a `test_custom_normaliser.py` (`pytest`-runnable) covering:

- **Idempotence.** For 100 random reference lines, `normalise(normalise(x)) == normalise(x)`.
- **Adversarial equivalences.** At least 15 pairs `(raw_A, raw_B)` where both should collapse to the same output. Examples:
  - `"Dr. Smith called 911"` == `"doctor smith called nine one one"`
  - `"It's 25% off."` == `"It is twenty five percent off"`
  - `"colour"` == `"color"` (only if you added the UK→US map)
- **Non-collapse.** At least 5 pairs `(raw_A, raw_B)` where the normaliser must *not* collapse. Examples:
  - `"911"` != `"9-1-1 was busy"` (context differs semantically)
  - `"twenty two"` != `"22 percent"`

The tests must pass. This is the safety net; treat them as production tests, not a nice-to-have.

### Part D — Author ITN rules for English

Install and initialise `nemo_text_processing.inverse_text_normalization.InverseNormalizer(lang="en")`. Verify the base grammar handles the common cases:

- `"call nine one one"` → `"call 911"`
- `"twenty five dollars"` → `"$25"`
- `"the tenth of march twenty twenty five"` → `"March 10, 2025"` (or the NeMo default; pin it and document)

Now push it further. Create a `itn_test_suite.csv` with ≥ 30 `(spoken, expected_written)` pairs across at least eight semiotic classes from chapter 06 (cardinals, ordinals, fractions, percentages, currency, times, dates, addresses, phone numbers, measurements, abbreviations). Include at least 5 cases where NeMo's default output differs from your expected written form; you will decide how to handle them in Part E.

### Part E — Extend and override

Pick two of the five default-vs-expected mismatches from Part D and *fix* them. Two options:

- **Fork the NeMo grammar.** NeMo grammars are `.grm` FST rules; you can inject your own class rules by subclassing the tagger and verbaliser. See NeMo Text Processing docs for the extension pattern.
- **Post-process the ITN output.** Cheap alternative — apply a small string rewrite after NeMo, but only for the specific rewrites you own. Cheaper to implement; less general.

Document which approach you chose and why. Update your test suite to assert the fixes hold.

### Part F — Integration with a real transcript

Take three Whisper transcripts you produced in Part A (or fresh ones). Whisper English decoders typically emit already-punctuated text with some digit-form numbers. Show both:

- **What Whisper emitted directly.**
- **What your ITN produces on top of Whisper's output.**

Note where Whisper already did the ITN (and where running yours a second time would double-convert — chapter 06 discusses this). Write a paragraph in `report.md` on your policy: which cases you defer to Whisper and which you re-run.

Then run the ITN on the *spoken-form* version of a transcript (either the raw wav2vec CTC output if you have it from a stretch task, or a hand-lowercased and hand-de-ITN'd version of your Whisper output). Show the before/after and confirm the written form matches what your JSON schema promises.

### Part G — Report

`report.md` (400–600 words) covering:

- The WER-scoring numbers table (raw / Whisper-normaliser / custom-normaliser) with one paragraph of interpretation.
- A short discussion of the three biggest rule categories that moved the WER: number expansion, casing, punctuation.
- The ITN test suite pass rate, before and after your two fixes in Part E.
- A ~150-word description of your policy for double-conversion with Whisper's implicit ITN.
- Two sentences on which languages you would prioritise adding to the ITN stack next and why.

## Starter guidance

- **Start with the Whisper normaliser, not from scratch.** Reading `whisper/normalizers/english.py` end-to-end takes 20 minutes and answers 80% of the design questions before you write your own.
- **`jiwer.compute_measures` returns WER, MER, and WIL together.** Reporting all three costs nothing and catches cases where WER moved for a boring reason (many small insertions).
- **NeMo ITN's `verbose=True`** prints the intermediate tag structure. Very useful when a rule fires unexpectedly.
- **Cache the NeMo FST compilation.** Point `cache_dir` at a persistent path so you do not pay the cold-start cost every run.
- **The reference numbers on LibriSpeech in the Whisper paper are with `EnglishTextNormalizer` applied.** If your `wer_normalised` is way off the paper number, check that you resampled to 16 kHz mono and used the same model size.
- **Do not use the WER normaliser to preprocess ITN input.** They are different artefacts (chapter 05); mixing them will confuse the ITN grammar (it expects natural spoken form, not punctuation-stripped text).

## Acceptance criteria

- [ ] `hypotheses.txt` + `references.txt` from a Whisper run on ≥ 200 LibriSpeech utterances.
- [ ] `wer_report.md` with three WER numbers: raw, Whisper-normaliser, custom-normaliser, and one paragraph interpretation.
- [ ] `custom_normaliser.py` implements the composed pipeline from Part B.
- [ ] `test_custom_normaliser.py` passes with at least 15 adversarial equivalences, 5 non-collapses, and idempotence checks over 100 samples.
- [ ] `itn_test_suite.csv` covers ≥ 30 pairs across ≥ 8 semiotic classes, with ≥ 5 default-vs-expected mismatches noted.
- [ ] Two mismatches are fixed via override / grammar extension; tests confirm the fixes.
- [ ] `report.md` (400–600 words) covers WER results, ITN test-suite pass rates, double-conversion policy, and priority next languages.

## Stretch goals

- **Add a second language.** Run the ITN suite for Spanish or French using NeMo's shipped grammars. Note which classes have gaps.
- **Neural ITN comparison.** Fine-tune a small T5 (`google/flan-t5-small`) on the Google TN Challenge dataset (Sproat & Jaitly 2016). Compare its outputs on your test suite to NeMo's; count silly errors.
- **Round-trip test.** Take 1 000 lines of written text; run TN (NeMo's forward grammar) to spoken form; run your ITN. Report round-trip accuracy per semiotic class. Failures are the highest-signal bug reports.
- **CLDR-driven currency defaults.** Extend the currency class in the ITN test suite to cover 5+ locales using the CLDR data files; ensure the correct symbol and placement per locale.
- **Confusion matrix per semiotic class.** For each ITN test failure, log the *actual* written form. Aggregate into a confusion matrix (expected class → produced form) and identify the two biggest classes to invest in next.

## Deliverables

Ship as a directory:

```
data/
  librispeech_manifest.csv   # or a link + hash for reproducibility
custom_normaliser.py
test_custom_normaliser.py
itn_test_suite.csv
itn_runner.py                # invokes NeMo ITN + your extensions
itn_extensions/              # your grammar fork / post-processor
wer_report.md
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
