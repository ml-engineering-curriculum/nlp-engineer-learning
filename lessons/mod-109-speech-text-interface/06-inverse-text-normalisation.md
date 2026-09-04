# Inverse Text Normalisation

## Motivation

An ASR model transcribing `"nine one one"` from audio is doing its job. Shipping `"nine one one"` in the transcript your dispatcher reads is a bug. Inverse Text Normalisation (ITN) is the step that maps *spoken form* — how a word is said — into *written form* — how the same word is expected to appear on the page: `nine one one` → `911`, `twenty-five dollars` → `$25`, `march tenth two thousand twenty five` → `March 10, 2025`.

This chapter is about building and integrating that mapping. Chapter 05 was about the forward direction; ITN is the reverse and is a distinct engineering artefact.

## The problem is not "write regexes"

The temptation with ITN is to sit down and write a few regexes and be done. This works for two hours and then breaks on the first real recording. The reasons it breaks:

- **Context matters.** `"two thousand"` becomes `"2000"` when it refers to a year, `"2,000"` when it refers to a quantity, `"$2000"` when preceded by `"dollars"`, and `"2:00 AM"` when the following word is `"a m"`. No regex composes cleanly across these.
- **Ambiguity is unavoidable.** `"one hundred and fifty"` can be `150` or `100.50`. Whichever policy you pick, other speakers will surprise you.
- **Order dependencies.** Currency wraps numbers, but numbers wrap date components, but dates wrap ordinals. Naive left-to-right regex substitution builds up the wrong intermediates.
- **Localisation is a fresh problem for every language.** `"twenty-five thousand"` and `"veinticinco mil"` and `"vingt-cinq mille"` all mean the same thing but require different grammars. Chinese and Japanese have myriad forms (`两千`, `二千`, digit + `千`) and no unified spec.

Reflecting these constraints, the field consolidated around two architectures: WFST grammars and neural seq2seq. Real production systems use one, the other, or a hybrid.

## WFST-based ITN (the NeMo recipe)

The dominant open-source approach is weighted finite-state transducers, first popularised for TTS text normalisation by Kestrel (Ebden & Sproat, ["The Kestrel TTS text normalization system"](https://www.cambridge.org/core/journals/natural-language-engineering/article/kestrel-tts-text-normalization-system/F0C18A3F596B75D83B75C479E23795DA), 2015) and ported to open source by NVIDIA in [`nemo_text_processing`](https://github.com/NVIDIA/NeMo-text-processing).

The core idea: an ITN grammar is a composition of small transducers, each responsible for one *semiotic class* — cardinals, ordinals, dates, times, currencies, addresses, abbreviations, measurements. Each transducer maps spoken form → written form on its class and passes everything else through. A top-level classifier tags each span of the transcript with its class, then the class-specific transducer rewrites it. A verbaliser (or in the NeMo pipeline, a token-level FST) assembles the final string.

The workflow:

```
"i'll be there at eleven thirty a m tomorrow"
       │
       ▼   (classifier tags: TIME("eleven thirty a m") wraps the numeric span)
       │
       ▼   (time transducer: "eleven thirty a m" → "11:30 AM")
       │
       ▼
"I'll be there at 11:30 AM tomorrow"
```

Concretely, using NeMo:

```python
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

itn = InverseNormalizer(lang="en", cache_dir=".cache/nemo-itn")

itn.inverse_normalize("i'll be there at eleven thirty a m tomorrow", verbose=False)
# → "I'll be there at 11:30 AM tomorrow"
itn.inverse_normalize("call nine one one", verbose=False)
# → "call 911"
itn.inverse_normalize("twenty five dollars", verbose=False)
# → "$25"
```

Properties that matter for production use:

- **Deterministic.** Same input → same output, every time. Regression tests are trivial.
- **Auditable.** Every rewrite corresponds to a specific transducer arc. When the output is wrong you can point to a specific grammar line.
- **Language coverage is patchy.** NeMo's ITN ships production-grade grammars for English, Spanish, German, French, Russian, Chinese, Vietnamese, Hindi, and a growing tail. Anything outside that list is "you write it, we host it."
- **Grammar authoring is not free.** A production-quality ITN grammar for a new language is a multi-week effort by someone with linguistics chops. Do not schedule a new locale as a one-week ticket.
- **Compilation is slow the first time.** NeMo compiles the FST graph on first use and caches it. Point `cache_dir` at a persisted location; a cold start is minutes.

The reference paper for the NeMo pipeline is Zhang, Bakhturina, Gorman & Ginsburg, ["NeMo Inverse Text Normalization: From Development to Production"](https://arxiv.org/abs/2104.05055), Interspeech 2021.

## Neural seq2seq ITN

The alternative — treat ITN as a sequence-to-sequence problem and train a T5/BART-like model on `(spoken, written)` pairs — is tempting because you avoid grammar authoring entirely. It comes with a specific failure mode you must understand before you deploy it.

Sproat & Jaitly (["RNN Approaches to Text Normalization: A Challenge"](https://arxiv.org/abs/1611.00068), 2016) framed the problem and released the Google TN Challenge dataset; Zhang, Sproat, Ng, Stahlberg, Peng, Gorman & Roark (["Neural Models of Text Normalization for Speech Applications"](https://direct.mit.edu/coli/article/45/2/293/95508/Neural-Models-of-Text-Normalization-for-Speech), *Computational Linguistics* 45(2), 2019) is the deeper analysis. The takeaway: neural TN/ITN produces mostly-correct output with occasional *silly errors* — hallucinated numbers, dropped digits, wrong dates. These errors are semantically catastrophic (a `£25` bill emitted as `£250` is worse than `£25` emitted as `twenty-five pounds`) even when their frequency is low.

If you go this route, the paper's recommended mitigation still stands: constrain the model with an FST-based coverage grammar. Generate candidates neurally, filter through the FST, fall back to identity on rejects. This is a hybrid architecture; you end up authoring an FST grammar anyway, just a smaller one.

The practical decision matrix:

| Setup | Approach |
|-------|----------|
| Language covered by NeMo, English-adjacent domain | NeMo ITN. Fork the grammars for domain vocab. |
| Language covered by NeMo, specialist domain | NeMo ITN + a light domain post-processor. |
| Language not covered by NeMo, high-value | Author a NeMo-style WFST grammar. Multi-week effort. |
| Language not covered by NeMo, exploratory | Neural seq2seq + FST guard. Accept the silly-error rate; monitor. |
| LLM-in-the-loop pipeline | Use the LLM to ITN as a whole task; test the hallucination rate carefully; keep an FST guard for critical classes (currency, dates). |

## Semiotic classes: the checklist

When scoping ITN for a language, walk the class checklist and decide the target written form for each. Missing a class here means the transcript is inconsistent for that class forever.

- **Cardinals** — `nine hundred and eleven` → `911` (locale-specific: comma vs. dot for thousands separator).
- **Ordinals** — `first`, `1st`, `21st`. Includes month days (`the tenth of March`).
- **Fractions and decimals** — `three point one four` → `3.14`; `two thirds` → `2/3` or `two-thirds`.
- **Percentages** — `twenty five percent` → `25%`.
- **Currency** — `twenty five dollars` → `$25`; locale-dependent symbol placement (`25 €` in French, `€25` in English).
- **Time** — `eleven thirty a m` → `11:30 AM`; 24-hour vs. 12-hour is a policy call.
- **Dates** — `march tenth twenty twenty five` → `March 10, 2025`. Locale-dependent ordering (`10/03/2025` vs. `03/10/2025`).
- **Addresses** — `one two three main street` → `123 Main Street`. Also house numbers, apartment numbers, postal codes.
- **Phone numbers** — `four one five five five five one two one two` → `(415) 555-1212` or `+1 415 555 1212`. Locale-dependent formatting.
- **Measurements** — `twenty degrees celsius` → `20°C`; unit abbreviation policies.
- **Abbreviations** — `doctor smith` → `Dr. Smith`; `united states` → `US` (or `U.S.`); `f b i` → `FBI`.
- **Acronyms with letters** — `n a s a` → `NASA` (if pronounced as a word) vs. `f b i` → `FBI` (if spelled out). This split is language-specific and needs a lookup table.
- **Email and URLs** — rarely spoken cleanly; usually left as spelled-out.
- **Emojis and special symbols** — usually out of scope; sanity-check that they do not slip into the transcript from your own ASR post-processing.

Pull the target written form from CLDR (the [Unicode CLDR data files](https://cldr.unicode.org/)) for the target locale wherever possible. UTS #35 ([Unicode Technical Standard #35: Locale Data Markup Language — Numbers / Dates / Units](https://www.unicode.org/reports/tr35/tr35-numbers.html)) is the standards reference for numeric and date formatting per locale. This is what iOS, Android, and every serious i18n library use; align with them and you get downstream consumer libraries for free.

## Integrating ITN with punctuation

There is a chicken-and-egg question: does ITN come before or after punctuation restoration? Both have arguments.

- **Punctuation first, then ITN.** Punctuation gives the ITN grammar structure to work with: a full stop after `two thousand twenty five` disambiguates it as a year sentence-final; a comma between `"eleven"` and `"thirty"` blocks the time reading. This is the recommended default and what NeMo assumes.
- **ITN first, then punctuation.** Sometimes cheaper if the punctuation model was trained on ITN'd text (which it often is, because the underlying corpora are already in written form). This can produce cleaner punctuation on numeric content.

The mainstream production pipeline is `ASR → verbatim cleanup → punctuation and casing → ITN`. Chapter 07 assumes this ordering when discussing punctuation.

Watch out for the case where Whisper *already* did ITN in its decoder (which happens because Whisper's training data has both spoken and written forms of numbers). If you then run your own ITN over Whisper's output you get double-conversions (`"25 dollars"` treated as words and rewritten to `"25 25 dollars"`). Either:

- Trust Whisper's ITN and skip your own for English; run yours only for languages where Whisper's is weak (mostly the long tail).
- Force Whisper's decoder toward spoken form (temperature > 0 sometimes helps, though this is unreliable) and always run your own ITN.

The second is cleaner if you can afford it; the first is often what you end up doing.

## Testing ITN

Same test-hygiene story as chapter 05. Two additional patterns matter specifically for ITN:

- **Class coverage matrix.** For every semiotic class, keep at least a dozen `(spoken, written)` golden pairs spanning the corner cases (compound numbers, negatives, fractions with cardinals, currency across scales). Assert `itn(spoken) == written`.
- **Round-trip vs. training data.** For a large sample of your reference transcripts (already in written form), *reverse-normalise* them into spoken form using your TN grammars (chapter 05 covers TN briefly; NeMo also ships forward TN), then apply ITN and assert the result matches the original. Round-trip failures are the highest-signal bug reports.

NeMo ships evaluation scripts (`nemo_text_processing/inverse_text_normalization/run_evaluate.py`) that operationalise this on the Google TN Challenge test set. Point them at your custom rules while you develop.

## Failure modes to expect

- **Over-eager class classifier.** The tagger classifies `"one plus one is two"` as `1 + 1 = 2` when the input was a description of a maths exercise. Fix: constrain classifiers on context (`"is"` between two cardinals is usually not equality); or leave it alone and rely on punctuation.
- **Digit-string preservation lost.** `"the meeting id is one two three four five six"` should stay as `"the meeting id is 123456"` (digit string, not the number 123 456). Some grammars will render it `"123,456"`, which is wrong for identifiers. Guard with a rule for `"id"`, `"code"`, `"number"` context; keep unformatted digit strings.
- **Currency ambiguity.** `"twenty five hundred dollars"` in English is `$2,500` but is not spoken that way in most other locales. Grammar must reject the pattern in the wrong locale.
- **Unit typos.** `"twenty degrees c"` becomes `20°C`; `"twenty degrees see"` becomes `20 degrees see` or crashes. Fix: robust homophone map in the pre-ITN pass, or a small acoustic-context rule.
- **Whisper's own casing conflicting with ITN's casing.** Whisper capitalises `"March"` on its own; ITN then treats `"March tenth"` as raw text (miscased) and misses the date. Fix: casefold before ITN; re-case after.
- **Numbers straddling segment boundaries.** VAD-based chunking may split `"one hundred and"` from `"twenty five"`. Fix: run ITN on the merged sentence, not on individual segments; or run it on segments and post-merge in a second pass.

## Chapter summary

- ITN maps spoken form to written form: `nine one one` → `911`. It is where transcripts become usable for humans and downstream systems.
- Regexes alone do not scale — semiotic-class conflicts and localisation force a grammar-based approach.
- The dominant open-source approach is WFST via NeMo (`nemo_text_processing`). Deterministic, auditable, but language coverage is patchy; new languages are multi-week efforts.
- Neural seq2seq ITN removes grammar authoring but hallucinates rare silly errors (dropped digits, wrong dates). Constrain with an FST guard.
- Walk the semiotic-class checklist for the target locale. Pull target forms from CLDR / UTS #35 rather than inventing them.
- Standard pipeline order is `ASR → cleanup → punctuation → ITN`. Watch for Whisper's implicit ITN causing double-conversions in English.
- Test ITN with class-coverage golden matrices *and* round-trip against your reference corpus.
- Common failures: over-eager classifiers, lost identifier digit strings, currency-locale mismatch, homophone unit typos, boundary-split numbers. Guard each one explicitly.
