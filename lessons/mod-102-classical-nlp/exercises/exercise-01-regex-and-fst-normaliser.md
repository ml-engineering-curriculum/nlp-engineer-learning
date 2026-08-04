# exercise-01: Regex And FST Normaliser

**Estimated effort:** 3 hours

## Objective

Build a small, testable text-normalisation pipeline that starts as regex substitutions and — for one semiotic class — is re-implemented as a `pynini` FST. The point is to feel the difference between "a chain of regex replacements" and "a composable, invertible FST cascade" on a task where both approaches are legitimate.

## Prerequisites

- Chapters [02](../02-regular-expressions-as-production-grammar.md) and [03](../03-finite-state-transducers-for-text.md).
- Chapter [04](../04-sentence-segmentation-lemmatisation-and-casing.md) for the front-door normalisation frame.
- Python 3.10+; `regex` (the third-party PyPI package, not stdlib `re`); `pynini ≥ 2.1` (`pip install pynini`); optionally `google-re2` for a comparison against a non-backtracking engine.

## Problem statement

Write a normaliser that converts noisy user-written text into a canonical form suitable for indexing. Your normaliser should:

### Part A — Front-door pipeline

Implement `normalise(text: str, lang: str = "en") -> str` that applies, in order:

1. **Encoding assumption + NFKC normalisation.**
2. **Whitespace collapse and trim** (Unicode-aware — use `\p{Zs}+` from the `regex` package).
3. **Case-folding** (`str.casefold()`, not `.lower()`).
4. **Structural regex clean-up** — URL, email, and phone-number placeholder replacement (e.g. `https://example.com` → `<URL>`). Use the `regex` package with Unicode property escapes.
5. **Currency normalisation** — implemented twice, once as regex and once as an FST (see Part B).

Ship a `pytest`-style test file with at least 20 positive and 20 negative examples for each substitution.

### Part B — Currency as an FST

Re-implement the currency normaliser using `pynini`. Support:

- Symbols: `$`, `€`, `£`, `¥`.
- Optional thousands separators: `1,000` and `1.000` (locale-dependent).
- Optional decimal fraction: `.30`.
- Output: `<CUR:USD:1000.30>`-style placeholders.

Compose the currency FST with the rest of the pipeline. Then produce the **inverse** FST (`<CUR:…>` → `$1,000.30`) with a *single* `pynini.invert(...)` call and demonstrate a round-trip test that passes on a held-out sample.

### Part C — Adversarial input

Feed your regex-only currency normaliser and your FST-based one at least ten adversarial inputs, including:

- Extreme repetition (`"$" * 1000 + "12"`).
- Nested numeric fragments (`$1,2,3,4.5.6`).
- Unicode look-alikes (`＄` full-width dollar, `€` in Arabic context).
- Concatenated values without separators (`$12$34`).

Report, per input, (a) what each pipeline produces, (b) how long each takes, and (c) whether the regex engine you chose exhibits catastrophic backtracking under a wall-clock timeout.

## Starter guidance

- Use the `regex` package's `\p{Sc}` (symbol, currency) escape for the currency symbol class.
- For the FST currency grammar, follow the `pynini` idioms from chapter 03: `pynutil.delete`, `pynutil.insert`, `pynini.cross`, and `pynini.compose(input_string, transducer)` + `shortestpath` + `.string()` to apply.
- Provide a `sigma_star` FST (identity over any Latin-plus-punctuation character) and compose your currency FST with it so unrelated text passes through unchanged.
- For adversarial timing, wrap regex calls in `signal.alarm` (POSIX) or run in a subprocess with a timeout.
- Log the compiled FST size (`.num_states()`, `.num_arcs()`) — if it explodes on a small grammar, you probably wrote `.closure()` where you meant `.plus()`.

## Acceptance criteria

- [ ] `normalise(...)` implemented, tested, and deterministic.
- [ ] Positive/negative example tables for every substitution, checked into the repo.
- [ ] Currency handled by both a regex pipeline and an FST pipeline; both agree on the 20+ positive examples.
- [ ] Inverse FST produces the original currency string for a round-trip test.
- [ ] Adversarial-input report includes at least one case where the regex-only pipeline degrades (either wrong output or catastrophic-backtracking timeout) and the FST behaves correctly.
- [ ] Write-up (a `README.md` in your solution folder) states, in ≤300 words, which layer of the pipeline is a rule and which is a grammar, and *why* — referencing the criteria in chapter 03.

## Stretch goals

- **RE2 comparison.** Rewrite the regex currency pipeline in `google-re2` and confirm it does not exhibit backtracking on the adversarial inputs.
- **Date FST.** Add a date semiotic class ("2024-07-04", "4th of July, 2024") and demonstrate FST composition with the currency grammar so a single compiled artefact handles both.
- **Non-English currency.** Extend to CHF, INR (with lakh/crore separators), and JPY (no decimals). Note where your FST grammar becomes locale-parameterised.
- **`nemo-text-processing` diff.** Load NVIDIA's `nemo_text_processing.text_normalization` English currency grammar and compare its output to yours on the same inputs. Explain any divergence.
