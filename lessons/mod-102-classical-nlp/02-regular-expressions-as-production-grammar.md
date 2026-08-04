# Regular Expressions as Production Grammar

## Motivation

Regex is the most under-taught of the load-bearing NLP tools. Every codebase has hundreds; almost every serious bug involving one falls into a small set of predictable categories — Unicode blindness, catastrophic backtracking, greediness, or an engine mismatch between where the pattern was authored and where it runs. This chapter treats regular expressions as production grammar: how they behave under load, how they interact with Unicode, and which engine to use where.

## What "regular" actually promises

A *regular language* is anything a finite automaton can recognise. Classic regular operations — concatenation, union (`|`), Kleene star (`*`) — compose into deterministic finite automata (DFA) that accept or reject a string in `O(n)` time, where `n` is the input length, independent of the pattern.

That guarantee is real, but almost no modern regex library implements only regular operations. PCRE, Python's `re`, Java's `java.util.regex`, JavaScript's `RegExp`, .NET, and Perl all add:

- **Backreferences** (`(\w+)\1`) — not regular. Turns matching into an NP-hard problem in the worst case.
- **Lookaround** (`(?=…)`, `(?<!…)`) — still linear in most engines, but complicates cost reasoning.
- **Non-greedy quantifiers** (`.*?`) — regular, but change matching semantics.
- **Recursive patterns** (PCRE `(?R)`) — recursive rather than regular; useful for parsing balanced structures, but sits outside the finite-automaton guarantee.

The upshot: when you write a "regex" in Python or PCRE, you are writing in a *strictly larger* language than what a DFA can accept. Two engines that both accept "regex" can accept different languages, at different costs.

## The engine landscape

| Engine                                 | Semantics             | Worst case                        | Where you meet it                                   |
|----------------------------------------|-----------------------|-----------------------------------|-----------------------------------------------------|
| Python `re` / PCRE / Perl / Java / JS  | Backtracking NFA      | Exponential (catastrophic BT)     | Application code, `spaCy` Matcher token regexes     |
| Rust `regex`, RE2 (Go, C++)            | DFA / NFA, no BT      | `O(n × m)` guaranteed             | `ripgrep`, Google infrastructure, log processing    |
| Intel Hyperscan                        | Multi-pattern DFA     | `O(n)` streaming                  | High-throughput IDS/IPS, PII scanning at scale      |
| ICU regex                              | Unicode-aware NFA     | Exponential (BT), but Unicode-safe| Any pipeline that needs Unicode properties          |
| POSIX ERE / BSD `grep`                 | DFA (limited)         | Linear                            | Shell one-liners; do not use for NLP pipelines      |

Two engineering rules follow:

1. **If the pattern is trusted (you wrote it) and rich features are needed** — Python `re`, PCRE, or `regex` (the third-party PyPI package with UAX #29 support) are fine, provided you review for backtracking.
2. **If the input is adversarial or high-throughput** — use RE2 (via `google-re2` on PyPI) or Rust `regex`. Both accept only patterns whose matching is `O(n × m)`, so pathological input cannot DoS the process.

## Catastrophic backtracking, with the pattern that gets everyone

The canonical example:

```python
import re
pattern = r"^(a+)+$"
re.match(pattern, "a" * 30 + "b")   # milliseconds
re.match(pattern, "a" * 50 + "b")   # seconds
re.match(pattern, "a" * 80 + "b")   # minutes
```

A backtracking NFA tries every partition of the input into groups. Each new `a` roughly doubles the search tree, hence `O(2ⁿ)`. Real-world variants include:

```
^(.*\s)*(user|USER)\s*=\s*(.+)$      # ambiguous quantifiers
^(([^:]+):)+([^:]+)$                 # nested repetition
(a|a)+b                              # alternation-inside-quantifier
```

The pattern is: **nested or overlapping quantifiers on ambiguous alternatives**. If you see `(…+)+`, `(…|…)*`, or `.*[…]*`, treat it as a smell.

Mitigations in order of preference:

- Switch to RE2 / Rust `regex`. They cannot backtrack; problem gone.
- Rewrite to eliminate ambiguity: `^a+$`, not `^(a+)+$`.
- Use possessive quantifiers (`a++`) or atomic groups (`(?>a+)`) in PCRE / Java to disable backtracking on that fragment.
- Set an engine-level timeout (Python `regex` supports `timeout=`; Java has `Pattern.matches(..., ..., timeoutMs)`).
- In multi-tenant systems, run untrusted patterns in a sandbox with a hard CPU deadline.

Reference: Cloudflare's July 2019 outage was a single regex with catastrophic backtracking in a WAF rule. Cloudflare's post-mortem, ["Details of the Cloudflare outage on July 2, 2019"](https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/), is the definitive reading on why this matters.

## Unicode-correct regex

`.`, `\w`, `\s`, `\d`, and case-insensitive matching all have Unicode traps. Rules that consistently produce correct behaviour:

- **Use Unicode property escapes** wherever possible: `\p{L}` (any letter), `\p{N}` (any digit), `\p{Zs}` (space separators), `\p{P}` (punctuation), `\p{M}` (marks/combining), `\p{Script=Han}`, `\p{Script=Arabic}`. These are defined by the Unicode Character Database (see [UAX #44](https://www.unicode.org/reports/tr44/)).
  - Available in: PCRE, Java, JavaScript ES2018+, .NET, ICU, third-party Python `regex`, Rust `regex` (with `unicode-property` feature).
  - **Not** available in Python's stdlib `re` — this is why the third-party `regex` package exists.
- **Match on grapheme clusters, not code points.** Python's `re` treats code points as characters, so `.` will happily split a family emoji into six matches. The `regex` package supports `\X` (a grapheme cluster) and, in v2023.10 and later, UAX #29 word/sentence break properties.
- **Case-fold, do not lowercase.** Turkish `İ` and German `ß` misbehave under `str.lower()`. The right operation is `casefold()` for Python and equivalent Unicode-locale casefolding in the engine — for regex, that means `re.IGNORECASE` with `re.UNICODE` (the default in Python 3) and, for correctness on Turkic, explicit ICU casefolding *before* matching.
- **Normalise before matching.** `é` (NFC) and `é` (NFD, `e` + `U+0301`) do not match with a naïve regex. Apply NFC/NFKC at the pipeline boundary (see chapter 05 of module 101) so the regex only ever sees one form.
- **Anchor to boundaries meaningfully.** `\b` in stdlib `re` uses ASCII-only word characters unless you pass `re.UNICODE`. In the third-party `regex` package, `\b` respects Unicode by default. In text with CJK (no whitespace), `\b` is often not what you want at all — use grapheme or word segmentation from chapter 04 instead.

## Where regex is the right tool

Good uses:

- **Deterministic patterns you can enumerate.** Phone numbers, dates in a known format, currency, IPs, URLs, email boundaries, HTTP status lines, log lines, XML tags in a controlled stream.
- **Fast-path filters before an expensive model.** "Reject anything not matching `^POST /api/v1/query` before sending to the classifier."
- **Tokeniser pre-splitting.** spaCy's token infix/prefix/suffix rules, Hugging Face `pre_tokenizers.Regex`, and OpenAI's `cl100k_base` pre-tokeniser are all patterns designed to punt on ambiguous cases and let the sub-word model decide.
- **PII redaction.** With careful review, regex catches structured PII (credit cards via Luhn + regex; SSN patterns; passport formats) with recall you can characterise deterministically. Free-form names and addresses are *not* a regex problem; use NER.

Bad uses:

- **Parsing recursive structures.** HTML, JSON, SQL, Markdown, source code. Use the language's real parser. The [famous StackOverflow answer](https://stackoverflow.com/a/1732454) is only half a joke.
- **Anything that requires understanding sentence structure.** Coreference, negation, sentiment, entity boundaries in general text.
- **Any pattern whose author cannot describe the regular language it accepts in one sentence.** If you cannot state the invariant, you cannot maintain the regex.

## Patterns worth memorising

A few recurring shapes worth pinning to muscle memory:

```
# 1. URL boundary (rough)
r'\bhttps?://[^\s<>"\']+'

# 2. IPv4
r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'

# 3. ISO-8601 date (Y-M-D)
r'\b\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b'

# 4. Email boundary (do not use for validation; use for splitting)
r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'

# 5. Currency amount
r'[$€£¥]\s?\d{1,3}(?:[,\.]\d{3})*(?:\.\d{2})?'

# 6. Whitespace-normalising split (Unicode-aware in Python `regex`)
r'\p{Zs}+'
```

Each of these has known holes (IPv4 does not validate ranges perfectly; email is a hopeless standard; currency depends on locale). They are useful because their holes are known. That is what "production grammar" means: patterns with a documented boundary condition, not patterns that "usually work".

## Testing regexes like code

Two habits that pay for themselves:

- **Write tests for the boundary.** For every regex you commit, add a table of positive and negative examples. If the regex ever changes, run the table.
- **Fuzz for backtracking.** Generate strings up to some length with random content and run each with a wall-clock timeout. Any regex that ever exceeds it is a DoS surface.

Both are cheap. A `pytest.mark.parametrize` block with 20 positive and 20 negative examples per regex is enough for most production patterns.

## When to graduate to an FST

If you find yourself:

- Composing several regex substitutions in a fixed order (e.g., "normalise `USD` to dollars, then digits to words, then case-fold").
- Wanting the pipeline to be *invertible* (spoken → written and back).
- Handling morphology (lemmas, plurals, conjugations) instead of surface form.
- Needing to guarantee finite state and a deterministic worst case.

...you are outgrowing regex. The next chapter shows how finite-state transducers generalise regex to input/output pairs, and why every production text-normalisation pipeline that has to be right (ASR, TTS, healthcare identifiers) is implemented as an FST cascade.

## Chapter summary

- Regex libraries mostly implement a language *larger* than "regular"; back-references and nested quantifiers can turn matching exponential.
- Choose the engine to fit the trust and throughput of the input: PCRE-family for expressive work, RE2 / Rust `regex` when the input is untrusted or high-QPS.
- Use Unicode property escapes (`\p{…}`), grapheme clusters (`\X`), NFC/NFKC normalisation at the boundary, and `casefold()` — not `lower()` — for case-insensitive matching.
- Test regexes with positive/negative tables and fuzz for pathological backtracking.
- Regex is production grammar for enumerable patterns and pipeline pre-splitting; move to FSTs once you need composition, inversion, or morphology.
