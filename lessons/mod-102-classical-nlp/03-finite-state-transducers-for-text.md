# Finite-State Transducers for Text Normalisation and Morphology

## Motivation

A regular expression *recognises* strings. A finite-state transducer (FST) *maps* input strings to output strings. That single generalisation — attaching an output tape to each transition in a finite-state machine — turns automata into the workhorse of morphology, phonology, and text normalisation for the last four decades of computational linguistics. Google's speech synthesis and speech recognition front-ends are FST cascades. NVIDIA's open-source `nemo-text-processing` — currently the reference implementation of production text normalisation and inverse text normalisation — is a stack of Thrax/Pynini grammars compiled to OpenFST.

If you want a normaliser that is deterministic, invertible, composable, and correct on the tail, FSTs are the tool. This chapter explains what they are, how they compose, and where they beat both regex and neural models in production.

## From automata to transducers

A finite-state acceptor (FSA) is a graph of states connected by transitions labelled with an input symbol. It accepts a string if some path from the start state to a final state consumes exactly that string.

An FST adds an *output* symbol to each transition. A path now consumes an input string and emits an output string. Two consequences:

- FSTs compute *relations* between strings. `three → 3`, `Doctor → Dr.`, and `+15551234567 → one five five five one two three four five six seven` are all FST-representable.
- Because the machine is finite, the relation is decidable in linear time in the input length.

Notationally, an FST is a 6-tuple `(Q, Σ, Δ, δ, q₀, F)` — states, input alphabet, output alphabet, transition relation, start state, final states — with each transition annotated `i:o` (input symbol : output symbol). An empty transition is `ε:ε`; a deletion is `x:ε`; an insertion is `ε:y`.

## The three operations that make FSTs powerful

Regex composes with `|` (union), `.` (concatenation), and `*` (star). FSTs compose with the same three *plus*:

- **Composition (`A ∘ B`).** If `A: X → Y` and `B: Y → Z`, then `A ∘ B: X → Z`. This is how a normaliser is built as a cascade: text → tokens → semiotic-class-labelled → verbalised → post-processed. Each stage is an FST; the whole pipeline is one FST after composition.
- **Inversion (`A⁻¹`).** Swap the input/output tapes. A TN grammar becomes an ITN grammar for free — the exact property that lets a TTS front-end (spoken form) and an ASR post-processor (written form) share a single source of truth.
- **Weighted transitions.** In a weighted FST (WFST) over a semiring (tropical, log, real), each transition has a cost. Shortest-path gives the best output; `n`-best gives ranked alternatives. This is the substrate for ASR decoding lattices and n-gram language models compiled to WFSTs.

Composition and inversion have no analogue in a regex substitution pipeline: you cannot re-order, invert, or reason globally about a chain of `re.sub` calls, but you can with FSTs.

## Two problems FSTs solve well

### 1. Text normalisation and inverse text normalisation

TTS front-ends need to convert "USD 12.30 on 07/04/2024" into a spoken form ("twelve dollars and thirty cents on the fourth of July, twenty twenty-four"). ASR back-ends need the reverse. FSTs win here because:

- The domain is a finite (large) set of *semiotic classes* — cardinals, ordinals, dates, times, currencies, addresses, phone numbers, URLs, measurements. Each has a grammar.
- Order matters: you must tokenise and classify before verbalising. Composition expresses this exactly.
- The mapping must be lossless and reviewable in both directions. WFSTs give you both.

The canonical modern implementation is NVIDIA's [`nemo-text-processing`](https://github.com/NVIDIA/NeMo-text-processing), which reimplements Sparrowhawk-era grammars in `pynini` (a Python front-end to OpenFST maintained by Google) and ships pre-built TN/ITN grammars for English, Spanish, German, Russian, Mandarin, and more. The primary paper is Zhang et al., "NeMo Inverse Text Normalization: From Development to Production", *Interspeech 2021*, [arXiv:2104.05055](https://arxiv.org/abs/2104.05055).

### 2. Morphological analysis and generation

An FST can encode a language's inflectional and derivational morphology as a bidirectional mapping between surface forms and morphemes:

```
surface: cats                 <->  lemma+features: cat+N+Pl
surface: taldemättömistäni    <->  talo+N+Ka+Neg+PxSg1+Pl+Ela   (Finnish)
surface: يَكْتُبُونَ                <->  ك ت ب+V+3+M+Pl+Impf              (Arabic)
```

Because the same FST runs in both directions, one grammar generates surface forms from a lexicon *and* analyses arbitrary surface forms into their morphemes. The reference implementations are:

- **Xerox `xfst` / `lexc`** (Beesley & Karttunen, "Finite State Morphology", CSLI, 2003). The historical standard; still used commercially.
- **`foma`** — an open-source, `xfst`-compatible engine (Hulden, EACL 2009). Standard for research and teaching. Project site: <https://fomafst.github.io/>.
- **HFST — Helsinki Finite-State Toolkit** — supports `xfst`, `lexc`, TWOL and its own APIs. Used for large open-source morphologies of Finnish, Sami, and other agglutinative languages. Project site: <https://hfst.github.io/>.

Morphological FSTs are how classical NLP still routinely beats neural lemmatisers on agglutinative and templatic languages (Finnish, Turkish, Arabic, Hebrew, Basque), where the neural model has to *learn* what an FST hand-written by a linguist can specify exactly.

## The FST toolchain, in one page

| Tool                              | What it is                                     | When to reach for it                                       |
|-----------------------------------|------------------------------------------------|------------------------------------------------------------|
| **OpenFST** (C++, Google)         | Low-level WFST library and shell tools         | Compiling large weighted grammars, ASR/TTS backends        |
| **Thrax** (Google)                | Grammar language on top of OpenFST             | Writing TN/ITN grammars declaratively                       |
| **pynini** (Google)               | Python bindings to OpenFST + Thrax semantics   | Everyday FST engineering; the current default              |
| **`nemo-text-processing`** (NVIDIA) | Pre-built TN/ITN grammars in `pynini`         | Production TTS/ASR normalisation for many languages         |
| **`foma`**                        | `xfst`/`lexc`-compatible engine (no weights)   | Morphological analysers, teaching                          |
| **HFST**                          | Multi-formalism FST toolkit                    | Large open-source morphologies                              |
| **Sparrowhawk** (Google, 2016)    | Open-source runtime for Kestrel-style grammars | Historical baseline; superseded by `nemo-text-processing`  |

Every mainstream open-source FST tool eventually compiles to OpenFST binaries. Learning `pynini` is the shortest path from "I know regex" to "I can ship an FST grammar".

## A minimal `pynini` example

The style of a TN grammar looks like this. It is deliberately verbose so the reader can see each stage of the composition.

```python
import pynini
from pynini.lib import pynutil

# Alphabet
SIGMA_STAR = pynini.union(*[chr(c) for c in range(0x20, 0x7F)]).closure()

# Simple cardinal number verbaliser: "12" -> "twelve"
digit = (
    pynini.cross("0", "zero")
    | pynini.cross("1", "one")
    | pynini.cross("2", "two")
    | pynini.cross("3", "three")
    # ... etc up to 9
)
teen = (
    pynini.cross("10", "ten")
    | pynini.cross("11", "eleven")
    | pynini.cross("12", "twelve")
    # ... etc up to 19
)

cardinal_1_19 = teen | digit

# Currency rule: "$12" -> "twelve dollars"
currency = (
    pynutil.delete("$")
    + cardinal_1_19
    + pynutil.insert(" dollars")
)

# Full normaliser (union of all classes; extend with dates, times, etc.)
normaliser = currency  # | date | time | phone | ...

# Apply to a string:
def normalise(s):
    return pynini.shortestpath(pynini.compose(s, normaliser)).string()

print(normalise("$12"))   # "twelve dollars"
```

Key idioms visible above:

- `pynini.cross(a, b)` — the transducer that maps `a` to `b`.
- `pynutil.delete(x)` — reads `x`, emits nothing (`x:ε`).
- `pynutil.insert(x)` — reads nothing, emits `x` (`ε:x`).
- `pynini.compose(input_string, transducer)` + `shortestpath` + `.string()` — the standard "apply this FST to this string" pattern.

Real grammars are hundreds of lines and organised by semiotic class. The `nemo-text-processing` [English TN grammar](https://github.com/NVIDIA/NeMo-text-processing/tree/main/nemo_text_processing/text_normalization/en) is the recommended reading once the toy example makes sense.

## When to reach for an FST instead of a regex

Signals that the problem has outgrown regex:

- **Multiple substitutions must run in a fixed, reasoned order.** Composition beats a chain of `re.sub`.
- **You need the inverse.** A TN grammar you can invert into ITN is the single biggest win.
- **The domain has classes, not just patterns.** "A currency amount consists of an optional symbol, an integer part, an optional fractional part, an optional trailing currency code" — that is a grammar, not a pattern.
- **The pipeline must be *complete*.** Any input either produces an output or is explicitly rejected — no silent no-op. FSTs give you that; ad-hoc regex chains do not.
- **Weights matter.** ASR lattice rescoring, disambiguating "1st" vs. "first", ranking alternate readings.

## Where FSTs do not help

- **Fuzzy semantics.** Sentiment, coreference, semantic role labelling. Nothing about a finite state helps here.
- **Vocabulary that grows online.** Named entities, brand names, hashtags. Rules assist but do not solve.
- **Very small problems.** A single date regex does not need Thrax.
- **Free-form generation.** FSTs are constrained by construction; that is a feature for normalisation and a limitation for generation.

## Debugging FSTs

Skills that transfer directly:

- **Round-trip test the inverse.** For any TN grammar, verify `ITN(TN(x)) == x` (up to normalisation) on a held-out sample.
- **Print intermediate lattices.** OpenFST's shell tools (`fstprint`, `fstdraw`) render a compiled FST as text or GraphViz. `pynini` FSTs expose the same via `pynini.print` and `.draw(...)`.
- **Test each semiotic class in isolation.** Compose only that sub-grammar with an input and inspect all paths (`pynini.shortestpath(..., nshortest=5)`).
- **Watch out for `sigma_star`.** Almost every TN grammar needs a "pass through anything not classified" identity FST, usually `sigma_star`. Getting this wrong produces the classic "my normaliser deleted a whole sentence" bug.

## Chapter summary

- FSTs generalise regular languages to *relations* between strings; the payoff is composition, inversion, and weighted decoding.
- Text normalisation (spoken ↔ written) and morphology (surface ↔ lemma + features) are the two production-critical use cases where FSTs still beat neural approaches on determinism, latency, and tail behaviour.
- The current default toolchain is OpenFST + `pynini`; `nemo-text-processing` ships production TN/ITN grammars for many languages, and `foma`/HFST cover morphology.
- Reach for an FST when you need ordered composition, an invertible pipeline, semiotic-class grammars, or weighted decoding. Stay with regex for one-shot patterns.
- The next chapter turns these primitives into a sentence-segmentation and lemmatisation front door that any downstream NLP component can rely on.
