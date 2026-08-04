# Terminology Constraints and Glossary Injection

## Motivation

Every serious MT deployment has a glossary: brand names that must be preserved (`iPhone`, `AWS`, `Amazon Prime`), product SKUs, legal terms with a mandated translation (`whereas → considérant`, `hereinafter → im Folgenden`), medical vocabulary tied to a standard (ICD-10, MedDRA), and forbidden translations (regional slurs, competitor brand names). The base NMT model does not know your glossary. Terminology-constrained decoding is the collection of techniques that forces the decoder to obey it — sometimes softly (a hint the model can override), sometimes hard (the target token is guaranteed to appear).

This chapter walks the four options that ship in production: **prompt / template injection**, **placeholder round-tripping**, **soft constraints via lexically-constrained decoding**, and **hard constraints via prefix / disjunctive-constraint beam search**. Getting the right balance between "the term is guaranteed present" and "the surrounding grammar remains fluent" is the craft.

## The requirement, formally

Given a source $x$, a target-language glossary $\mathcal{G} = \{(t_i^{\text{src}}, t_i^{\text{tgt}})\}$, and (optionally) a set of forbidden phrases $\mathcal{F}$, produce a translation $y$ such that:

- For every $(t_i^{\text{src}}, t_i^{\text{tgt}}) \in \mathcal{G}$ that matches $x$: $t_i^{\text{tgt}}$ appears in $y$ (positively constrained).
- No phrase in $\mathcal{F}$ appears in $y$ (negatively constrained).
- $y$ remains a fluent, adequate translation of $x$ under human judgement.

The three requirements pull against each other. A strict hard constraint that inserts every term will produce ungrammatical output; a soft hint that produces perfectly fluent text will silently drop the required term. The whole game is calibrating.

## Technique 1 — Prompt / template injection (soft, model-agnostic)

Prepend the required (source, target) glossary entries to the input as an inline instruction:

```
<glossary> iPhone → iPhone ; Amazon Prime → Amazon Prime ; battery health → état de la batterie </glossary>
Translate: The iPhone shows 87 % battery health under Amazon Prime.
```

Works with any encoder-decoder or decoder-only model that has seen a similar template during pretraining — instruction-tuned models (FLAN-T5, mT5-XXL-tuned) and LLMs handle this well; raw NLLB/Marian do not respond to it. If you are on an instruction-tuned base or an LLM, this is the cheapest correct baseline.

Trade-offs:

- **No guarantee.** The model can still drop or paraphrase the term.
- **Context bloat.** Long glossaries eat context. Retrieve only the entries that actually match the source (rules- or embedding-based match).
- **Best for LLM-based MT.** For NLLB/Marian, prefer placeholder round-tripping (below).

## Technique 2 — Placeholder round-tripping (hard, but you translate around a hole)

The technique the localisation industry has used since phrase-based MT. Steps:

1. Detect glossary matches in the source and replace each with a unique placeholder token (`__TERM_0__`, `__TERM_1__`, …).
2. Translate the placeholder-hosting source. The placeholder passes through the encoder and decoder unchanged if the tokeniser preserves it.
3. Replace each placeholder in the output with the target-side glossary entry.

```python
import re

def apply_glossary_before(text, glossary):
    """Returns (masked_text, ordered_target_terms)."""
    ordered = []
    def sub(m):
        src_term = m.group(0)
        tgt_term = glossary[src_term.lower()]
        i = len(ordered)
        ordered.append(tgt_term)
        return f"__TERM_{i}__"
    pattern = re.compile("|".join(map(re.escape, sorted(glossary, key=len, reverse=True))),
                          flags=re.IGNORECASE)
    masked = pattern.sub(sub, text)
    return masked, ordered

def restore_glossary_after(translation, ordered):
    for i, tgt in enumerate(ordered):
        translation = translation.replace(f"__TERM_{i}__", tgt)
    return translation
```

Two things that will silently break:

- **The tokeniser splits `__TERM_0__` into pieces.** Verify with `tokenizer.tokenize("__TERM_0__")`. If it's ≥ 2 tokens the model may reorder or drop them. Fix: add the placeholder as a special token, or pick a token pattern the SentencePiece already keeps whole (heuristic: a rare-but-valid unicode block, e.g. `⟦0⟧`, `⟦1⟧`).
- **The model reorders placeholders during translation.** For SVO ↔ SOV pairs (English ↔ Japanese) or long sentences with multiple placeholders, the output may permute them. Restoration by *positional index* then produces the wrong term substitution. Fix: use *content-tagged* placeholders (`__TERM_iphone__`) instead of numeric indices, or check that all placeholders survived and fall back to prompt injection if any were dropped.

Placeholder round-tripping is the workhorse of production localisation. The pattern is old and unglamorous but shipping in Trados, MemoQ, and every commercial MT-as-a-service tier. Use it as your baseline.

## Technique 3 — Lexically-constrained decoding (soft-to-hard, at decode time)

Introduced by Hokamp & Liu, ["Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search"](https://arxiv.org/abs/1704.07138) (*ACL 2017*) and extended by Post & Vilar's **Dynamic Beam Allocation** ([arXiv:1804.06609](https://arxiv.org/abs/1804.06609), *NAACL 2018*). At each beam step, keep track of *which* constraints have been satisfied on each hypothesis; ensure all constraints are met by the time the beam terminates.

The `transformers` API exposes two families of decode-time constraints:

- **`PhrasalConstraint`** — require an exact contiguous token sequence to appear.
- **`DisjunctiveConstraint`** — require one of several token sequences to appear (useful for morphological variants: "run", "ran", "running").

```python
from transformers import PhrasalConstraint, DisjunctiveConstraint

# Require "état de la batterie" and "Amazon Prime" both appear.
term_1 = tokenizer("état de la batterie", add_special_tokens=False).input_ids
term_2 = tokenizer("Amazon Prime", add_special_tokens=False).input_ids

constraints = [PhrasalConstraint(term_1), PhrasalConstraint(term_2)]

out = model.generate(
    input_ids,
    constraints=constraints,
    num_beams=8,                # wider beam is required for constrained decoding
    max_new_tokens=128,
    forced_bos_token_id=tgt_bos,
    early_stopping=True,
)
```

Warnings:

- **Wider beams are needed.** Constrained beam allocates part of the beam to satisfying each constraint. `num_beams=8`–`16` is a reasonable starting point.
- **Cost scales with the number of constraints.** For >10 constraints per sentence, cost gets prohibitive. Fall back to placeholder round-tripping for long glossaries.
- **Fluency degrades when constraints are tight.** The decoder may insert the required token where it does not fit grammatically. Chapter 10 has an evaluation angle — measure both term-match accuracy *and* fluency.
- **Not all tokenisers are equal.** For subword-heavy languages (Japanese, Thai), the same surface term can have multiple valid tokenisations. Use `DisjunctiveConstraint` with each valid tokenisation.

## Technique 4 — Prefix / forced-token constraints (deterministic)

For the specific case "the translation *must start with* a given prefix," use `prefix_allowed_tokens_fn`:

```python
def prefix_allowed(batch_id, input_ids):
    # Force the translation to start with "Le rapport a établi que"
    prefix_ids = tokenizer("Le rapport a établi que",
                            add_special_tokens=False).input_ids
    generated = input_ids.tolist()[-len(prefix_ids):]  # tokens generated so far
    step = len(generated)
    if step < len(prefix_ids):
        return [prefix_ids[step]]
    return list(range(tokenizer.vocab_size))
```

Useful for structured-output MT (subtitles that must fit a template) and for legal / regulatory boilerplate. For general terminology, prefer the phrasal / disjunctive constraint family.

## Technique 5 — Training-time term biasing

You can also *train* the model to respect a glossary rather than enforcing it at decode. Two patterns:

- **Copy-attention or copy tags at training.** Annotate training targets with `<c>…</c>` around glossary-matched spans; train the model to preserve them. Later works evolved into copy-attention pointer networks (see **Pointer-Generator** references in mod-106).
- **Terminology-aware fine-tuning.** Michel & Neubig, ["Extreme Adaptation for Personalized Neural Machine Translation"](https://arxiv.org/abs/1805.01817) (*ACL 2018*) fine-tune with client-specific term overrides in the input template. The model learns to obey a `<TERM src="X" tgt="Y">` markup family.
- **DiCE-based approaches.** Dinu et al., ["Training Neural Machine Translation to Apply Terminology Constraints"](https://arxiv.org/abs/1906.01105) (*ACL 2019*) inject the target term inline in the source with a special separator so the model learns to copy it verbatim.

Training-time approaches ship the highest-quality output but require the terminology at *training time* — impractical if the glossary changes per customer or per project. Combine with decode-time or placeholder techniques for late-bound terms.

## Which technique when

| Situation                                                         | Reach for                                                        |
|-------------------------------------------------------------------|------------------------------------------------------------------|
| Glossary changes per request; must-guarantee 100 % term appearance | Placeholder round-tripping.                                       |
| LLM-based MT; small glossary; can tolerate ~90 % term appearance   | Prompt injection.                                                 |
| Fixed enterprise glossary; want fluency and term appearance both   | Training-time term biasing.                                       |
| Handful of critical terms; per-request glossary                    | Phrasal/disjunctive constrained decoding with wide beam.          |
| Regulatory boilerplate at the start of every translation           | Prefix constraint.                                                |

Most production systems stack two: placeholder round-tripping as the default plus prompt injection for LLM-based fallback, or training-time term biasing plus decode-time disjunctive constraints for high-priority terms.

## Forbidden phrases and content filters

Symmetric to positive constraints: prevent certain target strings from appearing (competitor names, slurs, dated terminology). Two options:

- **Bad-words at generation.** `bad_words_ids` in `transformers` is a per-token blocklist; supply the token IDs of each forbidden phrase (all valid tokenisations if the phrase is multi-token):

  ```python
  bad = [tokenizer(w, add_special_tokens=False).input_ids for w in banned_phrases]
  out = model.generate(input_ids, bad_words_ids=bad, num_beams=4, ...)
  ```

- **Post-hoc filter with rewrite.** Generate; check for forbidden phrases; if any match, regenerate with `bad_words_ids` plus a diversity penalty. Prefer this pattern if you also want to log filtered outputs for compliance.

Both patterns fail on paraphrase — "Coca-Cola" is trivial to block, "a well-known cola brand" is not. For truly regulatory content-filter obligations, downstream classification (mod-103) is the right place.

## Measuring terminology adherence

Automatic metrics for terminology adherence are simple and worth reporting alongside BLEU / chrF / COMET:

- **Term-match accuracy.** For each (source, target, glossary) triple, count `fraction of glossary terms whose translation appears in y`. Report the mean.
- **Term-precision.** For each glossary term that the model *did* insert, was it in the right morphological form? (Especially important for morphologically rich languages — the German dative "dem iPhone" vs. genitive "des iPhones".)
- **Fluency delta.** Compare fluency (COMET, human) of the constrained output against the unconstrained baseline. A drop of >2 COMET points is a sign the constraint mechanism is too strict.

WMT ran the "Terminology Task" in [2021 and 2023](https://www2.statmt.org/wmt23/terminology-task.html) with formal metrics — worth reading for the evaluation methodology.

## Common failure modes and their fixes

- **Placeholder was reordered and content-tagged restore fails.** Verify with a placeholder-consistency check post-translation; log the failure rate.
- **Constraint beam produces garbled output.** Beam is too narrow. Raise `num_beams` to 8+ for 1–3 constraints; 16+ for 4+ constraints.
- **Term appears but in the wrong morphology.** Use `DisjunctiveConstraint` with all valid morphological forms, or switch to placeholder round-tripping and post-morph-inflect (chapter 09 has script/morphology tools).
- **LLM prompt injection ignores the glossary.** The template is not one the model has seen — try `<term src="X">Y</term>` markup, or retrieve only the top-3 matching glossary entries to reduce noise.
- **Terminology accuracy is 100 % but human raters flag "reads mechanical".** Constraints are too tight. Try soft-scoring instead of hard-constraining — decode with a per-token *bias* toward the glossary target rather than a hard requirement.

## Chapter summary

- Terminology-constrained decoding stacks four families: prompt/template injection (LLMs), placeholder round-tripping (localisation workhorse), lexically-constrained beam decoding (`PhrasalConstraint`, `DisjunctiveConstraint`), and prefix-forcing.
- Placeholder round-tripping is the modern production default for NLLB/Marian; verify the tokeniser preserves your placeholder as one token and use content-tagged placeholders to survive reordering.
- Decode-time constraints require wider beams and can degrade fluency; measure both term-match accuracy *and* fluency delta.
- Training-time approaches (copy tags, DiCE-style term markup) give the best quality but require the terminology at training time.
- Forbidden phrases via `bad_words_ids`; paraphrase-tolerant content filtering belongs downstream, not in the decoder.
- Report term-match accuracy alongside BLEU/chrF/COMET; WMT's Terminology Task is the reference evaluation methodology.
