# exercise-02: Terminology-Constrained Decoding

**Estimated effort:** 3 hours

## Objective

Build a terminology-controlled MT pipeline that enforces a customer-glossary of brand names, product terms, and forbidden phrases across three techniques — **prompt/template injection**, **placeholder round-tripping**, and **lexically-constrained decoding** (`PhrasalConstraint` / `DisjunctiveConstraint`) — then compare them on **term-match accuracy**, **fluency (chrF / COMET)**, and **latency**. The output is a decision guide you (and the next engineer) can reach for when a new customer glossary lands.

## Prerequisites

- Chapters [02](../02-encoder-decoder-nmt-architectures.md), [05](../05-domain-adaptation-for-production-mt.md), [06](../06-terminology-constraints-and-glossary-injection.md), [10](../10-mt-automatic-evaluation-bleu-chrf-comet-bleurt.md).
- Python 3.10+; `transformers`, `evaluate`, `sacrebleu`, `unbabel-comet`, `torch`.
- One encoder-decoder MT model — `facebook/nllb-200-distilled-600M` recommended; `Helsinki-NLP/opus-mt-en-de` as a lighter alternative.
- (Optional) One instruction-tuned model — `google/flan-t5-large` or an accessible LLM — for the prompt-injection arm.

## Setup

Language pair: **English → German** (morphologically rich enough to expose fluency degradation, well-served by NLLB and Marian).

### The glossary

Create a JSON glossary of 40–60 entries covering four categories:

- **Brand / product names (~15 entries):** proper nouns that must be preserved verbatim (`iPhone → iPhone`, `Amazon Prime → Amazon Prime`, `AWS → AWS`).
- **Translated technical terms (~15):** term must be rendered as a specific target string (`battery health → Batteriezustand`, `charging port → Ladeanschluss`, `two-factor authentication → Zwei-Faktor-Authentifizierung`).
- **Legal / boilerplate (~10):** register-locked translations (`hereinafter → im Folgenden`, `whereas → in Erwägung nachstehender Gründe`).
- **Forbidden phrases (~5–10):** target strings that must **not** appear (e.g. a competitor brand name, a deprecated product name, a mistranslation your model produces repeatedly).

Save as `glossary.json`:

```json
{
  "positive": [
    {"src": "iPhone", "tgt": "iPhone", "case_sensitive": true},
    {"src": "battery health", "tgt": "Batteriezustand", "case_sensitive": false}
  ],
  "forbidden": ["Handy von Apfel"]
}
```

### The evaluation corpus

Assemble **200 English sentences** across two subsets:

- **100 glossary-heavy sentences.** Synthesise or curate sentences that each contain 1–3 glossary terms in realistic context (support tickets, product descriptions, legal boilerplate). Human-write German references that use the mandated translations.
- **100 general sentences.** Sampled from FLORES-200 devtest so you can measure fluency impact on out-of-glossary text.

Ship both as `eval.jsonl` with `{"src", "ref", "matched_terms"}` per row.

## Problem statement

### Part A — Baseline (no constraints)

Translate the 200 eval sentences with `num_beams=4` and no constraint mechanism. Measure:

- **Term-match accuracy.** For each sentence with matched glossary terms, fraction of terms whose target string appears in the output (exact string match; also report a case-insensitive variant).
- **chrF++** vs. the human references.
- **COMET** (`Unbabel/wmt22-comet-da`).
- **Latency.** Median wall-clock per sentence, batch size 1.

Save as `baseline.py`.

### Part B — Prompt / template injection

Prepend the *relevant* glossary entries to the source in an inline template. Retrieve only the entries that match the source (exact substring match, or embedding similarity for the stretch goal). Two template variants worth trying:

```
Glossary: iPhone → iPhone; battery health → Batteriezustand.
Translate English to German: <SRC>
```

vs. XML-style markup:

```
<term src="iPhone" tgt="iPhone" />
<term src="battery health" tgt="Batteriezustand" />
<sent>The iPhone shows 87% battery health.</sent>
```

Run on the eval corpus with (a) NLLB (no instruction tuning; expect weak effect) and (b) an instruction-tuned model (`flan-t5-large` or an accessible LLM). Report both.

Save as `prompt_injection.py`.

### Part C — Placeholder round-tripping

Implement the localisation-industry workhorse:

1. Detect glossary matches in the source; replace each with a placeholder token. Pick a placeholder that survives the SentencePiece tokenizer as one piece — verify with `tokenizer.tokenize(placeholder)`. Recommended: `⟦0⟧`, `⟦1⟧`, ..., or `NLLB_TERM_0` added as a special token.
2. Translate the masked source with the standard pipeline.
3. Substitute placeholders back with the target-side glossary entries. Handle the **placeholder-was-reordered** case by using content-tagged placeholders (`⟦iphone⟧`) rather than numeric indices.
4. **Fallback:** if any placeholder is dropped or reordered ambiguously, fall back to prompt-injection or unconstrained decoding and log the fallback rate.

Report term-match accuracy, chrF++, COMET, latency, and the fallback rate.

Save as `placeholder.py`.

### Part D — Lexically-constrained decoding

Use `PhrasalConstraint` (and `DisjunctiveConstraint` for terms with morphological variants) with `model.generate(constraints=...)`. Recipe from chapter 06:

- `num_beams=8` (wider beam is required for constrained decoding).
- Convert each target-side glossary entry to its tokenized form with `tokenizer(term, add_special_tokens=False).input_ids`.
- For terms with morphological variants (`iPhone`, `iPhones`, `iPhone's`), wrap them in a `DisjunctiveConstraint`.
- For forbidden phrases: pass their token-id lists as `bad_words_ids`.

Report term-match accuracy, chrF++, COMET, latency, and any observed fluency degradation.

Save as `constrained.py`.

### Part E — Comparison matrix

Produce a Markdown table comparing baseline vs. the three constraint techniques across:

| Technique | Term-match acc. | chrF++ | COMET | Median latency (ms) | Forbidden-phrase leak rate |

Bold the winner in each column. In the write-up, name which technique you would ship for:

1. A customer with a small, fixed glossary and tight latency SLA.
2. A customer whose glossary changes per request and demands 100 % term appearance.
3. An LLM-backed MT service where prompt injection is native.
4. A regulated-content pipeline where forbidden phrases must never appear.

### Part F — Failure analysis

Sample 10 sentences where at least two of the three techniques disagreed on the output. For each, classify:

- **Term inserted, grammar broken** (constrained decoding is too tight).
- **Term missing entirely** (placeholder dropped / prompt ignored).
- **Term inserted in wrong morphology** (needs a `DisjunctiveConstraint` or post-inflection).
- **Fluency issue unrelated to term.**

Paste the source, reference, and each system's output in `failures.md`.

### Part G — Write-up

500–700 word `report.md` covering:

- Setup (glossary, corpus, models).
- Comparison matrix.
- Decision guide for the four customer profiles.
- Failure-mode summary from Part F.
- One extension you would build next (training-time term biasing? content-tagged placeholders? post-morph inflector?).

## Starter guidance

- **Verify your placeholder tokenizes to one piece** — `tokenizer.tokenize("⟦0⟧")` should return a single token, otherwise the model will re-order or drop the pieces. Add as a special token with `tokenizer.add_special_tokens({"additional_special_tokens": ["⟦0⟧", ...]})` if needed.
- **`PhrasalConstraint` requires a wider beam.** Start at `num_beams=8` for 1–3 constraints, `num_beams=16` for 4+.
- **`DisjunctiveConstraint` for morphological variants.** German cases (nominative "der iPhone", dative "dem iPhone", genitive "des iPhones") need three variants.
- **Case-sensitivity matters.** `iPhone` vs. `IPhone` vs. `iphone` — decide once, in the glossary schema, and enforce end-to-end.
- **Do not tokenize the glossary target string with `add_special_tokens=True`.** You will insert a BOS token in the middle of the sentence and the constraint will never fire.
- **Latency floor.** Constrained beam with `num_beams=16` is 3–5× slower than plain beam-4. Measure it; the trade-off is the whole point.
- **Report term-match accuracy on both string-level and morphology-normalised** (lowercase + strip punctuation). The gap between the two is a signal that you need `DisjunctiveConstraint` or post-inflection.

## Acceptance criteria

- [ ] `glossary.json` with 40–60 entries across brands, technical terms, boilerplate, and forbidden phrases; `eval.jsonl` with 200 sentences (100 glossary-heavy, 100 FLORES) and human references.
- [ ] `baseline.py` reports term-match accuracy, chrF++, COMET, latency for unconstrained NLLB or Marian.
- [ ] `prompt_injection.py` reports the same for at least one model (bonus if two).
- [ ] `placeholder.py` implements round-tripping with placeholder-survival verification and a fallback path; reports the same metrics plus fallback rate.
- [ ] `constrained.py` uses `PhrasalConstraint`/`DisjunctiveConstraint` with `num_beams>=8`; reports the same metrics.
- [ ] Comparison matrix in `report.md` with the four customer profiles and a shipped-technique recommendation.
- [ ] `failures.md` with 10 disagreement cases, each classified.
- [ ] `report.md` (500–700 words).

## Stretch goals

- **Embedding-retrieved glossary.** Instead of exact-substring matching, embed source sentences with a multilingual sentence encoder (LaBSE) and retrieve the top-K nearest glossary entries. Report how retrieval quality affects term-match.
- **Training-time term biasing.** Fine-tune the model with `<term src="X" tgt="Y">` markup in the source (DiCE-style; chapter 06). Compare against decode-time constraints on the same eval.
- **Post-morphology inflector.** For German, run the constrained output through a rule-based or model-based inflector to fix case-agreement issues in inserted glossary terms.
- **Content-filter arm.** Add a post-hoc regeneration pass when a forbidden phrase appears (regenerate with the phrase in `bad_words_ids`). Report the leak rate before and after.
- **Cross-language sweep.** Repeat for English → Japanese (SVO ↔ SOV; placeholder reordering is more likely) and comment on which technique degrades most.

## Deliverables

```
glossary.json
eval.jsonl
baseline.py
prompt_injection.py
placeholder.py
constrained.py
failures.md
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
