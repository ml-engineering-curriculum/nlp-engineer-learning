# exercise-04: Constrained and Structured Generation

**Estimated effort:** 3 hours

## Objective

Build a **structured summariser** that emits a JSON record for every input article, using constrained decoding so the output is *guaranteed* to parse and match a schema. Exercise the three constraint families from chapter 08 (regex, JSON schema, context-free grammar), the three structured-output patterns from chapter 09 (typed record, bulleted list, cite-then-generate), and the four libraries a real production system would consider (`PrefixConstrainedLogitsProcessor`, Outlines, Guidance, `lm-format-enforcer`). End with a benchmark of parse-success rate, schema-conformance rate, and semantic-quality rate across constrained vs. unconstrained baselines.

## Prerequisites

- Chapters [08](../08-constrained-decoding-regex-json-cfg.md) and [09](../09-structured-output-patterns.md). Optionally [12](../12-mitigating-hallucination.md) for the "constrained decoding as a faithfulness lever" framing.
- Python 3.10+; `transformers`, `datasets`, `pydantic>=2`, `outlines`, `lm-format-enforcer`. Optionally `guidance`, `xgrammar`, `lark`.
- An instruction-tuned decoder-only or encoder-decoder LLM. Recommended: `mistralai/Mistral-7B-Instruct-v0.2`, `meta-llama/Llama-3.1-8B-Instruct`, or `google/flan-t5-large`. GPU with ≥ 16 GB VRAM.
- API-side alternative: OpenAI structured outputs (`response_format={"type": "json_schema", ...}`) or Anthropic tool-use — note in the write-up if you take this path, since it limits the "swap the library" comparison in Part D.

## Dataset

Use news or dialogue articles as the summarisation source. Pick one:

- **CNN/DailyMail** dev subset (200 examples). You have the article text; you will invent the structured target.
- **SAMSum** dev subset (200 examples). Dialogue-summarisation shape, good for the action-items pattern.
- **XSum** dev subset (200 examples). Short-target, good for the "one-line title" pattern.

Recommendation: **SAMSum for the action-items schema (Part B) and CNN/DailyMail for the typed-record schema (Part A).**

## Problem statement

### Part A — Typed record with a JSON schema (Outlines or `lm-format-enforcer`)

Design and enforce this schema using a Pydantic model:

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Union

class ArticleSummary(BaseModel):
    title: str = Field(max_length=120)
    tl_dr: str = Field(max_length=280)
    key_entities: List[str] = Field(max_length=10)
    sentiment: Literal["positive", "neutral", "negative"]
    topic: Literal["earnings", "product_launch", "acquisition",
                   "leadership", "regulatory", "policy", "sports",
                   "other"]
    publication_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")   # ISO date via regex
```

Requirements:

- Generate 200 summaries **constrained** (via `outlines.generate.json`, `lm-format-enforcer`, or your chosen library) — a valid `ArticleSummary` per input.
- Generate 200 summaries **unconstrained** (same model, same prompt, `do_sample=False, num_beams=4`) — then attempt `json.loads()` + Pydantic validation.
- Report:
  - **Parse-success rate** — did `json.loads()` succeed?
  - **Schema-conformance rate** — did Pydantic validation succeed?
  - **Regex-conformance rate** — did the `publication_date` field match the ISO regex?
  - **Enum-conformance rate** — is `topic` one of the listed values (and not "Other" / "OTHER" / "misc")?

Present as a 4-row × 2-column table (constrained vs. unconstrained).

Save as `typed_record.py`.

### Part B — Bulleted list with per-item metadata

Design an action-items schema for SAMSum:

```python
class ActionItem(BaseModel):
    owner: str = Field(pattern=r"^[A-Z][a-zA-Z]+$")
    task: str = Field(max_length=200)
    priority: Literal["low", "medium", "high", "urgent"]
    due_date: Union[str, None] = Field(pattern=r"^\d{4}-\d{2}-\d{2}$|^None$")

class MeetingSummary(BaseModel):
    tl_dr: str = Field(max_length=280)
    action_items: List[ActionItem] = Field(max_length=8)
```

Requirements:

- Generate 200 SAMSum-dev outputs under the same library as Part A.
- Report parse / schema / regex / enum conformance rates as in Part A.
- Additionally, report:
  - **Fill-rate per field** — `owner`, `task`, `priority`, `due_date`. What fraction of the constrained items get non-filler values?
  - **Empty-list rate** — fraction of outputs with `action_items = []`. (The dialogue may genuinely have no action items; distinguish "abstained legitimately" from "filler abstention".)

### Part C — Cite-then-generate with a CFG

Define a Lark grammar that requires every emitted claim to be paired with a citation from a bounded ID space. The ID space is the enumerated sentence IDs of the source (0..N-1).

Sketch:

```
?start: claim+
claim: "CLAIM:" TEXT "[SRC:" SRC_ID "]" NL
SRC_ID: /[0-9]+/
TEXT: /[^\[\n]+/
NL: /\n/
```

Requirements:

- Precompute the source's sentence-ID space. Pass the source and its enumerated sentences into the prompt.
- Generate 100 CNN/DailyMail outputs under this grammar using Outlines' `generate.cfg` or XGrammar.
- Post-hoc validation:
  - Do the emitted `SRC_ID` values fall in `[0, N)` for each source? (They must, if the CFG is enumerated over the exact ID space; if you used a generic `[0-9]+` regex, they might not.)
  - Compute a lightweight NLI entailment score (`microsoft/deberta-v3-large-mnli`, chapter 11) between each emitted claim and its cited source sentence. Report the mean entailment probability.
- Compare against a matched unconstrained baseline: same prompt, no grammar, ask the model to emit claims + citations. Report parse rate, valid-citation rate, and mean entailment.

Save as `cite_then_generate.py`.

### Part D — Library swap and cost comparison

Reuse the schema from Part A. Run it through **two** additional libraries beyond the one you picked initially. For each library, measure:

- **Compile time** (grammar / schema compile latency, one-time).
- **Per-example latency** (mean seconds/example over 100 examples).
- **Parse-success rate** (should be 100 % for any real constrained library — anything less is a library bug).
- **Semantic-quality delta** — pick 20 random examples from each library's output, compare fields side-by-side by eye. Note any library-specific quirks (e.g., a library that biases toward short strings, or one that under-populates arrays).

Suggested combinations:

- Outlines + `lm-format-enforcer` + `PrefixConstrainedLogitsProcessor` (hand-rolled).
- Outlines + XGrammar (via vLLM if available) + Guidance.

Present as a Markdown table.

### Part E — When constrained decoding hurts

From chapter 08: constraints can force the model to emit filler or the-wrong-token when the constraint is too tight or unfamiliar. Design a stress test:

1. Take the Part A schema and add a required field:

   ```python
   revenue_change_pct: float = Field(ge=-100.0, le=1000.0)
   ```

2. Run 100 CNN/DailyMail articles through this stricter schema. Manually classify the `revenue_change_pct` value for 30 random outputs into:
   - **Correct** — the article gave the value, the model captured it.
   - **Wrong** — the article gave a value, the model got it wrong.
   - **Hallucinated** — the article did not give a value, the model made one up.
   - **Filler** — the model wrote `0.0` or another placeholder because it did not know.

3. Report the four counts and discuss which failure modes are *worse* under constrained decoding than under unconstrained.

Then re-run the same 100 articles under a schema that supports abstention (add `revenue_change_pct: Union[float, Literal["unknown"]]` or a discriminated union following chapter 09's tagged-union pattern). Re-classify — did the "filler" rate go down? Did anything get worse?

### Part F — Write-up

A 600–900 word `README.md` covering:

- The three schemas from Parts A / B / C.
- The parse / schema / regex / enum conformance tables from A and B.
- The CFG cite-then-generate results and mean entailment score.
- The library comparison from D.
- The stress-test findings from E, with the abstention-channel comparison.
- One "what next" idea — e.g., fine-tuning a small model on your schema so the constraint is unnecessary, or shipping the constraint at the vLLM / SGLang serving layer via XGrammar.

## Starter guidance

- **Design the schema first, then the prompt.** The schema is the contract; the prompt is the ask. Do not overload the prompt with "please return valid JSON" — the constraint enforces that.
- **`Literal[...]` and `enum` become `select`.** Under Outlines / Guidance, they compile to a tight token mask. Overuse enums where you can afford them — every enum is a chunk of hallucination surface removed.
- **`Field(pattern=r"...")` is the regex bridge.** Use it for dates, phone numbers, IDs, currency amounts.
- **`max_length` in Pydantic is characters; `max_length` in Outlines is tokens (roughly).** Read the library docs before assuming units.
- **Instruction-tune matters.** A base model without instruction-tuning will emit garbage inside the constraint — technically valid JSON with meaningless content. Use `-Instruct` or FLAN-family checkpoints.
- **Watch for constraint compile time.** Complex CFGs compile in seconds; if you regenerate the compiled artefact per request, throughput craters. Cache compiled constraints per schema hash.
- **Beam + constraint multiplies constraint state.** Each beam maintains its own constraint state. Start with `num_beams=1` under constraint; only raise if you can afford the latency.
- **Do not benchmark parse-success on the constrained pipeline** — it is 100 % by construction. The interesting number is *semantic quality under the constraint*, not conformance.

## Acceptance criteria

- [ ] `typed_record.py` implements the Part A schema; conformance-rate table (constrained vs. unconstrained) reported on 200 CNN/DailyMail examples.
- [ ] Part B action-items schema with fill-rate and empty-list rate reported on 200 SAMSum examples.
- [ ] `cite_then_generate.py` runs a CFG on 100 CNN/DailyMail articles with valid-citation rate and mean NLI entailment against the cited source sentence.
- [ ] Library comparison across ≥ 2 additional libraries with compile time, per-example latency, and semantic-quality notes.
- [ ] Stress test with a strict numeric field: 30 manual classifications; comparison against an abstention-channel schema.
- [ ] 600–900 word write-up covering all five parts and one "what next" idea.

## Stretch goals

- **XGrammar under vLLM.** Serve the Part A schema behind vLLM with XGrammar as the grammar backend. Report throughput (requests/sec) vs. the same model served without constraints.
- **OpenAI structured outputs comparison.** Re-run Part A against `gpt-4o-2024-08-06` or `gpt-4.1` with the same Pydantic schema via `response_format={"type": "json_schema", ...}`. Report parse rate, semantic quality on 20 random samples, and cost per request.
- **Anthropic tool-use.** Same as above for `claude-sonnet-4-6` or later, using tool-use to enforce the schema.
- **Fine-tune to natively emit the schema.** Fine-tune a small model (`google/flan-t5-base`) on 500 synthetic `(article, ArticleSummary)` pairs generated by a stronger teacher model. Compare the parse-success and semantic-quality rates of the fine-tuned model **without constraints** against the base model **with constraints**. Which wins?
- **Complex CFG target.** Design a CFG for a bounded SQL-like report ("SELECT metric FROM segment WHERE date_range LIMIT n"). Generate 50 outputs; validate parseability and semantic correctness manually.
- **Grammar-of-thought.** Use a CFG to force `<reasoning>...</reasoning><answer>...</answer>` structure over the summary; compare quality of the "answer" section against an unconstrained baseline.
