# Structured Output Patterns for Summarisation

## Motivation

Chapter 08 gave the mechanism — regex, JSON schema, CFG — for guaranteeing a structured output shape. This chapter is about the *design* question that comes before the mechanism: when your product asks for "a structured summary", what is the actual schema you commit to, and how do you organise fields so the model produces useful values instead of gaming the format?

The design decisions collected here recur in almost every production summariser:

- Free-form paragraph or typed record?
- One flat schema or nested?
- Cite the source per claim, per sentence, or per record?
- How does the model say "I don't know"?
- How does the model refuse when nothing salient exists?

Each of these has a common wrong answer that produces a technically-valid summary the downstream system cannot rely on.

## Four common structured-summary patterns

### Pattern 1: typed record

A flat JSON object with typed fields. The most common shape for downstream extraction:

```json
{
  "title": "Q3 revenue beats estimates by 8%",
  "summary": "AcmeCo reported Q3 revenue of $1.2B, up 12% YoY, exceeding analyst estimates of $1.11B. Growth was driven by the enterprise segment.",
  "key_entities": ["AcmeCo", "Q3", "$1.2B"],
  "sentiment": "positive",
  "topic": "earnings"
}
```

Guarantees: fields present, types correct, enums respected. Great for feeding a database, a dashboard, or a downstream classifier.

Failure mode: the model gets the *shape* right but the *content* wrong (invented entities, mismatched sentiment). Structure does not imply faithfulness — chapter 11 stacks on top.

### Pattern 2: bulleted list with per-bullet metadata

For meeting summaries, incident reports, and change logs, the natural shape is a list of items with structured metadata per item:

```json
{
  "action_items": [
    {"owner": "Alice", "task": "Draft migration RFC by 2026-08-15", "priority": "high"},
    {"owner": "Bob", "task": "Roll back the staging cache", "priority": "urgent"}
  ]
}
```

Guarantees the same as pattern 1 per bullet. Bounds the model's output length by the number of bullets — a natural throttle.

### Pattern 3: cite-then-generate

Every emitted claim is paired with a source span or source ID:

```json
{
  "claims": [
    {"text": "Revenue rose 12% YoY.", "citation": "para_3_sentence_1"},
    {"text": "Enterprise segment drove growth.", "citation": "para_4_sentence_2"}
  ]
}
```

This is the pattern for compliance-sensitive domains. The citation string is validated post-generation: does `para_3_sentence_1` exist in the source, and does it actually entail the claim (via NLI, chapter 11)?

Two implementation choices:

- **Grounded generation.** The model emits both text and citation in one pass, either via a fine-tune on citation-annotated data or via a schema that forces the citation before the text.
- **Cite-then-verify.** The model emits text; a separate pass aligns each claim to a source span; unaligned claims are flagged for review.

Both work. Grounded generation is stronger when the fine-tuning data exists; cite-then-verify is the fallback when it does not.

### Pattern 4: hierarchical / multi-section summary

A schema with named sections:

```json
{
  "one_line": "Q3 beat with strong enterprise growth.",
  "tl_dr": "Revenue $1.2B (+12% YoY), enterprise-led, guidance raised.",
  "detailed_summary": "AcmeCo reported...",
  "risks": ["Renewal rate ticked down.", "China revenue flat."]
}
```

Different sections have different length budgets, faithfulness thresholds, and audiences. A one-line title tolerates aggressive compression; a "risks" section must not invent risks. Schema-level metadata (per-field constraints) is the natural place to encode this.

## Designing the schema: five design rules

Rules that keep schemas useful over time:

1. **Every field has a *purpose*.** If a downstream system does not consume it, do not put it in the schema. Optional fields become required fields that no one enforces.
2. **Fields with bounded values are enums.** "sentiment" is `Literal["positive", "neutral", "negative"]`, not an open string. This is what constrained decoding is for.
3. **Required means required.** Do not make everything optional to "give the model flexibility". You will regret the missing-field handling downstream.
4. **Include an abstention channel.** A `"cannot_summarise": true` or a `"missing_fields": ["risks"]` field. Never make abstention implicit in emitting an empty string — the empty string is a valid value in most schemas, and you cannot distinguish "no risks" from "I could not find any".
5. **Reserve a `notes` field for freeform overflow.** The model will sometimes have relevant content that does not fit the schema. Giving it a `notes` field is better than watching it stuff the content into an unrelated typed field.

## Length budgets per field

Fields have different natural lengths. Enforce them at the schema level, not with a global `max_length`:

```python
class Summary(BaseModel):
    title: str = Field(max_length=120)             # ~one line
    tl_dr: str = Field(max_length=280)             # tweet-sized
    detailed_summary: str = Field(max_length=1500) # paragraph
    action_items: List[str] = Field(max_length=8)  # up to 8 items
```

Under Outlines / Pydantic, `max_length` is enforced by the generation state machine. Under a hand-rolled `PrefixConstrainedLogitsProcessor`, you have to enforce it yourself.

Note the field-specific budgets: they tell the model *implicitly* how much detail each field wants. A `title: str = Field(max_length=12)` is a signal to produce a headline; `detailed_summary: str = Field(max_length=1500)` is a signal to produce a paragraph.

## Enums and controlled vocabularies

Any categorical field — sentiment, topic, urgency, language, priority — should be an enum:

```python
Sentiment = Literal["positive", "neutral", "negative"]
Priority  = Literal["low", "medium", "high", "urgent"]
Topic     = Literal["earnings", "product_launch", "acquisition",
                    "leadership", "regulatory", "other"]
```

Two design decisions inside the enum:

- **Include an `other` catch-all.** Every taxonomy is incomplete. Without a catch-all, the model has to lie.
- **Include the *absence* value if that is meaningful.** For a sentiment enum, `neutral` is real. For a priority enum, `none` might be needed.

The `other` category also gives you a monitoring signal: if 30 % of your outputs are `topic: "other"`, your taxonomy has drifted from the corpus.

## Citations: the design decisions

If citations are part of your product, decide these before writing the schema:

- **Granularity.** Whole document, sentence, span, or noun phrase.
- **Format.** Numeric ID, source URL, `(doc_id, offset_start, offset_end)`, or full quoted text.
- **Multiplicity.** One citation per claim, or a list.
- **Validation.** Fail-closed (schema requires the citation to be a real ID and validation rejects unknown IDs) or fail-open (citation is a hint, validated post-hoc).

The most robust choice for compliance-sensitive domains: sentence-level granularity, `(doc_id, sentence_id)` tuples, single citation per claim, fail-closed with the ID space enumerated in the constraint. The trade-off: the model can only cite sentences you gave it; if the salient information is not in your provided sentences, it must abstain.

## Abstention as a first-class value

Every structured schema for summarisation needs an abstention channel. Common shapes:

```python
class Summary(BaseModel):
    title: Optional[str] = None
    tl_dr: Optional[str] = None
    missing_reason: Optional[str] = None
```

Or an explicit boolean:

```python
class Summary(BaseModel):
    can_summarise: bool
    summary_or_reason: str    # summary if can_summarise, reason otherwise
```

Or a tagged union (Pydantic discriminated union):

```python
class SuccessfulSummary(BaseModel):
    kind: Literal["ok"] = "ok"
    summary: str

class Abstain(BaseModel):
    kind: Literal["abstain"] = "abstain"
    reason: str

Result = Union[SuccessfulSummary, Abstain]
```

The tagged union is the strictest and best. Constrained decoding will make the model *choose* which arm of the union it is in; each arm then has its own required fields. No implicit "I don't know" hiding in an empty string.

## Multi-step generation: retrieve → summarise → structure

A common production pattern is three passes:

1. **Retrieve or extract** the relevant source spans.
2. **Summarise** them freeform (chapters 03–05).
3. **Structure** the freeform summary into the schema via constrained decoding.

The rationale: forcing a model to hit a JSON schema and produce a good summary *simultaneously* is harder than doing each separately. Splitting the pass lets each step use its natural strategy — beam search for the freeform pass, constrained beam for the structuring pass.

Costs: two model calls. Latency, tokens, complexity.

The alternative — one-pass, constrained-decoded summary generation with a good instruction-tuned model — works when the schema is simple and the model is strong. For complex schemas with nested cite-then-generate patterns, the two-pass shape is more reliable.

## Common failure modes

- **Filler in required fields.** The model writes "N/A" or "TBD" in fields it does not know. Add an abstention channel and monitor the filler rate.
- **Over-broad enums.** If `topic: "other"` is >20 % of outputs, refactor the taxonomy or accept that the field is not useful.
- **Hallucinated citations.** The model emits `sentence_id: 42` that does not exist. Fail-closed validation catches it; a constraint that enumerates valid IDs prevents it.
- **Repeated bullets.** Especially in action-item schemas. Add a semantic dedup pass or fine-tune with a diversity term.
- **Schema-satisfaction over content quality.** The output parses but is worthless. Structural evaluation must sit alongside content evaluation — chapter 10.

## Chapter summary

- Structured summarisation is a design problem before a decoding problem. Commit to a schema whose fields exist because a downstream system consumes them.
- Four patterns: typed record, bulleted list with metadata, cite-then-generate, hierarchical multi-section.
- Design rules: every field has a purpose, bounded values are enums, required means required, abstention is a first-class channel, keep a `notes` overflow.
- Enforce per-field length budgets and use enums for anything categorical.
- Citations: decide granularity, format, multiplicity, validation. Fail-closed with enumerated IDs is the strictest and best for compliance.
- Prefer explicit tagged-union abstention over implicit "empty string" abstention.
- Two-pass generation (summarise, then structure) is often more reliable than one-pass constrained generation for complex schemas.
