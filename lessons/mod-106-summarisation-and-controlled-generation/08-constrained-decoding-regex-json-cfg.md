# Constrained Decoding: Regex, JSON Schema, and Context-Free Grammars

## Motivation

The knobs from chapter 07 (`bad_words_ids`, `PhrasalConstraint`) are enough to force a word to appear or to suppress a token. They are not enough to force the output to be a valid JSON object with a specific schema, to match a regex, or to conform to an EBNF grammar. Those are the actual asks in every production system that consumes model output downstream: extraction pipelines, tool-calling APIs, structured summaries, code generators.

Unconstrained generation followed by a post-hoc validator is the tempting shortcut. It fails for two reasons: (a) even a 1 % invalid-output rate translates to visible errors in a production dashboard; and (b) retry-until-valid burns tokens and latency without addressing the underlying problem. The fix is *constrained decoding* — modifying the decoding step itself so that the model can *never* emit a token that would break the constraint.

This chapter covers the three constraint families you will actually use — regex, JSON schema, context-free grammars — and the four libraries that implement them: `PrefixConstrainedLogitsProcessor` in `transformers`, Outlines, Guidance, `lm-format-enforcer`, and XGrammar.

## The mechanism, in one line

At every decoding step, before sampling / argmax, mask out any token whose selection would violate the constraint. That is it. The rest is engineering: how to compute the valid-tokens mask fast, how to represent the constraint, how to handle regex/CFG/schema uniformly.

The mask is a length-$V$ vector of $\{0, -\infty\}$ added to the logits before the softmax. Combined with any decoding strategy (greedy, beam, nucleus), it guarantees:

- The output is a valid string under the constraint.
- The output is the model's best guess *given* the constraint.

The extra cost is a constraint-state update per step. For a regex it is a DFA transition; for a CFG it is an LR/GLR shift; for a JSON schema it is a JSON-parser state advance. Modern libraries compile the constraint into an FSA or trie once and then run the update in microseconds per step.

## The hand-rolled version: `PrefixConstrainedLogitsProcessor`

`transformers` ships a low-level hook that lets you specify, per step, which tokens are allowed:

```python
from transformers import PrefixConstrainedLogitsProcessor

def prefix_allowed_tokens_fn(batch_id, input_ids):
    # Return the list of token IDs that are allowed at this step.
    return allowed_ids_for(input_ids)

outputs = model.generate(
    input_ids,
    num_beams=4,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```

This is the primitive that every higher-level library reduces to. It is fine for small constraints (a fixed vocabulary, a short regex), painful for anything with real structure (JSON, CFG). Use it when the constraint is a one-off; reach for Outlines / Guidance / XGrammar for anything real.

## Regex-constrained generation

The simplest structured constraint. Compile the regex to a DFA, track the state after every generated character, and at each decoding step mask out tokens that would leave the DFA in an invalid state.

Two tricky bits:

- **Tokens don't align to characters.** A single token might contain characters that span a DFA transition. Modern implementations handle this by keeping a per-token *state map* — for each candidate token, the DFA state you would end up in if you emitted it.
- **Termination.** The regex has "accepting" states. You want the model to be able to emit `</s>` only when the DFA is in an accepting state. This is handled by masking `</s>` unless the state is accepting.

Outlines is the reference implementation:

```python
import outlines
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

# ISO date
generator = generate.regex(model, r"\d{4}-\d{2}-\d{2}")
print(generator("What is today's date?"))
# → "2024-11-21"

# Alphanumeric ID
generator = generate.regex(model, r"[A-Z]{2}-\d{6}")
print(generator("Give me an order ID:"))
# → "US-402193"
```

Regex-constrained generation is the right tool for:

- ISO dates, phone numbers, currency amounts, IDs.
- Fixed-vocabulary classification outputs.
- URL/UUID/hex-string formats.
- Constraining a summariser to a subset of the source vocabulary (chapter 12's extractive-guarantee trick).

## JSON-schema constrained generation

The most common ask in production. Given a JSON schema, generate a JSON object that (a) parses, (b) matches the schema's required keys and types, (c) respects enum values, and (d) respects string patterns (which are themselves regexes).

The mechanism reduces to a state machine over JSON: track whether we are inside a key, a value, an array, an object; at each step, the allowed tokens are those that keep the JSON parser and the schema state consistent.

Outlines with a Pydantic model or a raw JSON schema:

```python
from pydantic import BaseModel, Field
from typing import Literal, List

class Summary(BaseModel):
    title: str = Field(max_length=120)
    key_points: List[str]
    sentiment: Literal["positive", "neutral", "negative"]
    source_ids: List[int]

generator = generate.json(model, Summary)
result = generator("Summarise this article: ...")
# result is a Summary instance, guaranteed to parse.
```

Guidance:

```python
import guidance
from guidance import gen, select

@guidance
def summary(lm, article):
    lm += f"Article: {article}\n\n"
    lm += '{"title": "' + gen(name="title", stop='"', max_tokens=30) + '",'
    lm += ' "sentiment": "' + select(["positive", "neutral", "negative"], name="sentiment") + '"}'
    return lm
```

`lm-format-enforcer`, XGrammar, and OpenAI's structured-outputs API (which uses a compiled CFG under the hood) all offer the same guarantee: the returned string parses to a JSON object matching the schema.

Two things that JSON-schema constrained generation gives you for free:

- **Guaranteed parseability.** No `try/except` around `json.loads`.
- **Type safety.** Numbers are numbers; enums are enums; required fields are present.

Two things it does *not* give you:

- **Semantic correctness.** The model can still emit a wrong `sentiment` for the article. The schema only constrains form, not truth.
- **Faithfulness.** Structured summaries can still hallucinate fields. Chapters 11–12 apply on top.

## Context-free grammar (CFG) constrained generation

For anything more expressive than a regex — nested structures, SQL, code, DSLs, or bespoke output formats — a CFG is the right tool. The mechanism is the same as JSON: compile the grammar (typically in EBNF or Lark syntax), maintain a parser state, mask tokens that would break parseability.

Outlines with a Lark grammar:

```python
grammar = r"""
?start: statement+

statement: "SUMMARY:" TEXT | "ACTION:" TEXT | "OWNER:" NAME

NAME: /[A-Z][a-z]+/
TEXT: /[^\n]+/
"""

generator = generate.cfg(model, grammar)
print(generator("Summarise this meeting transcript: ..."))
```

XGrammar (Dong et al., 2024) is a fast alternative optimised for high-throughput serving; `llama.cpp`, `vLLM`, and `SGLang` are integrating it as their default grammar backend. XGrammar precompiles the grammar to a compressed FSA that lets it run the mask update in nanoseconds per step.

CFGs are the right tool for:

- Domain-specific output formats (report templates, radiology structured findings, legal briefs).
- Code / SQL generation with syntactic guarantees.
- Any output where JSON is too loose but hand-rolled regex is too rigid.

## Trade-offs among the libraries

| Library                   | Regex | JSON schema | CFG | Backend integration                                              | Notes                                                              |
|---------------------------|-------|-------------|-----|------------------------------------------------------------------|--------------------------------------------------------------------|
| `transformers` `PrefixConstrainedLogitsProcessor` | Manual | Manual | Manual | Native `transformers.generate`                              | Low-level primitive. Bring your own constraint compiler.           |
| **Outlines**              | ✔     | ✔ (Pydantic / JSON) | ✔ (Lark) | `transformers`, `vLLM`, `llama.cpp`, `MLX`                        | Reference implementation. Broad backend support.                    |
| **Guidance**              | Partial | ✔          | Partial | `transformers`, OpenAI, others                                    | Programmatic prompt/response templating; more prompt-language-ish. |
| **`lm-format-enforcer`**  | ✔     | ✔           | ✔   | `transformers`, `vLLM`, `TGI`                                     | Fast, minimal dependencies.                                        |
| **XGrammar**              | ✔     | ✔           | ✔   | `vLLM`, `SGLang`, `llama.cpp`                                     | High-throughput serving. Compressed grammar FSA.                    |
| **OpenAI structured outputs / Anthropic tool-use** | Regex-lite | ✔ | Limited | Proprietary                                             | Provider-side implementation. No local model support.               |

Recommendations:

- **Prototyping, research, or local-model deployment.** Outlines. Broadest coverage, cleanest API.
- **Production serving at scale.** XGrammar (via vLLM / SGLang). Best throughput.
- **API-only workflows.** OpenAI structured outputs or Anthropic tool-use. JSON schema is native.
- **When you need programmatic prompt templating alongside constraints.** Guidance.

## When constrained decoding hurts

Constrained decoding is not free. Three known costs:

- **Latency.** The mask update adds a per-step cost proportional to constraint complexity. For a simple regex or JSON schema, negligible; for a large CFG, a real fraction of decoding time.
- **Constraint gaming.** The model may satisfy the form of the constraint by emitting content-free filler ("N/A", empty arrays, "unknown") when it does not know. Distinguish "cannot answer" from "does not know" with an explicit abstention channel — a `sentinel_none` value in the schema, or an `"abstain": true` field.
- **Quality regressions on rare formats.** If the schema is unusual and the model has never seen anything like it in pretraining, constrained decoding will produce syntactically-valid but semantically-broken outputs. The fix is fine-tuning, not more constraint.

The pathological case is a constraint so tight that the *only* valid token at some step is the wrong one. The model has no way out. If your constraint set is that tight, either loosen the constraint or fine-tune the model to natively produce the target format.

## Combining constrained decoding with other strategies

Constrained decoding composes with all of chapter 06's strategies:

- **Constrained beam search.** Deterministic and structure-guaranteed. Standard for JSON/table outputs.
- **Constrained sampling.** Introduces variance while keeping structure. Useful for MBR-style multi-sample generation over a schema.
- **Constrained + `no_repeat_ngram_size`.** Compose naturally.

`num_beams > 1` with a constraint can be *slower* than expected because each beam maintains its own constraint state — beam width multiplies constraint-update cost. Benchmark before shipping.

## Chapter summary

- Constrained decoding masks out invalid tokens at every step, producing outputs guaranteed to match a regex, a JSON schema, or a CFG.
- The `PrefixConstrainedLogitsProcessor` primitive in `transformers` is the low-level hook. Outlines, Guidance, `lm-format-enforcer`, and XGrammar are the higher-level libraries; pick by backend and throughput needs.
- Regex handles dates, IDs, and fixed formats. JSON schema handles typed records. CFG handles anything more expressive — DSLs, code, template-heavy report formats.
- Constrained decoding gives parseability and type safety for free; it does not give semantic correctness or faithfulness. Chapters 11–12 stack on top for those.
- When constraints are too tight for the model's knowledge, it will game them with filler. Add an explicit abstention channel — do not overload structure to encode "I don't know".
