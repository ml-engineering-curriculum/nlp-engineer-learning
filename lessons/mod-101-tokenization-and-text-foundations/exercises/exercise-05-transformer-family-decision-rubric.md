# exercise-05: Transformer Family Decision Rubric

**Estimated effort:** 2 hours

## Objective

Turn chapter 07's decision rubric into a reusable artefact: a written rubric plus worked case studies. By the end of this exercise, you should be able to defend an architectural choice (encoder-only vs. encoder-decoder vs. decoder-only) to a review board with concrete reasoning, not habit.

## Prerequisites

- Chapters [07](../07-transformer-families-from-the-nlp-perspective.md) and [08](../08-kv-cache-attention-shape-and-sequence-budgets.md).
- Familiarity with Hugging Face `transformers` model classes is helpful for the empirical portion.

## Problem statement

The exercise has two halves: build the rubric, then apply it to five realistic scenarios and one empirical probe.

### Part A — Build the rubric

Produce `RUBRIC.md` containing:

1. **A decision tree** with clear branch questions (task type, conditioning input, latency budget, in-context learning need, training-data availability). Refine chapter 07's four questions into something you would actually hand a teammate.
2. **A comparison table** of the three families across at least these axes:
   - attention mask shape and information flow;
   - canonical pretraining objective;
   - typical model sizes / open-weights examples;
   - inference cost per generated token (prefill vs. decode);
   - KV-cache behaviour (chapter 08);
   - strengths, weaknesses, and one common misuse.
3. **A red-flag list** — three or more warning signs that a chosen family is wrong for the task (e.g. "the input is 3 sentences and the output is 3 words → you probably do not need a decoder-only 70B model").

### Part B — Case-study application

Apply the rubric to each of the following five scenarios. Pick a specific family (encoder-only, encoder-decoder, or decoder-only) *and* a size class (small ≤ 500M params, mid 1-15B, large ≥ 30B), and defend both choices in 4-8 sentences citing the rubric. Note explicit trade-offs.

1. **Ticket-router.** Classify inbound support tickets into 40 categories. Latency budget 50 ms p95. 500k historical labelled tickets. Target QPS 200.
2. **Legal contract summariser.** Produce a 200-word summary of a 40-page contract (~30k tokens after tokenisation). Batch overnight — latency does not matter. No labelled summaries; only raw contracts + a few hundred hand-written examples.
3. **On-device grammar checker.** Detect and correct grammatical errors in typed messages on a mobile phone. No cloud round-trip allowed. Ships on devices with 8 GB RAM.
4. **Multilingual FAQ chatbot.** Answer product questions in five languages, drawing on a knowledge base of 5000 articles. Must run in a customer's private cloud on a single A100.
5. **A code-completion IDE plugin.** Autocomplete the next 50 tokens of code as the user types, streaming. p50 latency budget 300 ms.

### Part C — Empirical probe

Pick *one* task where the "right" family is not obvious to you, and run a tiny empirical check:

- Encode the task input with a representative model from each of two candidate families (e.g. `bert-base-uncased` vs. `t5-small`, or `google/flan-t5-base` vs. `mistralai/Mistral-7B-v0.1`).
- Compare on: input tokens per example (fertility × input length), output tokens per example, per-example latency at batch=1 on your hardware, and KV-cache memory at generation (for decoder / encoder-decoder).
- Report whether the empirical numbers reinforce or contradict your rubric's answer. If they contradict it, update the rubric — that is the whole point.

## Starter guidance

- Do not confuse "task = generative-looking" with "task needs decoder-only". Classification can be phrased as generation, but paying for autoregressive decoding when a single forward pass would do is a common failure.
- Sequence-length budget matters. Chapter 08's formulas apply here — a 32k-token contract summarised by an encoder-decoder is very different in cost from the same task served by a 128k-context decoder-only model.
- Pretrained checkpoint availability is a real constraint. There is no strong open-weights encoder-decoder above ~11B parameters as of writing; if you specify "large encoder-decoder", say what checkpoint you would use and confirm it exists.
- For Part C, `time.perf_counter()` around a single `model.generate` call is fine — the goal is order-of-magnitude clarity, not benchmarking rigour.

## Acceptance criteria

- [ ] `RUBRIC.md` includes the decision tree, comparison table (all listed axes), and red-flag list.
- [ ] Every one of the five scenarios gets a family + size class recommendation with explicit reference to the rubric.
- [ ] At least one scenario's write-up notes a genuine trade-off (a reason the second-choice family could also be justified).
- [ ] Part C's empirical probe produces real numbers on your hardware for at least two candidate models on the same task input.
- [ ] The rubric is updated if the empirical result contradicts it, or a paragraph explains why the empirical numbers reinforce rather than refute the initial choice.

## Stretch goals

- **Serving-cost estimate.** For scenario 4 (multilingual FAQ chatbot), estimate cost per 1000 requests on the A100 for two candidate models — one encoder-decoder, one decoder-only. Use chapter 08's KV-cache formulas and published GPU utilisation numbers (or your own benchmark).
- **Rubric extension.** Add a fourth axis: fine-tuning cost. Under what data regime does encoder-only fine-tuning win regardless of family capability?
- **Historical case study.** Pick a paper where the authors switched families mid-project (e.g. moving from BERT to T5 or from an encoder-decoder MT model to a decoder-only LLM for translation). Summarise their justification and evaluate whether your rubric would have arrived at the same answer.
- **Failure post-mortem.** Describe a hypothetical (or real, if you have one) production incident caused by picking the wrong family. What signal, in the rubric, would have caught it earlier?
