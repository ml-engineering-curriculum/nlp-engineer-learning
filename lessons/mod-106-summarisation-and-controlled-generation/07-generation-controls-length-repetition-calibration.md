# Generation Controls: Length, Repetition, and Calibration Knobs

## Motivation

The last chapter picked *which* decoding strategy. This one picks the *knobs* on top: the length caps, the repetition penalties, the vocabulary constraints, the calibration levers. These are the flags you will actually spend time tuning on every summariser you ship. Skipping them is why so many fine-tuned models produce technically-correct summaries that are three sentences too long, or that end mid-word, or that repeat the article's first noun eleven times.

Every knob in this chapter maps to a specific failure mode. Learn the failure, then reach for the knob.

## Length control: `min_length`, `max_length`, and `length_penalty`

Three knobs, distinct meanings:

- **`max_length` / `max_new_tokens`.** Hard cap on total sequence length or newly-generated length. Once the cap is hit, generation stops mid-token.
- **`min_length` / `min_new_tokens`.** Hard floor. Below the floor, the model is forbidden from emitting `</s>`.
- **`length_penalty`.** During beam search, divides log-likelihood by `length^length_penalty`. $>1$ favours longer sequences (they get a smaller divisor once the ranking normalises); $<1$ favours shorter. Defaults to $1.0$ (neutral).

The three interact:

- Setting `max_length=142` with no `length_penalty` produces summaries that cluster near the cap because beam search prefers longer high-probability sequences.
- Setting `length_penalty=0.6` alone will produce very short summaries — probably too short.
- The right combo for a fixed-length product summary is `min_length=56, max_length=142, length_penalty=1.0` (the BART-CNN defaults), followed by dev-set tuning of the exact numbers.

If your target-length distribution is *bimodal* (short tweets and long articles), a single global length setting is wrong. Either train two models, use a conditional prefix ("write a tweet:" vs. "write an article:"), or accept that the model will regress to the mean.

### Length-controlled fine-tuning

If length is a first-class product requirement (e.g., "always a headline of at most 12 tokens"), enforce it *during training*, not just at inference. Two techniques:

- **Length-bucketed training.** Sort training examples by target length; ensure every batch contains examples from your target-length distribution.
- **Length control codes.** Prepend a length hint to the input: `"<len=short> summarize: ..."`. At inference, condition on the code that matches your desired output length. Introduced for machine translation (Kikuchi et al., 2016) and widely used since.

Both give tighter length adherence than post-hoc `max_length` truncation, which produces mid-sentence stops.

## Repetition control: `no_repeat_ngram_size` and `repetition_penalty`

Repetition is the single most common decoding failure mode. Two orthogonal knobs:

- **`no_repeat_ngram_size=n`.** Forbid generation of any n-gram that has already appeared in the output. `n=3` is the standard for summarisation — small enough to allow legitimate repetition of short phrases ("in the United States"), large enough to kill degenerate loops.
- **`repetition_penalty`.** Multiply the logit of any token that has already appeared by `1/penalty`. `1.0` is off; `1.1–1.3` is the useful range; above `1.3` and you start to see the model avoiding legitimate reuse of core entities.

Both work, both are cheap, and both should be turned on for summarisation. `no_repeat_ngram_size=3` is the more surgical of the two; `repetition_penalty` is softer and interacts with sampling temperature.

The failure mode that fools both is *sentence-level* repetition (two paraphrases of the same sentence). Neither knob catches it. If your model paraphrases the same claim twice, the fix is at training time (fewer repetitive training targets) or at post-processing (dedup the output sentences with an embedding-based check).

## Suppressing tokens: `bad_words_ids` and `suppress_tokens`

For any deployment with a forbidden vocabulary — competitor names, profanity, PII patterns — you can pass a list of token IDs to be suppressed at every step:

```python
bad_tokens = [tokenizer(word, add_special_tokens=False).input_ids
              for word in ["Competitor", "Slur1", "Slur2"]]

outputs = model.generate(input_ids,
                          bad_words_ids=bad_tokens,
                          num_beams=4)
```

Two caveats:

- **Suppression is per-token, not per-string.** "Competitor" tokenises to different IDs depending on preceding whitespace and case. Add all likely variants.
- **The model may route around suppression.** If you suppress "Competitor" the model may emit "Compet itor" or "C0mpetit0r". A stronger constraint is a full CFG or a post-hoc regex validator; see chapter 08.

For a positive vocabulary constraint (must include one of these words), see the disjunctive-constraints section below.

## Forcing tokens: `forced_bos_token_id`, `forced_eos_token_id`, and forced decoder prompts

Some tasks require the output to start or end with a specific token — most famously, the language token in multilingual seq2seq (mBART, NLLB, mT5 for translation) or an XML/JSON opening delimiter.

```python
outputs = model.generate(input_ids,
                          forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
```

For a longer forced prefix, pass `decoder_input_ids` explicitly:

```python
prefix = tokenizer("Summary:", return_tensors="pt").input_ids
outputs = model.generate(input_ids, decoder_input_ids=prefix)
```

The model generates *continuations* of the prefix; the prefix is not scored.

## Constrained beam search: `PhrasalConstraint` and `DisjunctiveConstraint`

For "must-include" content constraints:

```python
from transformers.generation import PhrasalConstraint, DisjunctiveConstraint

must_include = ["quarterly revenue"]
must_include_ids = [tokenizer(p, add_special_tokens=False).input_ids
                    for p in must_include]

outputs = model.generate(input_ids,
                          constraints=[PhrasalConstraint(ids)
                                       for ids in must_include_ids],
                          num_beams=6)
```

Constrained beam search (Anderson et al., 2017; Hokamp & Liu, 2017) modifies the beam search to guarantee the constraint is satisfied — at some latency cost. `num_beams` needs to be at least `2 * len(constraints)` for the algorithm to work well.

Chapter 08 covers the more general constrained-decoding frameworks (Outlines, Guidance, XGrammar) that handle regex, JSON schema, and CFG constraints beyond phrasal inclusion.

## Temperature calibration

For sampling strategies, temperature is the single biggest lever:

- $T = 0$ (in the limit): greedy. Deterministic.
- $T = 0.7$: mild sampling. Fluent-but-varied summaries; the default for chatbots.
- $T = 1.0$: raw distribution. Diverse; sometimes noticeably off.
- $T > 1.2$: high entropy. Increasingly incoherent.

For summarisation with sampling, `T ∈ [0.7, 0.9]` is a sensible band. For MBR-style sampling followed by rerank, use `T = 1.0` — you want diversity in the candidates because MBR handles the selection.

Temperature and top-$p$/top-$k$ *interact*. High $T$ + low $p$ = pick sharply from a broad distribution (a lot like unconditional sampling). Low $T$ + high $p$ = pick softly from a narrow distribution (close to greedy). Change one at a time.

## Label smoothing (training-time calibration)

Cross-entropy training on hard labels pushes the model to place all its probability on the gold token. This over-confidence hurts *sampling* at inference (probabilities are miscalibrated) and hurts *beam search* on rare-token vocabulary items. Label smoothing (Szegedy et al., 2016; Müller et al., 2019) softens the target distribution:

$$
\tilde{y}_i = (1 - \epsilon) \cdot y_i + \epsilon / V
$$

for smoothing factor $\epsilon \approx 0.1$ and vocabulary size $V$. `Seq2SeqTrainingArguments(label_smoothing_factor=0.1)` handles it in Hugging Face.

Label smoothing helps summarisation with BART; it hurts summarisation with some T5 checkpoints (whose pretraining already includes span smoothing). Always A/B on your dev set.

## Exposure bias and scheduled sampling

Teacher-forced training feeds the *gold* previous tokens at every step; inference feeds the *model's own* previous tokens. This mismatch — exposure bias — can accumulate over long sequences.

Scheduled sampling (Bengio et al., 2015), sequence-level fine-tuning with metrics as rewards (MRT — Shen et al., 2016; Ranzato et al., 2016), and reinforcement-learning-from-human-feedback (RLHF — Stiennon et al., 2020) all address exposure bias by exposing the model to its own outputs during training. In modern practice:

- For encoder-decoder summarisers, teacher forcing + label smoothing is the standard. Exposure bias is a known issue, but the fixes are heavy machinery for modest gains.
- For decoder-only LLMs, the RLHF/DPO stack from `mod-102`-era language modelling handles calibration much better than scheduled sampling ever did.

## Number-of-samples: how many to generate

For beam search, `num_return_sequences` controls how many of the top-$k$ beams are returned:

```python
outputs = model.generate(input_ids, num_beams=8, num_return_sequences=4)
```

For sampling, it controls how many independent samples are drawn. Use $\geq 32$ for MBR reranking; $4$–$8$ for user-facing "regenerate" or A/B options; $1$ for cost-sensitive single-shot serving.

## Batching and generation latency

Two performance knobs that matter in production but rarely appear in tutorials:

- **`use_cache=True`.** Cache attention keys/values across decoding steps. Essential — a 10× speedup for medium-length generation. `transformers` defaults to on.
- **KV-cache-aware batching.** Batch inputs of similar length together; short outputs in a batch of long ones waste compute. `vllm`, `TGI`, and `SGLang` handle this transparently; naive `.generate()` in a loop does not.

Neither changes output quality, but both change whether you can afford the strategy from chapter 06 in production.

## A checklist for shipping a summariser's decoding config

Before you ship, write down the specific values for:

- [ ] Strategy family (beam / nucleus / typical / contrastive)
- [ ] `num_beams` (or `top_p`, `top_k`, `typical_p` if sampling)
- [ ] `min_length` / `max_length` (or the `_new_tokens` variants)
- [ ] `length_penalty` (if beam)
- [ ] `no_repeat_ngram_size`
- [ ] `repetition_penalty`
- [ ] `temperature` (if sampling)
- [ ] `bad_words_ids` / `suppress_tokens` (if applicable)
- [ ] Any constraints (`PhrasalConstraint`, `DisjunctiveConstraint`, or a chapter-08 library)
- [ ] `use_cache=True`

Save as a `GenerationConfig`, version it with the model, and expose the values in your monitoring dashboard. Undocumented decoding is the number-one source of "why did the model start producing empty summaries" post-mortems.

## Chapter summary

- Length is controlled by `min_length` / `max_length` (hard bounds) and `length_penalty` (beam bias). Length-bucketed training or length control codes tighten adherence beyond post-hoc truncation.
- Repetition is controlled by `no_repeat_ngram_size` (surgical) and `repetition_penalty` (soft). Both should be on for summarisation.
- Suppression and forcing (`bad_words_ids`, `forced_bos_token_id`, `PhrasalConstraint`) handle vocabulary constraints below the level of a full grammar.
- Temperature and top-$p$/top-$k$ interact — change one at a time.
- Label smoothing softens over-confident cross-entropy; test on dev, but expect ~$0.1$ to help BART-style summarisers and be neutral-to-hurt on T5.
- Exposure bias is real but rarely worth the training complexity for standard summarisers.
- Ship a versioned `GenerationConfig`. Undocumented decoding is a bug.
