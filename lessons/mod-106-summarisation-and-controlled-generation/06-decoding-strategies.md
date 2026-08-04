# Decoding Strategies: Greedy, Beam, Nucleus, Typical, Contrastive

## Motivation

A trained encoder-decoder gives you $p_\theta(y_t \mid y_{<t}, x)$ — a distribution over the next token given the context. What actually reaches the user is one specific sequence, chosen by a *decoding strategy*. Different strategies produce dramatically different outputs from the same model on the same input: greedy is deterministic and terse, beam search is deterministic and mode-seeking, nucleus is stochastic and diverse, contrastive search punishes token-level repetition. Choose wrong and a state-of-the-art summariser produces "the the the" (repetition), "no summary" (empty), or fluent-but-invented text (hallucination).

This chapter walks the decoding zoo. The goal is not to memorise every parameter — the Hugging Face `GenerationConfig` has dozens — but to understand *why* each family exists, when each is appropriate, and how to spot the failure mode each is prone to.

## The two axes: search vs. sampling

Every decoding strategy sits somewhere on two axes:

1. **Search vs. sampling.** Search strategies (greedy, beam) try to find the highest-probability sequence. Sampling strategies (temperature, top-$k$, nucleus, typical) draw from a truncated distribution. Search is deterministic and mode-seeking; sampling is stochastic and diversity-preserving.
2. **How much of the distribution to consider at each step.** Greedy takes 1 token. Beam keeps $k$ hypotheses. Nucleus keeps enough tokens to reach cumulative mass $p$. Typical keeps tokens near the local entropy. Contrastive combines a top-$k$ prune with a repetition penalty.

Roughly:

- **Search wins on tasks with a small correct-answer set.** Summarisation (mostly), QA, translation.
- **Sampling wins on tasks where diversity is part of the deliverable.** Story generation, chatbot creativity, red-team prompt discovery.

Both axes matter. A summariser that beam-searches for the mode is boring but faithful; a chatbot that greedily picks the top token is monotone. Match the strategy to the task.

## Greedy decoding

At every step, pick $\arg\max_v p_\theta(v \mid y_{<t}, x)$.

```python
outputs = model.generate(input_ids, max_new_tokens=128,
                          num_beams=1, do_sample=False)
```

Fast, deterministic, and the natural baseline. Its problems:

- **Myopia.** A locally-optimal token can lead to a globally-suboptimal sequence.
- **Repetition loops.** Greedy is prone to "the cat sat on the mat and the cat sat on the mat" — once the model enters a repeating state, it can never escape without exploration.
- **Empty summaries.** For some seq2seq models, greedy will emit `</s>` immediately and produce an empty output.

Use greedy for cost-sensitive deployments where you have already validated that the failure modes above do not appear on your data. Otherwise, beam.

## Beam search

Keep the top-$k$ partial hypotheses at each step, expanding each and pruning back to $k$.

```python
outputs = model.generate(input_ids, max_new_tokens=128,
                          num_beams=4, do_sample=False,
                          length_penalty=1.0, no_repeat_ngram_size=3,
                          early_stopping=True)
```

- **`num_beams`.** Wider beam = higher-probability sequence found. But not always higher-quality — see below.
- **`length_penalty`.** $>1$ favours longer sequences; $<1$ favours shorter. Necessary because beam search's raw likelihood monotonically decreases with length.
- **`no_repeat_ngram_size`.** Suppress any n-gram that has already appeared. Essential for BART/T5 summarisers to avoid the trigram loop.
- **`early_stopping=True`.** Stop when all beams have hit `</s>`.

Beam is the workhorse for summarisation, translation, and factoid QA. The two classic surprises:

- **The beam-search curse.** As `num_beams` grows past ~5, quality often *decreases* (Stahlberg & Byrne, ["On NMT Search Errors and Model Errors"](https://arxiv.org/abs/1908.10090), *EMNLP 2019*; Meister & Cotterell, 2020). The mode of the model's distribution over sequences is often a degenerate short output or a repetitive one. Beams larger than what the paper reports rarely help.
- **The empty-string mode.** For some poorly-calibrated models, the highest-probability *sequence* is `</s>` — the empty output. Setting `min_length` prevents this.

## Diverse beam search

Beam search returns $k$ near-duplicate hypotheses. Diverse beam search (Vijayakumar et al., 2016) partitions the beams into groups and adds a diversity penalty *between* groups so that the returned $k$ hypotheses differ from each other.

```python
outputs = model.generate(input_ids, max_new_tokens=128,
                          num_beams=6, num_beam_groups=3,
                          diversity_penalty=1.0, do_sample=False)
```

Useful when you want $k$ *different* candidate summaries — for reranking with a downstream metric (MBR, faithfulness score), for user-facing "regenerate" buttons, or for training-time data augmentation.

## Sampling: temperature, top-$k$, and top-$p$ (nucleus)

Sampling strategies draw the next token from a modified distribution rather than taking the argmax:

- **Temperature $T$.** Rescale logits by $1/T$ before softmax. $T \to 0$ approaches greedy; $T = 1$ is the raw distribution; $T > 1$ flattens.
- **Top-$k$ sampling.** Keep only the $k$ highest-probability tokens, renormalise, sample. Fixes "long tail" degeneracy but ignores that the "reasonable" $k$ varies wildly by context (a period is often followed by one clear choice; an adjective by dozens).
- **Top-$p$ / nucleus sampling** (Holtzman et al., ["The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751), *ICLR 2020*). Keep the smallest set of tokens whose cumulative probability exceeds $p$, renormalise, sample. Adapts to local uncertainty — the nucleus is small at a period, large at an adjective. The default recommendation is $p = 0.9$–$0.95$ with $T = 0.7$–$1.0$.

```python
outputs = model.generate(input_ids, max_new_tokens=128,
                          do_sample=True, top_p=0.9, temperature=0.7)
```

Sampling's cost: variance. Two runs give two answers. This is bad for summarisation (users notice), good for creative generation (users expect it), and useful for MBR decoding (see below).

## Typical sampling

Locally typical sampling (Meister et al., ["Locally Typical Sampling"](https://arxiv.org/abs/2202.00666), *TACL 2023*) picks tokens whose *log-probability* is closest to the *local entropy* — i.e., tokens that are "typical" in an information-theoretic sense.

The motivation is that both top-$k$ / nucleus (which keep only high-probability tokens) and pure sampling (which keeps the tail) miss the sweet spot: fluent text often uses tokens that are neither the most probable nor the least, but the "expected" ones under the model's own uncertainty.

```python
outputs = model.generate(input_ids, max_new_tokens=128,
                          do_sample=True, typical_p=0.95)
```

Typical sampling is a good default for open-ended generation. For summarisation, beam or contrastive is usually preferred.

## Contrastive search

Contrastive search (Su et al., ["A Contrastive Framework for Neural Text Generation"](https://arxiv.org/abs/2202.06417), *NeurIPS 2022*) is a deterministic strategy designed to punish repetitive text. At each step, choose from the top-$k$ candidates the token that maximises:

$$
(1 - \alpha) \cdot p_\theta(v \mid y_{<t}, x) - \alpha \cdot \max_{y_{<t}} \text{sim}(h_v, h_{y_{<t}})
$$

where $h_\cdot$ are the hidden states and $\text{sim}$ is cosine similarity. The first term is the model's probability; the second term penalises tokens whose hidden state is too similar to any previously-generated token's hidden state.

```python
outputs = model.generate(input_ids, max_new_tokens=128,
                          penalty_alpha=0.6, top_k=4)
```

Contrastive search is a strong default for decoder-only LLMs when you want deterministic-but-diverse text. For encoder-decoder summarisers, beam search with `no_repeat_ngram_size` usually suffices.

## Contrastive decoding (a different thing)

Sometimes confused with contrastive search. Contrastive decoding (Li et al., ["Contrastive Decoding"](https://arxiv.org/abs/2210.15097), *ACL 2023*) uses *two* models — a strong "expert" and a weak "amateur" — and picks tokens with high expert-vs-amateur log-probability ratio. It suppresses tokens that both models like (fluent but generic) in favour of tokens the expert likes more than the amateur (fluent and specific).

Effective for open-ended generation with LLMs. Less commonly used in production summarisation because it requires two forward passes per step.

## Minimum Bayes Risk (MBR) decoding

Instead of returning the highest-probability sequence, MBR (Eikema & Aziz, ["Is MAP Decoding All You Need?"](https://arxiv.org/abs/2005.10283), *COLING 2020*) samples $N$ candidates, then returns the one with the highest expected utility against the other candidates under a chosen metric (BLEU, BERTScore, ROUGE):

$$
\hat{y} = \arg\max_{y \in Y} \sum_{y' \in Y} \text{sim}(y, y')
$$

MBR outperforms beam search on machine translation (Freitag et al., 2022) and is competitive on summarisation. Two costs: $N \geq 32$ samples per input, and you need a decent similarity metric.

## The strategy zoo, mapped to tasks

| Task                                        | First choice                              | Why                                                    |
|---------------------------------------------|-------------------------------------------|--------------------------------------------------------|
| Single-doc news summarisation               | Beam (`num_beams=4–5`, `no_repeat=3`)     | Faithful, deterministic; established default.          |
| Long-doc / multi-doc summarisation          | Beam or MBR (32 samples + BERTScore)      | MBR helps when the mode is degenerate on long inputs.  |
| Headline generation                         | Beam (`num_beams=6–8`, `length_penalty<1`) | Short outputs; length penalty steers toward brevity.   |
| Creative rewriting / paraphrase             | Typical or nucleus                        | Diversity is the deliverable.                          |
| Constrained / JSON output (chapter 08)      | Beam under a constraint                   | Determinism + schema conformance.                       |
| Story continuation, chatbot, red-teaming    | Nucleus / typical / contrastive search    | Fluency + diversity.                                    |
| Extractive-style QA / factoid               | Beam (`num_beams=4`)                      | Small correct-answer set.                              |

## The failure modes to know

- **Repetition.** Symptoms: `the the the`, or `Company X announced Company X announced`. Fix: `no_repeat_ngram_size=3`, `repetition_penalty=1.1–1.3`, or switch to contrastive search.
- **Truncation.** Model stops mid-sentence. Fix: raise `max_new_tokens`; if training targets were being truncated, retrain.
- **Empty output.** `</s>` on the first step. Fix: `min_length` or `min_new_tokens`; investigate why the model prefers the empty string.
- **Length drift.** Outputs consistently longer or shorter than reference. Fix: tune `length_penalty` and `min_length`/`max_length` against dev.
- **Hallucination amplified by beam width.** Symptom: increasing `num_beams` produces more confidently-wrong summaries. Fix: reduce beam width; add a faithfulness gate (chapter 11).
- **High variance across sampling seeds.** Symptom: two runs disagree. Fix: switch to beam or MBR for determinism.

## Setting up decoding in `transformers`

The `GenerationConfig` object bundles decoding parameters. Save it with the model and version it like any other artefact:

```python
from transformers import GenerationConfig

gen_config = GenerationConfig(
    num_beams=4,
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    min_length=56,
    max_length=142,
    early_stopping=True,
)
gen_config.save_pretrained("summariser/")

# At load time:
model.generation_config = GenerationConfig.from_pretrained("summariser/")
```

Any strategy change is a model change from the user's perspective. Track it.

## Chapter summary

- Every decoding strategy sits on two axes: search-vs-sample and how much of the distribution to consider. Match to the task.
- **Greedy** is fast and boring; **beam** is the summarisation workhorse; **nucleus/typical/contrastive** are for open-ended generation; **MBR** is beam's more principled cousin at higher cost.
- The **beam-search curse** — larger beams sometimes hurt quality — is real. Stay near `num_beams=4–5` for summarisation.
- Sampling introduces variance. Two runs give two answers. This is a bug in production summarisation and a feature in creative generation.
- Track `no_repeat_ngram_size`, `length_penalty`, `min/max_length`, and repetition penalties. Save them in a `GenerationConfig` alongside the model — decoding is versioned artefact, not a runtime toggle.
- Beam width does not fix hallucination. If anything, it makes it more consistent. Fix hallucination with faithfulness gates (chapters 11–12), not with decoding parameters.
