# Mitigating Hallucination in Generation

## Motivation

Chapter 11 measured hallucination. This chapter reduces it. The measurement half is the easy half — you can score a dev set and get a number — but the reduction half is where products live or die. A summariser with 30 % hallucination rate does not ship; a summariser with 3 % hallucination rate might, depending on the product and the failure cost.

There is no single "fix" for hallucination. The mitigation stack is layered: data, training, decoding, post-hoc verification, and product-level abstention. Each layer removes a chunk of the failure surface; each has a cost. This chapter walks the stack and gives the trade-offs.

## Layer 1: data curation

The single most effective mitigation is upstream of the model.

- **Prune training pairs where the summary contains unsupported content.** Run your best faithfulness metric (chapter 11) on the *training* set. Drop or downweight examples with low support scores. Papers that do this (CLIFF — Cao & Wang, 2021; and follow-ups) report noticeable faithfulness gains at modest ROUGE cost.
- **Match summary length distribution to source length.** Summaries that are systematically much shorter than the source encourage aggressive compression, which correlates with hallucination.
- **Remove reference summaries that contain outdated information.** For news corpora, references written years after the source often add hindsight facts that the source does not contain.
- **De-duplicate near-identical source-summary pairs.** Duplicates give the model brittle memorised behaviour.

Data curation is unglamorous and effective. It is the first thing to try, and often the only intervention needed for domain-specific corpora.

## Layer 2: training-time interventions

Interventions in the loss or the training procedure that reduce hallucination without changing the architecture.

### Contrastive learning (CLIFF)

Cao & Wang, ["CLIFF: Contrastive Learning for Improving Faithfulness and Factuality in Abstractive Summarization"](https://arxiv.org/abs/2109.09209), *EMNLP 2021* adds a contrastive loss that pushes the model to prefer faithful summaries over synthetically-perturbed unfaithful ones (entity swaps, negation flips, quantity changes). The result: BART/PEGASUS trained with CLIFF drops hallucination rate by 5–15 % on XSum-style data at ~0.5 ROUGE cost.

### Loss truncation and quality filtering

Kang & Hashimoto (2020) proposed *loss truncation* — during training, drop the top-$k$ % highest-loss examples in each batch. The intuition: high-loss examples are often mislabelled (references containing unsupported content), and giving up on fitting them stops the model from learning to hallucinate.

### Pointer-generator networks and copy mechanisms

For historically-important recipes (pre-transformer, and some transformer variants), pointer-generator networks (See, Liu & Manning, ["Get To The Point"](https://arxiv.org/abs/1704.04368), *ACL 2017*) let the decoder choose between generating a vocabulary token and *copying* a token from the source. When the model is uncertain, copying is often the right default.

Modern encoder-decoders (BART, T5, PEGASUS) do not have explicit copy heads, but their pretraining makes them strong implicit copiers — most fine-tuned summarisers copy 50–90 % of their output tokens directly from the source. Explicit copy heads are relevant when fine-tuning smaller models or when a hard extractive-verifiability guarantee is required.

### RLHF / DPO with faithfulness rewards

Reinforcement learning from human feedback (Stiennon et al., ["Learning to Summarize from Human Feedback"](https://arxiv.org/abs/2009.01325), *NeurIPS 2020*) and its offline cousin DPO can incorporate faithfulness into the reward. In practice: train a reward model on human-ranked faithfulness, then fine-tune with PPO / DPO. Expensive; the strongest known mitigation but requires human data at scale.

## Layer 3: decoding-time interventions

Interventions at inference that reduce hallucination without retraining.

### Constrained decoding with source-vocabulary restriction

Restrict the decoder's vocabulary to tokens that appear in the source. Extreme but effective: the summariser can only emit words from the input, which makes extrinsic hallucination almost impossible. Extractive-decoding baselines effectively do this by construction.

The looser variant, "soft copy" penalties, is what constrained decoding (chapter 08) is for. A `PhrasalConstraint` that must include the key entities, a regex that bounds numeric formats to the source, a JSON schema whose enums are drawn from source-mentioned values.

### Rerank $N$ candidates by faithfulness

Generate $N$ candidates with sampling or diverse beam; score each with a faithfulness metric (chapter 11); return the highest-scoring one:

```python
candidates = model.generate(input_ids, num_beams=8, num_return_sequences=8,
                             diversity_penalty=0.5, num_beam_groups=4)

scored = [(c, faithfulness_score(source, c)) for c in candidates]
best = max(scored, key=lambda x: x[1])
```

Effective; costs $N \times$ the decoding budget plus $N$ faithfulness scorings. Rerank at $N = 4$ often catches most low-hanging hallucinations.

### DoLa and contrastive decoding

Contrastive decoding (Li et al., 2023) and DoLa (Chuang et al., 2023) contrast layer outputs (or expert-vs-amateur models) to prefer tokens that are "more confidently" the model's belief than generic ones. Reduces some kinds of extrinsic hallucination in LLMs; less studied for encoder-decoder summarisers.

### Beam search with faithfulness reranking of partial hypotheses

Modify the beam-search scoring function to include a per-step or per-partial-hypothesis faithfulness signal. In practice this is fiddly — most implementations do the rerank post-hoc on complete candidates.

## Layer 4: post-hoc verification and editing

Interventions that run after generation.

### Cite-then-verify

Ask the model to produce a citation alongside each claim (chapter 09), then automatically verify that the cited span exists and entails the claim. Unverified claims are edited out, flagged, or replaced by an abstention.

Two implementation choices:

- **Structural.** The output schema requires a `citation` field per claim; validation post-hoc.
- **Editorial.** A separate "editor" pass takes the generated summary and revises it to remove unsupported content.

### Chain-of-Verification (CoVe)

Dhuliawala et al., ["Chain-of-Verification Reduces Hallucination in Large Language Models"](https://arxiv.org/abs/2309.11495), *ACL 2024 Findings* proposes:

1. Draft a summary.
2. Generate verification questions targeting the summary's claims.
3. Answer each question against the source.
4. Revise the draft to align with the answers.

Effective for LLM-based summarisers with strong instruction-following. Adds two extra model calls; latency multiplies.

### Rule-based numeric and entity checks

For domain-specific corpora, cheap regex-and-lookup checks catch a surprising fraction of hallucinations that NLI misses:

- **Numeric consistency.** Extract all numeric mentions from source and summary; flag any summary number not in the source (allowing for unit conversions).
- **Named entity consistency.** Extract all named entities; flag any summary entity not in the source.
- **Date consistency.** Extract all dates; flag summary dates not in the source.

These fire on the FRANK categories `predicate error` (numeric mismatch) and `entity error` most reliably. They complement NLI-based metrics; do not replace them.

## Layer 5: product-level abstention

The safety valve. When the pipeline cannot confidently produce a faithful summary, refuse.

- **Return the extractive summary instead.** For any input where the abstractive summary fails a faithfulness gate, fall back to a LEAD-N or BERTSumExt summary. Users get less-fluent but grounded content.
- **Return a structured "cannot summarise" object.** For structured-output products, an abstention arm of the tagged union (chapter 09) is the cleanest response.
- **Escalate to a human reviewer.** For low-volume, high-stakes domains.
- **Return the source verbatim.** Ugly but honest — a valid choice for compliance-critical text.

Abstention has a cost: the user does not get their summary. But an unfaithful summary has a *higher* cost in every domain where "the model was wrong" reaches a customer or an auditor.

## Composing the stack

A production abstractive summariser typically stacks at least three layers:

- Data curation (layer 1) — pre-training.
- Faithfulness-tuned fine-tune (layer 2) — training.
- Rerank + rule-based numeric check (layer 3–4) — decoding.
- Extractive fallback on gate failure (layer 5) — serving.

Each layer catches errors the others miss. A gate at the end of the pipeline that catches everything catches nothing — you would have had to have generated the failure first, and the gate becomes an escalating latency drag.

## Measuring the mitigation

Before/after any mitigation, re-run the chapter-11 faithfulness panel and compare:

- Faithfulness rate (SummaC > threshold).
- Definite-hallucination rate (SummaC below a strict threshold).
- ROUGE / BERTScore delta (mitigations often cost 0.5–2 ROUGE points; know your budget).
- Latency and cost per prediction (each layer adds).

A mitigation that improves faithfulness by 2 % but costs 3 ROUGE points is usually the wrong trade. A mitigation that improves faithfulness by 15 % at 0.5 ROUGE cost is a clear win. The right number for your product depends on the cost of an unfaithful summary in your domain.

## What does not work

Approaches that recur in tutorials and rarely help:

- **"Just increase beam width."** Larger beams find the *mode* of the model's distribution more precisely. If the mode is unfaithful, beam width makes it *more* consistent. See the beam-search curse (chapter 06).
- **"Just add 'be faithful' to the prompt."** Prompt-engineering-only interventions on non-instruction-tuned models are noise. On instruction-tuned models the effect is real but small (Adams et al., 2023 report ~2 pts).
- **"Just fine-tune longer."** More epochs on the same data amplifies whatever hallucination the data supports. Data curation first, then training length.
- **"Just switch to a bigger model."** Larger models are *more* fluent hallucinators, not more truthful ones. Faithfulness does not monotonically scale with parameters; the mitigation stack does.

## Chapter summary

- Hallucination is mitigated in layers: data curation, training-time interventions, decoding-time interventions, post-hoc verification, product-level abstention.
- **Data curation** — dropping unfaithful training pairs — is the highest ROI intervention and the first thing to try.
- **CLIFF-style contrastive training**, **loss truncation**, and **RLHF/DPO with faithfulness reward** are the training-time levers.
- **Rerank $N$ candidates by faithfulness** is the standard decoding-time intervention. Constrained decoding with source-vocabulary restrictions is the strong version.
- **Cite-then-verify** and **Chain-of-Verification** are the post-hoc patterns. Rule-based numeric/entity checks catch what NLI misses.
- **Product-level abstention** (extractive fallback, structured "cannot summarise", escalation) is the safety valve. Always have one.
- Increasing beam width, adding "be faithful" to a prompt, and switching to a bigger model do not fix hallucination. Data, training, decoding, verification, abstention do.
