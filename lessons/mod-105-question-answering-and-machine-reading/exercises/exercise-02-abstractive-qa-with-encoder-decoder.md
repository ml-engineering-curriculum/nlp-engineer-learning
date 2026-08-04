# exercise-02: Abstractive QA with Encoder-Decoder

**Estimated effort:** 3 hours

## Objective

Train an encoder-decoder abstractive QA model on an open-domain dataset, evaluate it with a *panel* of metrics (SQuAD F1, ROUGE-L, and BERTScore or BLEURT), and measure how often the model hallucinates content that is not supported by the source. This exercise makes the case that "single-metric" evaluation of generative QA is misleading and that faithfulness must be measured separately.

## Prerequisites

- Chapters [01](../01-question-answering-and-the-machine-reading-landscape.md), [04](../04-em-and-f1-the-squad-evaluation-protocol.md), [05](../05-abstractive-qa-with-encoder-decoder-models.md).
- `transformers`, `datasets`, `evaluate`, `torch`, `bert_score` (or `bleurt`), `rouge_score`, an NLI model such as `roberta-large-mnli` or `microsoft/deberta-v3-large-mnli`.
- GPU recommended for `flan-t5-large`; `flan-t5-base` is workable on CPU with patience.

## Dataset

Pick one:

- **MS MARCO QnA (v2.1)** — Nguyen et al., ["MS MARCO: A Human-Generated Machine Reading Comprehension Dataset"](https://arxiv.org/abs/1611.09268), *NIPS 2016 Workshop on Cognitive Computation*. `datasets.load_dataset("ms_marco", "v2.1")`. Passages plus abstractive human-written answers.
- **NarrativeQA** — Kočiský et al., ["The NarrativeQA Reading Comprehension Challenge"](https://arxiv.org/abs/1712.07040), *TACL 2018*. `datasets.load_dataset("deepmind/narrativeqa")`. Long-form abstractive QA over books and movie scripts.
- **ELI5** — Fan et al., ["ELI5: Long Form Question Answering"](https://arxiv.org/abs/1907.09190), *ACL 2019*. Long-form open-domain QA.

MS MARCO is the easiest ramp; NarrativeQA is closer to production long-context abstractive QA; ELI5 stresses long-form generation.

## Problem statement

### Part A — Baseline: extractive on the same data

Train (or reuse from exercise-01) an extractive model, apply it to your dataset, and report SQuAD F1 on the abstractive dev set. You will need to make one decision:

- For extractive scoring on abstractive gold answers, either restrict to questions where the gold appears as a substring of the passage, or accept low F1 as evidence of the extractive/abstractive mismatch.

Report the sample size of "extractable" questions and F1 on that subset.

### Part B — Train the abstractive model

Fine-tune `google/flan-t5-base` (or `-large` if you have the compute) using the recipe from chapter 05:

- Input template: `question: {q}  context: {c}`
- Target: raw answer text
- `DataCollatorForSeq2Seq`
- LR `1e-4`, effective batch `16`, `num_train_epochs = 3`, `weight_decay = 0.01`, `warmup_ratio = 0.06`
- `predict_with_generate = True`, `generation_max_length = 64`, `generation_num_beams = 4`

Save the checkpoint. Save training logs.

### Part C — Metric panel

Evaluate on dev with the following metrics, all against the gold answer set:

- **SQuAD EM and F1** (`evaluate.load("squad")`).
- **ROUGE-L** (`evaluate.load("rouge")`; report F-measure of the L variant).
- **BERTScore** (`evaluate.load("bertscore")` with `lang="en"` and `model_type="microsoft/deberta-xlarge-mnli"` or default) *or* **BLEURT** (`evaluate.load("bleurt", "BLEURT-20")`).

Report all four (or five, if you run both) numbers with 95 % bootstrap CIs.

Then compute *pairwise correlations* between the metrics across dev-set predictions. Plot as a heatmap.

### Part D — Faithfulness / hallucination measurement

Score every prediction for faithfulness by running an NLI model on `(context, prediction)` and reading off the entailment probability:

```python
nli = pipeline("text-classification", model="microsoft/deberta-v3-large-mnli", top_k=None)
scores = nli(f"{context} [SEP] {prediction}")
# entailment / neutral / contradiction probabilities
```

Report:

- Fraction of predictions with entailment probability > 0.5 (loosely: "faithful").
- Fraction with contradiction probability > 0.5 (loosely: "hallucinated").
- Correlation of entailment probability with each metric from Part C.

If you can, sample 20 predictions with entailment < 0.3 *and* high SQuAD F1 (the pathological case: high overlap with the gold but not supported by the context). Manually inspect and confirm the hallucination.

### Part E — Beam search vs. greedy vs. sampling

For 200 dev questions, generate answers under three decoding strategies:

- Greedy (`num_beams = 1`, `do_sample = False`).
- Beam search (`num_beams = 4`).
- Nucleus sampling (`do_sample = True`, `top_p = 0.9`, `temperature = 0.7`).

Report SQuAD F1 and mean entailment for each. Comment on the strategy that maximises F1 vs. the one that maximises faithfulness.

### Part F — Write-up

A 500–700 word `README.md` covering:

- Dataset, model, and hyperparameters.
- The full metric panel with CIs and the correlation heatmap.
- Faithfulness results and a discussion of what SQuAD F1 alone would have missed.
- The decoding-strategy comparison table.
- One example of a hallucination with the input, the prediction, and the NLI score.
- One thing you would try next (e.g., add an extractive fallback per chapter 05).

## Starter guidance

- Do not skip `DataCollatorForSeq2Seq`. `DefaultDataCollator` produces a silent-but-wrong loss for seq2seq.
- On MS MARCO, discard questions where `wellformed_answer` is empty — those were "unanswerable" per the annotators and do not have abstractive references.
- If you use FLAN-T5, zero-shot inference on your dev set is a useful baseline before fine-tuning. Report both numbers.
- For NarrativeQA, pass the *summary* field as context (not the full book) or use LongT5 (chapter 07) — the full text does not fit in T5's 512-token window.
- BERTScore is expensive; batch it and cache. On a large dev set, run it on a random 500-example subset if runtime is a problem.

## Acceptance criteria

- [ ] Extractive baseline reported on the "extractable" subset with sample-size annotation.
- [ ] Fine-tuning script (`train_abstractive_qa.py`) reproduces the Part B recipe.
- [ ] Four (or five) metrics reported with bootstrap CIs on the abstractive dev set.
- [ ] Metric correlation heatmap included in the write-up.
- [ ] NLI-based faithfulness statistics reported: fraction faithful, fraction hallucinated, correlation with each metric.
- [ ] Decoding-strategy comparison table with F1 and mean entailment for greedy / beam / sampling.
- [ ] 500–700 word write-up with at least one worked hallucination example.

## Stretch goals

- **BART vs. T5 vs. FLAN-T5.** Repeat with `bart-large` and `t5-large`. Which is best on faithfulness at fixed F1?
- **Extractive fallback.** Wire the extractive model from Part A as a fallback: if the abstractive prediction's entailment < 0.3 and an extractive span exists, prefer the extractive answer. Report the joint metric.
- **Constrained decoding.** For MS MARCO, constrain output to be a substring of the context using `PrefixConstrainedLogitsProcessor` or `Outlines`. Compare to unconstrained. Does the constraint hurt F1? Does it eliminate hallucination?
- **Label smoothing.** Train with `label_smoothing_factor = 0.1` and compare metrics. Any change to faithfulness?
- **LLM-as-judge.** Score 100 predictions with a strong LLM using a chapter-11-style rubric (correctness, faithfulness, completeness). Report per-dimension pass rate and correlation with the automatic metrics.
