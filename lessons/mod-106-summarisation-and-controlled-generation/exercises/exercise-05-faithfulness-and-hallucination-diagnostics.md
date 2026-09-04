# exercise-05: Faithfulness and Hallucination Diagnostics

**Estimated effort:** 2 hours

## Objective

Build and calibrate a **faithfulness panel** for a real summariser, following chapter 11. You will implement two orthogonal automatic metrics (NLI-based via SummaC and a QA-based signal), a light rule-based numeric check, a small LLM-as-judge, and ground everything against a **50-example human-labelled set** you annotate yourself. Then you will exercise the panel on outputs from three different decoding strategies to reproduce the "beam width amplifies hallucination" pattern chapter 12 warns about.

## Prerequisites

- Chapter [11](../11-faithfulness-and-hallucination-diagnostics.md); optionally [12](../12-mitigating-hallucination.md) for the mitigation framing.
- Outputs from exercise-01 (fine-tuned XSum summariser) or exercise-03 (multiple decoding strategies on the same model). At minimum you need `(source, gold, prediction)` triples for one summariser.
- Python 3.10+; `transformers`, `datasets`, `nltk`, `summac` (`pip install summac`), `scipy` (for correlation), an API-based LLM (Claude / GPT-4-class) for the LLM-as-judge cell — or a strong local instruction-tuned model if you prefer to stay offline.
- GPU with ≥ 12 GB VRAM for the NLI models.

## Dataset

Use ≥ 300 model predictions on a single dataset. Recommendation:

- **XSum dev subset** — one-sentence "extreme" summaries; the fastest way to expose hallucination.
- Alternatively, CNN/DailyMail dev subset if XSum is not available. CNN/DailyMail hallucinates less, so plan for smaller effect sizes.

You need three prediction sets *on the same 300 inputs*:

- **Predictions_beam4** — from exercise-01 (or generate now with `num_beams=4, no_repeat_ngram_size=3`).
- **Predictions_beam16** — same model, `num_beams=16`.
- **Predictions_nucleus** — same model, `do_sample=True, top_p=0.9, temperature=0.7`.

If you already have exercise-03 outputs, reuse them.

## Problem statement

### Part A — Method 1: NLI-based faithfulness (SummaC-Conv)

Implement the SummaC-Conv baseline from chapter 11:

```python
from summac.model_summac import SummaCConv
model = SummaCConv(models=["vitc"], granularity="sentence",
                   agg="mean", device="cuda", start_file="default")
scores = model.score(documents=sources, generated_summaries=predictions)
```

Alternative (hand-rolled sentence-level NLI, if `summac` install is problematic):

- For each `(source, prediction)`, split the prediction into sentences.
- For each `(source, prediction_sentence)`, run `microsoft/deberta-v3-large-mnli` and take the `ENTAILMENT` probability.
- Aggregate per prediction (mean and min).

Report on **each** of the three prediction sets:

- Mean SummaC score, median, and the 10th / 90th percentiles.
- Fraction with SummaC > 0.5 ("faithful").
- Fraction with SummaC < 0.3 ("definite hallucination").

Save the per-example scores to disk — Parts C, D, and E rescore against them.

### Part B — Method 2: QA-based faithfulness

Implement a lightweight QAGS-style scorer:

1. **Question generation.** For each prediction, generate 3–5 questions whose answers come from the prediction. Options:
   - `mrm8488/t5-base-finetuned-question-generation-ap` (T5 fine-tuned for QG).
   - Prompt an instruction-tuned LLM ("Given this text, generate 3 factoid questions whose answers are drawn from it. Return one per line.").
2. **QA against the source.** For each question, run an extractive QA model (`deepset/roberta-base-squad2`) with the source as context. Record the answer and confidence.
3. **Match.** Compare the QA answer against the prediction-derived answer using either:
   - String overlap (token F1 as in SQuAD evaluation).
   - Semantic similarity (`sentence-transformers` cosine).
4. **Aggregate.** Per prediction, average the question-level match scores. This is the QAGS-style faithfulness score in [0, 1].

Report on the same three prediction sets:

- Mean QAGS score.
- Fraction with QAGS > 0.5.

Runtime is significant (3–5 questions × QG + QA per prediction). If you need to trim, use a random 150-example subset and note it.

### Part C — Method 3: Rule-based numeric consistency

The blind spot chapter 11 flags: NLI models miss numeric-swap hallucinations. Implement a rule-based numeric checker:

- Extract all numeric mentions (integers, decimals, percentages, currency amounts) from source and prediction with a regex like `r"[\$€£]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:%|B|M|K)?"`.
- Compute a per-prediction "numeric consistency" score: fraction of prediction numbers that appear (exact match, up to unit normalisation) in the source.
- Flag any prediction where numeric consistency < 1.0.

Report on all three sets:

- Mean numeric consistency.
- Number-of-predictions with at least one unsupported numeric mention.
- 5 example predictions where SummaC scored high (> 0.7) but the numeric check flagged a hallucination — chapter 11's canonical NLI blind spot.

### Part D — Method 4: LLM-as-judge

For 100 random predictions from each of the three sets (300 total), run an LLM-as-judge with this rubric (adapted from chapter 11 and chapter 13):

```
You are evaluating whether a candidate summary is faithful to a source document.
A summary is FAITHFUL if every factual claim in it is directly supported by the source.
A summary is UNFAITHFUL if any claim is contradicted by, or absent from, the source.

Source:
<source>

Summary:
<summary>

Return valid JSON with this shape:
{"verdict": "faithful" | "unfaithful",
 "unsupported_claims": [list of exact spans from the summary],
 "reasoning": "one-sentence explanation"}
```

Ideally use a strong model (`claude-sonnet-4-6`, `gpt-4o`, or comparable). If cost is an issue, use `Meta-Llama-3.1-70B-Instruct` or a similar strong open-weight model. Constrain the output to a JSON schema (exercise-04 techniques).

Report per prediction set:

- Fraction judged faithful.
- Mean number of unsupported claims per prediction (from the JSON output).

### Part E — Method 5: Human ground truth (50 examples)

Manually label **50 predictions** yourself:

- Sample 50 predictions balanced across the three sets and across the SummaC-score distribution (e.g., 5 low / 5 mid / 5 high per set + 5 extras).
- For each, mark:
  - Overall faithfulness label: `faithful` / `unfaithful`.
  - Which FRANK category if unfaithful: `entity_error`, `predicate_error`, `circumstance_error`, `coreference_error`, `discourse_link_error`, `out_of_source_information` (chapter 11).
- Save as `human_labels.jsonl`.

If human labelling is genuinely impractical for you (be honest — this is a common shortcut that hides methodology bugs), replace with 50 labels *emitted by a stronger LLM judge* than the one used in Part D, and clearly mark them as "silver labels" throughout.

### Part F — Metric calibration

On the 50-example labelled subset (Part E), compute for each automatic metric (SummaC, QAGS, numeric consistency, LLM-as-judge verdict):

- **Correlation with the human label.** Pearson if the metric is continuous; Cohen's κ if it is binary.
- **Threshold analysis.** Find the score threshold that maximises F1 against the human label. Report precision / recall / F1 at that threshold.
- **False-positive examples.** For each metric, paste 2 examples where the metric said "faithful" but the human said "unfaithful". Discuss the failure mode.
- **False-negative examples.** For each metric, paste 2 where the metric said "unfaithful" but the human said "faithful". Discuss the failure mode.

Present as a correlation table (rows = metrics, columns = correlation, κ, F1, best-threshold) and a short qualitative section for each metric.

### Part G — The decoding × faithfulness interaction

Using all four metrics (SummaC, QAGS, numeric consistency, LLM-as-judge), compare the three decoding strategies (beam-4, beam-16, nucleus). Report:

- Mean faithfulness score per (metric × strategy).
- The strategy that wins on ROUGE-L (compute from the same predictions) vs. the strategy that wins on faithfulness. They will usually not be the same — this is chapter 06's beam-search curse and chapter 12's "beam width does not fix hallucination" warning in one plot.

Present as a Markdown table (rows = strategies, columns = ROUGE-L, SummaC, QAGS, numeric-consistency, LLM-judge-faithful-rate).

### Part H — Write-up

A 600–900 word `README.md` covering:

- Model and predictions used; how the three prediction sets were generated.
- All four automatic metric distributions (Part A–D) on all three prediction sets.
- The calibration table (Part F) with correlation and best-threshold F1 per metric.
- The decoding × faithfulness interaction table (Part G) and a one-paragraph interpretation.
- Two false-positive examples (per Part F) showing NLI's numeric blind spot most cleanly.
- One "what next" idea — e.g., adopting FactScore for a long-form product, adding a rerank-by-faithfulness pass (chapter 12), or replacing beam-4 with rerank-4 in serving.

## Starter guidance

- **Sentence-tokenise before running NLI.** Whole-summary NLI collapses to `neutral` most of the time; sentence-level is the SummaC-Conv formulation and works.
- **SummaC installation is finicky.** If the pip package fails to install, hand-roll it: sentence-tokenise the summary, chunk the source into 300-token windows, run pairwise NLI, take the max-entailment per summary sentence, aggregate by mean.
- **Rescale-with-baseline for BERTScore — but not for NLI.** NLI probabilities in `[0, 1]` are their own scale; SummaC's aggregation is already on `[0, 1]`.
- **Numeric extraction is finicky.** Watch for `"$1.2 B"` vs. `"$1.2B"` vs. `"1200 million"`. A perfectionist extractor is worth ~2 hours; a good-enough one for this exercise is 20 minutes of regex.
- **Do the human labelling first, not last.** Labelling 50 examples takes 45–90 minutes and *ground-truths every other number*. Rushing it invalidates Parts F and G.
- **LLM-as-judge on the same predictions with two seeds should almost agree.** If it doesn't, either your prompt is under-specified or the model is a poor judge for the task.
- **Correlation on 50 examples has wide CIs.** Report the point estimate and note the sample size; do not over-interpret a 0.05 delta in Pearson.

## Acceptance criteria

- [ ] SummaC scores computed and reported on all three prediction sets with mean / percentile / faithful-rate.
- [ ] QAGS-style scorer implemented and run; mean score reported.
- [ ] Rule-based numeric consistency scored and 5 SummaC-high-but-numerically-wrong examples pasted.
- [ ] LLM-as-judge run on 100 examples per prediction set with a structured output schema.
- [ ] 50 human-labelled predictions saved to `human_labels.jsonl` (or silver labels clearly marked).
- [ ] Metric calibration table with Pearson / κ / best-threshold F1 per automatic metric.
- [ ] Decoding × faithfulness interaction table (3 strategies × 5 metric columns).
- [ ] 600–900 word write-up covering all parts and one "what next" idea.

## Stretch goals

- **FactScore on a long-form set.** Take 30 GovReport or Multi-News predictions from exercise-02, decompose each into atomic facts with an LLM, verify each atomic fact against the source. Compare FactScore against SummaC on the same predictions.
- **DAE for predicate-level errors.** Add Goyal & Durrett's DAE (dependency-arc entailment) to the metric panel. Do DAE and SummaC agree on the numeric-consistency failures from Part C?
- **Faithfulness reranking (chapter 12, Layer 3).** For the beam-16 predictions, generate 8 candidates each with diverse beam; rerank by SummaC × ROUGE-L; return the top-1. Report the SummaC and ROUGE-L delta versus the original beam-16.
- **Cite-then-verify pipeline (chapter 12, Layer 4).** For 50 predictions, prompt the model to emit `(claim, source_sentence_id)` tuples; validate that the cited sentence entails the claim; remove unverified claims. Report the pre / post SummaC delta.
- **TRUE-benchmark reproduction.** Score at least two of your metrics on the TRUE meta-benchmark data (Honovich et al., 2022). Are your metric-to-human correlations in the ballpark of the paper's numbers?
- **Annotator agreement.** Have a second person label 20 of your 50 examples independently. Report Cohen's κ against you. Under 0.6 means either the rubric is under-specified or the task is genuinely hard — both are important findings.
