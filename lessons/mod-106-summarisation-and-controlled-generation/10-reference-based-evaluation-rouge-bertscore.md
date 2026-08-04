# Reference-Based Evaluation: ROUGE, BERTScore, and Their Limits

## Motivation

You have trained a summariser. Now you need a number. Reference-based metrics — those that compare model output against a human-written reference summary — are the fastest way to get one, and every paper in the field reports at least one. They are also the source of most misleading conclusions in summarisation research, because a metric that correlates well with human judgement on one dataset can be flat-out wrong on another.

This chapter walks the metrics you will actually use — **ROUGE-1/2/L**, **METEOR**, **BLEU**, **BERTScore**, **BLEURT**, **BARTScore**, **MoverScore** — explains what each measures, and gives the guardrails that stop you from over-interpreting a two-point difference.

Faithfulness metrics (which compare output against the *source*, not against a reference) are chapter 11. Human evaluation is chapter 13. This chapter is the reference-based side.

## ROUGE

Recall-Oriented Understudy for Gisting Evaluation (Lin, ["ROUGE: A Package for Automatic Evaluation of Summaries"](https://aclanthology.org/W04-1013/), *ACL 2004 Workshop*) is the standard reference-based metric for summarisation. Every paper reports it. Every checkpoint's model card cites it. It is the metric everyone loves to complain about.

The three ROUGE variants that matter:

- **ROUGE-N.** N-gram co-occurrence between candidate and reference. ROUGE-1 (unigrams) captures content overlap; ROUGE-2 (bigrams) captures fluency and local order.
- **ROUGE-L.** Longest common subsequence between candidate and reference. Order-sensitive without requiring contiguous matches. This is usually the flagship ROUGE score in papers.
- **ROUGE-W.** Weighted-length LCS, penalising fragmented matches. Rare in practice.

Each variant reports precision, recall, and F1. Papers usually report ROUGE-1/2/L F1.

```python
import evaluate

rouge = evaluate.load("rouge")
result = rouge.compute(
    predictions=candidates,
    references=references,
    use_stemmer=True,
    tokenizer=None,       # defaults to whitespace + punctuation
)
# result["rouge1"], result["rouge2"], result["rougeL"] — F-measure floats.
```

Two implementation details that quietly change the number:

- **Stemming.** `use_stemmer=True` collapses `run/runs/running`. Standard in published numbers; check before comparing.
- **Tokenisation.** Different tokenisers give slightly different scores. The `evaluate` library's default is close to the original Perl ROUGE-1.5.5, but not identical. Do not compare ROUGE scores across implementations.

### What ROUGE actually measures

ROUGE measures *lexical overlap*. It rewards summaries that reuse the reference's words. Two consequences:

- **Extractive systems that copy from the source have a natural advantage** if the reference is also close to the source (CNN/DailyMail).
- **Abstractive systems that paraphrase get punished** even when the paraphrase is correct ("won two of three" vs. "took the series").

ROUGE correlates moderately with human judgement on content adequacy for news summarisation. It correlates poorly on dialogue summarisation, on faithfulness, and on multi-reference tasks where the "gold" is one of many valid summaries.

### Interpreting ROUGE numbers

Sensible reference points (approximate, dataset-dependent — always look up the current SOTA):

- CNN/DailyMail: LEAD-3 ROUGE-L ≈ 32; strong abstractive systems ≈ 40+.
- XSum: LEAD-1 ROUGE-L ≈ 17; strong abstractive ≈ 36+.
- SAMSum (dialogue): LEAD-1 ≈ 20; strong abstractive ≈ 45+.

A 0.5-point ROUGE-L difference between two systems is often noise. A 2-point difference is usually real. A 5-point difference is almost always real. Bootstrap CIs on your dev set give you the true noise floor.

## BLEU and METEOR

Machine-translation heritage. **BLEU** (Papineni et al., 2002) is a precision-focused n-gram metric; **METEOR** (Banerjee & Lavie, 2005) adds recall, stemming, and synonym matching. Both underperform ROUGE on summarisation and are rarely reported as the primary metric. Include METEOR if your reviewers ask; skip BLEU.

## BERTScore

BERTScore (Zhang et al., ["BERTScore: Evaluating Text Generation with BERT"](https://arxiv.org/abs/1904.09675), *ICLR 2020*) replaces lexical overlap with *contextual embedding* similarity:

1. Encode both candidate and reference with a pretrained BERT-style model.
2. For each candidate token, find the reference token with the highest embedding cosine similarity.
3. Aggregate per-token similarities into precision, recall, and F1.

```python
bertscore = evaluate.load("bertscore")
result = bertscore.compute(
    predictions=candidates,
    references=references,
    lang="en",
    model_type="microsoft/deberta-xlarge-mnli",   # strongly recommended
    batch_size=32,
)
# result["precision"], result["recall"], result["f1"] — lists of floats.
```

BERTScore's advantages:

- **Paraphrase-tolerant.** Semantically-equivalent phrases score high even if the surface forms differ.
- **Language-agnostic** (given a suitable encoder).
- **Correlates better than ROUGE with human judgement** on abstractive summarisation, especially for content adequacy.

BERTScore's costs:

- **Slower.** BERT forward passes per pair.
- **Choice of scoring model matters a lot.** `deberta-xlarge-mnli` is currently the strongest general-purpose choice; the default `roberta-large` is fine but weaker.
- **Baseline rescaling matters.** Raw BERTScore F1 is compressed into a narrow band ($\sim 0.85$–$0.95$); use `rescale_with_baseline=True` for interpretable ranges near $[0, 1]$.
- **Not a faithfulness metric.** BERTScore still compares to a *reference*. Chapter 11 covers source-conditioned metrics.

## MoverScore

MoverScore (Zhao et al., 2019) uses Earth Mover's Distance over contextual embeddings — a soft alignment between candidate and reference tokens. It is more principled than BERTScore's greedy alignment but more expensive to compute. In practice it correlates with BERTScore closely enough that BERTScore is the default.

## BLEURT

BLEURT (Sellam, Das & Parikh, ["BLEURT: Learning Robust Metrics for Text Generation"](https://arxiv.org/abs/2004.04696), *ACL 2020*) is a *learned* metric: fine-tune BERT on human ratings of generation quality. It correlates better with human judgement than any n-gram metric and often better than BERTScore on adequacy.

Trade-offs:

- **Trained on English human ratings**, so cross-lingual applicability is limited to the multilingual BLEURT-20 variant.
- **The scoring model is heavier than BERTScore's** (a fine-tuned BERT-large).
- **The output range is unbounded** — BLEURT scores are calibrated to a specific dataset's rating distribution. Cross-dataset comparisons are dangerous.

Use BLEURT when you have a small dev set and want the best-correlated automatic metric. Report alongside BERTScore for triangulation, not as a replacement.

## BARTScore

BARTScore (Yuan, Neubig & Liu, 2021) reframes evaluation as generation: score $\log p_{\text{BART}}(\text{candidate} \mid \text{source})$ under a pretrained BART. A high score means BART finds the candidate a probable summary of the source.

Advantages:

- **Source-conditioned by default.** Unlike ROUGE/BERTScore, BARTScore can be computed without a reference (useful for faithfulness, chapter 11).
- **Cheap** — one BART forward pass per pair.
- **Correlates well with human judgement**, especially on faithfulness.

Trade-offs:

- **Depends on the scoring model's quality**. A BART trained on CNN/DailyMail will score CNN/DailyMail summaries most naturally; cross-domain use is noisier.
- **Log-probability scores are unbounded and hard to interpret**. Report relative rankings, not absolute numbers.

BARTScore is worth including as a third metric in every panel. It captures faithfulness signal that ROUGE and BERTScore miss.

## The metric panel

The single most important guardrail: **never report a single metric**. A minimum panel:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** for continuity with published baselines.
- **BERTScore** (with `deberta-xlarge-mnli`, `rescale_with_baseline=True`) for paraphrase tolerance.
- **BARTScore** or **BLEURT** for a learned or source-conditioned signal.
- **A faithfulness metric** (chapter 11) — the only source-conditioned check for hallucination.
- **Length statistics** — mean and 90th-percentile output length.

Report all of them, with 95 % bootstrap confidence intervals on the dev set. Show the correlation matrix. The interesting signal is often where two metrics *disagree* — one system wins on ROUGE but loses on BERTScore, meaning it copies more; one wins on BERTScore but loses on faithfulness, meaning it paraphrases more but says something wrong.

## Multi-reference evaluation

Some datasets provide multiple references per source (e.g., TAC 2008, some Multi-News subsets). ROUGE, BERTScore, and BLEURT all support multi-reference scoring: take the max over references (per candidate) or the average. Multi-reference scores are generally higher than single-reference scores because paraphrase penalties are softened.

If you have multi-reference data, use it. Single-reference ROUGE against a paraphrasable answer is unfair.

## Confidence intervals and statistical significance

A two-point ROUGE gap between two systems is often within noise. The right test:

- Compute per-example scores for both systems on the same dev set.
- Compute the mean difference $\bar{\Delta}$.
- Bootstrap: resample dev examples with replacement, recompute the mean difference, repeat 1000 times.
- Report $\bar{\Delta}$ and its 95 % CI.
- Alternatively, run a paired permutation test.

If the CI crosses zero, the improvement is not significant. Report it anyway with the CI so the reader knows.

## SummEval and metric meta-evaluation

Fabbri et al., ["SummEval: Re-evaluating Summarization Evaluation"](https://arxiv.org/abs/2007.12626), *TACL 2021* is the reference meta-benchmark for summarisation metrics. Key findings:

- No single automatic metric correlates well with human judgement on all dimensions (coherence, consistency, fluency, relevance).
- Learned metrics (BLEURT, BERTScore with strong encoders) outperform n-gram metrics on most dimensions.
- Human evaluation remains necessary for any high-stakes claim.

When your work is being submitted to a serious review — a paper, a compliance audit, a customer-facing launch — report human evaluation alongside automatic metrics. Chapter 13 covers the protocols.

## What ROUGE and BERTScore cannot see

Two blind spots that no reference-based metric catches:

- **Faithfulness.** A summary can score high on ROUGE/BERTScore against the reference and *contradict the source*. Both metrics compare to the reference, not to the source. Chapter 11.
- **Hallucination of unstated facts.** A summary can invent an entity that appears neither in the source nor in the reference; ROUGE/BERTScore will penalise this via low overlap but will not tell you *why*.

Reference-based metrics measure "how close is your output to *this reference*". They do not measure "is your output *true*". Every serious summarisation system needs a source-conditioned faithfulness metric on top.

## Chapter summary

- ROUGE-1/2/L is the standard reference-based summarisation metric — report it, but never alone. It rewards lexical overlap and punishes paraphrase.
- BERTScore (with `deberta-xlarge-mnli` and baseline rescaling) is the paraphrase-tolerant complement. BLEURT and BARTScore add a learned or source-conditioned signal.
- Always report ROUGE, BERTScore, one learned metric, a faithfulness probe (chapter 11), and length statistics. Show the correlation matrix. Bootstrap CIs are mandatory.
- A single-metric win is often a bug in the metric, not a real improvement.
- Reference-based metrics do not measure truth. Faithfulness needs a source-conditioned metric (chapter 11) and, for anything high-stakes, human review (chapter 13).
