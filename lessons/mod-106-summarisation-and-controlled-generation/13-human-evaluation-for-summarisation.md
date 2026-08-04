# Human Evaluation for Summarisation

## Motivation

Reference-based metrics (chapter 10) and faithfulness probes (chapter 11) get you 80 % of the way. The last 20 % is the one that stops product launches, ends academic reviews, and turns into compliance findings — and none of the automatic metrics can get you there alone. The SummEval meta-benchmark (Fabbri et al., 2021) is unambiguous: no single automatic metric correlates well with human judgement across all dimensions of summary quality.

This chapter is about the human evaluation you *should* be running on any summariser you plan to ship or publish. It covers the four dimensions that matter, the protocols to score them, the sample-size math, LLM-as-judge as a scalable approximation, and the failure modes of both human and LLM rating.

## The four dimensions

The community has converged on four axes for summary quality (Fabbri et al., 2021; Kryściński et al., 2019):

- **Coherence.** Does the summary read as a well-organised, structured piece of text, not a bag of disconnected sentences?
- **Consistency (faithfulness).** Is every claim in the summary supported by the source? No contradictions, no inventions.
- **Fluency.** Is the summary grammatical, well-formed, and free of typos or artefacts?
- **Relevance.** Does the summary cover the important content from the source, and only the important content?

These are (mostly) independent axes. A summary can be fluent-but-incoherent (grammatical sentences that do not flow), consistent-but-irrelevant (accurately restates trivia from the source), coherent-but-inconsistent (a fluent, well-organised lie).

Rate all four separately. A single "overall quality" score conflates them and gives you no signal on which dimension to improve.

## Rating scales

Two rating regimes dominate.

### Likert scales

Five-point (1–5) or seven-point (1–7) scales per dimension. Simple, cheap, and well-understood.

```
Coherence:   1 (very poor) — 2 — 3 — 4 — 5 (very good)
Consistency: 1 — 2 — 3 — 4 — 5
Fluency:     1 — 2 — 3 — 4 — 5
Relevance:   1 — 2 — 3 — 4 — 5
```

Known pathology: raters cluster ratings in the middle ("central tendency bias"), which compresses the effective scale. Anchoring each point with a concrete description (a "rubric") helps.

### Pairwise / A/B

Given two summaries of the same source, which is better on each dimension? Or: pick the better summary overall. Pairwise comparisons produce higher-agreement judgements than Likert because raters do not need to calibrate an absolute scale — they only need to decide "is A better than B?".

Pairwise is the standard when you are comparing two systems (a candidate vs. a baseline, a model before vs. after a mitigation). Likert is the standard when you need per-system absolute scores.

## Pyramid method (content-focused)

Nenkova & Passonneau, ["Evaluating Content Selection in Summarization: The Pyramid Method"](https://aclanthology.org/N04-1019/), *NAACL 2004* is a heavier-weight protocol for evaluating content selection specifically:

1. Multiple human annotators write reference summaries.
2. Annotators identify "Summary Content Units" (SCUs) — atomic content pieces — in each reference. Each SCU gets a weight equal to the number of references that included it.
3. A candidate summary is scored by the sum of SCU weights it covers, normalised by the maximum possible.

Pyramid produces a well-calibrated content-selection score but is expensive. The lightweight variant PyrEval (Gao, Sun & Passonneau, 2019) automates most of it.

Use Pyramid when content selection is the primary evaluation axis and you can afford the annotation cost. Skip it if consistency and fluency dominate your product concern.

## Sample-size math

How many summaries do you need to rate for a meaningful comparison? The answer depends on the effect size you want to detect and the rating variance.

Rules of thumb for pairwise comparisons:

- **50 pairs** — detect a 20-point win-rate gap with reasonable confidence.
- **200 pairs** — detect a 10-point gap.
- **500 pairs** — detect a 5-point gap.

For Likert on independent samples, budget ~200 examples per system for a stable per-dimension mean with confidence intervals ~$\pm 0.15$ on a 5-point scale.

Multiple raters per example (typically 3) trade higher cost for lower variance. Report inter-annotator agreement (Krippendorff's α, Fleiss' κ, or the correlation of individual raters with the mean) alongside the aggregate.

## Inter-annotator agreement

If your raters disagree, your evaluation is unreliable at any sample size.

- **Krippendorff's α** for ordinal ratings (Likert): $\geq 0.67$ is "acceptable"; $\geq 0.8$ is "good". Below $0.6$, either your rubric is unclear or the task is genuinely subjective.
- **Cohen's / Fleiss' κ** for categorical labels: same interpretation, different formula.
- **Pearson / Spearman correlation of individual raters vs. the mean** — a diagnostic for outlier raters.

Low agreement is a signal to *fix the rubric*, not to average harder. Rewrite the anchor descriptions with worked examples; retrain the raters; re-run.

## LLM-as-judge — scalable approximation

Ask a strong LLM (Claude 3.x, GPT-4.x) to rate summaries with the same rubric a human would use. Zheng et al., ["Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685), *NeurIPS 2023* is the reference; G-Eval (Liu et al., 2023) is a common protocol for summarisation specifically.

```
[System prompt]
You are evaluating a summary of a source document. Rate the summary on:
  - Coherence (1–5)
  - Consistency (1–5)
  - Fluency (1–5)
  - Relevance (1–5)
Return a JSON object with the four scores and a one-sentence justification for each.

[User]
Source: <source>
Summary: <candidate>
```

LLM-as-judge is:

- **Cheap and fast.** Pennies per rating, seconds per rating.
- **Consistent within a judge.** The same judge on the same rubric gives similar scores.
- **Well-correlated with humans on aggregate.** Averaged over hundreds of samples, LLM scores usually match human scores well enough to compare systems.

LLM-as-judge is *not*:

- **A replacement for humans in high-stakes evaluations.** Peiyi Wang et al., ["Large Language Models are not Fair Evaluators"](https://arxiv.org/abs/2305.17926), *ACL 2024 Findings* documents well-known biases: position bias (A vs. B ratings depend on the presentation order), length bias (longer summaries score higher), self-preference bias (a judge prefers outputs that resemble its own).
- **Unbiased across dimensions.** Judges score consistency and fluency well; relevance and coherence noisily.
- **Suitable for evaluating summaries generated by *the same model* as the judge.** Use a different-family judge or you will systematically over-score.

Mitigations for LLM-as-judge biases: shuffle presentation order, control for length by including length as a variable in the analysis, use two judges from different families and average.

## G-Eval

Liu et al., ["G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"](https://arxiv.org/abs/2303.16634), *EMNLP 2023* is a specific LLM-as-judge protocol that scores each dimension with a rubric of *chain-of-thought* steps and takes the *expected* score under the model's output distribution. Correlates better with humans than naive LLM-as-judge on summarisation.

## Human evaluation cost

Realistic numbers:

- **In-house SME reviewers.** $2–$5 per rating for standard news / dialogue text; higher for specialist domains (legal, medical). A 200-example pairwise study with 3 raters/example: $1200–$3000.
- **Crowd workers (Prolific, MTurk).** $0.20–$1.00 per rating. Lower quality; more variance; needs careful qualification tests.
- **LLM-as-judge.** ~$0.01 per rating for GPT-4-class, less for smaller models.

The right stack: **LLM-as-judge for every A/B, human for the launch and the milestone claim.** Never publish or ship on LLM ratings alone.

## Building a defensible evaluation report

For any serious summariser evaluation, produce:

- **The rubric.** Written definitions of each dimension with 1-point and 5-point anchor examples.
- **The sample.** 200+ examples per system, stratified by input length and domain.
- **The raters.** 3+ raters per example; report inter-annotator agreement.
- **The results.** Per-dimension mean with 95 % CIs; win-rate for pairwise; correlation with each automatic metric on the labelled subset.
- **Failure examples.** 10–20 low-scoring predictions per system, with the rating breakdown and rater comments.

This report is the artefact that stands up to scrutiny. Anything less is a marketing chart.

## Failure modes of human evaluation itself

Human evaluation is not magic. Its own failure modes:

- **Rubric drift.** Raters interpret the anchors differently over time. Ship the rubric; retrain quarterly.
- **Distraction / speed.** A rater doing 100 pairs in an hour is not doing the same task as a rater doing 20 in an hour.
- **Domain expertise mismatch.** A general-audience rater cannot evaluate a medical summary's consistency.
- **Ordering effects.** Rating summary B after A is not the same task as rating B alone. Randomise, or use pairwise deliberately.
- **Ceiling effects.** Modern summarisers often score 4–5 on all dimensions on easy corpora. Move to harder corpora (XSum, dialogue, long-doc) to get discriminating signal.

Every one of these is a rubric or process problem, not a rater problem. Fix the process.

## When human evaluation is required

Rule of thumb: any claim that carries a business, safety, or compliance risk requires human evaluation on a defensible sample.

- Publishing a paper claiming a new SOTA — required.
- Launching a production summariser in a regulated domain — required.
- A/B testing decoding strategies on a mature system — LLM-as-judge is usually enough.
- Iterating on prompts during development — automatic metrics + spot-checks are enough.

Match the evaluation rigour to the stakes.

## Chapter summary

- Rate summaries on four independent dimensions: coherence, consistency, fluency, relevance. A single "overall" score conflates them and loses signal.
- Likert scales for per-system absolute scores; pairwise A/B for comparisons. Pairwise produces higher agreement.
- Pyramid method for detailed content-selection evaluation; expensive but well-calibrated.
- Budget 200+ examples per system per comparison; 3 raters per example; report inter-annotator agreement (Krippendorff's α ≥ 0.67 is the target).
- LLM-as-judge scales cheaply and correlates with humans on aggregate. Not a substitute for human evaluation on high-stakes claims. Mitigate position, length, and self-preference biases.
- G-Eval is the recommended LLM-as-judge protocol for summarisation.
- Ship the rubric, the sample, the raters, the results, and failure examples. Anything less is marketing, not evaluation.
