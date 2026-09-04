# Human Evaluation: Rubrics, Position Bias, and Demographic Slicing

## Motivation

Automatic metrics are proxies. For any consequential deployment — a summariser that lawyers will act on, a translator that patients will read, a chat assistant that customers will trust — the ground truth of quality is a human rater. Automatic metrics are the fast leading indicator; human evaluation is the arbiter.

Getting human evaluation right is a design problem, not a metric problem. A poorly-designed rubric produces low inter-rater agreement and a number no one can act on. A protocol without position-bias controls silently favours whichever system is shown first (or second, depending on the bias direction). A rater pool that under-samples the demographic or dialect groups your model will serve produces an eval that says the model is fine when it is broken for the users you did not measure.

This chapter walks the design decisions: **rubric shape** (Likert, ranking, MQM, checklist), **rubric calibration** (anchors, examples, screening), **position-bias controls** (randomisation, presentation), **demographic and dialect slicing** (how to sample raters and inputs so the aggregate is honest), and **LLM-as-judge** as a scalable — and biased — approximation of human raters. mod-107 chapter 11 covers MT-specific human eval (DA, SQM, MQM); mod-106 chapter 13 covers summarisation human eval; this chapter is the cross-cutting authority on protocol design.

## Rubric shape: pick the one that matches the decision

Different rubric shapes measure different things. Pick the shape that matches the downstream question.

### Likert / continuous scalar

Rater assigns a numeric score to each output — 1-to-5 Likert, 0-to-100 continuous slider, or a 0-to-6 anchored scale (Google's SQM — Freitag et al., ["Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation"](https://arxiv.org/abs/2104.14478), *TACL 2021*).

- **Continuous slider (0–100).** Reduces anchoring effects, produces a distribution amenable to bootstrap statistics. Standard for MT direct assessment.
- **Anchored discrete (0–6).** Easier to train raters on, plays better with ordinal statistics. Google-style SQM is 0–6 with per-value verbal anchors ("nonsense" through "perfect").
- **5-point Likert.** Familiar to raters but coarse. Widely used in industry despite mild psychometric criticism (raters cluster in the middle, ceiling effects at 5).

Report per-rater z-standardised means, then mean over raters, then mean over items — do not average raw scores across raters without standardisation (raters differ in scale usage).

### Pairwise ranking (A vs. B)

Show two outputs; the rater picks one (or "tie"). Aggregate as win rate, or (with three or more systems) fit a Bradley-Terry model to derive latent quality scores.

Two properties make pairwise ranking attractive:

- **Higher agreement** than Likert scoring. Deciding "A better than B" is easier than choosing an absolute number.
- **Better sensitivity** to small quality gaps — the ranker sees both outputs at once.

The dominant weakness: pairwise ranking does not tell you *what kind* of error either output has, only which is worse. For diagnostic evaluation, combine pairwise with an error-tag rubric (below).

### MQM and other error-typed rubrics

**MQM** (Multidimensional Quality Metrics — Lommel, Uszkoreit, Burchardt, 2014; [themqm.info](https://themqm.info/)) has the rater mark each *span* of the output with an error type and a severity. For MT: accuracy vs. fluency vs. style vs. terminology; severity minor / major / critical.

- **Diagnostic.** Tells you which error categories are your systematic problems.
- **Expert-only.** Requires trained raters — 20 % of the crowd fails calibration screening; you cannot pay Mechanical Turk rates and get useful MQM.
- **Gold-standard.** WMT metrics shared tasks now use MQM as the primary human protocol.

Freitag et al. (2021) show MQM with expert translators is the most reliable protocol for MT; equivalent structured rubrics exist for summarisation (Fabbri et al., ["SummEval: Re-evaluating Summarization Evaluation"](https://arxiv.org/abs/2007.12626), *TACL 2021*, uses a 5-dimension Likert: coherence, consistency, fluency, relevance), for QA (correctness × completeness × faithfulness — mod-105 chapter 11), and for open-ended generation.

### Checklist / criterion satisfaction

For tasks whose quality decomposes into a fixed list of criteria — "does the summary include the numbers?", "is the code syntactically valid?", "does the response cite the source?" — a per-criterion checklist gives high agreement and is directly actionable. Aggregate as per-criterion pass rate.

Reach for a checklist when the criteria are enumerable and orthogonal. For scalar-quality-space tasks (creative writing, translation fluency), Likert or MQM is stronger.

## Rubric calibration

A rubric is only as good as the raters' shared understanding of it. Calibration is not optional.

### Anchors and examples

For every scale value or error category, ship worked examples in the rubric. "5 = flawless" is under-specified; "5 = flawless; example: <this translation>" is specific. Every serious MT / summarisation rubric ships an example pack.

Ship *both* good and bad examples per value — raters need to see the boundary between adjacent values, not just an instance of each.

### Screening set

Before real rating begins, every rater runs a 20–50 item calibration set of known-difficulty items. Reject raters whose calibration correlates poorly with the majority; retrain the borderline ones.

Screening reject rates for crowd raters typically run 20–30 %; for professional translators / experts, lower. Do not skip screening — an ungated crowd pool produces noise-dominated numbers.

### Guideline iteration

Run a small pilot (100–200 items) with 3–5 raters before the full batch. Compute per-item disagreement, inspect the highest-disagreement items, refine the guideline — usually a category boundary is under-specified or an anchor is ambiguous. Publish v1 of the guideline; run pilot v2; then run the full batch.

Iterating on the guideline *after* the full batch is running is a common protocol drift and destroys comparability across batches.

## Position bias and its controls

Human raters (and LLM-as-judge, below) have systematic position bias — they favour whichever option is shown first, or second, depending on task and platform. Zheng et al. (["Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685), *NeurIPS 2023 D&B*) documented position bias in GPT-4 pairwise judging as high as 65 %-vs-35 % for identical pairs presented in different orders.

Two controls:

- **Randomise order per item.** For pairwise, flip A/B position at random. Aggregate over the correct label, not the on-screen position.
- **Present in both orders and require agreement.** Score each pair twice with orders swapped; only count as a "win" for A if both orderings pick A. Halves throughput but eliminates position bias at the analysis level.

For Likert scoring of multiple outputs of the same input, randomise the order of the outputs *and* interleave with distractors (a different input) — otherwise raters compare within-input and produce ranking-like judgments dressed as scalars.

**Presentation matters.** Fixed-position "recommended" markers, subtle differences in whitespace or font, or a system that emits shorter (more scannable) outputs will bias the rater. Match presentation across systems byte-for-byte where possible.

## Sampling: what to evaluate and by whom

### Sampling the items

- **Sample size.** 200–500 items per system per language / domain for statistical significance in system-vs-system comparison. Fewer is exploratory; more is polish.
- **Source distribution.** Sample from the *test* distribution your users see, not from a benchmark that may not reflect it. Common pattern: benchmark items (for reproducibility) + production-sampled items (for realism), reported separately.
- **Difficulty stratification.** Sample equally across easy / medium / hard (by input length, syntactic complexity, or automatic-metric disagreement). Random sampling under-weights hard items where system differences matter most.
- **Same items for all systems.** Score every system on the same items in the same batch. Between-batch comparisons are much noisier than within-batch paired comparisons.

### Sampling the raters

- **Enough raters per item.** Minimum 2; 3 for Likert / DA; 3+ for MQM. Report inter-annotator agreement.
- **Anti-fatigue.** Cap sessions at ~2 hours; rotate items across raters.
- **Demographic diversity.** For any evaluation whose downstream users vary along a dimension that plausibly affects quality perception (native language, dialect, region, gender, expertise level), the rater pool should sample that dimension. A rater pool of exclusively college-educated US-English speakers cannot honestly evaluate a hate-speech classifier for African American English users.

## Demographic and dialect slicing

Two questions:

1. **Are the inputs the model is scored on representative of the demographic / dialect groups you serve?**
2. **Are the raters representative of those groups?**

Both matter. A model that performs badly on a dialect the input distribution does not include cannot be measured by an aggregate metric that does not include it. A rater pool that misinterprets AAE sentences as "less fluent" produces a fluency judgment that is culturally biased, not model quality.

**Slicing input data.** For classification / tagging / generation tasks over demographic-sensitive content:

- **Report per-group metrics.** Per gender / dialect / region / language / age-group F1, not just aggregate. The Dashboard for Model Cards (Mitchell et al., ["Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993), *FAT\* 2019*) is the canonical framing.
- **Highlight the worst-performing group.** The aggregate is dominated by the majority; the harm is on the minority. State the worst group's number explicitly.
- **Use dialect-labelled test sets where they exist.** For English, TwitterAAE (Blodgett et al., ["Demographic Dialectal Variation in Social Media"](https://arxiv.org/abs/1608.08868), *EMNLP 2016*) is a starting point. For classification harms, the HolisticBias dataset (Smith et al., ["'I'm sorry to hear that': Finding New Biases in Language Models with a Holistic Descriptor Dataset"](https://arxiv.org/abs/2205.09209), *EMNLP 2022*) covers 13 demographic axes.

**Slicing raters.** If you can afford it, run parallel rater pools stratified by the same demographic axes as your input slicing, and report per-pool metrics. If you cannot, at minimum publish the aggregate rater-pool demographics — reader can then judge whether the eval matches their user base.

## LLM-as-judge: scalable, biased, best-alongside-humans

Prompting a frontier LLM to rate outputs is now standard for scale. Zheng et al. (2023) showed GPT-4 pairwise judging correlates ≥ 80 % with human pairwise preferences on MT-Bench.

Known biases (Zheng et al.; also Chen et al., ["Humans or LLMs as the Judge? A Study on Judgement Bias"](https://arxiv.org/abs/2402.10669), *EMNLP 2024*):

- **Position bias.** GPT-4 favours the first-presented option on identical pairs. Randomise position and check with swapped-order scoring.
- **Verbosity bias.** LLM judges prefer longer answers. Control by matching output length across systems, or explicitly penalising length in the rubric.
- **Self-preference.** GPT-4 rates GPT-4-generated outputs higher than baseline. If evaluating a GPT-4-family model, use a *different* judge (Claude, Gemini, an open model) or use human eval for that comparison.
- **Style bias.** Judges over-weight tone / formatting / fluency and under-weight factual correctness. For faithfulness-critical tasks, ground the judge with a rubric that separately scores correctness vs. fluency.

**Recommended pattern.**

- Use LLM-as-judge for **iteration** (many system variants, low per-eval cost) with a rubric similar to what your humans use.
- Use human evaluation for **shipping decisions** (final quality gate, consequential deployments).
- Correlate the two on a small sample — Spearman between LLM-judge and human-judge scores should be > 0.6 for you to trust LLM iteration signal. If not, refine the judge prompt or fall back to human eval.

The GEMBA MT protocol (Kocmi & Federmann, ["Large Language Models Are State-of-the-Art Evaluators of Translation Quality"](https://arxiv.org/abs/2302.14520), *EAMT 2023*) is a worked example: same DA / MQM rubric humans use, applied to LLM prompts, calibrated against WMT expert MQM.

## Statistical significance for human evaluation

Same tools as chapter 07:

- **Bootstrap CI** on the per-rater z-standardised system means. 95 %, 1 000 resamples.
- **Paired bootstrap** on the system-vs-system difference. If the CI on the difference crosses zero, systems are not distinguishable.
- **Report inter-annotator agreement.** Cohen's / Fleiss' / Krippendorff's alpha for categorical rubrics; Pearson / Spearman correlation for scalar. Report both raw and standardised.

Report human-eval sample sizes and IAA so the reader can judge signal-to-noise. Twenty items rated by two raters is exploratory; 300 items by 3 raters with IAA > 0.6 is publishable.

## Ethics and operations

- **Fair pay.** Rating is skilled work; expert MQM is professional-translator skilled. Crowd platforms often underpay — you get what you pay for, and it is exploitative.
- **Content sensitivity.** Warn raters about sensitive content (harmful outputs, slurs, PII), provide opt-outs, sample carefully.
- **Anonymity.** Do not publish rater identities or link ratings to identifiable raters.
- **Consent and transparency.** Raters should know what their ratings are used for and how the aggregate will be reported.
- **Bias in the rater pool as a first-class risk.** Publish rater-pool demographics; if you cannot diversify, be explicit that the eval reflects a narrow perspective.

## A minimal human-evaluation protocol checklist

- [ ] **Rubric.** Shape (Likert / pairwise / MQM / checklist), anchor examples, published guideline document.
- [ ] **Screening.** Calibration set, pass criteria, published rater-pool composition.
- [ ] **Sampling.** Item count, source distribution, difficulty stratification, per-slice sub-samples.
- [ ] **Position bias.** Randomised order per item, or two-order scoring with agreement requirement.
- [ ] **Rater assignment.** Item count per rater, raters per item, rotation policy.
- [ ] **Demographic slicing.** Per-group input samples, per-group rater pools (or explicit acknowledgment of aggregate-only).
- [ ] **Aggregation.** Per-rater standardisation, per-item mean, per-system mean; do not average raw across raters.
- [ ] **Significance.** Paired bootstrap CI on system-vs-system differences; inter-annotator agreement (Cohen's / Fleiss' / Krippendorff's alpha; correlation for scalar).
- [ ] **LLM-as-judge (if used).** Judge model, prompt template, position-bias controls, correlation with human sample.
- [ ] **Ethics.** Rater compensation, content-sensitivity warnings, anonymity, published rater-pool composition.

## Chapter summary

- Human evaluation is the arbiter for consequential deployments. Rubric shape matches the decision: Likert / DA for scalar quality, pairwise for high-agreement ranking, MQM for diagnostic error analysis, checklist for enumerable criteria.
- **Calibrate the rubric.** Anchors + worked examples + screening set + pilot-then-iterate. Rater rejection at 20–30 % is normal for crowd pools.
- **Control position bias.** Randomise order or score in both orders with agreement. Match presentation across systems byte-for-byte.
- **Sample deliberately.** 200–500 items per system, stratified across difficulty, same items for all systems in the same batch. 2–3 raters per item, 3+ for MQM.
- **Slice by demographic / dialect.** Report per-group input metrics and, where feasible, per-group rater-pool metrics. Highlight the worst-performing group. Aggregate hides harm.
- **LLM-as-judge.** Correlates well on average, biased on position / verbosity / self-preference / style. Use for iteration, human for shipping, correlate on a sample.
- **Report significance and IAA.** Paired bootstrap CIs, Krippendorff's alpha (or equivalent) for the rubric type. Twenty items with two raters is exploratory; commit to the sample size needed for the delta you want to measure.
- Ethics: fair pay, anonymity, sensitivity warnings, published rater-pool composition. A human-eval report that hides the rater pool cannot be trusted for the users it did not measure.
