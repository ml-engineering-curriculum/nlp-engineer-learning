# Human Evaluation for MT: DA, SQM, and MQM

## Motivation

Automatic metrics — even the learned ones from chapter 10 — are an approximation of what a human would say about a translation. For consequential deployments (legal, medical, patient-facing, contractual), the automatic score is a monitoring tool; the ground truth is a human rater. This chapter walks the three protocols that dominate MT human evaluation — **Direct Assessment (DA)**, **Scalar Quality Metric (SQM)**, and **Multidimensional Quality Metrics (MQM)** — plus the annotator, sampling, and reliability design decisions that make the numbers you report actually mean something.

The MT field learned the hard way: automatic scores can improve while human-perceived quality regresses, and vice versa. WMT's yearly metric shared tasks (Freitag et al., ["Results of the WMT22 Metrics Shared Task"](https://aclanthology.org/2022.wmt-1.2/), *WMT 2022*) show sustained cases where COMET, chrF, and BLEU disagree on system rankings. Human evaluation is the arbiter; automatic evaluation is the leading indicator.

## The three protocols

### Direct Assessment (DA)

**Direct Assessment** (Graham et al., ["Continuous Measurement Scales in Human Evaluation of Machine Translation"](https://aclanthology.org/W13-2305/), *ACL 2013*; refined for WMT 2016+) asks the rater to score each translation on a 0–100 continuous slider given the source (or reference), with the rating question phrased as "how adequately does the candidate express the meaning of the source?" Design details that matter:

- **Continuous scale, not Likert.** Sliders reduce anchoring effects and produce a distribution amenable to statistical analysis.
- **Reference-based or source-based.** Source-based DA (SR-DA) shows the rater the source and asks them to judge; reference-based DA additionally shows a human reference. WMT has moved toward source-based (fewer rater biases from the reference).
- **Per-segment.** Ratings are per-segment (sentence), not per-document.
- **Bilingual raters required.** Raters must be fluent in both source and target. Monolingual-target-only raters can rate *fluency* but not adequacy.
- **Standardisation and outlier removal.** Scores are z-standardised per rater; raters whose score distribution correlates poorly with the majority are filtered.

DA is the WMT default protocol for shared-task human evaluation.

### Scalar Quality Metric (SQM)

Google's variant (Freitag et al., ["Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation"](https://arxiv.org/abs/2104.14478), *TACL 2021*) is a **0–6 discrete scale** with anchored labels: 0 = "nonsense / no meaningful information," 6 = "perfect." Middle values are anchored ("understandable but wrong in places," etc.). More constrained than DA but easier to train raters on; the discrete scale plays better with ordinal statistics.

SQM tends to be preferred when raters are non-experts (crowdworkers); DA when raters are professional translators.

### Multidimensional Quality Metrics (MQM)

**MQM** ([Lommel, Uszkoreit, Burchardt, 2014](https://themqm.info/); the current QT21 / GALA operationalisation) is a full error-typology annotation. Instead of a single score, raters mark each translation error with:

- **Error span** — the substring of the target that has an error.
- **Category** — from a hierarchy: *Accuracy* (mistranslation, addition, omission, untranslated), *Fluency* (grammar, register, spelling, punctuation), *Style*, *Terminology*, *Locale convention*, *Design / non-translatable*.
- **Severity** — Minor / Major / Critical.

An MQM score aggregates over spans with severity-weighted penalties (typically 1 point for Minor, 5 for Major, 10 for Critical, though the weights are configurable per project). Freitag et al. (2021, 2022) show MQM with expert translators as the most-reliable human-evaluation protocol for MT — the gold standard when budget allows.

**When to reach for MQM:**

- The output must meet a quality bar defined by error type (legal, medical, high-stakes UI).
- You need diagnostic information — *what kinds* of errors is my system making — rather than just a rank.
- You have access to professional translators as raters.

**When DA / SQM is enough:**

- You need a system-vs-system comparison and MQM cost is prohibitive.
- You are evaluating for a general-purpose deployment where any error type is roughly as bad as any other.

## Sampling: what to evaluate

Human evaluation is expensive; you must sample carefully.

- **Sample size.** 200–500 segments per system per language pair is the WMT-track minimum. For statistical significance in system-vs-system comparison, 500+ is safer.
- **Source distribution.** Sample from the *test* distribution you care about, not from a benchmark that may not reflect production traffic. A common pattern: FLORES-200 devtest for reproducibility + 200 production-sampled segments for realism, reported separately.
- **Difficulty stratification.** Sample equally across easy / medium / hard segments (by length, syntactic complexity, or by whether an automatic metric ranked one system near the other). Random sampling under-weights hard segments where system differences matter most.
- **Multiple systems in the same batch.** Show all systems' outputs for the same source, in randomised order, to the same rater. Between-rater comparisons on separate batches are much noisier than within-rater comparisons on the same batch.

## Rater design

Getting the raters right is the whole game.

- **Bilingual.** The rater must be genuinely fluent in both languages. "I studied Spanish in college" is not enough for professional MT rating.
- **Native in the target.** Adequacy rating tolerates non-native target speakers; fluency rating does not. Prefer native target speakers for MQM.
- **Number of raters per segment.** Two raters at minimum; three for MQM (majority-vote severity). Report inter-annotator agreement.
- **Screening.** Short calibration set with known-difficulty items before the main batch. Rejected raters are common — WMT operations typically screen out 20–30 % of crowd raters.
- **Training and calibration.** For MQM, expect a 1–2 hour training on the error typology and a calibration set of 50–100 segments before real rating begins.
- **Anti-fatigue.** Cap sessions at ~2 hours; rotate raters across languages if possible.

## Inter-annotator agreement

Report the raters' agreement so downstream readers can judge signal-to-noise.

- **DA / SQM:** Pearson or Spearman correlation between rater pairs on the same segments. Anything above ~0.5 is respectable; the target is > 0.7.
- **MQM:** F1 on error span identification (does rater A's span overlap with rater B's?) and Cohen's / Krippendorff's alpha on severity labels once spans are aligned.
- **Filtered corpus.** Report both raw and standardised (per-rater z-scored) numbers.

Low agreement is not necessarily a rater problem — it can indicate that the task itself is under-specified. For MQM, ambiguous severity boundaries are a common source; sharpen the guidelines.

## Aggregating rater scores

- **DA and SQM.** Per-rater z-standardise, then mean over raters, then mean over segments. Report system-level mean with paired bootstrap CI over segments.
- **MQM.** Sum severity-weighted penalties per segment, then mean over segments (lower is better). Alternatively, per-error-category breakdowns.

**Do not average raw scores across raters without standardisation.** Rater-scale differences (some raters use the full 0–100; others cluster around 40–60) dominate the numbers.

## Statistical significance

For DA / SQM system-vs-system comparison, use **paired bootstrap** over segments (same tool as SacreBLEU's `--paired-bs`, adapted to arbitrary scores):

```python
import numpy as np
def paired_bootstrap(scores_a, scores_b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(scores_a))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(np.mean(scores_a[s]) - np.mean(scores_b[s]))
    diffs = np.array(diffs)
    return {
        "mean_diff": diffs.mean(),
        "ci_95_low": np.percentile(diffs, 2.5),
        "ci_95_hi":  np.percentile(diffs, 97.5),
        "p_value":   float((diffs <= 0).mean()) * 2,  # two-sided vs. 0
    }
```

Report the CI and the $p$-value. If the CI crosses zero, the systems are not distinguishable.

## Design of a production evaluation cycle

A rhythm that works for production MT:

1. **Weekly / monthly automatic panel** on a fixed test set (FLORES-200 + your internal test set). Track SacreBLEU / chrF / COMET / COMET-Kiwi trendlines. Alert on regressions.
2. **Per-release human evaluation** — 300–500 segments per language pair, DA / SQM by professional raters. Compare new-version vs. previous-version with paired bootstrap.
3. **Quarterly MQM audit** — smaller sample (100–200 segments) with expert translators, focused on the top-N production language pairs and any pair flagged by the weekly automatic panel.
4. **Ad-hoc MQM on high-stakes changes** — new terminology system, new domain adaptation, new base model. Do not ship without a human eye on quality across error categories.

Store the eval outputs, source, and rater metadata in a structured schema — you will want to re-analyse them when a metric changes or a system evolves.

## Ethical and operational concerns

- **Rater fair pay.** Multilingual professional rating is skilled work; pay accordingly. Crowdsourcing platforms often underpay MT raters — you get what you pay for and it is exploitative.
- **Content sensitivity.** MT test sets can contain slurs, harmful content, or personal information. Warn raters, provide opt-outs, and screen the sample before annotation.
- **Rater privacy.** Do not tie rater identities to public rating outputs. Anonymise before publication.
- **Bias.** Raters have preferences (register, dialect, formality). Report rater demographics at the group level when possible; be transparent about the rater pool composition.

## LLM-as-judge for MT

Prompting a strong LLM (GPT-4-scale) to score translations is now a reasonable *approximation* of human evaluation for high-resource pairs. **GEMBA** (Kocmi & Federmann, ["Large Language Models Are State-of-the-Art Evaluators of Translation Quality"](https://arxiv.org/abs/2302.14520), *EAMT 2023*) formalised the pattern; correlations with expert MQM on WMT are competitive for the high-resource tail.

Caveats:

- **Low-resource languages.** LLM-as-judge quality drops sharply for languages the base model saw little of.
- **Cost.** Non-trivial per-segment at scale.
- **Correlated failure modes.** An LLM judge will systematically over-rate translations that look fluent; the same failure mode fluency-based metrics have always had. Do not use LLM-as-judge alone for high-stakes evaluation.

For iterative development, GEMBA-style LLM judging is a useful third metric alongside COMET. For consequential shipping decisions, back it up with human eval.

## Common failure modes and their fixes

- **"Rater agreement is low, so the results are unreliable."** Sometimes true — but low agreement can also mean the *task* is ambiguous. Inspect divergent items and consider guideline refinement.
- **"Two-rater DA average showed system B better."** With only two raters, one outlier drives the number. Use three raters or larger sample.
- **"Automatic COMET says A is better, human MQM says B is better."** Trust the MQM (it's the ground truth). Investigate whether COMET is over-rating fluent hallucinations; add reference-free QE and hallucination-detection to the pipeline.
- **"We averaged DA across all languages and got a score."** Aggregate averages are meaningless across languages with different rater pools and rating scales. Report per-language.
- **"Our raters are crowdworkers with two hours of MT experience."** For MQM, this will not work. Downgrade to SQM or invest in professional translators for MQM.

## Chapter summary

- Three protocols dominate MT human evaluation: **DA** (0–100 continuous slider, WMT default), **SQM** (0–6 anchored discrete, Google-style), and **MQM** (error-typed span annotation, gold-standard for consequential deployments).
- Reach for MQM when you have professional translators and need diagnostic error-type information; DA / SQM when you need a system-vs-system rank at lower cost.
- Sample 200–500 segments per system per language pair, stratified across difficulty; show all systems in randomised order to each rater; use 2–3 raters per segment.
- Per-rater z-standardise before averaging; report per-language, never aggregate averages across a heterogeneous language set.
- Paired bootstrap over segments for system-vs-system significance. If the 95 % CI on the difference crosses zero, the systems are not distinguishable.
- Rhythm: weekly automatic panel, per-release human DA/SQM, quarterly MQM audit, ad-hoc MQM on high-stakes changes.
- LLM-as-judge (GEMBA) is a fair *approximation* for high-resource pairs; not a substitute for MQM on consequential deployments.
- Pay raters fairly; anonymise; screen for content sensitivity.
