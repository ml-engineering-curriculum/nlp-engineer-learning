# Statistical Significance for NLP: Bootstrap, Paired Permutation, and How Not to Fool Yourself

## Motivation

A new system that "beats the baseline by 0.5 BLEU" or "improves F1 by 1.2 points" is almost never publishable — because on a 1 000-example test set, that improvement is usually within the noise of test-set sampling. A striking result over the past decade in NLP is that many reported "improvements" evaporate when someone runs a paired significance test. Dror et al. (["The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing"](https://aclanthology.org/P18-1128/), *ACL 2018*) surveyed the ACL corpus and found that the majority of papers claiming a metric improvement did not run any significance test at all — and of those that did, many used the wrong test.

This chapter walks the tests you actually need: **bootstrap resampling** (the general-purpose CI on any metric), **paired bootstrap** (the general-purpose system-vs-system comparison), and **paired permutation / approximate randomisation** (the non-parametric go-to for arbitrary metrics). It also covers the mistakes that inflate false-positive rates in practice — multiple testing across benchmarks, running-until-significant, ignoring seed variance, and confusing "significant" with "large enough to care about."

## The problem: a metric is a random variable

The BLEU / F1 / COMET score you report is a statistic computed on a *sample* — your test set. If you had sampled a different 1 000 sentences from the same underlying population, you would have gotten a different number. Significance testing quantifies the sampling distribution: how likely is the delta you observed if the two systems are equally good?

Two flavours of question:

- **How precise is my metric?** → confidence interval on a single system's score. Answer with a bootstrap.
- **Is system A better than system B?** → hypothesis test on the paired difference. Answer with a paired bootstrap or paired permutation test.

Both are settled by resampling — no parametric assumptions about the metric's distribution, which is what you want for BLEU / F1 / COMET / SQuAD-F1 / anything where the distribution is not obviously Gaussian.

## Bootstrap resampling: the CI on one system

The bootstrap principle (Efron, 1979) is: to estimate the sampling variability of a statistic, resample your data *with replacement* many times and recompute the statistic on each resample. The distribution of those recomputed statistics is a good approximation of the sampling distribution of the statistic itself.

For NLP metrics:

```python
import numpy as np

def bootstrap_metric_ci(metric_fn, per_item_data, n=1000, alpha=0.05, seed=0):
    """
    metric_fn(subset) -> float; subset is a list/array of per-item records
    (predictions + references) sufficient to recompute the metric.
    """
    rng = np.random.default_rng(seed)
    N = len(per_item_data)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)               # sample with replacement
        subset = [per_item_data[j] for j in idx]
        scores[i] = metric_fn(subset)
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {"mean": scores.mean(), "ci_low": lo, "ci_high": hi}
```

Standard configuration: `n = 1000` resamples (more does not add signal), 95 % CI (`alpha = 0.05`).

The bootstrap works for any metric that is a function of the per-item predictions and references — F1, BLEU, chrF, COMET, EM, MCC, ROUGE, per-slice everything. It does *not* work well when the metric is not naturally per-item — e.g., corpus-level BLEU's brevity penalty is nonlinear in aggregate token counts; the paired-bootstrap approximation still works but the strict theory is looser.

## Paired bootstrap: A vs. B on the same test set

For system-vs-system comparison on the *same* test set, use the paired variant: resample the same indices for both systems each iteration, compute `metric(A) - metric(B)` on the resample, take the distribution of that difference.

```python
def paired_bootstrap(metric_fn, data_a, data_b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(data_a)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        diffs[i] = (
            metric_fn([data_a[j] for j in idx]) -
            metric_fn([data_b[j] for j in idx])
        )
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())   # two-sided
    return {
        "mean_diff": diffs.mean(),
        "ci_low":    np.quantile(diffs, 0.025),
        "ci_high":   np.quantile(diffs, 0.975),
        "p_value":   float(p),
    }
```

**If the 95 % CI on the difference crosses zero, the systems are not distinguishable at the 5 % level.**

For BLEU, SacreBLEU ships a CLI directly: `sacrebleu ref.txt --input hyp_a.txt hyp_b.txt --paired-bs -m bleu chrf` — see mod-107 chapter 10 for the pattern.

The paired variant is strictly more powerful than the unpaired variant when the two systems are computed on the same items — items vary in intrinsic difficulty and pairing removes that nuisance variance.

## Paired approximate randomisation (permutation) test

The permutation test asks: if the labels "system A" vs. "system B" are exchangeable — i.e., the systems are equally good — how often would we observe a paired difference at least as large as what we saw?

For arbitrary metrics, the exact test enumerates all $2^N$ label swaps, which is intractable for $N > 20$. **Approximate randomisation** (Yeh, ["More Accurate Tests for the Statistical Significance of Result Differences"](https://aclanthology.org/C00-2137/), *COLING 2000*) samples $K$ random swaps instead:

```python
def paired_permutation(metric_fn, data_a, data_b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(data_a)
    observed = metric_fn(data_a) - metric_fn(data_b)
    count = 0
    for _ in range(n):
        mask = rng.integers(0, 2, N).astype(bool)  # per-item swap A<->B
        a2 = [b if m else a for a, b, m in zip(data_a, data_b, mask)]
        b2 = [a if m else b for a, b, m in zip(data_a, data_b, mask)]
        if abs(metric_fn(a2) - metric_fn(b2)) >= abs(observed):
            count += 1
    p = (count + 1) / (n + 1)                       # +1 add-one smoothing
    return {"observed_diff": observed, "p_value": p}
```

`n = 10 000` swaps is standard; `n = 1 000` is acceptable for quick checks. Report the observed difference and the $p$-value.

Approximate randomisation is Riezler & Maxwell's recommended default for MT ([Riezler & Maxwell, "On Some Pitfalls in Automatic Evaluation and Significance Testing for MT"](https://aclanthology.org/W05-0908/), *ACL WS 2005*). For classification F1, McNemar's test (below) is often cited alongside; the permutation test is the more general choice.

## McNemar's test for paired binary predictions

**McNemar's test** (Dietterich, ["Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"](https://direct.mit.edu/neco/article/10/7/1895/6224), *Neural Computation 1998*) compares two classifiers on the same test set at the level of individual predictions. Build the 2×2 disagreement table:

|  | B correct | B wrong |
|---|---|---|
| A correct | $n_{cc}$ | $n_{cw}$ |
| A wrong | $n_{wc}$ | $n_{ww}$ |

The test statistic is $\chi^2 = (|n_{cw} - n_{wc}| - 1)^2 / (n_{cw} + n_{wc})$ (with continuity correction), compared to $\chi^2$ with 1 degree of freedom. When $n_{cw} + n_{wc} < 25$, use the exact binomial test on the discordant pairs.

McNemar's is cheap, appropriate for paired binary correctness (accuracy, EM), and does not naturally extend to F1 / BLEU / COMET — for those, use paired bootstrap or permutation.

## Sample-size and power planning

Statistical significance is a function of test-set size, effect size, and variance. Before running an evaluation, estimate whether your test set is large enough to detect the difference you care about.

Rule-of-thumb for classification F1 or accuracy: a 1-point delta on a 500-example test set is rarely significant; on a 5 000-example test set it usually is (subject to per-class variance). For BLEU: a 0.5-BLEU delta on 1 000 sentences is typically not significant; a 1.5-BLEU delta usually is.

If you need a formal power analysis, simulate: generate synthetic (A, B) pairs at the effect size and per-item noise you expect, run paired bootstrap, count what fraction reach $p < 0.05$. That fraction is your power at that sample size.

## Multiple testing: the trap of many benchmarks

Every benchmark suite (GLUE, MTEB, `lm-eval-harness`) reports many per-task numbers. If you run paired significance on each of 10 tasks at $\alpha = 0.05$, the family-wise false-positive rate is roughly $1 - 0.95^{10} = 0.40$. You will find "significant improvements" on 40 % of tasks by chance alone.

Two standard corrections:

- **Bonferroni.** Divide $\alpha$ by the number of tests. For 10 tasks, require $p < 0.005$. Conservative — controls family-wise error rate but reduces power.
- **Benjamini-Hochberg (FDR).** Rank $p$-values, keep the largest $k$ such that $p_{(k)} \le k \alpha / m$ where $m$ is the number of tests. Controls false discovery rate; higher power than Bonferroni.

Report the correction and the corrected thresholds. Alternatively, report all raw $p$-values and let the reader correct themselves — but state up-front that you did not correct.

## Seed variance: the elephant in the room

The metric variance from *test-set sampling* is only half the story. Training runs are stochastic — different seeds produce different models with different metrics. Reimers & Gurevych (["Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging"](https://aclanthology.org/D17-1035/), *EMNLP 2017*) and Dodge et al. (["Show Your Work: Improved Reporting of Experimental Results"](https://arxiv.org/abs/1909.03004), *EMNLP 2019*) formalised what practitioners had observed for years: single-seed comparisons are unreliable at the delta sizes typically reported.

**Best practice.**

- Report metrics as **mean ± standard deviation over ≥ 3 seeds** (5 is safer, 10 is the ideal target when compute allows).
- The delta between two systems should exceed the *combined* seed-variance envelope, not just the test-sample bootstrap CI.
- If the delta is within seed variance, the "improvement" is a training-run lucky sample.
- For very expensive models where multi-seed is infeasible, be explicit: "Single-seed run; per-seed variance not characterised" — do not silently omit.

The Dodge et al. paper also introduced the "expected performance vs. compute budget" curve — plot the expected best-of-$k$ score as a function of $k$, so the reader can compare methods at equal compute rather than at equal cherry-picking.

## What significance does not tell you

Statistical significance says "the observed delta is unlikely under the null hypothesis of no difference." It does *not* say:

- **The delta is large enough to matter.** A 0.05-BLEU delta on a 500 000-sentence test set can be significant and irrelevant. Report the effect size (mean delta and CI on the delta) alongside the $p$-value.
- **The improvement will hold on your production traffic.** Test-set distribution may not match production. Report per-domain slices; deploy with observability (mod-112 chapter on monitoring).
- **The improvement is causally attributable to the change you made.** If you retrained end-to-end, changed the seed, and changed the loss, the delta is the joint effect. Ablate one change at a time.
- **The result is reproducible.** Significance under one seed does not imply the delta will replicate. Multi-seed reporting is the real robustness signal.

## Common failure modes

- **No significance test at all.** The modal failure mode. If you claim an improvement, test it.
- **Unpaired test on paired data.** Independent two-sample tests on the same test set discard the pairing structure and lose power. Use paired bootstrap or paired permutation.
- **Ignoring multiple testing across benchmarks.** 10 tasks × 5 % → 40 % family-wise false positive. Correct with Bonferroni or FDR, or report raw and note the number of comparisons.
- **Single seed.** Especially for fine-tuning experiments — seed-to-seed variance often exceeds the claimed effect.
- **Running until significant.** Adding more test data until $p < 0.05$ is $p$-hacking. Pre-register the test-set size or plan a power analysis.
- **Reporting only the $p$-value, no effect size.** A significant delta of 0.02 is not the same story as a significant delta of 2.0. Report both.
- **Confusing "no significant difference" with "the systems are equivalent."** Absence of evidence is not evidence of absence — a wide CI that contains zero also contains meaningful differences. If you want to *prove* equivalence, use an equivalence test (TOST — Lakens, 2017).

## A minimal significance-reporting checklist

For every system-vs-system claim in an evaluation report:

- [ ] Test set size and composition documented.
- [ ] Metric implementation cited and signature / checkpoint recorded (chapters 02–05).
- [ ] Per-item scores (or metric on paired resamples) computed for both systems.
- [ ] 95 % CI on each system's score via bootstrap resampling.
- [ ] 95 % CI on the paired difference via paired bootstrap.
- [ ] $p$-value from paired permutation (or paired bootstrap two-sided).
- [ ] For classification with binary correctness: McNemar's test alternative.
- [ ] Multi-seed variance reported for both systems (≥ 3 seeds); delta compared to combined seed envelope.
- [ ] Multiple-testing correction applied or explicitly deferred.
- [ ] Effect size reported alongside significance ("system A is +1.4 BLEU (95 % CI [0.6, 2.2], p < 0.01) over system B").

## Chapter summary

- The metric you report is a random variable. Report confidence intervals from bootstrap resampling (1 000 samples, 95 % CI) alongside the point estimate.
- **Paired bootstrap** for system-vs-system deltas on the same test set — resample the same indices for both systems, take the CI on the difference. If the CI crosses zero, not significant.
- **Paired approximate randomisation (permutation)** for arbitrary metrics — swap labels between systems with probability 0.5 many times, count how often the swapped difference exceeds observed.
- **McNemar's test** for two binary classifiers on the same test set — cheap and appropriate for paired correctness; not portable to F1 / BLEU.
- Correct for multiple testing across benchmarks: Bonferroni (conservative) or Benjamini-Hochberg (FDR-controlling).
- Report **multi-seed variance** (≥ 3 seeds); delta between systems should exceed the combined seed envelope, not just the bootstrap CI.
- Statistical significance ≠ practical significance. Report the effect size and CI on the delta; interpret whether the size matters for your use case.
- The reproducibility signal is multi-seed + reported CIs + pinned metric implementations + a paired significance test on the delta. The absence of any one of these is a reason to distrust the number.
