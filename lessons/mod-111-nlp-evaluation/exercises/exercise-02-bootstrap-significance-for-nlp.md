# exercise-02: Bootstrap Significance for NLP

**Estimated effort:** 2 hours

## Objective

Implement a **reusable significance-testing toolkit** for NLP metrics — bootstrap CI, paired bootstrap, paired approximate randomisation, and McNemar's test — and use it to re-examine three real system-vs-system comparisons on public benchmarks. Deliver evidence that at least one of your three comparisons is not significant despite a headline metric delta that would casually be called "an improvement."

## Prerequisites

- Chapter [07](../07-statistical-significance-for-nlp.md).
- Python 3.10+; `numpy`, `scipy`, `datasets`, `evaluate`, plus whatever task metric you need for the three comparisons.
- Optional: `sacrebleu` for its built-in `--paired-bs` to cross-check your implementation.

## Problem statement

### Part A — The toolkit

Build `sigtest/` — a small package with the following API:

```python
# sigtest/bootstrap.py
def bootstrap_ci(metric_fn, per_item_data, n=1000, alpha=0.05, seed=0):
    """Return {'mean': m, 'ci_low': lo, 'ci_high': hi, 'samples': array}."""

def paired_bootstrap(metric_fn, data_a, data_b, n=1000, alpha=0.05, seed=0):
    """Return {'mean_diff', 'ci_low', 'ci_high', 'p_value', 'samples'}."""

# sigtest/permutation.py
def paired_permutation(metric_fn, data_a, data_b, n=10000, seed=0):
    """Return {'observed_diff', 'p_value', 'null_samples'}."""

# sigtest/mcnemar.py
def mcnemar(correct_a, correct_b, use_exact_when_small=True):
    """Return {'n_cw', 'n_wc', 'chi2', 'p_value', 'method': 'chi2'|'exact'}."""
```

Requirements:

- **Reproducible under `seed`.** Same seed → identical samples. Use `numpy.random.default_rng(seed)`, not the legacy global RNG.
- **`metric_fn` is a callable taking a list of per-item records** — so the toolkit is metric-agnostic. Predictions and references live in the record.
- **Handle empty resamples gracefully.** For metrics that error on zero-support classes (macro-F1 with `zero_division=1`), pass through the flag; document.
- **Small (< 300 lines total).** This is a toolkit, not a framework.

Ship `tests/` with at least six unit tests:

- `bootstrap_ci` on a known-analytic case (mean of a Bernoulli sample; the CI should approach the Wilson interval for large `n`).
- `paired_bootstrap` on two identical systems (mean diff should be zero, CI should straddle zero, $p$ ≈ 1).
- `paired_bootstrap` on two synthetic systems with a constructed effect (correctly reject at $p < 0.05$).
- `paired_permutation` on the same two cases as above.
- `mcnemar` on Dietterich's ([Neural Computation 1998](https://direct.mit.edu/neco/article/10/7/1895/6224)) worked example — check the $\chi^2$ statistic matches.
- Determinism: same seed twice → same output; different seeds → different output.

### Part B — Cross-check against SacreBLEU

For a small pair of MT hypothesis files (nine sentences is enough for a smoke test; use FLORES-200 devtest first 100 en→de translated by two off-the-shelf models — same as exercise-01 B4), compute the paired bootstrap on BLEU using `sacrebleu --paired-bs` and using your `paired_bootstrap(sacrebleu_corpus_bleu, hyps_a, hyps_b)`. Your CI and $p$-value should be within Monte-Carlo variance of SacreBLEU's (they will not be identical because SacreBLEU uses its own RNG and default sample count).

Save the comparison as `sacrebleu_crosscheck.md` with both outputs and a one-paragraph "close enough — here's why" write-up.

### Part C — Three real comparisons

Use your toolkit to re-examine three system-vs-system comparisons. For each, report a full significance analysis:

**C1: Classification on GLUE MRPC.** Fine-tune (or use pretrained) `bert-base-uncased` and `roberta-base` on MRPC dev. Report accuracy + F1 for each system, bootstrap CIs, paired bootstrap on the difference, paired permutation on the difference, and McNemar on binary correctness. Include a `report.md` that says whether the two are distinguishable on the MRPC dev set, and — importantly — whether the deltas you observe are consistent with the deltas reported in the original RoBERTa paper. If not, explain (test-set overlap, seed variance, fine-tuning hyperparams).

Save as `c1_mrpc/`.

**C2: MT on FLORES-200 devtest.** Score two MT systems (e.g., `Helsinki-NLP/opus-mt-en-de` vs. `facebook/nllb-200-distilled-600M`) on eng→deu devtest. Report SacreBLEU, chrF, and COMET (`Unbabel/wmt22-comet-da`) with 95 % bootstrap CIs and paired-bootstrap difference on all three. Show the case: **either** the BLEU delta is significant while the COMET delta is not (or vice versa), **or** all three agree — either result is publishable, but the write-up must reason about *why*.

Save as `c2_mt/`.

**C3: Multi-seed classification.** Take a small classification task (SST-2 dev, 872 examples) and a light model (`distilbert-base-uncased`). Fine-tune the *same* model architecture and hyperparameters with **5 different seeds** to produce 5 versions of "system A." Do the same with 5 seeds for "system B" — a different model (`bert-base-uncased` or `roberta-base`). Report:

- Mean and std of accuracy across seeds for A and for B.
- 95 % CI on each seed's accuracy via bootstrap (per-seed test-set CI).
- Paired-bootstrap comparison between the *mean-per-seed* accuracy across the 5 seed pairings (approach 1) *and* between the best-seed of A and the best-seed of B (approach 2).
- Show whether the delta between A and B *exceeds* the combined seed-variance envelope.

The lesson: single-seed comparisons can be misleading when seed variance is comparable to model-vs-model variance. Your `report.md` must state whether the "which model wins" answer changes between the single-best-seed and mean-per-seed protocols.

Save as `c3_multiseed/`.

### Part D — The "not significant" write-up

At least one of C1 / C2 / C3 must produce a case where the point-estimate delta is what a casual reader would call an improvement (a few tenths of a point up to a few points depending on the task), but the paired significance test says the two systems are not distinguishable. Write `not_significant.md` (≤ 300 words) documenting that case: point deltas, CIs, $p$-value, and a plain-English "what a rigorous report would say instead of 'our model beats theirs by X.'"

If none of C1 / C2 / C3 spontaneously produces such a case (unlikely on your first pass), reduce the test-set size or increase the seed noise until it does — the exercise is about seeing what the significance test tells you when the honest answer is "cannot tell." Document the setup you had to adjust.

## Starter guidance

- **Random seeds are load-bearing.** Every RNG call must be threaded from the top-level `seed` argument. Do not silently seed from `time.time()`.
- **Bootstrap `n = 1000` for CIs; permutation `n = 10 000` for $p$-values.** More does not help; less risks noisy tails.
- **For BLEU / chrF / COMET, the metric is defined at the corpus level.** Your paired-bootstrap wrapper needs to pass the *resampled sentence list* to the corpus metric, not per-sentence metrics averaged.
- **Multi-seed fine-tuning is compute-heavy.** For C3, a `distilbert-base-uncased` on SST-2 fine-tune takes ~5 min on a single GPU per seed. Budget accordingly; do not run 20 seeds if 5 is enough.
- **Cache predictions.** Do not re-run models to score under different bootstrap seeds. Save per-item predictions once; run all significance tests on the cached predictions.
- **When in doubt, verify against SacreBLEU's paired-bs.** It is a well-tested implementation of the same paired-bootstrap on the same MT metrics.

## Acceptance criteria

- [ ] `sigtest/` package with `bootstrap_ci`, `paired_bootstrap`, `paired_permutation`, `mcnemar`, and ≥ 6 passing unit tests including two synthetic sanity checks and two identical-system sanity checks.
- [ ] `sacrebleu_crosscheck.md` showing your toolkit's `paired_bootstrap` on BLEU matches SacreBLEU's `--paired-bs` within Monte-Carlo variance.
- [ ] Three worked comparisons (`c1_mrpc/`, `c2_mt/`, `c3_multiseed/`) with per-metric bootstrap CIs, paired-bootstrap differences, paired-permutation $p$-values, McNemar (where applicable), and a `report.md` interpreting each.
- [ ] For C3, mean ± std over ≥ 5 seeds for both A and B, and an explicit statement of whether the "which wins" answer flips between single-best-seed and mean-per-seed protocols.
- [ ] `not_significant.md` documenting one case where the point-estimate delta is not statistically significant, with the correct interpretation.
- [ ] All scripts run under a single seed argument (`--seed 0`) and produce deterministic output.

## Stretch goals

- **Effect-size reporting.** Add Cohen's $d$ (for scalar metrics) or a normalised version for F1 to your significance output. Include in `report.md` for each comparison.
- **Multiple-testing correction.** For a benchmark suite with $k$ tasks, implement Bonferroni and Benjamini-Hochberg corrections and apply to a 5-task comparison (e.g., 5 GLUE tasks).
- **Power analysis.** For C1, simulate: at what test-set size does the C1 delta become significant with 80 % power? Use bootstrap over the pooled distribution.
- **Equivalence testing (TOST).** For a case where paired bootstrap says "no significant difference," run a TOST equivalence test with a domain-appropriate equivalence margin. Report both.
- **Randomisation baseline.** Compare the empirical false-positive rate of your `paired_permutation` at $p < 0.05$ under the null (two identical systems) to the nominal 5 % — over 500 replications.

## Deliverables

```
sigtest/
    __init__.py  bootstrap.py  permutation.py  mcnemar.py
tests/                  test_bootstrap.py  test_permutation.py  test_mcnemar.py
sacrebleu_crosscheck.md
c1_mrpc/                run.py  report.md  predictions.json
c2_mt/                  run.py  report.md  scores.json
c3_multiseed/           run.py  report.md  seed_runs/*.json
not_significant.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
