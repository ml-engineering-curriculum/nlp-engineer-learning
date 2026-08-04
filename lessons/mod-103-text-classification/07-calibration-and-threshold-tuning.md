# Calibration and Threshold Tuning for Cost-Sensitive Deployment

## Motivation

Two connected problems separate a classifier that scores well on validation from one that behaves well in production:

1. **Calibration.** The number your model emits as `p(class = 1) = 0.9` should mean "of all inputs where I say 0.9, roughly 90 % are actually class 1." Modern neural classifiers, and especially fine-tuned transformers, are systematically miscalibrated — they emit overconfident probabilities. Guo, Pleiss, Sun & Weinberger, ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599), *ICML 2017*, is the paper you should read once and reread.
2. **Threshold choice.** Even a perfectly calibrated model has to decide *at what probability* to fire. The default `p > 0.5` is a convention that presumes equal cost of false positives and false negatives, which almost no product actually has.

This chapter covers both. The methods are simple, cheap, and buy real quality on real deployments. Skipping them ships a model that looks good on the leaderboard and misfires on the traffic.

## What "calibrated" means, precisely

A classifier is perfectly calibrated if, for every emitted probability `p`:

```
P(y = 1 | model says p̂ = p) = p
```

You measure calibration with:

- **Reliability diagram.** Bin the emitted probabilities into (say) 10–20 bins by predicted probability, then plot the average predicted probability in each bin against the empirical fraction of positives in that bin. Perfect calibration = the diagonal.
- **Expected Calibration Error (ECE).** The weighted average absolute gap between bin-mean predicted probability and bin-mean empirical accuracy:
  ```
  ECE = Σ_b (|B_b| / N) · |acc(B_b) - conf(B_b)|
  ```
  Guo et al. define it. Range `[0, 1]`; lower is better. A fine-tuned encoder often lands at ECE `0.05`–`0.15` before calibration; a good target is `< 0.02`.
- **Brier score.** Mean squared error between predicted probabilities and one-hot labels. Decomposable into calibration + refinement components (Brier, "Verification of forecasts expressed in terms of probability", *Monthly Weather Review* 78:1, 1950). Useful because it is a proper scoring rule — improving it strictly rewards better-calibrated probabilities.
- **Negative log-likelihood.** The loss you already tracked in training. Improving calibration usually improves NLL on the held-out set.

Plot the reliability diagram before you touch anything. Two failure modes look almost identical in aggregate ECE but wildly different in the plot: *systematically overconfident* (S-shape below the diagonal) vs. *bimodal* (spikes near 0 and 1 with nothing in the middle). Different fixes apply.

## Temperature scaling: the single-parameter fix

Temperature scaling divides the logits by a single learned scalar `T > 0` before the softmax:

```
p_calibrated(k | x) = softmax(z(x) / T)_k
```

- If the model was overconfident, `T > 1` softens the distribution.
- If underconfident, `T < 1` sharpens it.
- **Rank order does not change.** Every threshold-independent metric (accuracy, AUROC, AUPRC) is preserved. That's the killer feature: you cannot make the model worse on any of those.

You fit `T` on a held-out calibration set by minimising NLL:

```python
import torch

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter=100):
    """logits: (N, K) tensor of validation-set logits; labels: (N,) class ids."""
    T = torch.nn.Parameter(torch.ones(1))
    optim = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def closure():
        optim.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optim.step(closure)
    return float(T.detach())
```

Apply at inference: emit `softmax(logits / T)` instead of `softmax(logits)`. That is the whole method. Guo et al. (2017) show it eliminates most calibration error on modern deep networks with one parameter.

Two caveats:

- Fit `T` on the *same* distribution you plan to serve, not on the training set (the model will look perfectly calibrated on its own training data even if it isn't).
- Refit `T` after every retrain. It is cheap enough to be part of the deployment pipeline.

## Isotonic regression: the nonparametric fix

Temperature scaling assumes the miscalibration is a single global rescaling. When the model is bimodal or has different failure modes at different confidence bands, you need something more flexible. Isotonic regression fits a monotonically non-decreasing function from predicted probability to empirical probability:

```python
from sklearn.isotonic import IsotonicRegression

# For binary classification, on a held-out set:
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(pred_probs_valid, y_valid)   # both shape (N,)
calibrated = iso.transform(pred_probs_test)
```

Notes:

- Nonparametric — no assumption about the shape of the miscalibration curve.
- Requires a decently sized calibration set (~1000+ examples per class) or it will overfit the piecewise-constant fit.
- Preserves rank order (isotonic function is monotonic), so AUROC and AUPRC are unchanged.
- For multi-class, fit one-vs-rest isotonic regressions and renormalise — but temperature scaling usually beats this on multi-class calibration when the calibration set is small.

`sklearn.calibration.CalibratedClassifierCV(method="isotonic")` wraps this end-to-end with cross-validated fitting; use it for linear baselines and shallow models.

## Platt scaling: the logistic-regression alternative

Platt scaling — Platt, ["Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods"](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf), *Advances in Large Margin Classifiers 1999* — fits a two-parameter sigmoid on top of scores:

```
p(y = 1 | s) = 1 / (1 + exp(A · s + B))
```

- Two parameters instead of one — slightly more flexible than temperature scaling on binary problems.
- The historical answer for SVM outputs (which are not probabilities to begin with).
- `sklearn.calibration.CalibratedClassifierCV(method="sigmoid")` implements it.
- For fine-tuned encoders, temperature scaling is usually the safer first choice. Platt scaling suits shallow models with well-behaved score distributions.

## What to calibrate to, and on what data

Two rules that keep this from going wrong:

1. **Calibrate on a held-out set that the training loop did not see.** A validation set the trainer used for early-stopping is fine only if it was not itself used for hyperparameter selection more than a couple of times. When in doubt, hold out a dedicated calibration split.
2. **Match the deployment base rate.** If your training set is oversampled (chapter 06) and deployment traffic has a very different class prior, calibration on the training-distribution validation set does not help. You either sample the calibration set from the deployment distribution or apply a base-rate correction — see Elkan, ["The Foundations of Cost-Sensitive Learning"](https://cseweb.ucsd.edu/~elkan/rescale.pdf), *IJCAI 2001*.

## Threshold tuning: the decision layer

Once probabilities are calibrated, you still need to convert them into decisions. The threshold is a hyperparameter you tune to a real metric on a real cost function.

### Cost-sensitive threshold selection

Give each decision cell in the confusion matrix a cost:

| | Predicted positive | Predicted negative |
| --- | --- | --- |
| Actual positive | `-b` (benefit of catching a positive) | `c_FN` (missing a positive) |
| Actual negative | `c_FP` (false alarm) | `0` (correctly ignored negative) |

The Bayes-optimal threshold for calibrated `p(y = 1 | x)` is:

```
τ* = c_FP / (c_FP + c_FN + b)
```

Elkan (2001) is the standard derivation. In practice you often don't have numeric costs; instead you have a *precision-at-recall constraint* ("must catch 95 % of fraud") or a *precision floor* ("false-positive rate < 0.1 %"). Both reduce to picking `τ` on a validation PR curve.

### F1-optimal threshold

For binary tasks where you care about F1:

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

p, r, thr = precision_recall_curve(y_valid, prob_valid)
f1 = 2 * p * r / (p + r + 1e-9)
best_thr = thr[np.argmax(f1[:-1])]   # thr has length len(p) - 1
```

This is a per-class computation; for multi-class or multi-label problems, do it per class and choose per-class thresholds.

### Multi-label threshold selection

Each label gets its own threshold. Never share a single `τ` across labels — their base rates and cost profiles differ. A minimal loop:

```python
best_thresholds = []
for k in range(K):
    p, r, thr = precision_recall_curve(y_valid[:, k], prob_valid[:, k])
    f1 = 2 * p * r / (p + r + 1e-9)
    best_thresholds.append(float(thr[np.argmax(f1[:-1])]))
```

For a downstream cost function (e.g., precision-at-recall), replace `f1` with the constraint check.

### Precision-recall constraints

Constraints of the form "recall ≥ 0.9, then maximise precision" are what most product asks look like. Compute:

```python
mask = r >= 0.9                # recall values where constraint holds
best_thr = thr[mask][np.argmax(p[mask][:-1])] if mask.any() else 0.5
```

Report both precision and recall at the chosen threshold — not just the F1 achievable.

### Cost-weighted metrics

If you have numeric costs and benefits, you can compute expected utility at each threshold and pick the argmax directly:

```python
def expected_utility(y_true, prob, threshold, c_fp=1.0, c_fn=10.0, b_tp=1.0):
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return b_tp * tp - c_fp * fp - c_fn * fn
```

Sweep `threshold ∈ np.linspace(0, 1, 1001)` and pick the argmax. Report the utility gap between `τ = 0.5` and the optimal `τ` — this is often the most compelling number in the room when convincing product to accept a threshold change.

## Calibration then threshold: the correct order

The order matters:

1. **Fine-tune the model** (chapter 04).
2. **Calibrate the probabilities** on a held-out set (temperature scaling / isotonic).
3. **Tune the threshold(s)** on a held-out set using the calibrated probabilities.
4. **Evaluate the final decisions** on a completely separate test set.

If you tune the threshold on uncalibrated probabilities and then calibrate, the threshold is now wrong (calibration transformed the probability scale). If you calibrate on the same split you'll pick the threshold on, both are overfit to that split. Use at minimum three splits: train, calibrate/threshold, test.

## Deploying and maintaining thresholds

- **Threshold is a config, not a constant.** Store it in a JSON alongside the model artefact, load at inference, and version it.
- **Monitor calibration in production.** Log emitted probabilities and observed outcomes; compute a rolling ECE and reliability diagram monthly. Retrain calibration when it drifts.
- **Watch base-rate drift.** If the fraction of positives moves — spam campaigns come and go, product taxonomies shift — the Bayes-optimal threshold moves with it. This is not a bug; it is why threshold is stored as a config.
- **Separate the calibrator from the model.** Ship a `.pt`/`.bin` and a small `calibration.json` (temperature or isotonic breakpoints). This lets you recalibrate without retraining.

## When not to calibrate

- **You only care about ranking, not probabilities.** Retrieval, top-`k` predictions, learning-to-rank — a well-calibrated model and a poorly calibrated one with the same rank order behave identically. Skip calibration.
- **Your downstream consumer is another model.** Sometimes a downstream classifier prefers raw logits; calibrated probabilities can even hurt it. Test end-to-end.
- **You applied a monotonic loss shaping (like focal loss) with no downstream probabilistic use.** Focal loss trades calibration for tail F1; if the deployment reads probabilities, calibrate afterwards or don't use focal loss.

## Chapter summary

- Modern neural classifiers, especially fine-tuned transformers, are miscalibrated (Guo et al. 2017). Measure ECE and reliability diagrams before and after.
- Temperature scaling (single learned scalar dividing logits) fixes most of the miscalibration for essentially zero cost and does not change ranking metrics.
- Isotonic regression and Platt scaling are the alternatives when calibration is nonlinear or the model is binary and has enough calibration data.
- Threshold tuning is a decision-layer job, driven by cost matrices or precision/recall constraints. Never inherit `p > 0.5` unless you can justify it.
- Do calibration *before* threshold tuning, on different held-out splits, and evaluate final decisions on a fresh test set.
- Ship calibration parameters and thresholds as config alongside the model; monitor and refit them on real production data as base rates drift.
