# exercise-03: Calibration and Threshold Tuning

**Estimated effort:** 3 hours

## Objective

Take a fine-tuned classifier from an earlier exercise, measure how miscalibrated it is, apply temperature scaling and isotonic regression, tune decision thresholds against a real cost function, and quantify the shipping impact of each step. By the end you should be able to defend both a calibration method and a threshold choice with concrete numbers and reliability plots.

## Prerequisites

- Chapters [06](../06-class-imbalance-weighting-sampling-and-focal-loss.md) and [07](../07-calibration-and-threshold-tuning.md).
- Exercise-01 (a fine-tuned encoder) or exercise-02's multi-label encoder — either works.
- Python 3.10+; `transformers`, `torch`, `scikit-learn`, `matplotlib` or `plotly`.

## Dataset

Ideally reuse an exercise-01 or exercise-02 dataset. If your dataset is well-balanced, deliberately pick something imbalanced for this exercise — the calibration and threshold-tuning story is more instructive under imbalance:

- **Jigsaw toxic comment** — highly imbalanced (~10 % toxic).
- **`hate_speech_offensive`** (Davidson et al., ["Automated Hate Speech Detection and the Problem of Offensive Language"](https://arxiv.org/abs/1703.04009), *ICWSM 2017*) — three-class, skewed distribution.
- **IMDb reviews** — balanced binary; still useful for the temperature-scaling story but you'll want to synthesise a cost matrix for the threshold-tuning part.
- **Financial PhraseBank** (Malo et al., ["Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts"](https://arxiv.org/abs/1307.5336), *JASIST 2014*) — three-class sentiment on financial news, moderately imbalanced.

## Problem statement

### Part A — Measure calibration

1. Fine-tune (or reuse) a binary or multi-class classifier. Reserve **three** splits: `train`, `calibration`, `test`. `calibration` should have at least 1000 examples per class where possible.
2. Compute on the raw (uncalibrated) test-set outputs:
   - **Expected Calibration Error (ECE)** with 15 bins.
   - **Brier score** (multi-class variant if applicable).
   - **Negative log-likelihood.**
   - **Reliability diagram** — 15-bin plot with the diagonal overlaid. Save the figure to disk.
3. Describe qualitatively what you see. Is the model overconfident (below diagonal)? Underconfident (above)? Bimodal? Class-dependent?

### Part B — Temperature scaling

1. Implement temperature scaling using the LBFGS snippet from chapter 07 (or Guo et al. 2017's reference implementation). Fit `T` on the calibration split's logits and labels — **not** on the test set.
2. Apply the fitted `T` at inference on the test set.
3. Recompute ECE, Brier, NLL, and the reliability diagram. Save the "before vs. after" plot side by side.
4. Confirm that ranking metrics (accuracy, AUROC, AUPRC) are unchanged — that is the temperature-scaling guarantee.
5. Report the fitted value of `T`. Interpret: `T > 1` = the model was overconfident by that factor; `T < 1` = underconfident.

### Part C — Isotonic regression

1. Fit `sklearn.isotonic.IsotonicRegression` on the calibration split. For multi-class, either fit per-class one-vs-rest and renormalise, or use `sklearn.calibration.CalibratedClassifierCV(method="isotonic")` on a wrapper around the encoder's outputs.
2. Recompute ECE, Brier, NLL, reliability diagram.
3. Compare to temperature scaling. Which method wins on ECE? On Brier? Which has more "wobble" in the reliability plot (sign of overfitting the calibration set)?
4. If your calibration set has fewer than ~500 examples per class, expect isotonic regression to overfit. Document this.

### Part D — Cost-sensitive threshold tuning

Invent (or research) a plausible cost matrix for your dataset. For a binary case:

```
c_FP = ...   # cost of a false positive
c_FN = ...   # cost of a false negative
b_TP = ...   # benefit of a true positive (optional)
```

Motivate the numbers with one paragraph explaining a realistic deployment (e.g., "for toxicity moderation, `c_FN = 10` because letting toxic content through has a higher reputational cost than a false-positive removal at `c_FP = 1`").

Then:

1. Sweep the decision threshold `τ ∈ [0, 1]` at 1001 points. Compute expected utility at each threshold on the calibration set.
2. Pick the argmax; report the chosen threshold, and the utility gap vs. `τ = 0.5`.
3. Report the precision, recall, and F1 achieved at the chosen threshold on the test set.
4. For the multi-class or multi-label case: pick per-class thresholds. Report each and the aggregate metrics.

### Part E — A precision-at-recall constraint

Now change the deployment constraint. Instead of a cost matrix, require:

> "The classifier must achieve **≥ 90 % recall** on the positive class; among thresholds that satisfy this, pick the one that maximises precision."

1. Pick the threshold satisfying this constraint using the validation PR curve.
2. Report the precision and recall at that threshold on the test set.
3. Compare to the F1-optimal threshold. How much precision did you give up for the recall guarantee?

### Part F — Combined story

Wrap everything into a single write-up (400–700 words) covering:

- The reliability-diagram evidence for miscalibration in your model.
- Which calibrator you would ship, and why (temperature vs. isotonic).
- Which threshold you would ship for the invented cost function, and which for the recall-constrained deployment.
- What monitoring you would add in production to catch calibration or threshold drift over time.
- One class or region of input space where the model is still miscalibrated after your best calibration — where, in production, you would still want a human in the loop.

## Starter guidance

- Save the **raw logits** on each split to disk once, then run calibration experiments on top of the saved logits. That way you are not repeatedly re-running the encoder.
- The `netcal` library (<https://github.com/EFS-OpenSource/calibration-framework>) implements ECE, reliability diagrams, temperature scaling, isotonic regression, and more, with sensible defaults. You can also implement each yourself in ~20 lines.
- Never fit calibration parameters on the training set — the model will look perfectly calibrated on data it memorised.
- Never tune the threshold on the test set — you will overfit the threshold and your reported numbers will be optimistic.
- For multi-class temperature scaling, use a single scalar `T`, not per-class temperatures. Per-class temperatures are called "vector scaling" (Guo et al. 2017) and usually help less than the added parameters cost.

## Acceptance criteria

- [ ] Three data splits (train / calibration / test), documented clearly.
- [ ] Reliability diagrams: uncalibrated, temperature-scaled, isotonic-calibrated. All three saved as PNGs.
- [ ] Metrics table with ECE, Brier score, NLL for all three variants.
- [ ] Fitted temperature `T` reported and interpreted.
- [ ] Invented cost matrix with a one-paragraph motivation.
- [ ] Threshold-sweep plot (utility vs. `τ`).
- [ ] Chosen threshold under (a) cost-sensitive optimisation, (b) precision-at-recall constraint. Both reported with precision, recall, F1 on the test set.
- [ ] 400–700 word write-up covering the "what would you ship" questions in Part F.

## Stretch goals

- **Per-class temperature vs. per-class isotonic.** For a multi-class problem where class base rates differ substantially, do per-class calibrators beat a single global calibrator? Report on your dataset.
- **Reliability under distribution shift.** If your dataset has a natural distribution shift (e.g., IMDb vs. Amazon reviews sentiment), calibrate on domain A and evaluate ECE on domain B. Does temperature scaling generalise? Does isotonic?
- **Focal loss + calibration.** Retrain the classifier with focal loss (chapter 06). Report the calibration story: focal loss usually makes calibration worse, and temperature scaling has to work harder. Quantify.
- **Beta calibration.** Kull, Filho & Flach, ["Beta Calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers"](https://proceedings.mlr.press/v54/kull17a.html), *AISTATS 2017*. Implement and compare to Platt scaling on your binary task.
- **Threshold as a function of a segment.** If your production traffic has clear segments (language, product line, time of day), fit per-segment thresholds and report aggregate utility. Where do segment-specific thresholds help vs. a single global threshold?
- **Long-horizon drift simulation.** Simulate base-rate drift by resampling the test set to have a different positive fraction than the calibration set. How quickly does your ECE and utility degrade? At what point would you want to refit calibration in production?
