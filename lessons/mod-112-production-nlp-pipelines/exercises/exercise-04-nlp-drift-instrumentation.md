# exercise-04: NLP Drift Instrumentation

**Estimated effort:** 3 hours

## Objective

Instrument the **four NLP-specific drift signal shapes** from chapter 6 — input-distribution shift, vocabulary OOV rate, output-label drift, language-mix drift — on a live inference service; build the three baseline shapes (training-corpus profile, rolling reference, point-in-time snapshot) into a `drift_baselines/` bundle shipped with the artifact; wire the alerts into a tiered scheme (page / Slack / dashboard); and prove the whole thing works with a **backtest**: synthetic-plus-real drift injections whose expected alerts you write down in advance and then verify. Deliver the instrumentation library, a running dashboard (or dashboard config), a `runbook.md` for the alerts, and a `backtest_results.md` documenting fire / no-fire against the expected alerts.

## Prerequisites

- Chapter [06](../06-nlp-drift-instrumentation.md); packaging discipline from chapter [05](../05-packaging-and-reproducible-shipping.md) for the baseline bundle.
- Python 3.10+; `transformers`, `numpy`, `pandas`, `scipy`, `matplotlib`; one of `prometheus_client` (for the metric surface) or `evidently` (for the higher-level dashboard); a language-ID library — [`fasttext`](https://fasttext.cc/docs/en/language-identification.html) with `lid.176.bin` is the most robust; `pycld3` is a lighter alternative.
- A running NLP service — reuse the one from exercise-01 or exercise-02, or scaffold a minimal one with `dslim/bert-base-NER`. Any classifier / NER / QA endpoint works as long as it emits predicted labels.

## Problem statement

### Part A — Assemble the corpora

You need three text corpora spanning distinct distributions so the drift injections are meaningful:

- **`train_reference/`**: 5-10 k documents from a "training-like" source. If your model is CoNLL-2003-trained, sample from the CoNLL training split; otherwise use `ag_news` train or the source you actually trained on.
- **`prod_normal/`**: 5-10 k documents that resemble the training source with the natural drift a real service sees (slight length differences, occasional non-English inputs, gradual vocabulary shift). Use CoNLL test + a modest slice from `wikiann/en` for realism; or synthesise by lightly perturbing `train_reference` with random truncation / capitalisation changes.
- **`prod_drifted/`**: a series of small drifted subsets you will inject during the backtest (Part F). Suggested five:
  - `drift_len_short`: 500 documents truncated to ≤ 40 chars.
  - `drift_len_long`: 500 documents padded / concatenated to > 1 500 chars.
  - `drift_language`: 500 non-English documents from `wikiann/de` or `wikiann/es`.
  - `drift_domain`: 500 documents from an out-of-domain source (`reuters21578` if you trained on Wikipedia, or the reverse) intended to move vocabulary and predicted labels.
  - `drift_output_label`: 500 documents you *filter* to force the model's predicted-label mix to shift (e.g. only prompts likely to yield `ORG` entities), so the label PSI moves without input drift being extreme.

Document each corpus's source, size, and known properties in `corpora.md`.

### Part B — Build the reference profiles (`drift_baselines/`)

Produce `build_baselines.py` that reads `train_reference/` and writes a bundle:

- `char_len_profile.json`: histogram of input character lengths (bins from chapter 6 recommendation) plus p50 / p95 / p99.
- `token_len_profile.json`: same on tokenised length.
- `subword_splits_profile.json`: mean subword-splits-per-word plus a histogram.
- `language_profile.json`: proportion per detected language.
- `char_class_profile.json`: proportions of `{ascii_letter, digit, punct, whitespace, non_ascii, emoji}` plus URL / mention / hashtag densities.
- `rare_token_ids.json`: token IDs whose corpus frequency is in the bottom decile — the "rare token" set from chapter 6.
- `label_profile.json`: proportion of each predicted label when the model runs over `train_reference/` (running the current production model against the training corpus is the canonical baseline for output-label PSI).

Include `hash.txt` — sha256s of every file. The bundle is the shipped baseline: it goes into the artifact manifest's `drift_baselines_sha` field.

Also produce, from `prod_normal/`, a **rolling-reference profile** covering the last simulated 30 days (bucket by simulated date). Store as `rolling_reference/`.

### Part C — Wire the metrics

Produce `drift/metrics.py` — the instrumentation library. It exposes:

- `record_input(text: str)`: computes per-input stats, emits Prometheus histograms / counters, updates the current rolling-window aggregation.
- `record_output(prediction: dict)`: same for predicted labels / confidences / NER-entity types.
- `snapshot() -> dict`: returns the current window's aggregate stats.
- `compute_psi(observed: dict, baseline: dict) -> float`: the PSI implementation from chapter 6.
- `compute_ks(observed: list[float], baseline: list[float]) -> tuple[float, float]`: KS statistic + p-value for continuous distributions.

Instrument the service (from exercise-01 / -02 or the minimal service): each `/extract` call runs `record_input` on the raw text and `record_output` on the response. Add an `/metrics` endpoint (`prometheus_client.generate_latest()`) so Prometheus can scrape.

### Part D — Compose the periodic drift-check job

Produce `drift/check.py` — a script (runnable as a scheduled job) that:

1. Loads the current-window aggregate (from Redis, disk, or a Prometheus query, whichever fits your stack).
2. Loads all three baselines (training, rolling-reference, point-in-time-since-last-deploy).
3. Computes each drift signal against the appropriate baseline:
   - Input length: KS against training profile, KS against rolling.
   - Language proportions: PSI per language against training, PSI against rolling.
   - Subword-splits-per-word: mean-delta and KS against training.
   - Rare-token rate: rate against training rate.
   - Label distribution: PSI per label + chi-square goodness-of-fit against training AND against point-in-time.
4. Emits a JSON summary: `{signal, baseline_shape, statistic, p_value, severity: tier}` per signal.
5. Fires an alert (via a stubbed webhook or a printed line — do not actually page anyone from a course exercise) for each Tier-1 signal, records Tier-2 to a Slack-style log file, appends Tier-3 to a weekly-review CSV.

Alert-tiering rules are the chapter-6 defaults:

- Tier 1 (page): input `empty_rate` > 5×, deploy-correlated label PSI > 0.5, latency SLO burn (out of scope for this exercise unless you wire it in).
- Tier 2 (Slack): output-label PSI in `[0.25, 0.5]`; per-language PSI > 0.25; rare-token rate up > 50 % from baseline.
- Tier 3 (weekly): PSI in `[0.1, 0.25]`; novel-token rate rising; drifts against the *training* baseline (slow shifts).

### Part E — The runbook

Produce `runbook.md` — an oncall guide for the Tier-1 and Tier-2 alerts. For each alert:

- What the metric measures (link to chapter 6).
- What likely-cause hypotheses to test (upstream traffic change, client bug, deploy regression, tokeniser mismatch).
- Which panels of the dashboard to look at.
- What mitigation moves are safe to take without waiting for a full postmortem (chapter 7 pattern).
- Escalation path — who owns which alert.

One page per alert; use the format the SRE workbook recommends (Trigger → Impact → Investigation → Mitigation → Escalation).

### Part F — The backtest

Produce `backtest.py` — the proof that the whole system works.

Set up: pre-populate the service with `prod_normal/` traffic for a simulated 30 days (a "warm" period during which no alerts should fire). Then, for each drifted subset from Part A, run a scripted injection: the service receives a batch of the drifted subset over a simulated 15-minute window; the check job runs at the end of the window; the results are recorded.

For each injection, write down in advance (in `expected_alerts.md`) which alerts you expect to fire and at what tier. Concretely:

- `drift_len_short`: expect Tier-2 KS-on-char-length + Tier-3 label-PSI drift.
- `drift_len_long`: expect Tier-2 subword-splits-per-word + Tier-2 label-PSI.
- `drift_language`: expect Tier-1 language-proportion PSI + Tier-2 rare-token rate.
- `drift_domain`: expect Tier-2 rare-token rate + Tier-2 label-PSI.
- `drift_output_label`: expect Tier-2 label-PSI without corresponding input drift — the "concept drift smell test."

Run the backtest; record fire / no-fire in `backtest_results.md`. For every unexpected fire or missed fire:

- If a false positive: tune the threshold, re-run, document the new threshold and why.
- If a false negative: check whether the signal is instrumented (a bug), the baseline is off (rebuild), or the threshold is too loose (tune).
- Do **not** tune thresholds to make the backtest pass at the cost of realistic sensitivity; document any trade-off explicitly.

### Part G — Report

Produce `report.md`:

1. Corpus summary (from `corpora.md`).
2. Baseline bundle summary — what's in `drift_baselines/`, hashes.
3. The metric surface — what is exposed on `/metrics`, per instrumented signal.
4. The check job — cadence and what it computes.
5. Alert tiering + runbook summary.
6. Backtest: expected vs. observed alerts per injection, per tier; any threshold tuning applied and the justification.
7. Retrospective: which signal is most / least useful; where you would invest instrumentation next.

## Starter guidance

- **Time can be simulated.** No need to run a real 30-day rolling window — the check job just needs the aggregation over a labelled window (`window_id`). Feed 30 windows of `prod_normal` as the "reference" and then the drift windows.
- **Storage of aggregations is up to you.** In-memory `dict` for a single-process backtest is fine; only introduce Redis / a real TSDB if you want to run the service as a separate process (a nice fidelity gain but not required).
- **Language-ID caveats.** fastText's `lid.176` works well on inputs > ~20 chars; short inputs often misidentify. Track a `low_confidence_lang` bucket rather than trusting every ID.
- **Do not use the model itself to compute drift on its own outputs.** The `record_output` path should record what the model *predicted* — separately, `check.py` compares aggregates. Do not close a feedback loop where the drift score influences the model's own predictions.
- **PSI implementation gotcha.** Add a small epsilon to zero-count bins to avoid `log(0)`; document the epsilon (`1e-6` is common). The result is sensitive to it in low-count regimes.
- **Backtest thresholds vs. real thresholds.** Backtest thresholds are calibrated against synthetic drift; real production thresholds are calibrated against historical incidents (chapter 6). Document the difference so a reader does not assume the backtest thresholds are directly deployable.

## Acceptance criteria

- [ ] `corpora.md` documents `train_reference/`, `prod_normal/`, and the five `prod_drifted/*` sets with sources, sizes, and known properties.
- [ ] `build_baselines.py` produces `drift_baselines/` with all six profile files plus `hash.txt`.
- [ ] `drift/metrics.py` instruments the four shapes (input distribution, vocabulary, output label, language mix); the service exposes `/metrics` in Prometheus format.
- [ ] `drift/check.py` computes KS + PSI + chi-square per signal per window against training and rolling baselines; emits the tiered JSON alerts.
- [ ] `runbook.md` covers every Tier-1 and Tier-2 alert with Trigger / Impact / Investigation / Mitigation / Escalation.
- [ ] `expected_alerts.md` documents predictions for each of the five injections BEFORE running the backtest.
- [ ] `backtest.py` runs the pre-populate + inject flow; `backtest_results.md` compares expected vs. observed per tier per injection; any threshold changes are justified with a re-run.
- [ ] Report captures the whole flow and the retrospective; instrumentation is reproducible from the repo alone.

## Stretch goals

- **Embedding-space drift.** Add MMD or Wasserstein distance between the pooled `[CLS]` embedding distributions of a sample from each window against the training-corpus embedding distribution. Threshold with a permutation-based p-value.
- **Multivariate drift with Alibi Detect.** Wire the [Alibi Detect](https://github.com/SeldonIO/alibi-detect) text-drift detector on the pooled embeddings; compare with your handcrafted signals on the injections.
- **Concept drift confirmation.** For the `drift_domain` injection, hand-label 200 documents' correct entities and score `seqeval` F1; show the quality drop that your Tier-2 alerts predicted.
- **Grafana dashboard.** Export a Grafana dashboard JSON that renders every instrumented signal + PSI / KS graphs per baseline shape. Include it in the deliverable.
- **CI gate.** Add a CI job that runs a mini-backtest (one injection) on every PR touching the model artifact — fails if the injection produces a different-shaped alert profile from the recorded baseline. This ties chapter 5's "CI gates on the artifact" story to chapter 6.
- **Real-service integration.** Deploy the instrumented service to a small local cluster (docker-compose with Prometheus + Grafana) and run the backtest against the real stack. Attach screenshots of dashboards during each injection.

## Deliverables

```
corpora.md
build_baselines.py
drift_baselines/          char_len_profile.json  token_len_profile.json  subword_splits_profile.json  language_profile.json  char_class_profile.json  rare_token_ids.json  label_profile.json  hash.txt
rolling_reference/        window_00.json ... window_29.json
drift/                    metrics.py  check.py  __init__.py
service.py                # instrumented endpoint (or diff against exercise-01/-02 service)
runbook.md
expected_alerts.md
backtest.py
backtest_results.md
report.md
dashboard.json            # optional: Grafana / Evidently dashboard
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
