# exercise-03: Release Gates and Rollback Design

**Estimated effort:** 3 hours

## Objective

Design and **partially implement** a release-gate and rollback plan for a specific NLP release. Deliver: (1) a gate specification document naming every blocking and warning gate; (2) a working `release_harness/` that runs the offline and slice gates against a candidate artifact and emits a machine-readable release report; (3) a rollout plan (canary %, soak duration, live-gate thresholds); (4) a rehearsed rollback plan with a tier-based SLA and a working rehearsal script; and (5) a filled-in release checklist for a mock promotion. The exercise operationalises chapter 3 on a candidate you build (or reuse from mod-112 exercise-01).

## Prerequisites

- Chapter [03 — Release gates, candidate selection, and rollback for NLP](../03-release-gates-and-rollback.md); complementary chapters [mod-111 ch. 7 (paired significance)](../../mod-111-nlp-evaluation/07-statistical-significance-for-nlp.md) and [mod-112 ch. 5 (packaging)](../../mod-112-production-nlp-pipelines/05-packaging-and-reproducible-shipping.md).
- Python 3.10+; `numpy`, `scipy`, `scikit-learn`, `pandas`, `datasets`, `seqeval` (if using an NER task); `mlflow` for registry demonstration (optional but recommended). A minimal serving stack (FastAPI + uvicorn) is enough for the live-gate simulation.
- A candidate artifact and a "previous production" artifact. Two easy sources:
    - **NER track.** `dslim/bert-base-NER` as "previous production"; a slightly-fine-tuned variant on a small CoNLL-2003 sample as "candidate."
    - **Classification track.** `distilbert-base-uncased-finetuned-sst-2-english` as previous; a distilroberta variant as candidate.
    - **Reuse.** If you completed [mod-112 exercise-01](../../mod-112-production-nlp-pipelines/exercises/exercise-01-spacy-projects-or-hf-pipeline-composition.md), reuse the pipeline as the candidate and its N-1 as the incumbent.
- A held-out eval set with declared slices — the CoNLL-2003 test split with per-entity-type slices works; SST-2 dev with per-length-bucket slices works.

## Problem statement

### Part A — Pick the release, name the gates

Write `gate_spec.md`:

- One paragraph naming the system, the candidate, and the incumbent.
- **Blocking gates.** Enumerate each. For each: what it measures, the failure threshold, the data / harness that measures it, the reason a failure blocks. At minimum:
    - Aggregate metric non-regression with paired significance (bootstrap or paired permutation — mod-111 ch. 7).
    - Slice-wise non-regression across at least three declared slices.
    - Latency parity (p50, p95, p99) within a stated tolerance.
    - Structured-output validity (if the task emits structured output).
    - Contamination sanity-check (mod-111 ch. 8) — if the eval was hand-curated after the pretraining cutoff, note that.
- **Warning gates.** Enumerate at least three that emit alerts but do not block — calibration drift, cost-per-request drift, top-1-confidence drift, etc.
- **Diagnostics.** The rest that inform the model card (per-class F1, confusion matrix, per-example error dump) but do not gate.
- **Explicit non-gates.** Two or three checks you *considered* and rejected, with the reason — e.g. "we do not gate on a proprietary benchmark because the licence forbids CI use."

Every gate has a **crisp fail condition** — a boolean computable from the harness output — not a subjective judgement.

### Part B — Build the release harness

Produce `release_harness/`:

```
release_harness/
    __init__.py
    gates/
        offline.py       # aggregate metric + slice matrix + paired significance
        operational.py   # latency parity, memory / GPU-fit sanity
        parity.py        # numeric parity for the exported artifact vs. reference
        contamination.py # simple leak check between eval and training-like corpora
    report.py            # emits release_report.json
    thresholds.yaml      # every threshold from gate_spec.md in one place
    run.py               # CLI: `python -m release_harness --candidate ... --incumbent ... --eval ...`
```

Requirements:

- **`offline.py`** loads the eval set, runs both models, computes the primary metric (F1 / accuracy / EM / etc.) plus per-slice metrics. Uses paired bootstrap (10 000 resamples) or paired permutation (10 000 permutations) for the aggregate delta CI. Rejects if the lower bound of the CI is below zero (i.e. the improvement is not statistically distinct from noise) on the aggregate; rejects if any declared slice regresses by more than the stated tolerance.
- **`operational.py`** benchmarks latency on a fixed 500-input canonical set; enforces the parity tolerance from `thresholds.yaml`.
- **`parity.py`** produces byte / numerical parity between an exported artifact (ONNX / TorchScript, borrow from mod-112 ch. 4) and the PyTorch reference. If you skip export, at minimum diff prediction distributions between two loads of the artifact to prove determinism.
- **`contamination.py`** is a lightweight leak check — n-gram overlap between eval texts and a training-like corpus. Not a formal contamination decontamination (mod-111 ch. 8 is the deep version); a quick sanity check that flags obvious overlap.
- **`report.py`** emits `release_report.json` — a structured, machine-readable record of every gate result and every measured value. The JSON is what the model-card generator (exercise from chapter 5) reads.

`run.py` executes end-to-end and exits non-zero on any blocking gate failure. That is the CI hook.

### Part C — The rollout plan

Write `rollout.md`:

- **Progressive-delivery shape.** Shadow → canary → 100 %. Explain the transitions and the criteria at each step.
- **Canary percentage schedule.** e.g. 1 % → 10 % → 50 % → 100 %, with time-in-state at each step.
- **Live-gate thresholds.** Which metrics are watched during canary, at what thresholds, on which panel of the dashboard. Draw the panel in `dashboard.md` (or a small Grafana JSON if you want to be thorough).
- **User-facing metric.** If applicable, name it and its expected error bar over the canary window.
- **Escalation.** Who decides to advance, who decides to abort, at what confidence level.

The rollout plan is a paper artifact — you are not deploying to a real cluster in this exercise — but it must be operationally realistic. A reviewer should be able to run it if you handed them the harness.

### Part D — The rollback plan and rehearsal

Write `rollback.md`:

- **Rollback shape.** Alias flip / traffic shift / feature-flag off. Pick one for this system with rationale.
- **Rollback SLA tier.** Tier-0 / -1 / -2 with the numeric threshold.
- **N-1 retention.** How is the previous artifact retained; where does the harness store it; how is retention monitored.
- **Non-rollbackable changes.** Explicit list. For this system, are there any? Schema changes? Data mutations? User-visible outputs already shipped?

And produce `rehearsal/`:

```
rehearsal/
    setup.sh              # brings up a mock "production" (two model versions loaded)
    rehearse.py           # simulates: candidate is serving; oncall detects issue; rollback executes
    teardown.sh
    RESULTS.md            # timing of the rehearsal, what went right, what would have gone wrong under load
```

`rehearse.py` starts a mock serving process with the candidate active; on a trigger signal, executes the rollback (alias-flip or a fake-registry version swap or a feature-flag toggle — pick the shape that matches your rollback plan); measures wall-clock time from trigger to serving-N-1; writes the timing to `RESULTS.md`. This is a rehearsal, not a load test — the point is to prove the mechanism works, not to benchmark it.

### Part E — Simulate a bad release

Produce `bad_release/` — a candidate with a **deliberate regression** that at least one blocking gate must catch. Suggestions:

- Corrupt the model's classification head to bias predictions toward one class → aggregate metric regresses.
- Fine-tune on data that under-represents one entity type → slice regresses even though aggregate improves.
- Export the model with a broken tokeniser mapping (mimic the tokenizer mismatch failure from mod-112 ch. 4) → runtime / parity gate catches.
- Introduce a latency regression (add `time.sleep(0.02)` to the model's forward or wrap in a slow tokeniser) → operational gate catches.

Run `release_harness` on the bad candidate. `bad_release/RESULTS.md` records: which gate fired, at what value, what the report JSON looked like, and what an oncall responding to that report would do next.

The point: **prove the harness catches what it should catch.** A harness that never fails is not a gate.

### Part F — The release checklist

Fill in `RELEASE_CHECKLIST.md` for a **mock promotion** of your (good) candidate. Use the template from chapter 3's summary. Every field populated, every sign-off named (even if the sign-off is you playing multiple roles). Attach `release_report.json` from Part B and the rehearsal results from Part D.

The completed checklist is the audit artifact. Chapter 5's responsible-release review reads it directly.

## Starter guidance

- **`thresholds.yaml` is the load-bearing file.** Every number the harness compares against lives there — the aggregate-regression tolerance, the slice-regression tolerance, the latency-parity tolerance, the paired-significance α. When a reviewer wants to challenge a threshold, they change the yaml, rerun, and see whether the release still passes.
- **Paired significance, not two-sample.** The candidate and incumbent are scored on the *same* examples; use paired resampling that respects the pairing (mod-111 ch. 7). A two-sample test on the same data is wrong and typically over-confident.
- **Slice tolerances are usually tighter than the aggregate.** A 0.5 F1 aggregate regression may be acceptable; a 3 F1 regression on the Spanish slice usually is not. Encode the asymmetry in `thresholds.yaml`.
- **Latency parity is *not* a benchmark competition.** The candidate does not need to be faster; it needs to be within tolerance. Set the tolerance based on the SLA headroom, not the model's raw speed.
- **`release_report.json` is the ML platform's contract.** Downstream tools (the model card generator, the release review, the drift monitor's baseline flip) read this file. Give it a schema (a JSON Schema or a Pydantic model) and pin `schema_version`.
- **The rehearsal must be reproducible.** `setup.sh` and `teardown.sh` should let anyone rerun the rehearsal end-to-end. An untested rollback is a paper procedure (chapter 3).
- **Do not overspecify the dashboard.** A drawn panel with axis labels is enough — you are not building a real Grafana instance in this exercise unless you want to as a stretch.

## Acceptance criteria

- [ ] `gate_spec.md` enumerates blocking gates with crisp fail conditions, warning gates, diagnostics, and explicit non-gates.
- [ ] `release_harness/` runs end-to-end: `python -m release_harness.run --candidate ... --incumbent ... --eval ...` exits non-zero on gate failure and emits `release_report.json`.
- [ ] Offline gate uses paired significance (bootstrap ≥ 5 000 resamples or paired permutation) on the aggregate delta and enforces slice-wise non-regression.
- [ ] Operational gate benchmarks latency on a fixed 500-input canonical set and enforces parity thresholds.
- [ ] `thresholds.yaml` centralises every threshold; changing a threshold changes the gate behaviour without code edits.
- [ ] `rollout.md` names shape, canary schedule, live-gate thresholds, escalation.
- [ ] `rollback.md` names shape, SLA tier, N-1 retention, non-rollbackable changes.
- [ ] `rehearsal/rehearse.py` runs the rollback end-to-end and `RESULTS.md` records the timing.
- [ ] `bad_release/` demonstrates that the harness catches at least one deliberate regression; `bad_release/RESULTS.md` records which gate fired and the report JSON.
- [ ] `RELEASE_CHECKLIST.md` is filled in end-to-end for the mock promotion and links to the report + rehearsal results.

## Stretch goals

- **Full contamination check.** Replace the n-gram overlap sanity in `contamination.py` with a proper decontamination (mod-111 ch. 8), including MinHash / LSH lookup against a representative pretraining slice. Publish a per-example contamination score in the report.
- **CI integration.** Wire the harness to GitHub Actions / GitLab CI so every PR touching the model artifact triggers a full harness run and posts the report JSON as a PR comment. A failing gate becomes a failing check on the PR.
- **Shadow-traffic simulation.** Extend the rehearsal to include a shadow-traffic phase: replay a saved traffic log through both artifacts and diff predictions. Include the prediction-parity number in the live gate.
- **Actual registry integration.** Register both artifacts in a real (local) MLflow model registry; execute the alias-flip rollback against the registry; verify the serving process picks up the change. This closes the loop between chapter-3 rollback shapes and mod-112 ch. 5 registry discipline.
- **Slice discovery.** Add a small `discover_slices.py` that groups eval examples by input length, language, and any other categorical you have; suggests slices to declare based on where the incumbent's variance is highest. Slice discovery is often more useful than the slices the team a priori guessed.
- **Two candidates, one selection.** Author a second candidate; run both through the harness; write a `SELECTION.md` that picks between them per chapter 3's candidate-selection rules (quality with significance × slice × cost × robustness). The selection rationale is a one-paragraph decision doc.

## Deliverables

```
gate_spec.md
release_harness/
    __init__.py
    gates/                offline.py  operational.py  parity.py  contamination.py
    report.py
    thresholds.yaml
    run.py
rollout.md
rollback.md
rehearsal/                setup.sh  rehearse.py  teardown.sh  RESULTS.md
bad_release/              (artifact + RESULTS.md + release_report.json for the bad run)
RELEASE_CHECKLIST.md      # populated for the mock promotion of the good candidate
release_report.json       # from the good candidate's run
dashboard.md              # panel sketch (optional if you build real Grafana)
README.md                 # how to run everything, choices made
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
