# Release Gates, Candidate Selection, and Rollback for NLP

## Motivation

A trained model is not a release. A release is the artifact plus the gates it passed, the candidate-selection decision that picked *this* artifact over its siblings, and the rollback procedure ready to run when the new version misbehaves. Skip any of the three and the team is one bad deploy away from an incident it cannot cleanly reverse.

This chapter is the release side of the NLP-specific system-design work. It sits on top of the evaluation vocabulary from mod-111 (metrics, benchmarks, significance) and the operational vocabulary from mod-112 (packaging, drift monitoring, postmortems). The contribution here is the *decision framework* — what set of gates does a candidate have to pass, how do you pick between candidates that pass, and how do you get out when the winner degrades in production.

## The release-candidate lifecycle

Every NLP release passes through five states. Naming them explicitly matters because the gate you apply depends on the state.

1. **Trained.** The training run finished; you have a checkpoint on disk and offline metrics on the training / held-out set. Not yet a release candidate.
2. **Candidate.** The checkpoint has cleared the offline gate (below) and been packaged as an artifact with a manifest (mod-112 chapter 5). Multiple candidates can be in this state simultaneously — hyperparameter variants, data-mix variants, base-model variants.
3. **Selected.** One candidate has been picked from the pool for staged rollout. The candidate-selection decision (below) is a documented judgement, not just an argmax.
4. **Staged.** The selected candidate is receiving a small fraction of production traffic (shadow, canary) with the previous version serving the rest.
5. **Production.** The candidate is serving 100 % of traffic. The previous version is retained per the rollback SLA below.

The lifecycle is one-directional per state except for one arrow: from any state, a candidate can be **demoted** back to Trained (needs re-baking) or **rejected** entirely. Rollback (below) is the specific case of demoting a Production candidate.

## The gate hierarchy

Not every check is a "gate" — only the ones that block promotion. The rest are diagnostic. A useful three-tier hierarchy:

- **Blocking gates.** Failure prevents promotion. Small, high-signal set. Every failure is investigated.
- **Warning gates.** Failure emits a Slack alert or writes to a review ticket, but does not block. Larger set; captures softer signals.
- **Diagnostics.** Populate the model card and inform future gate design. Not tied to promotion.

Everything below is about the blocking gates.

### Offline gate: candidate → released-candidate

The offline gate answers "did the training run produce something worth shipping candidates from?"

Composition:

- **Held-out task metric above baseline.** The primary NLP metric for the task (mod-111 chapter 1) on the held-out set, compared against the previous production model. **Reject if worse** on the aggregate; **investigate** if worse on a declared slice (per-language, per-entity-type, per-user-segment) even when aggregate is up.
- **Statistical significance of the delta.** Paired bootstrap or paired permutation test (mod-111 chapter 7) on per-example scores. An unsigned delta smaller than the bootstrap CI does not count as an improvement; treat as parity.
- **Calibration.** Expected Calibration Error (mod-111 chapter 2) below the accepted threshold; a calibration collapse can look like an aggregate quality win.
- **Contamination check.** Sanity-check the eval set against the pretraining corpus for the well-known cases (mod-111 chapter 8). A model that scored well because the test set leaked into pretraining is not a candidate.
- **Structured-output validity (if applicable).** For extraction / structured generation, the fraction of outputs that parse to schema. A model that gained 3 F1 by producing invalid JSON has not gained anything.
- **Toxicity / safety floor (if applicable).** For open-ended generation, a floor on measured toxicity, harmful-response rate, or jailbreak susceptibility, using the org's chosen benchmark. Below-floor performance is a block, not a warning.

The offline gate is a **strict boolean** — one aggregate pass rate per candidate, computed by an automated harness. Every field of the harness output is stamped into the artifact manifest (chapter 5).

### Slice gate: candidate → selected

The slice gate answers "did the candidate improve overall *without* silently regressing on a declared subset that matters?"

The declared slices come from the blueprint (chapter 1) and the domain-governance step (chapter 4). Typical:

- **Per language,** if the system is multilingual. A model whose English improved 2 F1 and whose Spanish dropped 4 F1 is worse for Spanish users; that trade-off is a business decision, not a technical one.
- **Per entity type,** for NER — regressions on rare types are common and worth catching.
- **Per user segment,** if segments are declared and measurable (enterprise vs. consumer, per-tenant, per-industry).
- **Per demographic / dialect,** where fairness is a stated goal (mod-111 chapter 9).
- **Per input length bucket,** where the model behaves qualitatively differently at short vs. long inputs.

The slice gate is a **matrix**, not a scalar. The candidate must pass on every declared slice, or the promotion decision must be escalated (a documented exception, not a silent pass). Chapter 5's model card carries the slice-wise results.

### Operational gate: selected → staged

Once a candidate is selected, an operational gate answers "will this artifact actually run in production."

Composition:

- **Latency parity.** Under the same load and hardware as the previous version, the candidate hits the same p50 / p95 / p99 within tolerance. mod-112 chapter 2 covers the SLA math. A model that scored better offline and is 2× slower is not shippable to a latency-sensitive product without a separate SLA conversation.
- **Cost parity.** Cost-per-request (chapter 2) within tolerance. A candidate that is 3× the previous model's per-token cost needs the finance side to sign off.
- **Runtime parity.** The exported artifact (ONNX / TorchScript / TensorRT — mod-112 chapter 4) produces numerically-equivalent outputs to the reference on a canonical input set. Silent fallback from an optimised execution provider to a slow one is a common regression (the incident in mod-112 chapter 7).
- **Interface parity.** The artifact's input / output schema matches what the serving layer expects. Version-bump gate: if the schema changed, the schema version must have changed and consumers must have been notified.
- **Memory / GPU-fit sanity.** The artifact fits on the target serving hardware with the required batch size.

The operational gate is a **canary rig** you run before touching real traffic — a benchmark suite against a known-representative input set with the artifact wired up as it will be in production.

### Live gate: staged → production

The live gate is monitored during the shadow / canary phase.

Composition:

- **Prediction-parity ratio against the incumbent.** In shadow mode (mod-112 chapter 7), the new artifact receives the same requests as production and its predictions are diffed against production's. The expected disagreement rate is *not* zero (that would mean nothing changed) but is well-characterised — if the disagreement is 20 % on a task with a stable label distribution, something is wrong.
- **Live drift signals.** The four NLP drift signals from mod-112 chapter 6 (input distribution, vocabulary, output-label, language mix) computed on the candidate's outputs vs. the incumbent's over the canary window. A candidate whose output-label distribution is materially different from the incumbent's on the same input distribution has changed the behaviour — that may be the intent, but it must be explicit.
- **User-facing metric.** If the system has a user-facing quality proxy (thumbs-up rate on a chat response, click-through on a search result, ticket-resolution rate), the canary window must show non-regressed values within the sensitivity of the metric. The metric has a large error bar on small samples — allow enough traffic in the canary to get a real answer.
- **Error / exception rate.** Any increase in serving errors or downstream exceptions is a block.
- **Cost per request under real distribution.** The offline cost gate uses synthetic loads; the live cost gate uses real inputs. Long-tail inputs (a big paste, a giant document) can dominate cost in ways synthetic canaries miss.

Canary soak duration is a trade-off: too short and the live gate does not have signal; too long and the release cadence stalls. A working default for a high-traffic service is one business day; a low-traffic service may need a week; a system with strong offline signals and low blast-radius can go faster.

## Candidate selection: picking one when several pass

The gate hierarchy is filtering. What is left is picking one candidate from the pool of passing candidates — often two or three variants that all cleared offline and slice gates.

Anti-pattern: **argmax on the aggregate metric**. The candidate with the highest test-set score is not automatically the right pick; the run-to-run variance on many NLP metrics is larger than the delta between the top two candidates. mod-111 chapter 7 (paired significance) is the diagnostic.

A defensible candidate-selection procedure has four inputs.

- **Aggregate quality delta with significance.** Not just "up 0.4 F1"; "up 0.4 F1, 95 % bootstrap CI [0.1, 0.7]" is a signal; "up 0.4 F1, CI [-0.2, 1.0]" is parity.
- **Slice-wise deltas.** A candidate up 0.4 F1 overall but down 3 F1 on the enterprise segment is worse for enterprise users. Which trade-off is acceptable is a business decision, not the ML engineer's alone.
- **Operational cost.** Two candidates with identical quality but different latencies / costs — the cheaper one wins unless there is a specific reason (upcoming feature that needs the larger model's capability).
- **Robustness.** Behaviour on adversarial / edge cases from the internal red-team set (chapter 5). A candidate that is 0.3 F1 better on the held-out set and materially worse on the red-team set is trading measured quality for hidden risk.

The selection decision is a document — one paragraph — attached to the artifact manifest, naming the candidate picked and the trade-off accepted. That document is the input to chapter 5's release review.

## Rollback: the reverse gear

The rollback strategy is a first-class design artifact — it is not an afterthought. Without a rehearsed rollback, "mitigate before you understand" (mod-112 chapter 7) is impossible.

### The three shapes of rollback

- **Alias flip.** The registry (MLflow, Vertex Model Registry, SageMaker Model Registry) has a `production` alias pointing at version N; a rollback flips it to N-1 and the serving layer reloads. mod-112 chapter 5 covered this. Time-to-rollback: a minute or two. Requires: N-1 is still available and still runnable.
- **Traffic shift.** The serving layer routes X % to N and 100-X % to N-1; a rollback sets X back to 0. Time-to-rollback: seconds. Requires: an active traffic-splitting mechanism (Argo Rollouts, Flagger, an in-house router).
- **Feature-flag off.** The *call site* is behind a flag that, when off, routes to a fallback (previous model, rule engine, degraded UX). Time-to-rollback: seconds; the safest during an active incident. Requires: a fallback code path that has itself been tested.

Every NLP release should have a documented rollback shape *before* the release is promoted. "We'll figure it out if it breaks" is not a plan.

### The rollback SLA

The rollback SLA is a promise, expressed as **maximum time from decision-to-rollback to fully-restored user traffic**. Typical thresholds:

- Tier-0 (user-facing, high-visibility): < 5 minutes.
- Tier-1 (internal, business-critical): < 30 minutes.
- Tier-2 (best-effort batch): < 4 hours.

Meeting the SLA requires that N-1 stays *runnable* — the artifact is retained, its dependencies still resolve, the inference server can still load it. mod-112 chapter 5's manifest discipline is what makes this real; every N-1 that gets purged in a cleanup job breaks the rollback SLA silently.

### What is *not* rollbackable

Some kinds of changes cannot be undone by an alias flip:

- **Schema changes.** If the new version emits a new field that a downstream consumer has already learned to expect, rolling back removes the field and breaks the consumer. Mitigation: version the schema and evolve consumers with feature flags of their own.
- **Data mutations.** If the model has been writing outputs to a durable store (labels applied, records enriched), rolling back does not un-label the historical records. Mitigation: capture the version alongside the output so downstream consumers can filter or reprocess.
- **User-visible behaviour that has already shipped.** If a user has seen an inappropriate output, no rollback un-shows it; the incident is already real. Mitigation: shadow / canary catches these before user impact.

The rollback design document lists the *non-rollbackable* changes explicitly, so the team goes into the release knowing which levers are one-way.

### Rehearsing rollback

An untested rollback is a paper procedure. The discipline: rehearse the rollback in a non-production environment at least once per release, and once per quarter against production during a scheduled game day. The Google SRE workbook chapter on ["Incident Response"](https://sre.google/workbook/incident-response/) frames this as part of general operational readiness; NLP releases inherit the discipline unchanged.

## Release-candidate templates and the shipping checklist

The release-gate and candidate-selection story fits into a **release checklist** — a short document filled in for every promotion. A working template:

```markdown
# Release: <model name> <version>

## Candidate metadata
- Artifact URI / registry version: ...
- Selected from candidate pool of N: ...
- Selection rationale (one paragraph): ...

## Offline gate results
- Aggregate metric (with paired significance): ...
- Slice-wise deltas (per declared slice): ...
- Calibration: ECE = ...
- Contamination check: ...
- Structured-output validity: ...
- Safety floor (if applicable): ...

## Operational gate results
- Latency parity (p50 / p95 / p99): ...
- Cost parity: ...
- Runtime parity (byte / numerical): ...
- Interface parity + schema version: ...

## Rollout plan
- Shape (shadow / canary / blue-green): ...
- Canary %, soak duration: ...
- Live gate SLAs to monitor: ...

## Rollback plan
- Rollback shape (alias flip / traffic shift / feature flag): ...
- Rollback SLA (Tier): ...
- Non-rollbackable changes: ...
- N-1 artifact retention verified: ...

## Sign-offs
- Owning engineer: ...
- Reviewing engineer: ...
- Product / business: ...
- Governance (chapter 5 review, if applicable): ...
```

Every release fills in every field. Empty fields are the audit trail's job to catch — a review process (chapter 5) that rubber-stamps unpopulated fields is not a review process.

## Post-release: closing the loop

Once the release is in production, the release gates become **monitors** — the same numeric checks that gated promotion continue to run against live traffic:

- The offline gate becomes a shadow eval on a rolling held-out corpus (mod-111 chapter 9's evaluation cadence).
- The operational gate becomes a Prometheus alert on latency / cost drift.
- The live gate is the drift-monitoring surface from mod-112 chapter 6.

Any gate that would have blocked the release now becomes a signal that the release should be *rolled back*. Symmetric gates — same checks before and after — mean the team's operational muscle memory carries directly from release day to steady state.

## Anti-patterns

Three failure modes worth naming so the exercise 3 design avoids them.

- **The unblocked promotion.** No blocking gates; every check is a warning; promotions ship on the argmax. Any regression that a gate would have caught gets caught in production instead. Fix: at minimum one blocking gate — aggregate metric non-regression with paired significance.
- **The unrehearsed rollback.** The rollback procedure exists on paper; nobody has run it. First rehearsal is during an incident; something breaks. Fix: rehearse per release in staging, per quarter in production.
- **The undocumented selection.** The candidate that shipped is the one someone picked in a stand-up; no record of why. Six months later a regression is traced to the selection; the reasoning is lost. Fix: one-paragraph rationale attached to the manifest for every promotion.

## Chapter summary

- A release is the artifact plus the gates it passed plus the candidate-selection decision plus the rollback plan. The three companions ship together, or the release is not a release.
- Release-candidate lifecycle: trained → candidate → selected → staged → production. Gates block transitions; demotion and rollback are the reverse arrows.
- Gate hierarchy: (1) offline gate — aggregate metric with paired significance, calibration, contamination, structured-output validity, safety floor; (2) slice gate — per-declared-slice matrix, escalate exceptions; (3) operational gate — latency / cost / runtime / interface parity; (4) live gate — shadow + canary with prediction parity, drift signals, user-facing metric, error rate, real-distribution cost.
- Candidate selection is not argmax. Rank on quality delta *with significance*, slice-wise deltas, operational cost, and robustness on the red-team set. Attach a one-paragraph rationale to the manifest.
- Rollback shapes: alias flip (registry), traffic shift (splitter), feature-flag off (call site). Every release has a documented shape and a tier-based SLA; retention of N-1 is what makes the SLA real. List non-rollbackable changes explicitly.
- Rehearse rollback per release in staging, per quarter in production. An untested rollback is a paper procedure.
- The chapter's artifact is a **release checklist** — filled in per promotion, sign-offs recorded, empty fields caught by the audit process (chapter 5).
- Post-release the gates become monitors — symmetric checks before and after — so operational muscle carries from release day into steady state.
