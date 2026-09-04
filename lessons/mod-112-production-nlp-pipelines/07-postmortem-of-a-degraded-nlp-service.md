# Postmortem of a Degraded NLP Service: End-to-End

## Motivation

Everything in this module — composition, latency profiling, batching, runtime export, packaging, and drift monitoring — earns its keep when a production NLP service degrades and you have to diagnose it under time pressure. The postmortem is the practice that turns an incident from a scary story into an operationally recoverable event: containment first, root cause second, structural fix third, and an artifact of the whole thing that changes future behaviour.

This chapter is the end-to-end methodology. It borrows heavily from the Google SRE workbook, ["Postmortem Culture: Learning from Failure"](https://sre.google/sre-book/postmortem-culture/), which is the reference every mature ops org converges on; the NLP-specific parts are the diagnostic moves that use the drift signals from chapter 6 and the artifact provenance from chapter 5.

## The blameless postmortem: what it is, and is not

The premise is that in a well-run system, a single person did not "cause" the incident. A person made a change; the system did not have the guardrails to catch the change; the change propagated to production; the incident occurred. Blame stops the analysis at the person. Blamelessness continues the analysis to the guardrails.

Concretely, a blameless postmortem:

- Names the actions and decisions that led to the incident without attaching moral weight to the person who made them ("the deploy included a base-model swap that was not caught by the numeric-parity gate" — not "Alice pushed the wrong model").
- Focuses on *how the system allowed the failure* — missing tests, missing alerts, missing rollback procedures, unclear ownership.
- Produces action items that fix the system, not the person — a new CI gate, a new alert, a clarified runbook.
- Is published widely inside the org. Postmortems are institutional memory; they only compound in value if everyone can read them.

Blame is corrosive to the reporting culture that lets you catch the next incident earlier. Every postmortem should be a document a junior engineer feels safe writing when they are the person named in the timeline.

## The incident lifecycle

The five phases and their standard timeboxes:

1. **Detection.** Alert fires (or a user reports; that is a monitoring bug in its own right). Timebox: seconds to minutes; if it took hours, the alerting is the problem, log that separately.
2. **Response.** Oncall pages in, opens the incident channel, declares severity, assigns incident commander. Timebox: minutes.
3. **Mitigation.** Actions that stop the user-visible harm — rollback, feature flag off, traffic shift, canary drain. **Not** root cause fix. Timebox: as fast as possible; measured by MTTR (mean time to mitigate).
4. **Root cause.** Understand *why* it happened. Timebox: hours to days after mitigation.
5. **Structural fix.** Action items that make the whole class of incident unlikely. Timebox: weeks; tracked to completion.

Mitigation before root cause is the discipline that separates a working ops team from one that leaves users broken while it debugs. A five-minute rollback is almost always better than a thirty-minute understanding.

## A worked NLP-specific incident timeline

The following is a schematic incident — a composite of failure modes the module has covered, not a real event — walking the diagnostic moves in sequence. It exists to show how the tools compose.

**T+0.** PagerDuty alert: `nlp_service_p95_latency` above SLO for 5 minutes. Concurrently, `nlp_output_label_psi[15m]` is 0.34 on the label `refund_request` (Tier-1 alert threshold).

**T+2m.** Oncall in incident channel. Incident commander declared. Two hypotheses on the board: (a) upstream traffic pattern change causing input drift; (b) recent deploy of the extractor service (30 minutes prior).

**T+4m — mitigation move 1: rollback candidate.** `mlflow models describe intent-ner@production` shows version `2.4.0` deployed 30 min ago; the previous version `2.3.5` is available. Oncall proposes rollback to `2.3.5`. Incident commander approves.

**T+6m.** Rollback complete via alias flip: `mlflow.set_registered_model_alias("intent-ner", "production", "2.3.5")`, service pods rolled. Latency drops back into SLO within 2 minutes; label PSI stays elevated (interesting).

**T+15m — verified stable.** Alerts clear on latency; label PSI still 0.28 (down from 0.34, above the Tier-2 threshold). Incident mitigated on latency; still under investigation on the label distribution.

**T+30m — root-cause investigation begins.** Everyone catches breath. Investigation splits:

*Latency thread:*
- `torch.profiler` snapshot from a pod on `2.4.0` (still available in a debug replica) shows a `TensorrtExecutionProvider` fallback to `CUDAExecutionProvider` — a subgraph the TensorRT export could not handle after a change in the model's forward pass to add a new pooling op. Fallback is 5× slower per forward.
- Reproduced on a canary rig with `2.4.0`. Chapter-4 lesson violated: the numeric-parity check ran, but the *latency-parity* check did not.

*Label-PSI thread:*
- The `refund_request` label rise correlates with a traffic-source change from an upstream marketing campaign that started at T-45m. Chapter-6 input-distribution panel shows the `char_len` p50 dropped by 30 % over the same window (shorter inputs from the campaign's landing page).
- Cross-checked against the training-corpus baseline: also elevated (PSI 0.19), meaning this is not just recent-normal drift, it is off the training distribution.
- The mitigation for this is not a rollback; it is a *concept-drift check* — do we still classify these shorter inputs correctly? A quick 200-sample re-label by two annotators (chapter 09 of mod-111 applies) shows 15 % of refund classifications are wrong.

**T+3h — mitigation move 2: routing.** A feature flag routes campaign-landing-page traffic through the older `intent-ner-2.2` model (which has different tokenisation and does better on shorter inputs) while the team retrains. Not a permanent fix; buys time.

**T+2d — structural fixes queued.**
- **CI gate:** add a latency-parity check to the export pipeline — a canary benchmark on 500 fixed inputs must be within 20 % of the PyTorch reference. Would have caught `2.4.0`.
- **Alert:** add an alert on `TensorrtExecutionProvider` fallback rate; any fallback in production should surface within minutes.
- **Retraining:** add the campaign traffic sample to the training corpus for the next scheduled retrain.
- **Drift-response runbook:** codify that a label-PSI spike combined with an input-length shift requires a concept-drift check (fresh labels) before conclusion.

Every one of those action items has an owner and a deadline. Chapter 06 monitoring caught the symptoms; chapter 05 provenance told the team which artifact was live and which one to roll back to; chapters 02/04 tools attributed the root cause; the structural fix closes the specific hole the incident exposed.

## The postmortem document: minimum viable structure

Every incident produces a document with the same sections. A working template:

- **Title, date, severity, incident commander, contributing responders.**
- **User-visible impact.** One paragraph: what did users experience, over what window, to what degree. Numbers, not adjectives.
- **Timeline.** Timestamped, one line per event. Detection, response actions, mitigations, when normal resumed, when the incident was declared over. Times in UTC; timezone converted for readability only.
- **What happened (technical).** The failure chain: the change that landed, the guardrail that missed, the propagation path, the resulting symptom. Include the specific metric graphs, log excerpts, and profile traces.
- **What went well.** What worked as designed — alerts that fired on time, tools that worked, decisions that reduced impact. Underemphasised in bad postmortems; important because it identifies the parts of the system that should be preserved.
- **What went poorly.** What did not work — alerts that fired too late or not at all, unclear ownership, missing runbooks, tools that were down or slow, decisions made under uncertainty that turned out wrong.
- **Where we got lucky.** Contributing factors that did not fire this time but could have. "The canary happened to catch a related issue two days prior, so the checkpoint was already suspicious." Names the guardrails you still need to build.
- **Root cause(s).** Ideally causal chain: the specific technical cause and one layer of "why did the system permit that."
- **Action items.** Each with an owner, a deadline, a priority, and a tracking issue. Not aspirational; concrete.
- **Supporting artifacts.** Links to the graphs, the logs, the incident channel transcript, the artifact manifests, the profile traces. Every claim in the document has a source.

Postmortem quality tracks with action-item completion rate. A team whose action items rot in the backlog is doing performative postmortem culture, not blameless engineering.

## NLP-specific diagnostic moves

The generic postmortem template lands. The NLP-specific moves add to it — from chapter 06's monitoring signals and chapter 05's artifact discipline — a small set of quick diagnostics an oncall should have muscle memory for.

- **Grab the artifact manifest for the current and previous production versions.** Diff them. Any change to `training.data.snapshot_date`, `training.hyperparameters`, `runtime.precision`, `runtime.calibration_set_sha`, or `runtime.target_hardware` is a high-probability suspect. A `patch`-versioned change that also changed `runtime.calibration_set_sha` is almost always the smoking gun for a quality regression.
- **Run the drift panel on the incident window.** PSI on label distribution, subword-splits-per-word, language-ID proportion, empty-input rate — all against the point-in-time baseline (the two weeks before the deploy). Anything that moved > 2σ is a lead.
- **Replay a canary from before-the-deploy through the new artifact.** If the previous artifact is still available (chapter 5's registry discipline), a canonical 500-input canary run through both versions surfaces changes in prediction distribution deterministically. This is the definitive "did the model change" test.
- **Slice the metrics by the manifest's declared slices.** `per_slice` in the manifest is what the team accepted as the eval floor; if `us-english` still scores at the trained level but `in-english` collapsed, the artifact regressed on a slice CI failed to gate.
- **Diff the tokeniser files.** Chapter 04's warning: a tokenizer mismatch is silently wrong. `diff` the `tokenizer.json` between the two versions; unexpected differences are almost always the bug.

A useful oncall runbook page is exactly these five moves as checkboxes with links to the commands. The first oncall who has to remember them under pressure will forget one; the runbook is the shared memory.

## Progressive delivery: shipping in a way that shortens future postmortems

Postmortems are cheaper when the change is caught before it reaches everyone. Three progressive-delivery patterns worth having in the deploy pipeline:

- **Canary deployment.** New version goes to a small proportion of pods (or a small % of traffic); the drift and latency signals are compared against the stable version for a soak period (30 min to hours) before promotion. Widely available: Argo Rollouts, Flagger, cloud LB primitives.
- **Shadow traffic.** New version receives production traffic in parallel, but responses are dropped; predictions are logged and diffed against production predictions offline. Catches numeric parity regressions on real data before any user sees them. The right shape for model deploys because it exercises the model on live distribution without user-visible risk.
- **Feature flags.** Not the model — the *call to the model*. A feature flag lets you turn off the new pipeline stage for a specific customer segment in seconds, without a deploy. Standard for any new inference path.

Adopt all three progressively — canary first (cheap, orchestrator-native), shadow when you have logging infrastructure, feature flags when you have the flag service. Each one reduces the blast radius of the next mistake.

## The retro-of-retros

Every quarter, review the postmortems as a batch:

- **Are the same root causes recurring?** If a category of incident (deploy caused, drift caused, capacity caused) is the top-3 root cause in two consecutive quarters, the structural fixes are not landing or are landing in the wrong place.
- **Are action items closing?** Stale action items are a leading indicator that postmortem culture is fading.
- **Is detection time trending down?** If MTTD (mean time to detect) is not improving, the monitoring layer is not being invested in. That is a management-level signal.
- **Is severity distribution changing?** More SEV1s over time is bad; more SEV3s that were caught early is *good* — it means detection is moving earlier and impacts are being contained faster.

The retro-of-retros is where operational maturity is measured. Individual postmortems fix specific incidents; the trend of postmortems tells you whether the whole system is getting more or less resilient.

## Chapter summary

- The postmortem is the practice that turns an incident from a scary event into a recoverable operational artifact. Blameless (fix the system, not the person), thorough (impact + timeline + root cause + action items + supporting artifacts), and public (institutional memory only compounds when everyone can read it).
- Incident lifecycle: detection → response → mitigation → root cause → structural fix. Mitigation before root cause: a fast rollback beats a slow understanding while users are affected.
- The postmortem document has a fixed structure — impact, timeline, technical narrative, what went well / poorly / lucky, root cause(s), action items with owners and deadlines, supporting artifacts. Every claim links to its source.
- NLP-specific diagnostics an oncall should have as muscle memory: diff the artifact manifest (chapter 5), run the drift panel over the incident window (chapter 6), replay a fixed canary through current and previous artifacts, slice metrics by the manifest's declared slices, diff the tokeniser files.
- Progressive delivery — canary, shadow, feature flags — reduces the blast radius of the next change. Adopt them in that order.
- The retro-of-retros (quarterly) tracks trends: recurring root causes, action-item closure rate, MTTD, severity distribution. Individual postmortems fix specific incidents; the trend measures whether the system is getting more resilient.
- Reference: Google SRE workbook, ["Postmortem Culture: Learning from Failure"](https://sre.google/sre-book/postmortem-culture/), for the generic template — this chapter's contribution is the NLP-specific moves layered on top.
