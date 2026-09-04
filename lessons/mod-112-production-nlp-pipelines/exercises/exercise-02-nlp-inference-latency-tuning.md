# exercise-02: NLP Inference Latency Tuning

**Estimated effort:** 3 hours

## Objective

Take a **stated latency SLA** for an off-the-shelf Transformer NER service, measure whether the current implementation meets it, attribute the time to specific stages of the forward pass, and — through a combination of batching, precision, and (if needed) a light runtime change — get the service inside the SLA. Deliver a **latency report** (`report.md`) with warmed histograms, per-stage attribution, a labelled trace, the specific knob you turned, before/after numbers, and a quality re-verification that shows the tuning did not silently regress accuracy.

## Prerequisites

- Chapter [02](../02-latency-profiling-under-slos.md); the batching / precision material from chapter [03](../03-batching-and-quantisation.md).
- Python 3.10+; `transformers>=4.40`, `torch>=2.2`, `datasets`, `numpy`, `pandas`, `matplotlib`; `py-spy` installed on `PATH`; optional `torch.profiler` (in-tree). GPU strongly recommended but not required — the exercise works on CPU with a smaller model.
- A machine you can benchmark on with a consistent load profile (idle desktop, dedicated cloud instance, or reserved lab machine). Note the CPU / GPU / driver / CUDA / PyTorch versions — every report pins these (chapter 2 rule).
- Familiarity with `datasets` for loading `conll2003` or `wikiann` en for representative inputs.

## Problem statement

### Part A — Write the SLA down completely

The service under test is a **Transformer NER inference endpoint** — one HTTP request, one input text (max 2 000 characters), one JSON response containing the list of entities. Choose *one* of these SLAs and record it in `sla.md` verbatim:

- **GPU-track SLA.** `p95 < 60 ms`, `p99 < 120 ms`, `p50 < 25 ms`, measured on 5-minute rolling windows, at the service layer (post-tokenisation through post-decoding), warmed up, English inputs, one document per request.
- **CPU-track SLA.** `p95 < 200 ms`, `p99 < 400 ms`, `p50 < 90 ms`, otherwise identical.

Note the percentile / window / work-unit / scope (chapter 2) explicitly.

### Part B — Build the baseline service and load generator

Produce `service.py` — a minimal FastAPI (or Flask; whichever you prefer) service with one endpoint `POST /extract` that returns `{"entities": [...]}`. Use `dslim/bert-base-NER` (CPU-track) or `Jean-Baptiste/roberta-large-ner-english` (GPU-track) — or pick a model and justify the pick in `report.md`. Do NOT enable any tuning yet: FP32, `batch_size=1`, no `torch.compile`, no dynamic batcher.

Produce `load.py` — a load generator that draws inputs from a representative distribution:

- 800 warm-up requests to prime the service (discarded from stats).
- 2 000 timed requests drawn from a mix: 60 % short (≤ 100 char), 30 % medium (100-500 char), 10 % long (500-2 000 char), matching typical production skew. Source the inputs from `conll2003` or `wikiann` en.
- Configurable concurrency (`--concurrency N`); measure at concurrency 1, 4, 16.
- Report per-request latency to `baseline.csv`.

Requirements on the harness itself:

- **Warm-up separately from measurement.** Chapter 2's most common mistake.
- **Client-measured latency**, not server-measured — measure from just before `httpx.post` to just after the response is parsed.
- **Report by percentile per concurrency**, not just the mean.

### Part C — Baseline measurement

Produce the baseline `report.md` section:

- Hardware pin (CPU model / core count, GPU model + driver, CUDA, cuDNN, PyTorch, Transformers, OS).
- Baseline latency table: p50 / p90 / p95 / p99 / p99.9 / max / mean at each concurrency.
- Histogram plot (`baseline_hist.png`) with vertical lines at your SLA percentiles.
- Latency-by-length scatter plot (`baseline_by_length.png`) showing the length dependence.
- Explicit statement: **do we pass the SLA?**

Almost certainly you do not on the GPU track (batch-1 + no compile is slower than the SLA on realistic input length); the CPU track may or may not depending on hardware.

### Part D — Attribute the time

Add per-stage timers to the service (chapter 2 pattern) and re-record 500 requests. Break out at least:

- Framework / HTTP-parse
- Tokenisation
- Model forward (with `torch.cuda.synchronize()` if GPU)
- Postprocess / decode (BIO decode + span assembly)
- JSON serialise

Report in `attribution.md` a per-stage p50 / p95 table. Comment on which stage(s) dominate. If the model forward is the dominant stage (usually is), take a `torch.profiler` snapshot on a warmed process and export a Chrome trace; open it and identify the top-3 kernels by CUDA time (GPU) or the top-3 ATen ops by CPU time (CPU). Include annotated screenshots or a summary table.

If instead a non-model stage dominates (e.g. postprocess or serialisation), run `py-spy record -o cpu_flame.svg --pid $(pgrep -f service) --duration 30` under load and interpret the flame graph. Include the SVG.

### Part E — Pick a knob, turn it, re-measure

Based on the attribution, pick **one** tuning move from the following menu and apply it. Do not stack multiple moves in one measurement — each measurement changes one thing so the effect is attributable.

- **Batching (dynamic).** Add a dynamic batcher — either homegrown (`asyncio.Queue` with `max_batch_size` and `max_batch_delay_ms`) or via a serving framework (BentoML, TorchServe, Triton). Sweep `(max_batch_size, max_batch_delay_ms)` over a grid; produce a heatmap of end-to-end p95 vs. batch parameters at concurrency 16. Pick the point that best fits the SLA and record it.
- **Length bucketing.** Pre-sort a moving window by input length before batching. Show the throughput / p95 improvement.
- **Precision (FP16 or BF16).** Wrap the forward in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` (BF16 if hardware supports) or `torch.float16`. Re-measure; re-check numeric parity (chapter 4 pattern) against the FP32 reference on a fixed 500-input canary.
- **`torch.compile`.** Wrap the model in `torch.compile(model, mode="reduce-overhead")`. Recompile-cost note: exclude the first N warmup calls per unique shape. Measure the shape-cache thrash if you did not length-bucket.
- **Dynamic INT8 quantisation (CPU-track only).** `torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`. Re-measure; report memory footprint change (`/proc/<pid>/status:VmRSS`).

Whichever move you pick, produce a **before/after** section in `report.md`: same table, same histogram overlaid (`compare_hist.png`), same by-length scatter, per-stage attribution re-run. Explicit statement of whether the SLA now holds.

### Part F — Verify quality did not regress

The pipeline still has to produce correct entities. Re-score the tuned service on the `conll2003` validation set using `seqeval` in `mode="strict"` with `scheme=IOB2` (mod-111 chapter 2). Report:

- Baseline F1 and CI.
- Tuned F1 and CI.
- Delta with a paired-bootstrap test (from mod-111 chapter 7).
- If the delta is a statistically significant regression of > 0.5 F1 anywhere per-slice, the tuning fails Part F — either reject the move, walk the knob back (larger batch, higher precision), or document explicitly that the quality drop is acceptable to the stated SLA constraint.

### Part G — Write the report

`report.md` is the deliverable. Order:

1. SLA (copied from `sla.md`).
2. Environment / hardware pin.
3. Baseline: table + histogram + by-length scatter + pass/fail.
4. Per-stage attribution + trace analysis.
5. Tuning move: what and why.
6. Tuned: table + histogram + attribution + pass/fail.
7. Quality re-verification: baseline vs. tuned F1 with CI and paired-bootstrap.
8. Retrospective: what you would try next (chapter 3 knobs not yet used, chapter 4 export candidates).

## Starter guidance

- **Warm up in-process, not by the load generator alone.** Add a `startup` handler that runs 100 forward passes before the service accepts traffic. Otherwise the first several load-gen requests skew the histogram even after your warm-up burst.
- **`torch.cuda.synchronize()` inside GPU timers.** Without it, wall-clock underestimates the forward pass by the async CUDA launch time.
- **Do not benchmark under `--reload` or a dev server.** Uvicorn with `--workers 1 --no-reload` for the baseline; the actual concurrency you configure is what you measure.
- **Do not compare across hardware runs.** If you re-run on a different machine, restate the pin and mark the numbers separately in the report.
- **Attribute before you tune.** The GPU forward is usually the bottleneck; do not accept that assumption without evidence. On CPU pipelines the tokenizer and JSON parser are often surprisingly significant.
- **When batching, tune against your target concurrency**, not against microbenchmarks that feed exactly-full batches. Real behaviour under bursty QPS is the only measurement that matters.
- **Log every measurement to CSV.** Recomputing percentiles from the raw data is the honest way; a report that quotes only the summary loses information.

## Acceptance criteria

- [ ] `sla.md` states the SLA verbatim (percentile / window / work-unit / scope).
- [ ] `service.py` implements the baseline service; `load.py` implements the specified load profile with warm-up, concurrency, length-mix, and per-request CSV output.
- [ ] `report.md` includes the hardware pin, baseline percentile table at three concurrencies, histogram plot, by-length scatter, pass/fail.
- [ ] Per-stage attribution table populated from server-side timers; either a `torch.profiler` Chrome trace summary or a `py-spy` flame graph committed (depending on where time lives).
- [ ] Exactly one tuning move applied; before/after tables and overlay histogram show the effect; SLA pass/fail after the move stated explicitly.
- [ ] `seqeval` `strict`+`IOB2` F1 on the CoNLL validation set for baseline and tuned; paired bootstrap on the delta; per-slice breakdown (per entity type at minimum). If the tuning regressed, document the walk-back or acceptance.
- [ ] All framework / library / driver versions pinned in `report.md`.
- [ ] Raw per-request latencies committed as CSV; percentile numbers in the report reproducible from the CSV.

## Stretch goals

- **SLA-vs-cost curve.** Add cost per 1M inferences at each configuration (approximating from cloud instance pricing) and produce a cost-vs-p95 Pareto plot. Which configuration is on the frontier?
- **Two-knob interaction.** After the single move, stack a *second* move (e.g. batching + FP16) and re-measure. Report whether the effects combine additively, multiplicatively, or (rarely) counteract.
- **Length-bucketed dynamic batching.** Instead of FIFO batching, hold a small buffer and dispatch length-similar batches. Compare p95 versus the length-agnostic dynamic batcher.
- **Compare against the runtime-export baseline (chapter 4).** Export the tuned model to ONNX and re-measure. How much of the remaining gap does the compiled runtime close?
- **Bursty-load stress test.** Instead of steady QPS, drive the load generator with a Poisson-arrival pattern with mean λ and burstiness parameter. Report how the p99 responds; discuss what breaks and how you would mitigate.
- **Sub-100 ms end-to-end for a longer document.** Take a 4 000-char input, chunk with overlap, run the model on chunks, merge spans. Measure whether the SLA still holds; document the chunking-and-merging choice.

## Deliverables

```
sla.md
service.py
load.py
report.md
attribution.md
baseline.csv
baseline_hist.png
baseline_by_length.png
compare_hist.png
trace.json                # torch.profiler Chrome trace, OR
cpu_flame.svg             # py-spy flame graph
tuned.csv
quality/                  seqeval_baseline.json  seqeval_tuned.json  paired_bootstrap.json
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
