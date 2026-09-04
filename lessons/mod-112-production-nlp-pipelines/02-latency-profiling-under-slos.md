# Profiling NLP Inference Latency Under Sub-100ms SLAs

## Motivation

"NLP is slow" is a claim made by teams that never profiled. The reality is more useful: NLP pipelines have a small number of specific latency drivers — tokenisation, per-token forward passes, decoding heads, CPU/GPU round-trips, and postprocessing — and each one is measurable and independently tunable. A pipeline that missed its 100 ms p95 target on the first deploy can almost always meet it once someone attributes the time correctly.

This chapter is the profiling half of the latency story. It covers what an SLA actually asks of you (which is not "average latency"), how to instrument a pipeline so per-stage cost is attributable, and how to read the profile so the tuning moves in chapter 3 (batching, quantisation) and chapter 4 (ONNX / TorchScript / TensorRT export) are informed rather than cargo-culted.

## What a sub-100ms SLA actually specifies

An SLA is a **percentile, an interval, a work-unit, and a scope**. "p95 latency under 100 ms" without the other three is meaningless. A complete SLA looks like:

> **99% of `POST /extract` requests over any rolling 5-minute window should return in under 250 ms; 95% under 100 ms; median under 40 ms — measured at the load balancer, warmed-up service, one document per request, English input up to 2 000 characters.**

The parts that matter:

- **Percentile.** Averages hide the tail. A p50 of 30 ms with a p99 of 4 s ships an unusable service. Report and target p50, p95, p99. Consequential systems also target p999.
- **Window.** Percentiles computed over "all time" are dominated by warm-up and unrepresentative periods. Rolling windows (typically 1 min to 5 min) are what SLO tooling (Prometheus histograms, SRE workbook) tracks.
- **Work-unit.** "One request" is ambiguous when a request may contain 1 or 100 documents. State the unit. If your API is batched, state per-batch and per-item SLAs separately.
- **Scope.** Where you measure changes the number. At the model layer (forward-pass time) it is smaller. At the service layer (adds tokenisation, decoding, serialisation) it is larger. At the load balancer (adds request framing, TLS, queueing) it is largest. Pick one, document it, measure at that layer.

The **percentile-cost asymmetry** is the key mental model. Halving the p50 is cheap — the median case is usually a small model on typical-length input. Halving the p99 is expensive — the p99 case is a long input, a full-cache-miss branch, or a garbage-collection pause. Tuning is dominated by tail work, not average work.

## The latency budget

Before you profile, allocate a **budget** across the pipeline stages and check whether it is feasible. For a 100 ms p95 end-to-end target on a document-level extraction pipeline:

| Stage | Budget (ms) | Notes |
|---|---:|---|
| HTTP framing + JSON parse | 3 | Framework baseline (FastAPI/Uvicorn) |
| Language ID + segmentation | 5 | Cheap models, CPU |
| Tokenisation (Transformer) | 5 | `PreTrainedTokenizerFast` on CPU |
| Model forward (Transformer NER) | 40 | The main event |
| Postprocess: span decoding | 5 | Argmax + BIO decoding |
| Entity linker candidate gen | 15 | ANN over KB, CPU or GPU |
| Entity linker rerank forward | 15 | Small cross-encoder |
| Structured output serialisation | 5 | Pydantic → JSON |
| Slack for tail | 7 | Absorb GC pauses, warm-cache misses |
| **Total** | **100** | |

If the budget does not close on paper it will not close in production. Two moves when it does not: **collapse stages** (fold entity linking into the NER model as a joint head), or **move stages off the request path** (async post-enrichment where the client is happy with a first response). Both are architectural decisions, not tuning knobs.

## Warm-up: the number-one profiling mistake

The first 100 calls to a Transformer service are unrepresentative. On a cold service:

- **Model weights are not yet in the OS page cache.** First `torch.load` from disk pays real I/O cost.
- **CUDA kernels are compiled on first use** (JIT for many ops). First forward is 2–10× slower.
- **Python allocator warmth.** cPython's small-object allocator and PyTorch's caching allocator warm up after ~100 forward passes.
- **cuDNN benchmark cache.** With `torch.backends.cudnn.benchmark = True`, the first forward per unique input shape searches for a fast kernel; subsequent forwards with the same shape are cached.

Every latency measurement must **warm up before recording**. A standard protocol: run 100 warm-up forward passes with representative inputs; discard the timings; then record the next 1 000. Any benchmark tool that does not do this — including naïve `time.time()` loops — reports a warm-up-contaminated number.

## Instrumentation, from coarse to fine

Reach for the coarsest tool that answers your question.

### End-to-end histograms (`time.perf_counter` + `numpy.percentile`)

The right first tool. Wrap the entry point, run N representative requests, dump per-request latencies, compute percentiles:

```python
import time, numpy as np
from your_service import extract

lat = []
warmup = 200
for i, doc in enumerate(load_requests(1200)):
    t = time.perf_counter()
    extract(doc)
    dt = (time.perf_counter() - t) * 1000
    if i >= warmup:
        lat.append(dt)

arr = np.array(lat)
for p in [50, 90, 95, 99, 99.9]:
    print(f"p{p}: {np.percentile(arr, p):.1f} ms")
```

This tells you *whether* you are inside the SLA. It does not tell you *why not*.

### Per-stage timers

Instrument each pipeline stage:

```python
from contextlib import contextmanager
from collections import defaultdict

STAGE_MS = defaultdict(list)

@contextmanager
def timed(stage):
    t = time.perf_counter()
    try:
        yield
    finally:
        STAGE_MS[stage].append((time.perf_counter() - t) * 1000)

with timed("tokenize"):
    tokens = tok(text, return_tensors="pt")
with timed("forward"):
    out = model(**tokens.to(device))
```

Add `torch.cuda.synchronize()` inside `timed("forward")` on GPU — otherwise `perf_counter` returns before the kernel completes and per-stage numbers are wrong.

### `torch.profiler` for the model forward

`torch.profiler.profile` gives you per-op CUDA-kernel and CPU-op time inside a forward pass:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as p:
    for _ in range(20):
        model(**batch)
        torch.cuda.synchronize()

print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20))
p.export_chrome_trace("trace.json")
```

Chrome trace viewer (`chrome://tracing`) opens the JSON. Look for:

- **Big blue matmul bars** — normal, this is the transformer.
- **Long gaps between kernels** — CPU is bottlenecking the launch; consider `torch.compile` or a compiled runtime (chapter 4).
- **Postprocess ops on GPU followed by a `.cpu()` sync** — the postprocess should probably run on CPU if it is small, or stay on GPU if it is repeated per batch.

### `py-spy` for CPU-bound services

`py-spy` (Ben Frederickson) samples a running Python process and produces a flame graph without instrumenting the code:

```bash
py-spy record -o profile.svg --pid $(pgrep -f your_service) --duration 60
py-spy top --pid $(pgrep -f your_service)
```

Use `py-spy top` for a live `top`-style view, `py-spy record` for a flame graph. It is the right tool for "the service is slow, I do not know where" on CPU pipelines. Requires ptrace privileges on Linux (`--privileged` in Docker, or `SYS_PTRACE` capability).

### `cProfile` for one-off deep dives

Standard-library `cProfile` gives function-level counts and cumulative time. Best when you already have a suspect function:

```python
import cProfile, pstats
p = cProfile.Profile()
p.enable(); extract(doc); p.disable()
pstats.Stats(p).sort_stats("cumulative").print_stats(30)
```

`cProfile` overhead is real (~2×) — do not use it to compare *timings* against un-profiled runs, only to compare *proportions* between functions.

## Reading a latency histogram

The shape of the latency distribution tells you what to fix.

- **Long right tail, flat median.** Something in the tail requests is different — long input, cold cache, garbage collection. Slice the histogram by input length and plot separately.
- **Bimodal.** Two populations. Common causes: cache-hit vs. cache-miss, one language vs. another, one document type routed to a slower branch.
- **Sawtooth on a time-series plot.** GC or memory-pressure pattern. Check `gc.set_debug(gc.DEBUG_STATS)`, watch RSS, and consider `PYTHONMALLOC=malloc` + jemalloc for large-heap services.
- **Step function correlated with QPS.** Queueing delay; the server is CPU/GPU-saturated and requests wait. This is not a model-latency problem — it is a capacity problem. Add a replica or reduce per-request cost.

Slicing is essential. `p95 = 120 ms` is one number; `p95 by input length bucket = {<200 chars: 30, 200-1000: 80, 1000-2000: 180}` is a plan.

## Latency vs. throughput vs. cost — the three-way trade

Tuning any one usually costs you the other two. Concretely:

- **Batching increases throughput and per-item cost-efficiency, but increases per-request latency** (an item waits for the batch to fill). Chapter 3 covers dynamic batching.
- **Model quantisation reduces per-item latency and memory (fits more replicas per GPU), but risks small quality loss.** Chapter 3 covers INT8 / FP16 trade-offs.
- **Runtime export (ONNX / TorchScript / TensorRT) reduces per-item latency and enables aggressive kernel fusion, but adds a build step and a compatibility surface.** Chapter 4 covers this.
- **Distillation** (smaller model trained to imitate a larger) is the biggest single latency lever but requires retraining and lands in a different chapter (mod-108 chapter 8).
- **More replicas** trade money for tail latency: extra capacity means less queueing at the p99.

State the SLA constraint first, then decide which knobs you have room to turn. A team told to "make it faster" without a target usually optimises the wrong stage.

## Measuring on the right hardware

Latency numbers are only comparable on identical hardware, driver stack, and library versions. Pin these in every benchmark report:

- CPU model, core count, `cat /proc/cpuinfo | grep 'model name' | uniq`.
- GPU model, driver version, `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`.
- CUDA version (`nvcc --version`), cuDNN version (`python -c "import torch; print(torch.backends.cudnn.version())"`).
- PyTorch / Transformers versions (`pip freeze | grep -E "torch|transformers|onnx"`).
- Container / OS.

Benchmarks without this pin are unreproducible; benchmarks *with* it are directly comparable when someone else re-runs on the same stack.

## The profiling loop

The order of operations, made explicit:

1. Write the SLA down completely (percentile, window, unit, scope).
2. Build the latency budget by stage.
3. Warm up. Record end-to-end p50/p95/p99.
4. If out of SLA, add per-stage timers with `torch.cuda.synchronize()` where relevant.
5. The stage over budget: reach for `torch.profiler` (forward), `py-spy` (CPU service), or `cProfile` (function suspect).
6. Slice the histogram by input length, language, route.
7. Pick one lever from chapter 3 (batching, quantisation) or chapter 4 (runtime export). Change one thing. Re-profile.

The failure mode is skipping steps 1-2 and 6-7. Every skipped step turns tuning into guessing.

## Chapter summary

- A production SLA is a percentile, a window, a work-unit, and a scope — write all four down before you measure. Target p95 and p99; the tail dominates the tuning.
- Allocate a per-stage latency budget on paper before you optimise. If the budget does not close, collapse stages or move work off the request path — do not tune your way out of an architecturally unfeasible target.
- Warm up for at least 100 forward passes before recording — CUDA kernel JIT, cuDNN benchmark cache, Python allocator, and OS page cache all inflate cold measurements.
- Instrument coarse-to-fine: end-to-end histograms first, per-stage timers with `torch.cuda.synchronize()` next, `torch.profiler` for forward-pass ops, `py-spy` for CPU services, `cProfile` for suspect functions.
- Read the latency distribution shape: long tail (input skew), bimodal (branching), sawtooth (GC), step-with-QPS (queueing). Slice by input length, language, and route — a single p95 is not actionable.
- Latency, throughput, and cost trade three ways. Chapter 3 (batching + quantisation) and chapter 4 (ONNX / TorchScript / TensorRT) turn the specific knobs; this chapter is what tells you which knob is worth turning.
