# exercise-03: ONNX or TorchScript Export and Quantisation

**Estimated effort:** 2 hours

## Objective

Export a Transformer NER model to a compiled runtime (**ONNX Runtime** or **TorchScript**), quantise it to INT8 using a **calibration set sampled from a realistic input distribution**, and prove three things: (1) the exported / quantised model is numerically close enough to the PyTorch reference to be safe to ship, (2) it materially reduces inference latency on the target hardware, and (3) it does not silently regress quality — including on per-slice breakdowns. Deliver a `report.md` with parity numbers, latency numbers, quality numbers per slice, and the specific export + quantisation configuration pinned.

## Prerequisites

- Chapters [03](../03-batching-and-quantisation.md) and [04](../04-onnx-torchscript-tensorrt-export.md).
- Python 3.10+; `transformers>=4.40`, `torch>=2.2`, `datasets`, `numpy`, `seqeval`; for the ONNX track: `onnx`, `onnxruntime>=1.18` (CPU) or `onnxruntime-gpu` (GPU), `optimum[onnxruntime]`. For the TorchScript track: PyTorch core is enough.
- The `dslim/bert-base-NER` model (or another CoNLL-2003-trained NER model) and the `conll2003` dataset.
- The latency-harness from exercise-02 (recommended, not required) — pinning to that harness makes the exercises compose.

## Problem statement

### Part A — Pick a track

- **Track ONNX.** Export via Hugging Face Optimum, run under ONNX Runtime with the CPU or CUDA execution provider, quantise via Optimum's `ORTQuantizer` with a real calibration set.
- **Track TorchScript.** Export via `torch.jit.trace`, run in Python (or, for stretch, in a small C++ LibTorch harness). Quantise via `torch.quantization` (dynamic quant is the safe default; static PTQ is the stretch).

The ONNX track is the more production-shaped default. The TorchScript track is instructive when your deployment already speaks LibTorch or when you want to see PyTorch's own quantisation flow end-to-end.

### Part B — Export

Produce `export.py` that produces `artifact/` containing the exported model, the tokenizer files, and a `signature.json`.

`signature.json` fields (from chapter 4):

```json
{
  "format": "onnx" | "torchscript",
  "opset": 17,
  "precision": "fp32",
  "input_names": ["input_ids", "attention_mask"],
  "dynamic_axes": {"input_ids": {"0": "batch", "1": "seq"}},
  "source_model": "dslim/bert-base-NER",
  "source_revision": "<hf-commit-sha>",
  "exporter_version": "optimum==X.Y.Z, onnxruntime==A.B.C, torch==P.Q.R",
  "target_hardware": "cpu-avx512-vnni" | "cuda:0-a100" | "cpu-generic"
}
```

Every field is required. If a field does not apply for your track, populate it with an explicit `null` and comment in `report.md`.

### Part C — Numeric parity check (FP32)

Produce `parity.py` that:

- Loads the PyTorch model and the exported model.
- Runs both on 500 sentences from the CoNLL-2003 validation split.
- Reports `max_abs_logit_diff`, `mean_abs_logit_diff`, `argmax_flip_rate` (fraction of tokens where the argmax label differs), and the count of *entities* whose set changed (any change to the seqeval-extracted set of `(start, end, type)` triples).

Success criterion at FP32: `max_abs_logit_diff < 1e-4`, `argmax_flip_rate == 0`, entity set delta = 0. If any of these fail, the export is wrong; do not proceed to quantisation. Debug it (usually an op that fell back to a fp64 kernel, an opset mismatch, or a `dynamic_axes` misconfiguration).

Report the numbers in `report.md`.

### Part D — Build the calibration set (deliberately)

Chapter 3's key point: calibration-set quality drives PTQ outcomes more than the algorithm. Assemble `calibration.jsonl` with **200-500 examples** that reflects the input distribution you expect at inference time. Explicit requirements:

- Cover the length distribution: at least 20 % short (≤ 40 tokens), 60 % medium (40-150), 20 % long (150-256).
- Cover *domains* if your source has them: if using CoNLL-2003, include some Reuters news items plus (if you have any) a small out-of-domain sample from `wikiann/en` or a modern news feed — enough to catch a calibration gap that a pure CoNLL calibration set would miss.
- Every calibration item has an ID and a documented source.
- Ship `calibration.jsonl.sha256` alongside the file; the signature must record `calibration_set_sha`.

Produce `build_calibration.py` that assembles the file from named sources and prints the sha.

### Part E — Quantise

- **Track ONNX.** Use `optimum.onnxruntime.ORTQuantizer` with `AutoQuantizationConfig` — either `avx512_vnni(is_static=True, per_channel=True)` (recommended on modern x86) or `avx512(is_static=True)` (older CPUs). For GPU-INT8, use the appropriate config. Point the calibration reader at your `calibration.jsonl`. Save to `artifact-int8/`; update the signature.
- **Track TorchScript.** Start with `torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)` — no calibration required for dynamic quant, but explain in `report.md` why static quant would need it and would probably do better on your workload.

Update `signature.json` with `precision: "int8"`, `calibration_set_sha`, and (ONNX) `quantization_config`.

### Part F — Numeric parity check (INT8)

Re-run `parity.py` against the quantised model. Report the same four numbers. INT8 expectations (chapter 4):

- Logit diffs can be much larger than FP32 (`atol ~ 0.1-0.5` is common).
- `argmax_flip_rate`: report and interpret. Under ~1 % of tokens is typical; higher may be acceptable, but must be investigated.
- Entity-set delta: report the fraction of documents where the extracted entity set changes. This is what actually matters downstream.

If the entity-set delta is > 5 %, treat as a stop-ship signal for the quantisation configuration and either (a) improve the calibration set, (b) switch to per-channel quantisation, (c) fall back to FP16 / BF16 as a less aggressive alternative and re-measure.

### Part G — Latency benchmark

Benchmark PyTorch (baseline), exported FP32, and quantised INT8 on the same hardware, same input distribution, and record:

- Per-sample latency at batch 1: p50, p95, p99 over 1 000 warmed inputs.
- Per-sample latency at batch 8 and batch 32.
- Throughput (samples/sec) at each batch size.
- Peak resident memory (`/proc/<pid>/status:VmRSS` for CPU, `torch.cuda.max_memory_allocated()` for GPU).

Report as a table in `report.md`, columns: baseline / FP32 export / INT8 quant. Include the same environment pin from exercise-02 (CPU / GPU / CUDA / driver / library versions). If you did exercise-02, reuse its harness verbatim; if not, write a minimal one.

### Part H — Quality re-verification (per-slice)

Score all three artifacts on the CoNLL-2003 validation set with `seqeval` in `mode="strict"`, `scheme=IOB2`. Report:

- Overall micro-F1 with 95 % bootstrap CI (mod-111 chapter 2 pattern).
- Per-entity-type F1 (PER / ORG / LOC / MISC).
- Per-input-length-bucket F1 (short / medium / long, matching the calibration buckets).
- Paired bootstrap for the delta between PyTorch baseline and quantised INT8.

The rule from chapter 3: **a > 1-point macro-F1 drop on any slice is a stop-ship signal until investigated.** If the overall number is flat but per-length or per-type shows a bigger drop, name the slice in `report.md` and describe what would change — better calibration data covering that slice, per-channel quant, or reverting to FP16.

### Part I — The report

`report.md` sections, in order:

1. Track chosen and rationale.
2. Environment / hardware pin.
3. Export: signature block quoted from `signature.json`.
4. FP32 parity: numbers table + pass/fail.
5. Calibration set: composition summary + sha256.
6. Quantisation: config + INT8 signature update.
7. INT8 parity: numbers table + flip-rate interpretation.
8. Latency table (baseline / FP32 / INT8; three batch sizes; percentiles + throughput + memory).
9. Quality table (overall + per-type + per-length; deltas with paired-bootstrap).
10. Ship / no-ship decision with justification.

Every claim in the report cites its source file (CSV, JSON) or a specific benchmark run.

## Starter guidance

- **Do not skip the FP32 parity check.** If FP32 export does not match, the INT8 will not match a fortiori, and you will spend the rest of the exercise chasing the wrong bug.
- **Calibration coverage over calibration size.** 500 well-distributed examples beat 5 000 all from one domain.
- **Bundle the tokenizer.** Chapter 4 rule; without it your artifact is not self-contained. Ship `tokenizer.json`, `tokenizer_config.json`, `config.json`, `special_tokens_map.json` in the artifact directory.
- **Turn off `torch.backends.cudnn.benchmark = True` for the PyTorch baseline** when your inputs are variable-shape (unless you length-bucket). It thrashes the kernel cache and inflates cold measurements — chapter 3.
- **Report the argmax-flip rate, not just the logit delta.** A large logit delta with zero flips is safe; a small logit delta with 3 % flips is not.
- **On CPU, INT8 wins on AVX-512 VNNI machines.** On older CPUs the win is smaller. Note your CPU generation in the pin so the numbers are interpretable.

## Acceptance criteria

- [ ] `export.py` produces `artifact/` with the exported FP32 model, tokenizer files, and a complete `signature.json`.
- [ ] `parity.py` reports the four numeric-parity numbers for FP32; FP32 passes (`max_abs_logit_diff < 1e-4`, zero flips, zero entity-set delta) before proceeding.
- [ ] `calibration.jsonl` covers the required length distribution and (at least two) domain sources; `calibration.jsonl.sha256` matches the `signature.json` `calibration_set_sha`.
- [ ] Quantised `artifact-int8/` produced; signature updated with precision and calibration hash.
- [ ] INT8 parity numbers reported; entity-set delta discussed; if > 5 % the stop-ship response is documented.
- [ ] Latency table with three artifacts × three batch sizes; percentiles, throughput, peak memory; hardware pin included.
- [ ] Quality table with overall + per-type + per-length F1, CIs, and paired-bootstrap; > 1-point per-slice drop investigated and documented.
- [ ] `report.md` follows the required structure and includes an explicit ship / no-ship decision.

## Stretch goals

- **Add a FP16 / BF16 midpoint.** Export a mixed-precision variant and add a column to the latency and quality tables. Which precision is the best trade for your target SLA?
- **Static PTQ on TorchScript.** Convert the model to a quantisable module, run static calibration with `torch.quantization.prepare` + `convert`, and compare with dynamic quant on the same inputs.
- **Cross-runtime bake-off.** Export the same model to both ONNX Runtime and TorchScript, and (if your fleet is NVIDIA) an ORT `TensorrtExecutionProvider` variant. Add a fourth column to the latency table and pick the winner.
- **LibTorch C++ harness.** Wrap the TorchScript artifact in a minimal C++ inference harness and benchmark it against the Python one. Report the Python-vs-C++ latency delta.
- **Batch-1 vs. dynamic-batched latency.** Build a small dynamic-batcher around the INT8 artifact (using exercise-02's harness) and re-report p50 / p95 / p99 vs. batch 1 at concurrency 16.
- **Portability check.** Copy the artifact to a different machine (different CPU generation, or CPU when built on GPU) and re-run parity + latency. Where does the artifact travel well and where does it not?

## Deliverables

```
export.py
parity.py
build_calibration.py
benchmark.py
calibration.jsonl
calibration.jsonl.sha256
artifact/                 model.onnx OR model.pt  tokenizer.json  tokenizer_config.json  config.json  special_tokens_map.json  signature.json
artifact-int8/            (same layout with quantised model + updated signature)
report.md
tables/                   latency.csv  quality.csv  parity.csv
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
