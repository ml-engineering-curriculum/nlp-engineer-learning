# Batching and Quantisation: Tuning the Model Forward Pass

## Motivation

Chapter 2 attributed the latency; this chapter turns two of the levers that most often move a Transformer forward from "over SLA" to "in SLA": **batch composition** and **numeric precision**. Both are model-agnostic — they apply to any Transformer NER, classifier, cross-encoder, or generator — and both interact with each other and with the runtime choices in chapter 4. Getting them right is often the difference between needing a bigger GPU and shipping on the one you have.

The failure mode is that both look like "free" wins in isolation and are not. Bigger batches always help throughput on the microbenchmark and always hurt latency on real traffic. Quantisation usually preserves accuracy on a general benchmark and sometimes silently destroys it on a domain slice you care about. This chapter walks the trade-offs so the tuning is deliberate.

## Batching: what it actually does

A Transformer forward is a stack of large matrix multiplications (batch × seq × hidden × hidden). Modern GPUs (NVIDIA A100 / H100, consumer RTX) are throughput-optimised: the same matmul on batch 32 does not cost 32× the time of batch 1 — it costs perhaps 2–4× the time, because the launch overhead, the kernel setup, and the memory-hierarchy warmup amortise across the batch.

The **per-item throughput** curve as a function of batch size looks like this qualitatively:

- **Batch 1.** Latency-optimal per item; wastes GPU utilisation.
- **Batch 2–8.** Per-item cost drops sharply as launch overhead amortises.
- **Batch 16–64.** Per-item cost approaches an asymptote — this is the "sweet spot" on most GPUs for Transformer inference.
- **Batch > 64.** Per-item cost roughly flat; total latency growing linearly; increasingly memory-bound.

The exact numbers depend on model size, sequence length, and hardware — always measure on your stack.

**Latency**, however, grows monotonically with batch size in a request-queueing setting: each item in a batch of N waits for the whole batch to finish. If the SLA is a per-item p95, batching to N means every item pays the batch-N forward-pass cost even if it could have been served alone in a fraction of that time.

## Static, dynamic, and continuous batching

Three shapes of batching to know:

- **Static batching.** The batch size is fixed at pipeline construction. Simplest; used by offline processing (`nlp.pipe(batch_size=64)`, `for batch in DataLoader(ds, batch_size=32)`). Correct when you have a large corpus to process end-to-end and no per-item SLA.
- **Dynamic batching (a.k.a. server-side batching, request batching).** The server accumulates incoming requests for a small time window (e.g. 5 ms) or until it hits a max batch size, then runs one forward pass. Every mainstream inference server implements this: [Triton Inference Server](https://github.com/triton-inference-server/server) via its dynamic batcher, [BentoML](https://docs.bentoml.com/) via `bentoml.Runner(batching=True)`, [TorchServe](https://pytorch.org/serve/) via `batch_size` + `max_batch_delay`. The tuning knobs are the batch-max and the wait-max.
- **Continuous batching (a.k.a. iteration-level batching, in-flight batching).** For autoregressive decoders: the server steps generation one token at a time across many active sequences, inserting new requests into the running batch and evicting finished ones. This is the batching shape used by vLLM, TGI, and TensorRT-LLM — an order-of-magnitude throughput improvement over static batching for LLM serving. Yu et al., ["Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) is the reference paper.

For encoder-only NLP (classification, NER, cross-encoder reranking), dynamic batching is the right default. For generation, use a server that does continuous batching — do not roll your own.

## Tuning dynamic batching against an SLA

The core knob is `(max_batch_size, max_batch_delay_ms)`. The intuition:

- **Higher `max_batch_size`** — better throughput, worse p99 for any single item in a full batch.
- **Longer `max_batch_delay_ms`** — better batch fill under low QPS, adds a fixed queueing tax to every request.

A concrete tuning recipe against a 100 ms p95:

1. Measure the batch-1 forward cost. Call it `f(1)` ms.
2. Build the curve `f(b)` for `b in [1, 2, 4, 8, 16, 32, 64]` at your typical sequence length. Take a per-item p95 for each.
3. For each batch size, compute `p95_end_to_end(b) ≈ max_batch_delay + f(b) + fixed_overhead`. The maximum `b` such that this fits your budget is your `max_batch_size`.
4. Set `max_batch_delay_ms` to (a) small (1–5 ms) if traffic is bursty and you care more about tail than throughput, or (b) larger (10–20 ms) if traffic is steady and you care more about throughput than one-sigma tail.
5. Load-test at target QPS, verify the histogram, iterate.

For example, a BERT-base NER with `f(1) = 8 ms`, `f(32) = 40 ms`, a 100 ms p95 budget, and 5 ms of pre/post overhead: `max_batch_size = 32`, `max_batch_delay_ms = 10`. Under a QPS that fills batches on average, most items pay ~55 ms; at the p99 an item can pay `max_batch_delay + f(32) + overhead ≈ 55 ms`; the SLA holds.

## Sequence-length matters more than batch size

Transformer cost is roughly `O(batch × seq × hidden²)` plus `O(batch × seq² × hidden)` for the attention matmul. On typical short inputs (up to a couple hundred tokens) the linear term dominates and batching helps most. On long inputs (thousands of tokens) the quadratic attention term dominates and batching helps less.

Two important sequence-level moves:

- **Length bucketing.** Group items by length before batching. A batch of `[10, 12, 15]`-token items runs in a fraction of the time of a batch of `[10, 12, 2000]`-token items because everything pads to the max. Length-bucketed batching (a.k.a. sorted batching) can double or triple throughput on real traffic. See `transformers`' `LengthGroupedSampler`.
- **Chunking long inputs.** For encoder inputs beyond the model's context window (512 for BERT, 4096 for many modern encoders), chunk with overlap, run each chunk, and merge. Merging strategy is task-specific — chapter 07 of mod-104 covers span-tagging chunk merging; chapter 06 of mod-105 covers QA chunk merging.

## Numeric precision: FP32 → FP16/BF16 → INT8

Model weights and activations live at some numeric precision. Reducing precision reduces memory footprint (fits more replicas per GPU) and speeds up matmuls (Tensor Cores on modern NVIDIA GPUs are ~2× faster on FP16/BF16 than on FP32, and INT8 matmuls are faster still on hardware that supports them).

The four operating points:

- **FP32.** Full precision. Trainings default; inference default only when you have not looked at it.
- **FP16 (half precision).** Half the bytes, ~2× the tensor-core throughput. Numerically fragile: some ops (softmax, layernorm) overflow. In practice, run in mixed precision — FP16 for matmuls, FP32 for the sensitive ops. Standard on GPU inference.
- **BF16 (bfloat16).** Same 16-bit footprint as FP16 but with FP32-range exponents; much more numerically stable. Preferred over FP16 where hardware supports it (Ampere / Hopper NVIDIA GPUs, Google TPUs). No mixed-precision juggling — you can run the whole model in BF16.
- **INT8 quantisation.** 4× the memory reduction, INT8 matmuls on hardware that supports them. Quality-sensitive; requires either post-training quantisation (PTQ) or quantisation-aware training (QAT).

### Post-training quantisation (PTQ)

PTQ takes a trained FP32/FP16 model and produces an INT8 version without further training. Two flavours:

- **Dynamic quantisation.** Weights quantised offline; activations quantised on the fly per forward. PyTorch's `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` does this in one call. Simple, safe, works out of the box on CPU. Modest speed-up (~2×) because activations still transit FP32.
- **Static quantisation.** Weights *and* activations quantised offline against a **calibration set** — a representative sample of inputs used to compute per-channel activation scales. Larger speed-up (~4× on CPU, similar on INT8-capable GPU) but requires the calibration step and is more sensitive to input distribution.

The **calibration set matters more than the algorithm**. Post-training quantisation calibrated on the wrong distribution will silently produce a model that scores 0.85 F1 on your general test set and 0.60 on the domain slice the calibration missed. Rules of thumb:

- 100–500 examples is usually enough for calibration; more helps at diminishing returns.
- The calibration set must reflect production input distribution: length, language mix, domain, formatting. Sampling from production logs is ideal.
- Always evaluate the quantised model on the same held-out set you used to accept the FP32 model, plus your per-slice breakdowns from mod-111 chapter 2. A macro-F1 drop of >1 point on any slice is a stop-ship signal until investigated.

### Quantisation-aware training (QAT)

QAT inserts "fake-quant" nodes into the training graph so the model learns weights that quantise well. Higher engineering cost, one more training run, but recovers most of the quality loss PTQ leaves on the table for aggressive INT8. Reach for QAT when PTQ loses more accuracy than you can accept and you cannot swap for a smaller FP16 model.

### Hugging Face Optimum for quantisation

[Hugging Face Optimum](https://huggingface.co/docs/optimum/) provides one-shot APIs for exporting and quantising Transformer models across ONNX Runtime, OpenVINO, Neural Compressor, and TensorRT backends:

```python
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model = ORTModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER", export=True
)
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir="ner-int8", quantization_config=qconfig)
```

Chapter 4 covers the export half of that recipe (`export=True`) in detail. The point here: quantisation is a supported first-class operation with a mainstream toolchain; there is no need to hand-roll it.

## Bringing batching and quantisation together

The two levers compose but must be measured together:

- **Quantisation changes the batch-size sweet spot.** INT8 matmuls saturate the memory bus differently than FP16; the batch size that maximises throughput moves. Re-sweep `f(b)` after quantising.
- **Quantisation reduces per-item cost more than it reduces per-batch cost.** The launch and kernel overhead is unchanged; the matmul is faster. Net effect: the batch-1 latency improves more (relative) than the batch-64 latency.
- **Batching and quantisation both eat into your quality/latency margin.** Each in isolation may be fine; combined they may push you off the SLA of a downstream evaluation. Always re-run the full eval panel (mod-111) after applying both.

## What not to do

Anti-patterns that appear in every second production incident:

- **Never benchmark quantised models on the training set.** Overfits your quality signal to the exact data the calibration used.
- **Do not pick a batch size from a microbenchmark and ship it to a queued server.** Microbenchmarks feed the model exactly one full batch at a time; real servers deal with bursty QPS where dynamic batching is what actually matters.
- **Do not turn on `torch.backends.cudnn.benchmark = True` for variable-shape inputs.** The kernel-search cache grows unbounded; every new `(batch, seq)` shape pays the search cost. Fine for fixed shapes (padded to a bucket); harmful for varied shapes.
- **Do not report a latency improvement without re-reporting quality.** "20 ms faster" that costs 3 F1 points is a regression, not an improvement. The postmortem in chapter 7 walks through one where this happened.

## Chapter summary

- Batching amortises GPU launch and setup overhead across items — throughput rises sharply from batch 1 to 8-16, then plateaus. Latency, however, grows monotonically with batch size in a queued-server setting; the SLA percentile is what caps you.
- Dynamic batching (`max_batch_size`, `max_batch_delay_ms`) is the right shape for encoder-only NLP servers; continuous batching (vLLM / TGI / TensorRT-LLM) is required for LLM generation.
- Sequence-length effects dominate at long context; length bucketing (`LengthGroupedSampler`) and chunking-with-merge are the standard moves.
- Numeric-precision options: FP32 (default), FP16 (needs mixed-precision juggling), BF16 (preferred where hardware supports it), INT8 via PTQ (dynamic or static + calibration) or QAT.
- Calibration set quality drives PTQ outcomes more than algorithm choice. Sample from production logs, cover length / language / domain slices, and re-run the full eval panel (per-slice, not just aggregate) after quantising.
- Batching and quantisation compose but must be re-measured together. Chapter 4 turns the third big lever — the runtime the quantised model runs on.
