# Runtime Export: ONNX, TorchScript, TensorRT, and When Each Wins

## Motivation

A Transformer trained in PyTorch runs, by default, through the Python interpreter dispatching one op at a time to CUDA. That is the most flexible and the slowest way to serve it. Every production Transformer runtime — ONNX Runtime, TorchScript, TensorRT, OpenVINO, TVM — trades some of that flexibility for aggressive kernel fusion, graph-level optimisation, and interpreter removal.

This chapter is about picking the right target for your workload, understanding what each one actually does, and knowing where the compatibility bear traps are. The three you will meet most often in NLP serving are ONNX Runtime, TorchScript, and TensorRT — with `torch.compile` as an in-process alternative that has narrowed the gap significantly since PyTorch 2.0.

## What a runtime actually does

A trained PyTorch model at inference time does five things:

1. Trace or eagerly interpret Python control flow.
2. Dispatch each op through the ATen dispatcher.
3. Launch a CUDA kernel per op (or a fused kernel when one exists).
4. Return tensors to Python.
5. Repeat.

A compiled runtime replaces steps 1–4 with:

1. A **static graph** captured once at export time (traced or scripted).
2. **Kernel fusion** — chains of pointwise ops merged into one kernel; matmul + bias + activation into one fused kernel; the whole attention block into a single [FlashAttention](https://github.com/Dao-AILab/flash-attention)-style kernel on capable hardware.
3. **Constant folding, dead-code elimination, layout optimisation** — the graph is rewritten before any kernels launch.
4. A **serving loop** that hands tensors between fused kernels with no Python in the middle.

The result: 2–10× lower latency and higher throughput on the same weights and hardware, with the caveats below.

## `torch.compile` — the in-process option

Since PyTorch 2.0, `torch.compile(model)` returns a version of `model` that JIT-compiles the forward graph on first call and reuses it afterwards. It uses TorchDynamo to capture the graph and Inductor (with Triton kernels) to generate fused CUDA. No export step, no separate runtime process.

```python
import torch
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").cuda().eval()
model = torch.compile(model, mode="reduce-overhead")

with torch.inference_mode():
    for batch in batches:
        out = model(**batch)
```

Modes: `default`, `reduce-overhead` (CUDA graphs; best for small, repeated shapes — great for served inference), `max-autotune` (spends longer compiling for best runtime; use for long-running batch jobs).

The trade-off: `torch.compile` recompiles on *shape changes*. Variable sequence-length inputs are a footgun — every unique `(batch, seq)` triggers a fresh compile until the cache saturates. Fixes: length-bucketing (chapter 3) so the number of unique shapes is bounded; padding to a fixed max length; or `torch._dynamo.config.cache_size_limit` tuned up.

When it wins: same-process PyTorch service, GPU, small-to-medium models, low overhead cost to add. When it loses: heavy multi-GPU serving, need to serve outside Python, need to target CPU + INT8 (ONNX Runtime is usually cleaner there).

## ONNX + ONNX Runtime

[ONNX](https://onnx.ai/) is an interchange format for computation graphs; [ONNX Runtime](https://onnxruntime.ai/) is Microsoft's inference engine that consumes ONNX and dispatches to `CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`, `OpenVINOExecutionProvider`, and others. Hugging Face's [Optimum](https://huggingface.co/docs/optimum/) is the mainstream way to export a Transformer to ONNX and run it under ONNX Runtime.

Export + serve, end to end:

```python
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = ORTModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER", export=True
)
model.save_pretrained("ner-onnx")
tok.save_pretrained("ner-onnx")

# Later, at serve time:
model = ORTModelForTokenClassification.from_pretrained("ner-onnx", provider="CUDAExecutionProvider")
inputs = tok("Barack Obama visited Berlin.", return_tensors="pt")
outputs = model(**inputs)
```

What ONNX Runtime gives you:

- **Cross-framework portability.** The same ONNX file runs in Python, C++, C#, Java, JavaScript (via `onnxruntime-web`), and on mobile.
- **CPU-first execution** that is genuinely competitive for encoder-only NLP with INT8. On modern x86 with AVX-512 VNNI, an INT8 BERT-base on ONNX Runtime is often 2–4× faster than the PyTorch equivalent on the same CPU.
- **Graph optimisation levels** (`SessionOptions.graph_optimization_level`: `ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC`, `ORT_ENABLE_EXTENDED`, `ORT_ENABLE_ALL`). Turn to `ALL` for production; export-time constant folding is included.
- **Quantisation toolchain** (`optimum.onnxruntime.ORTQuantizer`) integrated with the export — see chapter 3.

Compatibility caveats:

- **Op coverage.** Not every PyTorch op has an ONNX equivalent; exotic custom ops or dynamic control flow fail export. For standard `transformers` architectures this is almost never a problem; for research-stage code it can be. Set `opset` version explicitly (`export(..., opset=17)`) and pin it.
- **Dynamic axes.** ONNX exports have to declare which axes are dynamic (batch, sequence). If you export with a fixed shape and try to run with a different one, you get a runtime error. Optimum handles this correctly by default for the standard Transformer heads.
- **Version pins.** `onnx`, `onnxruntime`, and the exporter version all interact. Pin them together in the same requirements file; upgrade in tandem.

## TorchScript

TorchScript is PyTorch's own graph representation with two modes: **tracing** (`torch.jit.trace(model, example_inputs)`) records the ops a forward call executes on the example input; **scripting** (`torch.jit.script(model)`) statically compiles Python to TorchScript IR. The output is a `.pt` file callable from a C++ TorchScript runtime (LibTorch) with no Python dependency.

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").eval()
tok = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
example = tok("Barack Obama visited Berlin.", return_tensors="pt")

traced = torch.jit.trace(model, (example["input_ids"], example["attention_mask"]))
traced.save("ner-traced.pt")
```

When TorchScript wins:

- **You already deploy C++ services** (a common shape at latency-sensitive Java/Go/Rust shops via a small C++ shim). LibTorch is the least-friction path.
- **You want to stay inside PyTorch semantics** without translating to an interchange format.

When TorchScript loses (and this is most of the time in 2025):

- **Op coverage for scripting is incomplete** — annotations required for Python control flow, exceptions, and some optional-tensor shapes. Tracing avoids this but silently bakes in the control flow of the traced call, which is wrong for models with input-dependent branches.
- **The PyTorch team's focus has shifted to `torch.compile`.** New optimisations land there first.
- **ONNX Runtime has caught up on CPU** and is more portable, so the historical TorchScript case ("PyTorch-native, C++ deployable, fast") is now less distinct.

Reach for TorchScript when your deployment shape specifically wants it. Otherwise start with `torch.compile` (in-process) or ONNX Runtime (cross-language, more optimisations).

## TensorRT

[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) is a proprietary inference SDK that produces highly-optimised engines for NVIDIA GPUs. It does the most aggressive kernel fusion of any of these runtimes, has specialised INT8 and FP8 support tied to Tensor Cores, and can produce 2–5× lower latency than ONNX Runtime on the same NVIDIA GPU for many Transformer inference workloads.

The two paths in:

- **`onnxruntime` with `TensorrtExecutionProvider`** — ONNX in, TensorRT-optimised subgraphs. Easiest; ONNX Runtime falls back to CUDA EP for ops TensorRT does not support.
- **`tensorrt` directly** or via [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for LLM serving. Higher engineering cost; harder to update.

TensorRT-specific properties:

- **Engines are hardware-specific.** An engine built for A100 does not necessarily run on H100 or L4; you build one engine per GPU SKU you deploy on. Build steps take minutes to hours for large models; script and cache them.
- **INT8 calibration** is first-class and often best-in-class — the tooling is designed for it and the resulting engines saturate Tensor Cores more effectively than PTQ under other runtimes.
- **FP8** on Hopper (H100) and later is TensorRT's headline feature. If your fleet is H100-class and workloads justify it, FP8 inference is currently the fastest way to serve.
- **Vendor lock-in.** TensorRT is NVIDIA-only, closed-source, and evolves on its own schedule. Fine when NVIDIA is your fleet; disqualifying when portability matters.

## The decision, distilled

The one-page recommendation:

| Deployment shape | First choice | Second choice | Notes |
|---|---|---|---|
| Same-process Python, GPU, low ops overhead | `torch.compile` | ONNX Runtime + CUDA EP | Use `reduce-overhead` mode; watch shape-cache thrash |
| Cross-language service (Java / Go / Rust / C#) | ONNX Runtime | TorchScript + LibTorch | ONNX Runtime has the widest language support |
| CPU-only serving | ONNX Runtime + INT8 (PTQ) | OpenVINO (Intel CPUs) | AVX-512 VNNI is the sweet spot for INT8 encoders |
| Latency-critical NVIDIA GPU serving | TensorRT (direct or via ORT-TRT EP) | ONNX Runtime + CUDA EP | Rebuild engines per GPU SKU |
| Autoregressive LLM serving | vLLM or TGI (continuous batching) | TensorRT-LLM | Chapter 3; not a compiled-runtime pick, a serving-framework pick |
| Mobile / edge | ONNX Runtime Mobile or TFLite | — | Model size / battery matters more than latency |

Two rules of thumb regardless of pick:

- **Benchmark your model, on your hardware, on your input distribution.** The rankings above are correct on average; they are not correct for every model. A quick sweep with all three feasible backends on 500 representative inputs beats picking based on Twitter hype.
- **Pin everything.** ONNX opset, ONNX Runtime version, TensorRT version, driver version, CUDA version. Cross-runtime numeric differences are almost always version-drift artefacts.

## Numeric parity: the always-check step

A compiled runtime is only useful if it produces the same outputs the PyTorch model does. Every export must pass a **numeric-parity check** before it goes into production:

```python
import numpy as np, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification

tok = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
pt = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").eval()
onnx = ORTModelForTokenClassification.from_pretrained("ner-onnx")

for text in probe_texts:
    x = tok(text, return_tensors="pt")
    with torch.inference_mode():
        pt_logits = pt(**x).logits.numpy()
    ort_logits = onnx(**x).logits
    # FP32 export: atol=1e-5 is realistic; FP16/INT8 exports: larger, and check argmax equality
    assert np.allclose(pt_logits, ort_logits, atol=1e-4), text
```

For FP32 exports, expect atol < 1e-4. For FP16 / BF16 exports, atol grows to ~1e-2; check that the **argmax** (i.e. the decoded label) matches, not just the logit values. For INT8 exports, expect a fraction of items where argmax flips — measure that fraction on a held-out set, and re-run the full eval panel (per-slice) from mod-111. A flip rate above ~1% is a stop-ship signal without further investigation.

## Bundle the tokenizer too

Every runtime export gives you a compiled model. Your service also runs a tokenizer, and the tokenizer version + files matter as much as the model — a tokenizer mismatch is silently wrong (BPE / WordPiece rules encode differently) and often not caught until quality regresses in production.

- Ship `tokenizer.json` (fast tokenizer, `tokenizers` library) alongside the exported model in the same artifact.
- ONNX Runtime has an `onnxruntime-extensions` package with tokenizer ops built in — you can bake the tokenizer into the ONNX graph for a single self-contained artifact.
- Pin the `tokenizers` and `transformers` versions in the requirements accompanying the artifact.

Chapter 5 makes this a formal packaging requirement.

## Chapter summary

- Runtime export replaces Python-op-dispatch inference with a compiled static graph of fused kernels. Typical wins on NLP encoders: 2–4× (CPU with INT8 via ONNX Runtime), 2–5× (NVIDIA GPU via TensorRT), similar range with `torch.compile` for in-process PyTorch serving.
- `torch.compile` (PyTorch 2.0+) is the cheapest in-process option — mode `reduce-overhead` for served inference — but recompiles on shape changes, so length-bucket or fix your input shapes.
- ONNX Runtime is the cross-language, cross-hardware default; Optimum makes export one line. It has the widest language bindings and the best CPU story for INT8 encoders.
- TorchScript's historical case has narrowed against `torch.compile` and ONNX Runtime; reach for it when your deployment specifically wants LibTorch in C++.
- TensorRT is the latency-optimal NVIDIA-only option — best kernel fusion, first-class INT8 / FP8, engines built per GPU SKU. Vendor-locked; use when NVIDIA is the fleet and latency wins matter enough to accept the ops cost.
- Every export needs a numeric-parity check against the PyTorch reference and a re-run of the full eval panel (mod-111 per-slice) at the target precision. FP32: `atol < 1e-4`. INT8: check argmax flip rate.
- Ship the tokenizer with the model in the same artifact; a tokenizer mismatch is a silent quality bug. Chapter 5 turns this into a packaging requirement.
