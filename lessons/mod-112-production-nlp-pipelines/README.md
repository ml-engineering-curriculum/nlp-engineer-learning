# mod-112 · Production NLP Pipelines: Composition, Latency, Drift Monitoring

The engineering discipline that turns a Transformer sitting in a notebook into a running service you can hold to a sub-100 ms SLA, ship deterministically, monitor for the failure modes NLP actually has, and postmortem when it degrades. The chapters walk composition (spaCy Projects + Hugging Face pipelines), the latency stack (profiling → batching → quantisation → runtime export via ONNX / TorchScript / TensorRT), packaging and versioning (spaCy Projects DAG, Docker, MLflow Model Registry), the four NLP-specific drift signal shapes (input distribution, vocabulary OOV, output-label, language mix), and end-to-end incident postmortem.

**Estimated effort:** 12 hours

## Learning objectives

- Compose document-level NLP pipelines (segmentation → classification → NER → linking → structured output) with spaCy Projects or Hugging Face pipelines.
- Profile and tune NLP inference latency under sub-100 ms SLAs; reason about batching, quantisation, and ONNX / TorchScript / TensorRT export trade-offs.
- Version, package, and ship an NLP pipeline as a reproducible artifact (spaCy Projects DAG, Docker, MLflow registry).
- Instrument NLP-specific drift signals (input-distribution shift, vocabulary OOV rate, output-label drift, language-mix drift).
- Run an end-to-end production-regression postmortem on a degraded NLP service.

## Chapters

1. [Composing document-level NLP pipelines: spaCy and Hugging Face](01-pipeline-composition-hf-and-spacy.md)
2. [Profiling NLP inference latency under sub-100ms SLAs](02-latency-profiling-under-slos.md)
3. [Batching and quantisation: tuning the model forward pass](03-batching-and-quantisation.md)
4. [Runtime export: ONNX, TorchScript, TensorRT, and when each wins](04-onnx-torchscript-tensorrt-export.md)
5. [Packaging, versioning, and shipping the pipeline as a reproducible artifact](05-packaging-and-reproducible-shipping.md)
6. [Instrumenting NLP-specific drift signals](06-nlp-drift-instrumentation.md)
7. [Postmortem of a degraded NLP service: end-to-end](07-postmortem-of-a-degraded-nlp-service.md)

## Exercises

- [exercise-01 · spaCy Projects or HF pipeline composition](exercises/exercise-01-spacy-projects-or-hf-pipeline-composition.md)
- [exercise-02 · NLP inference latency tuning](exercises/exercise-02-nlp-inference-latency-tuning.md)
- [exercise-03 · ONNX or TorchScript export and quantisation](exercises/exercise-03-onnx-or-torchscript-export-and-quantisation.md)
- [exercise-04 · NLP drift instrumentation](exercises/exercise-04-nlp-drift-instrumentation.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
