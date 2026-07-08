# mod-112-production-nlp-pipelines: Production NLP Pipelines: Composition, Latency, Drift Monitoring

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 12 hours

## Learning objectives

- Compose document-level NLP pipelines (segmentation -> classification -> NER -> linking -> structured output) with spaCy Projects or Hugging Face pipelines
- Profile and tune NLP inference latency under sub-100ms SLAs; reason about batching, quantisation, ONNX / TorchScript / TensorRT export trade-offs
- Version, package, and ship an NLP pipeline as a reproducible artifact (spaCy Projects DAG, Docker, MLflow registry)
- Instrument NLP-specific drift signals (input-distribution shift, vocabulary OOV-rate, output-label drift, language-mix drift)
- Run an end-to-end production-regression postmortem on a degraded NLP service

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
