# Resources for mod-112 · Production NLP Pipelines

Primary docs, standards, and papers. Prefer these over blog posts when digging deeper.

## Pipeline composition

- [spaCy — Language pipeline documentation](https://spacy.io/usage/processing-pipelines) — component ordering, `nlp.pipe`, batching, custom `@Language.component` and `@Language.factory` decorators.
- [spaCy Projects — official guide](https://spacy.io/usage/projects) — `project.yml` structure, commands / workflows, remotes, packaging.
- [spaCy — `spacy package` CLI reference](https://spacy.io/api/cli#package) — turning a trained pipeline into an installable Python wheel.
- [Hugging Face `transformers.pipeline` — documentation](https://huggingface.co/docs/transformers/main_classes/pipelines) — task-specific pipelines, `batch_size`, `device`, custom `Pipeline` subclassing.
- [Hugging Face `datasets` — streaming and mapping](https://huggingface.co/docs/datasets/stream) — the data side of streamed pipeline composition.
- [spaCy-transformers — using HF models as spaCy components](https://github.com/explosion/spacy-transformers) — the bridge between the two ecosystems.
- [Haystack — pipeline framework](https://docs.haystack.deepset.ai/docs/pipelines) — one of the mainstream DAG runners for multi-component HF pipelines.
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/) — typed record models for inter-stage contracts.

## Latency profiling

- [Google SRE workbook — implementing SLOs](https://sre.google/workbook/implementing-slos/) — the reference for percentile / window / work-unit thinking; ties directly to chapter 2's SLA framing.
- [`torch.profiler` — documentation](https://pytorch.org/docs/stable/profiler.html) — per-op CPU / CUDA profiling, Chrome trace export.
- [PyTorch profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — worked walk-through of the profiler.
- [`py-spy` — sampling profiler for Python](https://github.com/benfred/py-spy) — flame graphs without instrumentation.
- [`cProfile` — Python standard library](https://docs.python.org/3/library/profile.html) — deterministic function-level profiling.
- [Prometheus histogram best practices](https://prometheus.io/docs/practices/histograms/) — how to build latency histograms that give correct percentiles at query time.

## Batching, quantisation, and runtime export

- [Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) — the paper that introduced continuous / iteration-level batching, the foundation of vLLM and TGI.
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) — mainstream server with dynamic batching, model repository, ensemble scheduling.
- [BentoML documentation](https://docs.bentoml.com/) — runners, dynamic batching, model store.
- [TorchServe documentation](https://pytorch.org/serve/) — PyTorch's own model server with `batch_size` and `max_batch_delay`.
- [vLLM](https://github.com/vllm-project/vllm) and [Hugging Face TGI](https://github.com/huggingface/text-generation-inference) — continuous-batching serving for LLM generation.
- [PyTorch quantization documentation](https://pytorch.org/docs/stable/quantization.html) — dynamic vs. static PTQ vs. QAT.
- [Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017](https://arxiv.org/abs/1706.04599) — the calibration reference cross-cited in mod-111.
- [Hugging Face Optimum — ONNX Runtime integration](https://huggingface.co/docs/optimum/onnxruntime/overview) — export and quantise Transformers with one API.
- [Hugging Face Optimum — quantisation guide](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization) — dynamic vs. static PTQ under Optimum + ORT.
- [ONNX — specification](https://onnx.ai/onnx/) and [ONNX opset versioning](https://onnx.ai/onnx/operators/) — the interchange format and its op-set evolution.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — execution providers, graph optimisation levels, quantisation tooling, extensions.
- [`torch.compile` documentation](https://pytorch.org/docs/stable/generated/torch.compile.html) and [tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) — TorchDynamo + Inductor as an in-process alternative to export.
- [TorchScript documentation](https://pytorch.org/docs/stable/jit.html) — tracing, scripting, LibTorch runtime.
- [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA-specific inference runtime and LLM-serving library.
- [safetensors](https://github.com/huggingface/safetensors) — safe weight-serialisation format used by `transformers`.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — fused attention kernel referenced in chapter 4.

## Packaging and reproducibility

- [Google SRE book — Release engineering](https://sre.google/sre-book/release-engineering/) — the "how software gets to production" chapter that underpins chapter 5.
- [Semantic Versioning 2.0.0](https://semver.org/) — the versioning spec chapter 5 maps onto NLP artifacts.
- [MLflow Model Registry documentation](https://mlflow.org/docs/latest/model-registry.html) — registered models, versions, aliases, transitions.
- [MLflow Models — documentation](https://mlflow.org/docs/latest/models.html) — model flavours (`pyfunc`, `transformers`), signature inference.
- [Hugging Face Model Cards](https://huggingface.co/docs/hub/model-cards) — the model-card spec and template.
- [Hugging Face Hub — `huggingface_hub` client](https://huggingface.co/docs/huggingface_hub) — programmatic model / dataset publishing.
- [`pip-tools`](https://github.com/jazzband/pip-tools) and [`uv pip compile`](https://docs.astral.sh/uv/pip/compile/) — reproducible Python dependency locking.
- [Docker best practices for Python](https://docs.docker.com/language/python/best-practices/) — layer ordering, non-root users, multi-stage builds.
- [Argo Rollouts](https://argo-rollouts.readthedocs.io/) and [Flagger](https://docs.flagger.app/) — progressive-delivery controllers for canary / blue-green.

## Drift detection and monitoring

- [Lipton, Wang, & Smola, *Detecting and Correcting for Label Shift with Black Box Predictors*, ICML 2018](https://arxiv.org/abs/1802.03916) — label-shift detection reference from chapter 6.
- [Gama et al., *A Survey on Concept Drift Adaptation*, ACM Computing Surveys 2014](https://dl.acm.org/doi/10.1145/2523813) — the canonical concept-drift survey.
- [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — production ML principles including monitoring guidance.
- [Evidently AI — open-source ML monitoring](https://github.com/evidentlyai/evidently) — data-drift, target-drift, text-drift dashboards including PSI / KS / chi-square and text-specific descriptors.
- [NannyML documentation](https://nannyml.readthedocs.io/) — performance estimation without ground truth (CBPE) and multivariate drift detection.
- [Alibi Detect documentation](https://docs.seldon.io/projects/alibi-detect/en/stable/) — MMD, LSDD, learned-kernel drift detectors including text.
- [fastText — language identification (`lid.176`)](https://fasttext.cc/docs/en/language-identification.html) — the standard language-ID model referenced in chapter 6 and mod-107.
- [pycld3 / cld3](https://github.com/bsolomon1124/pycld3) — lightweight Compact Language Detector v3.
- [Population Stability Index — Karakoulas, *Empirical Validation of Retail Credit-Scoring Models*, RMA Journal 2004](https://www.researchgate.net/publication/238478420) — origin of the PSI convention widely used in ML monitoring.
- [Prometheus documentation](https://prometheus.io/docs/) and [Grafana documentation](https://grafana.com/docs/) — the mainstream open-source metric + dashboard stack.
- [OpenTelemetry — semantic conventions for metrics](https://opentelemetry.io/docs/specs/semconv/general/metrics/) — for teams standardising on OTel over raw Prometheus.

## Postmortems and incident response

- [Google SRE book — Postmortem Culture: Learning from Failure](https://sre.google/sre-book/postmortem-culture/) — the reference for blameless postmortems; chapter 7's foundation.
- [Google SRE workbook — Managing incidents](https://sre.google/workbook/incident-response/) — the incident-lifecycle framework (detection → response → mitigation → root cause → structural fix).
- [PagerDuty Incident Response documentation](https://response.pagerduty.com/) — practical incident-command guidance widely adopted outside Google.
- [Etsy — Debriefing Facilitation Guide (John Allspaw)](https://extfiles.etsy.com/DebriefingFacilitationGuide.pdf) — practitioner's guide to running blameless postmortems.
- [SRE workbook — SLOs, SLIs and error budgets](https://sre.google/workbook/implementing-slos/) — the framework for deciding when to page vs. Slack vs. dashboard.
- [Learning from Incidents in Software (learningfromincidents.io)](https://www.learningfromincidents.io/) — cross-company practitioner community and reading list on incident learning.
