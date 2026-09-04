# Packaging, Versioning, and Shipping the Pipeline as a Reproducible Artifact

## Motivation

A pipeline that exists only in a notebook is not a pipeline. A pipeline that ships as "the checkpoint on `/mnt/nfs/models/latest`" is worse — it appears to work until someone else needs to reproduce a number, roll back a bad deploy, or diagnose a regression whose model version is no longer available. Packaging is the discipline that makes the artifact addressable, reproducible, and auditable.

The three moves this chapter covers are what most production NLP teams converge on:

1. **A DAG that produces the artifact** — spaCy Projects, DVC, or a Makefile of well-shaped scripts. Anyone can re-run it and get a byte-identical (or at least verified-equivalent) output.
2. **A container that runs the artifact** — a Dockerfile that pins the OS, Python, framework, and library versions and installs the artifact deterministically.
3. **A registry that stores versions of the artifact** — MLflow Model Registry, Hugging Face Hub, spaCy's own package registry, or the object-storage-with-metadata pattern.

Miss any of the three and reproducibility breaks in a predictable direction: no DAG means every model is a snowflake; no container means every environment drifts; no registry means every rollback is a treasure hunt.

## What "reproducible artifact" actually means

Reproducibility is an operational property, not an aesthetic one. A reproducible NLP pipeline artifact is one where, given the artifact identifier, an engineer can:

- **Re-run inference** on new data and get outputs identical to what production would produce (byte-identical for FP32; numerically close for FP16/INT8 with a stated tolerance).
- **Re-run the training/build** and produce the same (or verifiably equivalent) artifact, given the same data snapshot and hyperparameters.
- **Retrieve every input** — data hash, code commit, config file, tokenizer files, base-model checkpoint SHA, hyperparameters, hardware notes — via the artifact's metadata.
- **Bind the artifact to a decision** — this artifact scored these numbers on this eval suite, was approved by this stakeholder, was deployed at this timestamp.

Falling short on any of these means a future incident cannot be resolved without guesswork.

## spaCy Projects as the pipeline DAG

Chapter 1 introduced the spaCy Projects `project.yml`. For packaging, its properties are:

- **`assets`** are declared with checksums and URLs. `spacy project assets` fetches them; if the local checksum matches, no download happens; if not, the file is refetched and re-verified.
- **`commands`** declare `deps` (input paths) and `outputs` (output paths); each command is skipped when both hash-match the previous run. This gives you Makefile semantics without a Makefile.
- **`workflows`** compose commands in an explicit order.
- **`spacy project push` / `pull`** synchronise artifacts against a remote (S3, GCS, Azure Blob, local mount) so the DAG output can be shared across the team without re-training.

A packaging-ready `project.yml` (extending chapter 1's fragment):

```yaml
title: "Customer-intent NER pipeline"
vars:
  version: "0.2.0"
  gpu_id: 0
  base_model: "en_core_web_lg"
directories: ["assets", "corpus", "training", "packages", "metrics"]
assets:
  - dest: "assets/train.jsonl"
    checksum: "8e6d..."
    url: "s3://our-corpus/intent/train.jsonl"
  - dest: "assets/dev.jsonl"
    checksum: "3a1c..."
    url: "s3://our-corpus/intent/dev.jsonl"
commands:
  - name: "convert"
    script: ["python scripts/to_spacy.py assets/train.jsonl corpus/train.spacy",
             "python scripts/to_spacy.py assets/dev.jsonl corpus/dev.spacy"]
    deps: ["assets/train.jsonl", "assets/dev.jsonl", "scripts/to_spacy.py"]
    outputs: ["corpus/train.spacy", "corpus/dev.spacy"]
  - name: "train"
    script: ["python -m spacy train configs/config.cfg --output training/ --gpu-id ${vars.gpu_id}
             --paths.train corpus/train.spacy --paths.dev corpus/dev.spacy"]
    deps: ["corpus/train.spacy", "corpus/dev.spacy", "configs/config.cfg"]
    outputs: ["training/model-best"]
  - name: "evaluate"
    script: ["python -m spacy benchmark accuracy training/model-best corpus/dev.spacy --output metrics/eval.json"]
    deps: ["training/model-best", "corpus/dev.spacy"]
    outputs: ["metrics/eval.json"]
  - name: "package"
    script: ["python -m spacy package training/model-best packages
             --name intent_ner --version ${vars.version} --force
             --meta-path meta.json"]
    deps: ["training/model-best"]
    outputs: ["packages/en_intent_ner-${vars.version}"]
remotes:
  s3: "s3://our-models/spacy-projects/intent-ner"
workflows:
  all: ["convert", "train", "evaluate", "package"]
```

`spacy project run all` runs the workflow. `spacy project push` pushes new artifacts to the remote. The `meta.json` on `spacy package` embeds version, description, license, dependency pins, and the eval numbers — the package is self-describing.

The output of `spacy package` is a Python package with a versioned name (`en_intent_ner-0.2.0`) that installs with `pip install ./en_intent_ner-0.2.0-py3-none-any.whl`. That wheel is the shipping unit.

## Hugging Face artifacts

For Hugging Face-based pipelines, the equivalent set of moves:

- **`save_pretrained(dir)`** on model, tokenizer, and any config produces a directory that `from_pretrained(dir)` restores. Ship this directory as the artifact.
- **Model cards.** Every artifact needs a `README.md` in the model directory following the [Hugging Face model card spec](https://huggingface.co/docs/hub/model-cards) — intended uses, out-of-scope uses, training data description, evaluation numbers, license, contact. Non-optional for anything shipped to a serving fleet.
- **Push to Hub or private mirror.** `push_to_hub(repo_id, private=True)` sends the artifact to a Hub repo — the Hub commit SHA is the version. For teams that cannot use huggingface.co, self-host with the same API surface via the Hub's on-prem edition or via `datasets`/`huggingface_hub`-compatible object storage.
- **`safetensors`.** Save weights in [safetensors](https://github.com/huggingface/safetensors) format (`safe_serialization=True`) — safer than pickle-based `.bin`, no arbitrary code execution on load. This is the current default in `transformers`.

For runtime-exported artifacts (chapter 4), ship the exported model plus the original tokenizer as one directory: `model.onnx`, `tokenizer.json`, `tokenizer_config.json`, `config.json`, `special_tokens_map.json`, `README.md`, and a `signature.json` capturing the export config (opset, precision, target hardware, calibration set hash).

## The container

The artifact is the model + tokenizer + metadata. The container is the code that runs them plus its environment. Both are needed and both need to be pinned.

A production Dockerfile for an NLP inference service:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Two-step install for a fast rebuild loop: deps first, code second
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ /app/src/
COPY artifacts/en_intent_ner-0.2.0 /app/artifacts/en_intent_ner-0.2.0

ARG GIT_SHA=unknown
ARG ARTIFACT_VERSION=0.2.0
ENV GIT_SHA=${GIT_SHA} ARTIFACT_VERSION=${ARTIFACT_VERSION}

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8080/healthz || exit 1

USER 1000:1000
CMD ["python", "-m", "src.server"]
```

The properties that matter:

- **Base image is pinned to a specific tag.** `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, not `nvidia/cuda:latest`. `latest` breaks reproducibility on the next re-pull.
- **`requirements.lock` is pinned, not `requirements.txt` with ranges.** Use `pip-tools` (`pip-compile`) or `uv pip compile` to lock; commit the lock file.
- **The artifact is copied into the image, not downloaded at start.** Runtime downloads add a startup dependency on the artifact store and break air-gapped deploys. Copy the versioned artifact directory (or install the versioned wheel) at build time.
- **`GIT_SHA` and `ARTIFACT_VERSION` are baked in as env vars.** The `/version` endpoint of the service reports both. Chapter 6's drift dashboard slices by both.
- **`HEALTHCHECK` runs one real forward.** Not just "does the process live" — "does the model produce a valid output on a canary input." Chapter 7's canary comparison depends on this.
- **Non-root user.** Every production container. Baseline security hygiene.
- **Explicit `EXPOSE` and no `--net=host`.** The container is a black box the orchestrator schedules; do not depend on host networking.

## Model registries: MLflow and the alternatives

A registry is a versioned, queryable store of model artifacts and their metadata. The three properties it must provide:

- **Immutable versioning.** A registered model version, once created, does not change. Rollback goes back to a specific version.
- **Metadata.** Training run link (params, metrics, code SHA, data hash), eval numbers, tags, staged/production stage transitions.
- **Programmatic access.** `mlflow.pyfunc.load_model("models:/intent-ner/Production")` in a container fetches the current production version deterministically.

### MLflow Model Registry

MLflow is the most-used open registry. Concepts: **Runs** (a single training execution with params, metrics, artifacts), **Models** (a named registered model), **Model Versions** (immutable versioned snapshots of a model), **Stages** (`None`, `Staging`, `Production`, `Archived` — soft-deprecated in newer versions in favour of aliases and tags).

```python
import mlflow
from mlflow.models.signature import infer_signature

with mlflow.start_run() as run:
    mlflow.log_params(hparams)
    mlflow.log_metrics(eval_scores)
    signature = infer_signature(X_sample, model.predict(X_sample))
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=IntentNERWrapper(model, tokenizer),
        signature=signature,
        registered_model_name="intent-ner",
        pip_requirements="requirements.lock",
    )
```

At serve time:

```python
model = mlflow.pyfunc.load_model("models:/intent-ner@production")
```

The alias (`@production`) is decoupled from the version number — promote a version, all consumers pick it up on next load. Rollback is one alias update.

### Alternatives

- **spaCy's own package + pip index.** `spacy package` produces installable wheels; a private PyPI index (Artifactory, CodeArtifact, self-hosted `pypiserver`) is a valid registry for spaCy-shaped artifacts. Simpler than MLflow if you are all-spaCy.
- **Hugging Face Hub or private mirror.** `push_to_hub(...)` + Hub commit SHAs give you the same immutability and metadata story with a UI. Standard for HF-heavy teams.
- **Object storage + a metadata table.** S3 (or GCS / Azure Blob) with a Postgres or DynamoDB table indexing `(name, version) → (s3_path, metadata_json)`. Rolls your own MLflow with less overhead; the right choice when you already have the ops for it.
- **W&B Artifacts, Neptune, DagsHub, ClearML.** Comparable to MLflow with different UX. Pick based on what the rest of the org uses.

The pick matters less than the discipline. Registration must be **required by CI** — no artifact reaches production that did not enter through the registry.

## Semantic versioning for NLP artifacts

Pick a version scheme and enforce it. A working default derived from SemVer:

- **Major (`X.0.0`).** Breaking output schema change (e.g. new entity types, changed span-offset semantics, dropped labels), or a base-model swap that changes numeric outputs beyond tolerance. Downstream consumers *will* need to update.
- **Minor (`0.X.0`).** New capability that is backwards-compatible on the schema (added a new optional field, added a new language, added a new component). Consumers benefit but do not have to change.
- **Patch (`0.0.X`).** Retraining on refreshed data with the same architecture and schema; bug fixes; quantisation changes. Should not change output *shape*; may change output *values* within a stated tolerance.

Tag the artifact and the container image with the same version. Do not ship "the container is `v0.2.0` but the artifact inside it is `v0.1.3`" — this is a debugging nightmare when it goes wrong.

## The artifact manifest

Every artifact ships with a machine-readable manifest that anchors reproducibility. A minimum-viable manifest:

```json
{
  "name": "intent-ner",
  "version": "0.2.0",
  "created_at": "2025-04-18T14:22:03Z",
  "git": {
    "repo": "org/nlp-pipelines",
    "sha": "9f8a1c2b...",
    "branch": "main"
  },
  "training": {
    "framework": "spacy",
    "framework_version": "3.7.4",
    "base_model": "en_core_web_lg",
    "config_sha": "b0d1...",
    "data": {
      "train_sha": "8e6d...",
      "dev_sha": "3a1c...",
      "snapshot_date": "2025-04-15"
    },
    "hyperparameters": { "learning_rate": 3e-5, "batch_size": 32, "epochs": 20 }
  },
  "runtime": {
    "format": "onnx",
    "opset": 17,
    "precision": "int8",
    "calibration_set_sha": "cc4a...",
    "target_hardware": "cpu-avx512-vnni"
  },
  "eval": {
    "suite_version": "eval-v3.1",
    "primary_metric": {"name": "seqeval-strict-f1", "value": 0.874},
    "per_slice": {
      "us-english": 0.891, "uk-english": 0.876, "in-english": 0.842
    },
    "confidence_interval": {"low": 0.868, "high": 0.881}
  },
  "schema_version": "1.2.0",
  "license": "Apache-2.0",
  "contact": "ml-platform@example.com"
}
```

Every field on that manifest earns its place because someone has needed it in a post-incident retro at some point. `training.data.snapshot_date` lets you correlate against upstream data-source changes. `runtime.calibration_set_sha` lets you tell whether a quality regression is a calibration problem. `eval.per_slice` lets you diff against the previous version and see which slices degraded.

Ship this file inside the artifact directory (e.g. `manifest.json`) *and* register it as structured metadata in your registry.

## CI: what to gate on

The CI job that trains and packages a new artifact should fail (not warn, fail) on any of:

- Eval numbers below a stated floor (per-slice, not just aggregate) — regression gate.
- Manifest missing any required field — schema gate.
- Numeric parity between the PyTorch and exported (ONNX / TensorRT) versions outside tolerance — export gate.
- Latency above the SLA target on a canary benchmark — perf gate.
- Container fails to boot with a valid response to the healthcheck — smoke gate.

A working pipeline: PR opens → CI trains → CI evals → CI builds container → CI runs canary → CI registers artifact (as `staging`) → human review → promote to `production` alias. Chapter 7's rollback story depends on this pipeline having produced a working previous version to fall back to.

## Chapter summary

- A reproducible NLP artifact is one an engineer can re-run inference on, re-run the build for, trace every input of, and bind to a decision — via a stable identifier. Every artifact needs a DAG, a container, and a registry.
- spaCy Projects (`project.yml`) is the canonical DAG shape for spaCy pipelines: declared assets with checksums, commands with `deps`/`outputs` for Make-style caching, workflows for order, remotes for team sharing. `spacy package` produces an installable versioned wheel.
- Hugging Face pipelines ship via `save_pretrained`/`push_to_hub`; every artifact needs a model card, a fast tokenizer bundled in, and safetensors weights. Runtime-exported artifacts add a `signature.json` capturing export config.
- The container is separate from the artifact. Pin base image tag, use a lock file, copy the artifact in at build time, bake `GIT_SHA` and `ARTIFACT_VERSION` as env vars, healthcheck a real forward, non-root user.
- A model registry (MLflow, HF Hub, private wheels, or S3+metadata table) enforces immutable versioning, metadata, and programmatic loading via aliases (`models:/intent-ner@production`). Registration must be a CI gate.
- SemVer maps to NLP artifacts as major = schema break, minor = additive capability, patch = same-shape retrain / bug fix. Container tag and artifact version must match.
- Every artifact ships with a machine-readable manifest capturing training config, data hashes, runtime signature, eval numbers per slice, and CIs. This is what an on-call reads in an incident.
- CI gates on regression (per-slice), schema, numeric parity, latency, and boot smoke. Chapter 6 monitors the running artifact; chapter 7 postmortems what monitoring caught.
