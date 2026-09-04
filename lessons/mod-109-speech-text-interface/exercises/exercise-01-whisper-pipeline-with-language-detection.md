# exercise-01: Whisper Pipeline With Language Detection

**Estimated effort:** 2 hours

## Objective

Build an end-to-end batch ASR pipeline that ingests a directory of mixed-language audio files, runs language identification, routes to the appropriate Whisper checkpoint / language flag, and emits transcripts conforming to the chapter 02 JSON schema — with per-segment `language` tags, word-level times from WhisperX-style forced alignment, and a small routing table you own.

This exercise is the *shape* every subsequent module exercise composes with: normalisation, punctuation, and diarisation post-processing all consume the JSON you produce here.

## Prerequisites

- Chapters [01](../01-speech-text-interface-landscape.md), [02](../02-asr-outputs-whisper-and-wav2vec.md), [03](../03-language-detection-and-mixed-language-routing.md), [04](../04-timestamp-alignment-and-forced-alignment.md).
- Python 3.10+; `faster-whisper` or `whisperx`, `torch`, `torchaudio`, `pyannote.audio` (optional, for a stretch task), `speechbrain` (for the alternative LID model), `pydantic` or `jsonschema` for schema validation.
- A GPU is strongly recommended. `faster-whisper` int8 will run on CPU but takes ~10× longer.

## Data

You will assemble a small mixed-language evaluation set from public corpora. Do not synthesise — use real audio.

- **Recommended.** ~30 files, 20–60 seconds each, spanning at least four languages (e.g. English, Spanish, French, Mandarin). Pull from:
  - Mozilla Common Voice ([Ardila et al., LREC 2020](https://arxiv.org/abs/1912.06670)) via the `common_voice_16_1` dataset on Hugging Face, filtered to short clips per locale.
  - FLEURS ([Conneau et al., 2022](https://arxiv.org/abs/2205.12446)) via `google/fleurs`, which is designed for LID evaluation.
- **Include at least three adversarial files:**
  - A file whose first 5 seconds are music / silence, then speech in a specific language.
  - A file with sequential code-switching (start in one language, switch to another mid-file — you can concatenate two short Common Voice clips).
  - A short file (< 3 s) to stress LID confidence.

Save the audio and a manifest CSV with columns `filepath,expected_language,notes`.

## Problem statement

### Part A — Baseline LID with Whisper

Load `faster-whisper`'s `large-v3` (or `openai/whisper` `large-v3`). For each file:

1. Read the audio, resample to 16 kHz mono if needed.
2. Call `detect_language` (or equivalent — see chapter 03) to get the top-1 language and its probability.
3. Also record the top-2 probability so you can compute the top-1-vs-top-2 margin.

Emit a `lid_baseline.csv` with columns `filepath, expected, whisper_top1, whisper_top1_prob, whisper_top2, whisper_top2_prob, correct`.

Compute overall accuracy and per-language accuracy against `expected_language`.

### Part B — Cross-check LID with SpeechBrain VoxLingua107

Do the same with [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) (chapter 03). Emit `lid_speechbrain.csv` with the same columns. Note that VoxLingua107 uses its own language-code alphabet — build a small mapping so you compare against the same normalized code.

Produce an agreement table: rows = files, columns = `whisper`, `speechbrain`, `expected`, `agree_ww_sb`. Report the number of disagreements and inspect them by hand; write a paragraph in `report.md` classifying each disagreement (which model was right; why the wrong one was wrong).

### Part C — Calibrated routing decision

Write a routing function that takes the two LID predictions and returns one of `{"supported_high_conf", "supported_low_conf", "fallback_multilingual", "reject"}` per the table in chapter 03. Your policy must:

- Require Whisper top-1 probability > 0.85 *and* `(top1 - top2) > 0.15` for `supported_high_conf`.
- Route to `supported_low_conf` when both LIDs agree but confidence is below the threshold.
- Route to `fallback_multilingual` when LIDs disagree.
- Route to `reject` on files < 1.5 s or where both LIDs return low-confidence unrelated results.

Emit `routing.csv` per file. Report the counts per bucket and one sentence per bucket about what you would do with it in production.

### Part D — Whisper transcription with per-segment language

Transcribe every non-rejected file with `faster-whisper`:

- `supported_high_conf`: force `language=<detected>` and use the appropriate size (e.g. `medium.en` for high-confidence English, `large-v3` otherwise). Justify your size mapping in `report.md`.
- `supported_low_conf` / `fallback_multilingual`: `language=None` (let Whisper LID per 30 s window; enables sequential code-switching support).

For each segment in Whisper's output, record its detected language (from `faster-whisper`'s `info.language` or the per-segment field). Preserve `avg_logprob`, `no_speech_prob`, and `compression_ratio` — you will use them for hallucination filtering in acceptance criteria.

### Part E — Word-level alignment

Run word-level forced alignment on the top-N supported files. Use whichever is easier for you:

- `faster-whisper` with `word_timestamps=True` (fastest; uses cross-attention post-processing — chapter 02).
- `whisperx.align` with the appropriate per-language wav2vec 2.0 alignment model (higher accuracy; chapter 04).

If a segment's language has no alignment model available, leave the `words` field out of that segment rather than fabricating.

### Part F — Emit schema-conforming JSON

Assemble the output into per-file `.json` files matching the chapter 02 schema:

- Top-level `language` (majority language), `duration`, `model`, `pipeline_version`.
- `segments[i]` with per-segment `language` (BCP-47 tag; see part G below), `start`, `end`, `text`, `avg_logprob`, `no_speech_prob`, `compression_ratio`.
- `segments[i].words[j]` when alignment succeeded.

Validate every file against a JSON Schema (Draft 2020-12) drawn from chapter 02's example. Reject and log any file that fails validation.

### Part G — ISO-639-1 → BCP-47 mapping

Whisper emits ISO-639-1 codes (`en`, `es`, `zh`). Your JSON contract says BCP-47 (`en-US`, `es-MX`, `zh-Hans`). Write a small `language_map.py` with a documented default per language (e.g. `"en" → "en-US"`, `"pt" → "pt-BR"`) and apply it before writing JSON. Document the assumptions in `report.md`.

### Part H — Report

`report.md` (500–800 words) covering:

- LID overall + per-language accuracy for both models (Parts A / B) with a short table.
- Disagreement analysis (5–10 sentences with 2–3 concrete file-level examples).
- Routing bucket counts and rationale (Part C).
- Choice of Whisper size mapping (Part D) and any hallucination-filtering thresholds you applied.
- One-paragraph write-up of an adversarial file (music intro, code-switch, or short clip) and how the pipeline handled it.
- Two sentences on where the pipeline would still fail and what you would do next.

## Starter guidance

- **Prefer `faster-whisper` over `openai/whisper`.** Faster, better memory, native word-level timestamps. `WhisperModel.transcribe(..., word_timestamps=True, vad_filter=True)` covers most of what you need.
- **Do VAD-trim before LID.** A five-second music intro is a common LID failure (chapter 03). If you skip VAD, mention it in the disagreement analysis when it inevitably bites.
- **Speak-brain LID is CPU-friendly.** You can run part B on a laptop while Whisper runs on GPU. Do the LID work first, in bulk; it informs the routing before you transcribe.
- **`faster-whisper` per-window LID.** When `language=None`, `faster-whisper` re-runs LID per window; check the `Segment.language` (if available in your version) or reason from `info.all_language_probs`.
- **Do not fabricate words for un-aligned segments.** If alignment fails or the language has no model, drop the `words` field for that segment. Downstream reads "no words" as "not available", not as "no speech."
- **Keep the JSON schema in a separate file** (`schema.json`) and reference it from your validator. It is a chapter-02 artefact worth reifying.

## Acceptance criteria

- [ ] `data/manifest.csv` lists ≥ 20 audio files spanning ≥ 4 languages, including at least 3 adversarial cases.
- [ ] `lid_baseline.csv` and `lid_speechbrain.csv` report per-file LID and overall + per-language accuracy.
- [ ] `routing.csv` classifies every file into one of `{supported_high_conf, supported_low_conf, fallback_multilingual, reject}` using the calibrated policy from Part C.
- [ ] `pipeline.py` runs the full pipeline end-to-end on the manifest and writes one JSON transcript per non-rejected file.
- [ ] Every emitted JSON passes validation against `schema.json`.
- [ ] All per-segment `language` fields are BCP-47 tags (not raw two-letter codes).
- [ ] Word-level times are present for at least one file per language you have an alignment model for; absent (not fabricated) for the rest.
- [ ] `report.md` (500–800 words) covers LID accuracy, disagreements, routing decisions, size mapping, one adversarial-file walkthrough, and next-step failures.

## Stretch goals

- **Add diarisation.** Wire in `pyannote/speaker-diarization-3.1` on the longer files and merge speakers into the JSON per chapter 08. Requires a Hugging Face auth token and pyannote install.
- **Cross-lingual translation surface.** For a subset of non-English files, also decode with `<|translate|>` (task="translate" in `faster-whisper`) and emit a parallel `translation_en` field. Do not overwrite the same-language `text`.
- **Confidence calibration.** Fit a temperature-scaling parameter (Guo et al., ICML 2017) on a held-out subset to recalibrate Whisper LID probabilities. Report expected calibration error (ECE) before and after.
- **Text-side LID cross-check.** Run `fastText lid.176.bin` or CLD3 on the emitted transcript; log disagreement between acoustic LID and text LID as a QA signal.
- **Streaming variant.** Adapt the pipeline to consume a streaming chunk source (simulate with `torchaudio.io.StreamReader`) and emit partial JSON events. Compare per-file transcription latency (from audio-in to final segment emit) against the batch pipeline.

## Deliverables

Ship as a directory:

```
data/
  manifest.csv
  audio/           # or a small README pointing at the corpora
pipeline.py
language_map.py
schema.json
lid_baseline.csv
lid_speechbrain.csv
routing.csv
outputs/
  <file>.json     # one per non-rejected file
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
