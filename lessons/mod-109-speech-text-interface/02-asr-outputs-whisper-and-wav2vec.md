# ASR Outputs: Whisper and wav2vec 2.0

## Motivation

Every choice you make downstream — the alignment strategy, the ITN grammar, the punctuation restoration hook, the JSON schema you hand to the frontend — depends on what the ASR actually returned. Not "a transcript" (that word does more work than any word should). A structured object with tokens, times, probabilities, language tags, and, for Whisper, an ambient prompt window. Getting the output-format decision right is the first real engineering decision in this module.

This chapter maps the two families you will spend the most time on — Whisper and wav2vec 2.0 — from their internal representations to the JSON / VTT / SRT surfaces you actually ship. It ends with a matrix so you can pick a format for a downstream consumer without going back to the model card.

## Whisper: what actually comes out

Whisper (Radford et al., ["Robust Speech Recognition via Large-Scale Weak Supervision"](https://arxiv.org/abs/2212.04356), 2022) is an encoder-decoder transformer. The decoder emits a stream of *tokens* that includes both text and *special tokens* controlling the task, language, and timestamps. Understanding this token stream is the difference between "I trust the JSON" and "I keep hitting weird edge cases."

The decoder-side vocabulary includes:

- **Language tokens** — one per language, of the form `<|en|>`, `<|es|>`, `<|zh|>`, ... These select the language the decoder should transcribe. On the multilingual checkpoints, the first token produced (or a forced-decoder-input token) is the language.
- **Task tokens** — `<|transcribe|>` (same-language) or `<|translate|>` (multilingual → English). Whisper is a translation model too; treat that as a feature you can *opt into*, not something you fall into by accident.
- **Timestamp tokens** — a dense grid of `<|0.00|>`, `<|0.02|>`, ..., `<|30.00|>` at 20 ms resolution over each 30 s window. Segment start / end timestamps are emitted around each utterance. These are *predicted*, not deterministic — chapter 04 covers the "Whisper timestamps drift" problem in detail.
- **`<|nospeech|>`** — the model's estimate that the current 30 s window contains no speech. Useful as a VAD signal but not a substitute for a proper VAD (chapter 04).
- **Text tokens** — SentencePiece byte-BPE tokens over ~52 k entries (the multilingual `large-v3` vocabulary is expanded to ~51 865 entries plus the specials).

The reference `openai/whisper` Python package returns something like this per audio file:

```json
{
  "text": "  This is the whole transcript, joined with spaces.",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 6.4,
      "text": " This is the first segment.",
      "tokens": [50364, 639, 307, 264, 700, 3172, 13, 50684],
      "temperature": 0.0,
      "avg_logprob": -0.184,
      "compression_ratio": 1.31,
      "no_speech_prob": 0.014
    },
    ...
  ]
}
```

Everything downstream depends on this schema. Things worth pinning:

- `text` is the concatenation of segment texts, sometimes with a leading space; do not treat it as authoritative for word-level use.
- `segments[i].start` and `end` are in seconds relative to the file, not the current 30 s window. `seek` is the sample offset of the window they were decoded from.
- `no_speech_prob` and `avg_logprob` are the two knobs the reference decoder uses to *drop* segments as hallucinated silence. Preserve them — they are the cheapest hallucination filter you get.
- Whisper's `text` typically arrives with punctuation and casing already restored. This is not a real punctuation-restoration model; it is a byproduct of the training data. Chapter 07 discusses when to trust it and when to override it.

The Hugging Face `transformers` implementation returns a similar object via `pipeline("automatic-speech-recognition")` with `return_timestamps=True` or `return_timestamps="word"`. The keys are slightly different — you get `chunks` instead of `segments` at the pipeline layer — but the underlying token stream is the same. See the [Whisper docs](https://huggingface.co/docs/transformers/en/model_doc/whisper) and the [`AutomaticSpeechRecognitionPipeline` reference](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) for the pipeline schema.

`faster-whisper` (SYSTRAN's CTranslate2 port) returns segment and word objects with `.start`, `.end`, `.text`, `.probability`, and (for word-level) `.word`. It is faster and it exposes word-level timestamps directly, which is why it is the workhorse implementation for most production Whisper deployments.

## Whisper checkpoint sizes and when to pick which

The public checkpoints (as of `whisper-large-v3` and `-v3-turbo`):

| Checkpoint | Parameters | English-only? | Typical use |
|------------|-----------|---------------|-------------|
| `tiny` / `tiny.en` | 39 M | Both variants | On-device demos, wake-word-adjacent tasks, low-quality bulk transcription. |
| `base` / `base.en` | 74 M | Both variants | Fast, mediocre. Sometimes the right call for real-time. |
| `small` / `small.en` | 244 M | Both variants | The "cheap default." Handles most clean-speech English well. |
| `medium` / `medium.en` | 769 M | Both variants | Quality jump over `small`; often the sweet spot for offline batch. |
| `large-v2` / `large-v3` | 1.55 B | Multilingual only | Best public quality; the multilingual default. `large-v3` has a larger 128-mel input than `-v2`. |
| `large-v3-turbo` | 809 M | Multilingual only | Distilled decoder over `large-v3` encoder; ~4× faster decode, small quality delta. Ships with `openai/whisper` from v20240930. |

The `*.en` variants are English-only *and* they drop the multilingual behaviour: no LID, no translation task, English-tuned tokeniser. Prefer them when you know the language is English, because they are meaningfully more accurate at the same size on English audio.

## wav2vec 2.0: what actually comes out

wav2vec 2.0 (Baevski et al., ["wav2vec 2.0"](https://arxiv.org/abs/2006.11477), 2020) is a self-supervised acoustic encoder — the decoder is *not* included. Fine-tuning attaches a CTC head that predicts character or phoneme tokens at each 20 ms frame. Concretely, a wav2vec 2.0 fine-tune returns:

- **A dense logits tensor** of shape `(T, V)` where `T` is the number of 20 ms frames (~50 per second) and `V` is the small vocabulary size — commonly 32 for English character-level fine-tunes (26 letters + apostrophe + space + `<pad>` + `<unk>` + `<s>` + `</s>`).
- **CTC-decoded text** after applying a greedy or beam-search decoder to the logits. The decoder collapses repeated frames and removes blanks.
- **Per-frame timestamps** — because each logit corresponds to a fixed 20 ms slice of audio, the alignment between decoded characters and audio frames is *tight and monotonic*. This is the property WhisperX exploits for forced alignment (chapter 04).

wav2vec 2.0 outputs *lowercase, unpunctuated text*. There is no LID, no translation, no timestamps-as-tokens, no `no_speech_prob`. Restoring punctuation and casing is your problem (chapter 07); combining it with a language model at decode time is the standard way to raise quality (see the pyctcdecode / KenLM section below).

The Hugging Face wrapper (`Wav2Vec2ForCTC` + `Wav2Vec2Processor` or `Wav2Vec2ProcessorWithLM`) returns something like:

```python
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

logits = model(input_values).logits
pred_ids = logits.argmax(dim=-1)
transcript = processor.batch_decode(pred_ids)
# → ["A MAN SAID TO THE UNIVERSE SIR I EXIST"]
```

To get word-level timestamps you can inspect the `pred_ids` frame-by-frame, run through `Wav2Vec2CTCTokenizer.decode` with `output_word_offsets=True`, and multiply by the 20 ms frame stride. Or delegate to `pyctcdecode` or WhisperX, which do this and add language-model rescoring.

Family variants you should recognise: **HuBERT** (Hsu et al., 2021), **WavLM** (Chen et al., 2021), **XLS-R** (Babu et al., 2022) — all self-supervised acoustic encoders with the same fine-tuning story. **Meta MMS** (Pratap et al., ["Scaling Speech Technology to 1,000+ Languages"](https://arxiv.org/abs/2305.13516), 2023) fine-tunes the same architecture across 1 100+ languages and is the go-to fallback for a language Whisper does not do well on.

## Whisper vs. wav2vec 2.0 at a glance

| Property | Whisper | wav2vec 2.0 (CTC fine-tune) |
|----------|---------|-----------------------------|
| Architecture | Encoder-decoder transformer | Encoder + CTC head |
| Tokens emitted | SentencePiece BPE text tokens + specials | Characters (typically) |
| Timestamp granularity | Segment-level (native); word-level via `return_timestamps="word"` (approximate) | Frame-level (exact 20 ms) |
| Casing / punctuation | Ships restored (mimicking training data) | Lowercase, unpunctuated |
| Language ID | Yes (multilingual checkpoints) | No |
| Translation to English | Yes (via `<|translate|>` task token) | No |
| VAD hint | `no_speech_prob` per segment | None; needs an external VAD |
| Best at | Multilingual, batch, "just make it a good transcript" | Forced alignment, single-language fine-tuning, streaming CTC |
| Weak at | Real-time streaming; word-level timestamps in isolation | Multilingual generalisation; casing / punctuation out of the box |

The complementary shape — Whisper for transcription, wav2vec 2.0 for forced alignment — is *exactly* how WhisperX (chapter 04) is built and is a good default for offline pipelines.

## Choosing an output format

The ASR gives you a structured object. You now have to pick what to hand downstream. There is no universal answer; there is a right answer per consumer.

### JSON with segments and words (recommended default)

Ship a JSON object with a fixed schema — file-level metadata (`language`, `duration`, `model`, `pipeline_version`), then a list of segments, each with `start`, `end`, `text`, `speaker`, and an optional `words` list carrying `word`, `start`, `end`, `probability`. This is what most modern pipelines ship: WhisperX writes it, `faster-whisper` writes it, Deepgram / AssemblyAI / Azure return it.

A concrete recommended shape:

```json
{
  "language": "en-US",
  "duration": 328.4,
  "model": "whisper-large-v3-turbo",
  "pipeline_version": "asr-post-v3.4",
  "segments": [
    {
      "id": 0,
      "speaker": "SPEAKER_00",
      "start": 0.24,
      "end": 4.10,
      "text": "Welcome back to the podcast.",
      "words": [
        {"word": "Welcome", "start": 0.24, "end": 0.72, "probability": 0.99},
        {"word": "back",    "start": 0.72, "end": 0.98, "probability": 0.98},
        ...
      ]
    },
    ...
  ]
}
```

Two things worth pinning as *contract*, not implementation detail:

- **Speakers are strings, not integers.** Diarisers return anonymous string IDs; you may later resolve them against a name registry. Do not paint yourself into an int corner.
- **`language` is a BCP-47 tag** (`en-US`, `zh-Hans`, `fr-CA`) — not an ISO 639-1 code (`en`, `zh`, `fr`). This is the contract used by browsers, iOS, Android, and every downstream Unicode-aware library. Chapter 09 of mod-107 covers this.

### WebVTT (subtitles for the web)

WebVTT ([W3C Recommendation](https://www.w3.org/TR/webvtt1/)) is what browsers consume in `<track kind="subtitles">`. It is line-oriented, cue-based, and has real support for speaker labels via `<v Speaker>` voice spans.

```
WEBVTT

00:00:00.240 --> 00:00:04.100
<v SPEAKER_00>Welcome back to the podcast.

00:00:04.100 --> 00:00:07.900
<v SPEAKER_01>Thanks for having me.
```

Ship WebVTT when the consumer is `<video>` or a JW Player / Video.js / Bitmovin embed. Do *not* try to jam word-level timestamps into WebVTT unless you are prepared to use its ugly per-word `<00:00:00.240>` inline timestamps — most players will render them fine but few analytics tools will parse them.

### SRT (SubRip)

Older subtitle format. No formal spec, so profile carefully. Widely supported, universally hated. Emit it when you must:

```
1
00:00:00,240 --> 00:00:04,100
Welcome back to the podcast.

2
00:00:04,100 --> 00:00:07,900
Thanks for having me.
```

Note: SRT uses `,` for the millisecond separator; WebVTT uses `.`. This trips people up.

### Plain text (last resort)

Punctuated, cased, one paragraph per speaker turn or one paragraph per some heuristic segment size. Ship it when the consumer is `grep`, an editor, or a document-oriented downstream NLP task (summarisation, classification) that will re-chunk anyway. Attach the JSON in parallel; do not force downstream to reconstruct times from plain text.

## The format-vs-consumer matrix

| Consumer | Recommended format | Why |
|----------|-------------------|-----|
| Downstream summarisation / classification / RAG | JSON (segments only, no need for words) or plain text | The consumer needs a document, not times; keep the JSON for `speaker` and `language` metadata. |
| Timestamped search / video seek | JSON (with `words`) | You will index words → times for jump-to-word playback. |
| Subtitles / captions in a browser | WebVTT | Native `<track>` support and `<v>` voice tags. |
| Subtitles in a legacy tool / burnt into video via `ffmpeg` | SRT | `ffmpeg -i in.mp4 -vf "subtitles=in.srt" out.mp4` still expects SRT. |
| Analyst inspection / spreadsheet | CSV with columns `start,end,speaker,text` | Excel-friendly; short. |
| Voice-of-customer analytics with sentiment / topics | JSON with segments + speaker | Analytics wants speaker turns and full text but rarely per-word. |
| Compliance / call recording archives | JSON *and* WebVTT | JSON for programmatic access, WebVTT for the compliance UI. |

## Chunking, batching, and long-form audio

Whisper is trained on 30 s windows. Anything longer is chunked. Three strategies are common; know which one the implementation you use is doing:

- **Sliding 30 s windows with condition-on-previous-text** — the reference `openai/whisper` behaviour. Each 30 s window is decoded with the previous window's decoded text prepended as decoder prompt. Prone to hallucination on silent / noisy chunks and to *runaway repetition* if the previous window is corrupt. The reference implementation guards against both with `no_speech_threshold`, `logprob_threshold`, and a temperature fallback (`temperature_increment_on_fallback`).
- **VAD-based chunking** (`faster-whisper` with `vad_filter=True`, WhisperX) — an external VAD (typically Silero) segments the audio into speech regions ≤ 30 s each, and Whisper decodes each region independently. Almost always more robust than sliding-window; recommended default for long-form batch.
- **Hugging Face pipeline chunking** (`chunk_length_s`, `stride_length_s`) — the HF `AutomaticSpeechRecognitionPipeline` chunks the audio into fixed-length segments with a configurable stride (overlap) and merges outputs. Simpler than VAD; usually inferior on real-world recordings.

Regardless of strategy, the JSON you ship must have `start` / `end` in seconds **relative to the original file**, not to the chunk. Every implementation gets this right for you; verify at least once on a > 30-minute file.

## Timestamps: honest limitations

Whisper's native timestamp tokens are predicted by the same decoder that predicts the text. They are calibrated to segment boundaries, not word boundaries, and empirically drift under conditions like fast speech, overlapping speakers, and long stretches of silence. WhisperX (Bain et al., ["WhisperX"](https://arxiv.org/abs/2303.00747), 2023) exists specifically because Whisper's raw timestamps are not accurate enough for subtitle-quality alignment. Chapter 04 goes deep; the takeaway here is:

- Do not promise your downstream consumer word-level timestamps *from Whisper alone*. Word-level `chunks` in the HF pipeline are approximate.
- If your consumer needs word-accurate times, insert a forced-alignment stage (WhisperX / wav2vec 2.0 CTC / MFA). Chapter 04.

## The output-format contract you will end up owning

If you author a single document at the start of the project, make it the transcript-JSON schema, and version it. A representative minimal schema:

```json
{
  "$id": "https://example.com/schemas/asr-transcript-v3.json",
  "type": "object",
  "required": ["language", "duration", "model", "pipeline_version", "segments"],
  "properties": {
    "language": {"type": "string", "description": "BCP-47 tag."},
    "duration": {"type": "number", "description": "Seconds."},
    "model": {"type": "string"},
    "pipeline_version": {"type": "string"},
    "segments": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "start", "end", "text"],
        "properties": {
          "id": {"type": "integer"},
          "speaker": {"type": ["string", "null"]},
          "start": {"type": "number"},
          "end": {"type": "number"},
          "text": {"type": "string"},
          "words": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["word", "start", "end"],
              "properties": {
                "word": {"type": "string"},
                "start": {"type": "number"},
                "end": {"type": "number"},
                "probability": {"type": "number"}
              }
            }
          }
        }
      }
    }
  }
}
```

Version the schema. When you add fields (confidence bands, sentiment scores, per-word language tags for code-switching), bump the version and record both in the file. Downstream consumers cache aggressively; a silently changing schema is how you break a dashboard six months from now.

## Chapter summary

- Whisper emits a token stream with text + language + task + timestamp specials; the standard JSON output has `text`, `language`, and per-segment `start`, `end`, `text`, `tokens`, `avg_logprob`, `compression_ratio`, `no_speech_prob`.
- wav2vec 2.0 (and HuBERT / WavLM / XLS-R) emits per-frame logits over a small character vocabulary; CTC-decoded text is lowercase, unpunctuated, and comes with *tight* frame-level alignment.
- Whisper is the multilingual, batch, "just make a good transcript" default; wav2vec 2.0 is the forced-alignment default. The `Whisper + wav2vec-alignment` combination is what WhisperX ships.
- Pick output format by consumer: JSON with segments and words is the recommended default; WebVTT for browsers; SRT for legacy tools; plain text as a last resort.
- Own the transcript-JSON schema and version it. `speaker` is a string; `language` is a BCP-47 tag.
- Long-form audio requires chunking. Prefer VAD-based chunking over sliding-window; verify that timestamps in your output JSON are file-relative, not chunk-relative.
- Whisper's native timestamps are segment-calibrated and drift for word-level use; insert forced alignment (chapter 04) when word-accurate times matter.
