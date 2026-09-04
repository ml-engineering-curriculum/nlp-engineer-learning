# Timestamp Alignment and Forced Alignment

## Motivation

Word-accurate timestamps are the difference between a transcript that a user can search and click-through and a transcript that only reads well. Every downstream feature that involves *jump to this word in the video*, *highlight this word as the speaker says it*, *cut this clip from this second to this second*, or *diarisation merge* — depends on times you can trust.

Whisper's native timestamps are not those times. This chapter explains why, and what to reach for instead.

## Whisper's native timestamps: what they actually mean

Chapter 02 covered the shape: Whisper emits `<|0.00|>`, `<|0.02|>`, ... tokens on a 20 ms grid around each utterance segment, and the standard output surfaces them as `segments[i].start` and `end`. Two properties surprise people:

- **They are *predicted* by the decoder**, not derived from the audio. The decoder learns during pre-training to place timestamp tokens at plausible utterance boundaries. It is doing pattern-matching, not measurement.
- **They are segment-calibrated.** The training signal for the timestamp tokens comes from the weak-supervision labels, which are usually sentence- or segment-level. Word-level timestamps that some implementations expose (via `return_timestamps="word"` in HF pipeline, or `word_timestamps=True` in `faster-whisper`) are derived by *heuristic post-processing* over the cross-attention weights, not by primary training signal.

Concretely, the errors you will observe on real audio:

- **Segment start / end typically off by 100–500 ms**, sometimes more on long silence or fast speech. Fine for subtitles when the display is 3–5 s per cue; visible as a lag when the display is per-word karaoke.
- **Word-level timestamps drift more.** The cross-attention post-processing produces monotonic times but not necessarily accurate ones. Common failure mode: consecutive short words share the same start time because the attention peaks were smeared.
- **Hallucinated segments** — Whisper occasionally emits text and timestamps for audio regions that were silent or contained no speech. `no_speech_prob` catches most of these; alignment against the actual audio catches the rest.
- **Timestamp drift accumulates on long files** when the model conditions on previous text (`condition_on_previous_text=True`). VAD-based chunking (`faster-whisper` with `vad_filter=True`, or WhisperX) usually eliminates this by resetting context between speech regions.

The Whisper paper (Radford et al., ["Robust Speech Recognition via Large-Scale Weak Supervision"](https://arxiv.org/abs/2212.04356), 2022) and the WhisperX paper (Bain, Huh, Han & Zisserman, ["WhisperX"](https://arxiv.org/abs/2303.00747), 2023) both discuss this; WhisperX explicitly exists to fix the word-timestamp problem.

## What "forced alignment" means

Forced alignment is the two-input problem: given (a) the audio and (b) the reference text, produce the start and end time of each word (or phoneme, or character) in the audio. It is *not* transcription — the text is assumed correct. That reframing is what makes it tractable: no vocabulary uncertainty, no language uncertainty, just time.

Two flavours are common in practice:

- **CTC-based alignment.** Feed the audio to a wav2vec 2.0 / HuBERT / WavLM CTC model; get frame-level character logits at 20 ms resolution; find the maximum-likelihood monotonic alignment between the reference text (character-tokenised through the model's vocabulary) and the frames. The alignment algorithm is essentially Viterbi over the CTC lattice — the same recipe used at CTC decode time, restricted to the given text. Tight, deterministic, well-defined for supported languages.
- **Traditional GMM/HMM forced alignment.** MFA (Montreal Forced Aligner — McAuliffe et al., ["Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi"](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html), 2017) trains a GMM/HMM on a pronunciation dictionary and aligns via Viterbi. Older, phoneme-level, works well when a good G2P dictionary exists (mostly high-resource languages). Slower than CTC alignment; more accurate for phonetics research.

For the text-side pipeline you will almost always want CTC alignment. It runs on GPU, is well-supported in Hugging Face, and produces the exact grain (word-level timestamps in the reference text's tokenisation) that downstream consumers want.

## WhisperX: the pattern to internalise

WhisperX composes Whisper for transcription with wav2vec 2.0 for alignment. The pipeline it implements is the pattern you should internalise even if you do not use the library:

```
audio
  │
  ▼
Silero VAD ─► speech regions ≤30 s
  │
  ▼
Whisper (batched) on each region  ─► segments with approximate times + text
  │
  ▼
wav2vec 2.0 CTC + Viterbi on each segment ─► word-accurate times for each word
  │
  ▼
(optional) pyannote diarisation
  │
  ▼
merge: attach speaker labels to words via time overlap
```

The alignment model must be *for the transcript's language*. WhisperX ships a small registry of per-language wav2vec 2.0 alignment models (`WAV2VEC2_ASR_BASE_960H` for English, `VOXPOPULI_ASR_BASE_10K_FR` for French, etc.). If the language is not covered, alignment falls back to Whisper's native timestamps and you should downgrade the promise you make to the consumer.

Concrete usage:

```python
import whisperx

device = "cuda"
audio = whisperx.load_audio("clip.wav")

model = whisperx.load_model("large-v3", device, compute_type="float16")
result = model.transcribe(audio, batch_size=16)
# result["segments"] carries approximate times + text + language

align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result_aligned = whisperx.align(
    result["segments"], align_model, metadata, audio, device,
    return_char_alignments=False,
)
# result_aligned["segments"][i]["words"] now has {"word", "start", "end", "score"}
```

The `score` is the CTC log-probability normalised over frames, not a Whisper decoder log-prob. Do not conflate the two: they are on different scales.

## When you cannot use WhisperX (and what to do instead)

WhisperX is a good default but not always the right tool. Reasons to build your own:

- **Language not covered by shipped alignment models.** Meta's MMS (Pratap et al., ["Scaling Speech Technology to 1,000+ Languages"](https://arxiv.org/abs/2305.13516), 2023) has wav2vec 2.0 CTC fine-tunes for 1 100+ languages and is the default fallback. `torchaudio.pipelines.MMS_FA` exposes a ready-to-use forced-alignment stack with the same interface pattern.
- **You need phoneme-level timestamps** (linguistics work, TTS training, prosody analysis). Use MFA.
- **You need to align to a locked reference transcript** — closed-captioning legal texts, dubbing scripts. In that case Whisper's transcription is not what you want at all; use MMS-FA or MFA directly.

`torchaudio` also provides the primitives (`torchaudio.functional.forced_align`, `torchaudio.pipelines.MMS_FA`) if you want to build a bespoke pipeline. The [torchaudio Forced Alignment tutorial](https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html) is worth reading through once.

## Word- vs segment- vs phoneme-level: choosing the grain

Different consumers need different grains. Pin the grain contract upstream so the alignment cost matches the payoff.

| Grain | Cost | Accuracy | When you need it |
|-------|------|----------|------------------|
| Segment (Whisper native) | Free | ±100–500 ms | Search over segment text; long-form subtitles at 3–5 s cadence. |
| Word (WhisperX / MMS-FA) | +1× ASR compute | ±30–80 ms typical | Click-to-play word highlighting; karaoke display; diarisation merge; word-level search. |
| Phoneme (MFA / MMS-FA phoneme mode) | +2–3× ASR compute | ±20–40 ms typical | TTS training data; lyrics-to-song alignment; linguistics; lip-sync verification. |

Do not upgrade the grain for the whole file when only one downstream needs it. It is common to ship the JSON with segment-level times by default and re-align to word- or phoneme-level on demand for the small fraction of files that get clicked through.

## Timestamps and long-form audio

The single most common alignment bug in production: timestamps computed on a chunk that was decoded independently, expressed relative to the chunk, and never rebased to the file. If your file is one hour long and your chunk is 30 s, times end up in `[0, 30)` for every word and everything falls over.

The fix is boring but load-bearing: whenever you chunk, remember the chunk offset in the file, and shift every emitted time by that offset before you write JSON. `faster-whisper` and WhisperX get this right for you. Custom pipelines built around HF's chunk-and-batch trick sometimes do not — test on at least one file longer than 30 minutes and print times of the last few words.

Related trap: VAD-chunked pipelines *drop* the silences between speech regions. If you compute `duration` as the sum of segment durations you will be under-reporting by exactly the total silence. Compute `duration` from the audio file's own header, not from the transcript.

## Overlapping speech

Forced alignment assumes one speaker per span. When two speakers overlap, CTC alignment silently attributes the overlap to whichever speaker the acoustic likelihood favours; the other speaker's words in that overlap get compressed or dropped in the aligned times.

Two mitigations:

- **Post-hoc via diarisation overlap output.** Chapter 08 covers pyannote's overlap-aware segmentation. When the diariser marks a region as overlapped, treat any word timestamps inside it as approximate and mark them accordingly in the JSON (`overlap: true`).
- **True target-speaker ASR.** Recent work (e.g. Kanda et al., ["Serialized Output Training for End-to-End Overlapped Speech Recognition"](https://arxiv.org/abs/2003.12687), 2020, and follow-ups) trains ASR to transcribe multiple speakers in one pass. This is squarely in speech-specialist territory (chapter 09).

## Alignment quality: how to know it is working

Alignment errors are boring to eyeball. Set up numerical checks and run them in CI on a fixed test set. Suggested minimum:

- **Word Alignment Error Rate (WAER)** — for a set of files with manual word-level timestamps, average absolute error in word start times. `< 100 ms` is achievable with WhisperX on clean speech; `< 200 ms` on real-world conversational audio.
- **Overlap fraction** — for a set of consecutive words, count how often `word[i].end > word[i+1].start` (they overlap). Should be near zero; non-zero suggests the alignment collapsed on that segment.
- **Duration histogram** — histogram of word durations. Real speech is roughly log-normal with a mode near 200 ms and few words below 50 ms. A spike at exactly the frame duration (20 ms) means the aligner defaulted to the frame stride for words it could not localise.
- **Boundary silence check** — proportion of segments whose start / end lands inside silence longer than the frame stride. If most starts / ends are inside speech you have late / early boundary artefacts.

Pin the baseline; alert on regressions. Alignment quality decays silently otherwise.

## Persisting alignment in the JSON

Segment and word records in the transcript JSON already accommodate this (chapter 02). Two conventions worth adopting:

- **`words` are optional per segment.** If alignment succeeded, the field is present and populated. If it failed for that segment (short segment, hallucination, language not supported), leave the field out rather than emitting fabricated word times.
- **Record the aligner in the file metadata.** `alignment: {"model": "WAV2VEC2_ASR_BASE_960H", "language": "en"}` at the file level. This is the single most useful field when triaging a downstream "why are these times wrong?" ticket six months later.

## Chapter summary

- Whisper's native timestamps are decoder-predicted, segment-calibrated, and drift for word-level use. Word-level chunks are derived from cross-attention post-processing, not primary training signal.
- Forced alignment reformulates the problem: given audio *and* text, produce times. CTC-based alignment on wav2vec 2.0 / HuBERT is the standard tool.
- WhisperX composes Whisper transcription with wav2vec 2.0 alignment; internalise the pipeline shape even if you do not use the library. Alignment model must match the transcript's language.
- Fallbacks: Meta MMS for the 1 100+ language tail, MFA for phoneme-level or reference-transcript alignment.
- Pick the alignment grain (segment / word / phoneme) per consumer; do not upgrade the whole pipeline for one downstream.
- Rebase all times to file-relative on chunking; compute `duration` from the audio header, not the transcript.
- Overlap is a first-class problem: mark overlap regions from the diariser and downgrade the accuracy promise for word times inside them.
- Measure alignment quality with WAER, word-overlap fraction, duration histograms, and boundary-silence checks. Alignment decays silently.
