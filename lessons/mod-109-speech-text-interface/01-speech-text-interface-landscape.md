# The Speech / Text Interface Landscape

## Motivation

You have been dropped into a room where someone points at a directory of `.wav` files and says "make the transcripts useful." The audio has already been captured. Someone else, probably a speech engineer or an ASR vendor, will produce the raw transcripts. Your job — as an NLP engineer — is the *text side* of that pipeline: choosing the ASR output format that downstream consumers can actually use, deciding what language each file is in, aligning timestamps to the words that matter, cleaning up the numbers and abbreviations, restoring punctuation and casing to a legible transcript, and slotting speaker labels in the right places.

This module owns that interface. It does not train the acoustic model. It does not tune the beam search. It does not chase 0.4 WER improvements on LibriSpeech test-other. Those are the *speech engineer's* problems, and chapter 09 is explicit about the handoff.

Why the split matters: speech modelling and NLP post-processing are two very different skills, with different tools, different papers, and different failure modes. A team that expects the ASR to "just do it all" ends up with transcripts that are technically low-WER but downstream-hostile — no punctuation, wrong casing, `9 1 1` instead of `911`, no speaker labels, and timestamps that drift because they were computed on Whisper's decoded segments instead of via forced alignment. That team then hires a speech PhD to fix a text problem. This module exists so you can fix the text problem without hiring the PhD.

## The pipeline shape you will see 90 % of the time

Almost every production speech-to-useful-text pipeline decomposes into the same stages. Some are optional; some are collapsed inside a hosted API. But the shape is the same:

```
audio in
   │
   ▼
[VAD / segmentation]        (Silero, WebRTC VAD, or Whisper's own chunker)
   │
   ▼
[language ID / routing]     (Whisper's LID, or a dedicated LID model — chapter 03)
   │
   ▼
[ASR]                       (Whisper, wav2vec 2.0, HuBERT, NeMo Parakeet — chapter 02)
   │
   ▼
[timestamp alignment]       (WhisperX / wav2vec CTC alignment, MFA — chapter 04)
   │
   ▼
[ITN + punctuation + case]  (spoken → written form, restore punctuation — chapters 06, 07)
   │
   ▼
[diarisation merge]         (RTTM + word timestamps → per-speaker turns — chapter 08)
   │
   ▼
useful text
```

The chapters map roughly one-to-one onto these stages. Chapter 05 covers the *inverse* transform (written → spoken) used at training time and for WER scoring; the pipeline above is the read direction.

## What "text-side" ownership actually means

The concrete deliverables the rest of the org will hold you accountable for:

- A **transcript format** that downstream consumers can consume without rewriting parsers — JSON with segments, word-level times, speaker labels, and language; or WebVTT / SRT if the consumer is a video player; or a two-column CSV if it is an analyst.
- A **normalisation contract**: numbers, currencies, dates, addresses, and abbreviations rendered in written form consistently, with a way to round-trip back to spoken form for downstream ASR training or WER scoring.
- A **punctuation and casing contract**: whether the transcript ends every utterance with punctuation, whether proper nouns are capitalised, whether it uses UK or US conventions.
- A **language and locale contract**: BCP-47 language tags on every segment, correct routing for language-mixed audio, sensible fallbacks for the "unsupported language" case.
- A **speaker labelling contract**: how many speakers you emit, how you handle overlap, how you resolve `SPEAKER_00`-style anonymous labels against a name registry if one exists.

None of these are "the ASR's job." All of them are yours.

## What is *not* in this module

Ownership goes the other way too. If someone asks you to do any of the following, the correct move is usually "hand off to a speech specialist" (chapter 09):

- Train or fine-tune an acoustic model from scratch (wav2vec pretraining, Whisper distillation, RNN-T training).
- Tune the beam-search decoder or CTC blank penalty for latency vs. accuracy on the acoustic side.
- Diagnose or fix WER regressions caused by acoustic-model behaviour (background noise robustness, reverberation, accents, low-bitrate telephony).
- Design or tune a keyword-spotting / wake-word model.
- Design a full custom diarisation model (embedding extractor + clustering). Post-processing an existing pipeline's output is yours; designing the pipeline is not.
- Handle real-time streaming ASR at the audio buffer level (chunking, endpointing, look-ahead policies).

You should recognise these problems well enough to route them, and you should know the vocabulary well enough to have a productive conversation with the speech team. You should not be the one solving them.

## The three model families you will interact with

You will spend most of your time on the output side of three families. Chapter 02 goes deep; here is the vocabulary.

- **Whisper** (Radford et al., ["Robust Speech Recognition via Large-Scale Weak Supervision"](https://arxiv.org/abs/2212.04356), 2022). Encoder-decoder transformer trained on ~680 k hours of multilingual, weakly supervised audio. Outputs *text tokens* directly, including language and task control tokens, timestamp tokens, and (in the multilingual checkpoints) a translation-to-English option. This is the default when you do not have a compelling reason to pick something else. Sizes: `tiny` (39 M) through `large-v3` and `large-v3-turbo`.
- **wav2vec 2.0 / HuBERT / WavLM** (Baevski et al., 2020; Hsu et al., 2021; Chen et al., 2021). Self-supervised acoustic encoders — no built-in decoder. Fine-tuned with a CTC head to emit character or subword tokens. Output is *phoneme- or character-level*, with tight monotonic alignment to the audio frames. This is what WhisperX uses under the hood for forced alignment (chapter 04), and what you fine-tune when you need a domain-specific ASR for a language Whisper cannot handle.
- **NeMo Parakeet / Canary and USM-family models** (NVIDIA, Google). Production-oriented Conformer-Transducer or attention-decoder models with strong throughput characteristics. Frequently the choice at scale when Whisper's decoder is too slow or the transcript quality on your domain is not sufficient. You will encounter them mostly as an API or model card; the interface concerns are the same.

You do not need to know their acoustic details. You need to know:

- What tokens they emit (subword, character, phoneme).
- Whether they emit timestamps natively (Whisper: yes but coarse; wav2vec CTC: yes at frame level; NeMo: yes via RNN-T alignment).
- Whether they do language ID internally (Whisper: yes for the multilingual checkpoints; wav2vec: no).
- What their output looks like in JSON and how "chunk", "segment", and "word" are defined for that model.

## The three axes that pin an ASR deployment

Before you touch a config file, pin three things:

| Axis | Options | What it decides |
|------|---------|-----------------|
| **Latency** | Batch (offline) · Near-real-time (< 5 s) · Real-time streaming (< 500 ms) | Whether you can use a decoder-heavy model like Whisper (batch-friendly, weak for streaming) or need an RNN-T / CTC streaming model. Whether VAD chunking is one-shot or continuous. |
| **Language coverage** | Single-language · Small multilingual set · "Anything" | Whether you need LID up front, whether you can specialise per language, and whether you can afford to fall back to a lower-quality generalist model like `whisper-large-v3` for the tail. |
| **Domain** | Broadcast / read speech · Meetings / conversational · Call-centre / telephony · Medical / legal / financial | Whether your normalisation and ITN rules can be built off a generic English recipe or need domain vocabulary (drug names, legal citations, ticker symbols). Whether diarisation matters. |

A "single-language, batch, broadcast" pipeline is almost solved by a Whisper checkpoint plus the reference English normaliser and no diarisation. A "multilingual, near-real-time, call-centre" pipeline needs LID, per-language ITN, streaming punctuation, and diarisation. The pipeline shape is the same; the levers you actually have to pull are radically different.

## Where the module lives inside the track

This module sits between the multilingual side (mod-107) and the data-engineering / evaluation modules (mod-110, mod-111):

- **mod-107 (Machine Translation & Multilingual NLP)** owns the multilingual encoders and locale handling. When you route a Whisper output through translation-to-English, you are inside mod-107, not this module. When you decide the language tag on a segment, you are inside this module. The BCP-47 chapter of mod-107 (chapter 09 there) is a load-bearing prerequisite; you should be comfortable with language tags before you start chapter 03.
- **mod-110 (NLP Data Engineering)** owns the corpus curation and Unicode / normalisation infrastructure at scale — you will use their tooling for large transcript batches.
- **mod-111 (NLP Evaluation)** owns generic evaluation methodology. WER, MER, and CER-flavoured metrics are covered briefly here (chapter 05) because they are inseparable from text normalisation, but the general framework lives in mod-111.
- **mod-112 (Production NLP Pipelines)** owns the serving stack. Streaming ASR post-processing (chapter 07) borrows from mod-112 for its buffering primitives.

## The failure modes to keep in your head

Because the boundary between speech and NLP is fuzzy, the same symptoms can come from either side. Learn to sort them, because misrouted bugs waste weeks:

- **"The transcript says '9 1 1' instead of '911'."** ITN failure. Yours. Chapter 06.
- **"The transcript is all one paragraph with no punctuation."** Punctuation-restoration failure. Yours. Chapter 07.
- **"The transcript randomly capitalises words."** Truecasing failure. Yours. Chapter 07. (Or: Whisper generated cased tokens directly and you are getting its raw output — decide which layer owns it.)
- **"The transcript is confidently in French but the audio is Spanish."** LID failure. Yours. Chapter 03.
- **"The timestamps are off by 3 seconds."** Chunking / alignment failure. Yours. Chapter 04.
- **"The transcript is missing words." / "It hallucinates content that was not spoken."** Acoustic-model failure or VAD-chunk-boundary hallucination. Speech engineer's, but you diagnose with the timestamped output (chapter 04) and route.
- **"Speaker A's turn overlaps speaker B's and both are labelled Speaker A."** Diarisation failure. Post-processing is yours (chapter 08); the underlying model is the speech engineer's.

You do not need to be able to *fix* every one of these, but you must be able to *classify* every one of them.

## Chapter summary

- This module owns the text-side of the ASR pipeline: format, LID, alignment, normalisation, ITN, punctuation, casing, and diarisation post-processing. It does not own the acoustic model.
- The canonical pipeline shape is: VAD → LID → ASR → alignment → ITN + punctuation + case → diarisation merge → useful text. Chapters 02–08 map onto those stages.
- Three model families dominate: Whisper (encoder-decoder, batch, multilingual, ships timestamps and LID), wav2vec 2.0 / HuBERT / WavLM (SSL acoustic encoders + CTC, tight alignment, needed for forced alignment), and NeMo Parakeet / USM (production throughput).
- Pin three axes before touching a config: latency (batch / NRT / streaming), language coverage (one / small set / anything), and domain (broadcast / conversational / telephony / vertical).
- Learn to classify failure modes across the speech / text boundary so you can route bugs correctly. Chapter 09 makes the handoff explicit.
