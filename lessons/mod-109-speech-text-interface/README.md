# mod-109 · Speech / Text Interface for NLP Engineers

Where ASR meets NLP: Whisper and wav2vec 2.0 output formats, language detection and mixed-language routing, timestamp alignment, text normalisation for WER and training data preparation, inverse text normalisation for spoken → written form, streaming punctuation and truecasing, diarisation post-processing (RTTM, overlap, speaker turns), and the durable ownership boundary between the NLP engineer and the speech specialist.

**Estimated effort:** 8 hours

## Learning objectives

- Operate ASR systems (Whisper, wav2vec 2.0) at the text-side: format selection, language detection, timestamp alignment.
- Build text-normalisation and inverse-text-normalisation pipelines for ASR pre/post-processing (numbers, currency, abbreviations, casing).
- Implement punctuation restoration and capitalisation models for streamed transcripts.
- Post-process diarisation output (speaker turns, overlap handling) and route language-mixed audio correctly.
- Recognise where deep speech modelling stops and where to hand off to a speech specialist.

## Chapters

1. [The speech / text interface landscape](01-speech-text-interface-landscape.md)
2. [ASR outputs: Whisper and wav2vec 2.0](02-asr-outputs-whisper-and-wav2vec.md)
3. [Language detection and mixed-language routing](03-language-detection-and-mixed-language-routing.md)
4. [Timestamp alignment and forced alignment](04-timestamp-alignment-and-forced-alignment.md)
5. [Text normalisation for ASR](05-text-normalisation-for-asr.md)
6. [Inverse text normalisation](06-inverse-text-normalisation.md)
7. [Punctuation and capitalisation restoration](07-punctuation-and-capitalisation-restoration.md)
8. [Diarisation post-processing](08-diarisation-post-processing.md)
9. [When to escalate to a speech specialist](09-when-to-escalate-to-a-speech-specialist.md)

## Exercises

- [exercise-01 · Whisper pipeline with language detection](exercises/exercise-01-whisper-pipeline-with-language-detection.md)
- [exercise-02 · Text normalisation and ITN for ASR](exercises/exercise-02-text-normalisation-and-itn-for-asr.md)
- [exercise-03 · Punctuation restoration model](exercises/exercise-03-punctuation-restoration-model.md)

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.

## Also in this module

- `labs/` — long-form hands-on labs (added on a later authoring cycle).
- `quizzes/` — knowledge checks (added on a later authoring cycle).
- [`resources.md`](resources.md) — primary sources, standards, and further reading.
