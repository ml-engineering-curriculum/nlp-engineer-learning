# Language Detection and Mixed-Language Routing

## Motivation

Language identification is the first decision the pipeline has to make and the first one it can be wrong about silently. If you route a French clip to an English-only model you get gibberish with high acoustic confidence; if you route it to the wrong locale-specific ITN grammar you get correct words with the wrong currency symbol; if you assume "one file = one language" and the speaker code-switches, half the transcript ends up unusable.

This chapter is about deciding *what language every span of audio is in* — cheaply, with calibrated confidence, and in a way that survives real-world audio where two languages coexist in the same call.

## What "language detection" actually names

Three related tasks get mashed together under LID. Keep them separate:

- **File-level LID.** One file, one predicted language tag. Fine for podcasts, audiobooks, most broadcast content. Wrong for meetings and calls.
- **Segment-level LID.** A language tag per VAD segment or per fixed window (typically 30 s to match Whisper's window). Handles speaker-per-language conversations well.
- **Word-level LID (code-switching detection).** A language tag per word or short span. Needed for `"Let's grab coffee, ¿te parece?"` sentences. Much harder, small research area; production systems usually approximate with segment-level LID plus a per-segment ASR call.

Whisper's built-in LID is *file-* or *window-level*. It is a good default for the first two tasks and useless for the third. Chapter 09 of mod-107 introduces BCP-47 tags; this chapter assumes you are comfortable writing `en-US` or `zh-Hans` rather than `en` or `chinese`.

## Whisper's built-in LID

Whisper's multilingual checkpoints (everything except the `*.en` variants) predict a language token as the first decoded token after the `<|startoftranscript|>` special. The `detect_language` helper in `openai/whisper` runs the encoder on the first 30 s window, feeds only the `<|startoftranscript|>` token to the decoder, and returns the log-probability distribution over the ~100 language tokens.

The reference call:

```python
import whisper

model = whisper.load_model("large-v3")
audio = whisper.load_audio("clip.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

_, probs = model.detect_language(mel)
lang = max(probs, key=probs.get)
print(lang, probs[lang])   # 'en' 0.9823
```

Important properties to internalise:

- **It only sees the first 30 s.** If those 30 s are silence, music, or a jingle in a different language than the body, LID is wrong. Pre-VAD trimming helps; a VAD pass first almost always improves LID quality for real-world files.
- **The output is an `iso-639-1` code**, not a BCP-47 tag. Map it (`"en" → "en-US"` or `"en-GB"` per your default policy). Never emit the raw two-letter code to a downstream consumer that expects BCP-47.
- **There is no `<|unknown|>` token.** If the language is not in the ~100 Whisper knows, it will pick the closest neighbour with high confidence.
- **Confidence is not calibrated.** The winning probability can be `0.99` on a language where the acoustic evidence is genuinely ambiguous. Chapter 04 discusses uncertainty-based fallbacks; the short version is that `no_speech_prob` and `avg_logprob` correlate with LID reliability better than the LID probability itself does.
- **`transcribe(..., language=None)` runs LID internally per 30 s window** in the current `openai/whisper` and `faster-whisper` implementations. Each window can, in principle, be tagged with a different language. This is *not* real code-switching detection — the decoder is still forced to one language per window — but it is enough to route each segment to the right downstream ITN and punctuation model.

The reference LID benchmark in the Whisper paper is FLEURS (Conneau et al., ["FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech"](https://arxiv.org/abs/2205.12446), 2022). Table 8 in the Whisper paper reports LID accuracy per checkpoint size on FLEURS; `large-v3` is above 95 % top-1 on the majority-well-represented languages and considerably lower on the long tail.

## Alternatives to Whisper LID

You will reach for a non-Whisper LID when Whisper is not already running (LID as a *routing* step *before* choosing an ASR), or when Whisper's LID is not accurate enough for your language set.

- **SpeechBrain `spkrec-langid-voxlingua107`** ([model card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)) — an ECAPA-TDNN speaker/language embedding classifier trained on VoxLingua107 (Valk & Alumäe, ["VoxLingua107: a Dataset for Spoken Language Recognition"](https://arxiv.org/abs/2011.12998), 2020). ~107 languages; small, fast, runs on CPU. Good default for a *pre-ASR* routing hop.
- **NVIDIA NeMo `TitaNet-LID`** — production-oriented, similar architecture.
- **Silero LID** (`snakers4/silero-lang`) — small, fast, fewer languages; sometimes the right call in low-resource settings.
- **fastText `lid.176.bin`** ([Facebook fastText](https://fasttext.cc/docs/en/language-identification.html)) and **CLD3** — these are *text* LID models. Use them on the ASR output, not the audio. Useful for a cross-check ("Whisper said Portuguese, the text LID says Spanish — inspect this file"), not for routing.

The pattern most production pipelines converge on:

```
audio ─► VAD ─► segment-level acoustic LID (SpeechBrain / TitaNet)
                        │
                        ▼
        per-segment ASR call with `language=<detected>`
                        │
                        ▼
        text LID on the emitted transcript as a sanity check
```

The acoustic LID makes the routing decision; the text LID catches disagreements. Files where they disagree by more than a small margin get flagged for review.

## Calibrating LID confidence

Raw softmax probabilities from any of these models are wildly overconfident. You need a threshold policy, and you need it derived from data.

The workflow, once:

1. Take ~500–2 000 held-out files spanning your target languages *and* your expected out-of-scope languages (unknown, silent, music).
2. Run the LID model. Record top-1 language, top-1 probability, top-2 probability, and the ground-truth tag.
3. Bin the top-1 probability into deciles. In each bin, compute accuracy: fraction of files where top-1 equals ground truth.
4. The threshold you want is the smallest probability at which accuracy in that bin exceeds your requirement (95 %, 99 %, whatever the routing tolerates).
5. Anything below the threshold routes to a *fallback path* — a general-purpose Whisper checkpoint, a human-in-the-loop queue, or a "reject with reason: low LID confidence" response.

Also worth measuring: the *margin* between top-1 and top-2 probabilities. Low margin is a stronger uncertainty signal than low top-1 alone. In practice, a policy of `top1 > 0.85 AND (top1 - top2) > 0.15` catches most silent-window and music-window mis-LIDs without hurting accuracy on legitimate speech.

Standard temperature-scaling / calibration references (Guo, Pleiss, Sun & Weinberger, ["On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599), 2017) apply here as they do everywhere.

## Handling code-switched audio

Real code-switching splits into two rough regimes.

**Sequential code-switching** — speaker A speaks language X for 30 seconds, then language Y for 30 seconds; or the interviewer is in English and the interviewee is in Spanish. This is what Whisper's per-30 s LID handles competently. Combined with diarisation (chapter 08), you can even attach a language tag per speaker.

Pipeline:

```
audio ─► VAD ─► 30s-ish windows
        │
        ▼
        per-window LID + per-window ASR (language forced from LID)
        │
        ▼
        segments carry `language` field individually
```

Every segment in the output JSON has its own `language` field. Downstream ITN, punctuation, and casing steps route per-segment. `resources.md` links VoxPopuli which has real multilingual sequential switching for evaluation.

**Intra-sentential code-switching** — `"Yesterday I went to the mercado to buy some carne."` This is genuinely hard. Whisper decodes the whole sentence in one language, and the words in the other language come out either mis-recognised or (helpfully) hallucinated into orthography of the decoded language. There is no clean text-side fix.

The pragmatic options, in decreasing order of quality:

1. **Use an ASR that supports mixed decoding natively.** Some domain models (particularly Indian-subcontinent Hindi-English and North-American Spanish-English) exist as pre-mixed acoustic models. This is a speech-engineer decision (chapter 09) — if the customer needs this, escalate.
2. **Post-hoc word-level LID + selective re-transcription.** Run a text LID over the transcript at word/short-span level. For spans flagged as a different language, re-run the audio window around that span with the other language forced. Expensive; noisy; occasionally correct.
3. **Accept the loss and document it.** Flag files as `mixed=true` in metadata and set expectations with the consumer.

Do not oversell a code-switching capability you do not have. The failure mode where you claim mixed-language support and silently mis-transcribe is worse than the failure mode where you decline the file.

## Routing decisions the pipeline actually makes

Given LID output, the pipeline has a small set of routes to pick between. Make the mapping explicit and version it.

| LID output | Route |
|------------|-------|
| Supported language, high confidence | Language-specific ASR (or forced-language Whisper) + language-specific ITN + language-specific punctuation |
| Supported language, low confidence | Multilingual Whisper `large-v3` with `language=None` (let the model decide per window) + generic normalisation + multilingual punctuation |
| Out-of-scope but recognisable language | Multilingual Whisper as above, mark file `unsupported_language=true`, downstream may reject |
| Unrecognisable / silence / music | Reject with reason; do not run the ASR |
| Mixed (multiple high-confidence languages detected) | Per-segment routing; mark file `mixed=true` |

The versioned routing table lives in code, not in a data scientist's head. When it changes you bump the pipeline version tag in the JSON (chapter 02) so downstream can invalidate caches.

## Common failure modes

- **Music intro fools LID.** The 30 s intro is instrumental with English lyrics or a French station ID. LID picks English or French; the actual talk is Mandarin. Fix: VAD-trim before running LID; alternatively, run LID on the first N speech-only windows and take a majority vote.
- **Long silence in the first 30 s.** LID confidence collapses. Fix: same — VAD first.
- **Very short files (< 3 s).** LID accuracy degrades sharply. Fix: raise the low-confidence threshold; consider routing all short files to the multilingual generalist.
- **Similar language families are confused.** Norwegian / Danish / Swedish; Portuguese / Galician / Spanish; Serbian / Croatian. The predicted language is often the "wrong right answer" — a related language with slightly different orthography. Fix: cluster related languages in your routing table so a mis-LID inside the cluster still routes to a competent ASR.
- **Region drift.** Whisper predicts `zh` (Mandarin) confidently on a Cantonese clip. LID at the ISO-639-1 level cannot tell them apart. Fix: use a LID model with language *and region* labels for the languages where this matters (Cantonese / Mandarin, Egyptian / MSA, Latin-American / Iberian Spanish).
- **Silent LID drift on window transitions.** The first 30 s is English; the second is Spanish; per-window LID gets both right, but the transition happens mid-word and both windows attribute the boundary word to their own language. Fix: post-process the boundary via forced alignment (chapter 04); attribute the boundary word to whichever side its acoustic centroid falls in.

## Emitting the language on every segment

Every emitted segment gets a `language` field. Not `text` and a top-level file `language` — that pattern breaks the moment anyone code-switches. The schema from chapter 02 accommodates this trivially:

```json
{
  "language": "en-US",
  "segments": [
    {"id": 0, "language": "en-US", "start": 0.24, "end": 3.10, "text": "Welcome back."},
    {"id": 1, "language": "es-MX", "start": 3.10, "end": 6.80, "text": "Bienvenidos otra vez."}
  ]
}
```

The top-level `language` is a *majority summary* — the language of most of the audio — kept for backward compatibility with consumers that only look at the file-level tag. The per-segment tag is authoritative.

## Chapter summary

- LID splits into file-level, segment-level, and word-level tasks. Whisper handles the first two natively; the third needs specialist tooling and is rarely a text-side problem alone.
- Whisper LID sees only the first 30 s, is not calibrated, and emits ISO-639-1 codes. Map to BCP-47 before shipping.
- Alternatives: SpeechBrain VoxLingua107, NeMo TitaNet-LID, Silero for acoustic LID; fastText / CLD3 for *text* LID as a cross-check on the transcript.
- Calibrate LID confidence on your own held-out set. Use both top-1 probability and top-1-vs-top-2 margin as thresholds. Route low-confidence to a fallback path.
- Sequential code-switching is tractable: per-window LID + per-window ASR + per-segment language tags. Intra-sentential code-switching is genuinely hard; do not oversell it.
- Emit the routing table explicitly and version it. Every segment carries its own `language` field in BCP-47.
- Classic failure modes are silence / music / short clips fooling LID and similar-language-family confusion. Guard against them with VAD, higher thresholds for short files, and language-family cluster routing.
