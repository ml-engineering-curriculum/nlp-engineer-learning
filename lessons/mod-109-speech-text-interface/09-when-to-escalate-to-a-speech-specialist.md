# When to Escalate to a Speech Specialist

## Motivation

Chapter 01 named the boundary. This chapter makes it concrete: the specific problems that arrive at your desk labelled "ASR issue" that are not yours to fix, the vocabulary you need to hand them off cleanly, and the artefacts you produce as part of that handoff. This is short and prescriptive by design — when a speech-text pipeline has a quality problem, "which team owns this?" should not be a debate.

## The boundary in one paragraph

The NLP engineer owns the *text-side* of the speech pipeline: format contract, language routing, alignment merging, normalisation, ITN, punctuation, casing, diarisation merge, and downstream consumption. That includes fine-tuning small text-side models (punctuation restoration, truecasing) and authoring grammars (ITN). The speech specialist owns the *acoustic-side*: acoustic-model training and adaptation, decoder tuning, VAD, wake-word / keyword spotting, streaming ASR latency policies, and diarisation model training. Shared surface: the ASR output JSON, the diarisation RTTM, the timestamp contract, and the audio-quality feedback loop where text-side observations inform acoustic-side priorities.

## Escalation triggers: symptoms and owners

Sorting bugs to the right owner is most of the game. A representative catalogue:

| Symptom | Owner | Why |
|---------|-------|-----|
| Numbers like `"9 1 1"` instead of `"911"` | You (chapter 06) | ITN failure. |
| No punctuation or wrong capitalisation | You (chapter 07) | Punctuation / truecasing failure. |
| Off-by-a-few-hundred-ms word timestamps | You (chapter 04) | Alignment failure. |
| Segment start / end off by 3 s | You (chapter 04) | Chunking failure — verify times were rebased to file. |
| Wrong language detected for the file | You (chapter 03) | LID routing. |
| Wrong speaker attributed to a word | You (chapter 08) | Merge failure. |
| Whole regions of audio transcribed in the wrong language | Both — LID routing on your side, possibly acoustic-model bias on theirs | Start with your LID; if the LID was correct and the ASR still produced the wrong language, escalate. |
| Speaker count wrong on every file (systematically) | Speech | Diariser hyperparameters or model. |
| Speaker count wrong on some files (occasionally) | You + speech | Check the file's `min_speakers` / `max_speakers` bounds first. |
| High WER on domain vocabulary (drug names, tickers, product names) | Both | Custom dictionary or biasing (yours to author); acoustic fine-tune (theirs). |
| High WER on accented speech | Speech | Acoustic-model failure. Provide labelled examples; do not attempt to fix from text side. |
| High WER on low-bitrate telephony | Speech | Acoustic-model failure; possibly needs a telephony-tuned model (Whisper is trained on 16 kHz clean; 8 kHz phone lines behave differently). |
| Hallucinated segments in long silence | Speech-adjacent — the acoustic model or its VAD chunker | You can filter with `no_speech_prob` / `avg_logprob` (chapter 02); recurring hallucination on silence is the speech engineer's to root-cause. |
| Runaway repetition (`"the the the the"`) | Speech-adjacent | Decoder-side failure. Filter aggressively; escalate the model-level fix. |
| Missing words in overlapping speech | Speech | Target-speaker ASR is a research problem. Preserve the overlap flags on your side (chapter 08); communicate the gap; do not paper over it. |
| Streaming latency too high | Both | The streaming ASR latency budget is theirs; the punctuation commit-window is yours (chapter 07). Split the budget. |
| Wake-word not firing / firing too often | Speech | Wake-word models are a specialised sub-field. Not text-side. |
| Timestamp drift accumulating over an hour | Speech-adjacent | Chunking / conditioning policy. You should switch to VAD-based chunking (chapter 04) as a first move; if that does not fix it, escalate. |
| WER regression after acoustic model swap | Speech | Model change is theirs; provide the labelled regression set to help them triage. |

If you can classify a bug to a row in this table without opening the audio, most escalations become one-line JIRA tickets: "row X on the escalation matrix; here is the file id."

## Vocabulary you need to have the conversation

You are not the acoustic expert but you have to hold a conversation with one. A minimum vocabulary:

- **VAD** — voice activity detection. Silero and WebRTC are the common ones; matters for chunking and endpointing.
- **CTC vs. attention decoder vs. RNN-T** — the three decoder families. Whisper is attention-decoder; wav2vec 2.0 is CTC; Parakeet is RNN-T. Streaming-friendliness increases in that order.
- **Endpointing** — deciding when the current utterance is over in a streaming setting. Latency-critical; VAD does part of it.
- **Beam width / temperature** — decoder search hyperparameters. Trade off latency and quality.
- **Forced alignment** — chapter 04; you already know this one.
- **Speaker embedding / speaker verification** — the acoustic representation used for diarisation and re-identification.
- **WER / MER / CER / WIL** — evaluation metrics (chapter 05 covers WER). MER = Match Error Rate; WIL = Word Information Lost; CER = Character Error Rate. The Morris, Maier & Green paper (["From WER and RIL to MER and WIL"](https://www.researchgate.net/publication/221489635_From_WER_and_RIL_to_MER_and_WIL_improved_evaluation_measures_for_connected_speech_recognition), Interspeech 2004) is the standard reference.
- **DER / JER** — diarisation error rate; Jaccard error rate.
- **G2P** — grapheme-to-phoneme. Used in MFA-style alignment and TTS.
- **Mel spectrogram / log-mel** — the standard acoustic feature. Whisper uses 80-channel log-mel (128 in v3).
- **Sampling rate / bit depth** — 16 kHz mono is the standard input for most models. 8 kHz telephony is different enough to need a different model or resampling.

Knowing these words is enough for the conversation to be productive. You do not need to know how to fine-tune wav2vec 2.0. You need to know when the answer is "please fine-tune wav2vec 2.0."

## The handoff artefact

When escalating an acoustic issue, ship a reproducible package. The minimum bundle:

1. **The audio file(s)** demonstrating the issue. Ideally short (< 30 s each), ideally 10–20 files spanning the failure mode.
2. **The transcript(s) your pipeline produced**, verbatim, with timestamps. This is the JSON from chapter 02, exactly as it left your pipeline. Do not clean it up first — the speech engineer needs to see what your side saw.
3. **The reference (ground-truth) transcript(s)** if you have them. If you do not, mark "no reference; qualitative issue" and describe the symptom in text.
4. **The metric** — a WER, DER, or attribution-accuracy number showing the delta between what you expected and what you got.
5. **A one-sentence hypothesis**. Not a demand, a hypothesis. "This looks like acoustic-model behaviour on telephony bandwidth" is useful; "the ASR is broken" is not.
6. **The rest of the pipeline metadata** — model id and version, VAD version, punctuation model version, ITN grammar version. Same triage discipline as any bug report.

This bundle is worth templating in your team's issue tracker. The speech engineer's first request otherwise will be for exactly these six items.

## The failure loop across the boundary

Some failures live cleanly on one side; some genuinely straddle the boundary and can only be fixed by both sides talking. Recurring patterns:

- **Domain vocabulary.** New product names, drug names, ticker symbols. Text-side biasing (grammar-based ITN post-process, custom dictionary for casing, LLM correction pass) fixes some of it. Acoustic-side biasing (Whisper's `initial_prompt` parameter, wav2vec KenLM shallow fusion, hotword lists in production ASRs) fixes more. Neither alone is enough. Joint work: you supply the vocabulary list; they supply the acoustic biasing hooks.
- **Accent robustness.** Systematic mis-recognition on accents underrepresented in Whisper training. Text-side fixes (post-hoc word correction) are Band-Aids. Real fix: fine-tune the acoustic model on accented data or switch to a model with better accent coverage (MMS for many; Canary for others). Their call; you supply the accented eval set and the WER-per-accent breakdown.
- **Streaming latency budget.** End-to-end latency is `audio-in → text-out`. Their share is `audio-in → ASR partial`; your share is `partial → punctuated / cased / ITN'd text`. Neither can hit an aggressive budget alone. Joint work: agree on a target, measure each side, negotiate.
- **Speaker identification.** Diarisation as a model is theirs; the mapping to human names is yours; the vocal-enrolment pipeline that connects them is joint.

The rule of thumb: if the metric moves when you swap a model, it is probably them; if the metric moves when you change a JSON field or a grammar, it is probably you. When it moves both ways, it is a shared problem and needs a shared owner.

## When there is no speech specialist

Real situation on small teams. You are the closest thing.

Three defensive postures:

- **Use hosted ASRs where possible.** Deepgram, AssemblyAI, Azure Speech, Google Speech-to-Text. Their teams do the acoustic work; you consume the JSON. This is often the right answer on any team below ~5 engineers.
- **Do not train acoustic models.** Fine-tuning wav2vec 2.0 or Whisper from scratch is possible but is a serious project; a false-start burns weeks. Pick a pre-trained model, adapt with text-side techniques, and stop.
- **Escalate to consulting or vendor support.** Both major hosted vendors and the pyannote / NeMo teams offer consulting. When a genuine acoustic problem lands on a team without a specialist, buy the help.

The escalation matrix in this chapter still applies — being the whole team does not change the ownership boundary of a problem; it just means you know the shape of what you are punting on.

## Chapter summary

- The boundary between text-side and acoustic-side is durable and worth respecting: ITN, LID routing, alignment merge, punctuation, casing, diarisation merge, format contract are yours; acoustic-model training, decoder tuning, VAD, wake-word, streaming latency at the audio level, diarisation model training are theirs.
- Sort bugs to the escalation matrix before you invest engineering time. Most "ASR broken" tickets are text-side once you look at them.
- Learn the minimum acoustic vocabulary (VAD, CTC vs. attention vs. RNN-T, endpointing, forced alignment, WER / DER, G2P, mel spectrogram, sampling rate) so the handoff is productive.
- The handoff artefact is a repeatable bundle: audio + your JSON + reference + metric + hypothesis + pipeline metadata. Template it.
- Straddling problems (domain vocab, accents, streaming latency, speaker identification) need joint work. Communicate the split; agree on shared metrics.
- On teams without a speech specialist: prefer hosted ASRs, avoid training acoustic models, and buy consulting for genuine acoustic problems. The boundary does not change; only your options for who staffs the other side do.
