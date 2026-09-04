# Diarisation Post-Processing

## Motivation

Diarisation answers the question *who spoke when*. On its own it is a stream of `(speaker_id, start, end)` triples — a time-oriented view of the audio with no words attached. Your job at the text-side is to join it to the transcript: attach a speaker label to each word (or each segment), handle overlaps sensibly, and produce a per-speaker view a downstream can consume.

You do not train the diariser. You do own the *merge* — a small pile of code that decides which words belong to which speaker turn, what to do when the diariser labels an overlap ambiguously, and how to resolve `SPEAKER_00` into `"Clara"` when a name registry exists.

## What a diariser actually returns

Two common surfaces:

- **RTTM (Rich Transcription Time Marked)** — the standard file format from NIST's Rich Transcription evaluations. Space-delimited, one line per speaker turn. The LDC docs ([RTTM format v13](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)) are the spec; every serious diariser reads and writes it.
- **JSON / Python objects** — pyannote returns an `Annotation` object with `(segment, track, label)` iteration; NeMo returns similar. All map trivially to RTTM.

RTTM line, unpacked:

```
SPEAKER  meeting_abc  1   12.340  4.820   <NA>  <NA>  SPEAKER_00  <NA>  <NA>
   type    file-id chan  start  duration                 label
```

Key properties:

- **Start is absolute in the file**, in seconds; duration is also seconds. End = start + duration; do the addition once and stop worrying about it.
- **Speaker labels are anonymous strings**, typically `SPEAKER_00`, `SPEAKER_01`, ... They are consistent *within a file* but not across files. Do not assume `SPEAKER_00` in file A is the same person as `SPEAKER_00` in file B.
- **Turns may overlap.** Two `SPEAKER` lines with intersecting time intervals means the model detected overlapping speech. Modern overlap-aware pipelines (pyannote v3) emit these routinely; older tools may not.
- **Non-speech regions are absent.** No line means silence or noise. Do not infer that missing time is `SPEAKER_UNKNOWN`.

The reference open-source tool for diarisation is pyannote.audio (Bredin et al., ["pyannote.audio: neural building blocks for speaker diarization"](https://arxiv.org/abs/1911.01255), ICASSP 2020; overlap-aware update: Bredin & Laurent, ["End-to-end speaker segmentation for overlap-aware resegmentation"](https://arxiv.org/abs/2104.04045), Interspeech 2021). The Park et al. survey (["A Review of Speaker Diarization: Recent Advances with Deep Learning"](https://arxiv.org/abs/2101.09624), 2022) is a good broader read.

Concretely with pyannote 3.x:

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HF_TOKEN",
)
diarisation = pipeline("clip.wav")

for turn, _, speaker in diarisation.itertracks(yield_label=True):
    print(f"{turn.start:.2f}\t{turn.end:.2f}\t{speaker}")

diarisation.write_rttm(open("clip.rttm", "w"))
```

`pyannote.audio` also has an End-to-End Neural Diarisation (EEND) family variant (Fujita, Kanda, Horiguchi, Nagamatsu & Watanabe, ["End-to-End Neural Speaker Diarization with Self-Attention"](https://arxiv.org/abs/1909.06247), ASRU 2019) that handles overlap natively in a single model. Pipeline vs. EEND is a speech-engineer decision; both feed the same RTTM downstream.

## Merging diarisation with word timestamps

The merge is where post-processing lives. You have two aligned streams:

- Words with `(word, start, end, probability)` from forced alignment (chapter 04).
- Speaker turns with `(speaker, start, end)` from the diariser.

The simple rule that works most of the time: assign each word to the speaker whose turn has maximum *time-overlap* with the word's interval. Ties (equal overlap) go to the turn with the earlier start time. Words with zero overlap against all turns fall back to `unknown`.

```python
def assign_speaker(word_start, word_end, turns):
    best, best_overlap = None, 0.0
    for turn in turns:
        overlap = max(0.0, min(word_end, turn.end) - max(word_start, turn.start))
        if overlap > best_overlap:
            best, best_overlap = turn, overlap
    return best.speaker if best else "unknown"
```

WhisperX ships an equivalent merge in `whisperx/diarize.py` if you want a reference implementation. Wire it up if you use WhisperX; port the same rule if you assemble the pipeline yourself.

Once every word has a speaker, group consecutive same-speaker words back into segments to emit into the JSON. Segment boundaries appear whenever the speaker changes; that gives you the per-speaker turn shape most downstream consumers want.

## Overlapping speech

Overlap is where the pipeline stops being simple. Three flavours of overlap you will see:

- **Short backchannel overlap.** Speaker B says `"mmhm"` while Speaker A is talking. The word-level merge silently attributes the `"mmhm"` to whichever speaker's turn has more time-overlap, which is typically A. B's backchannel disappears. Usually fine.
- **Simultaneous speech.** Both speakers talk for 5 seconds. Whisper transcribes only one of them (whichever was louder / closer to the mic); the other's words never make it into the transcript at all. This is a *transcription* failure, not a diarisation failure — the words are gone by the time you see the transcript.
- **Interruption at speaker change.** A finishes mid-word, B starts mid-word. Diariser marks the boundary approximately; forced-aligner splits the word across two speakers. Word attribution is ambiguous.

What to do:

1. **Preserve the diariser's overlap output.** If pyannote flagged a region as overlapping (via its overlap-detection or its `Overlap` label), pass that through in your JSON. A downstream can then decide to display both speakers, mark the segment as `overlap`, or apply a specialised model for that region.
2. **Add an `overlap` boolean per word or per segment.** Recommended default: word inherits `overlap=true` if the diariser's overlap regions cover more than half of the word's duration.
3. **Do not average speakers.** `"SPEAKER_00/01"` is not a useful label. Pick one (or `unknown`) and record ambiguity separately.
4. **Understand what your transcript is missing.** If the ASR only transcribed one speaker in the overlap region, the *other* speaker's turn in your JSON will have no words — an empty `words` array for a real time interval. That is the correct behaviour. Do not fabricate.

Overlap in general is a hard research problem. Chapter 09 discusses when to escalate to the speech specialist; overlap-heavy corpora (interviews, dinner parties, board meetings) are one of the earlier escalation triggers.

## Anonymous labels and identity resolution

`SPEAKER_00` is not a name. Turning it into one is a family of problems, roughly ordered by difficulty:

- **Within-file consistency.** Diarisers already provide this. If the same person speaks at 0:15 and 5:30, both windows get `SPEAKER_00`.
- **Cross-file re-identification.** Same speaker across multiple files should get the same anonymous label. Requires a *speaker verification* model (e.g. ECAPA-TDNN, WavLM speaker embeddings) run once per turn to produce embeddings, then a clustering pass across the file set. Speaker-engineer territory when done well; the text-side just consumes the cluster IDs.
- **Name resolution against a registry.** You have `SPEAKER_00`; you have a list of expected participants (attendee list, calendar invite, host + guest). Assign names via a mix of:
  - **Vocal enrolment** — a speaker embedding for each known name; nearest-neighbour lookup on each turn. Requires enrolment data.
  - **Contextual disambiguation** — parse the transcript itself: the first speaker who says `"Hi, this is Clara"` is Clara.
  - **Role priors** — in a call recording, the caller is the host; in a podcast, the first non-intro speaker is the guest.

None of these are the NLP engineer's core responsibility, but you own the *mapping table* — a small piece of state per file that says "for this file, `SPEAKER_00` = `Clara`, `SPEAKER_01` = `Marcus`" — and the code that applies it to the JSON. Version it. Speakers get relabelled; your mapping should be an artefact, not an implicit lookup buried in serving code.

## The JSON shape after merge

Working from the chapter 02 schema, the merged form looks like:

```json
{
  "language": "en-US",
  "duration": 328.4,
  "model": "whisper-large-v3-turbo",
  "diarisation": {
    "model": "pyannote/speaker-diarization-3.1",
    "num_speakers": 2,
    "speaker_names": {"SPEAKER_00": "Clara", "SPEAKER_01": "Marcus"}
  },
  "segments": [
    {
      "id": 0,
      "speaker": "SPEAKER_00",
      "start": 0.24,
      "end": 4.10,
      "text": "Welcome back to the podcast.",
      "overlap": false,
      "words": [
        {"word": "Welcome", "start": 0.24, "end": 0.72, "probability": 0.99, "overlap": false},
        ...
      ]
    },
    ...
  ]
}
```

Notes:

- `speaker` remains the anonymous label; downstream reads `diarisation.speaker_names` to get a display name. Keeps the mapping addressable.
- `overlap` is set once per word (from the diariser's overlap regions) and once per segment (if any word in the segment overlaps). Cheap to compute; enormous downstream value.
- `diarisation` block is optional at the file level. When the file was mono / single-speaker / diariser skipped, omit the block cleanly rather than emitting `num_speakers: 1`.

## Evaluating diarisation output

You are not scoring the diariser (that is the speech engineer's job) but you may be scoring the *merge*. Two orthogonal metrics:

- **DER (Diarisation Error Rate)** — the standard NIST metric ([NIST Rich Transcription Evaluation](https://www.nist.gov/itl/iad/mig/rich-transcription-evaluation)). Sum of three errors as a fraction of speaker time: miss (speech classified as non-speech), false alarm (non-speech classified as speech), and speaker confusion (speech attributed to the wrong speaker). The `md-eval` tool from `sctk` is the reference scoring implementation. `pyannote-metrics` ([documentation](https://pyannote.github.io/pyannote-metrics/)) is the Python re-implementation you should reach for first.
- **Word-level speaker attribution accuracy** — for each word in the transcript, is it labelled with the correct speaker? This is the metric that matters for your merge specifically. Compute against a small human-labelled set.

Also worth tracking:

- **Speaker over-segmentation rate** — number of predicted turns / number of true turns. > 1.5 means the diariser is over-splitting; the merge inherits the churn.
- **Overlap recall** — for time regions with true overlap, does the pipeline preserve the overlap flag? Missing overlap silently loses one speaker's words.
- **Backchannel loss** — for known backchannels in the reference (`"mmhm"`, `"right"`), what fraction survive the merge to appear in the transcript?

Report DER for the underlying model (a system quality number), attribution accuracy for the merge (your quality number), and be very explicit in every dashboard which is which. Team-blame arguments over "the diariser is broken" usually turn out to be one side's DER regression and the other side's attribution regression, and no one can tell which until the numbers are labelled.

## Common failure modes

- **Timestamp drift between transcript and diariser.** The two pipelines were run against different chunkings of the audio. Words end up half a second off from their true diariser turn, and the merge gets attribution wrong at boundaries. Fix: run diariser and forced alignment against the same audio; verify both operate on the file in the same sample-rate and channel configuration.
- **Speaker count wrong.** Diariser silently over-clusters (too many speakers) or under-clusters (too few). Fix: consult the diariser's hyperparameters (`num_speakers`, `min_speakers`, `max_speakers` in pyannote); when the count is known, pin it. When unknown, verify against a small labelled set periodically.
- **Same person split into two speakers.** They spoke very quietly for a stretch and the embedding drifted. Fix: post-process by clustering speaker embeddings across the file and merging speakers whose centroids are close.
- **Overlap regions labelled as one speaker only.** Overlap-detection was disabled or the pipeline is old (pre-v3). Fix: upgrade or add an explicit overlap-detection pass.
- **DIHARD-style noise.** In noisy environments (Ryant et al., ["The Second DIHARD Diarization Challenge"](https://arxiv.org/abs/1906.07839), Interspeech 2019, is the benchmark), DER degrades sharply. Fix: know your target environment; if it is close to DIHARD-style, escalate the acoustic setup (chapter 09).
- **Cross-file re-identification without vocal enrolment.** `SPEAKER_00` in the morning meeting is a different person from `SPEAKER_00` in the afternoon meeting; a naive per-day rollup mislabels everything. Fix: never assume cross-file identity without an explicit re-id pass.

## Chapter summary

- Diarisation output arrives as RTTM (or an equivalent object). Merge attaches the labels to the transcript via time-overlap between words and turns.
- The workhorse merge rule: each word gets the speaker whose turn maximally overlaps its time interval; tie-break on earlier turn start.
- Overlap is a first-class problem: preserve the diariser's overlap output, mark `overlap` on words and segments, do not fabricate speakers or words.
- Anonymous labels (`SPEAKER_00`) are within-file consistent but not cross-file. Identity resolution against a name registry is a versioned, per-file mapping the NLP engineer owns.
- Report DER for the model, word-level speaker-attribution accuracy for the merge; keep them labelled separately. Also track over-segmentation, overlap recall, and backchannel loss.
- Common failures: timestamp drift between transcript and diariser, speaker-count errors, split identities, missed overlap, DIHARD-style noise, unsafe cross-file re-id.
