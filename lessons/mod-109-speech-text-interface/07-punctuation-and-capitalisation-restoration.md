# Punctuation and Capitalisation Restoration

## Motivation

Raw ASR output that lands in your pipeline from a wav2vec 2.0 fine-tune is a lowercase, unpunctuated wall of text. Whisper's output is punctuated and cased but not always the way you want and not always where you would put the boundary. Downstream summarisation, sentiment, translation, and human legibility all degrade sharply on unpunctuated text: sentence boundaries disappear, proper nouns lose their signal, and the transcript reads as one long sigh.

This chapter is about the models and pipelines that put punctuation and casing back — both offline and in streaming mode where the "current" transcript is a moving target.

## Framing the problem

Two related but distinguishable tasks travel together:

- **Punctuation restoration.** Given a lowercase, punctuation-less sequence of words, predict which punctuation mark (if any) belongs *after* each word: `.`, `,`, `?`, `!`, `-`, `:`, `;`, `"`, or nothing. Sequence labelling, one label per token.
- **Truecasing.** Given a lowercase word sequence, decide the case of each character: all-lowercase, initial capital, all-uppercase, mixed. Also sequence labelling, one label per token, but the label set is typically 3 or 4 cases rather than 8+ punctuation marks.

They interact — capitalisation follows sentence-final punctuation — but you can train them jointly or separately. Most public reference models do them jointly; some production stacks split them because the truecasing distribution shifts less between domains than the punctuation distribution does.

The classic reference is Alam, Khan & Alam (["Punctuation Restoration using Transformer Models for High- and Low-Resource Languages"](https://aclanthology.org/2020.wnut-1.18/), W-NUT 2020), which pins down the BERT-based sequence-labelling recipe on the IWSLT 2011 dataset. Everything that follows is a variation on that theme.

## The sequence-labelling recipe

At a mechanical level:

- Take an encoder (BERT / RoBERTa / DeBERTa / XLM-R) fine-tuned on token classification.
- Vocabulary of labels: `O` (no punctuation after this token), `COMMA`, `PERIOD`, `QUESTION`, `EXCLAM`, ... (whatever the target set is; keep it small).
- Loss: standard token-classification cross-entropy over the label set.
- At inference, feed a window of the transcript; get a label per token; interleave the predicted punctuation back into the token stream.

Training data is trivial to synthesise: take any well-punctuated corpus (news, books, Wikipedia, well-transcribed audiobooks), strip the punctuation and lowercase everything to build the input, use the original as the labels. Because supervision is free, dataset size is not the bottleneck; matching your target domain is.

For truecasing, the label set is typically `{LOWER, TITLE, UPPER, MIXED}` — the last for weird internal casings like `iPhone` where you need a shipped dictionary lookup rather than a learned rule. Sanchez (["Word-Casing: A New Look at the Automatic Case Restoration Problem"](https://arxiv.org/abs/1903.11222), 2019) is a good survey of the rule-vs-statistical-vs-neural trade-offs.

## Reference public models

Two commonly-used checkpoints from Hugging Face are worth knowing about because they cover most language + punctuation needs out of the box:

- [`oliverguhr/fullstop-punctuation-multilang-large`](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) — XLM-RoBERTa-large fine-tuned to predict `.`, `,`, `?`, `-`, `:` for English, German, French, Italian. Small label set, decent multilingual reach.
- [`felflare/bert-restore-punctuation`](https://huggingface.co/felflare/bert-restore-punctuation) — BERT-base fine-tuned on IWSLT / TED transcripts for English. Older but well-documented.

Both take lowercase, unpunctuated input; both output a token-classification distribution. Use them as reference implementations and starting points, not as production defaults — they were trained on public corpora and shift under domain change.

Concrete usage of the multilingual one:

```python
from deepmultilingualpunctuation import PunctuationModel

m = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")
m.restore_punctuation(
    "my name is clara and i live in berkeley what about you"
)
# → 'My name is Clara, and I live in Berkeley. What about you?'
```

`deepmultilingualpunctuation` is the reference client library and does token→label→string reassembly for you. If you skip it, the plumbing is 30 lines of transformers-native code, but you inherit responsibility for windowing (below).

## Choosing the training data

Off-the-shelf models are trained on written text — Wikipedia, news, books. Written text has a different punctuation distribution than spoken transcripts:

- Written text has more commas than spoken transcripts, because writers use them for embedded clauses that speakers just pause through.
- Written text has fewer question marks than conversational transcripts (customer support, interviews).
- Written text has virtually no filler-word transitions (`"um, so, ..."`); spoken transcripts are full of them, and the model has to learn either to keep them uncommaed or to comma them in the "right" spots.
- Written text uses more semicolons and dashes than transcripts should. Restrict your label set upfront.

If your target is *transcript legibility*, the right training corpora are ones already in transcript form — TED / IWSLT (see the [IWSLT proceedings](https://aclanthology.org/venues/iwslt/)), publicly available broadcast news, hearing transcripts, court records. Fine-tune the public checkpoint on your target-domain transcripts for one or two epochs; even a few thousand well-punctuated transcript sentences move the numbers meaningfully.

Watch the domain in your source data too. A model trained on news commas will overpunctuate podcasts and undercomma technical talks.

## Windowing and long transcripts

Encoder inputs cap at 512 tokens (BERT) or 4 096 (Longformer) or similar. A one-hour transcript is many multiples of that. The offline solution is straightforward:

- Split the transcript into overlapping windows (256 tokens with 64-token overlap is a reasonable default).
- Run the model on each window.
- In the overlap zone, average the label logits from both windows before argmaxing. This eliminates the boundary discontinuity where the last token of window N and the first token of window N+1 disagree because each was decoded without the other's context.
- Reassemble.

Two edge cases:

- **Sentence starts near a window boundary.** Long-range dependencies (an opening quote, a subordinate clause anchor) fall outside the window. Larger overlap helps; a Longformer / DeBERTa-v2 with a bigger context window helps more.
- **Token vs. word alignment.** BERT tokenises with WordPiece; labels are typically only meaningful on the first subword of each word. Standard convention: label the first subword and mask (`-100`) the rest during training and use only the first subword's prediction at inference. If you skip this you get half your labels on `##suffix` tokens and everything drifts.

## Streaming punctuation

Streaming ASR (chapter 09 is explicit that the acoustic streaming problem is the speech engineer's) still surfaces streaming *text* at the interface: partials arrive every 200–500 ms, and the punctuation model needs to keep up without flip-flopping.

Two failure modes to avoid:

- **Flapping.** The comma the model placed after token N gets deleted when token N+1 arrives because the enlarged context now argues for a different label. Users see punctuation appear and vanish. Bad.
- **Latency-tail dropouts.** The model waits for a right-context margin before committing any label; a long pause at the end of a stream means no punctuation ever appears until the finaliser runs. Also bad.

The standard technique is a *commit window*: only commit labels for tokens that have at least N tokens of right context (typically N=3–5). Beyond that horizon, labels are treated as tentative and may be revised without user visibility. When the ASR emits a `<|final|>` marker (VAD-endpointed silence), commit all remaining tokens with whatever labels the model currently predicts.

Guerreiro, Zerva, van Stigt, Rei & Martins (["Multilingual punctuation restoration for streaming speech"](https://arxiv.org/abs/2203.14976), 2022) is the reference paper for the streaming problem specifically. It formalises the latency-vs-accuracy trade-off (very analogous to the streaming translation literature — see also Kim et al., ["StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection"](https://arxiv.org/abs/2406.06097), 2024).

Practical streaming pipeline:

```
partial transcript (rolling window of ~200 tokens)
   │
   ▼
punctuation + case model → per-token label logits
   │
   ▼
commit-window filter: emit labels for tokens with ≥N right context; keep the rest tentative
   │
   ▼
diff against the last emitted state → send only the delta to the UI
```

Keeping the last-emitted state per session on the server side is worth doing; sending a diff instead of the whole partial transcript on every tick saves both bandwidth and re-render churn on the client.

## Truecasing details

Casing tends to be more consistent than punctuation across domains: sentence-initial capitals, proper nouns, acronyms. A joint punctuation+case model handles the first two adequately.

For proper nouns and acronyms it helps to seed the model with a domain dictionary, especially for names, product names, and brand acronyms. Two integration patterns:

- **Post-processing lookup.** After the model outputs cased tokens, do a case-insensitive lookup against your dictionary and overwrite the model's choice for exact matches. Cheap, deterministic, easy to update.
- **Feature injection.** Concatenate a boolean feature to each token indicating "this token matched an entry in the proper-noun dictionary." Requires re-training but is more robust than post-processing.

Watch out for the failure mode where the ASR produced the wrong homophone (`"Marc"` → `"mark"` and now casing gives you `"Mark"` instead of `"Marc"`). Truecasing cannot fix acoustic errors and does not try to; document that boundary with the consumer.

The classic truecasing literature (Susanto, Chieu & Lu, ["The Impact of Truecasing on Neural Machine Translation"](https://aclanthology.org/W16-2308/), WMT 2016) framed the value of truecasing for downstream tasks; the reference for approaches is Sanchez, ["Word-Casing"](https://arxiv.org/abs/1903.11222).

## When to skip your own model and use Whisper's output

Whisper's decoder produces punctuated, cased text natively. On clean-speech, monolingual English it is often close enough that adding your own punctuation model helps at the margins and adds latency. Two questions decide it:

1. **Is Whisper's punctuation acceptable on a fixed evaluation slice?** Score punctuation-F1 (below) on 100–200 transcripts against a human-punctuated reference. If Whisper alone is above your quality bar, stop. Do not add a model to solve a problem you do not have.
2. **Are you running Whisper at all in the streaming path?** If the answer is wav2vec-CTC for streaming, then Whisper's punctuation is not available and you need your own punctuation model at the interface.

Most production stacks end up with *both*: use Whisper's punctuation where it lands well; run a punctuation model in the wav2vec streaming path; unify the two behind the JSON schema.

## LLM-based punctuation and casing

An LLM will punctuate and case a transcript perfectly well if you ask it. Two reasons this shows up in production, and two reasons to be cautious.

Reasons to use it:

- **Language coverage is free.** The LLM handles any language the model was trained on without a fine-tune.
- **You already run an LLM in the pipeline** (for summarisation, classification, or downstream RAG). Punctuation is a free side effect and eliminates a separate model.

Reasons to be cautious:

- **Silent rewriting.** LLMs tend to fix words, split sentences differently, or correct disfluencies. If your contract is "verbatim transcript with punctuation added," LLM-based punctuation may violate it. Constrain with a *diff check*: assert that removing your applied punctuation and casing yields the exact original transcript. Reject and retry on failure.
- **Cost and latency.** A dedicated small model runs orders of magnitude faster and cheaper. If you are billing per token or per second, dedicated is almost always the right call for high-volume batch.

If you go the LLM route, the diff-check guard is not optional. Log it as a metric and alert when the rewrite rate goes above your tolerance.

## Evaluation: what to measure

Punctuation and casing are notoriously bad targets for pure accuracy. Two metric families to run in parallel:

- **Token-level F1 per label.** Precision, recall, F1 for each of `PERIOD`, `COMMA`, `QUESTION`, `TITLE`, `UPPER`. Report per-label; a joint F1 hides that your commas are terrible even when periods are good. The IWSLT punctuation-restoration papers use exactly this framing.
- **Slot-level F1.** For sentence boundaries specifically: precision and recall of the *set* of sentence-ending positions (period, question, exclam) against the reference. This is the metric that correlates with downstream summarisation quality — an off-by-one period is a slot-level error but a token-level exact match; you want to see both.

Also worth reporting:

- **Word Error Rate before and after.** If your model rewrites words (bad), WER will show it.
- **Case-sensitive vs. case-insensitive WER.** Delta = truecasing error rate.
- **Latency per token in streaming mode.** Report p50 and p99 for the commit-window delay.

Alam et al. (above) tabulates the same metrics on IWSLT so you have a public reference to calibrate against.

## Chapter summary

- Punctuation restoration and truecasing are token-level sequence-labelling tasks; the reference recipe is a BERT-family encoder over lowercase, unpunctuated text.
- Reference public checkpoints: `oliverguhr/fullstop-punctuation-multilang-large` (XLM-R, multilingual) and `felflare/bert-restore-punctuation` (English, IWSLT).
- Training data must match your target domain — spoken transcripts, not written prose. Fine-tune public checkpoints on a few thousand target-domain sentences.
- For long transcripts, window with overlap and average logits in the overlap zone. Label only the first subword of each word.
- Streaming punctuation needs a commit-window (N tokens right context) to prevent flapping and a `<|final|>`-triggered flush for the tail.
- Whisper produces usable punctuation and casing on clean English; measure before you replace it. Most stacks end up with both a Whisper path and a wav2vec+punctuation path.
- LLM-based punctuation works but silently rewrites text. Guard with a diff check that removing applied marks yields the original tokens.
- Evaluate with per-label F1, slot-level F1 for sentence boundaries, case-sensitive vs. -insensitive WER delta, and streaming latency percentiles.
