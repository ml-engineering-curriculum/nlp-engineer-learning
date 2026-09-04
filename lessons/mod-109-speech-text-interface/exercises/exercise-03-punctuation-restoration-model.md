# exercise-03: Punctuation Restoration Model

**Estimated effort:** 2 hours

## Objective

Fine-tune a token-classification model to restore punctuation and truecasing on unpunctuated, lowercase transcripts, evaluate it against an off-the-shelf public baseline, then wire it into a streaming-friendly commit-window inference loop that handles the flapping / latency-tail failure modes from chapter 07.

The exercise stops short of writing a full streaming server (mod-112) but shows every mechanism the real thing depends on.

## Prerequisites

- Chapters [02](../02-asr-outputs-whisper-and-wav2vec.md), [05](../05-text-normalisation-for-asr.md), [07](../07-punctuation-and-capitalisation-restoration.md).
- Python 3.10+; `transformers>=4.40`, `datasets`, `torch`, `evaluate`, `seqeval`, `scikit-learn`.
- A GPU is strongly recommended for training. Inference and evaluation run on CPU.

## Data

Two options — recommended default first.

- **Recommended.** IWSLT 2011/2012 English TED transcripts, the standard benchmark for punctuation restoration (as used by Alam, Khan & Alam, ["Punctuation Restoration using Transformer Models for High- and Low-Resource Languages"](https://aclanthology.org/2020.wnut-1.18/), W-NUT 2020, and earlier by Che, Wang & Ma, ["Punctuation Prediction for Unsegmented Transcript Based on Word Vector"](https://aclanthology.org/L16-1103/), LREC 2016). Available packaged on Hugging Face as `unography/iwslt-punctuation` or similar third-party mirrors; if none is stable at the time you take the exercise, fall back to option 2 and note the substitution in `report.md`.
- **Alternative.** Any well-punctuated English corpus: use `wikipedia` (`20220301.simple` for size), TED-LIUM transcripts (Rousseau et al., 2018), or the `openwebtext` corpus. Strip punctuation and lowercase to build training inputs; keep the original as labels. This is the training-data trick from chapter 07.

Aim for ~100 k sentences of training data and ~5 k of held-out test data.

## Problem statement

### Part A — Prepare the labelled data

For each sentence in your corpus:

1. Tokenise the *punctuated, cased* sentence into words.
2. For each word, record:
   - Its lowercase form as the model input.
   - Its case label: `LOWER`, `TITLE`, `UPPER`.
   - The punctuation mark that appears *after* it (if any): map to one of `O` (nothing), `COMMA` (`,`), `PERIOD` (`.`), `QUESTION` (`?`), `EXCLAM` (`!`), `DASH` (`-`), `COLON` (`:`). Anything else is folded into `O`.
3. Emit `(words, punct_labels, case_labels)` triples.

Save `train.jsonl` and `test.jsonl` with these records. Sanity-check: report the distribution of punctuation labels and casing labels; the majority class should be `O` and `LOWER` respectively (usually 80–90 %).

### Part B — Baseline: off-the-shelf model

Run [`oliverguhr/fullstop-punctuation-multilang-large`](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) on your test set via the `deepmultilingualpunctuation` library. For each test sentence:

1. Feed the *lowercase, unpunctuated* input.
2. Get the model's punctuated + cased output.
3. Reconstruct per-token punctuation and case labels from the model's output.
4. Score against your ground-truth labels.

Report token-level precision, recall, and F1 per label for both punctuation and casing. Also report *slot-level* F1 for sentence boundaries: treat each period / question / exclam as a sentence-ending "slot" and compute F1 over slot positions.

Cache the scores in `baseline_report.md`.

### Part C — Fine-tune your own model

Fine-tune a BERT-family encoder for joint punctuation + casing prediction:

- Backbone: `distilbert-base-uncased` (fast) or `bert-base-uncased` (better quality). For multilingual work substitute `xlm-roberta-base`.
- Task: token classification with *two* heads — one 7-way over punctuation labels, one 3-way over case labels. Alternatively, a single flattened 21-way head (`{punct} × {case}`) works too; report which you chose.
- Tokenise inputs at the subword level; assign labels only to the first subword of each word, mask the rest with `-100` (chapter 07's alignment note).
- Loss: sum of the two cross-entropies. Use `class_weight` inversely proportional to sqrt of frequency to counter the `O` / `LOWER` majority.
- Trainer: `transformers.Trainer` with `TrainingArguments(per_device_train_batch_size=32, num_train_epochs=3, learning_rate=5e-5, warmup_ratio=0.1)`. Log every 200 steps.

Evaluate on the same test set with the same metric harness as Part B. Report the per-label F1 and slot-level F1 in `train_report.md`.

### Part D — Windowing for long transcripts

Pick 5 of your test transcripts of ≥ 1 000 tokens each. Run your model over them with:

- **Naive.** Split into non-overlapping 256-token windows; concatenate outputs.
- **Overlapped + logit-averaged.** 256-token windows with 64-token stride; in the overlap zone, average the logits from both windows before argmax.

Compute the per-label F1 for both approaches on the long transcripts. The overlap-averaged version should show a small improvement, especially near window boundaries. Report the delta in `train_report.md`.

### Part E — Streaming commit-window inference

Write a `streaming_inference.py` that simulates a streaming source:

- Read a long transcript token-by-token (simulating ASR partials arriving one token at a time).
- Maintain a rolling buffer of the last 200 tokens.
- Every K tokens (K=5 is fine), re-run the model on the rolling buffer.
- **Commit-window policy.** Only *emit* labels for tokens with at least `N=4` tokens of right context. Tokens closer to the end of the buffer are held as tentative and may be revised.
- On a simulated `<|final|>` marker (end of the transcript, or a manually-inserted marker), flush all remaining tokens with their current labels.

Log two things:

- **Emission latency** — for each token, how many wall-clock milliseconds elapsed between its arrival and its committed label emission. Report p50, p90, p99.
- **Flapping** — the fraction of tokens whose *tentative* label was different from its final *committed* label. Should be near zero if the commit window is doing its job; non-zero flapping means N is too small.

Report both in `streaming_report.md`.

### Part F — Compare policies

Sweep `N ∈ {0, 2, 4, 8}` on the streaming runner and plot both metrics (latency and flapping) as a function of N. Chapter 07 predicts the classic latency-vs-quality trade-off. Comment on which N is a good default for your model in `streaming_report.md`.

### Part G — Consolidate

`report.md` (400–600 words):

- Data prep summary (Part A) — corpus, sizes, label distributions.
- Baseline vs. fine-tuned comparison table on per-label F1 (Parts B / C).
- Windowing lift for long transcripts (Part D).
- Streaming trade-off summary with the N sweep numbers (Parts E / F).
- One paragraph on what breaks first: which label do you get wrong most often, and what would you change (more data, richer label set, LLM refinement pass) to move it.

## Starter guidance

- **Use the `deepmultilingualpunctuation` library for the baseline.** It handles token→label→string reassembly, which is fiddly if you do it yourself.
- **Label alignment on subwords.** `is_split_into_words=True` in the tokeniser gives you `word_ids` back per subword; iterate and set the label to `-100` for anything past the first subword. Missing this is the #1 silent bug in this exercise.
- **Class weights matter.** Without them, the model happily predicts `O` and `LOWER` for everything and gets 90% token-level accuracy while being useless.
- **Do not use accuracy as your primary metric.** Per-label F1 on `PERIOD`, `COMMA`, `QUESTION`, `TITLE`, `UPPER` is what tells you whether the model works.
- **Streaming simulation is not a real streaming server.** For this exercise a `for token in tokens: buffer.append(token); if len(buffer) % K == 0: model(buffer)` loop is enough. Do not build a websocket server; that is mod-112.
- **Latency measurements need `time.perf_counter()`**, not `time.time()`. `time.time()` has millisecond-ish granularity that lies to you on CPU.
- **Report Whisper as a reference.** Bonus: for a few test sentences, also transcribe the equivalent audio (if you have paired data) with Whisper `medium.en` and record the punctuation Whisper produces natively. Comment on whether your model outperforms Whisper's implicit punctuation.

## Acceptance criteria

- [ ] `data/train.jsonl` and `data/test.jsonl` with `(words, punct_labels, case_labels)` triples and reported label distributions.
- [ ] `baseline_report.md` with per-label precision / recall / F1 and slot-level F1 for the off-the-shelf model.
- [ ] `train.py` fine-tunes a token-classification model with joint punctuation + casing heads.
- [ ] `train_report.md` with per-label F1 for the fine-tuned model and the naive-vs-overlapped windowing delta on long transcripts.
- [ ] `streaming_inference.py` simulates a streaming source with commit-window policy; logs per-token latency and flapping.
- [ ] `streaming_report.md` with the N sweep (latency vs. flapping) and one recommended default.
- [ ] `report.md` (400–600 words) summarising all of the above.

## Stretch goals

- **Multilingual.** Repeat with `xlm-roberta-base` on a multilingual mix (English, Spanish, French). Report per-language F1.
- **LLM-based punctuation baseline.** Prompt a small open LLM (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) with a few-shot punctuation restoration prompt over your test sentences. Add a diff-check guard: verify the LLM did not alter the underlying words. Compare F1 vs. the fine-tuned model, and count violation rate.
- **CRF or structured decoder on top.** Add a linear-chain CRF layer on the punctuation head; measure the F1 delta on adjacent-label transitions (`COMMA`+`PERIOD` two words apart is unusual; CRF should suppress it).
- **Truecasing dictionary override.** Post-process outputs with a small dictionary of proper nouns and acronyms (chapter 07). Report the delta on the `MIXED` case if you added the label.
- **Real streaming rig.** Wire your commit-window logic into a `websockets` server that ingests one token per message and emits diff events. Not a full production stack — just enough to confirm the shape.

## Deliverables

Ship as a directory:

```
data/
  train.jsonl
  test.jsonl
baseline.py
baseline_report.md
train.py
train_report.md
streaming_inference.py
streaming_report.md
report.md
checkpoints/
  <best model dir>       # or a link + hash
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
