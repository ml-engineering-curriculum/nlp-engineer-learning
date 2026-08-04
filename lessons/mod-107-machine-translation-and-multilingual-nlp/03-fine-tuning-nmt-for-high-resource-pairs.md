# Fine-Tuning NMT for High-Resource Language Pairs

## Motivation

"High-resource" is a loose term. In practice it means the pair has enough parallel supervision — typically more than a million sentence pairs — that you can fine-tune a pretrained NMT model from scratch on it and expect the fine-tune to move the needle. The recipe below is what you should be able to reproduce in an afternoon whenever a new domain dataset arrives for a mature language pair (English ↔ German, French, Spanish, Chinese, Japanese, and a dozen others). It is deliberately boring — most of the interesting work in production MT happens in the *data pipeline* upstream and the *evaluation* downstream (chapters 10–12), not in the training loop.

## The task

Given a parallel corpus $\{(x_i, y_i)\}$ where $x_i$ is a source sentence and $y_i$ is its human translation, fine-tune $p_\theta(y \mid x, L_s, L_t)$ to maximise

$$\sum_i \sum_t \log p_\theta(y_{i,t} \mid y_{i,<t}, x_i)$$

with teacher-forced cross-entropy. Standard seq2seq, no MT-specific loss required. The complications are all in tokenisation, language-code plumbing, and data hygiene.

## Data preparation: the parts that actually bite

Parallel corpora are usually noisier than the paper reports suggest. Budget a full day on cleaning; the training itself is minutes on top.

### Length and length-ratio filtering

Discard pairs where either side is empty, where either side exceeds ~200 tokens, or where the length ratio (`len(source) / len(target)`) falls outside `[0.5, 2.0]`. Very asymmetric pairs are almost always misaligned sentences from a documnent-level aligner. **Bicleaner** or **Bicleaner AI** (Prompsit) is the standard filter used in WMT participants — worth reaching for if you are dealing with web-scraped data.

### Deduplication

Deduplicate on `(hash(source), hash(target))` and also on source alone. Web-scraped parallel data (Paracrawl, OPUS) can have double-digit duplication rates. Duplicates inflate your training set, bias evaluation, and — worse — often leak into your test split. Deduplicate *before* splitting.

### Sentence-alignment sanity

If the corpus is document-aligned, re-verify with a fast sentence aligner (**Vecalign** with LASER embeddings, or **hunalign**). Feed only aligned sentence pairs to the trainer.

### Test-set contamination

The single most common way to embarrass yourself: your training corpus contains sentences from FLORES-200 or WMT test sets, and your reported BLEU is inflated. Deduplicate your training set against the FLORES `devtest` and any WMT test sets you plan to report against. `datasketch` MinHashLSH at token-shingle level is the standard tool.

### Language-code hygiene

Every training pair should be tagged with a source and target language code in the *exact* format the model expects (see chapter 02). Mixed-code pairs (some rows tagged `de_DE`, some `de`) silently train the model on one and evaluate on the other. Pick a canonical code, convert everything to it, assert at load time.

## Preprocessing: `text_target` and the collator

Both encoder-decoder tokenisers we use (Marian, mBART, NLLB) accept `text_target` for the label side. For mBART / NLLB, the tokeniser also needs a `src_lang` — set it *before* preprocessing.

```python
from transformers import AutoTokenizer

MODEL = "facebook/nllb-200-distilled-600M"
SRC   = "eng_Latn"
TGT   = "deu_Latn"

tok = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC)

MAX_INPUT  = 128
MAX_TARGET = 128

def preprocess(batch):
    inputs = tok(
        batch["src"],
        max_length=MAX_INPUT,
        truncation=True,
        padding=False,
    )
    with tok.as_target_tokenizer():
        labels = tok(
            batch["tgt"],
            max_length=MAX_TARGET,
            truncation=True,
            padding=False,
        )
    inputs["labels"] = labels["input_ids"]
    return inputs
```

Notes:

- **`as_target_tokenizer()`** is the mBART/NLLB idiom that swaps the source-language BOS for the target-language one when tokenising labels. `text_target=...` is the newer API and works for most tokenisers; for NLLB and mBART, use `as_target_tokenizer()` to be explicit.
- **`padding=False`** — let the collator pad dynamically. Wildly variable sentence lengths across languages make static padding wasteful.
- **`max_length=128`** for sentence-level MT. For paragraph MT, raise to 512 and expect longer training time.
- **`DataCollatorForSeq2Seq`** — not `DefaultDataCollator`. It pads inputs and labels independently, replaces label pad tokens with `-100`, and (for NLLB / mBART) manages the language-BOS tokens in `decoder_input_ids`. Using `DefaultDataCollator` produces a silently-wrong loss.

## The workhorse training recipe

Sensible defaults for fine-tuning NLLB or mBART on a ~1 M-pair in-domain corpus with a single A100 or comparable GPU:

```python
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC)

args = Seq2SeqTrainingArguments(
    output_dir="nmt-en-de-domain",
    learning_rate=3e-5,                        # NLLB / mBART fine-tune LR
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,             # effective batch 64
    num_train_epochs=3,
    warmup_ratio=0.06,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    bf16=True,                                 # fp16 on older GPUs
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=make_mt_metrics(tokenizer, tgt_lang=TGT),
)
trainer.train()
```

Where the settings come from:

- **`learning_rate=3e-5`** for NLLB / mBART / Marian fine-tuning. `1e-5` if the fine-tune is destabilising the base model; `5e-5` if you are hitting a plateau and have plenty of data.
- **`warmup_ratio=0.06`** and **`weight_decay=0.01`** are the BART/mBART fine-tuning defaults and remain a solid choice.
- **`label_smoothing_factor=0.1`** is the standard MT setting (Vaswani et al., 2017). It helps almost universally on NMT.
- **`generation_num_beams=4`** — the standard beam width for NMT. Larger widths often hurt (the "beam-search curse", Stahlberg & Byrne, ["On NMT Search Errors and Model Errors"](https://arxiv.org/abs/1908.10090), *EMNLP 2019*).
- **`metric_for_best_model="bleu"`** — chapter 10 argues you should always report chrF and COMET too, but for early-stopping the SacreBLEU number is a reasonable proxy.
- **`bf16=True`** on Ampere+ GPUs; **`fp16=True`** on Volta / Turing. Do not train fp32 unless you are debugging numerical issues.

## `forced_bos_token_id` at training time

For mBART and NLLB, the decoder starts with the target-language BOS token. The trainer must know this so it teacher-forces the right label. `DataCollatorForSeq2Seq` handles it *if* the model config has `decoder_start_token_id` set correctly — which it does for the base checkpoints, but is worth asserting:

```python
tgt_bos = tokenizer.convert_tokens_to_ids(TGT)  # NLLB, e.g. deu_Latn
assert model.config.decoder_start_token_id in (tgt_bos, tokenizer.eos_token_id, None)
model.config.forced_bos_token_id = tgt_bos       # for generate()
```

If you skip the assert and the config has a stale start token, your fine-tune trains a model that translates to the wrong language for the first token and then course-corrects — you will notice at eval time with a mysteriously flat BLEU.

## Compute metrics: BLEU + chrF at training time

Chapter 10 covers the metric stack in depth. For training-loop metrics, wire up **SacreBLEU** (Post, ["A Call for Clarity in Reporting BLEU Scores"](https://arxiv.org/abs/1804.08771), *WMT 2018*) so your numbers are reproducible:

```python
import numpy as np
import evaluate

sacrebleu = evaluate.load("sacrebleu")
chrf      = evaluate.load("chrf")

def make_mt_metrics(tokenizer, tgt_lang):
    def compute(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # SacreBLEU expects list-of-lists on the reference side.
        refs = [[r] for r in decoded_labels]
        return {
            "bleu":  sacrebleu.compute(predictions=decoded_preds, references=refs)["score"],
            "chrf":  chrf.compute(predictions=decoded_preds, references=refs)["score"],
        }
    return compute
```

A few things every tutorial gets wrong:

- **SacreBLEU expects raw detokenised strings.** Do not re-tokenise the predictions or references — SacreBLEU handles this internally and pins the tokeniser version in its signature. Using `nltk.translate.bleu_score.corpus_bleu` on already-tokenised strings gives non-reproducible numbers.
- **Replace `-100` before decoding.** Labels have `-100` where the collator inserted padding; passing those to `tokenizer.batch_decode` raises.
- **Report BLEU and chrF together.** BLEU is fragile across languages (chapter 10); chrF is more robust for morphologically rich or non-space-separated languages.

## Learning-rate schedules and effective batch size

Two things dominate stability:

- **Effective batch size.** MT training benefits from large effective batches — 32 K tokens per step is the fairseq default. Use `gradient_accumulation_steps` aggressively if VRAM is the bottleneck.
- **Warmup.** Skip warmup and the first few hundred steps blow up loss on the multilingual embedding layer. `warmup_ratio=0.06` (about the first 6 % of training) is the safe default.

You will occasionally see the Adafactor optimiser recommended for MT — it saves VRAM at similar quality on the T5 family. For encoder-decoder MT specifically, AdamW with the settings above is the field default.

## What to expect: rough numbers

Rough guidance, not benchmarks:

- **Marian OPUS-MT en-de**, fine-tuned on ~500 k in-domain pairs from a public news corpus (WMT), gains 1–3 chrF over the base checkpoint on WMT `test`. In-domain gains on domain-specific test sets are usually 3–10 chrF.
- **NLLB-200 distilled-600M**, fine-tuned on ~1 M in-domain pairs for the same pair, closes most of the gap to NLLB-3.3B on in-domain test while remaining CPU-serveable.
- **mBART-50 many-to-many**, fine-tuned on a specific pair, often *loses* on other pairs (catastrophic forgetting). Chapter 05 covers mitigations.

Do not chase headline WMT numbers if your production distribution is not WMT. In-domain evaluation is the metric that matters for shipping.

## Inference recipe

The generation counterpart to the trainer above. Chapter 10 wraps this in a full evaluation harness.

```python
translator = build_translator(MODEL, src_lang=SRC, tgt_lang=TGT)  # helper from ch.02
predictions = translator(dev_srcs, num_beams=4, max_new_tokens=128)
```

Add at inference:

- `no_repeat_ngram_size=3` — cheap protection against decoder repetition.
- `length_penalty=1.0` — neutral; tune on dev if the model over- or under-generates.
- `early_stopping=True` — stop when all beams have hit `</s>`.

## Common failure modes and their fixes

- **BLEU plateaus at zero.** You forgot `forced_bos_token_id`, or `src_lang` is wrong on the tokeniser, or the model is emitting `</s>` immediately (`min_new_tokens=2` at generation is a quick sanity check). Inspect a raw prediction — if it's an empty string or the wrong language, it is a code plumbing bug, not a training problem.
- **Loss is silently zero.** You are using `DefaultDataCollator` instead of `DataCollatorForSeq2Seq`. The labels are padded with `0` instead of `-100` and cross-entropy is comparing padding to padding.
- **Repeated phrases in output.** Add `no_repeat_ngram_size=3` at generation. If it persists, your training data has repeated targets (bad data quality).
- **Model regresses on other language pairs after fine-tuning.** You catastrophically-forgot. Chapter 05 has adapters/LoRA and mixed-batch strategies as the mitigation.
- **Non-reproducible BLEU across runs.** You are re-tokenising before scoring. Use `sacrebleu` on raw strings.

## Chapter summary

- The high-resource NMT fine-tune reduces to standard seq2seq cross-entropy with label smoothing, warmup, and `DataCollatorForSeq2Seq`.
- Data cleaning (length filter, deduplication, sentence-alignment sanity, test-contamination) is where the wins and disasters are — the training loop is the boring part.
- Wrap language-code plumbing in a helper (`src_lang`, `forced_bos_token_id`) and assert config invariants — the number-one silent bug is a missing or wrong language code.
- Report SacreBLEU and chrF at training time (not `nltk.translate.bleu_score`); report COMET at final evaluation (chapter 10).
- Rough delta expectation: 3–10 chrF on an in-domain test set from an in-domain fine-tune of NLLB-200 or Marian.
