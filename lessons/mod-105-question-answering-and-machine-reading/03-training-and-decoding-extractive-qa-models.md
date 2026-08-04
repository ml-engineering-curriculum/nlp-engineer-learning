# Training and Decoding Extractive QA Models

## Motivation

Chapter 02 defined the extractive QA formulation. This chapter is the "workhorse recipe" — the end-to-end training script you paste, tune, and reuse whenever a new extractive-QA dataset lands on your desk. The Hugging Face `run_qa.py` example is our reference implementation; we walk through the pieces that matter, the ones that always need to be re-tuned, and the ones you can leave alone.

The recipe has three parts: **preprocessing** (align characters to tokens across a sliding window), **training** (fine-tune an encoder with a two-headed span classifier), and **decoding** (turn per-chunk logits into an answer string). Each part has a small number of load-bearing decisions that dominate the final metric.

## The head: two linear layers on the encoder

`AutoModelForQuestionAnswering` wraps any HF encoder and adds a `nn.Linear(hidden_size, 2)`. The two output columns are the start and end logits per token. That is the entire architectural change over the base encoder — no CRF, no attention over positions, nothing exotic. All the machine-reading intelligence lives in the pretrained encoder; the head is a thin classifier.

Loss is:

$$
\mathcal{L} = \tfrac{1}{2}\bigl(\text{CE}(\text{start\_logits}, y_\text{start}) + \text{CE}(\text{end\_logits}, y_\text{end})\bigr)
$$

with `y_start, y_end` clamped to `[0, sequence_length - 1]`. Positions marked with `start_positions = end_positions = 0` (CLS) count as "no answer" and produce a loss gradient on the CLS logit; on SQuAD 1.1 this only happens for sliding-window chunks that lose the answer, but the same mechanism is what SQuAD 2.0 relies on for genuine unanswerable questions.

## Which encoder to start with

Pick based on domain, budget, and language(s):

| Encoder                            | Params  | Strengths                                                                     | When to reach for it                            |
|------------------------------------|---------|-------------------------------------------------------------------------------|-------------------------------------------------|
| `bert-base-uncased`                | 110 M   | Reference implementation; matches every paper.                                | Reproducing published numbers.                  |
| `roberta-base` / `roberta-large`   | 125 M / 355 M | Consistent SQuAD improvements over BERT with the same recipe.            | Default English extractive-QA starting point.   |
| `microsoft/deberta-v3-base` / `-large` | 184 M / 435 M | State-of-the-art among encoder-only readers of comparable size.         | Best "just works" quality-per-parameter today.  |
| `xlm-roberta-base` / `-large`      | 278 M / 559 M | Multilingual; strong on XQuAD, MLQA, TyDi.                                  | Cross-lingual extractive QA.                    |
| `allenai/longformer-base-4096`     | 149 M   | 4k-token context via sliding-window attention.                                | Long single-document QA. See chapter 07.        |
| Domain encoders (`SciBERT`, `BioClinicalBERT`, `LEGAL-BERT`) | ~110 M | Pretrained on in-domain text.                                                 | Domain corpora where general pretraining underperforms. |

Start small (`roberta-base` or `deberta-v3-base`) so you can iterate on preprocessing and evaluation quickly; only scale up once the metric-improvement-per-hour curve is clearly positive.

## Hyperparameters that transfer

The following defaults come from Devlin et al. (2019) and have held up across the encoder families above. They are a good starting point; treat them as priors, not constants.

- Learning rate `3e-5` (BERT-base), `2e-5` or `1e-5` for `-large` variants.
- Batch size 12–32 per GPU; use gradient accumulation to reach an effective batch of 32.
- `num_train_epochs = 2` — SQuAD converges fast; a third epoch usually helps by ≤ 0.3 F1 and can start memorising.
- `warmup_ratio = 0.1`, `weight_decay = 0.01`, linear LR schedule.
- `max_seq_length = 384`, `doc_stride = 128` (the paper defaults; every framework echoes them).
- `fp16 = True` on Ampere+; `bf16 = True` on Hopper. Halves memory with no metric cost.
- Freeze token type embeddings on RoBERTa (`token_type_ids` are not learned) — the `AutoTokenizer` handles this for you if you pass in the tokenised pair correctly.

## The training loop, minus the boilerplate

Using `transformers.Trainer`:

```python
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model     = AutoModelForQuestionAnswering.from_pretrained("microsoft/deberta-v3-base")

# `preprocess_train` and `preprocess_valid` follow chapter 02's sliding-window recipe.
train_ds = raw_train.map(preprocess_train, batched=True, remove_columns=raw_train.column_names)
valid_ds = raw_valid.map(preprocess_valid, batched=True, remove_columns=raw_valid.column_names)

args = TrainingArguments(
    output_dir="qa-run",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DefaultDataCollator(),
    tokenizer=tokenizer,
    # compute_metrics uses the SQuAD post-processor (chapter 04)
    compute_metrics=make_squad_compute_metrics(valid_ds, raw_valid),
)
trainer.train()
```

Two things worth calling out:

- The validation dataset here is the *tokenised* one (chunked), but `compute_metrics` also needs `raw_valid` (one row per question) so it can aggregate chunk logits back to a single answer per question. Passing both through a closure or `functools.partial` is the standard pattern.
- `DefaultDataCollator` (not `DataCollatorForTokenClassification` — that is chapter 04 of the NER module) is correct here because the labels are two integers per example, not per-token tags.

## Decoding: `postprocess_qa_predictions` in detail

The reference implementation lives in `transformers/examples/pytorch/question-answering/utils_qa.py`. In pseudocode:

```
for each original example:
    candidates = []
    for each tokenised chunk of that example:
        for start_idx in top_k(start_logits, k=20):
            for end_idx in top_k(end_logits, k=20):
                if start_idx not in context: continue
                if end_idx not in context: continue
                if end_idx < start_idx: continue
                if end_idx - start_idx + 1 > max_answer_length: continue
                score = start_logits[start_idx] + end_logits[end_idx]
                span_text = context[offset_map[start_idx][0] : offset_map[end_idx][1]]
                candidates.append((span_text, score))
    prediction = candidates[argmax(score)]
```

The `k=20` top-k prune is what turns an $O(L^2)$ search per chunk into an $O(k^2)$ one. Setting `k=1` (pure argmax) drops SQuAD F1 by roughly a point because it forbids the model from recovering when the best start and best end disagree slightly.

`max_answer_length = 30` matches SQuAD 1.1's distribution; longer answers exist but are rare. For domains with long extractive answers (legal clauses, medical summaries) bump it to 60–100.

## The SQuAD 2.0 null-score adjustment

SQuAD 2.0 uses the same architecture but adds a null-score comparison during decoding:

```
null_score = start_logits[CLS] + end_logits[CLS]
best_non_null_score = max(candidate_scores)
if best_non_null_score - null_score < threshold:
    predict ""  # unanswerable
else:
    predict best_non_null_span
```

The `threshold` is tuned on the dev set — chapter 09 shows how. For SQuAD 1.1 you set `threshold = -inf` (always emit a non-null span) and the mechanism is quiet.

## Sanity-checking a fresh training run

Before you invest a full training run, verify with a 100-example smoke test:

1. **Preprocess and print.** Print 10 tokenised examples with subword tokens, `sequence_ids`, and the substring `context[offset_map[start_pos][0]:offset_map[end_pos][1]]`. It should exactly equal the gold `answer_text`.
2. **Overfit a tiny subset.** Train on 100 examples for 20 epochs with a high LR. Training loss should approach zero. If it does not, your loss or label alignment is wrong.
3. **First epoch on the full data.** After one epoch on SQuAD 1.1, `deberta-v3-base` should hit F1 ≥ 85 on dev. If you see F1 < 60, the post-processor is broken or the sliding window is dropping answers.

Any of these three checks fails, stop and fix — pushing forward on a broken setup burns compute without teaching you anything.

## When the workhorse is not enough

Reach for one of the extensions later in the module when:

- Contexts do not fit in a sliding window of a 512-token model → chapter 07 (Longformer, LongT5, Fusion-in-Decoder).
- Questions require reasoning over multiple passages → chapter 08 (multi-hop).
- Many questions are unanswerable → chapter 09 (SQuAD 2.0, calibration).
- The answer is not literally a span (paraphrase, aggregation, summary) → chapter 05 (abstractive) or chapter 06 (decoder-only closed-book / instruction-tuned).

## Chapter summary

- The extractive-QA head is one linear layer producing start and end logits per token; the loss is averaged cross-entropy on both.
- Defaults from Devlin et al. (2019) — LR `3e-5`, batch 32, 2 epochs, `max_seq_length = 384`, `doc_stride = 128` — transfer across every encoder we recommend.
- `postprocess_qa_predictions` (top-*k* over start and end, filtered by context region and max length, aggregated across chunks) is the decoder — bugs here are invisible in training loss.
- Always smoke-test with a 10-example alignment print, a 100-example overfit, and a 1-epoch dev-F1 check before running full training.
