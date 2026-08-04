# Abstractive QA with Encoder-Decoder Models

## Motivation

Extractive QA (chapters 02–04) is the right tool when the answer is a substring of the context. Real user questions often are not:

- *"How much did the plaintiff win?"* — the answer is `$4.2 million`, but the document says `four million two hundred thousand dollars`.
- *"Summarise the plan's medical benefits."* — the answer is a multi-sentence paraphrase across three sections.
- *"Which quarter grew fastest?"* — the answer requires comparison, not extraction.
- *"Why did the migration fail?"* — the answer requires stitching together evidence from a log and a design doc.

For any of these, the reader must be able to *generate* the answer, not merely locate it. That is what abstractive QA is for. This chapter covers the encoder-decoder recipe (T5, BART, FLAN-T5, LongT5); chapter 06 covers the decoder-only closed-book variant.

## The task, formally

You are given $(q, c, a)$ where $a$ is now a *free-form* answer string that need not be a substring of $c$. The model is trained to maximise

$$
\log p_\theta(a \mid \text{template}(q, c))
$$

with standard next-token cross-entropy over the target sequence. The template is text-to-text: a prompt that concatenates $q$ and $c$ into an input string, and a target that is the raw answer string.

For T5-family models the standard template is:

```
input:  question: {q}  context: {c}
target: {a}
```

For BART:

```
input:  {q} </s> {c}
target: {a}
```

There is no separate span head. All the QA-specific structure lives in the training data and the decoding strategy.

## Which model to start with

| Model                          | Params  | Context (train) | Notes                                                                        |
|--------------------------------|---------|-----------------|------------------------------------------------------------------------------|
| `t5-base` / `t5-large`         | 220 M / 770 M | 512           | Reference implementation; classic C4 pretraining.                            |
| `google/flan-t5-base` / `-large` / `-xl` | 250 M / 780 M / 3 B | 512 | Instruction-tuned; better zero-shot QA out of the box (Chung et al., 2022).  |
| `facebook/bart-base` / `-large`| 140 M / 400 M | 1024          | Denoising objective; stronger on abstractive summarisation-style QA.         |
| `google/long-t5-tglobal-base` / `-large` | 250 M / 800 M | 4096 (up to 16 k) | Long-context abstractive QA. See chapter 07.                                 |
| `google/pegasus-large`         | 568 M   | 1024          | Gap-sentence pretraining; strong on QA that resembles summarisation.         |

Practical default for English abstractive QA on modest-length contexts: **FLAN-T5-large**. It has already been instruction-tuned on many QA-style tasks, so it starts closer to a usable answer distribution than raw T5. When context length matters (chapter 07), swap to LongT5.

## Data preparation

Use `AutoTokenizer.__call__` for the input and `text_target=...` for the target (or the older manual pattern with `as_target_tokenizer()`):

```python
def preprocess(examples):
    inputs  = [f"question: {q}  context: {c}"
               for q, c in zip(examples["question"], examples["context"])]
    targets = examples["answer_text"]

    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=64,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

Load with a **`DataCollatorForSeq2Seq`** — it pads inputs and labels independently and replaces the label pad token with `-100` so cross-entropy ignores padding. Using `DefaultDataCollator` here is a common bug that produces silently-wrong loss.

## Training recipe

Encoder-decoder QA fine-tuning defaults:

- Learning rate `1e-4` (T5 family — the paper is written in Adafactor's LR regime) or `3e-5` (BART with AdamW).
- Batch size 8–16, gradient accumulation to reach 32 effective.
- `num_train_epochs = 3` — abstractive QA usually benefits from a third epoch that extractive QA does not.
- `weight_decay = 0.01`, `warmup_ratio = 0.06`.
- `predict_with_generate = True` in `Seq2SeqTrainingArguments`, `generation_max_length = 64`, `generation_num_beams = 4`.
- `label_smoothing_factor = 0.1` sometimes helps abstractive; test on your dev set.

```python
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

args = Seq2SeqTrainingArguments(
    output_dir="abstractive-qa",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.06,
    predict_with_generate=True,
    generation_max_length=64,
    generation_num_beams=4,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
    compute_metrics=make_generation_metrics(tokenizer),
)
trainer.train()
```

## Decoding strategies

At inference time you pick a decoding strategy. QA is not summarisation — the space of correct answers is small — so the strategies that dominate open-ended text generation are usually the wrong default here.

- **Beam search (`num_beams = 4` or `5`).** Best default for factoid QA. Predictable, deterministic (given the model), and biases toward high-probability short answers.
- **Greedy (`num_beams = 1`).** Slightly cheaper, marginally worse on factoid QA. Fine for cost-sensitive deployments.
- **Nucleus / top-p sampling.** Introduces variability that is *bad* for factoid QA. Use only for creative or long-form QA where multiple answers are acceptable.
- **Constrained decoding.** Restrict the output to a lexicon or a schema (e.g., dates in ISO format). See `transformers`' `PrefixConstrainedLogitsProcessor` and Outlines / lm-format-enforcer / Guidance for structured decoding libraries. Chapter 09 uses this for answerability.

Set `no_repeat_ngram_size = 3` to suppress the classic T5 failure mode where the model gets stuck in a repeated phrase, and `length_penalty = 1.0` (higher favours longer answers, lower shorter).

## Evaluation for abstractive QA

The SQuAD normaliser + F1 (chapter 04) *still applies* and remains a useful signal. But because the answer surface can vary, single-reference F1 systematically under-scores correct paraphrases. Practical evaluation stack:

- **SQuAD F1 and EM** — cheap sanity check. Report them but do not treat them as the ceiling.
- **ROUGE-L** (Lin, ["ROUGE: A Package for Automatic Evaluation of Summaries"](https://aclanthology.org/W04-1013/), *ACL 2004 Workshop*). Longest-common-subsequence overlap; captures word-order-tolerant paraphrase better than F1.
- **BERTScore** (Zhang et al., 2020). Embedding cosine similarity. Correlates better with humans on paraphrase, at the cost of a scoring model.
- **BLEURT** (Sellam, Das & Parikh, ["BLEURT: Learning Robust Metrics for Text Generation"](https://arxiv.org/abs/2004.04696), *ACL 2020*). Learned metric fine-tuned on human ratings.
- **LLM-as-judge / answer-equivalence classifier.** Chapter 11.
- **Human evaluation.** Chapter 11.

The right minimum is: SQuAD F1 (for continuity with extractive baselines) + ROUGE-L (for paraphrase tolerance) + one embedding-based or learned metric (BERTScore or BLEURT). Never report only one metric — they disagree, and the disagreement is the interesting signal.

## Hallucination — the new failure mode

Extractive models cannot hallucinate: the worst they can do is pick the wrong span. Abstractive models can *invent*. Concretely, an abstractive QA system trained on SQuAD-style data can produce answers that:

- Contradict the context (fluent, plausible, wrong).
- Blend the context with pretraining priors (adding an unstated fact).
- Answer a *different* question (usually a nearby one).

Two mitigations you should adopt from day one:

- **Faithfulness metric.** Train or borrow a natural-language-inference model and score `entailment(context, predicted_answer)`. Anything below a threshold is a candidate hallucination. See Honovich et al., ["TRUE: Re-evaluating Factual Consistency Evaluation"](https://arxiv.org/abs/2204.04991), *NAACL 2022*.
- **Extractive fallback / re-check.** For questions where an extractive answer exists in the context, run both models and prefer the extractive answer when it disagrees with the generative one. This is a common production pattern for compliance-sensitive domains.

Chapter 11 formalises human rubrics for faithfulness; chapter 09 extends it with abstention thresholds.

## When to prefer abstractive over extractive

Choose abstractive when at least one of these is true:

- The answer requires *combining* information from multiple sentences.
- The natural answer is a *paraphrase* of the source phrasing (units, tense, aggregation).
- The user expects a *complete sentence* rather than a phrase.
- The dataset has *no consistent span* — different annotators highlighted different substrings for the same "correct" answer.

Choose extractive when *any* of these is true:

- The product requires a citation into the source text.
- The domain has zero tolerance for invented content (legal, medical, finance).
- The answer is always a proper noun, a date, or a numeric literal in the source.
- The corpus is too small to fine-tune a generator without severe overfitting; the pretrained encoder + span head is cheaper.

Some production systems run *both* and reconcile — the pattern is often called "generative answer + extractive citation" — where the generative model produces the natural-language answer and the extractive model produces the citing span.

## Chapter summary

- Abstractive QA turns the task into text-to-text: template `(q, c)` into an input, cross-entropy over the answer target, seq2seq decoding at inference.
- FLAN-T5, T5, BART, and LongT5 (chapter 07) are the workhorse encoder-decoders. FLAN-T5-large is a strong default; move to LongT5 when context length matters.
- Use `DataCollatorForSeq2Seq`, beam search with modest beams (4–5), and evaluate with SQuAD F1 *plus* ROUGE-L and an embedding-based or learned metric — never a single number.
- Abstractive models introduce hallucination as a failure mode. Add a faithfulness check (NLI-based) and consider an extractive fallback when the product cannot tolerate invented content.
