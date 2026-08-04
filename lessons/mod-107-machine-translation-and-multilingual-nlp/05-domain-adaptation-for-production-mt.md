# Domain Adaptation for Production MT

## Motivation

A base NMT model trained on web text will translate "the plaintiff filed a motion in limine" as "the plaintiff filed a motion in limine" only if it has seen legal English before. Feed the same sentence to a general-purpose Marian or NLLB and you may get "the plaintiff filed a motion in a limousine" (real error, real vendor). Domain adaptation is the collection of techniques that shifts a general MT model into a *specific* distribution — legal, medical, financial, software, biomedical, subtitles — where the vocabulary, register, and syntax differ enough that the general model underperforms noticeably.

This chapter covers the four techniques that actually ship in production MT: **continued fine-tuning**, **adapter / LoRA-based adaptation**, **mixed-batch fine-tuning to fight catastrophic forgetting**, and **retrieval-augmented adaptation** (fuzzy-match TM injection). Terminology injection — the other big production lever — gets its own chapter (06).

## What "domain" means for MT

Three axes of shift matter, and they need different mitigations:

- **Lexical shift.** In-domain vocabulary the base model does not know or translates wrong. "Non-Hodgkin lymphoma" translated word-by-word instead of as a term. → Terminology injection (ch. 06) and in-domain fine-tuning.
- **Register / stylistic shift.** Legal English is verbose and Latinate; social-media English is elliptical. UI microcopy is imperative and short. → Fine-tuning on in-domain (source, target) pairs.
- **Structural shift.** In-domain sentences differ in length, punctuation, structure. Software localisation strings have placeholders (`{user_name}`, `%s`); subtitles have line-break constraints; legal documents have citation formats. → Preprocessing (placeholder round-tripping), format constraints, and sometimes structural fine-tunes.

A production MT system usually needs all three levers. The stack in this chapter targets 1 and 2; chapter 06 targets the terminology aspect of 1; the structural aspect of 3 is usually handled at the data-pipeline level (chapter 10 of mod-110-nlp-data-engineering).

## Technique 1 — Continued fine-tuning ("domain-tune")

The most straightforward technique: take a strong general-purpose model and continue training on in-domain parallel data.

```python
# Base: NLLB-200 distilled 600M fine-tuned on WMT en→de. In-domain: 100k legal pairs.
args = Seq2SeqTrainingArguments(
    output_dir="nllb-en-de-legal",
    learning_rate=1e-5,                # LOWER than the general fine-tune (3e-5)
    per_device_train_batch_size=16,
    num_train_epochs=2,                # FEWER epochs than general fine-tune
    warmup_ratio=0.1,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="chrf",       # in-domain evaluation set
    save_total_limit=2,
)
```

Two changes vs. the chapter 03 recipe:

- **Lower learning rate (`1e-5`).** You are refining an already-fine-tuned model. A higher LR overwrites too much of the general knowledge.
- **Fewer epochs.** In-domain corpora are small; 2 epochs usually suffices. Watch for the eval-loss uptick that signals overfitting; early-stop hard.

**Watch for catastrophic forgetting.** If you evaluate the domain-tuned model on *general* text (WMT, FLORES), quality usually drops by 2–10 chrF. That is fine if the domain-tuned model only ever sees in-domain input; it is not fine if it serves the general endpoint. Mitigations below.

## Technique 2 — Mixed-batch fine-tuning

To keep general performance while gaining domain performance, mix in a fraction of the general training data with each batch. A common recipe:

- Every training batch is 50 % in-domain, 50 % general (or 70/30 for domain-heavy setups).
- Same learning rate as the pure domain-tune, or slightly lower.

```python
from datasets import concatenate_datasets, interleave_datasets

in_domain = load_dataset("your/legal-en-de")["train"]
general   = load_dataset("wmt19", "de-en")["train"].shuffle(seed=42).select(range(500_000))

mixed = interleave_datasets(
    [in_domain, general],
    probabilities=[0.5, 0.5],     # 50/50; try 0.7/0.3 too
    seed=42,
    stopping_strategy="all_exhausted",
)
```

Chu et al., ["An Empirical Comparison of Domain Adaptation Methods for Neural Machine Translation"](https://arxiv.org/abs/1701.03214) (*ACL 2017*) is the reference paper — mixed fine-tuning was the winner across their setups. Two decades later it still is.

## Technique 3 — Parameter-efficient adaptation: adapters and LoRA

The mid-2020s production default is *not* full fine-tuning per domain. Full fine-tuning gives you N base-model-sized checkpoints per N domains; parameter-efficient techniques give you one base + N small deltas.

### Adapter modules

Adapter modules — small bottleneck feed-forward layers inserted after each transformer block — were introduced for NMT specifically by Bapna & Firat, ["Simple, Scalable Adaptation for Neural Machine Translation"](https://arxiv.org/abs/1909.08478) (*EMNLP 2019*). The base model is frozen; only the adapter weights train. Each domain gets its own ~2 % adapter overlay.

### LoRA

LoRA (Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), *ICLR 2022*) is the modern generalisation — replaces adapter modules with a low-rank update $W \leftarrow W + BA$ on attention projections. Cheaper than adapters (no forward-path overhead once merged), and the `peft` library makes it drop-in for NLLB / mBART:

```python
from peft import LoraConfig, get_peft_model, TaskType

base = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
lora = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base, lora)
model.print_trainable_parameters()
# trainable params: 5,373,952 || all params: 620M || trainable%: 0.87
```

Then train with the same `Seq2SeqTrainer` you would use for full fine-tuning. Storage per domain: one ~20 MB `.safetensors` file instead of a 2.4 GB checkpoint.

**Serving.** Two patterns:

1. **Merged inference.** For each request, load the domain LoRA into the base weights (`peft` provides `merge_and_unload()`), serve, then unload. High latency per domain switch.
2. **Per-request LoRA.** vLLM and TGI support hot-swapping LoRA adapters per request. Latency overhead is small; storage overhead is trivial. This is the pattern that made per-domain adaptation viable at production scale.

### When to reach for which

- **10 k–200 k in-domain pairs, single domain.** Full continued fine-tune is fine.
- **Multiple domains, want to serve from one base.** LoRA per domain.
- **Very small in-domain (< 10 k pairs).** LoRA — full fine-tune will overfit.
- **On-device or memory-constrained.** LoRA — you can distribute per-domain 20 MB deltas over the wire.

## Technique 4 — Retrieval-augmented MT (fuzzy-match TM injection)

Translation memories (TMs) — human-produced (source, target) pairs from prior translations — are the standard tool of the localisation industry. Even a general-purpose NMT model benefits from being *shown* the closest TM matches at inference time.

The **kNN-MT** family (Khandelwal et al., ["Nearest Neighbor Machine Translation"](https://arxiv.org/abs/2010.00710), *ICLR 2021*) formalises this: at each decoding step, retrieve $k$ nearest-neighbour target tokens from a datastore of prior (source-context, target-token) pairs and interpolate with the model's output distribution. In practice, three simpler patterns cover most production usage:

- **Prepend the top TM match to the input.** `"REF: <source_match> ||| <target_match> ||| SRC: <source>"` — the model uses the reference like a few-shot example. Cheap; works with any encoder-decoder.
- **Concatenate multiple matches in a prompt template.** Same idea, more context. Watch for context-length inflation.
- **kNN-MT proper.** Requires a datastore build (FAISS over decoder hidden states) and an interpolated decoding loop. Best quality when the TM is large and high-overlap with the input distribution; heavier infrastructure.

For most production systems, the prompt-prepend pattern is what you should try first. TM libraries (Trados, MemoQ) already store the closest matches; you just have to wire them into the request.

## Domain-adaptive pretraining

Before fine-tuning, you can also continue *pretraining* the model on monolingual in-domain text with the model's original objective (denoising for mBART, span-corruption for T5/mT5). This is expensive — you need enough in-domain monolingual data to move the pretraining loss meaningfully — but pays off when the domain vocabulary is far from the base model's pretraining distribution (biomedical text, legal documents in a specific jurisdiction, technical manuals).

The pattern:

1. **Domain-adaptive pretraining** on monolingual in-domain corpora (both languages independently if multilingual).
2. **Continued fine-tuning** on in-domain parallel data.
3. **Task-specific fine-tuning** on the final production distribution.

Gururangan et al., ["Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"](https://arxiv.org/abs/2004.10964) (*ACL 2020*) formalised this recipe for classification; the same shape works for MT.

## Handling placeholders and markup: the boring but critical part

Software localisation, subtitles, and support-ticket translation all inject non-linguistic tokens into the source: `{user_name}`, `%s`, `<b>bold</b>`, `[[link]]`, `\n`. A raw NMT model will translate these into local text, capitalise them, or drop them. Two patterns work:

- **Placeholder masking.** Before translation, replace each placeholder with a canonical unique token (`__PH_1__`, `__PH_2__`). After translation, put them back. Requires that the tokeniser preserve `__PH_1__` as one piece — either add them as special tokens or design a token that survives the model's SentencePiece.
- **Tag-aware training.** Fine-tune with placeholders in place; add a training-time loss that penalises dropping or duplicating them. Heavier but higher quality.

For HTML/XML: use a tag-preserving preprocessor (Moses `wrap-xml.pl` is the classical tool; newer options in the [`sacrerebleu`](https://github.com/mjpost/sacrebleu) and [`slugit`](https://github.com/openoffice/translate-toolkit) ecosystems) that splits tags out, translates the text, and reattaches. The alternative — hoping the NMT model handles the tags — routinely fails on nested markup.

## Evaluating domain adaptation

- **Report on the *in-domain* test set as primary.** This is what you shipped for.
- **Also report on the *general* baseline (FLORES, WMT).** This is your forgetting-monitor. If general chrF drops by more than 3–5, either your mixed-batch ratio is too aggressive or your LoRA rank is too high.
- **Terminology accuracy is a first-class metric.** Chapter 06 covers term-match accuracy; report it alongside chrF for any domain where terminology matters.
- **Human evaluation on 100 in-domain sentences.** Especially for legal / medical, automatic metrics can pass while a subject-matter expert flags severe errors.

## Common failure modes and their fixes

- **In-domain chrF up, general chrF down 10.** Catastrophic forgetting from pure domain fine-tune. Switch to mixed-batch fine-tune, or move to LoRA and serve base + domain-specific overlay.
- **LoRA fine-tune plateaus below full fine-tune.** Raise rank (8 → 16 → 32). Add more target modules (currently only `q_proj, v_proj`? Add `k_proj, out_proj`). If it still lags, the domain shift is large enough that you need full fine-tuning.
- **Placeholders come back malformed.** Placeholder masking upstream is missing or the mask token got split by the tokeniser. Verify with `tokenizer.tokenize("__PH_1__")` — should be one token.
- **TM injection makes output *worse*.** The retrieved matches are low-similarity or off-domain. Add a fuzzy-match score threshold — only inject matches above 70 % similarity.
- **Model translates domain terms word-by-word.** Domain fine-tune alone did not lift the vocabulary — chapter 06's terminology injection is the fix.

## Chapter summary

- Domain adaptation for production MT stacks four techniques: continued fine-tuning, mixed-batch training against forgetting, parameter-efficient (LoRA / adapter) fine-tunes, and retrieval-augmented decoding with translation memories.
- Continued fine-tune uses a *lower* learning rate (`1e-5`) and fewer epochs than the general recipe; watch for eval-loss uptick from overfitting.
- Mixed-batch fine-tuning is the classical fix for catastrophic forgetting. LoRA overlays are the modern fix — one base checkpoint + per-domain 20 MB deltas served per request via vLLM / TGI.
- Domain-adaptive pretraining on monolingual in-domain text is worth it when the domain vocabulary is far from the base model's pretraining distribution.
- Placeholders and markup are handled by masking (round-tripping) or tag-aware training — not by trusting the model to leave them alone.
- Report on *both* in-domain and general test sets; general chrF is the forgetting monitor.
