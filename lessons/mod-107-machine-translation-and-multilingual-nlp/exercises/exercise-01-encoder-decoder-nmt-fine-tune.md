# exercise-01: Encoder-Decoder NMT Fine-Tune

**Estimated effort:** 3 hours

## Objective

Fine-tune an encoder-decoder NMT model (Marian, mBART-50, or NLLB-200) on a real parallel corpus for a chosen language pair, evaluate it against the pretrained baseline with a proper metric panel (SacreBLEU + chrF + COMET), and report per-domain and per-length breakdowns. This is the "workhorse recipe" you should be able to reproduce from scratch whenever a new parallel dataset lands for a mature language pair.

## Prerequisites

- Chapters [01](../01-machine-translation-and-multilingual-landscape.md), [02](../02-encoder-decoder-nmt-architectures.md), [03](../03-fine-tuning-nmt-for-high-resource-pairs.md), [10](../10-mt-automatic-evaluation-bleu-chrf-comet-bleurt.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `sacrebleu`, `unbabel-comet`, `torch`, `sentencepiece`.
- A GPU strongly recommended (Marian works on CPU; NLLB-distilled-600M fine-tunes comfortably on a single A100/L4/T4).

## Language pair and dataset

Pick one pair with enough parallel data to make fine-tuning meaningful. Two recommended setups:

- **High-resource, general.** English ↔ German on **WMT19 news** (`datasets.load_dataset("wmt19", "de-en")`) — well over 10 M pairs. Base model: `facebook/nllb-200-distilled-600M` or `Helsinki-NLP/opus-mt-en-de`.
- **Mid-resource, in-domain.** English ↔ Swahili on **OPUS-100** (`datasets.load_dataset("Helsinki-NLP/opus-100", "en-sw")`) — ~500 k pairs. Base model: `facebook/nllb-200-distilled-600M`.

Recommendation: **English → German on a downsampled 500 k subset** of WMT19 for a runnable-in-an-afternoon exercise. Report at the end whether you would expect the trends to hold at full scale.

Regardless of pair, the evaluation set must include **FLORES-200 devtest** for the pair (`datasets.load_dataset("facebook/flores", "eng_Latn-deu_Latn", split="devtest")`) so the numbers are comparable across submissions.

## Problem statement

### Part A — Data hygiene

Before touching the model, clean the parallel data:

1. **Length + length-ratio filter.** Drop pairs where either side is empty, exceeds 200 SentencePiece tokens, or whose length ratio falls outside `[0.5, 2.0]`.
2. **Deduplicate.** Deduplicate on `(source, target)` and on source alone.
3. **Contamination check.** N-gram-overlap-check your training split against FLORES-200 devtest for the pair. Report the number of exact-match sentences removed.
4. **Language-code hygiene.** Confirm every pair is tagged with the model's expected code format (`eng_Latn` for NLLB; `en` for Marian).

Report before/after counts in `data_stats.md`. Save the cleaned dataset as `train.jsonl` / `dev.jsonl`.

### Part B — Baseline evaluation

Before fine-tuning, evaluate the pretrained model on FLORES-200 devtest and on your held-out dev split:

- SacreBLEU with the correct tokenizer (`13a` for German/English/most Latin, `zh` for Chinese, `ja-mecab` for Japanese).
- chrF++ (`word_order=2`).
- COMET (`Unbabel/wmt22-comet-da`) with the checkpoint name recorded.

Also record the SacreBLEU signature string and the COMET checkpoint hash.

Save as `baseline.py` + `baseline_report.md`.

### Part C — Fine-tune

Fine-tune using the chapter-03 recipe. Minimum requirements:

- Wrap language-code plumbing in a `build_translator(model, src_lang, tgt_lang)` helper (chapter 02); never hardcode `forced_bos_token_id` at call sites.
- `AutoModelForSeq2SeqLM` and `AutoTokenizer` with `src_lang` set.
- `DataCollatorForSeq2Seq` (not `DefaultDataCollator`).
- LR `3e-5`, effective batch `32`, `num_train_epochs=2` (or as many as fit in 90 min), `warmup_ratio=0.06`, `weight_decay=0.01`, `label_smoothing_factor=0.1`.
- `predict_with_generate=True`, `generation_max_length=128`, `generation_num_beams=4`.
- `bf16=True` (or `fp16=True`), `load_best_model_at_end=True`, `metric_for_best_model` set to `chrf`.

Save training logs, the best checkpoint, and `train.py`.

### Part D — Metric panel with confidence

Score the fine-tuned model on the same evaluation sets from Part B and produce a comparison table with:

- SacreBLEU, chrF++, COMET for baseline and fine-tuned.
- 95 % bootstrap CI over segments (1 000 resamples) for each metric.
- Paired-bootstrap significance (`sacrebleu --paired-bs`) between baseline and fine-tuned for SacreBLEU + chrF.

Present as a Markdown table in `report.md`. Bold any delta whose CI does not cross zero.

### Part E — Slice-level breakdown

Split the FLORES devtest into three roughly-equal length buckets (short: < 15 source tokens, medium: 15–40, long: > 40). Also, if your dataset has a domain field (WMT19 does — news vs. other), split by domain.

Report SacreBLEU / chrF / COMET per slice for the fine-tuned model. Comment on where the fine-tune helps most and where it does not.

### Part F — Write-up

500–700 word `report.md` covering:

- Dataset, model, hyperparameters, training runtime.
- Data-hygiene stats (Part A).
- Baseline vs. fine-tuned metric table with CIs and paired-bootstrap p-values.
- Slice breakdown and interpretation.
- One "what would I try next" (bigger base? more epochs? in-domain LoRA?).

## Starter guidance

- **Use `DataCollatorForSeq2Seq`.** `DefaultDataCollator` silently trains the model to emit padding.
- **NLLB needs the target BOS.** Set `forced_bos_token_id=tok.convert_tokens_to_ids("deu_Latn")` at generation time; without it you often get English out of an English → German model. Wrap in the helper from chapter 02.
- **`src_lang` mutates the tokenizer instance.** If you share the tokenizer across training and eval, set `src_lang` once at construction and treat the tokenizer as read-only.
- **Test-set contamination is common.** WMT training data historically overlapped FLORES source sentences. Always dedupe.
- **COMET is slow.** Batch at `--batch_size 8` on GPU; `16+` if VRAM allows. Cache the reference-encoded features if you evaluate multiple systems.
- **Report the SacreBLEU signature.** Papers and PRs without it are non-reproducible; the exercise will not be accepted without it.
- **Do not report BLEU alone for Chinese/Japanese.** Use `sacrebleu --tokenize zh` (Chinese) or `ja-mecab` (Japanese); prefer chrF or COMET as primary.

## Acceptance criteria

- [ ] `data_stats.md` reports before/after counts for length filter, dedupe, and FLORES contamination check.
- [ ] `baseline.py` produces SacreBLEU + chrF++ + COMET for the pretrained model on FLORES-200 devtest and on your dev split, with signatures and checkpoint names recorded.
- [ ] `train.py` fine-tunes the model with the Part C recipe; run logs and the best checkpoint are saved.
- [ ] Metric-panel comparison table with 95 % bootstrap CIs and paired-bootstrap p-values (SacreBLEU + chrF) between baseline and fine-tuned.
- [ ] Slice-level breakdown by length (short/medium/long) and by domain if available.
- [ ] `report.md` (500–700 words) covers all of the above and one "what next" idea.
- [ ] `build_translator(model, src, tgt)` helper wraps the language-code plumbing; no raw `forced_bos_token_id` at call sites.

## Stretch goals

- **Marian vs. NLLB head-to-head.** Fine-tune `Helsinki-NLP/opus-mt-en-de` and `facebook/nllb-200-distilled-600M` with matched compute. Report which is stronger *in-domain* and which is stronger on FLORES.
- **LoRA vs. full fine-tune.** Add `peft` LoRA (rank 16, `q_proj`, `v_proj`) as a third arm. Report the parameter-count vs. quality trade-off.
- **MBR decoding.** For 200 dev inputs, sample 16 candidates per input (nucleus, `top_p=0.9`), score each with COMET-Kiwi, pick the highest. Compare against `num_beams=4` beam.
- **Third language pair.** Repeat the whole pipeline for a mid-resource pair (`en ↔ sw` or `en ↔ vi`). Compare the deltas.
- **Terminology teaser.** Pick 5 brand names or product terms and evaluate whether the fine-tune preserved them. Foreshadows exercise-02.

## Deliverables

Ship as a directory with:

```
data_stats.md
baseline.py
train.py
metric_panel.py
build_translator.py
report.md
```

Reference solutions live in the paired [`nlp-engineer-solutions`](https://github.com/ml-engineering-curriculum/nlp-engineer-solutions) repository.
