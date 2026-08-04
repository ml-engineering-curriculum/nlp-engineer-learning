# exercise-01: Abstractive Summarisation with BART or T5

**Estimated effort:** 3 hours

## Objective

Fine-tune an encoder-decoder summariser (BART or T5) end-to-end on a real dataset, evaluate it against strong extractive baselines and a *panel* of automatic metrics (ROUGE, BERTScore, faithfulness), and write up the trade-offs. This is the "workhorse recipe" you should be able to reproduce from scratch in an afternoon whenever a new summarisation dataset lands.

## Prerequisites

- Chapters [01](../01-summarisation-and-controlled-generation-landscape.md), [02](../02-extractive-summarisation-baselines.md), [03](../03-abstractive-summarisation-with-bart-t5-pegasus-mt5.md), [10](../10-reference-based-evaluation-rouge-bertscore.md).
- Python 3.10+; `transformers`, `datasets`, `evaluate`, `torch`, `rouge_score`, `bert_score`, `sumy` (for classical extractive baselines).
- A GPU strongly recommended for `-large` variants. CPU + `-base` is workable but slow.

## Dataset

Pick one:

- **CNN/DailyMail** — Hermann et al., ["Teaching Machines to Read and Comprehend"](https://arxiv.org/abs/1506.03340), *NeurIPS 2015*. `datasets.load_dataset("cnn_dailymail", "3.0.0")`. The most-copied summarisation benchmark. Highly extractive by construction; LEAD-3 is famously hard to beat.
- **XSum** — Narayan, Cohen & Lapata, ["Don't Give Me the Details, Just the Summary!"](https://arxiv.org/abs/1808.08745), *EMNLP 2018*. `datasets.load_dataset("EdinburghNLP/xsum")`. Single-sentence "extreme" summaries; the standard stress test for hallucination.
- **SAMSum** — Gliwa et al., ["SAMSum Corpus"](https://arxiv.org/abs/1911.12237), *EMNLP 2019 Workshop*. `datasets.load_dataset("Samsung/samsum")`. Dialogue summarisation; a good "not news" alternative.

Recommendation: **XSum**, because it exposes both the extractive-vs-abstractive gap and the hallucination surface faster than any other public dataset.

## Problem statement

### Part A — Extractive baselines

Before fine-tuning anything, run and score three baselines on the dev set:

- **LEAD-N.** First $N$ sentences of the source. Use $N = 3$ for CNN/DailyMail; $N = 1$ for XSum; $N = 3$ for SAMSum.
- **LexRank.** Use `sumy`'s `LexRankSummarizer` with the same sentence budget as LEAD-N.
- **Oracle-extractive.** Greedy sentence-picker that maximises ROUGE-L against the gold reference — this is the *ceiling* for any extractive system.

Report ROUGE-1 / ROUGE-2 / ROUGE-L (F1) for each on the full dev set.

Save as `baselines.py`.

### Part B — Fine-tune the abstractive model

Fine-tune one of `facebook/bart-base`, `facebook/bart-large`, `google/flan-t5-base`, or `google/pegasus-large` using the recipe from chapter 03:

- `AutoModelForSeq2SeqLM` and `AutoTokenizer`.
- `DataCollatorForSeq2Seq` — not `DefaultDataCollator`.
- Input template: raw source for BART/PEGASUS; `"summarize: " + source` for T5-family.
- LR `3e-5` (BART/PEGASUS) or `1e-4` (T5-family), effective batch `32`, `num_train_epochs = 3`, `warmup_ratio = 0.06`, `weight_decay = 0.01`, `label_smoothing_factor = 0.1` (BART/PEGASUS only).
- `predict_with_generate = True`, `generation_max_length = 128` (or the dataset-typical target length), `generation_num_beams = 4`.
- `bf16 = True` (or `fp16 = True`), `load_best_model_at_end = True`, `metric_for_best_model = "rougeL"`.

Save training logs and the best checkpoint. Ship as `train.py`.

### Part C — Automatic metric panel

Evaluate the fine-tuned model on the dev set with:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** via `evaluate.load("rouge")` with `use_stemmer=True`.
- **BERTScore** via `evaluate.load("bertscore")` with `model_type="microsoft/deberta-xlarge-mnli"` and `rescale_with_baseline=True`.
- **Length statistics** — mean and 90th-percentile output length in tokens, compared against the reference length distribution.

Report all with 95 % bootstrap CIs (1000 resamples over dev examples).

Present as a Markdown table alongside the extractive baselines from Part A.

### Part D — Faithfulness spot check

Even though chapter 11 has the full faithfulness stack, do a lightweight check now:

- For 100 random dev predictions, compute a sentence-level NLI faithfulness score against the source using `microsoft/deberta-v3-large-mnli`. Report the mean entailment probability and the fraction of summaries with mean entailment < 0.5.
- Sample 5 low-entailment predictions and manually inspect. Classify each as "faithful (metric wrong)", "extrinsic hallucination", or "intrinsic hallucination".

Report the counts and paste the 5 examples in the write-up.

### Part E — Decoding-strategy sanity check

Pick 200 dev inputs. Generate summaries with three decoding strategies:

- Greedy (`num_beams=1, do_sample=False`).
- Beam (`num_beams=4, no_repeat_ngram_size=3`).
- Nucleus (`do_sample=True, top_p=0.9, temperature=0.7`).

Report ROUGE-L and mean output length for each. Comment briefly on which strategy is stable, which produces the longest / shortest, and whether any degenerates into repetition.

### Part F — Write-up

A 500–700 word `README.md` covering:

- Dataset, model, hyperparameters.
- The metric table (baselines + fine-tuned + oracle) with CIs.
- The faithfulness spot-check numbers and the manual inspection of 5 low-scoring predictions.
- The decoding-strategy comparison and any observed pathologies.
- One thing you would try next.

## Starter guidance

- **Use `DataCollatorForSeq2Seq`.** `DefaultDataCollator` produces a silently-wrong loss that trains the model to emit padding. This is the bug every seq2seq tutorial gets wrong.
- **Prefix for T5.** `t5-*` and `flan-t5-*` need `"summarize: "` prepended. `bart-*` and `pegasus-*` do not.
- **Tune `max_target_length` per dataset.** CNN/DailyMail summaries are ~60 tokens; XSum are ~20; SAMSum ~30. Truncating targets silently is a common bug — check that <1 % of your training targets are being truncated.
- **Report ROUGE with `use_stemmer=True`.** This is the reproducibility default in the field.
- **BERTScore is expensive.** Batch it (`batch_size=32`). If dev is large, run BERTScore on a random 500-example subset instead of the full dev set.
- **On XSum, expect low ROUGE and high hallucination.** LEAD-1 ROUGE-L ~17; PEGASUS-XSum ROUGE-L ~36 out of the box. Do not chase CNN/DailyMail-style numbers.

## Acceptance criteria

- [ ] `baselines.py` reports LEAD-N, LexRank, and oracle-extractive ROUGE on the dev set.
- [ ] `train.py` fine-tunes an encoder-decoder with the Part B recipe; run logs and the best checkpoint are saved.
- [ ] Metric panel reported (ROUGE-1/2/L, BERTScore, length stats) with 95 % bootstrap CIs on the full dev set.
- [ ] Faithfulness spot-check reported on 100 random predictions; 5 low-scoring predictions manually inspected and classified.
- [ ] Decoding-strategy comparison table (greedy / beam / nucleus) with ROUGE-L and mean output length.
- [ ] 500–700 word write-up including the baseline vs. fine-tuned comparison and one "what next" idea.

## Stretch goals

- **BART vs. PEGASUS vs. FLAN-T5 head-to-head.** Fine-tune all three with matched compute. Which is strongest on your dataset? Which is *most faithful* at fixed ROUGE?
- **Zero-shot baseline.** Before fine-tuning, evaluate `facebook/bart-large-cnn` or `google/pegasus-xsum` zero-shot on your dev set. Report the delta between zero-shot and fine-tuned — how much is the pretraining vs. how much is the fine-tune?
- **Length control.** Add a length control code to the input template (`"<len=short> summarize: ..."`) and train with it. Evaluate whether the model respects the code at inference.
- **BARTScore.** Add BARTScore to the metric panel. Compare its rankings with ROUGE and BERTScore on 20 predictions where they disagree.
- **Ablate `label_smoothing_factor`.** Train with and without `label_smoothing=0.1`. Report the ROUGE delta and any qualitative differences.
