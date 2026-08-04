# exercise-02: N-gram Language Model and Perplexity

**Estimated effort:** 2 hours

## Objective

Train two n-gram language models on the same corpus — one with a "toy" smoothing scheme you implement, one with modified Kneser-Ney via KenLM — then report perplexity on a held-out set, characterise their behaviour, and use one of them as a lightweight anomaly detector for structured text. Confirm empirically that classical LM smoothing is the difference between a curiosity and a shippable component.

## Prerequisites

- Chapter [05](../05-ngram-language-models-and-perplexity.md).
- Python 3.10+; `kenlm` (`pip install https://github.com/kpu/kenlm/archive/master.zip`); a copy of KenLM's `lmplz` and `build_binary` binaries on PATH (build from source or install a system package).
- ~50-500 MB of text in one language of your choice.

## Problem statement

### Part A — MLE and add-`k` from scratch

Implement, in ≤150 lines of Python:

- `train_ngram_mle(sentences: Iterable[list[str]], n: int) -> dict[tuple[str,...], dict[str, float]]` — MLE bigram or trigram estimates.
- `train_ngram_addk(sentences, n, k=1.0, vocab_size) -> dict[...]` — add-`k` smoothed estimates.
- `perplexity(model, sentences, n) -> float` — corpus perplexity, handling `<s>` boundary padding and `</s>` end-of-sentence per chapter 05.

Requirements:

- Handle unseen words explicitly. Report per-corpus OOV rate.
- Handle zero-probability contexts (MLE) by returning `float("inf")` and record the count.

Report:

- Perplexity of MLE on train and test.
- Perplexity of add-1 and add-0.01 on test.
- OOV rate on test.

Do **not** implement Kneser-Ney by hand; that is the point of Part B.

### Part B — KenLM modified Kneser-Ney

Train a 3-gram and a 5-gram KenLM model on the same corpus:

```bash
lmplz -o 3 < train.txt > model3.arpa
lmplz -o 5 < train.txt > model5.arpa
build_binary -q 8 model5.arpa model5.bin
```

Report:

- Perplexity of each KenLM model on the same held-out set as Part A. Confirm it dramatically beats the toy models.
- Model file size on disk for each `.bin`.
- Query throughput (sentences/sec) with `kenlm.Model.score` on a fixed batch of 10 000 sentences.

### Part C — LM as an anomaly detector

Pick a *structured* text stream — server access logs, DNS query names, code snippets in one language, JSON payloads with a single schema — where you can distinguish "normal" from "anomalous" records without ambiguity. Train a KenLM model on the normal stream.

Then:

- Score a held-out normal set and a labelled anomalous set (or synthesise anomalies: injections, typos, code from a different language).
- Report the ROC curve (or precision/recall at a chosen threshold) of "per-token log-probability" as an anomaly score.
- Discuss what perplexity cannot catch: give at least one class of anomaly that your LM misses and explain why.

## Starter guidance

- Do the tokenisation *once* and cache. Comparing perplexity across models requires identical tokenisation and identical vocabulary.
- Include `<s>` at the start of each sentence for `n > 1`; count `</s>` as one token in the denominator for perplexity, consistent across all models.
- KenLM expects one sentence per line. Ensure sentence segmentation is deterministic (use ICU per chapter 04, not `str.split`).
- For anomaly detection, use `model.full_scores(sentence, bos=True, eos=True)` and examine the per-token `(log_prob, ngram_length, oov)` triples — this is where the debugging happens.
- If your Part A implementation is slower than 30 seconds on 100k sentences, you are doing something suspicious. Profile before adding "just one more Counter".

## Acceptance criteria

- [ ] All three MLE / add-1 / add-0.01 numbers reported on train and held-out, plus OOV rate.
- [ ] KenLM 3-gram and 5-gram perplexities reported, model file sizes, and query throughput.
- [ ] Perplexity gap between hand-implemented smoothing and KenLM stated explicitly; the direction and rough magnitude should match chapter 05's exposition (KenLM lower by a wide margin).
- [ ] Anomaly-detection experiment: chosen corpus described, anomalous set constructed, ROC or precision/recall reported, and one class of anomaly the LM misses is discussed with reasoning.
- [ ] A short write-up (`README.md`) states which of the four LMs (MLE, add-1, add-0.01, KenLM) you would actually ship for the anomaly-detection use case, and why.

## Stretch goals

- **Character-level LM.** Retrain your models over characters (or bytes) instead of tokens. Compare perplexity, model size, and OOV behaviour. Discuss where character-level wins.
- **Interpolation with a neural LM.** Score the same held-out set with a small pretrained transformer LM (e.g., `gpt2` via `transformers`) and combine log-probs with the n-gram model (linear interpolation on log-prob at 0.5). Does the combined perplexity beat either alone on your domain?
- **Perplexity as a domain classifier.** Train one KenLM per domain (news, tweets, code) and classify a document by lowest per-token perplexity. Compare to fastText classification on the same three domains.
- **Pruning ablation.** Train KenLM with and without `--prune`. Report the perplexity / model-size trade-off.
