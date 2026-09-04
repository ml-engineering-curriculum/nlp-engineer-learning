# Extractive QA Metrics: SQuAD F1 and EM

## Motivation

Extractive QA has a stable community-standard metric — SQuAD F1 and Exact Match with the official normaliser — and yet the reported numbers routinely diverge because implementations diverge on details. A homegrown SQuAD scorer will differ from `evaluate.load("squad")` by 1–3 F1 points, usually because it drops the multi-reference `max`, skips the SQuAD normaliser, or scores subword strings instead of the raw context slice.

This chapter defines the SQuAD protocol crisply, extends it to SQuAD 2.0 (where "no answer" is a valid prediction), extends it further to the multilingual SQuAD variants (XQuAD, MLQA, TyDi QA) where the normaliser is language-specific, and covers what to do when the reference set is too tight to capture answer equivalence. mod-105 chapter 04 is the training-side reference for extractive QA; this chapter is the evaluation-side authority.

## SQuAD F1 and EM — the metric

Both metrics are per-question and macro-averaged over the dataset.

**Exact Match**:

$$\text{EM}(p, R) = \max_{r \in R} \mathbb{1}\bigl[\text{normalize}(p) = \text{normalize}(r)\bigr]$$

**Token-level F1**:

$$\text{F1}(p, r) = \frac{2 \cdot \text{overlap}(p, r)}{|\text{normalize}(p)| + |\text{normalize}(r)|}, \quad \text{F1}(p, R) = \max_{r \in R} \text{F1}(p, r)$$

where `overlap` is the bag-of-tokens intersection count. Dataset-level scores are the arithmetic mean over questions. The `max` over the reference set $R$ is the multi-reference protocol from Rajpurkar et al. (["SQuAD: 100,000+ Questions for Machine Comprehension of Text"](https://arxiv.org/abs/1606.05250), *EMNLP 2016*): SQuAD dev and test questions carry multiple crowd-collected references and *any* match counts. Using only `references[0]` is a common bug that silently lowers reported EM by 2–5 points.

## The SQuAD normaliser

Verbatim from the official `evaluate.py` and reproduced in `evaluate.load("squad")`:

1. **Lowercase.**
2. **Strip ASCII punctuation** (`string.punctuation`).
3. **Remove leading articles** `a`, `an`, `the` (as whole words, anywhere).
4. **Collapse whitespace.**

```python
import re, string

def squad_normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s
```

`"Barack H. Obama"` and `"the barack h obama"` both normalise to `"barack h obama"` and count as an exact match. `"1998"` and `"1,998"` both normalise to `"1998"` — the normaliser is a strong regulariser of surface form.

## SQuAD 2.0: adding "no answer"

SQuAD 2.0 (Rajpurkar, Jia, Liang, ["Know What You Don't Know"](https://arxiv.org/abs/1806.03822), *ACL 2018*) adds unanswerable questions to the mix. The reference for unanswerable is `{"text": [], "answer_start": []}`; the correct prediction is the empty string.

Two evaluation extensions matter:

- **`HasAns` vs `NoAns` split.** Report EM/F1 on answerable-only questions (`HasAns_exact`, `HasAns_f1`) and accuracy on unanswerable-only questions (`NoAns_exact`, treated as EM), then the overall aggregate. A model can look middling in aggregate while being great at answering and terrible at abstaining, or vice versa; the split is diagnostic.
- **Best-threshold F1.** Predictions carry a `no_answer_probability` (or a `best_non_null_score` and a `null_score` — the delta is the threshold). The `squad_v2` metric sweeps thresholds and reports the best achievable F1 alongside the F1 at your emitted threshold. Difference between the two tells you whether your model is calibrated for the abstain decision or just picked a bad threshold.

```python
import evaluate

squad_v2 = evaluate.load("squad_v2")
predictions = [
    {"id": qid, "prediction_text": pred, "no_answer_probability": p_none}
    for qid, pred, p_none in preds
]
references = [{"id": qid, "answers": ex["answers"]} for qid, ex in gold.items()]
result = squad_v2.compute(predictions=predictions, references=references)
# {"exact": 82.1, "f1": 85.3,
#  "HasAns_exact": 78.0, "HasAns_f1": 85.6,
#  "NoAns_exact": 86.0,
#  "best_exact": 83.2, "best_f1": 86.4, "best_exact_thresh": 0.5, ...}
```

Report the four numbers. `best_f1 − f1` above ~0.5 is a signal that your calibration is off.

## Multilingual variants

The SQuAD protocol travels to other languages via three main benchmarks; each has its own normaliser you must use.

- **XQuAD** (Artetxe, Ruder, Yogatama, ["On the Cross-lingual Transferability of Monolingual Representations"](https://arxiv.org/abs/1910.11856), *ACL 2020*) — 11 languages, translations of SQuAD 1.1 dev. Uses a per-language article-stripping list.
- **MLQA** (Lewis et al., ["MLQA: Evaluating Cross-lingual Extractive QA"](https://arxiv.org/abs/1910.07475), *ACL 2020*) — 7 languages, aligned test sets. Ships its own scorer with per-language normalisers (`mlqa/mlqa_evaluation_v1.py`). Do not use the English SQuAD normaliser on MLQA — the F1 will be systematically wrong on languages where the article rule doesn't apply.
- **TyDi QA** (Clark et al., ["TyDi QA: A Benchmark for Information-Seeking QA in Typologically Diverse Languages"](https://arxiv.org/abs/2003.05002), *TACL 2020*) — 11 typologically diverse languages, information-seeking questions. Uses its own scorer with per-language token-normalisation (Chinese and Japanese are character-tokenised for F1; Thai and Korean have specific rules).

The general rule: for non-English extractive QA, use the benchmark's own scorer. Do not paste the English SQuAD normaliser on the assumption that "F1 is F1." The article step, the tokenisation step, and the whitespace step all interact with the target language.

## Common failure modes and their fixes

- **Whitespace in the predicted span.** Post-processing sometimes emits a leading or trailing space from the offset map. The SQuAD normaliser strips it, but downstream string comparisons will fail.
- **Sub-word artefacts.** Decoding token IDs (`tokenizer.decode`) instead of slicing the original context by `offset_mapping` can introduce `##` fragments, lose spaces, or lower-case when the model was cased. Always slice the raw context.
- **`references[0]` bug.** SQuAD dev has multiple references per question. Use all of them; that is the point of the `max`.
- **Unanswerable on SQuAD 1.1.** An empty prediction on a SQuAD 1.1 question is a wrong answer. If your model can abstain, evaluate on SQuAD 2.0 (or evaluate the answerable subset only).
- **Language-inappropriate normaliser.** The English SQuAD normaliser on a French or Japanese eval set drops the wrong things. Use the MLQA / TyDi / XQuAD scorer.
- **Aggregation direction.** SQuAD is macro-averaged over questions. Do not micro-average token overlaps across the dataset — that over-weights long-answer questions.
- **Cross-implementation drift.** A homegrown scorer is almost certainly wrong on one of the above. Use `evaluate.load("squad")` or `evaluate.load("squad_v2")`.

## Per-slice reporting for SQuAD-family evaluation

Aggregate EM/F1 hides most of what a QA team needs to fix. Standard slices:

- **Per question type.** Extract the first interrogative word (who / what / when / where / why / how / other). `why` and `how` are systematically harder than `who` and `when`.
- **Per answer length bucket.** 1-token, 2–5, 6+. Long answers are harder and are under-represented in the aggregate mean.
- **Per context length.** Under 100 tokens vs. sliding-window cases. If long-context F1 is much lower, the post-processor is dropping the right span across a chunk boundary — a bug you can fix.
- **Per topic / domain.** If you built your own test set, break out by source domain (Wikipedia biography vs. legal vs. medical).
- **HasAns vs. NoAns for SQuAD 2.0.** Non-optional.
- **Best-threshold delta.** `best_f1 − f1_at_your_threshold` per slice — where the delta is large, calibration is the problem, not extraction.

## When SQuAD F1 is not enough

The SQuAD normaliser is generous on surface variation but blind to semantic equivalence. `"the capital of France"` and `"Paris"` are both correct answers to "What is Paris?" and both zero to each other under SQuAD F1. Three escalations:

- **BERTScore or a learned metric.** For abstractive-flavoured extractive QA (extract-and-rephrase pipelines), BERTScore-F1 gives partial credit for paraphrase. Report *alongside* SQuAD F1, not instead of.
- **Answer-equivalence classifier.** Bulian et al. (["Tomayto, Tomahto"](https://arxiv.org/abs/2202.07654), *EMNLP 2022*) train a small model to judge whether two answer strings are equivalent given the question. Corrects paraphrase misses; costs a model to trust.
- **LLM-as-judge.** Prompt a strong LLM with `(question, gold_answer, predicted_answer)` and ask for a yes/no equivalence. Cheap and reasonably accurate for open-domain QA; non-deterministic and biased toward fluent outputs. Chapter 09 covers LLM-as-judge as a general pattern.

For SQuAD-style extractive QA — where the answer is a substring of a supplied context — EM and F1 with the official normaliser remain the right defaults precisely because they are cheap, deterministic, and comparable across the literature. Use the richer metrics as *additional* signal on paraphrase-heavy cases, not as replacements.

## MRQA and beyond

The **MRQA 2019 shared task** (Fisch et al., ["MRQA 2019 Shared Task: Evaluating Generalization in Reading Comprehension"](https://arxiv.org/abs/1910.09753), *MRQA workshop*) collates 18 extractive-QA datasets into a common SQuAD-format schema, so you can score cross-dataset generalisation with a single scoring function. If you are evaluating an extractive-QA model for out-of-distribution robustness, MRQA is the canonical benchmark and its scorer inherits SQuAD's normaliser.

## Chapter summary

- SQuAD F1 and EM: per-question, `max` over the reference set, macro-averaged over the dataset. Always use the official normaliser (lowercase, strip punctuation, drop `a/an/the`, collapse whitespace) and always use all references.
- SQuAD 2.0 adds `HasAns` / `NoAns` splits and best-threshold F1. Report all four: overall EM/F1, answerable-only F1, unanswerable accuracy, best-threshold F1. `best_f1 − f1` diagnoses calibration.
- Multilingual variants (XQuAD, MLQA, TyDi) have language-specific normalisers — use the benchmark's own scorer, not the English SQuAD normaliser.
- Always slice the original context by `offset_mapping` for the predicted span. Never decode subword IDs, or you introduce artefacts the normaliser cannot rescue.
- Report per question type, per answer length, per context length, and (for SQuAD 2.0) per HasAns/NoAns. Bootstrap CIs on all metrics.
- Escalate to BERTScore, an answer-equivalence classifier, or LLM-as-judge only when paraphrase equivalence is the failure mode you need to measure — as *additional* signal, not replacements for SQuAD F1.
