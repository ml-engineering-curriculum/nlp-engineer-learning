# EM and F1: The SQuAD Evaluation Protocol

## Motivation

Exact-match (EM) and token-level F1 are the metrics you will see on every extractive-QA leaderboard, in every paper, and in every internal evaluation dashboard. They are also the metrics most likely to be *reported wrong*: applied with the wrong normaliser, macro-averaged incorrectly, or computed against a single reference when multiple were provided. A model that beats a strong baseline by 3 F1 under one implementation can *lose* by 1 F1 under another, purely because of these details.

This chapter defines the SQuAD evaluation protocol as it appears in the official `evaluate.py` script (Rajpurkar et al., 2016) and codifies the pitfalls. We use it directly for SQuAD 1.1 and 2.0, and we build on it in chapter 11 when reference answers stop being sufficient.

## Exact Match (EM)

Exact match is a binary score per question:

$$
\text{EM}(p, R) = \max_{r \in R} \mathbb{1}\bigl[\text{normalize}(p) = \text{normalize}(r)\bigr]
$$

where $p$ is the predicted answer string, $R = \{r_1, \dots, r_k\}$ is the set of reference answers for the question, and $\text{normalize}(\cdot)$ is the SQuAD text normaliser (below). The dataset-level EM is the mean over all questions.

The `max` over $R$ is important: SQuAD dev/test questions often have 3–5 references collected from different crowd workers, and *any* match counts. Using only `references[0]` is a common bug that silently lowers reported EM by 2–5 points.

## Token-level F1

F1 is the harmonic mean of precision and recall computed at the *word-token* level between the predicted answer and each reference. Formally:

$$
\text{F1}(p, r) = \frac{2 \cdot \text{overlap}(p, r)}{|p| + |r|}
$$

where $|p|$ and $|r|$ are the numbers of normalised word tokens in the prediction and reference, and $\text{overlap}(p, r)$ is the count of matching tokens treating each side as a bag (multiset). The dataset-level F1 is:

$$
\text{F1}(p, R) = \max_{r \in R} \text{F1}(p, r), \qquad \text{F1}_\text{dataset} = \frac{1}{N} \sum_i \text{F1}(p_i, R_i)
$$

Again, `max` over references and then macro-average over questions. The macro-average is the SQuAD convention; do not micro-average or you will over-weight questions with longer answers.

## The SQuAD normaliser

The normaliser is spelled out in `squad_evaluate.py` (Rajpurkar et al., 2016) and reproduced verbatim in `evaluate.load("squad")` and `evaluate.load("squad_v2")`. In order:

1. **Lowercase.** `"United States" → "united states"`.
2. **Remove punctuation.** All ASCII punctuation stripped (`string.punctuation`).
3. **Remove articles.** The words `a`, `an`, `the` are deleted anywhere in the string.
4. **Collapse whitespace.** Runs of whitespace → single space; leading/trailing space stripped.

```python
import re, string

def squad_normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s
```

This is intentionally aggressive. `"Barack H. Obama"` and `"the barack h obama"` both normalise to `"barack h obama"` and count as an exact match. It also means EM is not a strict spelling check: `"1998"` and `"1,998"` both normalise to `"1998"`.

For non-English SQuAD variants (XQuAD, MLQA, TyDi) the article-stripping step is *language-specific* and often skipped entirely — MLQA ships its own normaliser per language. Do not paste the English SQuAD normaliser onto a Japanese eval set and call it done.

## Using `evaluate` in practice

```python
import evaluate

squad_metric = evaluate.load("squad")   # or "squad_v2" for SQuAD 2.0

predictions = [{"id": qid, "prediction_text": pred} for qid, pred in preds.items()]
references  = [{"id": qid, "answers": example["answers"]} for qid, example in gold.items()]

result = squad_metric.compute(predictions=predictions, references=references)
# {"exact_match": 87.3, "f1": 93.1}
```

Two implementation details worth calling out:

- SQuAD 1.1 answers come in a `{"text": [...], "answer_start": [...]}` dict. Every string in `text` is a valid reference for the `max` over $R$.
- SQuAD 2.0 encodes an unanswerable question as `{"text": [], "answer_start": []}`. `predictions` must include a `no_answer_probability` field for SQuAD 2.0; the `squad_v2` metric uses it to compute the best F1 achievable across threshold sweeps.

## Micro pitfalls that cost F1 points

- **Whitespace in the predicted span.** Post-processing sometimes emits a leading or trailing space from the offset map. The SQuAD normaliser will strip it, but a downstream string comparison against the raw context will fail.
- **Sub-word artefacts.** If you concatenate subword tokens by decoding token IDs (`tokenizer.decode`) instead of slicing the original context by `offset_mapping`, you can introduce `"##"` fragments or lose spaces. Always slice the original context.
- **Multiple gold spans.** SQuAD dev has multiple references per question; use them all. SQuAD train has one, and that is normal.
- **Article ambiguity in languages other than English.** `"un chat"` (fr) normalises differently under the English normaliser and can silently drop French articles you meant to keep.
- **Unanswerable question scoring on SQuAD 1.1.** If your model emits an empty string on any SQuAD 1.1 question, it counts as a wrong answer (EM = 0, F1 = 0). Under SQuAD 2.0 it can be correct — check which metric you loaded.

## Per-slice reporting

The single number is a poor summary. When you report SQuAD-family results, break out at minimum:

- **Overall EM and F1** with 95% bootstrap confidence intervals (resample the questions with replacement 1000 times, take the 2.5th and 97.5th percentiles).
- **Per question-type F1.** SQuAD questions fall into who / what / when / where / why / how / other — use the first interrogative word as a proxy. `"why"` questions are systematically the hardest.
- **Per answer-length bucket.** 1-token answers, 2–5 tokens, 6+ tokens. Short answers are easier and dominate the average.
- **Per context-length bucket.** Under 100 tokens vs. sliding-window cases. If your F1 is much lower on long contexts, the post-processor is dropping the right span across a chunk boundary.
- **For SQuAD 2.0:** report answerable-only F1 and unanswerable-only F1 (accuracy on "no answer" predictions) separately, plus the best-threshold F1 across the sweep. A model that is great on answerable questions but terrible at abstaining looks middling in the aggregate.

## The alternatives — and why SQuAD F1 is still the default

- **BLEU / ROUGE-L / METEOR.** Designed for summarisation and translation; they reward *n-gram* overlap, not answer correctness. Too generous on plausible but wrong answers.
- **BERTScore** (Zhang et al., ["BERTScore: Evaluating Text Generation with BERT"](https://arxiv.org/abs/1904.09675), *ICLR 2020*). Embedding-based; useful for abstractive QA (chapter 05) but overkill for extractive.
- **Answer-equivalence classifiers.** A learned "are these two answers equivalent?" model (e.g., Bulian et al., ["Tomayto, Tomahto"](https://arxiv.org/abs/2202.07654), *EMNLP 2022*). Corrects paraphrase misses, at the cost of an extra model to trust and maintain.
- **LLM-as-judge.** Chapter 11 treats this in detail. Correct for open-ended QA; overkill and non-deterministic for SQuAD-style extractive QA.

For SQuAD-style extractive QA, EM and F1 with the official normaliser remain the right defaults precisely because they are cheap, deterministic, and comparable across the entire literature. Use richer metrics as *additional* signal, not as replacements.

## Chapter summary

- EM is a normalised string match; F1 is token-overlap. Both are computed against a *set* of references with `max`, then macro-averaged over questions.
- The SQuAD normaliser (lowercase, strip punctuation, drop `a/an/the`, collapse whitespace) is deliberately aggressive and language-specific to English. Use language-appropriate normalisers on non-English variants.
- Always slice the original context by `offset_mapping` for the predicted answer string — never decode subword IDs, or you introduce artefacts that the normaliser cannot rescue.
- Report EM and F1 with bootstrap CIs and slice by question type, answer length, and context length. A single-number leaderboard entry hides the failure modes you actually need to fix.
