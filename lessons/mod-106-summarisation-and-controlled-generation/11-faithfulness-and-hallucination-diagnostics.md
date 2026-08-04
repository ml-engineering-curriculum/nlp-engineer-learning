# Faithfulness and Hallucination Diagnostics

## Motivation

An abstractive summariser can produce a fluent, high-ROUGE, high-BERTScore summary that *contradicts the source*. The classical Maynez et al. (2020) study of XSum systems reported that upward of 60 % of BART/PEGASUS summaries contained at least one hallucinated span, and that ROUGE was essentially uncorrelated with the human faithfulness judgement.

"Faithfulness" (or "factual consistency") is the source-conditioned complement to the reference-based metrics of chapter 10. Where ROUGE asks "does the candidate look like the reference?", faithfulness asks "does the candidate say anything the source does not support?". The two questions are almost independent — a summary can be faithful and lexically distant from the reference, or lexically close and blatantly unfaithful.

This chapter covers the diagnostic side: how to *measure* faithfulness. Chapter 12 covers the mitigation side.

## A working taxonomy of hallucination

The literature is inconsistent on terminology. A useful working split (Maynez et al., 2020; Ji et al., 2023):

- **Intrinsic hallucination.** The output contradicts the source. "The company reported Q3 revenue of $1.4B" when the source says $1.2B. Detectable by an entailment check.
- **Extrinsic hallucination.** The output states something not in the source, which may or may not be true. "AcmeCo's revenue growth was driven by its Frankfurt data centre expansion" when the source is silent on Frankfurt. May be true (in which case it is background knowledge) or false (in which case it is invention). Detectable only by comparison to a *knowledge source*, not to the input document.

The FRANK typology (Pagnoni, Balachandran & Tsvetkov, 2021) refines this further — entity error, predicate error, circumstance error, coreference error, discourse-link error, out-of-source information — and is the vocabulary of choice for error analysis in papers.

In production, the practical binary is:

- **Faithful.** Every claim in the output is supported by the source.
- **Unfaithful.** At least one claim is either contradicted by, or absent from, the source.

Most production faithfulness gates check for the *absence of contradiction* rather than the presence of support — the latter is a much harder inference.

## Method 1: NLI-based faithfulness

Natural Language Inference (NLI) models classify a `(premise, hypothesis)` pair as `entailment`, `neutral`, or `contradiction`. If you use the source as premise and a candidate sentence as hypothesis, the NLI model gives you a faithfulness signal directly.

The naive approach — one call per `(source, whole_summary)` — often produces mostly `neutral` predictions, because whole-summary hypotheses stretch beyond what a single NLI decision can handle. Two refinements make it work in practice:

- **Sentence-level.** Break the summary into sentences; run NLI on `(source, summary_sentence)` for each; aggregate (mean, min, or fraction-entailed).
- **Chunked source, sentence-level candidate.** Run NLI over every `(source_chunk, summary_sentence)` pair; take the maximum entailment probability per summary sentence. This is the SummaC-Conv formulation (Laban et al., ["SummaC"](https://arxiv.org/abs/2111.09525), *TACL 2022*), and it is the current baseline for NLI-based faithfulness.

```python
from summac.model_summac import SummaCConv

model = SummaCConv(models=["vitc"], bins="percentile",
                   granularity="sentence", nli_labels="e",
                   device="cuda", start_file="default", agg="mean")

scores = model.score(documents=[source_text], generated_summaries=[summary])
# scores["scores"][0] ∈ [0, 1] — higher = more faithful.
```

For a hand-rolled version:

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="microsoft/deberta-v3-large-mnli",
               top_k=None, device=0)

def faithfulness(source, summary, threshold=0.5):
    scores = []
    for sent in sent_tokenise(summary):
        result = nli(f"{source} [SEP] {sent}")
        entail = next(r["score"] for r in result if r["label"] == "ENTAILMENT")
        scores.append(entail)
    return sum(s >= threshold for s in scores) / max(len(scores), 1)
```

The choice of NLI model matters: `microsoft/deberta-v3-large-mnli` and `roberta-large-mnli` are the two most-used baselines; ViT-C (Vitamin C — Schuster et al., 2021) is the recommended default for SummaC because it was trained on contrastive-consistency data.

## Method 2: QA-based faithfulness (QAGS, QuestEval, FEQA)

The QA-based family (Wang, Cho & Lewis, ["QAGS"](https://arxiv.org/abs/2004.04228), *ACL 2020*; Durmus, He & Diab, ["FEQA"](https://arxiv.org/abs/2005.03754), *ACL 2020*; Scialom et al., ["QuestEval"](https://arxiv.org/abs/2103.12693), *EMNLP 2021*) reframes faithfulness as a QA game:

1. Generate a set of questions whose answers are drawn from the *summary*.
2. Ask a QA model to answer each question given the *source*.
3. Compare the QA model's answers to the answers from the summary.
4. If the answers match, the summary claim is supported.

```
Summary: "Alice won the tournament with a score of 21."
Question generated: "What was the score of Alice's win?"
Source: "In the finals, Alice defeated Bob 21-18."
QA model on source: "21-18"        # matches → supported.

Summary: "Alice, the reigning champion, won."
Question generated: "Is Alice the reigning champion?"
Source: (does not mention reigning champion status)
QA model on source: (no answer / abstains) → unsupported.
```

QA-based methods correlate with human judgement about as well as SummaC on standard benchmarks (see the TRUE meta-benchmark — Honovich et al., 2022). They are slower — each summary requires question generation and multiple QA calls — but produce interpretable per-claim scores that are useful for error analysis.

## Method 3: Fine-grained atomic-fact evaluation (FactScore)

Min et al., ["FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation"](https://arxiv.org/abs/2305.14251), *EMNLP 2023* decomposes a long generation into atomic factual claims and checks each independently:

1. Decompose the summary into atomic facts ("Alice won", "the tournament was chess", "Alice scored 21", ...).
2. Verify each atomic fact against a knowledge source (either the input document or an external corpus).
3. Report precision as the fraction of supported facts.

FactScore is the go-to for long-form generation where a single per-sentence score is too coarse. Cost: LLM calls for decomposition and verification, so budget accordingly. Useful especially for evaluating LLM-generated summaries of books, meetings, or multi-document briefs.

## Method 4: LLM-as-judge for faithfulness

Ask a strong LLM ("does this summary contradict any statement in the source?" or "list any summary claims not supported by the source"). Convenient, expensive, and biased. See chapter 13 for the rubric and the known issues.

LLM-as-judge is *good enough* for engineering dashboards and A/B tests. It is *not* good enough for a compliance audit or a paper claim without a human-rated subset for calibration.

## Method 5: Learned faithfulness metrics (FactCC, DAE)

- **FactCC** (Kryściński et al., ["Evaluating the Factual Consistency of Abstractive Text Summarization"](https://arxiv.org/abs/1910.12840), *EMNLP 2020*). BERT-based binary classifier trained on synthetically-perturbed summaries (entity swaps, pronoun swaps, sentence deletions). Fast; brittle to real-world hallucination patterns not in the synthetic training set.
- **DAE** (Goyal & Durrett, ["Evaluating Factuality in Generation with Dependency-Level Entailment"](https://arxiv.org/abs/2010.05478), *EMNLP 2020*). NLI at the dependency-arc level; catches predicate-level errors that whole-sentence NLI misses.

Both are cheap to run and useful in combination with SummaC. Do not rely on either alone.

## Choosing a faithfulness metric

The TRUE meta-benchmark (Honovich et al., 2022) evaluates these methods against human-labelled datasets. The current practical recommendation:

| Task                                              | First choice                          | Second-line                          |
|---------------------------------------------------|---------------------------------------|--------------------------------------|
| Sentence-level scoring for a production dashboard | SummaC-Conv with ViT-C NLI            | AlignScore                           |
| Long-form (books, multi-doc)                      | FactScore                             | QAGS with a strong QA model          |
| Error taxonomy / research analysis                | FRANK typology + QAGS / FEQA          | SummaC + DAE                         |
| Quick prototype / triage                          | Hand-rolled sentence-level NLI        | LLM-as-judge with a rubric           |

Report at least two orthogonal metrics — e.g., SummaC + LLM-as-judge, or SummaC + FactScore. Different metrics catch different failure modes; agreement across two independent signals is stronger evidence than a single metric.

## Faithfulness gates in production

A faithfulness gate is a decision rule that either accepts, flags, or rejects a candidate summary based on a faithfulness score. Common patterns:

- **Threshold rejection.** If faithfulness < 0.5, reject and either re-generate with a different decoding strategy or abstain.
- **Rerank.** Generate $N$ candidates, rank by faithfulness × ROUGE, return the top-1.
- **Extractive fallback.** If the abstractive summary fails the gate and an extractive summary exists, return the extractive one.
- **Human review queue.** Below-threshold predictions go to a human. Feasible for low-volume, high-stakes domains (legal, medical); not feasible at web scale.

Choose the pattern by cost tolerance. Rerank at $N = 4$ is cheap and effective for most production settings. Extractive fallback is the compliance default.

## Faithfulness metric limitations

Every automatic faithfulness metric has known blind spots:

- **NLI models miss numerics.** "The company revenue was $1.4B" vs. "$1.2B" is often labelled `neutral` by NLI models because both statements are structurally identical. Add a rule-based numeric-consistency check on top.
- **NLI models over-flag rare-entity claims.** If the model has not seen "AcmeCo" during pretraining, it will label any claim about AcmeCo as `neutral` even when the source supports it. Symptom: high false-negative rate on domain-specific corpora.
- **QA-based methods depend on the QA model's coverage.** If the QA model cannot find the answer in the source when a human easily can, you get spurious "unsupported" flags.
- **LLM-as-judge inherits the judge's biases.** Position bias, length bias, self-preference bias (Wang et al., 2024).

Always calibrate your chosen metric on a small human-labelled subset before trusting the automatic numbers. Report metric-to-human correlation on that subset alongside the aggregate score.

## Reporting faithfulness

The minimum panel for a serious faithfulness evaluation:

- Fraction of summaries with faithfulness score above threshold (e.g., SummaC > 0.5).
- Fraction with score below threshold — the "definite hallucination" rate.
- Distribution of scores (mean, median, quartiles).
- Per-error-type breakdown if you tag a subset with the FRANK typology.
- Correlation with human ratings on a labelled subset.

Report by *decoding strategy* — beam vs. sampling faithfulness rates often diverge — and by *input length* — long-doc summarisation faithfulness typically drops with input length.

## Chapter summary

- Faithfulness is source-conditioned; reference-based metrics (chapter 10) do not measure it. Both must be reported for abstractive summarisers.
- **Hallucination taxonomy:** intrinsic (contradicts source) vs. extrinsic (unstated); the FRANK typology refines this further. Most production gates check for absence of contradiction.
- **Methods.** NLI-based (SummaC), QA-based (QAGS/QuestEval/FEQA), atomic-fact (FactScore), learned classifiers (FactCC, DAE), and LLM-as-judge. Each has blind spots; combine two.
- **Production gates.** Threshold rejection, rerank, extractive fallback, human review — pick by cost tolerance.
- Automatic metrics need calibration against a small human-labelled subset. Report metric-to-human correlation alongside aggregate scores.
- Faithfulness is not something you fix by tuning beam width. Chapter 12 covers the actual mitigation stack.
