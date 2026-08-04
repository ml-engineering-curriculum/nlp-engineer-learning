# Human Evaluation Rubrics for QA

## Motivation

Chapter 04 established EM and F1 as the reference metrics for extractive QA. Chapter 05 layered ROUGE-L, BERTScore, and BLEURT for the abstractive setting. Chapter 08 added supporting-fact and joint F1 for multi-hop reasoning. Chapter 09 added risk-coverage AUC for abstention. Yet on real user queries — non-factoid questions, long-form answers, retrieval-augmented responses over private corpora — *none* of these numbers is sufficient to decide whether the system is good enough to ship.

Two reasons. First, most production QA has no gold-answer set: users ask things nobody has ever asked before. Second, even where a gold answer exists, correctness is often multi-dimensional — a legally correct answer might be phrased confusingly, or a fluent answer might be subtly wrong on a fact the user cannot verify. Reference-based metrics collapse those dimensions into a single number.

This chapter is the rubric side of QA evaluation: how to define what "good" means, how to measure it with humans, and how to approximate that measurement cheaply with LLM-as-judge.

## When reference-based metrics stop being enough

Concretely, reach for rubric-based evaluation when at least one of the following holds:

- The answer form is *long-form* (a paragraph, not a phrase).
- The corpus of possible answers is open-ended.
- The user's success criteria include *how* the answer was said — tone, hedging, sourcing.
- The domain is high-stakes and a fluent-but-wrong answer must count as a hard failure.
- The system has no ground-truth gold answers (fresh production traffic).

The rubric replaces "does the string match?" with "does the answer satisfy the following criteria?"

## The core rubric dimensions

Below is a canonical rubric for QA. Every production evaluation I have seen is a variation on it — extend or trim per product, but resist the urge to invent new dimensions before you have measured the ones below.

- **Correctness.** Is the answer factually right? For open questions, is it consistent with authoritative sources?
- **Faithfulness / groundedness.** For retrieval-augmented answers: are the claims in the answer supported by the retrieved passages? A correct answer with no support in the retrieval is *unfaithful* and should be flagged even when correct.
- **Completeness.** Does the answer address the whole question, including any sub-parts? Partial answers are a common failure mode.
- **Relevance.** Did the model answer the *asked* question, or a nearby easier one?
- **Citation quality.** Are the cited sources the ones that actually support the answer? Do they exist? Are they authoritative for the claim?
- **Clarity / fluency.** Is the answer well-written for the audience?
- **Calibration / abstention.** When the model was uncertain, did it hedge or abstain appropriately? An overconfident wrong answer is worse than a correctly hedged one.
- **Safety / policy.** Did the answer respect content and access-control policies?

For a given product you will likely elevate two or three of these to "must pass" (typically Correctness, Faithfulness, and Safety) and treat the rest as "should score high on average".

## Rating scales that work

Two conventions dominate:

- **Binary rubric (pass / fail per dimension).** Best for high-stakes deployments and for LLM-as-judge (LLMs calibrate binary decisions better than fine-grained scales). Report per-dimension pass rate.
- **Likert 1–5 per dimension.** Better resolution when you are comparing near-tied systems. Report per-dimension mean and inter-annotator agreement (Krippendorff's $\alpha$ or Cohen's $\kappa$).

Avoid ordinal 1–7 or 1–10 scales; the fine granularity is illusory and inter-annotator agreement collapses. Also avoid mixing scales inside one rubric — humans anchor differently and the numbers stop being comparable.

## Sampling: what to actually rate

You cannot rate everything. Sampling policy is a design decision:

- **Random uniform sample** of production queries per week. Baseline for tracking regressions. Sample size: at least 100 per version to detect a 5 percentage-point pass-rate change with modest confidence.
- **Stratified sample** by question type or intent. Prevents rare-but-important categories (legal, medical) from being under-sampled.
- **Adversarial / red-team sample.** Prompts designed to trigger known failure modes. Never a substitute for random; a *complement* that tells you what your worst case looks like.
- **Regression fixtures.** A hand-curated set of questions that must always pass. Small (30–100), stable, and used as a merge-blocking test.

Every rubric-based evaluation should include at least the random and regression samples. Adversarial and stratified are added as the product matures.

## Inter-annotator agreement — the sanity check nobody runs

Before trusting an evaluation, measure whether your annotators agree with each other. Have two annotators independently rate the same 50–100 samples. Compute:

- **Cohen's $\kappa$** for binary rubrics per dimension. > 0.7 is workable, > 0.8 is good.
- **Krippendorff's $\alpha$** for Likert scales. Same thresholds.
- **Pairwise agreement** by dimension. Dimensions with low agreement need a clearer definition or better examples.

Low agreement is not "annotators are bad" — it is "the rubric is under-specified" or "the task itself is subjective". Both are your problem to fix before the numbers mean anything.

## LLM-as-judge — the cost-effective approximation

Human evaluation is slow and expensive. Using a strong LLM to score answers against a rubric — LLM-as-judge — is now standard practice for at-scale QA evaluation. The reference framing is Zheng et al., ["Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685), *NeurIPS 2023*.

The recipe:

1. Write a *judge prompt* that describes the task, the answer, the rubric dimension, and the required output format (binary or scale).
2. Feed the judge model `(question, [context], predicted_answer, [reference_answer])` for the dimension being scored.
3. Parse the judge's output and aggregate over the eval set.

Two failure modes to design against from the start:

- **Judge bias.** LLM judges systematically prefer longer, more confidently phrased, or better-formatted answers, even when they are wrong (positional-bias, verbosity-bias, self-preference-bias — Zheng et al., 2023). Randomise answer order in pairwise comparisons; check calibration against human labels on a subsample.
- **Judge miscalibration.** LLM judges are more lenient than humans in aggregate (Wang et al., ["Large Language Models are not Fair Evaluators"](https://arxiv.org/abs/2305.17926), *ACL 2024 Findings*). Never report LLM-judge pass rate as if it were a human pass rate; always ground it against a periodic human audit.

The pragmatic pattern for a shipping team:

- **LLM-as-judge on 100 % of eval traffic** for tracking week-over-week regressions.
- **Human evaluation on a stratified 5–10 % sample** for calibration against the LLM judge.
- **When LLM-judge and human disagree > 5 percentage points on a dimension, retune the judge prompt or downgrade that dimension to human-only.**

RAGAS (Es et al., 2024) is a widely used LLM-as-judge library for retrieval-augmented QA specifically — Faithfulness, Answer Relevance, Context Precision, Context Recall. Use it as scaffolding for the RAG-specific dimensions; still evaluate Correctness and Safety separately with humans.

## Pairwise vs. pointwise scoring

Pointwise ("rate this answer 1–5 on correctness") is what the rubric above describes. Pairwise ("which of these two answers is better?") is the other common protocol.

Pairwise wins when you are *comparing systems* — model A vs. model B on the same queries. Elo ratings from pairwise comparisons power Chatbot Arena and are highly reliable for relative ranking. Pointwise wins when you are *measuring absolute quality* — did we hit the 95 % faithfulness target?

Reach for pairwise when landing a model change; reach for pointwise when reporting the shipped system's quality to a product stakeholder.

## Building an evaluation set that outlasts models

The evaluation set is often the most valuable artefact of a QA project — models come and go, but a good eval set catches regressions from every future change. Invest in it accordingly:

- **Version it.** `evaluation-set-v1.jsonl`, `evaluation-set-v2.jsonl`, with a changelog. When you add or remove samples, cut a new version.
- **Annotate it richly.** Every sample carries the rubric answers, the reasoning behind them, and the reviewer. Regressions are easier to argue about when the intent is captured.
- **Track it in the same repo as the code.** Not a Google sheet, not a wiki. A JSONL file under source control, so the diff shows up in code review.
- **Include a "hard" subset.** The 20 % of questions your current model gets wrong. Ship the model when its behaviour on that subset improves; ignore aggregate wins on the easy 80 %.

## Chapter summary

- Reference-based metrics (EM, F1, ROUGE, BLEURT) stop being enough for long-form, open-ended, high-stakes, or reference-free QA. Rubric-based evaluation replaces them.
- The canonical rubric dimensions are correctness, faithfulness, completeness, relevance, citation quality, clarity, calibration, and safety. Elevate two or three to "must pass" per product.
- Use binary or Likert 1–5 scales, sample randomly plus stratified plus regression fixtures, and always measure inter-annotator agreement before trusting the numbers.
- LLM-as-judge scales rubric evaluation but is biased and miscalibrated by default — pair it with a periodic human audit and track judge-vs-human divergence per dimension.
- Version, richly annotate, and source-control the evaluation set. It will outlast several model generations and is often the highest-leverage artefact your QA team owns.
