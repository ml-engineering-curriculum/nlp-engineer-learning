# Multi-Hop QA and Question Decomposition

## Motivation

A single-hop question — "In what year was Barack Obama born?" — can be answered from a single sentence. A multi-hop question — "In what year was the 44th U.S. president born?" — requires *combining* facts: first identify who the 44th U.S. president was, then look up their year of birth. Multi-hop QA is the setting where models most obviously *reason*, and it is the setting where a strong single-hop system silently fails without any change in aggregate metric.

Multi-hop QA also probes something the SQuAD-style formulation cannot: whether the model followed the right reasoning chain, or just happened to guess the right final answer. Two systems can tie on final-answer F1 while one produces defensible chains and the other invents them entirely. Chapter 11 comes back to this in the evaluation rubric; this chapter covers the modelling side.

## The canonical datasets

- **HotpotQA** (Yang et al., ["HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"](https://arxiv.org/abs/1809.09600), *EMNLP 2018*). 113k questions each requiring reasoning over two Wikipedia paragraphs. Two settings: *distractor* (10 paragraphs given, 2 are gold), *fullwiki* (retrieve from all of Wikipedia). Supporting-fact annotations enable chain evaluation.
- **2WikiMultiHopQA** (Ho et al., ["Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps"](https://arxiv.org/abs/2011.01060), *COLING 2020*). Templated multi-hop questions with explicit evidence and reasoning-type labels — bridge, comparison, composition, inference.
- **MuSiQue** (Trivedi et al., ["MuSiQue: Multihop Questions via Single-hop Question Composition"](https://arxiv.org/abs/2108.00573), *TACL 2022*). Multi-hop questions constructed by composing single-hop questions, with each hop verified independently. Harder than HotpotQA for models that were secretly single-hopping.
- **StrategyQA** (Geva et al., ["Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies"](https://arxiv.org/abs/2101.02235), *TACL 2021*). Yes/no questions whose reasoning steps are *implicit* — the decomposition itself must be inferred.

The distractor / fullwiki split in HotpotQA matters. Distractor mode isolates *reader* quality (given the right passages plus distractors, can you reason correctly?). Fullwiki mode measures *end-to-end* quality including retrieval. Report which one you evaluated on; they are not comparable.

## Reasoning types you will encounter

- **Bridge (composition).** *"Who directed the movie based on the novel written by X?"* — Find the novel, find the movie, find the director.
- **Comparison.** *"Which is taller, Everest or K2?"* — Look up each entity, compare a shared attribute.
- **Intersection.** *"Which actor appeared in both movies A and B?"* — Enumerate cast of each, intersect.
- **Set-difference / negation.** *"Which of the following presidents was NOT a general?"* — More common in temporal or set-reasoning benchmarks.

Reasoning type shapes the right approach: bridge questions decompose cleanly into sub-questions; comparison and intersection benefit from a structured prompt or a schema; implicit-strategy questions require the model to invent the decomposition.

## Strategy A: end-to-end reader (long-context or FiD)

Feed all candidate passages to a long-context reader (chapter 07) or a Fusion-in-Decoder model and let the internal attention do the reasoning. Simple, cheap at inference (one forward pass), and increasingly competitive as long-context models improve.

**Strengths.** No pipeline. No decomposition module to maintain.

**Weaknesses.** No explicit reasoning chain — evaluation on *chain* correctness is impossible unless you fine-tune the model to emit citations or supporting-fact tags. Failure modes are hard to debug.

**When to use.** Cost-sensitive production, moderate hop depth (2 hops), evaluations that only care about the final answer.

## Strategy B: iterative retrieve-and-read

An agent-style pipeline:

1. Retrieve passages for the original question.
2. Read: either produce a partial answer or ask a follow-up sub-question.
3. If follow-up, retrieve for the sub-question and go to step 2.
4. Terminate when the model produces a final answer or hits a hop budget.

Concrete instances: IRCoT (Trivedi et al., ["Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"](https://arxiv.org/abs/2212.10509), *ACL 2023*), Self-Ask (Press et al., ["Measuring and Narrowing the Compositionality Gap in Language Models"](https://arxiv.org/abs/2210.03350), *EMNLP 2023 Findings*), ReAct (Yao et al., ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629), *ICLR 2023*).

**Strengths.** Explicit chain, easier to debug and evaluate per-hop. Scales to arbitrary hop depth.

**Weaknesses.** More retrieval calls per question (latency, cost). Error compounding — a wrong first hop poisons subsequent retrievals. Requires a working retriever from the `rag-engineer` track.

**When to use.** 3+ hop questions, open-domain, cases where you need to show reasoning to the user.

## Strategy C: explicit decomposition then solve

Two-stage pipeline: a decomposer LLM turns the multi-hop question into an ordered list of single-hop sub-questions; a solver answers each sub-question in sequence, feeding earlier answers forward.

**Strengths.** Clean separation of concerns. Sub-questions are re-usable as evaluation units — you can measure hop-level accuracy independently. Least-to-most prompting (Zhou et al., ["Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"](https://arxiv.org/abs/2205.10625), *ICLR 2023*) formalises this.

**Weaknesses.** Decomposition failures (missing a hop, ordering hops wrong, generating a non-decomposable sub-question) propagate. Requires prompt engineering discipline on the decomposer.

**When to use.** Enterprise QA over structured knowledge graphs or SQL databases where each sub-question maps to a query. Also strong for MuSiQue-style benchmarks where the compositional structure is explicit.

## Chain-of-thought and self-consistency

Chain-of-thought prompting (Wei et al., 2022) — "let's think step by step" — helps a lot on multi-hop QA. It exposes the model's reasoning and gives you a chain to evaluate, at the cost of longer outputs (higher latency and cost) and a new failure mode: fluent but wrong reasoning that arrives at the wrong answer confidently.

Self-consistency (Wang et al., 2023) — sample $k$ chains and majority-vote on the final answer — reliably improves multi-hop accuracy at the cost of $k$ forward passes. On HotpotQA-scale benchmarks, `k=5` typically buys 3–6 F1 over greedy CoT.

For production multi-hop systems the pragmatic pattern is: retrieve → CoT reader with $k=3$–$5$ samples → majority-vote → keep the winning chain as the "explanation" surfaced to the user.

## Evaluating multi-hop QA properly

The SQuAD F1 protocol (chapter 04) still applies to the final answer. But two additional metrics are important:

- **Supporting-fact F1** (HotpotQA convention). For each question, the annotator marked which sentences contain evidence. Score the model's predicted supporting sentences with precision/recall/F1. A model that produces the right answer without citing the right evidence should not be rewarded the same as one that does.
- **Joint F1** — the joint (answer AND supporting-facts) metric HotpotQA reports. This is the number to lead with when comparing multi-hop systems, because it penalises the specific failure mode where the model gets the answer right by guessing.

For datasets without supporting-fact annotations (StrategyQA, MuSiQue-Ans-only), fall back to answer F1 + a manual chain audit on a sampled subset.

## The specific failure to check for: single-hop shortcuts

A common finding across multi-hop benchmarks: models can achieve high final-answer F1 by *pattern-matching* rather than reasoning. Chen & Durrett, ["Understanding Dataset Design Choices for Multi-hop Reasoning"](https://arxiv.org/abs/1904.12106), *NAACL 2019*, showed that a large fraction of HotpotQA can be solved from a single passage plus surface features of the question. Jiang & Bansal, ["Avoiding Reasoning Shortcuts"](https://arxiv.org/abs/1906.07132), *ACL 2019*, quantified the shortcut phenomenon and proposed adversarial data augmentation.

Practical audits before you trust a multi-hop metric:

1. Report supporting-fact F1 alongside answer F1. If answer F1 is 75 but supporting-fact F1 is 40, the model is guessing.
2. Ablate individual passages at inference and measure the answer-F1 drop. A robust multi-hop model should degrade sharply when either gold passage is removed. A shortcut-taking model degrades only slightly.
3. Evaluate on MuSiQue in addition to HotpotQA. MuSiQue was constructed specifically to close the shortcut gap.

## Chapter summary

- Multi-hop QA requires combining evidence across passages. HotpotQA, 2WikiMultiHopQA, MuSiQue, and StrategyQA are the standard benchmarks; the *distractor* vs. *fullwiki* setting matters.
- Three architectures: end-to-end reader (long-context or FiD), iterative retrieve-and-read (IRCoT, Self-Ask), and explicit decomposition. Cost, chain-visibility, and hop-depth tolerance drive the choice.
- Chain-of-thought plus self-consistency is the dominant prompting pattern for multi-hop with LLMs; the majority-vote chain becomes the user-facing explanation.
- Report supporting-fact F1 and joint F1 alongside answer F1, and audit for single-hop shortcuts — MuSiQue and passage ablations are the standard checks.
