# Closed-Book QA with Decoder-Only LLMs

## Motivation

The extractive and encoder-decoder recipes both assume a context $c$ is supplied at inference time. A decoder-only LLM in the "closed-book" setting has no context — only the question — and must answer from its parameters alone. This is what happens when you type a question into a chat interface and the model does not consult any external tool.

Closed-book QA is worth understanding in this module for three reasons: it is *what a raw LLM does by default*, so every retrieval-augmented system silently falls back to it whenever retrieval misses; it is the natural home for "common knowledge" questions where a corpus lookup is overkill; and it is the setting where hallucination is easiest to produce and hardest to detect. Roberts, Raffel and Shazeer's ["How Much Knowledge Can You Pack Into the Parameters of a Language Model?"](https://arxiv.org/abs/2002.08910) (*EMNLP 2020*) is the reference framing.

## The formulation

Given only a question $q$, produce an answer $a$:

$$
a = \arg\max_a p_\theta(a \mid \text{prompt}(q))
$$

There is no separate training objective — the LLM was already trained on general text and, for instruction-tuned variants, on QA-shaped conversations. The knob you control is the *prompt*.

Concretely you author a prompt template and use the same generation infrastructure as chapter 05:

```
prompt = f"""Answer the following question concisely.
If you do not know the answer, say "I don't know."

Question: {q}
Answer:"""
```

Zero-shot works out of the box on any modern instruction-tuned decoder (Llama 3.1 Instruct, Mistral Instruct, Qwen Instruct, GPT-4-class API models). Few-shot — providing several example `(question, answer)` pairs in the prompt — usually adds 3–10 EM points, especially on formatting-sensitive answers (dates, IDs, numbers).

## Prompt patterns that transfer

- **Instructions before content.** Put the task description at the top; put the question at the bottom near the answer marker. Instructions after content are more likely to be ignored (Liu et al., ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172), *TACL 2024*).
- **Explicit answer marker.** End the prompt with `Answer:` (or `A:`, or `Response:`) so the model has a clear place to start. Without it, some models will restate the question.
- **Format specification.** If you need `YYYY-MM-DD`, say so. Most instruction-tuned models will comply on the first try.
- **Explicit abstention licence.** Chapter 09 goes deeper, but even in closed-book you should authorise the model to abstain ("If you are not sure, say 'I don't know.'"). Without it, an LLM will hallucinate to please.
- **Few-shot exemplars for format.** Two or three `Q:… A:…` exemplars pin the answer format without teaching new facts.

Chain-of-thought (Wei et al., ["Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903), *NeurIPS 2022*) — asking the model to reason step by step before answering — helps on questions that require arithmetic or multi-step reasoning. It hurts on simple factoid questions by giving the model more surface to hallucinate on. Reserve it for chapter 08's multi-hop setting.

## Evaluation is the same, but the failure modes shift

Use the SQuAD-style metrics (EM, F1) from chapter 04 as a baseline, with the caveats from chapter 05 (ROUGE-L, embedding-based, LLM-as-judge). What changes is *what* you are measuring — and what you should be worried about.

- **Recall of parametric knowledge.** Does the model know the fact at all? Benchmarks: TriviaQA closed-book (Joshi et al., 2017), Natural Questions closed-book (Kwiatkowski et al., 2019), MMLU (Hendrycks et al., ["Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300), *ICLR 2021*).
- **Freshness.** Was the fact true at pretraining cut-off? Was it true when the user asked? A model trained in 2023 will confidently give 2023-era answers about the current CEO of a company that changed leadership in 2024.
- **Hallucination rate.** How often does the model produce a confident, plausible, wrong answer? Benchmarks: TruthfulQA (Lin, Hilton & Evans, ["TruthfulQA: Measuring How Models Mimic Human Falsehoods"](https://arxiv.org/abs/2109.07958), *ACL 2022*).
- **Calibration.** When the model *says* it is confident, is it right? Almost always no, by default (Kadavath et al., ["Language Models (Mostly) Know What They Know"](https://arxiv.org/abs/2207.05221), 2022, is a starting reference).
- **Abstention rate.** How often does the model correctly say "I don't know"? Given the abstention licence above, this should be non-zero. If it is zero, the model is over-confident and needs post-hoc calibration.

## Confidence signals from a decoder-only model

A closed-book LLM gives you three signals you can use to estimate answer confidence, from cheapest to most expensive:

1. **Per-token log-probabilities.** If the API exposes them (many do; `logprobs` in OpenAI-style APIs, `output_scores=True` in HF `generate`), sum the log-probs of the answer tokens and threshold. Simple; correlates weakly with correctness for hard questions.
2. **Self-consistency** (Wang et al., ["Self-Consistency Improves Chain of Thought Reasoning"](https://arxiv.org/abs/2203.11171), *ICLR 2023*). Sample $k$ answers with temperature > 0 and take the majority. Agreement across samples is a stronger correctness signal than any single log-prob.
3. **Self-critique / verification.** Prompt the model to check its own answer against the question (or against retrieved evidence, moving toward RAG). Costs an extra call but often catches obvious hallucinations.

None of these is a substitute for retrieval when factual accuracy matters. They are a way to *decide when to retrieve*, which is one of the harder engineering problems in QA systems today.

## When closed-book QA is the right choice

Closed-book is the right choice when *all* of these hold:

- The question is about widely known, stable facts (basic geography, arithmetic, common-sense reasoning, popular culture up to the model's cut-off).
- The user does not need a citation.
- The cost of a rare wrong answer is acceptable (or an abstention licence is in place).
- Latency and cost of retrieval outweigh the accuracy benefit (e.g., a chit-chat interface).

It is the *wrong* choice when any of these hold:

- The question is about your company's private data, a specific customer, or anything post-cut-off.
- The user expects provenance.
- The domain is legal, medical, financial, or otherwise low-tolerance for hallucination.
- Model updates and knowledge updates need to be decoupled (retrieval lets you fix a fact without retraining).

For everything in the second list, use retrieval-augmented generation — the reader/generator side of which is the abstractive recipe in chapter 05, wired to a retriever from the `rag-engineer` track (chapter 10).

## Closed-book fine-tuning — briefly

You *can* fine-tune a decoder-only model to be a better closed-book QA system on a target distribution (Roberts et al., 2020). The recipe is standard causal-LM fine-tuning:

- Template `Q: {q}\nA: {a}\n\n` (or your preferred instruction-tuning format).
- Loss on the answer tokens only (mask `-100` on the prompt).
- LoRA/QLoRA (Hu et al., ["LoRA"](https://arxiv.org/abs/2106.09685), *ICLR 2022*; Dettmers et al., ["QLoRA"](https://arxiv.org/abs/2305.14314), *NeurIPS 2023*) if full fine-tuning is out of budget.

Two caveats. First, cramming facts into weights is data-inefficient — a single fact costs you many training examples' worth of gradient updates, and the model still often gets it wrong. Second, fine-tuning teaches the model to be *more confident on the training distribution*, which increases hallucination when you push it off-distribution. Prefer retrieval-augmented generation whenever the facts have any hope of living in a corpus you control.

## Chapter summary

- Closed-book QA is generation from parameters alone: no context, only prompt. It is what every decoder-only LLM does by default.
- Prompt patterns matter — instruction first, question last, explicit abstention licence, format specification. Chain-of-thought helps on reasoning, hurts on simple factoid.
- Evaluation is SQuAD-style metrics *plus* freshness, hallucination, and calibration signals. Self-consistency and log-prob thresholds are cheap confidence signals; retrieval is the real fix.
- Reach for closed-book on stable, public, no-citation questions. Reach for retrieval-augmented (chapter 10 boundary) on private, fresh, or provenance-required questions.
