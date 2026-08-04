# exercise-03: Long-Context and Multi-Hop QA

**Estimated effort:** 3 hours

## Objective

Build a QA system that must reason over evidence longer than a 512-token window *and* combine information from multiple passages. Compare a sliding-window extractive baseline against a long-context reader and (optionally) a chain-of-thought decoder-only LLM. Evaluate not just final-answer F1 but also supporting-fact F1 and depth-of-evidence stratification, so that "aggregate wins" that come from single-hop shortcuts are exposed.

## Prerequisites

- Chapters [02](../02-extractive-qa-and-the-squad-formulation.md), [07](../07-long-context-qa-with-sliding-window-and-fid.md), [08](../08-multi-hop-qa-and-question-decomposition.md).
- `transformers`, `datasets`, `evaluate`, `torch`.
- A GPU with ≥ 24 GB is comfortable for a Longformer-base run; smaller GPUs can use Longformer-base at reduced batch or drop to `allenai/longformer-base-4096` in inference-only mode.

## Dataset

Use **HotpotQA** in the **distractor** setting as the primary dataset:

```python
from datasets import load_dataset
raw = load_dataset("hotpot_qa", "distractor")
```

Each example provides 10 paragraphs (2 gold, 8 distractors), a question, an answer, and supporting-sentence annotations. This is the setting that isolates *reading* quality from retrieval quality.

Optional additional datasets for depth:

- **2WikiMultiHopQA** — Ho et al., 2020. `datasets.load_dataset("2wikimultihopqa")`. Explicit reasoning-type labels.
- **MuSiQue-Ans** — Trivedi et al., 2022. `datasets.load_dataset("dgslibisey/MuSiQue")`. Shortcut-resistant multi-hop questions.

## Problem statement

### Part A — Sliding-window extractive baseline

Concatenate the 10 HotpotQA paragraphs into a single context (with paragraph delimiters). Fine-tune `deberta-v3-base` (or reuse the exercise-01 model) on HotpotQA with the sliding-window preprocessor from chapter 02:

- `max_seq_length = 384`, `doc_stride = 128`.
- Chunks that do not contain the answer get `start = end = 0`.
- Standard extractive-QA training.

Evaluate on the HotpotQA dev set. Report:

- Final-answer EM and F1 using the HotpotQA official scoring (word-overlap F1 with lowercase + strip punctuation + strip articles).
- Supporting-fact F1 — for every chunk, predict which sentences are "supporting" if the chunk contains any part of the gold answer (weak heuristic; the point is to force you to think about the supporting-fact evaluation).
- Joint F1 (element-wise product of answer-F1 and support-F1, then averaged).

### Part B — Long-context reader

Fine-tune `allenai/longformer-base-4096` on the same data with `max_seq_length = 4096`, `doc_stride = 0` (no sliding needed at this length). Configure global attention on the `[CLS]` token and on every token of the question (per the Longformer paper's QA recipe).

Report the same three metrics as in Part A on the same dev set.

Compare to Part A. In particular, is the long-context reader's *supporting-fact* F1 better, or only its final-answer F1?

### Part C — Depth-of-evidence stratification

Bucket the dev questions by the *position of the second supporting paragraph* in the concatenated context:

- Bucket 1: both supporting paragraphs are in the first 512 tokens.
- Bucket 2: second supporting paragraph is between tokens 512 and 2048.
- Bucket 3: second supporting paragraph is between tokens 2048 and 4096.

Report answer F1 per bucket for both Part A and Part B models. Comment on the sliding-window model's degradation in buckets 2 and 3.

### Part D — Chain-of-thought with a decoder-only LLM (optional but recommended)

Prompt a decoder-only instruction-tuned LLM (e.g., a Llama-3 Instruct variant or an API model you have access to) with:

- The 10 paragraphs.
- The question.
- An instruction to reason step by step and cite the two supporting paragraphs by index.

Do this in two modes:

- **Single-sample CoT.**
- **Self-consistency** — sample 5 chains with `temperature = 0.7` and majority-vote.

Report answer F1 and supporting-fact F1 for both modes on a 500-example dev subset (full dev is expensive on an API).

### Part E — Shortcut audit

Sample 100 dev questions your best model gets *right*. For each, remove one of the two gold paragraphs at inference and re-run the model. Bucket:

- **Robust** — model now abstains or gets the answer wrong (correct behaviour: it needed the second hop).
- **Shortcut** — model still gets the answer right without the second paragraph (it was single-hopping).

Report the shortcut rate. Discuss which reasoning-type buckets (bridge / comparison) shortcut more.

If you ran on MuSiQue (in Stretch Goals) instead, repeat this audit there and note whether MuSiQue's construction reduces the shortcut rate as claimed by Trivedi et al.

### Part F — Write-up

A 500–700 word `README.md` covering:

- Dataset and preprocessing.
- Metric tables for Part A and Part B (answer F1, supporting-fact F1, joint F1).
- Depth-of-evidence stratification chart from Part C.
- If Part D was done: CoT and self-consistency results.
- Shortcut-audit rate from Part E, with two worked examples (one robust, one shortcut).
- One thing you would try next.

## Starter guidance

- HotpotQA's official scoring is *not* the SQuAD scoring — it uses `hotpot_evaluate_v1.py` published by the dataset authors. Grab it or re-implement it from the paper. The joint F1 formulation in particular is worth reading from the paper directly (Yang et al., 2018, §5).
- For Longformer with global attention on the question, use `LongformerForQuestionAnswering` and set the `global_attention_mask` manually before the forward pass — the default is zeros.
- Chunk-boundary answers are more common on concatenated HotpotQA contexts than on SQuAD. Increase `doc_stride` to 200 if your sliding-window model consistently fails on Bucket 2 questions.
- Self-consistency with 5 samples is 5× the LLM cost. On API models, budget for it.
- The shortcut audit is the point of the exercise. Do not skip it.

## Acceptance criteria

- [ ] Sliding-window baseline (Part A) reports answer, supporting-fact, and joint F1 on the HotpotQA dev set.
- [ ] Long-context reader (Part B) reports the same three metrics on the same dev set.
- [ ] Depth-of-evidence stratification (Part C) reported for both models, in a table or chart.
- [ ] Part D done for at least the single-sample CoT mode on a 500-example dev subset (self-consistency and full-dev are optional).
- [ ] Shortcut-audit rate from Part E, with two worked examples.
- [ ] 500–700 word write-up with a hypothesis about which model is best per reasoning type.

## Stretch goals

- **MuSiQue-Ans.** Repeat Parts A, B, and E on MuSiQue. How much lower is the shortcut rate?
- **IRCoT-style iterative retrieve-and-read.** Implement a simple two-hop pipeline that uses the first-round answer to retrieve additional paragraphs from the distractor set. Compare to Part B.
- **FiD (Fusion-in-Decoder).** Train a small FiD variant on top of `flan-t5-base` treating each paragraph as an independent passage. Report the metric set.
- **Decomposition prompting.** Ask an LLM to decompose each question into sub-questions before answering (chapter 08's Strategy C). Compare answer F1 to the direct CoT baseline.
- **Reasoning-type breakdown.** On HotpotQA, use the `type` field (bridge vs. comparison) to report per-type answer F1. Where do the sliding-window and long-context models diverge most?
