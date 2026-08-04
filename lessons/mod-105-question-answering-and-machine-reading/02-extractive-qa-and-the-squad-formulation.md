# Extractive QA and the SQuAD Formulation

## Motivation

Extractive QA is the "predict a span" workhorse. It is the right choice whenever a correct answer already exists in the source text and the product needs to cite it — legal contract questions, medical record lookups, customer-support answers grounded in a knowledge-base article, search snippets. Because the answer must be a substring of the context, extractive QA is *structurally incapable* of the specific class of hallucination where a model invents a plausible fact from thin air. That property is the reason the formulation refuses to die even in the LLM era.

This chapter defines the formulation, the input encoding, and the two subtle but load-bearing details that new implementers get wrong: **offset mapping** (translating between characters, subwords, and words) and **sliding windows** (handling contexts longer than the model's maximum sequence length).

## The task, formally

You are given a dataset of triples $(q, c, a)$ where $q$ is a natural-language question, $c$ is a context passage, and $a$ is a contiguous substring of $c$ (i.e., a *span*). The model must predict the character offsets $(s, e)$ of the span in $c$.

In practice we predict at the *token* level and translate back to characters afterwards. Given the tokenised sequence

```
[CLS] q_1 q_2 … q_m [SEP] c_1 c_2 … c_n [SEP]
```

the model produces two logit vectors of shape `[sequence_length]`: one for the start position, one for the end position. The predicted span is $(i, j)$ where $i \le j$, both are inside the context region, and $\text{start\_logit}_i + \text{end\_logit}_j$ is maximised over all valid $(i, j)$.

The dominant training loss is cross-entropy on the start distribution and cross-entropy on the end distribution, averaged. This is Devlin et al.'s original BERT fine-tuning recipe (2019) and every subsequent HF `AutoModelForQuestionAnswering` implementation still uses it.

## SQuAD and its relatives

The canonical training data:

- **SQuAD 1.1** (Rajpurkar, Zhang, Lopyrev & Liang, ["SQuAD: 100,000+ Questions for Machine Comprehension of Text"](https://arxiv.org/abs/1606.05250), *EMNLP 2016*). 100k+ crowd-sourced questions on Wikipedia paragraphs; every question is answerable and the answer is always a span.
- **SQuAD 2.0** (Rajpurkar, Jia & Liang, ["Know What You Don't Know"](https://arxiv.org/abs/1806.03822), *ACL 2018*). Adds 50k adversarially crafted *unanswerable* questions. This changes evaluation and calibration — chapter 09 covers it in depth.
- **Natural Questions** (Kwiatkowski et al., ["Natural Questions: A Benchmark for Question Answering Research"](https://aclanthology.org/Q19-1026/), *TACL 2019*). Real Google queries paired with entire Wikipedia pages. Introduces the *long answer / short answer* distinction and the "no answer" case.
- **TriviaQA** (Joshi, Choi, Weld & Zettlemoyer, ["TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension"](https://arxiv.org/abs/1705.03551), *ACL 2017*). Trivia-style questions with distantly supervised evidence documents.
- **XQuAD, MLQA, TyDi QA** — multilingual extractive QA. Used for cross-lingual transfer evaluation.

For chapters 02–04 we work in the SQuAD 1.1 setting: every question has exactly one answerable span in the given context. Chapter 09 upgrades to SQuAD 2.0.

## Input encoding

A tokenised example must expose four things to the training loop:

1. **`input_ids`, `attention_mask`, `token_type_ids`** — the model input, with the question in segment 0 and the context in segment 1 (or the equivalent for models without `token_type_ids`, like RoBERTa).
2. **`start_positions` and `end_positions`** — the *token indices* of the answer's first and last subword tokens.
3. **`offset_mapping`** — a per-token list of `(char_start, char_end)` tuples into the *original* context string. This is what lets you translate predicted token indices back to a substring of the raw context at inference time.
4. **`sequence_ids`** — which tokens belong to the question, the context, or special tokens. Used to mask out non-context positions when searching for the best span at inference.

The Hugging Face `tokenizers` library returns `offset_mapping` and `sequence_ids` when you pass `return_offsets_mapping=True` and access the resulting `BatchEncoding`. Google's original BERT code did the same thing by hand with `tokens_to_original_index` maps; the modern API is much less error-prone.

## Aligning a character-level answer to token indices

The training data gives you `answer_text` and `answer_start` (character offset into the context). To convert to `start_positions` and `end_positions` (token indices), scan the `offset_mapping`:

```python
def find_token_span(offsets, sequence_ids, answer_start, answer_end):
    # First and last indices that belong to the context.
    ctx_start = next(i for i, sid in enumerate(sequence_ids) if sid == 1)
    ctx_end   = len(sequence_ids) - 1 - next(
        i for i, sid in enumerate(reversed(sequence_ids)) if sid == 1
    )

    # If the answer is fully outside this window, mark it unanswerable.
    if offsets[ctx_start][0] > answer_start or offsets[ctx_end][1] < answer_end:
        return 0, 0  # CLS is the canonical "no answer" position

    # Otherwise walk forward and backward to the first/last token that overlaps.
    tok_start = ctx_start
    while tok_start <= ctx_end and offsets[tok_start][0] <= answer_start:
        tok_start += 1
    tok_end = ctx_end
    while tok_end >= ctx_start and offsets[tok_end][1] >= answer_end:
        tok_end -= 1
    return tok_start - 1, tok_end + 1
```

Three easy-to-miss failure modes:

- If the answer straddles a window boundary in a sliding-window setup, it must be marked "no answer" in *that* window even though the underlying dataset is SQuAD 1.1. Otherwise you train the model to predict a wrong span.
- The `(char_start, char_end)` returned by fast tokenisers is *half-open* in some conventions — always eyeball a few examples before trusting the walk.
- If your tokeniser strips whitespace or normalises Unicode, `offset_mapping` is against the *original* string, but the training answer offset may be against a *normalised* string. Read once, then decide which representation is canonical.

## Sliding-window (chunked) encoding for long contexts

BERT-family encoders top out at 512 tokens; DeBERTa-v3-base at 512; RoBERTa-large at 512; Longformer at 4096 (chapter 07 covers the long-context specialists). A single Natural Questions article is far longer than 512 tokens, so the standard trick is a **sliding window with overlap** at tokenisation time:

- Split the context into overlapping chunks of `max_length - question_length - special_tokens`. Use `stride` (typically 128) so adjacent chunks share a prefix.
- Each chunk becomes an independent training / inference example. The `overflow_to_sample_mapping` returned by HF fast tokenisers tells you which original example each chunk came from.
- At training time, chunks where the answer does not fall fully inside are marked with `start=end=0` (CLS position). The model learns to predict "no answer" for those chunks, which is exactly the calibration signal chapter 09 formalises.

The HF tokenisation call is:

```python
tok = tokenizer(
    questions,
    contexts,
    max_length=384,
    truncation="only_second",
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
```

Note that `truncation="only_second"` truncates only the context, never the question — otherwise you can lose the interrogative content that makes the answer unique.

## Post-processing: from per-chunk logits back to an answer string

At inference time each chunk yields start/end logit vectors. To get one predicted answer per original question:

1. For each chunk, take the top-*k* start indices and top-*k* end indices (typically `k=20`).
2. Enumerate candidate `(start, end)` pairs subject to: both indices are inside the context region (`sequence_ids == 1`), `end >= start`, and `end - start + 1 <= max_answer_length` (typically 30 tokens).
3. Score each candidate as `start_logit + end_logit`.
4. Merge candidates across all chunks belonging to the same original example.
5. Convert the highest-scoring candidate back to a substring via `offset_mapping`.

The Hugging Face `run_qa.py` example encodes this exact algorithm in `utils_qa.py` (`postprocess_qa_predictions`). It is worth reading once, because subtle bugs at this stage are invisible in training loss and only show up as a persistent 5-point EM/F1 gap.

## What can go wrong (and pass tests)

- **Off-by-one in the token span.** Model appears to work; F1 is 3 points below the paper. Always print a handful of decoded answers next to the gold answers early in training.
- **Question truncation.** With `truncation="longest_first"` a long question can lose its final word (`"what year did X do Y?"` becomes `"what year did X do"`). Use `"only_second"`.
- **Sliding-window answer split.** An answer that starts in one chunk and ends in the next is marked unanswerable in *both* chunks; the model never sees it. Increase `stride` or `max_length` if this happens often.
- **Wrong context region at inference.** Predicting an "answer" from inside the question or from special tokens. Always mask with `sequence_ids` before decoding.

## Chapter summary

- Extractive QA predicts a contiguous span of a supplied context by producing start and end token distributions and choosing the highest-scoring valid pair.
- The SQuAD 1.1 recipe — question and context concatenated, cross-entropy on start and end positions — is still the reference implementation for encoder-only readers.
- Two ideas do the load-bearing work: **offset mapping** (subword ↔ character translation) and **sliding-window encoding** (long context → overlapping chunks with a "no answer" fallback).
- Post-processing top-*k* spans across all chunks — not simply argmax of the logits — is what turns per-chunk predictions into a single answer string, and where implementation bugs most often hide.
