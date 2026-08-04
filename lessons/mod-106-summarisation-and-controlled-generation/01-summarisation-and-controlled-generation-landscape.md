# The Summarisation and Controlled-Generation Landscape

## Motivation

Summarisation is the task most NLP engineers ship first and second-guess forever. Every product with a text surface eventually asks for one: turn a 40-message support thread into a two-line issue, distil a 30-page contract into an executive-friendly briefing, collapse an hour-long meeting transcript into action items, replace a wall of log lines with "here is what broke and where". Underneath, it is the same task — read text, produce shorter text, do not make things up — and it is the task on which almost every modern generative failure mode was first named.

*Controlled generation* is the other half of the story. Once we can generate text, we almost never actually want the raw model output. We want a JSON object with typed fields for a downstream system, or a summary that matches a template, or a bounded vocabulary, or a citation into the source. Constrained decoding, structured outputs, and faithfulness-aware training are the levers we pull to turn "the model produced a plausible paragraph" into "the model produced an artefact my product can rely on".

This module owns both. We start from the classical extractive baselines, move through the modern encoder-decoder recipe (BART, T5, PEGASUS, mT5), climb into long-document and multi-document strategies, and end on the parts nobody puts on the tutorial poster: the decoding zoo, the constraint stack, and the faithfulness diagnostics that stop a fluent summariser from lying with a straight face.

## A working taxonomy

Every summarisation product pins itself along three orthogonal axes. Confusing them is the leading source of "we built the wrong summariser" incidents.

| Axis                     | Options                                                                                          | What it decides                                                                                                    |
|--------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Approach**             | Extractive (select sentences/spans) · Abstractive (generate new text) · Hybrid (extract then rewrite) | Whether hallucination is even *possible*, whether citations are free, and what the base model must be.             |
| **Input scope**          | Single-document · Multi-document · Long-document (single, > context window) · Query-focused         | Whether you need chunking, retrieval, or hierarchical fusion; whether "query" is part of the input.                |
| **Output shape**         | Free-form paragraph · Bullet list · Headline · Structured record (JSON/table) · Highlight spans     | Whether you can hand the output to a downstream system or a human, and whether constrained decoding is required.   |

A concrete product usually pins each axis. "Extractive, single-doc, highlight spans" is a research-paper annotator. "Abstractive, multi-doc, JSON record" is a competitive-intelligence brief pulled from newswire. "Hybrid, long-document, bullet list" is a meeting-notes summariser stacked over a transcript. "Query-focused, multi-doc, paragraph with citations" is the RAG-style QA/summarisation hybrid you see in every "chat with your docs" product.

## Extractive vs. abstractive — the choice that decides your failure mode

Everything in this module downstream of the taxonomy flows from one question: is the summariser allowed to *invent* text or must it only *select* it?

- **Extractive summarisation** picks whole sentences (or spans) from the source and returns them. It cannot hallucinate — the worst it can do is choose the wrong sentence or produce a summary that reads unnaturally. Citation is free: every sentence is already anchored to the source. It struggles when a natural summary requires paraphrase, aggregation, or fluent stitching.
- **Abstractive summarisation** generates new text conditioned on the source. It can produce short, fluent, paraphrased summaries that no extractive system could. It can also confidently make things up — invent numbers, swap named entities, add unstated claims — and the fluency of the output *hides* the error from casual review.
- **Hybrid ("extract-then-abstract" or "cite-then-generate")** is the pattern most production systems converge on. An extractive stage or a retriever narrows the input; an abstractive stage rewrites it into readable prose; a citation head or extractive verifier grounds each claim in a source span. Chapter 09 (structured outputs) and chapter 12 (mitigating hallucination) both push you toward this pattern.

If your domain has zero tolerance for invented content — legal, medical, financial, regulated — start extractive and only add abstractive when you can afford the faithfulness gates from chapters 11–12. If your users care about fluency more than provenance — casual news summaries, TL;DRs, chat messages — start abstractive with a strong pretrained encoder-decoder.

## What "controlled" means in this module

"Controlled" is doing a lot of work. It covers at least four different kinds of control an NLP engineer might reach for:

1. **Length control** — enforce a token or sentence budget (headline, tweet, bullet list, exec summary). Handled by `max_length`, `min_length`, `length_penalty`, and length-aware fine-tuning. See chapter 07.
2. **Format control** — force the output to conform to a regex, a JSON schema, or a grammar (bullet list, ISO date, typed record). Handled by constrained decoding libraries — Outlines, Guidance, `lm-format-enforcer`, XGrammar, or the `PrefixConstrainedLogitsProcessor` in `transformers`. See chapters 08–09.
3. **Content control** — steer *what* the model talks about (aspect-based, query-focused, keyword-forced, or entity-anchored summaries). Handled by input templating, disjunctive constraints, and control-code fine-tuning.
4. **Faithfulness control** — enforce that every claim is supported by the source. Handled by copy attention, decoding-time re-ranking, post-hoc verification, and abstention. See chapters 11–12.

A production summariser typically stacks two or three of these. A JSON summariser with a fixed schema, a token budget per field, and an NLI-based faithfulness gate is a normal ask, not an exotic one.

## Landmark datasets

You should recognise these by name because papers, benchmarks, and pretrained checkpoints assume you do:

- **CNN/DailyMail** (Hermann et al., 2015; See, Liu & Manning, 2017). ~300 k news articles paired with bullet-point highlights. The workhorse benchmark for English abstractive news summarisation. Highly extractive by construction — a strong LEAD-3 baseline is famously hard to beat here.
- **XSum** (Narayan, Cohen & Lapata, 2018). ~230 k BBC articles paired with a single-sentence "extreme" summary. Much more abstractive than CNN/DailyMail; the go-to stress test for hallucination.
- **Gigaword / Newsroom** — headline generation and diverse news summarisation styles.
- **SAMSum / DialogSum** — dialogue and chat summarisation.
- **arXiv / PubMed / GovReport / BillSum / BigPatent** — long-document summarisation over scientific, legislative, and patent text.
- **Multi-News / WikiSum** — multi-document summarisation.
- **QMSum** — query-focused meeting summarisation.
- **MLSum / XL-Sum** — multilingual summarisation over dozens of languages.
- **SCROLLS** — a suite for long-context summarisation and QA.

We will pull from these as running examples throughout the module. When in doubt, prototype on XSum (it exposes hallucination fastest) and CNN/DailyMail (it exposes extractive-vs-abstractive trade-offs).

## What each chapter owns

- **Chapters 02–05: architectures and inputs.** Extractive baselines, the encoder-decoder recipe, and the long/multi-doc strategies that push past a 1 k-token window.
- **Chapters 06–07: decoding.** How to turn a trained model into text: greedy, beam, nucleus, typical, contrastive — plus the length/repetition knobs that ship with them.
- **Chapters 08–09: constraint and structure.** Regex, JSON schema, and CFG-based constrained decoding; structured-output patterns for summaries.
- **Chapters 10–13: evaluation and faithfulness.** ROUGE and BERTScore (and why they mislead), NLI- and QA-based faithfulness diagnostics, mitigation patterns, and human-evaluation rubrics.

## The two questions to keep in your head

Two questions run through the entire module. If you can answer both for your product before starting, most of the design decisions fall out on their own:

1. **Where can a fact enter the summary?** Only from the source (extractive / heavily-grounded abstractive) or also from the model's parametric knowledge (unconstrained abstractive)? This decides your faithfulness stack.
2. **Who consumes the output?** A human, a downstream system, or both? This decides your format stack: paragraph, bullet list, or structured record, and how strictly you need to constrain the decoder.

Both questions are product decisions dressed up as engineering ones. Answer them early, write them down, and hold the design accountable to them.

## Chapter summary

- Summarisation and controlled generation are the same problem viewed from two angles: "produce shorter text" and "produce text with guarantees". Every serious summariser needs both.
- Pin three axes before you build: approach (extractive / abstractive / hybrid), input scope (single / long / multi / query-focused), output shape (paragraph / list / record).
- Extractive cannot hallucinate but cannot paraphrase. Abstractive can do both. Hybrid ("extract-then-abstract" or "cite-then-generate") is the pattern most production systems land on.
- "Controlled" spans length, format, content, and faithfulness. This module gives you a lever for each.
- Two anchor questions for every product decision: where can a fact enter the summary, and who consumes the output.
