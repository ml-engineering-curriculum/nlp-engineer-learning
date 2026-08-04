# exercise-04: Domain-Adapted Tokenizer Training

**Estimated effort:** 3 hours

## Objective

Train a domain-adapted tokenizer for a specialised corpus, quantify how much of the pretraining tokenizer's fertility problem it fixes, and produce a written recommendation for whether to (a) extend the existing model with new tokens, (b) continue-pretrain with a new tokenizer, or (c) leave everything alone.

This is the applied version of the metrics-driven workflow in chapter 06.

## Prerequisites

- Chapter [06](../06-training-a-domain-adapted-tokenizer.md).
- Chapters [02](../02-byte-pair-encoding-and-byte-level-bpe.md) and [04](../04-sentencepiece-and-language-family-selection.md) for the algorithm choices.
- Python 3.10+, `tokenizers ≥ 0.15`, `transformers ≥ 4.40`, `datasets`.

## Problem statement

Pick a domain corpus that is *demonstrably different* from generic web text. Suggested options (pick one, or bring your own of comparable size and specificity):

- **Biomedical:** [PubMed Abstracts / PMC subset](https://pubmed.ncbi.nlm.nih.gov/download/) — full of Latin binomials, drug names, gene identifiers.
- **Legal:** [EDGAR filings](https://www.sec.gov/edgar), [case-law datasets](https://free.law/), or the `pile-of-law` slice on Hugging Face.
- **Source code:** the `code-parrot/github-code` slice, or a single-language filter (Python, Go, Rust).
- **Chemistry / materials:** SMILES strings, IUPAC names, materials abstracts.
- **A non-English or low-resource language slice** where an English tokenizer performs badly.

Target ~50-500 MiB of raw text (large enough that tokenizer training is not trivial, small enough that it finishes in minutes).

### Deliverables

You will produce and defend the following:

1. **Baseline measurement.** Load an existing tokenizer that is a plausible starting point for the domain (`gpt2`, `bert-base-uncased`, `meta-llama/Llama-2-7b-hf` — pick and justify). Measure on your held-out slice: fertility (tokens/word and tokens/byte), coverage / OOV rate, round-trip lossiness, and sequence-length p95.
2. **Domain tokenizer.** Train a new tokenizer that targets the same vocabulary size, using either:
   - `AutoTokenizer.train_new_from_iterator` (keeps normaliser + pre-tokeniser of the source) — recommended when you want format compatibility;
   - or a from-scratch pipeline (BPE, WordPiece, or Unigram) with your own normaliser / pre-tokeniser choice — recommended for a materially different domain.
   Document every choice. Save the tokenizer to disk.
3. **Extension tokenizer.** Take the baseline tokenizer and add the top-K domain-specific tokens (K in the range 500-5000) via `add_tokens` — say, the K most frequent whitespace-words in the domain that the baseline tokenizer fragments into ≥ 3 pieces. Save this as a second experimental tokenizer.
4. **Comparison table.** Re-measure all four metrics on the held-out slice for baseline, extended, and fully-retrained. Also report:
   - vocabulary size;
   - embedding-matrix memory cost implied at hidden size `d = 4096`, fp16 (this is a proxy for "what would resizing the model cost me?").
5. **Vocabulary inspection.** Print 30 tokens sampled uniformly from your new tokenizer's vocabulary and 30 tokens from the tail (least frequent by trainer order). Identify: (a) domain morphemes the tokenizer learned, (b) boilerplate / garbage tokens, (c) tokens you did not expect. Comment on all three.
6. **Recommendation.** In `RECOMMENDATION.md`, give a one-page answer: which of (baseline / extended / fully-retrained) would you ship, given a hypothetical downstream task and traffic pattern that you specify? Cite specific numbers from the table.

## Starter guidance

- Split your corpus deterministically (`train` 90%, `eval` 10%) *before* touching any tokenizer, so the evaluation stays honest.
- For the extension tokenizer, sort words by `existing_tokenizer.encode(word).ids` length and pick the top-K worst offenders as your candidate additions. Filter out garbage (control chars, all-numeric, over-length) before adding.
- Do not skip round-trip lossiness. WordPiece and lower-cased tokenizers routinely lose whitespace and casing; if the baseline has 5% lossiness on your corpus, that is data for the decision.
- Report memory in **numbers**: `vocab_size * hidden_dim * bytes_per_param`. Do not hand-wave "the embeddings would grow" — quantify it.
- If your domain is code or math, add `pre_tokenizers.Digits(individual_digits=True)` and consider a `Punctuation` splitter. These small choices dramatically change token quality.

## Acceptance criteria

- [ ] Domain corpus and its origin are documented; split is deterministic and reproducible.
- [ ] Baseline metrics are recorded before any training; you can point at the code that produced them.
- [ ] Domain tokenizer trains successfully to the target vocabulary size and its config is saved alongside the tokenizer file.
- [ ] Extension tokenizer adds a defensible set of tokens (not a random top-K) and is saved separately.
- [ ] Comparison table includes fertility (tokens/word and tokens/byte), OOV rate, round-trip lossiness, sequence-length p95, vocabulary size, and embedding memory at `d=4096`.
- [ ] Vocabulary inspection identifies at least one domain morpheme win and at least one boilerplate / garbage token per new tokenizer.
- [ ] `RECOMMENDATION.md` gives a concrete ship/extend/leave-alone answer with cost/benefit reasoning tied to your numbers.

## Stretch goals

- **Continue-pretraining sanity check.** Take a small pretrained model (`gpt2` — 124M is enough), resize its embeddings to your new tokenizer, and run a short continue-pretraining pass on the domain train split. Report loss curves; discuss whether the new tokens converge in the compute you can afford.
- **Down-stream ablation.** Fine-tune a downstream classifier (or evaluate a zero-shot task) with each tokenizer variant. Confirm or refute your recommendation on task metric.
- **PII / leakage scan.** Automatically scan your final vocabulary for tokens that look like emails, SSNs, common passwords, or long identifiers. Report and, if any exist, retrain with the offending data filtered out.
- **Two-corpus tokenizer.** Train a single tokenizer on a *mixture* of the domain corpus and a generic web slice (50/50), and compare fertility on both. Is the compromise better than either extreme? At what mixture ratio does the domain benefit start to flatten?
