# exercise-02: SentencePiece Unigram vs. WordPiece Experiment

**Estimated effort:** 3 hours

## Objective

Train a WordPiece tokenizer (BERT-style) and a SentencePiece Unigram tokenizer on the same corpus, then design an experiment that surfaces where each shines and where each falls over. The goal is to make the trade-offs from chapter 03 and chapter 04 concrete on data you can inspect.

## Prerequisites

- Chapters [03](../03-wordpiece-and-unigram-language-model.md) and [04](../04-sentencepiece-and-language-family-selection.md).
- Chapter [06](../06-training-a-domain-adapted-tokenizer.md) for the metrics (fertility, coverage, round-trip lossiness).
- Python 3.10+, `tokenizers ≥ 0.15`, `sentencepiece ≥ 0.2`.

## Problem statement

Pick a multilingual, morphologically-diverse corpus with at least three languages, one of which uses a non-Latin script. Train two tokenizers at the same target vocabulary size (32k is a good default), then run a matched evaluation and write up the trade-offs.

### Step 1 — Corpus

Choose *one* of:

- Three Wikipedia language slices (e.g. English, Finnish, Japanese) sampled to comparable sizes.
- A multilingual dataset like [`oscar`](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) or [`mC4`](https://huggingface.co/datasets/mc4) filtered to three languages.
- A domain corpus you already have, provided it spans at least two scripts.

Keep the total under ~500 MiB so training finishes in minutes, not hours.

Split into `train` (90%) and `eval` (10%) with a fixed random seed.

### Step 2 — Train two tokenizers

**WordPiece** via `tokenizers`:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

wp = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wp.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=None)
wp.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
wp.decoder = decoders.WordPiece()
trainer = trainers.WordPieceTrainer(
    vocab_size=32_000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    continuing_subword_prefix="##",
)
wp.train(files=["train.txt"], trainer=trainer)
```

**SentencePiece Unigram** via `sentencepiece`:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="train.txt",
    model_prefix="unigram-32k",
    vocab_size=32_000,
    model_type="unigram",
    character_coverage=0.9995,
    byte_fallback=True,
    normalization_rule_name="nmt_nfkc",
    split_digits=True,
)
```

Document your choices: normaliser, pre-tokeniser, `character_coverage`, whether you enable `byte_fallback`, and why.

### Step 3 — Matched evaluation

On the `eval` split, per language, compute:

- **Fertility** (tokens per whitespace-word, or per grapheme for CJK).
- **Coverage** — fraction of tokens that are not `[UNK]` (WordPiece) or byte-fallback (Unigram). Note the semantic difference in your write-up.
- **Round-trip lossiness** — fraction of eval sentences where `decode(encode(x)) != x`.
- **Sequence-length distribution** — mean and p95 tokens per sentence per language.
- **Sub-word regularisation demo (Unigram only)** — pick 5 morphologically-interesting words and show at least three alternative segmentations each, obtained with `SentencePieceProcessor.encode(..., enable_sampling=True, alpha=0.1)`.

Present all metrics in a single table for direct comparison.

### Step 4 — Failure-mode inspection

Find and report at least one concrete example of each of the following:

- A word/sentence that WordPiece maps to `[UNK]` and Unigram does not (or vice-versa).
- A CJK or Devanagari sentence where the two tokenizers produce noticeably different fertilities.
- A round-trip failure for WordPiece (whitespace, casing, or accents lost) that Unigram+SentencePiece handles losslessly.
- A case where Unigram's argmax segmentation is arguably worse than WordPiece's longest-match, morphologically speaking.

For each, quote the tokens and give a one-paragraph explanation grounded in chapter 03 or 04.

## Starter guidance

- Save every tokenizer file with the corpus hash and training config in its filename. You will re-train many times.
- Write a single `evaluate(tokenizer, texts)` helper that returns a dict of the metrics above; call it once per tokenizer per language.
- For CJK, split on grapheme clusters (via `grapheme` or `uniseg`), not on `str.split()`, when computing per-character fertility.
- Do not rely on defaults you did not choose. Log the resolved config (`tokenizer.to_str()` or the `spm.model` protobuf) into your run notes.

## Acceptance criteria

- [ ] Two tokenizers trained on the same split with matched vocabulary size and documented configuration.
- [ ] A single comparison table showing fertility, coverage, round-trip lossiness, and sequence-length p95 per language, per tokenizer.
- [ ] At least one worked example per failure mode listed in Step 4, with correct explanations.
- [ ] A short recommendation: given a hypothetical downstream task (you pick — retrieval, translation, on-device classification), which tokenizer would you use and why? Reference specific numbers from your table.
- [ ] The Unigram sub-word regularisation demonstration produces genuinely different segmentations across samples (not just permutations of whitespace).

## Stretch goals

- **BPE via SentencePiece.** Train a third tokenizer with `model_type="bpe"` and add it to your comparison. Discuss whether SentencePiece-BPE looks more like Unigram or more like character-level BPE on your corpus.
- **Fine-tune a small classifier.** Fine-tune a tiny encoder (e.g. `distilbert-base-multilingual-cased`) with each tokenizer on a downstream task like XNLI's language-matched subset, holding hyperparameters constant. Report whether tokenizer choice moves the eval metric enough to matter.
- **Sub-word regularisation ablation.** Train a downstream model twice with Unigram: once with `alpha=0` (deterministic) and once with `alpha=0.1` (sampled) segmentations. Discuss regularisation vs. instability.
- **Vocabulary-diff visualisation.** Compute the set difference between the two vocabularies and eyeball 50 tokens from each side. What kind of pieces does each algorithm learn that the other does not?
