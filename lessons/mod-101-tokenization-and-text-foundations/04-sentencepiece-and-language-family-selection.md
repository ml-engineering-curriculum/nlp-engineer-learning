# SentencePiece and Choosing a Scheme per Language Family

## Motivation

BPE, WordPiece, and Unigram LM are algorithms. `sentencepiece` (Kudo & Richardson, EMNLP 2018 System Demonstrations) is a **library** that implements two of them (BPE and Unigram) with a design goal that matters far more than the algorithm choice: it is language- and script-agnostic, and it round-trips losslessly.

This chapter covers what SentencePiece does that raw BPE/WordPiece do not, and then how to pick a tokenization scheme when your data mixes languages, scripts, or morphological types.

## What SentencePiece actually adds

Traditional tokenizer stacks assume a whitespace-based pre-tokeniser: split on spaces, then apply BPE/WordPiece per word. That assumption breaks the moment you leave Latin-script text:

- Chinese and Japanese generally have no inter-word whitespace.
- Thai and Lao have no inter-word whitespace but do have inter-sentence whitespace.
- Arabic has whitespace but heavy internal morphology and shaping.
- Korean has whitespace but agglutinative morphology across it.

SentencePiece treats the input as a **raw stream of Unicode characters**, including whitespace, and lets the sub-word algorithm learn where the "words" are. To keep the pipeline reversible, it maps whitespace to a printable placeholder — the `▁` character (U+2581, "lower one eighth block") — before training. Every token that begins with `▁` marks a word boundary; tokens without `▁` are word-internal.

```
"Hello world"      → ["▁Hello", "▁world"]
"新しいトークナイザー"     → ["▁新しい", "トークナ", "イザー"]     # (illustrative segmentation)
```

Decoding is a single string replace of `▁` back to space, which round-trips perfectly. This is why SentencePiece is the default for T5, mT5, ALBERT, XLNet, NLLB, and virtually every multilingual model.

### Two knobs you must set

- **`model_type`** — `bpe`, `unigram` (default and recommended by the authors), `char`, or `word`.
- **`character_coverage`** — the fraction of characters in the training corpus the base vocabulary must cover. The upstream docs recommend `1.0` for languages with small alphabets (English, most European) and `0.9995` for languages with large character sets (Japanese, Chinese, Korean, historical texts) so that ultra-rare glyphs do not bloat the base vocabulary.

Other useful knobs:

- **`byte_fallback=True`** — falls back to UTF-8 bytes for any character not covered, giving you the "no UNK" property of byte-level BPE without paying for bytes on every character.
- **`normalization_rule_name`** — one of `nmt_nfkc`, `nfkc`, `nmt_nfkc_cf` (case-fold), or a custom rule set. Applied before tokenisation.
- **`user_defined_symbols`** — tokens that will never be split (custom tags, mask tokens, template markers).
- **`split_by_whitespace`, `split_by_unicode_script`, `split_digits`** — pre-tokeniser hints; set them explicitly rather than trusting defaults.

## Training SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="unigram-32k",
    vocab_size=32_000,
    model_type="unigram",
    character_coverage=0.9995,       # multilingual / CJK
    byte_fallback=True,
    normalization_rule_name="nmt_nfkc",
    user_defined_symbols=["<sep>", "<mask>"],
    split_digits=True,               # each digit becomes its own piece, better for arithmetic
    max_sentence_length=8192,
)

sp = spm.SentencePieceProcessor(model_file="unigram-32k.model")
sp.encode("Hello, world!", out_type=str)
# ['▁Hello', ',', '▁world', '!']
```

Because SentencePiece is a self-contained tool, its models load in every serving framework without a Python dependency chain — the `.model` file is a Protocol Buffer that the C++ / Rust / JS bindings all read.

## Choosing a scheme per language family

There is no universal answer, but the following heuristics catch most cases. Confirm each with the diagnostic metrics from chapter 06 (fertility, coverage, downstream loss) on your own corpus.

### English and other Latin-script analytic languages

Sub-word matters least here — words are short, morphology is light. Byte-level BPE at 32-50k vocabulary is the standard choice and is what every English-first decoder-only model ships. WordPiece is a fine alternative if you are extending BERT-family encoders.

### Germanic / Slavic / Romance languages with moderate morphology

Byte-level BPE at 32-50k still works, but expect higher fertility on inflected forms. Unigram LM with sub-word regularisation can help small-data fine-tuning.

### Highly agglutinative languages (Turkish, Finnish, Hungarian, Korean, Swahili)

Fertility can be 2-3× that of English at the same vocab size. Options:

- **Grow the vocabulary** (64-128k) if you can afford the embedding cost.
- **Use SentencePiece Unigram** with `split_by_unicode_script=True` and `byte_fallback=True`; sub-word regularisation helps regularise inflection.
- **Add domain morphemes as `user_defined_symbols`** if you have a curated list.

### Chinese, Japanese, Korean (CJK)

No inter-word whitespace, large character inventories.

- SentencePiece Unigram with `character_coverage=0.9995` is the industry default; it handles the no-whitespace assumption natively.
- For Chinese-only or Japanese-only encoders (BERT-Chinese, RoBERTa-Chinese), WordPiece with a per-character pre-tokeniser is common: each Han character becomes its own initial piece.
- Do not run raw byte-level BPE with a Latin-tuned pre-tokeniser on CJK — you will burn 2-4 tokens per Han character and destroy your context budget.

### Arabic

- Bidirectional text and script shaping (see chapter 05) mean **normalisation is more important than algorithm choice**. Apply NFKC and one of the Arabic-specific normalisation policies (removing tatweel, unifying alef forms) *before* training.
- SentencePiece Unigram with `nmt_nfkc` handles most cases; add per-corpus normalisation rules for diacritics (harakat) depending on whether they are semantically important for your task.
- Consider `split_digits=True` — Arabic-Indic digits (٠-٩) and ASCII digits can otherwise collapse into rare mixed tokens.

### Indic scripts (Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gurmukhi, Odia, ...)

- Grapheme clusters matter (see chapter 05). A conjunct like `क्ष` is three code points (क + ्  + ष) but one perceived character. Applying BPE at the code-point level splits conjuncts arbitrarily.
- SentencePiece with `split_by_unicode_script=True` and `byte_fallback=True` handles the multi-script case; consider a normaliser that composes independent vowel signs into their combined forms.
- Multilingual Indic models (IndicBERT, MuRIL) use SentencePiece Unigram vocabularies in the 100-200k range because they cover 10+ scripts.

### Code, formulas, and mixed-format corpora

- Code benefits from **splitting digits and punctuation**, keeping indentation tokens whole (e.g. tokens for common indent widths), and *not* stripping newlines. Popular code models use byte-level BPE with tuned pre-tokenisers (StarCoder2, CodeLlama).
- Chemistry SMILES, biomedical acronyms, and DNA sequences need custom initial alphabets and `user_defined_symbols` for reagent tokens or gene names; otherwise fertility explodes.

## Decision checklist

Before committing to a tokenizer scheme, answer:

1. **What is my pretrained starting point?** If any, use its tokenizer verbatim. Don't retrain unless you are also retraining the model.
2. **What scripts are in my corpus, and in what proportions?** Latin-only → BPE is fine. Mixed scripts or non-whitespace scripts → SentencePiece.
3. **How much morphology am I encoding?** Heavy morphology → Unigram LM (regularisation helps) or larger vocabulary.
4. **What is my context budget per document?** If the median document is close to the model's context window, aggressively minimise fertility.
5. **What is my downstream task?** Generation and translation are unforgiving of detokenisation errors; classification is not. This changes how strict you need to be about lossless round-trip.

## Chapter summary

- SentencePiece is a library, not an algorithm. It runs BPE or Unigram over a raw Unicode stream with `▁` as an explicit whitespace token, so it round-trips losslessly across every script.
- `character_coverage`, `byte_fallback`, `normalization_rule_name`, and `split_*` are the knobs that matter most.
- Language family determines defaults: Latin → byte-level BPE; CJK/Arabic/Indic/multilingual → SentencePiece Unigram; agglutinative → larger vocabulary or Unigram.
- Chapter 05 goes one level deeper into Unicode itself — segmentation, normalisation, and script-specific pitfalls that every tokenizer inherits.
