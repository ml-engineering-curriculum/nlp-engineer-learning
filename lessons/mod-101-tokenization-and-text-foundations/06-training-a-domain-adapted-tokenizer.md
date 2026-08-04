# Training a Domain-Adapted Tokenizer with `tokenizers`

## Motivation

Most NLP work does not start from scratch — it starts from a pretrained model with a fixed vocabulary. But there are three cases where you need your own tokenizer:

1. **You are training a base model from scratch** (rare, but happens in dedicated labs).
2. **Your domain drifts so far from the pretraining corpus that fertility ruins your economics** (biomedical, legal, code, chemistry, low-resource languages).
3. **You are *extending* an existing tokenizer** — adding tokens for company jargon, product names, control tokens, or a new language.

This chapter shows how to do all three with the Hugging Face `tokenizers` library, and — more importantly — how to *measure* whether the new tokenizer actually helps.

## The `tokenizers` pipeline stages

`tokenizers` (the Rust-backed sister of `transformers`) exposes each pipeline stage as a swappable component:

| Stage             | Purpose                                              | Common choices                                                               |
|-------------------|------------------------------------------------------|------------------------------------------------------------------------------|
| `normalizers`     | Unicode / casing / accent transformations            | `NFC`, `NFKC`, `Lowercase`, `StripAccents`, `BertNormalizer`, `Sequence`     |
| `pre_tokenizers`  | Coarse split before sub-word modelling               | `Whitespace`, `ByteLevel`, `Metaspace`, `BertPreTokenizer`, `Punctuation`, `Digits`, `Split` |
| `models`          | The sub-word algorithm                               | `BPE`, `WordPiece`, `Unigram`                                                |
| `trainers`        | Training config for the model                        | `BpeTrainer`, `WordPieceTrainer`, `UnigramTrainer`                           |
| `post_processors` | Add special tokens, build templates for pairs        | `TemplateProcessing`, `ByteLevel`, `BertProcessing`, `RobertaProcessing`     |
| `decoders`        | Reverse the pipeline to get text back                | `ByteLevel`, `WordPiece`, `Metaspace`                                        |

Build a tokenizer by composing them:

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders, processors

tokenizer = Tokenizer(models.BPE(unk_token=None))

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC(),
    # add domain-specific rules here, e.g. lowercase for a case-insensitive encoder
])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32_000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train(files=["domain-corpus.txt"], trainer=trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.save("domain-bpe-32k.json")
```

Every choice above is a decision an NLP engineer should be able to defend on request.

## Extending an existing tokenizer

Sometimes you want the pretrained model's tokens *plus* a few new ones (product names, control tokens, a domain-specific script). Two approaches:

### 1. `add_tokens` on the existing tokenizer

```python
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
num_added = tok.add_tokens(["neupogen", "onc-201", "her2neu"])

model = AutoModel.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(tok))
```

- New tokens get random embeddings. You must fine-tune to teach the model their meaning.
- `add_special_tokens={"additional_special_tokens": [...]}` is the analogous call for special tokens that should not be split.
- After `resize_token_embeddings`, remember to re-tie the LM head if your architecture ties weights, and to include the new tokens in any downstream tokenizer / vocab files.

This is the right tool for a handful of tokens. Do not use it for thousands — the fertility improvement will not justify the embedding cost.

### 2. Train a new tokenizer on your corpus

For substantial domain drift, train a new tokenizer and either replace the model's tokenizer (only viable if you are also retraining or continuing pretraining) or merge vocabularies carefully. There is no free lunch: you cannot swap the tokenizer of a trained model without retraining, because every embedding row is defined by its token ID.

The Hugging Face convenience wrapper `AutoTokenizer.train_new_from_iterator` retrains only the vocabulary while keeping the same normaliser and pre-tokeniser as the source:

```python
from transformers import AutoTokenizer

base = AutoTokenizer.from_pretrained("gpt2")
new_tok = base.train_new_from_iterator(
    text_iterator=corpus_iterator(),
    vocab_size=32_000,
    new_special_tokens=["<|domain_start|>", "<|domain_end|>"],
)
new_tok.save_pretrained("gpt2-domain")
```

This produces a tokenizer that is *format-compatible* with the source (same special tokens, same encoding conventions) but with a domain-specific vocabulary — useful when you are pretraining from scratch on a new corpus but want to reuse the source's tooling.

## Metrics that tell you whether your tokenizer is any good

Do not choose a tokenizer on vibes. Every experiment in this module should report at least the following.

### Fertility

Average number of tokens per whitespace-delimited word (or per grapheme for CJK). Lower is better, subject to vocabulary-size constraints.

```python
def fertility(tokenizer, texts):
    total_tokens = 0
    total_words = 0
    for t in texts:
        total_tokens += len(tokenizer.encode(t).ids)
        total_words += len(t.split())
    return total_tokens / max(total_words, 1)
```

Report fertility on:

- a general-domain reference corpus (Wikipedia sample, C4);
- your target domain;
- optionally, a code / math / non-Latin subsample if you care.

If your domain fertility is > 1.4× the reference, a domain tokenizer is likely to pay off.

### Coverage

Fraction of tokens that are *not* the fallback (byte-fallback bytes, `<unk>`, or long-tail single-character tokens). Low coverage indicates that the vocabulary is not learning morphemes from the domain.

### OOV rate

For non-byte-level tokenizers, the fraction of `<unk>` per corpus. Should be near zero on the training distribution. If it is not, either your `min_frequency` is too high or your normalisation is broken.

### Round-trip lossiness

```python
def roundtrip_error_rate(tokenizer, texts):
    errors = 0
    for t in texts:
        if tokenizer.decode(tokenizer.encode(t).ids) != t:
            errors += 1
    return errors / len(texts)
```

For generation and translation tokenizers, this should be exactly 0 on your test set. SentencePiece is designed to guarantee it; byte-level BPE also achieves it if the pre-tokeniser is set correctly; WordPiece typically does not (whitespace and casing are lost).

### Downstream loss

The only metric that ultimately matters: does the new tokenizer improve your task loss / metric at fixed compute? Run a short pretraining or fine-tuning ablation with the old and new tokenizers, holding everything else constant.

Anecdotal fertility wins can vanish when the new tokenizer trades sequence length for a larger embedding matrix. Always confirm end-to-end.

## Vocabulary-quality anti-patterns to look for

Print your vocabulary before you ship it. Common problems:

- **Boilerplate tokens.** Long tokens like `<!DOCTYPEhtml>`, `SPDX-License-Identifier`, or `https://www.` show up when the training corpus was not cleaned. They inflate vocab size for no downstream benefit.
- **Merged whitespace runs.** `                                ` (long spaces) is a common token in code corpora. Sometimes desirable (indentation), sometimes a bug (crawled HTML). Decide deliberately.
- **Missing morphemes.** For an inflected language, if no token contains the plural, tense, or case suffix, fertility on that morphology will be bad. Inspect a stratified sample.
- **Duplicate near-tokens.** `Hello`, `hello`, `HELLO`, `_hello`, `Ġhello` may all exist. This is often correct (case-sensitive byte-level BPE), but if you have a case-insensitive downstream task, add a lowercasing normaliser at training time.
- **Domain leakage.** A vocabulary trained on scraped internal data can leak identifiers, credentials, or PII into token strings. Redact before training, or filter after.

## Chapter summary

- The `tokenizers` library treats each pipeline stage as a swappable component; construct explicitly, do not rely on defaults.
- Extending a pretrained tokenizer with `add_tokens` is right for a handful of tokens; for substantial domain drift, train a new tokenizer (usually with `train_new_from_iterator`) and continue-pretrain the model.
- Fertility, coverage, OOV rate, and round-trip lossiness are the minimum diagnostics; downstream loss is the tiebreaker.
- Inspect your vocabulary before shipping. Half the "bad tokenizer" bugs in production are visible on a five-minute skim of the token list.
- The next chapter shifts from tokens to the models that consume them: encoder, encoder-decoder, and decoder-only transformers.
