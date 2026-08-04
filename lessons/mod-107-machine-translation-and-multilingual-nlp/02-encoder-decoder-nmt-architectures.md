# Encoder-Decoder NMT Architectures: Marian, mBART, NLLB, M2M-100

## Motivation

Almost every NMT system you will build or ship this decade is an encoder-decoder transformer. The four checkpoints below cover the practical span — from a 74 M-parameter bilingual Marian model that fits on a Raspberry Pi to a 3.3 B-parameter NLLB checkpoint that translates between 200 languages — and all four load through the same `AutoModelForSeq2SeqLM` / `AutoTokenizer` API. Choosing between them is a decision about *language coverage*, *quality per parameter*, and *how you address the target language at inference time*. This chapter is the model-selection guide.

## The four lineages

| Model family                                              | Params (typical)          | Coverage             | Direction              | Tokeniser                       | Language addressing                                                          |
|-----------------------------------------------------------|---------------------------|----------------------|------------------------|---------------------------------|------------------------------------------------------------------------------|
| **Marian** `Helsinki-NLP/opus-mt-*`                       | ~74 M                     | Usually one pair     | Bilingual              | SentencePiece per pair          | None — model *is* the pair.                                                  |
| **mBART-25 / mBART-50**                                   | ~610 M                    | 25 / 50 languages    | Many-to-many (mBART-50)| SentencePiece 250 k             | `src_lang` / `tgt_lang` on tokeniser; language codes like `en_XX`, `de_DE`.  |
| **M2M-100**                                               | 418 M / 1.2 B             | 100 languages        | Many-to-many           | SentencePiece 128 k             | `forced_bos_token_id = tokenizer.get_lang_id(tgt)`; codes like `en`, `de`.   |
| **NLLB-200** `facebook/nllb-200-*`                        | 600 M / 1.3 B / 3.3 B     | 200 languages        | Many-to-many           | SentencePiece 256 k             | `src_lang` on tokeniser + `forced_bos_token_id`; BCP-47-ish codes (`eng_Latn`, `deu_Latn`, `zho_Hans`). |

All four share the same transformer encoder-decoder scaffolding — a stack of encoder blocks (self-attention + FFN), a stack of decoder blocks (masked self-attention + cross-attention + FFN), a shared source-target embedding, and a linear+softmax head over the vocabulary. The differences that matter to you as an engineer are (a) how the model is told *which* language it should be producing and (b) how many languages the tokeniser and shared embedding are trying to cover at once.

## Marian: the small, fast bilingual default

Marian NMT is the C++ training and inference framework that produced the **OPUS-MT** family of ~1 500 open pre-trained bilingual translators. In `transformers`, they load as `MarianMTModel` / `MarianTokenizer` under names like `Helsinki-NLP/opus-mt-en-de`.

```python
from transformers import MarianMTModel, MarianTokenizer

name = "Helsinki-NLP/opus-mt-en-de"
tok  = MarianTokenizer.from_pretrained(name)
mdl  = MarianMTModel.from_pretrained(name)

batch  = tok(["The board approved the amendment."], return_tensors="pt")
out    = mdl.generate(**batch, num_beams=4)
print(tok.batch_decode(out, skip_special_tokens=True))
# ['Der Vorstand hat den Änderungsantrag angenommen.']
```

Why you would reach for it:

- **Latency and memory.** ~74 M parameters. Fits into <200 MB fp16 memory. Comfortable on CPU with `int8` quantisation.
- **Simplicity.** No language code plumbing — the model *is* the pair.
- **Coverage of pairs the big models undertrained.** OPUS-MT has explicit checkpoints for pairs (Finnish ↔ Estonian, Basque ↔ Spanish, Chinese ↔ Japanese) that mid-size NLLB variants may lag on.

Why you would not:

- **One-per-pair explosion.** Serving 100 language pairs means 100 model instances. Multi-lingual NLLB serves them from one.
- **Older training data.** OPUS-MT models were retrained periodically but not on the scale NLLB was.

## mBART / mBART-50: multilingual denoising encoder-decoder

**mBART** (Liu et al., ["Multilingual Denoising Pre-training for Neural Machine Translation"](https://arxiv.org/abs/2001.08210), *TACL 2020*) pretrains a single encoder-decoder on monolingual corpora across 25 languages with a **BART-style denoising objective** — corrupt the input with span masking and sentence permutation, reconstruct. **mBART-50** (Tang et al., 2020) extended this to 50 languages and added a many-to-many MT fine-tune (`mbart-large-50-many-to-many-mmt`).

```python
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

name = "facebook/mbart-large-50-many-to-many-mmt"
tok  = MBart50TokenizerFast.from_pretrained(name, src_lang="en_XX")
mdl  = MBartForConditionalGeneration.from_pretrained(name)

batch  = tok("The board approved the amendment.", return_tensors="pt")
out    = mdl.generate(
    **batch,
    forced_bos_token_id=tok.lang_code_to_id["de_DE"],
    num_beams=4,
)
print(tok.batch_decode(out, skip_special_tokens=True))
```

Two model-specific things worth internalising:

- **Language codes are the tokeniser's problem.** `src_lang="en_XX"` on the tokeniser prepends the source-language BOS. `forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]` at generation forces the *decoder* to start with the target-language token. Skip either and you get random-language output.
- **The 50-language codes are mBART-specific.** `en_XX`, `de_DE`, `zh_CN`, etc. — not BCP-47, not ISO 639-1, not what NLLB uses. Every family invented its own.

Where mBART shines: general-purpose multilingual generation beyond pure MT — summarisation (multilingual XSum), paraphrase, style transfer. When you want a *multilingual encoder-decoder* that is not narrowly a translator, mBART-50 is the modern default in the mBART family; mT5 is the alternative from Google (chapter 07).

## M2M-100: the first massively many-to-many

**M2M-100** (Fan et al., ["Beyond English-Centric Multilingual Machine Translation"](https://arxiv.org/abs/2010.11125), *JMLR 2021*) was Meta's first massively many-to-many MT system — 100 languages, non-English-pivot training (Spanish ↔ Japanese without going through English), 418 M / 1.2 B / 12 B parameter tiers on Hugging Face.

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

name = "facebook/m2m100_418M"
tok  = M2M100Tokenizer.from_pretrained(name)
tok.src_lang = "en"

mdl  = M2M100ForConditionalGeneration.from_pretrained(name)
batch = tok("The board approved the amendment.", return_tensors="pt")
out   = mdl.generate(
    **batch,
    forced_bos_token_id=tok.get_lang_id("ja"),
    num_beams=4,
)
```

M2M-100 was a landmark because it proved that direct non-English MT beats pivoting through English. In practice today it has been largely superseded by NLLB-200 (same team, more languages, better quality). Use M2M-100 if (a) you specifically need one of its supported languages that NLLB does not cover well, (b) you are working from a paper that reports M2M-100 numbers, or (c) parameter budget forces `418M`.

## NLLB-200: the modern many-to-many default

**NLLB-200** (NLLB Team et al., ["No Language Left Behind"](https://arxiv.org/abs/2207.04672), 2022) is the current baseline for multilingual MT — 200 languages, direct translation between any pair, and released alongside **FLORES-200**, the reference evaluation benchmark. Parameter tiers on Hugging Face:

- `facebook/nllb-200-distilled-600M` — CPU-friendly, distilled from the 3.3 B teacher. Reasonable quality; the default "small NLLB" choice.
- `facebook/nllb-200-1.3B` — a strong general-purpose choice.
- `facebook/nllb-200-3.3B` — the flagship publicly-released checkpoint; best quality for the model card but heavy at inference.
- `facebook/nllb-200-distilled-1.3B` — distilled 1.3 B; a good latency/quality compromise.
- 54.5 B mixture-of-experts checkpoint — research-only, not on the Hub.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

name = "facebook/nllb-200-distilled-600M"
tok  = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
mdl  = AutoModelForSeq2SeqLM.from_pretrained(name)

batch = tok("The board approved the amendment.", return_tensors="pt")
out   = mdl.generate(
    **batch,
    forced_bos_token_id=tok.convert_tokens_to_ids("deu_Latn"),
    max_new_tokens=128,
    num_beams=4,
)
print(tok.batch_decode(out, skip_special_tokens=True))
```

The things worth memorising:

- **NLLB codes are `language_Script`** where `language` is an ISO 639-3 code and `Script` is an ISO 15924 code — `eng_Latn`, `deu_Latn`, `zho_Hans`, `zho_Hant`, `arb_Arab`, `hin_Deva`, `pes_Arab` (Iranian Persian), `bod_Tibt`. This is close to BCP-47 but with an underscore separator. The full list of 200 codes is in the model card.
- **`src_lang` on the tokeniser + `forced_bos_token_id` on `generate` are both required.** Forgetting either produces silently wrong output (the model translates from the wrong assumed source or into whatever language the model guesses).
- **`forced_bos_token_id=tok.convert_tokens_to_ids("deu_Latn")`** — NLLB does not expose a `lang_code_to_id` helper the way mBART does; you look up the language code as a normal special token.

Use NLLB-200 as the default multilingual MT starting point unless you have a specific reason not to.

## The three ways models tell you a language code

The single most bug-prone aspect of multilingual MT is the language-code plumbing. It differs across families and each has its own convention:

| Family      | Where you set source                      | Where you set target                                      | Code format                                    |
|-------------|-------------------------------------------|-----------------------------------------------------------|------------------------------------------------|
| Marian      | Implicit — model = pair                    | Implicit — model = pair                                    | N/A                                            |
| mBART-50    | `tokenizer.src_lang="en_XX"`               | `forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]`   | `{iso639-1}_{country}` custom (`en_XX`, `zh_CN`) |
| M2M-100     | `tokenizer.src_lang="en"`                  | `forced_bos_token_id=tokenizer.get_lang_id("de")`          | ISO 639-1 (`en`, `de`, `zh`)                    |
| NLLB-200    | `tokenizer.src_lang="eng_Latn"`            | `forced_bos_token_id=tok.convert_tokens_to_ids("deu_Latn")`| ISO 639-3 + ISO 15924 (`eng_Latn`, `zho_Hans`)  |

Wrap the plumbing in a small helper so you never write raw language codes at the call site — that is where the bugs come from.

## Vocabulary size and its cost

All four lineages use SentencePiece (Kudo & Richardson, ["SentencePiece: A simple and language independent subword tokenizer"](https://arxiv.org/abs/1808.06226), *EMNLP 2018*) with a shared source-target vocabulary. Sizes:

- Marian OPUS-MT: ~32 k per-pair
- M2M-100: ~128 k
- mBART-50: ~250 k
- NLLB-200: ~256 k

Vocabulary size is not free:

- **Embedding + LM head.** For a 512-dim model, a 256 k vocabulary is ~130 M parameters *just for the embedding matrix* — a quarter of the whole `distilled-600M` checkpoint. Fine-tuning it dominates memory.
- **Tokens per sentence.** A bigger multilingual SentencePiece has more full-word pieces for common languages; a smaller one shatters into subwords. Rough guide: NLLB tokenises English at ~1.2 tokens/word, Japanese at ~2.5–3, Amharic at ~4+. Budget context window accordingly.
- **Batching.** Longer token sequences (low-resource languages, non-Latin scripts) mean smaller batches at the same VRAM.

For high-resource pairs with production latency budgets, a small Marian model beats a 3.3 B NLLB on cost-per-token by an order of magnitude — often at similar quality *in-domain*. For coverage or unusual pairs, NLLB is the answer.

## Reference `translate()` helper

A single function you can adapt to whichever family you are on. All the family-specific plumbing lives here; call sites stay clean.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def build_translator(model_name: str, src_lang: str, tgt_lang: str):
    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if "nllb" in model_name:
        forced_bos = tok.convert_tokens_to_ids(tgt_lang)
    elif "mbart" in model_name:
        forced_bos = tok.lang_code_to_id[tgt_lang]
    elif "m2m100" in model_name:
        forced_bos = tok.get_lang_id(tgt_lang)
    else:                                             # Marian: no forced BOS
        forced_bos = None

    def translate(texts, num_beams=4, max_new_tokens=256):
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True)
        kwargs = {"num_beams": num_beams, "max_new_tokens": max_new_tokens}
        if forced_bos is not None:
            kwargs["forced_bos_token_id"] = forced_bos
        out = mdl.generate(**batch, **kwargs)
        return tok.batch_decode(out, skip_special_tokens=True)

    return translate
```

Every exercise and lab in this module builds on top of a helper like this. Do not skip it — hardcoding `forced_bos_token_id` at every call site is the fastest route to a silent regression when you swap model families.

## What each family is best at

| If you need…                                                | Reach for                                                      |
|-------------------------------------------------------------|----------------------------------------------------------------|
| One high-resource pair, latency < 100 ms, small footprint    | `Helsinki-NLP/opus-mt-<src>-<tgt>` (Marian) with int8 quant.   |
| Many pairs served from one checkpoint, quality > speed       | `facebook/nllb-200-1.3B` or `-3.3B`.                            |
| Many pairs, latency-sensitive                                | `facebook/nllb-200-distilled-600M` or `-distilled-1.3B`.        |
| Multilingual *generation* beyond MT (summ., paraphrase)      | `facebook/mbart-large-50` or `google/mt5-large` (chapter 07).   |
| A low-resource language pair not well-served elsewhere       | Start with NLLB; fall back to OPUS-MT if the pair exists.       |
| Extremely low-resource, source is monolingual-only           | NLLB base + back-translation (chapter 04).                      |

## Common failure modes and their fixes

- **Empty or wrong-language output.** You forgot `forced_bos_token_id` (mBART / M2M / NLLB) or set `src_lang` on the wrong side. Wrap the plumbing in a helper as above and never touch raw codes at the call site.
- **Truncated translations.** `max_new_tokens` default is often 20. Set explicitly — 128 for sentences, 512+ for paragraphs. Also set `max_length` on the tokeniser (typically 1024 for these models; anything longer is truncated silently).
- **Non-Latin languages come out mangled.** Almost always a tokenisation issue upstream (Unicode normalisation, mixed script). Chapter 09.
- **Model with a 250 k vocabulary trains OOM.** The embedding is a quarter of your parameters. Try `gradient_checkpointing_enable()` on the model, drop batch size, use LoRA on the transformer blocks only (chapter 05).
- **Same input produces different tokenisation across processes.** SentencePiece is deterministic, but `src_lang` mutates the tokeniser instance. If you share a tokeniser across a translation pool, make it read-only after construction.

## Chapter summary

- Four encoder-decoder families cover 99 % of production NMT: Marian (small bilingual), mBART / mBART-50 (multilingual encoder-decoder for generation), M2M-100 (first many-to-many), NLLB-200 (modern many-to-many default).
- Each family invented its own language-code convention. Wrap the plumbing in a helper; never hardcode codes at the call site.
- Vocabulary size dominates memory for the big multilingual checkpoints — 256 k tokens is ~130 M embedding parameters.
- NLLB-200 is the modern default when you need coverage; Marian is the modern default when you need per-pair speed. mBART-50 is the modern default when you need *generation* in many languages and not just translation.
- The FLORES-200 benchmark (chapter 12) is the reference evaluation set for the whole massively-multilingual tier.
