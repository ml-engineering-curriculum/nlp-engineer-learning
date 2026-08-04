# sentence-transformers, Pooling, and the Similarity Contract

## Motivation

`sentence-transformers` (Reimers & Gurevych, ["Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://arxiv.org/abs/1908.10084), *EMNLP 2019*) is the library the rest of this module lives inside. It is the reason `model.encode(texts)` works today, and it is what wraps every training loop, pooling head, loss, and evaluation harness we will touch. If you know one library from this module, know this one.

But the library is also opinionated. Its defaults — CLS-token vs. mean pooling, cosine vs. dot-product similarity, prefix conventions on the input, `normalize_embeddings=True` — are decisions you inherit whether or not you meant to. This chapter is the tour of the library and the decisions it makes for you, plus the three that you should always make explicitly for the model at hand.

## Why raw BERT is not a sentence encoder

The Sentence-BERT paper opens with a demonstration you should be able to reproduce from memory: mean-pooling BERT-base's last hidden states, or using the raw `[CLS]` token, yields sentence embeddings that are *worse than averaging GloVe vectors* on STS benchmarks. BERT was pretrained with masked-language-modelling and next-sentence-prediction; neither objective ever asks it to produce a similarity-preserving single vector for a whole sentence.

The Sentence-BERT recipe fixes this with two ingredients:

1. **A pooling layer** on top of the last transformer hidden states (mean, CLS, or max) that produces one fixed-size vector per input.
2. **Contrastive fine-tuning** (chapter 04) on sentence pairs with a similarity objective — so that "similar sentences get similar vectors" becomes something the model was actually trained to do.

Every model in the modern bi-encoder lineage — MPNet, MiniLM, BGE, E5, Nomic Embed — is a variant on that recipe. They differ in backbone, pooling, loss, training data, and negatives — but they all inherit the shape.

## The library's architecture

A `SentenceTransformer` is a Python `nn.Sequential` of *modules*, each transforming a batch of dicts. The three you will touch are:

| Module              | What it does                                                              |
|---------------------|---------------------------------------------------------------------------|
| `Transformer`       | Wraps a Hugging Face `AutoModel` — produces per-token hidden states.       |
| `Pooling`           | Reduces the token sequence to one vector per input (mean, CLS, max, or a combination). |
| `Normalize`         | L2-normalises the pooled vector to unit length. Optional.                  |
| `Dense` (optional)  | A linear projection layer, sometimes trained for dim-reduction (`128d`, `256d`). |

You can inspect any pretrained model's stack by loading it and printing:

```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-mpnet-base-v2")
print(m)
# SentenceTransformer(
#   (0): Transformer({'max_seq_length': 384, ...})
#   (1): Pooling({'pooling_mode_mean_tokens': True, ...})
#   (2): Normalize()
# )
```

That printout is a *contract*, not a decoration. It tells you what pooling the model expects, whether outputs are L2-normalised (so cosine and dot product are equivalent), and what the max input length is. Reading it before you use a model saves an entire class of "why do my scores look wrong" bugs.

## Pooling: the three choices

Three pooling modes cover 99 % of what you will see. Given per-token hidden states $H = [h_1, \dots, h_L]$ from a transformer:

- **Mean pooling.** $v = \frac{1}{L} \sum_{i=1}^L h_i$, masked over real (non-padding) tokens. The `sentence-transformers` default. Best-supported by empirical evidence for sentence similarity.
- **CLS pooling.** $v = h_\text{CLS}$ — the vector at the special first token. Default for `bert-base-*` classification, and used by many BGE and E5 variants which were trained to produce useful CLS representations. Empty out-of-the-box for BERT until fine-tuned.
- **Max pooling.** $v_j = \max_i h_{ij}$, element-wise max over token dim. Rare in modern models but occasionally combined with mean pooling in older Sentence-BERT variants.

You do not get to pick pooling at inference time — you have to use whatever the model was trained with. If you build a `SentenceTransformer` from a raw HF encoder, you pick pooling once at construction and stick with it:

```python
from sentence_transformers import SentenceTransformer, models

word_embed = models.Transformer("microsoft/deberta-v3-base", max_seq_length=384)
pooling    = models.Pooling(
    word_embed.get_word_embedding_dimension(),
    pooling_mode="mean",              # or "cls", "max"
)
model = SentenceTransformer(modules=[word_embed, pooling])
```

## Why pooling matters — a debugging story

If you take a model trained with mean pooling and call it with CLS pooling, or vice versa, the vectors are *garbage-looking-plausible* — same dimension, same magnitudes, similar cosine ranges — but not the ones the model was trained to emit. Every downstream ranking will silently degrade by 10–30 % depending on the task. The library protects you from this only if you load a full published `SentenceTransformer` config; the moment you pick the encoder yourself, you own the pooling.

Concrete symptom: BGE and E5 checkpoints exist as both Hugging Face `AutoModel` checkpoints (loadable with `transformers`) and full `SentenceTransformer` bundles. If you load `intfloat/e5-large-v2` with plain `transformers.AutoModel`, you must know that E5 uses `pooling_mode=mean` and normalises the output, and re-apply both. Load it via `SentenceTransformer("intfloat/e5-large-v2")` and both are handled automatically.

## Input prefixes: a modern-training convention

Several 2022–2024 embedding models require a *task-prefix* on the input. This looks silly and is deeply important:

| Model family                  | Query prefix          | Passage prefix         |
|-------------------------------|-----------------------|------------------------|
| E5 (`intfloat/e5-*`)          | `"query: "`           | `"passage: "`          |
| BGE English (`BAAI/bge-*-en-*`) | `"Represent this sentence for searching relevant passages: "` (query only) | (none)             |
| BGE M3 (`BAAI/bge-m3`)         | (none — model handles) | (none — model handles) |
| Nomic Embed v1                | `"search_query: "`    | `"search_document: "`  |
| GTE (`thenlper/gte-*`)         | (none)                | (none)                 |

These are not decoration. E5 was trained with those literal prefixes; feeding `"where is the eiffel tower"` (no prefix) at query time and `"the eiffel tower is..."` (no prefix) at index time drops MTEB retrieval scores by several points. The prefixes tell the model "this is a query" vs. "this is a document," which is *especially* useful for asymmetric retrieval where query and document distributions look very different.

The rule: for every embedding model you deploy, read the model card, find the query and passage prefix (or confirm none), and *encode with them explicitly* at both index time and query time. Do not let the library "default" a prefix in — some do, some don't.

## Similarity functions and the normalisation contract

Three similarity functions, one contract:

| Function                     | Formula                                | When it is right                                           |
|------------------------------|----------------------------------------|------------------------------------------------------------|
| Cosine similarity            | $u \cdot v \; / \; (\|u\| \|v\|)$        | Any model trained with cosine loss (SBERT, MPNet, BGE, E5) |
| Dot product                  | $u \cdot v$                            | Any model trained with dot-product loss (some DPR variants, Cohere `input_type=search_document`) |
| Negative Euclidean distance  | $-\|u - v\|_2$                          | Rare; some older BERT-style STS models                     |

The important simplification: **if vectors are L2-normalised to unit length, cosine and dot product are equivalent.** That is why every modern bi-encoder either normalises at the output (a `Normalize` module) or expects the caller to normalise before comparison. `normalize_embeddings=True` on `model.encode(...)` handles it for you.

Two failure modes here are common enough to memorise:

- **Comparing normalised vectors with raw dot product without normalising the query.** Cosine on the doc side, dot on the query side; ranking works but with subtly wrong magnitudes when combined with score-based filtering thresholds.
- **Indexing raw (unnormalised) vectors into a FAISS `IndexFlatIP` index expecting `IndexFlatL2` behaviour, or vice versa.** FAISS respects whatever geometry you pick — it does not check. Chapter 11.

## The reference encode call

The library's `model.encode()` handles batching, tokenisation, GPU transfer, and pooling. The full argument list you should know:

```python
model.encode(
    texts,                          # str or list[str]
    batch_size=32,                  # tune to VRAM
    show_progress_bar=True,
    convert_to_tensor=True,         # keeps output on GPU as torch.Tensor
    convert_to_numpy=False,         # or True to move to CPU as np.array
    normalize_embeddings=True,      # L2-normalise output
    device="cuda",                  # or "cpu", "mps"
    output_value="sentence_embedding", # or "token_embeddings" for ColBERT-style
    precision="float32",            # or "float16", "int8", "ubinary", "binary" (v2.7+)
    prompt_name=None,               # look up a named prompt from model config
    prompt=None,                    # explicit prefix override
)
```

Every one of those flags either affects the numeric output or the throughput. `precision="binary"` in particular is chapter 11 territory — a 32× compression in exchange for a small quality drop, and one of the first knobs to reach for when serving costs matter.

## Building a bi-encoder from a raw HF checkpoint

Sometimes the checkpoint you want isn't published as a `SentenceTransformer` bundle — it's a bare HF encoder trained by a research group or a domain-adapted PubMedBERT you fine-tuned yourself. You wrap it manually:

```python
from sentence_transformers import SentenceTransformer, models

def wrap_encoder(hf_name: str, pooling_mode: str = "mean", normalize: bool = True) -> SentenceTransformer:
    word_embed = models.Transformer(hf_name, max_seq_length=512)
    pooling    = models.Pooling(
        word_embed.get_word_embedding_dimension(),
        pooling_mode=pooling_mode,
    )
    modules = [word_embed, pooling]
    if normalize:
        modules.append(models.Normalize())
    return SentenceTransformer(modules=modules)

bio = wrap_encoder("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
```

This is the constructor exercise 04 asks you to use for the domain-adaptation exercise.

## The training loop, one screen

`sentence-transformers` v3 unified around the Hugging Face `Trainer` API. A minimal training script:

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset

model = SentenceTransformer("microsoft/mpnet-base")

train = Dataset.from_dict({
    "anchor":   ["What is the capital of France?", "Who wrote Hamlet?"],
    "positive": ["The capital of France is Paris.", "Shakespeare wrote Hamlet."],
})

loss = losses.MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="./bi-encoder",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
)

SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train, loss=loss,
).train()
```

Everything meaningful in this module — the loss (chapter 04), the batch size (chapter 04–05), the negatives (chapter 05), the training curriculum (chapter 06) — is a variation on that skeleton. Read the `sentence-transformers` v3 docs before assuming any argument does what its name implies.

## Long inputs, truncation, and the max-length trap

Every encoder has a hard token limit — 512 for BERT-family (all the classic sentence-transformers), 8192 for Nomic Embed and some newer long-context encoders, 32 k for a handful of frontier models. If you feed longer inputs, the library silently truncates and only the first `max_seq_length` tokens survive.

This is the single most common bug when moving from a benchmark corpus (short sentences) to production documents (long articles):

- **Symptom.** Retrieval quality on long documents is inexplicably bad. The end of the document — often the most content-bearing part — never made it into the vector.
- **Diagnosis.** Log the token count per document vs. `model.max_seq_length` and count truncated docs.
- **Fix.** Chunk documents before encoding (`rag-engineer` territory), or switch to a long-context encoder (Nomic, `jina-embeddings-v3`, `bge-m3` supports 8k), or use a hierarchical encoder that pools chunk vectors.

Long-context embedding is a live research area. Do not assume any bi-encoder handles it out of the box.

## Chapter summary

- `sentence-transformers` is a stack of `Transformer → Pooling → (Normalize)` modules and is the training and inference substrate for the whole module.
- Raw BERT is not a sentence encoder. The Sentence-BERT recipe adds pooling and contrastive fine-tuning.
- Pooling mode (mean / CLS / max) is a *training-time contract*; you cannot change it at inference without silently degrading quality.
- Modern models often require task-prefixes (`"query: "`, `"passage: "`). Read the model card and apply them explicitly at both index and query time.
- Cosine and dot product are equivalent iff vectors are L2-normalised. Every modern bi-encoder either normalises internally or expects `normalize_embeddings=True`.
- `model.encode()` batches, pools, normalises, and (v2.7+) quantises. Every one of its flags is meaningful; know them all.
- Wrap any HF encoder into a `SentenceTransformer` with explicit `Pooling` and `Normalize` modules when the checkpoint is not pre-bundled.
- The maximum sequence length is a hard trap. Log truncation rates before trusting retrieval quality on long documents.
