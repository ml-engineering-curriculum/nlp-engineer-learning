# Multi-Stage Training and Cross-Encoder Distillation

## Motivation

Almost every strong open bi-encoder — E5, BGE, GTE, Nomic Embed, jina-embeddings-v3 — is trained in *stages*, not with a single loop. There is a *pretraining* stage on massive noisy pair data, a *supervised fine-tune* on cleaner labelled pairs, and, for the leaderboard-topping models, a *distillation* stage where a strong cross-encoder produces soft targets that pass ranking knowledge to the fast bi-encoder. Chapter 05 introduced the negatives; this chapter introduces the training-stage architecture that stitches them into a working pipeline.

The distillation part is the important one for anyone who wants a "small model that punches above its weight." A 22 M-parameter MiniLM distilled from a 400 M-parameter cross-encoder can hold its own against a 1 B-parameter unaligned bi-encoder on MTEB retrieval, at 20× the throughput. That is the whole story of the modern "small, cheap, good enough" retrieval stack.

## The standard three-stage pipeline

The published recipes converge on the following shape:

```
        ┌─── Stage 1 ────┐   ┌─── Stage 2 ────┐   ┌─── Stage 3 ────┐
        │ Pretrain on    │   │ Supervised     │   │ Cross-encoder  │
input   │ weakly labelled│──▶│ fine-tune on   │──▶│ distillation   │──▶ shipped
model   │ pair data      │   │ labelled pairs │   │ on ranked triples│  bi-encoder
        │ (billions of   │   │ + hard negs    │   │ + soft scores   │
        │  pairs, in-batch)  │ (chapter 05)   │   │                │
        └────────────────┘   └────────────────┘   └────────────────┘
```

Stage 1 gives the model a general notion of similarity. Stage 2 aligns it to labelled relevance judgments and hard negatives. Stage 3 transfers fine-grained ranking knowledge from a much larger cross-encoder.

Each stage has its own loss, batch size, data, and stopping criterion. Attempting to do all three in one loop is the most common self-inflicted quality loss in this space.

## Stage 1: weakly-supervised pretraining

The point of stage 1 is *not* to be right — it is to give the model exposure to hundreds of millions of pair-level similarity examples so that the encoder's representations are already retrieval-shaped when supervised training begins.

The data:

- **Common Crawl QA pairs** — `(question, title)` pairs mined from FAQ pages, sub-headings, `<h1>` / `<p>` pairs. E5 was trained on this.
- **Reddit `title` / `body` pairs** — (submission-title, top-comment) or (subreddit, post) pairs. Big, noisy, cheap.
- **StackExchange `question` / `answer` pairs** — cleaner than Reddit; still weakly labelled.
- **S2ORC citation pairs** (Lo et al., 2020) — (paper title, cited paper title). The bibliographic backbone of the scientific-embedding models (SPECTER, SciNCL).
- **Wikipedia section pairs** — (section-header, section-body) or (article-title, first-paragraph). Used by Contriever and many others.
- **NLI datasets** (SNLI, MNLI) — treat `(premise, entailment)` as positive pairs. Small but clean.
- **Amazon / Yelp reviews / product Q&A** — same-product pairs. Big and free where licences permit.

Typical stage-1 scale: 100 M–1 B pairs, batch size 1024–8192 (via `CachedMultipleNegativesRankingLoss`), 1–2 epochs, learning rate 2e-5, `MultipleNegativesRankingLoss` with in-batch negatives only. This stage does not use hard negatives — the point is *breadth*, and the noise in the data means "hard negative mining" mostly surfaces false negatives anyway.

The E5 paper (Wang et al., 2022) calls this stage *"contrastive pre-training on unsupervised pairs"*; the Contriever paper (Izacard et al., ["Unsupervised Dense Information Retrieval with Contrastive Learning"](https://arxiv.org/abs/2112.09118), 2022) is a well-documented pure-stage-1 recipe worth reading end-to-end.

## Stage 2: supervised fine-tuning with hard negatives

Now the labelled pairs come in, and hard-negative mining (chapter 05) kicks in.

Typical stage-2 data:

- **MS MARCO passage ranking** (Nguyen et al., 2016) — 500 k queries with 1 positive each, from Bing search logs. The reference supervised retrieval dataset.
- **Natural Questions** (Kwiatkowski et al., 2019) — 300 k real Google queries with Wikipedia passage answers.
- **HotpotQA** (Yang et al., 2018) — multi-hop QA with paragraph-level supporting evidence.
- **FEVER** — claim-evidence pairs from Wikipedia.
- **GooAQ, PAQ, TriviaQA, ELI5** — additional labelled QA-style pair datasets.

Typical hyperparameters:

- Batch size 128–512, with 1 positive + 7–15 mined hard negatives per query.
- `MultipleNegativesRankingLoss` (or `CachedMultipleNegativesRankingLoss`).
- Learning rate 1e-5–3e-5 with linear warmup.
- 1–3 epochs. Longer causes overfit to the training queries and hurts MTEB.
- Hard-negative refresh every 1 epoch (ANCE-style) or once end-of-stage-1 (single-shot).

Evaluate on held-out MS MARCO dev + BEIR (see chapter 08 for MTEB, which subsumes BEIR).

## Stage 3: cross-encoder distillation

The final stage — optional, but the one that produces most of the MTEB-leaderboard-topping models.

The idea (Reimers & Gurevych, ["Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation"](https://arxiv.org/abs/2004.09813), 2020, and generalised in many papers since): a bi-encoder cannot outperform a cross-encoder on ranking, but it can *match its ordering* on the pairs the cross-encoder was asked to score. Train the bi-encoder to produce the cross-encoder's *ordering*, not just a binary correct/incorrect signal.

The mechanic:

1. For each query, retrieve top-K candidates from the current bi-encoder (or BM25).
2. Score every (query, candidate) pair with a strong cross-encoder to get a soft score $s^\text{teacher}(q, d) \in \mathbb{R}$.
3. Train the bi-encoder with a loss that matches the *differences* in teacher scores across candidates.

The reference loss for this is `MarginMSELoss` (Hofstätter et al., ["Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation"](https://arxiv.org/abs/2010.02666), 2020), which trains the bi-encoder to reproduce the *margin* between a positive and a negative candidate:

$$
\mathcal{L}_\text{margin-MSE} = \left((s^\text{student}(q, d^+) - s^\text{student}(q, d^-)) - (s^\text{teacher}(q, d^+) - s^\text{teacher}(q, d^-))\right)^2
$$

Two properties matter:

- **Margin-based, not absolute.** The bi-encoder does not need to match the cross-encoder's *values* — only their differences. This is important because bi-encoder cosines and cross-encoder logits live on different scales.
- **Ranks negatives smoothly.** A near-positive negative gets a small margin; a truly off-topic negative gets a large margin. The bi-encoder learns "how wrong" each negative is, not just that it is wrong.

`sentence-transformers` API:

```python
from sentence_transformers import losses
loss = losses.MarginMSELoss(model)
```

The training example format is a triple `(query, pos, neg)` with an extra field `label = teacher_score(q, pos) - teacher_score(q, neg)`. `sentence-transformers` v3's `SentenceTransformerTrainer` handles this natively when you pass a dataset with an appropriately named label column.

An alternative loss family — often used in the MS MARCO papers — is **listwise KL divergence** (Zeng, Zamani & Vinay, 2022, and predecessors): match the softmax over teacher scores across a whole candidate list. Slightly more expressive than pairwise margin, slightly more sensitive to teacher noise. Both work; pick one and iterate.

## The "SimCSE" family: unsupervised pretraining without pairs

An entirely orthogonal training angle: what if you have *no* pair data at all, not even weak? **SimCSE** (Gao, Yao & Chen, ["SimCSE: Simple Contrastive Learning of Sentence Embeddings"](https://arxiv.org/abs/2104.08821), *EMNLP 2021*) solves this by *manufacturing* positives from a single input.

- **Unsupervised SimCSE.** Encode the same sentence twice, once with the model's dropout on. The two dropped-out encodings are the positive pair; all other sentences in the batch are negatives. Dropout is the augmentation.
- **Supervised SimCSE.** Use SNLI/MNLI entailment pairs as positives.

The trick is disarmingly simple. On STS benchmarks it beats a wide range of more complex approaches and set the modern standard for "start a sentence encoder from nothing." Reach for it when:

- You need a warm-start for a new domain encoder and only have monolingual text.
- You want a fast, cheap baseline before investing in labelled pairs.
- You want to teach a fine-tuned classifier's encoder to also produce useful sentence vectors as a byproduct.

**TSDAE** (Wang, Reimers & Gurevych, ["TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning"](https://arxiv.org/abs/2104.06979), *EMNLP 2021 Findings*) is the domain-adaptation cousin: pretrain by reconstructing the original sentence from a corrupted (deleted-token) version. Better than SimCSE for domain adaptation on niche corpora because it exposes the model to domain vocabulary, worse than SimCSE for pure STS.

Neither replaces stage 1–3 for production quality, but both are the right first move when you have zero labelled pairs.

## The full modern recipe, in one script's worth of pseudocode

```python
# ------------- Stage 1: weakly-supervised pretrain -------------
model = SentenceTransformer("microsoft/mpnet-base")     # backbone
train_stage1 = load_weakly_labelled_pairs()             # 300M pairs
trainer(
    model, train_stage1,
    loss=losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32),
    per_device_train_batch_size=1024,
    num_train_epochs=1,
    learning_rate=2e-5,
).train()

# ------------- Stage 2: supervised fine-tune -------------
train_stage2 = load_msmarco_and_nq_with_hard_negatives(
    model_for_mining=model,                             # ANCE-style self-mining
    num_negatives=7,
    cross_encoder_denoiser="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
trainer(
    model, train_stage2,
    loss=losses.MultipleNegativesRankingLoss(model),
    per_device_train_batch_size=128,
    num_train_epochs=2,
    learning_rate=1e-5,
).train()

# ------------- Stage 3: cross-encoder distillation -------------
teacher = CrossEncoder("BAAI/bge-reranker-large")
train_stage3 = build_distillation_dataset(
    queries, candidates_from_stage2_model, teacher,
)
trainer(
    model, train_stage3,
    loss=losses.MarginMSELoss(model),
    per_device_train_batch_size=256,
    num_train_epochs=1,
    learning_rate=5e-6,                                 # lower LR for distillation
).train()
```

Not exercise-01. Exercise-01 does *one* of these stages in depth. But this is the full shape you should have in your head when you read a model card for E5, BGE, or `text-embedding-3`.

## Practical trade-offs and when to skip stages

- **No labelled pairs at all?** Start with SimCSE or TSDAE, then GPL for domain adaptation (chapter 09). Skip stage 2, do a light stage 3 with a public cross-encoder.
- **Small labelled dataset (10–100 k pairs)?** Skip stage 1 (use a public E5/BGE base as your starting weights), do stage 2 heavily, skip stage 3.
- **Big labelled dataset + big compute?** Full three-stage recipe. Expect the biggest lift from stage 3.
- **Domain-specific corpus but no labels?** Stage 1 on your domain text via SimCSE/TSDAE, then GPL for stage 2 (chapter 09), skip stage 3.
- **Serving cost dominates?** Distil from a bigger `sentence-transformers` model into a smaller one using MSE-on-embeddings — exactly the recipe in Reimers & Gurevych (2020) for cross-lingual distillation, but with same-language teacher and student. Small quality loss, big throughput gain.

## Evaluation between stages

Never proceed to the next stage without evaluating the current one on a held-out MTEB slice or your target retrieval eval. The usual pattern:

- **After stage 1:** evaluate on MTEB STS. Retrieval numbers will still be poor; that's expected.
- **After stage 2:** evaluate on MTEB retrieval (BEIR subset — NFCorpus, SciFact, TREC-COVID at minimum). This is where you decide if the mined negatives are helping.
- **After stage 3:** evaluate on the same retrieval slice. The lift over stage 2 is your distillation ROI.

Chapter 08 covers MTEB properly. The point here: staged training only makes sense if you evaluate after each stage.

## Chapter summary

- Strong open embedding models are trained in stages: weakly-supervised pretrain → supervised fine-tune with hard negatives → cross-encoder distillation.
- Stage 1 uses massive noisy pair corpora (Reddit, StackExchange, Common Crawl QA), batch size 1000+, in-batch negatives only. Contriever and E5 documented recipes.
- Stage 2 uses cleanly labelled QA/retrieval datasets with mined hard negatives. Batch size 128–512.
- Stage 3 distils a strong cross-encoder into the bi-encoder with `MarginMSELoss` — the bi-encoder learns the cross-encoder's *ordering* of candidates. This is where the leaderboard-topping quality comes from.
- SimCSE (dropout-as-augmentation) and TSDAE (denoising autoencoder) are the "no-pair-data" pretraining recipes — useful for warm-starting new-domain encoders.
- Skip stages when supervision or compute forces it: no labels → SimCSE + GPL; small labels → skip stage 1; small serving budget → distil the teacher into a smaller student.
- Evaluate on MTEB between every stage. Staged training only works with staged evaluation.
