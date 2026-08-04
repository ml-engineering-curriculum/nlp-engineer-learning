# Extractive Summarisation Baselines: LEAD-N, TextRank, and BERTSum

## Motivation

Before you fine-tune a 400M-parameter encoder-decoder to abstract from your data, run the two dumbest baselines you can think of. Extractive summarisation is embarrassingly effective on many corpora — famously so on news, where "just take the first three sentences" (LEAD-3) has beaten many published models on CNN/DailyMail ROUGE. Every ambitious summariser is measured against these baselines whether you build them or not; you may as well build them and know the number.

Extractive baselines also do three practical jobs no abstractive model can:

1. **They put a floor on your metrics.** If LEAD-3 scores ROUGE-L 30 and your fine-tuned BART scores 31, you have not built a summariser — you have built an expensive way to reformat the article.
2. **They diagnose your dataset.** A tiny gap between LEAD-3 and a strong extractive model tells you the corpus is extractive by construction; a big gap tells you paraphrase carries real signal.
3. **They cannot hallucinate.** For compliance-sensitive domains, an extractive summariser is often the *only* summariser you are allowed to ship without a heavy verification stack.

This chapter covers the three baselines you should be able to write in an afternoon: **LEAD-N** (positional), **TextRank / LexRank** (graph-based), and **BERTSumExt** (neural sentence classification).

## The extractive formulation

Given a source document $D$ of $n$ sentences $s_1, \dots, s_n$, produce a subset $S \subseteq \{1, \dots, n\}$ of sentence indices that (a) fits within a length budget (a token limit or a target sentence count $k$) and (b) maximises informativeness with respect to $D$.

The three families differ in how they choose $S$:

- **Positional (LEAD-N).** Pick the first $k$ sentences unconditionally.
- **Graph-based (TextRank, LexRank).** Build a graph over sentences, run a PageRank-style random walk, pick the top-$k$ by centrality.
- **Neural (BERTSumExt).** Score each sentence with a pretrained encoder + a binary classification head, pick the top-$k$ by score.

All three produce a summary that is a concatenation of source sentences. Provenance is free; hallucination is impossible; the ceiling is bounded by how well the "right" content can be assembled without paraphrase.

## Baseline 1: LEAD-N

The simplest baseline in the field. For each document, return the first $N$ sentences.

```python
def lead_n(document: str, n: int = 3, sent_tokenizer=None) -> str:
    sentences = sent_tokenizer(document)
    return " ".join(sentences[:n])
```

Trivial. Also strong: LEAD-3 was the state of the art for years on CNN/DailyMail because news is written with an inverted-pyramid structure — the most important facts appear first. See Narayan, Cohen & Lapata (2018) for the specific numbers.

Where LEAD-N is *weak*:

- **Dialogue.** The first three lines of a Slack thread are `hi`, `hi`, `so`. Use TextRank or a neural summariser.
- **Long, structured documents.** The first three sentences of a research paper are the title, the affiliation list, and half of the abstract's first sentence.
- **Non-Western news, technical documentation, transcripts.** Any genre without the inverted-pyramid convention.

Always ship LEAD-N on your dataset. It is the honesty check on every headline number that follows.

## Baseline 2: TextRank and LexRank

The graph-based family (Mihalcea & Tarau, 2004; Erkan & Radev, 2004) turns extractive summarisation into a centrality problem.

1. Represent every sentence as a bag of words (TextRank) or a TF-IDF vector (LexRank).
2. Compute a similarity score $w_{ij}$ between every pair of sentences $(s_i, s_j)$ — cosine for LexRank, word-overlap normalised by log-length for TextRank.
3. Build a weighted graph over sentences with $w_{ij}$ as edge weights.
4. Run a PageRank-style random walk to convergence. Each sentence gets a centrality score.
5. Return the top-$k$ sentences by score, presented in their original document order.

```python
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def lexrank(sentences: list[str], k: int = 3, threshold: float = 0.1) -> list[str]:
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sim   = cosine_similarity(tfidf)
    sim[sim < threshold] = 0.0                # discard weak edges
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph, alpha=0.85)
    ranked = sorted(range(len(sentences)),
                    key=lambda i: -scores[i])
    picked = sorted(ranked[:k])
    return [sentences[i] for i in picked]
```

The `sumy` library ships production-quality implementations of TextRank, LexRank, LSA, SumBasic, and KL-Sum. There is rarely a reason to write your own from scratch outside of instructional exercises.

TextRank / LexRank beat LEAD-N on dialogue, on long documents, and on any domain where importance is spread through the text rather than concentrated at the top. They lose to LEAD-N on news benchmarks.

## Baseline 3: BERTSumExt — neural extractive summarisation

BERTSumExt (Liu & Lapata, ["Text Summarization with Pretrained Encoders"](https://arxiv.org/abs/1908.08345), *EMNLP 2019*) is the canonical neural extractive summariser. The idea:

1. Encode every sentence of the document *together*, using a single BERT-style encoder with `[CLS]` tokens prepended to each sentence.
2. Extract the final-layer hidden state at each `[CLS]` position — one representation per sentence.
3. Pass those `[CLS]` vectors through a small transformer or LSTM ("inter-sentence encoder") to model discourse.
4. A binary classification head over each sentence position predicts $p(s_i \in S)$.

Training data is constructed by *oracle labelling*: for each `(document, gold_summary)` pair, greedily pick the subset of source sentences whose union maximises ROUGE against the gold summary. Those sentences become the positive labels; every other sentence is negative.

```python
from transformers import AutoModel, AutoTokenizer
import torch, torch.nn as nn

class BERTSumExt(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.inter_sent = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, cls_positions):
        h = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        # gather [CLS] hidden states — one per sentence
        cls_h = torch.stack([h[b, cls_positions[b]] for b in range(h.size(0))])
        sent_h = self.inter_sent(cls_h)
        logits = self.classifier(sent_h).squeeze(-1)                # (B, num_sents)
        return logits
```

At inference, take the top-$k$ sentences by logit, either with a fixed $k$ or with a trigram-blocking post-processor (Paulus et al., 2018) that suppresses picking a sentence whose trigrams already appear in the running summary.

BERTSumExt tends to beat LEAD-3 by 1–3 ROUGE points on CNN/DailyMail and by more on XSum-style corpora where the extractive ceiling is higher than any single positional heuristic would find.

## Which baseline to run when

- **News, corporate reports, structured documents with headlines-first conventions.** LEAD-N is a legitimate baseline and sometimes the winner. Run it first.
- **Dialogue, transcripts, unstructured long-form.** LEAD-N is a distractor. Go straight to TextRank/LexRank or BERTSumExt.
- **Compliance-sensitive domains where hallucination is disqualifying.** Extractive is often the *only* viable approach — ship the best of the three and treat abstractive as a stretch goal behind a faithfulness gate (chapter 12).
- **Any dataset before touching an encoder-decoder.** Run LEAD-N + TextRank as headline numbers before you spend a GPU-day fine-tuning BART. If your fine-tuned model does not beat both, you have a data or an evaluation bug, not a modelling one.

## Evaluating extractive summarisers

The metric protocols in chapter 10 apply — ROUGE-1/2/L against the reference summaries, with the same tokenisation and stemming rules as the papers you compare against. Two nuances specific to extractive systems:

- **Report a length-normalised score** (average summary tokens per document) alongside ROUGE. An extractive system that returns 200 tokens will beat one that returns 60 on recall-heavy metrics even if the 60-token one is more useful.
- **Compute the "oracle" extractive ROUGE.** Run the greedy oracle labeller on the dev set and evaluate the oracle summaries. This is the *ceiling* for any extractive system on this data. If your neural model is 2 ROUGE below oracle and 5 ROUGE above LEAD-3, you are close to the extractive limit; further gains require abstractive.

## The extractive ceiling and the case for abstractive

For each dataset, plot three numbers: LEAD-N, BERTSumExt-style neural extractive, and the greedy-oracle extractive ROUGE. The oracle is the highest score any extractive system can achieve; the abstractive systems in the next chapter regularly *beat it* on XSum and other paraphrase-heavy corpora.

That gap — abstractive scoring above the extractive oracle — is the case for the encoder-decoder recipe. Any summary that requires collapsing "won two out of three" and "lost the third" into "took the series" is impossible extractively and trivial abstractively. When the gap is small (CNN/DailyMail), extractive stays competitive; when the gap is large (XSum, dialogue), abstractive is the only path.

## Chapter summary

- Extractive summarisation selects existing sentences and cannot hallucinate; three baselines suffice for 90 % of ideation.
- **LEAD-N** is the strongest possible dumb baseline on inverted-pyramid text (news, reports). Ship it before anything else.
- **TextRank / LexRank** win on dialogue, transcripts, and long documents where importance is diffuse. Use `sumy` unless you're teaching.
- **BERTSumExt** is the strongest extractive baseline; train it against greedy-oracle labels and use trigram-blocking at inference.
- Report LEAD-N, TextRank, BERTSumExt, and oracle-extractive ROUGE together. The gap between BERTSumExt and oracle bounds your extractive headroom; the gap between oracle and your dreamt-of abstractive model is the case for the next chapter.
