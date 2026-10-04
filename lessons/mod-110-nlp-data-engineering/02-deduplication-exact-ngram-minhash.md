# Deduplication: Exact, N-gram, and MinHash

## Motivation

Nobody argues about whether to dedup a training corpus any more. The question is which of the three tiers you run, in which order, with what thresholds. The paper that made the field take dedup seriously is Lee et al., ["Deduplicating Training Data Makes Language Models Better"](https://arxiv.org/abs/2107.06499), *ACL 2022* — on C4 and RealNews they showed that removing exact and near-duplicate text (a) reduces how often a trained model emits verbatim training strings by an order of magnitude, (b) improves perplexity on held-out data, and (c) measurably shrinks eval-set contamination because the eval set itself overlaps the train set less often.

Carlini et al., ["Quantifying Memorization Across Neural Language Models"](https://arxiv.org/abs/2202.07646), *ICLR 2023*, pushed the knife further: the probability that a model emits a given training string verbatim grows roughly log-linearly with the number of duplicates of that string in training. If a snippet appears once, you probably cannot extract it; if it appears a few hundred times, you almost certainly can. Dedup is therefore (i) a quality lever, (ii) a privacy lever, and (iii) an evaluation-integrity lever.

This chapter walks the three tiers you will actually run — exact, substring/n-gram, MinHash + LSH — plus the graph problem that falls out of near-dup detection. Boilerplate removal and language routing are related but separate; see chapters [04](04-deboilerplating-and-document-quality.md) and [03](03-language-routing-and-corpus-shards.md).

## What "duplicate" means at corpus scale

Three different equivalences, each catching a different failure mode:

| Tier       | Equivalence                                              | Catches                                              | Misses                                           |
|------------|----------------------------------------------------------|------------------------------------------------------|--------------------------------------------------|
| Exact      | normalised byte-identity                                 | republished articles, mirror sites, cached dumps     | anything with a timestamp, a tracker, a diff    |
| Substring  | one document contains a long span of another             | boilerplate-free copies with injected ads, templates | reordered or lightly paraphrased                 |
| Near-dup   | Jaccard over k-shingles above a threshold                | re-templated copies, paraphrases, OCR variants      | semantic duplicates that share no shingles      |

A corpus-wide dedup pass almost always runs all three, in that order, because each tier is cheap enough to run on the output of the previous one and each catches a strictly different pattern. The ordering matters: exact dedup collapses the corpus hard enough that the subsequent substring and MinHash passes have one or two orders of magnitude less work to do.

## Order of operations in a pipeline

A CCNet-style pipeline (Wenzek et al., ["CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data"](https://arxiv.org/abs/1911.00359), *LREC 2020*) does:

```
raw crawl
  -> HTML extraction + basic cleaning
  -> paragraph-level exact dedup (across the whole shard set)
  -> language identification + routing  (chapter 03)
  -> perplexity-based quality filter    (chapter 04)
  -> per-language near-dup pass
```

Why dedup before routing? Because a popular English paragraph that is mirrored across a hundred domains should be collapsed to one copy *before* you spend LID cycles on each copy. And because routing-then-dedup forces the near-dup pass to run per-language, which is fine — the LSH is sharded by language anyway — but you want the exact pass to see the whole corpus so that cross-language duplicates (user-agent strings, cookie banners, license notices rendered in English on non-English pages) collapse too.

Cross-document dedup is a **corpus-wide** operation. You cannot dedup shard-by-shard and call it done, because near-duplicates cluster across shards in a Zipfian way: a single news wire story shows up on 50 domains spread across every shard you have.

## Tier 1: exact deduplication

The algorithm is: hash a normalised byte stream of each document (or paragraph), group by hash, keep one representative per group.

The three decisions you have to make:

### 1. Normalisation

Normalise aggressively or you will leave 90% of exact duplicates on the floor. A pragmatic recipe:

```python
import hashlib
import unicodedata

def normalise(text: str) -> bytes:
    # Unicode NFC so composed/decomposed accents collapse.
    text = unicodedata.normalize("NFC", text)
    # Collapse runs of whitespace to a single space.
    text = " ".join(text.split())
    # Optional: casefold for case-insensitive dedup. Decide once, per corpus.
    text = text.casefold()
    return text.encode("utf-8")

def doc_hash(text: str) -> str:
    return hashlib.sha256(normalise(text)).hexdigest()
```

Three things to notice:

- Whitespace collapsing is almost always wanted. Scrapers emit the same paragraph with different trailing newlines, non-breaking spaces, and tab-vs-space indent.
- NFC is cheap and makes "café" (one code point) equal to "café" (two code points). You will see both in the same corpus.
- Casefolding is a judgement call. For training-data dedup it is usually fine; for anything evaluation-adjacent, leave case alone so you do not quietly conflate "polish" and "Polish".

### 2. Granularity: document or paragraph

Document-level dedup is one hash per document. Fast, cheap, misses the most common pattern in web data: the same paragraph (a license footer, a cookie banner, a news-wire boilerplate) repeated across millions of otherwise-distinct documents.

Paragraph-level dedup is one hash per paragraph (split on blank lines, or on a sentence tokeniser). You keep the first instance of each paragraph and drop subsequent copies, which can leave documents with holes — some pipelines reassemble, others drop any document that loses more than X% of its paragraphs.

Lee et al. report that paragraph-level is where the big wins live. The intuition: web corpora have a long tail of partially-overlapping documents, and no document-level hash catches that.

### 3. Hash choice and distribution

For the exact pass, both SHA256 (crypto, 32 bytes) and MurmurHash3 (non-crypto, 16 bytes with `mmh3.hash128`) work. MurmurHash is 5-10x faster per hash and the collision probability for a corpus of billions of paragraphs is negligible <!-- needs-research: birthday-bound numbers for mmh3_128 at 10^10 inputs --> ; SHA256 gives you a stable identifier you can use as a primary key across systems and over years. Use SHA256 when hashes outlive the pipeline (e.g., for dataset cards, see chapter [11](11-dataset-cards-datasheets-and-licensing.md)); MurmurHash when they are a one-shot grouping key.

Shard the dedup by a prefix of the hash. Each worker only sees paragraphs whose hash starts with its assigned bytes:

```bash
# Pseudo: split by 2-byte hash prefix across 256 workers.
for shard in input/*.jsonl; do
    jq -r '.text' "$shard" \
      | python normalise_and_hash.py \
      | awk -v OFS='\t' '{print substr($1,1,2), $0}' \
      > "work/${shard##*/}.pre"
done
# Then sort-merge by shard prefix; each prefix is independent work.
```

With a 2-byte prefix you get 65,536 independent groups and perfectly linear scaling until you saturate disk. MapReduce, Spark, Ray, Beam, any framework that gives you a group-by on a hash — all fine. The *idea* is prefix-sharding; the framework is a detail.

## Tier 2: substring and n-gram overlap

Exact dedup collapses byte-identical paragraphs. Substring dedup collapses the case where document A contains document B as a span, or where A and B share a long contiguous span even though neither is a subset of the other.

The Lee et al. tool, [`deduplicate-text-datasets`](https://github.com/google-research/deduplicate-text-datasets), builds a **suffix array** over the entire corpus and reports every pair of substrings of length ≥ 50 bytes that occurs in two different documents. Suffix arrays give you this in O(n log n) build time and O(k) query per hit, where n is the total corpus length — tractable up to multi-terabyte scale on a single large box.

The policy question on top of the index: when you find a span of length L shared between A and B, what do you remove?

- Remove the span from the shorter document, keeping the longer intact.
- Remove the span from both documents and keep the surrounding context.
- Drop the shorter document entirely if more than X% of its length is covered by shared spans.

The last policy is what Lee et al. use in practice: a document is dropped if any 50-token span appears elsewhere in the corpus, which is aggressive but what you want for pre-training. For instruction tuning or domain adaptation it is often too aggressive — a shared code snippet or stock disclaimer gets a whole document killed — so you dial up the "fraction-of-document covered" threshold instead.

A lightweight alternative when you cannot afford a corpus-wide suffix array: hash every overlapping 50-token window (or 50-character window for CJK) and treat window-level hash collisions as duplicate evidence. This is strictly weaker than a suffix array — it only catches spans that align to your window boundaries — but it parallelises exactly like Tier 1 and is often good enough.

```python
# Rolling 50-token window hashes, keyed by document.
def window_hashes(tokens, w=50, step=1):
    for i in range(0, len(tokens) - w + 1, step):
        yield hashlib.blake2b(" ".join(tokens[i:i+w]).encode(),
                              digest_size=16).hexdigest()
```

Group the output by window hash, and any hash that appears across more than one document is a duplicate-span candidate. In practice you set a `step` larger than 1 (say 10 or 25) to trade recall for cost.

## Tier 3: near-duplicate detection with MinHash + LSH

Exact and substring dedup both require byte-level overlap. They do not catch the case of two documents that share 80% of their content but with scattered edits — a news article republished with the lede rewritten, a product description templated across SKUs, an OCR'd page where every third word has a different error. For these you need an *approximate* set-similarity structure.

The classical reference is Broder, "On the Resemblance and Containment of Documents", *SEQUENCES 1997*, which introduced MinHash; and Leskovec, Rajaraman, Ullman, *Mining of Massive Datasets* (2nd ed., 2014), chapter 3, which is still the clearest exposition of locality-sensitive hashing. For an engineering-grade write-up targeted at LLM corpora, see [`datasketch`'s own documentation](https://ekzhu.com/datasketch/lsh.html).

### Shingles

Represent each document as the set of its contiguous k-token (or k-character) shingles. k = 5 tokens is a common default for English prose; k = 9 characters is a common default for mixed-script data where tokenisation varies.

```python
def shingles(text: str, k: int = 5) -> set[str]:
    tokens = text.lower().split()
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}
```

The **Jaccard similarity** between two documents is `|A ∩ B| / |A ∪ B|`. Two near-identical documents have Jaccard close to 1; two unrelated documents have Jaccard close to 0. The threshold at which you call them duplicates is a policy choice — 0.7-0.9 is the usual range for pre-training corpora.

Computing Jaccard pairwise over N documents is O(N²), which is impossible at corpus scale. MinHash + LSH gets you an approximate answer in O(N) time with tunable precision/recall.

### MinHash

For each of `num_perm` random hash functions `h_i`, the MinHash signature of a document D is:

```
sig_i(D) = min { h_i(s) : s ∈ shingles(D) }
```

The deep fact is that `Pr[sig_i(A) == sig_i(B)] == Jaccard(A, B)`. So if you compute `num_perm = 128` MinHash signatures per document and compare how many positions match between two signatures, you have an unbiased estimate of their Jaccard. The variance drops as 1/`num_perm`; 128 is a standard choice, 256 for stricter work.

```python
from datasketch import MinHash

def minhash_of(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles(text, k=5):
        m.update(s.encode("utf-8"))
    return m

a = minhash_of("the quick brown fox jumps over the lazy dog")
b = minhash_of("the quick brown fox leaps over the lazy dog")
print(a.jaccard(b))  # ~0.5 — one shingle differs
```

### LSH banding

Even with cheap signatures, pairwise comparison is still O(N²). LSH fixes this by hashing *bands* of the signature into buckets, so that only documents landing in the same bucket for at least one band are compared.

Split the signature of length `L = num_perm` into `b` bands of `r` rows each (`b * r = L`). Hash each band to a bucket. Two documents collide in a bucket iff all `r` rows in some band are equal. The probability of their colliding in *at least one* band given Jaccard `t` is:

```
P(collision | Jaccard = t) = 1 - (1 - t^r)^b
```

That curve is an S-curve, and `(b, r)` tunes where the step is. The step centre is approximately `(1/b)^(1/r)`. A couple of concrete choices:

| `num_perm` | `b` | `r` | Step centre | Behaviour                                   |
|------------|-----|-----|-------------|---------------------------------------------|
| 128        | 32  | 4   | ~0.56       | Permissive; catches Jaccard ≥ ~0.5          |
| 128        | 16  | 8   | ~0.74       | Standard; targets ≥ ~0.7                    |
| 128        | 8   | 16  | ~0.84       | Strict; only very near duplicates           |

Plot the curves yourself to pick:

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 101)
for b, r in [(32, 4), (16, 8), (8, 16)]:
    p = 1 - (1 - t**r)**b
    plt.plot(t, p, label=f"b={b}, r={r}")
plt.xlabel("Jaccard"); plt.ylabel("P(collision)"); plt.legend(); plt.show()
```

The intuition: increasing `r` makes the S-curve steeper (fewer false positives, more false negatives); increasing `b` shifts the step left (more candidates everywhere). You tune these to match your target threshold `t` — `datasketch.MinHashLSH(threshold=t, num_perm=L)` picks `(b, r)` for you by optimising a weighted false-positive/false-negative objective.

### A worked example with `datasketch`

```python
from datasketch import MinHash, MinHashLSH

docs = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "the quick brown fox leaps over the lazy dog",
    "d3": "a completely different sentence about numerical analysis",
    "d4": "the quick brown fox jumps over the lazy dog yesterday",
}

def mh(text):
    m = MinHash(num_perm=128)
    for s in shingles(text, k=5):
        m.update(s.encode())
    return m

lsh = MinHashLSH(threshold=0.7, num_perm=128)
sigs = {k: mh(v) for k, v in docs.items()}
for k, m in sigs.items():
    lsh.insert(k, m)

print(lsh.query(sigs["d1"]))   # ['d1', 'd4']  — d2 drops out at 0.7
```

Three things to notice when you run this on a real corpus:

- The LSH index is memory-resident; `datasketch` has `MinHashLSH` backed by Redis or Cassandra if you need to shard it, and the Lee et al. tool has a Rust equivalent.
- You insert and query simultaneously: the first document in a bucket seeds the bucket, and every subsequent duplicate hits it. Processing order matters for *which* document becomes the representative (see the graph section below).
- The signatures are a few KB per document, so a billion-document corpus is a few TB of MinHash state. Shard the LSH by language and build each in parallel; cross-language near-duplicates are rare enough to ignore in practice.

## From pairwise hits to a dedup decision: the graph

LSH gives you candidate near-duplicate pairs. In a web corpus, those pairs form a graph with:

- Many small components (a document mirrored on five domains).
- A handful of giant components (a page template re-rendered across millions of pages).

You cannot just drop one side of each pair, because near-duplication is transitive-ish: if A ~ B and B ~ C but A and C are below threshold, dropping B first means A and C both survive, and dropping A first leaves B surviving to overlap C. The clean approach:

1. Verify each candidate pair with its exact Jaccard (over the actual shingle sets, not the MinHash estimate). Drop pairs below the real threshold — MinHash has false positives.
2. Treat the surviving pairs as edges in an undirected graph.
3. Compute **connected components** via union-find (`scipy.sparse.csgraph.connected_components` for small graphs; GraphFrames or a sharded union-find for corpus-scale).
4. For each component, pick exactly one canonical representative; drop the rest.

Policies for the canonical pick, in order of what people actually use:

- **Keep oldest.** First-seen timestamp wins. Works for news-style corpora where the original matters.
- **Keep highest-quality.** Rank by document-quality score (perplexity under a reference LM, length, non-boilerplate fraction — chapter [04](04-deboilerplating-and-document-quality.md)) and keep the top one.
- **Keep longest.** Longest document in the component wins; a reasonable proxy for "most informative".
- **Keep random.** Deterministic hash-of-doc-id tie-break; what you fall back to when you have no quality signal.

The biggest trap: a mega-component with 10M documents, all variants of the same template, where any pick is wrong because the whole component is boilerplate. Giant components (say, >10k nodes) deserve a human inspection pass or an automatic "drop the whole component" rule, not a canonical pick. Flag and sample them — this is almost always the signal that your Tier 2 or boilerplate filter needs to run before near-dup, not after.

## Boilerplate is not dedup's job

Aggressive near-dup will happily remove legitimate shared text that is not a duplicate in any meaningful sense:

- License notices (MIT, Apache, Creative Commons) appearing on every page of every doc site.
- "Jump to navigation" / "Jump to search" strings on every Wikipedia-like page.
- Cookie banners, GDPR notices, newsletter sign-up CTAs.

These collapse to a single surviving instance under near-dup, which deletes a lot of pages' worth of real content wrapped around the shared boilerplate. If you see your corpus losing documents in a Zipf-shaped pattern after near-dup, chances are your pipeline is treating boilerplate as duplicate content. The fix is a boilerplate filter at paragraph or block level before near-dup — see chapter [04](04-deboilerplating-and-document-quality.md).

## A direct comparison

| Tier                  | Catches                                     | Cost                                                 | False positives                       | False negatives                                     |
|-----------------------|---------------------------------------------|------------------------------------------------------|---------------------------------------|-----------------------------------------------------|
| Exact (hash)          | byte-identical after normalisation          | O(N), one hash per doc/paragraph; parallelises hard  | essentially zero                      | anything with a diff of any kind                    |
| Substring (suffix arr.)| long shared spans across documents          | O(n log n) index on total corpus length n; one big box or careful shard | low; a shared quote survives | reordered or paraphrased content                    |
| MinHash + LSH         | high-Jaccard documents (near-paraphrase)    | O(N) insert, O(N × candidates/doc) verify            | 1-5% at typical `(b, r)`; verify to drop | semantic duplicates with no shingle overlap        |

Read "false positives" as "pairs flagged as duplicates that are not"; read "false negatives" as "actual duplicates the tier misses". The three tiers compose because their false-negative sets are nearly disjoint.

## Evaluation and eval-set contamination

Dedup is not just a pre-training quality lever; it is the mechanism by which you prevent your evaluation set from leaking into training. Standard practice:

- Compute MinHash signatures of every eval example (e.g., every HumanEval problem, every MMLU question, every held-out LAMBADA passage).
- Query the training-corpus LSH with each eval signature. Any training document that lands in the same bucket as an eval example is a contamination suspect.
- Report contamination rate (fraction of eval items with any training hit above threshold) alongside your eval numbers.

Lee et al. and the C4 "redpajama" rebuild (Together AI, 2023, [github.com/togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data)) both publish contamination numbers for their public eval sets; use theirs as a reference point when you report yours.

A failure mode worth naming: dedup within your training corpus does not remove eval contamination unless you include the eval set as a *query* against the dedup graph. If HumanEval problem #42 was never in your training set but a near-paraphrase of it was, your trained model is still contaminated — the dedup never saw the eval set as such.

## Chapter summary

- Dedup is a quality, privacy, and eval-integrity lever; Lee et al. and Carlini et al. are the papers that quantified why.
- Run three tiers in order: exact (hash of normalised bytes, paragraph-level), substring (suffix array over corpus or windowed hashes), near-dup (MinHash + LSH).
- Normalise aggressively before hashing (NFC, whitespace collapse, usually casefold). Shard the exact pass by a hash prefix; the work parallelises trivially.
- Suffix-array substring dedup is Lee et al.'s big win over bare hashing; the lightweight substitute is window-hash n-gram overlap.
- MinHash estimates Jaccard; LSH banding `(b, r)` lets you tune the S-curve step centre to your target threshold. Verify candidate pairs with exact Jaccard before trusting them.
- Near-dup produces a graph; compute connected components and pick one canonical representative per component. Watch for mega-components — those are boilerplate, not duplicates.
- Dedup is a corpus-wide pass that runs before language routing (chapter [03](03-language-routing-and-corpus-shards.md)) and alongside quality filtering (chapter [04](04-deboilerplating-and-document-quality.md)). Treat eval sets as queries against the dedup index so contamination is detected, not hidden.
