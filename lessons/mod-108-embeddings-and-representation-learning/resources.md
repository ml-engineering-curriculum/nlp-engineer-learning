# Resources for mod-108 · Embeddings & Representation Learning for Text

Prefer primary sources: papers, standards, and official documentation. Blog posts and library docs are included only where they are the canonical reference for a tool or idea.

## Sentence and document embedding foundations

- **Sentence-BERT:** Nils Reimers, Iryna Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", *EMNLP 2019*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).
- **`sentence-transformers` documentation:** <https://www.sbert.net/> — training overview, pooling, losses, examples.
- **`sentence-transformers` GitHub:** <https://github.com/UKPLab/sentence-transformers>.
- **word2vec (Skip-gram / CBOW):** Tomas Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", *ICLR 2013*. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781); "Distributed Representations of Words and Phrases and their Compositionality", *NeurIPS 2013*. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546).
- **GloVe:** Jeffrey Pennington, Richard Socher, Christopher D. Manning, "GloVe: Global Vectors for Word Representation", *EMNLP 2014*. [aclanthology.org/D14-1162](https://aclanthology.org/D14-1162/).
- **FastText:** Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov, "Enriching Word Vectors with Subword Information", *TACL 2017*. [arXiv:1607.04606](https://arxiv.org/abs/1607.04606).
- **ELMo:** Matthew Peters et al., "Deep Contextualized Word Representations", *NAACL 2018*. [arXiv:1802.05365](https://arxiv.org/abs/1802.05365).
- **BERT:** Jacob Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", *NAACL 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- **MPNet:** Kaitao Song et al., "MPNet: Masked and Permuted Pre-training for Language Understanding", *NeurIPS 2020*. [arXiv:2004.09297](https://arxiv.org/abs/2004.09297).
- **MiniLM:** Wenhui Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers", *NeurIPS 2020*. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957).

## Bi-encoder / dense retrieval architectures

- **DPR (Dense Passage Retrieval):** Vladimir Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", *EMNLP 2020*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906).
- **Contriever (unsupervised dense retrieval):** Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave, "Unsupervised Dense Information Retrieval with Contrastive Learning", *TMLR 2022*. [arXiv:2112.09118](https://arxiv.org/abs/2112.09118).
- **SimCSE:** Tianyu Gao, Xingcheng Yao, Danqi Chen, "SimCSE: Simple Contrastive Learning of Sentence Embeddings", *EMNLP 2021*. [arXiv:2104.08821](https://arxiv.org/abs/2104.08821).
- **E5 (weakly-supervised contrastive pretraining):** Liang Wang et al., "Text Embeddings by Weakly-Supervised Contrastive Pre-training", 2022. [arXiv:2212.03533](https://arxiv.org/abs/2212.03533). Models: [`intfloat/e5-large-v2`](https://huggingface.co/intfloat/e5-large-v2), [`intfloat/e5-mistral-7b-instruct`](https://huggingface.co/intfloat/e5-mistral-7b-instruct).
- **BGE (C-Pack):** Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, "C-Pack: Packaged Resources To Advance General Chinese Embedding", 2023. [arXiv:2309.07597](https://arxiv.org/abs/2309.07597). Models: [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5), [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3).
- **GTE:** Zehan Li et al., "Towards General Text Embeddings with Multi-stage Contrastive Learning", 2023. [arXiv:2308.03281](https://arxiv.org/abs/2308.03281).
- **Nomic Embed:** Zach Nussbaum, John X. Morris, Brandon Duderstadt, Andriy Mulyar, "Nomic Embed: Training a Reproducible Long Context Text Embedder", 2024. [arXiv:2402.01613](https://arxiv.org/abs/2402.01613).
- **Jina Embeddings v2 / v3:** Michael Günther et al., "Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents", 2023. [arXiv:2310.19923](https://arxiv.org/abs/2310.19923). "jina-embeddings-v3", 2024. [arXiv:2409.10173](https://arxiv.org/abs/2409.10173).
- **`text-embedding-3` announcement (OpenAI):** <https://openai.com/index/new-embedding-models-and-api-updates/>.
- **Cohere embed v3:** <https://cohere.com/blog/introducing-embed-v3>.
- **Voyage AI embeddings docs:** <https://docs.voyageai.com/docs/embeddings>.

## Contrastive objectives and training tricks

- **InfoNCE / CPC:** Aaron van den Oord, Yazhe Li, Oriol Vinyals, "Representation Learning with Contrastive Predictive Coding", 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748).
- **`MultipleNegativesRankingLoss` (docs):** <https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss>.
- **Triplet loss (FaceNet):** Florian Schroff, Dmitry Kalenichenko, James Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering", *CVPR 2015*. [arXiv:1503.03832](https://arxiv.org/abs/1503.03832).
- **CoSENT (blog + `sentence-transformers` implementation):** Jianlin Su, "CoSENT (Chinese)", 2022. <https://kexue.fm/archives/8847>. `sentence-transformers.losses.CoSENTLoss`.
- **GradCache (scaling contrastive batches):** Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan, "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup", *RepL4NLP 2021*. [arXiv:2101.06983](https://arxiv.org/abs/2101.06983).
- **MegaBatchMarginLoss / large-batch retrieval training:** see `sentence-transformers` losses docs, `MegaBatchMarginLoss`.
- **TSDAE (unsupervised sentence embedding pretraining):** Kexin Wang, Nils Reimers, Iryna Gurevych, "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning", *EMNLP 2021 Findings*. [arXiv:2104.06979](https://arxiv.org/abs/2104.06979).
- **Contrastive tension:** Fredrik Carlsson et al., "Semantic Re-tuning with Contrastive Tension", *ICLR 2021*. [openreview](https://openreview.net/forum?id=Ov_sMNau-PF).
- **Alignment / uniformity view of contrastive learning:** Tongzhou Wang, Phillip Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere", *ICML 2020*. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242).

## Hard-negative mining, curriculum, and distillation

- **ANCE (approximate nearest-neighbour negatives):** Lee Xiong et al., "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval", *ICLR 2021*. [arXiv:2007.00808](https://arxiv.org/abs/2007.00808).
- **RocketQA:** Yingqi Qu et al., "RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering", *NAACL 2021*. [arXiv:2010.08191](https://arxiv.org/abs/2010.08191).
- **RocketQAv2 (joint training):** Ruiyang Ren et al., "RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking", *EMNLP 2021*. [arXiv:2110.07367](https://arxiv.org/abs/2110.07367).
- **GPL (Generative Pseudo Labeling):** Kexin Wang, Nandan Thakur, Nils Reimers, Iryna Gurevych, "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval", *NAACL 2022*. [arXiv:2112.07577](https://arxiv.org/abs/2112.07577).
- **MarginMSE distillation for retrieval:** Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, Allan Hanbury, "Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation", 2020. [arXiv:2010.02666](https://arxiv.org/abs/2010.02666).
- **SBERT `mine_hard_negatives` utility:** <https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.mine_hard_negatives>.
- **doc2query (query generation):** Rodrigo Nogueira, Wei Yang, Jimmy Lin, Kyunghyun Cho, "Document Expansion by Query Prediction", 2019. [arXiv:1904.08375](https://arxiv.org/abs/1904.08375). Model: [`doc2query/msmarco-t5-base-v1`](https://huggingface.co/doc2query/msmarco-t5-base-v1).
- **PROMPTAGATOR (few-shot dense retrieval by prompted synth):** Zhuyun Dai et al., "Promptagator: Few-shot Dense Retrieval From 8 Examples", *ICLR 2023*. [arXiv:2209.11755](https://arxiv.org/abs/2209.11755).
- **INPARS-v2:** Vitor Jeronymo et al., "InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval", 2023. [arXiv:2301.01820](https://arxiv.org/abs/2301.01820).

## Cross-encoders, rerankers, and late-interaction models

- **monoBERT (the original cross-encoder reranker):** Rodrigo Nogueira, Kyunghyun Cho, "Passage Re-ranking with BERT", 2019. [arXiv:1901.04085](https://arxiv.org/abs/1901.04085).
- **monoT5:** Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, Jimmy Lin, "Document Ranking with a Pretrained Sequence-to-Sequence Model", *EMNLP 2020 Findings*. [arXiv:2003.06713](https://arxiv.org/abs/2003.06713). Model: [`castorini/monoT5-base-msmarco`](https://huggingface.co/castorini/monoT5-base-msmarco).
- **ColBERT:** Omar Khattab, Matei Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", *SIGIR 2020*. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832).
- **ColBERTv2 (PLAID compression):** Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia, "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction", *NAACL 2022*. [arXiv:2112.01488](https://arxiv.org/abs/2112.01488). Repo: <https://github.com/stanford-futuredata/ColBERT>. Model: [`colbert-ir/colbertv2.0`](https://huggingface.co/colbert-ir/colbertv2.0).
- **SPLADE (learned sparse retrieval):** Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant, "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking", *SIGIR 2021*. [arXiv:2107.05720](https://arxiv.org/abs/2107.05720). SPLADEv2: Thibault Formal et al., "From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective", *SIGIR 2022*. [arXiv:2205.04733](https://arxiv.org/abs/2205.04733).
- **bge-reranker family (model cards):** [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base), [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large), [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3).
- **Mixedbread rerankers:** [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1).
- **Cohere Rerank:** <https://cohere.com/rerank>.
- **RAGatouille (ColBERT ergonomics):** <https://github.com/AnswerDotAI/RAGatouille>.
- **Answer.AI `rerankers`:** <https://github.com/AnswerDotAI/rerankers>.
- **`pylate` (ColBERT training on top of sentence-transformers):** <https://github.com/lightonai/pylate>.

## Multi-stage training and curriculum

- **E5 recipe (paper details staging):** Wang et al., 2022. [arXiv:2212.03533](https://arxiv.org/abs/2212.03533).
- **BGE / C-Pack recipe:** Xiao et al., 2023. [arXiv:2309.07597](https://arxiv.org/abs/2309.07597).
- **GTE multi-stage recipe:** Li et al., 2023. [arXiv:2308.03281](https://arxiv.org/abs/2308.03281).
- **Multilingual E5 (report):** Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei, "Multilingual E5 Text Embeddings: A Technical Report", 2024. [arXiv:2402.05672](https://arxiv.org/abs/2402.05672).
- **BGE M3-Embedding (multi-function multilingual):** Jianlv Chen et al., "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings", 2024. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).
- **Contriever (pure stage-1 recipe):** Izacard et al., 2022. [arXiv:2112.09118](https://arxiv.org/abs/2112.09118).
- **Sentence-transformers "Training Overview" docs:** <https://www.sbert.net/docs/sentence_transformer/training_overview.html>.
- **`MultiDatasetBatchSamplers` (staged / weighted training):** <https://www.sbert.net/docs/package_reference/sentence_transformer/sampler.html>.

## Retrieval benchmarks

- **MS MARCO passage ranking:** Tri Nguyen et al., "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset", *NIPS 2016 workshop*. [arXiv:1611.09268](https://arxiv.org/abs/1611.09268). Dataset: <https://microsoft.github.io/msmarco/>.
- **Natural Questions:** Tom Kwiatkowski et al., "Natural Questions: a Benchmark for Question Answering Research", *TACL 2019*. [aclanthology.org/Q19-1026](https://aclanthology.org/Q19-1026/).
- **TriviaQA:** Mandar Joshi, Eunsol Choi, Daniel S. Weld, Luke Zettlemoyer, "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension", *ACL 2017*. [arXiv:1705.03551](https://arxiv.org/abs/1705.03551).
- **HotpotQA:** Zhilin Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", *EMNLP 2018*. [arXiv:1809.09600](https://arxiv.org/abs/1809.09600).
- **BEIR (zero-shot retrieval benchmark):** Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", *NeurIPS 2021 Datasets and Benchmarks*. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663).
- **LoTTE:** part of the ColBERTv2 release. [arXiv:2112.01488](https://arxiv.org/abs/2112.01488).
- **NFCorpus:** Vera Boteva, Demian Ghelfi, Dan Wolczyk, Erin Yeo, Michael Chen, "A Full-Text Learning to Rank Dataset for Medical Information Retrieval", *ECIR 2016*. <https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/>.
- **SciFact:** David Wadden et al., "Fact or Fiction: Verifying Scientific Claims", *EMNLP 2020*. [arXiv:2004.14974](https://arxiv.org/abs/2004.14974).
- **TREC-COVID:** Kirk Roberts et al., "TREC-COVID: rationale and structure of an information retrieval shared task for COVID-19", *JAMIA 2020*. <https://ir.nist.gov/trec-covid/>.
- **FiQA-2018 (financial opinion mining and QA):** <https://sites.google.com/view/fiqa/>.
- **`pyserini` (Lucene / Anserini Python bindings for retrieval baselines):** Jimmy Lin et al., "Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations", *SIGIR 2021 demo*. [arXiv:2102.10073](https://arxiv.org/abs/2102.10073). Repo: <https://github.com/castorini/pyserini>.
- **`rank_bm25` (Python BM25):** <https://github.com/dorianbrown/rank_bm25>.

## MTEB and evaluation

- **MTEB (paper):** Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers, "MTEB: Massive Text Embedding Benchmark", *EACL 2023*. [arXiv:2210.07316](https://arxiv.org/abs/2210.07316).
- **`mteb` Python package:** <https://github.com/embeddings-benchmark/mteb>.
- **MTEB leaderboard (live):** <https://huggingface.co/spaces/mteb/leaderboard>.
- **MTEB(medical):** Ling Xiong et al., "Benchmarking Medical Text Embedding Models", 2024. [arXiv:2404.01617](https://arxiv.org/abs/2404.01617).
- **LEXTREME (legal benchmark, background for MTEB(law)):** Ilias Chalkidis et al., "LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain", 2023. [arXiv:2301.13126](https://arxiv.org/abs/2301.13126).
- **CoIR (code retrieval benchmark):** Xiangyang Li et al., "CoIR: A Comprehensive Benchmark for Code Information Retrieval Models", 2024. [arXiv:2407.02883](https://arxiv.org/abs/2407.02883).
- **STS benchmarks family:** Daniel Cer et al., "SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Cross-lingual Focused Evaluation", *SemEval 2017*. [aclanthology.org/S17-2001](https://aclanthology.org/S17-2001/); STSBenchmark dataset card: <https://huggingface.co/datasets/mteb/stsbenchmark-sts>.
- **BIOSSES (biomedical STS):** Gizem Soğancıoğlu, Hakime Öztürk, Arzucan Özgür, "BIOSSES: a semantic sentence similarity estimation system for the biomedical domain", *Bioinformatics 2017*. <https://tabilab.cmpe.boun.edu.tr/BIOSSES/>.
- **SciDocs (scientific reranking / classification):** Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel S. Weld, "SPECTER: Document-level Representation Learning using Citation-informed Transformers", *ACL 2020*. [arXiv:2004.07180](https://arxiv.org/abs/2004.07180).
- **AskUbuntu duplicate question dataset:** <https://github.com/taolei87/askubuntu>.
- **Banking77:** Iñigo Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", *ACL 2020 NLP4ConvAI*. [arXiv:2003.04807](https://arxiv.org/abs/2003.04807).
- **ArXiv clustering task (MTEB):** part of MTEB, [arXiv:2210.07316](https://arxiv.org/abs/2210.07316).
- **Trec-DL 2019 / 2020:** Nick Craswell et al., "Overview of the TREC 2019 deep learning track", 2020. [arXiv:2003.07820](https://arxiv.org/abs/2003.07820). "Overview of the TREC 2020 deep learning track", 2021. [arXiv:2102.07662](https://arxiv.org/abs/2102.07662).
- **`ranx` (IR metrics library):** Elias Bassani, "ranx: A Blazing-Fast Python Library for Ranking Evaluation and Comparison", *ECIR 2022 demo*. Repo: <https://github.com/AmenRa/ranx>.

## Domain adaptation

- **PubMedBERT:** Yu Gu et al., "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing", *ACM Transactions on Computing for Healthcare 2021*. [arXiv:2007.15779](https://arxiv.org/abs/2007.15779). Model: [`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).
- **BioBERT:** Jinhyuk Lee et al., "BioBERT: a pre-trained biomedical language representation model for biomedical text mining", *Bioinformatics 2020*. [arXiv:1901.08746](https://arxiv.org/abs/1901.08746). Model: [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1).
- **ClinicalBERT:** Emily Alsentzer et al., "Publicly Available Clinical BERT Embeddings", *ClinicalNLP 2019*. [arXiv:1904.03323](https://arxiv.org/abs/1904.03323). Model: [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).
- **SciBERT:** Iz Beltagy, Kyle Lo, Arman Cohan, "SciBERT: A Pretrained Language Model for Scientific Text", *EMNLP 2019*. [arXiv:1903.10676](https://arxiv.org/abs/1903.10676). Model: [`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased).
- **Legal-BERT:** Ilias Chalkidis et al., "LEGAL-BERT: The Muppets straight out of Law School", *EMNLP 2020 Findings*. [arXiv:2010.02559](https://arxiv.org/abs/2010.02559). Model: [`nlpaueb/legal-bert-base-uncased`](https://huggingface.co/nlpaueb/legal-bert-base-uncased).
- **FinBERT:** Yi Yang, Mark Christopher Siy Uy, Allen Huang, "FinBERT: A Pretrained Language Model for Financial Communications", 2020. [arXiv:2006.08097](https://arxiv.org/abs/2006.08097).
- **SPECTER (citation-informed scientific embeddings):** Cohan et al., 2020. [arXiv:2004.07180](https://arxiv.org/abs/2004.07180). Model: [`allenai/specter`](https://huggingface.co/allenai/specter).
- **SciRepEval / SPECTER2:** Amanpreet Singh et al., "SciRepEval: A Multi-Format Benchmark for Scientific Document Representations", *EMNLP 2023*. [arXiv:2211.13308](https://arxiv.org/abs/2211.13308). Models: [`allenai/specter2_base`](https://huggingface.co/allenai/specter2_base).
- **SciNCL:** Malte Ostendorff, Nils Rethmeier, Isabelle Augenstein, Bela Gipp, Georg Rehm, "Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings", *EMNLP 2022*. [arXiv:2202.06671](https://arxiv.org/abs/2202.06671).
- **`pritamdeka/S-PubMedBert-MS-MARCO` (biomedical bi-encoder):** <https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO>.
- **`pile-of-law`:** Peter Henderson et al., "Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset", *NeurIPS 2022 D&B*. [arXiv:2207.00220](https://arxiv.org/abs/2207.00220).
- **LexGLUE:** Ilias Chalkidis et al., "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English", *ACL 2022*. [arXiv:2110.00976](https://arxiv.org/abs/2110.00976).
- **MedRAG toolkit / benchmarks:** Guangzhi Xiong, Qiao Jin, Zhiyong Lu, Aidong Zhang, "Benchmarking Retrieval-Augmented Generation for Medicine", 2024. [arXiv:2402.13178](https://arxiv.org/abs/2402.13178).

## Multilingual and cross-lingual embeddings

- **LaBSE:** Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, Wei Wang, "Language-agnostic BERT Sentence Embedding", *ACL 2022*. [arXiv:2007.01852](https://arxiv.org/abs/2007.01852). Model: [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE).
- **LASER / LASER-3:** Mikel Artetxe, Holger Schwenk, "Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond", *TACL 2019*. [arXiv:1812.10464](https://arxiv.org/abs/1812.10464). NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation", 2022. [arXiv:2207.04672](https://arxiv.org/abs/2207.04672). Repo: <https://github.com/facebookresearch/LASER>.
- **Cross-lingual knowledge distillation for sentence embeddings:** Nils Reimers, Iryna Gurevych, "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation", *EMNLP 2020*. [arXiv:2004.09813](https://arxiv.org/abs/2004.09813).
- **`paraphrase-multilingual-MiniLM-L12-v2` model card:** <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>.
- **XLM-R (backbone for multilingual encoders):** Alexis Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", *ACL 2020*. [arXiv:1911.02116](https://arxiv.org/abs/1911.02116).
- **mMARCO (multilingual MS MARCO):** Luiz Bonifacio et al., "mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset", 2021. [arXiv:2108.13897](https://arxiv.org/abs/2108.13897).
- **MIRACL:** Xinyu Zhang et al., "MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages", 2022. [arXiv:2210.09984](https://arxiv.org/abs/2210.09984).
- **XOR-TyDi QA:** Akari Asai, Jungo Kasai, Jonathan H. Clark, Kenton Lee, Eunsol Choi, Hannaneh Hajishirzi, "XOR QA: Cross-lingual Open-Retrieval Question Answering", *NAACL 2021*. [aclanthology.org/2021.naacl-main.46](https://aclanthology.org/2021.naacl-main.46/). <!-- needs-research: confirm arXiv ID (candidate 2010.11856 collides with XQuAD) -->

- **MLDR (multilingual long-document retrieval; part of BGE-M3 release):** Chen et al., 2024. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).
- **Tatoeba (bitext mining benchmark):** <https://tatoeba.org/>. Test set used in MTEB derived from LASER benchmarks.
- **BUCC bitext-mining task:** Pierre Zweigenbaum, Serge Sharoff, Reinhard Rapp, "Overview of the Third BUCC Shared Task: Spotting Parallel Sentences in Comparable Corpora", *BUCC 2018*. <https://comparable.limsi.fr/bucc2018/>.

## Serving, quantisation, and dimensionality

- **Matryoshka Representation Learning:** Aditya Kusupati et al., "Matryoshka Representation Learning", *NeurIPS 2022*. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147).
- **Matryoshka in `sentence-transformers`:** <https://www.sbert.net/examples/training/matryoshka/README.html>.
- **Binary and int8 embedding quantisation (blog + reference impl):** Aamir Shakir, Tom Aarsen et al., "Embedding Quantization" (Hugging Face + Mixedbread), 2024. <https://huggingface.co/blog/embedding-quantization>.
- **`sentence-transformers` `quantize_embeddings` / `semantic_search_faiss` / `semantic_search_hnsw`:** <https://www.sbert.net/docs/package_reference/util.html>.
- **Text Embeddings Inference (TEI):** <https://github.com/huggingface/text-embeddings-inference>. Docker image and dynamic batching for encoder inference.
- **ONNX Runtime for transformers:** <https://onnxruntime.ai/docs/tutorials/huggingface.html>.
- **NVIDIA Triton Inference Server:** <https://github.com/triton-inference-server/server>.
- **BetterTransformer (kernel-level speedups for encoders):** <https://huggingface.co/docs/optimum/bettertransformer/overview>.
- **FAISS:** Jeff Johnson, Matthijs Douze, Hervé Jégou, "Billion-scale similarity search with GPUs", *IEEE Transactions on Big Data 2019*. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734). Repo: <https://github.com/facebookresearch/faiss>.
- **HNSW:** Yu. A. Malkov, D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", *IEEE TPAMI 2018*. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320).
- **ScaNN (Google):** Ruiqi Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization", *ICML 2020*. [arXiv:1908.10396](https://arxiv.org/abs/1908.10396). Repo: <https://github.com/google-research/google-research/tree/master/scann>.
- **PLAID (ColBERTv2 index compression):** Keshav Santhanam et al., "PLAID: An Efficient Engine for Late Interaction Retrieval", *CIKM 2022*. [arXiv:2205.09707](https://arxiv.org/abs/2205.09707).
- **`vLLM` (used for generation; relevant for embedding + LLM co-serving):** <https://github.com/vllm-project/vllm>.

## The embedding / RAG-engineer boundary

- **RAG (original):** Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", *NeurIPS 2020*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- **Atlas (retrieval-augmented LM training):** Gautier Izacard et al., "Atlas: Few-shot Learning with Retrieval Augmented Language Models", *JMLR 2023*. [arXiv:2208.03299](https://arxiv.org/abs/2208.03299).
- **Chunking strategies (deep dive, non-primary):** LlamaIndex "Chunking Strategies" documentation. <https://docs.llamaindex.ai/en/stable/optimizing/production_rag/>. (Blog reference; treat as an entry point, not primary.)
- **Hybrid dense-sparse retrieval evidence:** Sebastian Bruch, Siyu Gai, Amir Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval", *ACM TOIS 2023*. [arXiv:2210.11934](https://arxiv.org/abs/2210.11934).

## Libraries and reference implementations

- **`sentence-transformers`** — <https://github.com/UKPLab/sentence-transformers>
- **Hugging Face `transformers`** — <https://huggingface.co/docs/transformers/index>
- **Hugging Face `datasets`** — <https://huggingface.co/docs/datasets/index>
- **Hugging Face `evaluate`** — <https://huggingface.co/docs/evaluate/index>
- **`mteb`** — <https://github.com/embeddings-benchmark/mteb>
- **`beir`** — <https://github.com/beir-cellar/beir>
- **`pyserini`** — <https://github.com/castorini/pyserini>
- **`rank_bm25`** — <https://github.com/dorianbrown/rank_bm25>
- **`faiss`** — <https://github.com/facebookresearch/faiss>
- **`hnswlib`** — <https://github.com/nmslib/hnswlib>
- **`ScaNN`** — <https://github.com/google-research/google-research/tree/master/scann>
- **`peft` (LoRA / adapters)** — <https://github.com/huggingface/peft>
- **`optimum`** — <https://github.com/huggingface/optimum>
- **Text Embeddings Inference (TEI)** — <https://github.com/huggingface/text-embeddings-inference>
- **ColBERT** — <https://github.com/stanford-futuredata/ColBERT>
- **RAGatouille** — <https://github.com/AnswerDotAI/RAGatouille>
- **Answer.AI `rerankers`** — <https://github.com/AnswerDotAI/rerankers>
- **`pylate`** — <https://github.com/lightonai/pylate>
- **`GPL` (reference implementation)** — <https://github.com/UKPLab/gpl>

## Companion tracks

- **`mod-101` (this track)** — subword tokenisation, encoder foundations. The vocabulary-shatter warnings in chapter 09 build on it.
- **`mod-102` (this track)** — classical NLP. Static word embeddings and BM25 baselines that chapter 05 measures against.
- **`mod-103` (this track)** — text classification. The kNN and linear-probe use cases in chapter 08's classification category live there.
- **`mod-105` (this track)** — question answering. Consumers of the dense-retrieval encoders trained here.
- **`mod-107` (this track)** — machine translation and multilingual NLP. The multilingual-encoder shelf (XLM-R, mT5) that chapter 10 builds embeddings on.
- **`mod-111` (this track)** — NLP evaluation methodology. Statistical significance, dataset governance, and human evaluation that ground exercise-03's selection defence.
- **`mod-112` (this track)** — production NLP pipelines. Serving, observability, and rollout for the encoders trained here (chapter 11).
- **`rag-engineer` track** — chunking, ANN indexing, hybrid scoring, and RAG pipeline ownership. Chapter 12 draws the boundary explicitly.
- **`ml-platform-engineer` track** — GPU capacity, batching infra, model registries. Underlies chapter 11's serving-stack choices.
