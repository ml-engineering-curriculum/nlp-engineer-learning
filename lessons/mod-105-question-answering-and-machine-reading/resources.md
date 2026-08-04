# Resources for mod-105 · Question Answering & Machine Reading Comprehension

Prefer primary sources: papers, standards, and official documentation. Blog posts are included only where they are the canonical explanation.

## Extractive QA — datasets and formulation

- **SQuAD 1.1:** Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang, "SQuAD: 100,000+ Questions for Machine Comprehension of Text", *EMNLP 2016*. [arXiv:1606.05250](https://arxiv.org/abs/1606.05250).
- **SQuAD 2.0:** Pranav Rajpurkar, Robin Jia, Percy Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD", *ACL 2018*. [arXiv:1806.03822](https://arxiv.org/abs/1806.03822).
- **Natural Questions:** Tom Kwiatkowski et al., "Natural Questions: A Benchmark for Question Answering Research", *TACL 2019*. [aclanthology.org/Q19-1026](https://aclanthology.org/Q19-1026/).
- **TriviaQA:** Mandar Joshi, Eunsol Choi, Daniel S. Weld, Luke Zettlemoyer, "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension", *ACL 2017*. [arXiv:1705.03551](https://arxiv.org/abs/1705.03551).
- **NewsQA:** Adam Trischler et al., "NewsQA: A Machine Comprehension Dataset", *Rep4NLP 2017*. [arXiv:1611.09830](https://arxiv.org/abs/1611.09830).
- **XQuAD:** Mikel Artetxe, Sebastian Ruder, Dani Yogatama, "On the Cross-lingual Transferability of Monolingual Representations", *ACL 2020*. [arXiv:1910.11856](https://arxiv.org/abs/1910.11856).
- **MLQA:** Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian Riedel, Holger Schwenk, "MLQA: Evaluating Cross-lingual Extractive Question Answering", *ACL 2020*. [arXiv:1910.07475](https://arxiv.org/abs/1910.07475).
- **TyDi QA:** Jonathan H. Clark et al., "TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages", *TACL 2020*. [arXiv:2003.05002](https://arxiv.org/abs/2003.05002).
- **SQuADShifts:** John Miller et al., "The Effect of Natural Distribution Shift on Question Answering Models", *ICML 2020*. [arXiv:2004.14444](https://arxiv.org/abs/2004.14444).
- **AdversarialQA:** Max Bartolo, Alastair Roberts, Johannes Welbl, Sebastian Riedel, Pontus Stenetorp, "Beat the AI: Investigating Adversarial Human Annotation for Reading Comprehension", *TACL 2020*. [arXiv:2002.00293](https://arxiv.org/abs/2002.00293).

## Abstractive and long-form QA

- **MS MARCO:** Tri Nguyen et al., "MS MARCO: A Human-Generated Machine Reading Comprehension Dataset", *NIPS 2016 Workshop on Cognitive Computation*. [arXiv:1611.09268](https://arxiv.org/abs/1611.09268).
- **NarrativeQA:** Tomáš Kočiský et al., "The NarrativeQA Reading Comprehension Challenge", *TACL 2018*. [arXiv:1712.07040](https://arxiv.org/abs/1712.07040).
- **ELI5:** Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, Michael Auli, "ELI5: Long Form Question Answering", *ACL 2019*. [arXiv:1907.09190](https://arxiv.org/abs/1907.09190).
- **Qasper:** Pradeep Dasigi et al., "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers", *NAACL 2021*. [arXiv:2105.03011](https://arxiv.org/abs/2105.03011).
- **SCROLLS:** Uri Shaham et al., "SCROLLS: Standardized CompaRison Over Long Language Sequences", *EMNLP 2022*. [arXiv:2201.03533](https://arxiv.org/abs/2201.03533).

## Closed-book QA and prompting

- **Closed-book QA framing:** Adam Roberts, Colin Raffel, Noam Shazeer, "How Much Knowledge Can You Pack Into the Parameters of a Language Model?", *EMNLP 2020*. [arXiv:2002.08910](https://arxiv.org/abs/2002.08910).
- **Chain-of-thought:** Jason Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", *NeurIPS 2022*. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903).
- **Self-consistency:** Xuezhi Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", *ICLR 2023*. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).
- **Least-to-most prompting:** Denny Zhou et al., "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models", *ICLR 2023*. [arXiv:2205.10625](https://arxiv.org/abs/2205.10625).
- **ReAct:** Shunyu Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", *ICLR 2023*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
- **MMLU:** Dan Hendrycks et al., "Measuring Massive Multitask Language Understanding", *ICLR 2021*. [arXiv:2009.03300](https://arxiv.org/abs/2009.03300).
- **TruthfulQA:** Stephanie Lin, Jacob Hilton, Owain Evans, "TruthfulQA: Measuring How Models Mimic Human Falsehoods", *ACL 2022*. [arXiv:2109.07958](https://arxiv.org/abs/2109.07958).
- **Kadavath et al., "Language Models (Mostly) Know What They Know"**, 2022. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221).

## Long-context readers

- **Longformer:** Iz Beltagy, Matthew E. Peters, Arman Cohan, "Longformer: The Long-Document Transformer", 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150).
- **BigBird:** Manzil Zaheer et al., "Big Bird: Transformers for Longer Sequences", *NeurIPS 2020*. [arXiv:2007.14062](https://arxiv.org/abs/2007.14062).
- **LongT5:** Mandy Guo et al., "LongT5: Efficient Text-to-Text Transformer for Long Sequences", *NAACL 2022 Findings*. [arXiv:2112.07916](https://arxiv.org/abs/2112.07916).
- **Fusion-in-Decoder (FiD):** Gautier Izacard, Édouard Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", *EACL 2021*. [arXiv:2007.01282](https://arxiv.org/abs/2007.01282).
- **Lost in the Middle:** Nelson F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", *TACL 2024*. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).
- **Needle-in-a-Haystack:** Greg Kamradt, "Needle In A Haystack — Pressure Testing LLMs", 2023. [github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack).

## Multi-hop QA

- **HotpotQA:** Zhilin Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", *EMNLP 2018*. [arXiv:1809.09600](https://arxiv.org/abs/1809.09600).
- **2WikiMultiHopQA:** Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, Akiko Aizawa, "Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps", *COLING 2020*. [arXiv:2011.01060](https://arxiv.org/abs/2011.01060).
- **MuSiQue:** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal, "MuSiQue: Multihop Questions via Single-hop Question Composition", *TACL 2022*. [arXiv:2108.00573](https://arxiv.org/abs/2108.00573).
- **StrategyQA:** Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, Jonathan Berant, "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies", *TACL 2021*. [arXiv:2101.02235](https://arxiv.org/abs/2101.02235).
- **IRCoT:** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal, "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions", *ACL 2023*. [arXiv:2212.10509](https://arxiv.org/abs/2212.10509).
- **Self-Ask:** Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, Mike Lewis, "Measuring and Narrowing the Compositionality Gap in Language Models", *EMNLP 2023 Findings*. [arXiv:2210.03350](https://arxiv.org/abs/2210.03350).
- **Multi-hop shortcut analysis:** Jifan Chen, Greg Durrett, "Understanding Dataset Design Choices for Multi-hop Reasoning", *NAACL 2019*. [arXiv:1904.12106](https://arxiv.org/abs/1904.12106).
- **Avoiding reasoning shortcuts:** Yichen Jiang, Mohit Bansal, "Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model Development for Multi-Hop QA", *ACL 2019*. [arXiv:1906.07132](https://arxiv.org/abs/1906.07132).

## Retrieval-augmented generation (link out to `rag-engineer` track)

- **RAG:** Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", *NeurIPS 2020*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- **REALM:** Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang, "REALM: Retrieval-Augmented Language Model Pre-Training", *ICML 2020*. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909).
- **DPR (Dense Passage Retrieval):** Vladimir Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", *EMNLP 2020*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906).
- **HyDE:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan, "Precise Zero-Shot Dense Retrieval without Relevance Labels", *ACL 2023*. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496).
- **RAGAS:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation", *EACL 2024 Demo*. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217).

## Evaluation metrics

- **ROUGE:** Chin-Yew Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", *ACL 2004 Workshop*. [aclanthology.org/W04-1013](https://aclanthology.org/W04-1013/).
- **BLEU:** Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu, "BLEU: A Method for Automatic Evaluation of Machine Translation", *ACL 2002*. [aclanthology.org/P02-1040](https://aclanthology.org/P02-1040/).
- **METEOR:** Satanjeev Banerjee, Alon Lavie, "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments", *ACL 2005 Workshop*. [aclanthology.org/W05-0909](https://aclanthology.org/W05-0909/).
- **BERTScore:** Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi, "BERTScore: Evaluating Text Generation with BERT", *ICLR 2020*. [arXiv:1904.09675](https://arxiv.org/abs/1904.09675).
- **BLEURT:** Thibault Sellam, Dipanjan Das, Ankur P. Parikh, "BLEURT: Learning Robust Metrics for Text Generation", *ACL 2020*. [arXiv:2004.04696](https://arxiv.org/abs/2004.04696).
- **Answer equivalence classifier:** Jannis Bulian, Christian Buck, Wojciech Gajewski, Benjamin Boerschinger, Tal Schuster, "Tomayto, Tomahto. Beyond Token-level Answer Equivalence for Question Answering Evaluation", *EMNLP 2022*. [arXiv:2202.07654](https://arxiv.org/abs/2202.07654).
- **Faithfulness / factual consistency:** Or Honovich et al., "TRUE: Re-evaluating Factual Consistency Evaluation", *NAACL 2022*. [arXiv:2204.04991](https://arxiv.org/abs/2204.04991).

## Calibration, abstention, and selective prediction

- **Temperature scaling:** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, "On Calibration of Modern Neural Networks", *ICML 2017*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- **Selective classification:** Yonatan Geifman, Ran El-Yaniv, "Selective Classification for Deep Neural Networks", *NeurIPS 2017*. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500).
- **Selective QA and abstention:** Amita Kamath, Robin Jia, Percy Liang, "Selective Question Answering under Domain Shift", *ACL 2020*. [arXiv:2006.09462](https://arxiv.org/abs/2006.09462).

## Foundational transformer architectures used in this module

- **BERT:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", *NAACL 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- **RoBERTa:** Yinhan Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692).
- **DeBERTa / DeBERTa-v3:** Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", *ICLR 2021*. [arXiv:2006.03654](https://arxiv.org/abs/2006.03654); Pengcheng He, Jianfeng Gao, Weizhu Chen, "DeBERTaV3", 2021. [arXiv:2111.09543](https://arxiv.org/abs/2111.09543).
- **T5:** Colin Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", *JMLR 2020*. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683).
- **BART:** Mike Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", *ACL 2020*. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461).
- **FLAN-T5:** Hyung Won Chung et al., "Scaling Instruction-Finetuned Language Models", 2022. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416).
- **XLM-R:** Alexis Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", *ACL 2020*. [arXiv:1911.02116](https://arxiv.org/abs/1911.02116).

## Human evaluation and LLM-as-judge

- **LLM-as-judge:** Lianmin Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", *NeurIPS 2023*. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
- **LLM judges are not fair evaluators:** Peiyi Wang et al., "Large Language Models are not Fair Evaluators", *ACL 2024 Findings*. [arXiv:2305.17926](https://arxiv.org/abs/2305.17926).

## Libraries and reference implementations

- **Hugging Face `transformers`** — <https://huggingface.co/docs/transformers/index>
- **Hugging Face `run_qa.py` example** — <https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering>
- **Hugging Face `evaluate`** — <https://huggingface.co/docs/evaluate/index>
- **Hugging Face `datasets`** — <https://huggingface.co/docs/datasets/index>
- **`bert_score`** — <https://github.com/Tiiiger/bert_score>
- **`bleurt`** — <https://github.com/google-research/bleurt>
- **`rouge_score`** — <https://github.com/google-research/google-research/tree/master/rouge>
- **RAGAS** — <https://github.com/explodinggradients/ragas>
- **`Outlines` (constrained decoding)** — <https://github.com/dottxt-ai/outlines>
- **`Guidance` (constrained decoding)** — <https://github.com/guidance-ai/guidance>

## Companion tracks

- **`rag-engineer` track** — retrieval, chunking, reranking, hybrid search. Owns everything upstream of the reader's passage input.
- **`mod-104` (this track)** — sequence labelling for extraction tasks that are QA-adjacent (NER, relation extraction).
- **`mod-111` (this track)** — deeper NLP evaluation methodology, including inter-annotator agreement, statistical significance, and dataset governance.
