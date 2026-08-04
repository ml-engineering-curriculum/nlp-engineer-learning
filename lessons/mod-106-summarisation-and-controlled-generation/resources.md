# Resources for mod-106 · Summarisation & Controlled Generation

Prefer primary sources: papers, standards, and official documentation. Blog posts and library docs are included only where they are the canonical reference for a tool or an idea.

## Encoder-decoder architectures used in this module

- **BART:** Mike Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", *ACL 2020*. [arXiv:1910.13461](https://arxiv.org/abs/1910.13461).
- **T5:** Colin Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", *JMLR 2020*. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683).
- **PEGASUS:** Jingqing Zhang, Yao Zhao, Mohammad Saleh, Peter J. Liu, "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization", *ICML 2020*. [arXiv:1912.08777](https://arxiv.org/abs/1912.08777).
- **mT5:** Linting Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer", *NAACL 2021*. [arXiv:2010.11934](https://arxiv.org/abs/2010.11934).
- **FLAN-T5:** Hyung Won Chung et al., "Scaling Instruction-Finetuned Language Models", 2022. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416).
- **LongT5:** Mandy Guo et al., "LongT5: Efficient Text-to-Text Transformer for Long Sequences", *NAACL 2022 Findings*. [arXiv:2112.07916](https://arxiv.org/abs/2112.07916).
- **Longformer / LED:** Iz Beltagy, Matthew E. Peters, Arman Cohan, "Longformer: The Long-Document Transformer", 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150).
- **BigBird / BigBird-PEGASUS:** Manzil Zaheer et al., "Big Bird: Transformers for Longer Sequences", *NeurIPS 2020*. [arXiv:2007.14062](https://arxiv.org/abs/2007.14062).
- **PEGASUS-X:** Jason Phang, Yao Zhao, Peter J. Liu, "Investigating Efficiently Extending Transformers for Long Input Summarization", *EMNLP 2023*. [arXiv:2208.04347](https://arxiv.org/abs/2208.04347).
- **PRIMERA (multi-doc):** Wen Xiao, Iz Beltagy, Giuseppe Carenini, Arman Cohan, "PRIMERA: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization", *ACL 2022*. [arXiv:2110.08499](https://arxiv.org/abs/2110.08499).
- **BART for CNN/DailyMail (Hugging Face reference):** [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn).
- **PEGASUS-XSum reference:** [`google/pegasus-xsum`](https://huggingface.co/google/pegasus-xsum).

## Summarisation datasets and benchmarks

- **CNN/DailyMail:** Karl Moritz Hermann et al., "Teaching Machines to Read and Comprehend", *NeurIPS 2015*. [arXiv:1506.03340](https://arxiv.org/abs/1506.03340). Summarisation formulation via See, Liu & Manning (2017), [arXiv:1704.04368](https://arxiv.org/abs/1704.04368).
- **XSum:** Shashi Narayan, Shay B. Cohen, Mirella Lapata, "Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization", *EMNLP 2018*. [arXiv:1808.08745](https://arxiv.org/abs/1808.08745).
- **Gigaword (headlines):** David Graff et al., "English Gigaword", *LDC2003T05*. Sentence-summarisation split via Rush, Chopra & Weston (2015), [arXiv:1509.00685](https://arxiv.org/abs/1509.00685).
- **Newsroom:** Max Grusky, Mor Naaman, Yoav Artzi, "Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies", *NAACL 2018*. [arXiv:1804.11283](https://arxiv.org/abs/1804.11283).
- **Multi-News (multi-doc):** Alexander R. Fabbri, Irene Li, Tianwei She, Suyi Li, Dragomir Radev, "Multi-News: A Large-Scale Multi-Document Summarization Dataset", *ACL 2019*. [arXiv:1906.01749](https://arxiv.org/abs/1906.01749).
- **arXiv / PubMed (long-doc):** Arman Cohan et al., "A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents", *NAACL 2018*. [arXiv:1804.05685](https://arxiv.org/abs/1804.05685).
- **BillSum:** Anastassia Kornilova, Vlad Eidelman, "BillSum: A Corpus for Automatic Summarization of US Legislation", *EMNLP 2019 Workshop*. [arXiv:1910.00523](https://arxiv.org/abs/1910.00523).
- **BigPatent:** Eva Sharma, Chen Li, Lu Wang, "BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization", *ACL 2019*. [arXiv:1906.03741](https://arxiv.org/abs/1906.03741).
- **SAMSum (dialogue):** Bogdan Gliwa et al., "SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization", *EMNLP 2019 Workshop*. [arXiv:1911.12237](https://arxiv.org/abs/1911.12237).
- **DialogSum:** Yulong Chen et al., "DialogSum: A Real-Life Scenario Dialogue Summarization Dataset", *ACL 2021 Findings*. [arXiv:2105.06762](https://arxiv.org/abs/2105.06762).
- **QMSum (query-focused meeting):** Ming Zhong et al., "QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization", *NAACL 2021*. [arXiv:2104.05938](https://arxiv.org/abs/2104.05938).
- **GovReport:** Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, Lu Wang, "Efficient Attentions for Long Document Summarization", *NAACL 2021*. [arXiv:2104.02112](https://arxiv.org/abs/2104.02112).
- **WikiSum (multi-doc, long):** Peter J. Liu et al., "Generating Wikipedia by Summarizing Long Sequences", *ICLR 2018*. [arXiv:1801.10198](https://arxiv.org/abs/1801.10198).
- **MLSum (multilingual):** Thomas Scialom et al., "MLSUM: The Multilingual Summarization Corpus", *EMNLP 2020*. [arXiv:2004.14900](https://arxiv.org/abs/2004.14900).
- **XL-Sum (multilingual):** Tahmid Hasan et al., "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages", *ACL 2021 Findings*. [arXiv:2106.13822](https://arxiv.org/abs/2106.13822).
- **SCROLLS (long-context suite):** Uri Shaham et al., "SCROLLS: Standardized CompaRison Over Long Language Sequences", *EMNLP 2022*. [arXiv:2201.03533](https://arxiv.org/abs/2201.03533).

## Extractive summarisation

- **LexRank:** Güneş Erkan, Dragomir R. Radev, "LexRank: Graph-based Lexical Centrality as Salience in Text Summarization", *JAIR 2004*. [arXiv:1109.2128](https://arxiv.org/abs/1109.2128).
- **TextRank:** Rada Mihalcea, Paul Tarau, "TextRank: Bringing Order into Texts", *EMNLP 2004*. [aclanthology.org/W04-3252](https://aclanthology.org/W04-3252/).
- **SumBasic:** Ani Nenkova, Lucy Vanderwende, "The Impact of Frequency on Summarization", *Microsoft Research Tech Report* MSR-TR-2005-101, 2005.
- **BERTSumExt / BERTSumAbs:** Yang Liu, Mirella Lapata, "Text Summarization with Pretrained Encoders", *EMNLP 2019*. [arXiv:1908.08345](https://arxiv.org/abs/1908.08345).
- **MatchSum:** Ming Zhong et al., "Extractive Summarization as Text Matching", *ACL 2020*. [arXiv:2004.08795](https://arxiv.org/abs/2004.08795).

## Decoding strategies

- **Beam search analysis:** Ashwin K. Vijayakumar et al., "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models", 2016. [arXiv:1610.02424](https://arxiv.org/abs/1610.02424).
- **Nucleus / top-p sampling:** Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi, "The Curious Case of Neural Text Degeneration", *ICLR 2020*. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751).
- **Typical sampling (locally typical decoding):** Clara Meister, Tiago Pimentel, Gian Wiher, Ryan Cotterell, "Locally Typical Sampling", *TACL 2023*. [arXiv:2202.00666](https://arxiv.org/abs/2202.00666).
- **Contrastive search / SimCTG:** Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, Nigel Collier, "A Contrastive Framework for Neural Text Generation", *NeurIPS 2022*. [arXiv:2202.06417](https://arxiv.org/abs/2202.06417).
- **Contrastive decoding:** Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, Mike Lewis, "Contrastive Decoding: Open-ended Text Generation as Optimization", *ACL 2023*. [arXiv:2210.15097](https://arxiv.org/abs/2210.15097).
- **Beam-search curse for MT/summarisation:** Felix Stahlberg, Bill Byrne, "On NMT Search Errors and Model Errors: Cat Got Your Tongue?", *EMNLP 2019*. [arXiv:1908.10090](https://arxiv.org/abs/1908.10090).
- **Minimum Bayes Risk (MBR) decoding:** Bryan Eikema, Wilker Aziz, "Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation", *COLING 2020*. [arXiv:2005.10283](https://arxiv.org/abs/2005.10283).
- **Speculative decoding:** Yaniv Leviathan, Matan Kalman, Yossi Matias, "Fast Inference from Transformers via Speculative Decoding", *ICML 2023*. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).
- **Hugging Face `GenerationConfig`:** <https://huggingface.co/docs/transformers/main_classes/text_generation> and the "How to generate" guide: <https://huggingface.co/blog/how-to-generate>.

## Constrained and structured generation

- **Grammar-based constrained decoding:** Kanishka Misra et al., "Comparing Rule-Based and Grammar-Based Constrained Decoding for Structured Prediction with Language Models" — surveyed in the `outlines` and `guidance` docs.
- **Outlines (regex / JSON-schema / CFG):** Rémi Louf, Will Kuhn, Nicola Buonocore et al., ["Outlines: Guided Generation with Large Language Models"](https://github.com/dottxt-ai/outlines). See also Willard & Louf, "Efficient Guided Generation for Large Language Models", 2023. [arXiv:2307.09702](https://arxiv.org/abs/2307.09702).
- **Guidance:** Microsoft, ["Guidance"](https://github.com/guidance-ai/guidance).
- **`lm-format-enforcer`:** Noam Gat, ["lm-format-enforcer"](https://github.com/noamgat/lm-format-enforcer).
- **JSON-schema / OpenAI structured outputs:** OpenAI, "Introducing Structured Outputs in the API", 2024. <https://openai.com/index/introducing-structured-outputs-in-the-api/>.
- **XGrammar (grammar-guided fast decoding):** Yixin Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models", 2024. [arXiv:2411.15100](https://arxiv.org/abs/2411.15100).
- **Hugging Face constrained decoding docs:** `PrefixConstrainedLogitsProcessor`, `DisjunctiveConstraint`, `PhrasalConstraint` — <https://huggingface.co/docs/transformers/generation_strategies#constrained-beam-search>.
- **JSONSchema Draft 2020-12:** <https://json-schema.org/specification>.

## Copy, coverage, and grounding mechanisms

- **Pointer-Generator + Coverage:** Abigail See, Peter J. Liu, Christopher D. Manning, "Get To The Point: Summarization with Pointer-Generator Networks", *ACL 2017*. [arXiv:1704.04368](https://arxiv.org/abs/1704.04368).
- **CopyNet:** Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li, "Incorporating Copying Mechanism in Sequence-to-Sequence Learning", *ACL 2016*. [arXiv:1603.06393](https://arxiv.org/abs/1603.06393).
- **Coverage penalty (post-hoc):** Yonghui Wu et al., "Google's Neural Machine Translation System", 2016. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144).

## Long-document strategies

- **Hierarchical / chunk-and-fuse (LangChain map-reduce, refine):** LangChain docs, "Summarization" chain type — <https://python.langchain.com/docs/tutorials/summarization/>. LlamaIndex "response synthesis modes" — <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes/>.
- **Recursive summarisation (books):** Jeff Wu, Long Ouyang, Daniel M. Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, Paul Christiano, "Recursively Summarizing Books with Human Feedback", 2021. [arXiv:2109.10862](https://arxiv.org/abs/2109.10862).
- **Chain-of-density prompting:** Griffin Adams, Alexander R. Fabbri, Faisal Ladhak, Eric Lehman, Noémie Elhadad, "From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting", 2023. [arXiv:2309.04269](https://arxiv.org/abs/2309.04269).
- **Lost in the middle:** Nelson F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", *TACL 2024*. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).

## Reference-based evaluation

- **ROUGE:** Chin-Yew Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", *ACL 2004 Workshop*. [aclanthology.org/W04-1013](https://aclanthology.org/W04-1013/).
- **BLEU:** Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu, "BLEU: A Method for Automatic Evaluation of Machine Translation", *ACL 2002*. [aclanthology.org/P02-1040](https://aclanthology.org/P02-1040/).
- **METEOR:** Satanjeev Banerjee, Alon Lavie, "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments", *ACL 2005 Workshop*. [aclanthology.org/W05-0909](https://aclanthology.org/W05-0909/).
- **BERTScore:** Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi, "BERTScore: Evaluating Text Generation with BERT", *ICLR 2020*. [arXiv:1904.09675](https://arxiv.org/abs/1904.09675).
- **MoverScore:** Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger, "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance", *EMNLP 2019*. [arXiv:1909.02622](https://arxiv.org/abs/1909.02622).
- **BLEURT:** Thibault Sellam, Dipanjan Das, Ankur P. Parikh, "BLEURT: Learning Robust Metrics for Text Generation", *ACL 2020*. [arXiv:2004.04696](https://arxiv.org/abs/2004.04696).
- **BARTScore:** Weizhe Yuan, Graham Neubig, Pengfei Liu, "BARTScore: Evaluating Generated Text as Text Generation", *NeurIPS 2021*. [arXiv:2106.11520](https://arxiv.org/abs/2106.11520).
- **`rouge_score`:** <https://github.com/google-research/google-research/tree/master/rouge>.
- **`bert_score`:** <https://github.com/Tiiiger/bert_score>.
- **Hugging Face `evaluate`:** <https://huggingface.co/docs/evaluate/index>.
- **Reproducibility of ROUGE:** Rodrigo Nogueira, "A Note on the Evaluation of Generative Models", 2019. [arXiv:1902.03511](https://arxiv.org/abs/1902.03511).
- **SummEval:** Alexander R. Fabbri et al., "SummEval: Re-evaluating Summarization Evaluation", *TACL 2021*. [arXiv:2007.12626](https://arxiv.org/abs/2007.12626).

## Faithfulness and hallucination

- **Faithfulness survey (summarisation):** Joshua Maynez, Shashi Narayan, Bernd Bohnet, Ryan McDonald, "On Faithfulness and Factuality in Abstractive Summarization", *ACL 2020*. [arXiv:2005.00661](https://arxiv.org/abs/2005.00661).
- **Hallucination survey:** Ziwei Ji et al., "Survey of Hallucination in Natural Language Generation", *ACM Computing Surveys 2023*. [arXiv:2202.03629](https://arxiv.org/abs/2202.03629).
- **FactCC:** Wojciech Kryściński, Bryan McCann, Caiming Xiong, Richard Socher, "Evaluating the Factual Consistency of Abstractive Text Summarization", *EMNLP 2020*. [arXiv:1910.12840](https://arxiv.org/abs/1910.12840).
- **DAE (Dependency Arc Entailment):** Tanya Goyal, Greg Durrett, "Evaluating Factuality in Generation with Dependency-Level Entailment", *EMNLP 2020 Findings*. [arXiv:2010.05478](https://arxiv.org/abs/2010.05478).
- **SummaC:** Philippe Laban, Tobias Schnabel, Paul N. Bennett, Marti A. Hearst, "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization", *TACL 2022*. [arXiv:2111.09525](https://arxiv.org/abs/2111.09525).
- **QAGS:** Alex Wang, Kyunghyun Cho, Mike Lewis, "Asking and Answering Questions to Evaluate the Factual Consistency of Summaries", *ACL 2020*. [arXiv:2004.04228](https://arxiv.org/abs/2004.04228).
- **QuestEval:** Thomas Scialom et al., "QuestEval: Summarization Asks for Fact-based Evaluation", *EMNLP 2021*. [arXiv:2103.12693](https://arxiv.org/abs/2103.12693).
- **FEQA:** Esin Durmus, He He, Mona Diab, "FEQA: A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization", *ACL 2020*. [arXiv:2005.03754](https://arxiv.org/abs/2005.03754).
- **TRUE meta-benchmark:** Or Honovich et al., "TRUE: Re-evaluating Factual Consistency Evaluation", *NAACL 2022*. [arXiv:2204.04991](https://arxiv.org/abs/2204.04991).
- **FactScore:** Sewon Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation", *EMNLP 2023*. [arXiv:2305.14251](https://arxiv.org/abs/2305.14251).
- **Chain-of-Verification (CoVe):** Shehzaad Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models", *ACL 2024 Findings*. [arXiv:2309.11495](https://arxiv.org/abs/2309.11495).
- **Cliff / CLIFF:** Shuyang Cao, Lu Wang, "CLIFF: Contrastive Learning for Improving Faithfulness and Factuality in Abstractive Summarization", *EMNLP 2021*. [arXiv:2109.09209](https://arxiv.org/abs/2109.09209).
- **FRANK typology of factual errors:** Artidoro Pagnoni, Vidhisha Balachandran, Yulia Tsvetkov, "Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics", *NAACL 2021*. [arXiv:2104.13346](https://arxiv.org/abs/2104.13346).

## Human evaluation

- **Pyramid method:** Ani Nenkova, Rebecca Passonneau, "Evaluating Content Selection in Summarization: The Pyramid Method", *NAACL 2004*. [aclanthology.org/N04-1019](https://aclanthology.org/N04-1019/).
- **Lightweight pyramid (PyrEval):** Yanjun Gao, Chen Sun, Rebecca J. Passonneau, "Automated Pyramid Summarization Evaluation", *CoNLL 2019*. [aclanthology.org/K19-1038](https://aclanthology.org/K19-1038/).
- **Summarisation with human feedback:** Nisan Stiennon et al., "Learning to Summarize from Human Feedback", *NeurIPS 2020*. [arXiv:2009.01325](https://arxiv.org/abs/2009.01325).
- **DUC/TAC guidelines:** NIST Text Analysis Conference summarisation tracks — <https://tac.nist.gov/>.
- **LLM-as-judge:** Lianmin Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", *NeurIPS 2023*. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
- **LLM judges are not fair evaluators:** Peiyi Wang et al., "Large Language Models are not Fair Evaluators", *ACL 2024 Findings*. [arXiv:2305.17926](https://arxiv.org/abs/2305.17926).
- **G-Eval:** Yang Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment", *EMNLP 2023*. [arXiv:2303.16634](https://arxiv.org/abs/2303.16634).

## Libraries and reference implementations

- **Hugging Face `transformers`** — <https://huggingface.co/docs/transformers/index>
- **Hugging Face `run_summarization.py` example** — <https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization>
- **Hugging Face `evaluate`** — <https://huggingface.co/docs/evaluate/index>
- **Hugging Face `datasets`** — <https://huggingface.co/docs/datasets/index>
- **`Outlines`** — <https://github.com/dottxt-ai/outlines>
- **`Guidance`** — <https://github.com/guidance-ai/guidance>
- **`lm-format-enforcer`** — <https://github.com/noamgat/lm-format-enforcer>
- **`XGrammar`** — <https://github.com/mlc-ai/xgrammar>
- **`SummaC`** — <https://github.com/tingofurro/summac>
- **`FactSumm`** — <https://github.com/Huffon/factsumm>
- **`bert_score`** — <https://github.com/Tiiiger/bert_score>
- **`rouge_score`** — <https://github.com/google-research/google-research/tree/master/rouge>
- **`sumy` (classical extractive baselines)** — <https://github.com/miso-belica/sumy>
- **LangChain summarisation chains** — <https://python.langchain.com/docs/tutorials/summarization/>
- **LlamaIndex response synthesis** — <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes/>

## Companion tracks

- **`rag-engineer` track** — retrieval, chunking, reranking for retrieval-augmented summarisation over corpora. Owns everything upstream of the summariser's context.
- **`mod-105` (this track)** — abstractive QA — shares the encoder-decoder recipe and hallucination surface.
- **`mod-111` (this track)** — deeper NLP evaluation methodology (statistical significance, annotator agreement, dataset governance).
- **`mod-112` (this track)** — production NLP pipelines — how to deploy summarisers with the faithfulness gates and abstention rules from this module.
