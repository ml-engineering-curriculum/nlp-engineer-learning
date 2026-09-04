# Resources for mod-104 · Sequence Labelling and Information Extraction

Prefer primary sources: papers, standards, and official documentation. Blog posts and tutorials are included only where they are the canonical or most-referenced explanation.

## Textbooks and canonical references

- **Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. (in progress)** — <https://web.stanford.edu/~jurafsky/slp3/>. Chapters 8 (Sequence Labelling for Parts of Speech and Named Entities), 17 (Information Extraction), and 22 (Coreference Resolution) cover the classical grounding for the whole module.
- **Manning & Schütze, *Foundations of Statistical Natural Language Processing*, MIT Press, 1999.** Older but still the reference for the classical HMM / CRF era.
- **Sutton & McCallum, ["An Introduction to Conditional Random Fields"](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf), *Foundations and Trends in Machine Learning*, 2012.** The CRF tutorial that every course cites.

## Tagging schemes and entity-level evaluation

- **Ramshaw & Marcus, ["Text Chunking Using Transformation-Based Learning"](https://aclanthology.org/W95-0107/), *Third Workshop on Very Large Corpora, 1995*.** Origin of IOB / BIO tagging.
- **Tjong Kim Sang & De Meulder, ["Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"](https://aclanthology.org/W03-0419/), *CoNLL 2003*.** Defines the CoNLL-2003 dataset, tagging scheme, and evaluation script.
- **Ratinov & Roth, ["Design Challenges and Misconceptions in Named Entity Recognition"](https://aclanthology.org/W09-1119/), *CoNLL 2009*.** BILOU scheme and modelling-choice ablations.
- **Reimers & Gurevych, ["Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging"](https://arxiv.org/abs/1707.09861), *EMNLP 2017*.** Seed variance in sequence-tagging benchmarks.
- **Nakayama, ["seqeval: A Python framework for sequence labeling evaluation"](https://github.com/chakki-works/seqeval).** Reference implementation of entity-level P/R/F1.
- **CoNLL-2003 official evaluation script (`conlleval`)** — <https://www.clips.uantwerpen.be/conll2003/ner/>.

## Encoders and tokenisation

- **Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), *NeurIPS 2017*.**
- **Devlin, Chang, Lee & Toutanova, ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805), *NAACL 2019*.**
- **Liu et al., ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692), *arXiv 2019*.**
- **He, Liu, Gao & Chen, ["DeBERTa: Decoding-enhanced BERT with Disentangled Attention"](https://arxiv.org/abs/2006.03654), *ICLR 2021*.**
- **He, Gao & Chen, ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543), *ICLR 2023*.**
- **Sanh, Debut, Chaumond & Wolf, ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"](https://arxiv.org/abs/1910.01108), *NeurIPS Workshop 2019*.**
- **Conneau et al., ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116), *ACL 2020*.** XLM-R.
- **Joshi et al., ["SpanBERT: Improving Pre-training by Representing and Predicting Spans"](https://arxiv.org/abs/1907.10529), *TACL 2020*.** Span pretraining objective, natural fit for span-based IE and coref.

## Subword-to-word alignment

- **Sennrich, Haddow & Birch, ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909), *ACL 2016*.** BPE, the underlying subword algorithm for RoBERTa/DeBERTa.
- **Kudo & Richardson, ["SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"](https://arxiv.org/abs/1808.06226), *EMNLP 2018*.**
- **Hugging Face `transformers` — Token Classification tutorial** — <https://huggingface.co/docs/transformers/tasks/token_classification>. Canonical `word_ids` + `-100` alignment recipe.
- **Wu, He & Liu, ["Not All Attention Is Needed: Gated Attention Network for Sequence Data"](https://arxiv.org/abs/1912.00349), *AAAI 2020*.** First-subword-only alignment analysis.

## CRF heads and structured decoders

- **Lafferty, McCallum & Pereira, ["Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"](https://repository.upenn.edu/cis_papers/159/), *ICML 2001*.** The CRF paper.
- **Lample, Ballesteros, Subramanian, Kawakami & Dyer, ["Neural Architectures for Named Entity Recognition"](https://aclanthology.org/N16-1030/), *NAACL 2016*.** Bi-LSTM + CRF, the pre-transformer NER workhorse.
- **Ma & Hovy, ["End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF"](https://aclanthology.org/P16-1101/), *ACL 2016*.**
- **Souza, Nogueira & Lotufo, ["Portuguese Named Entity Recognition using BERT-CRF"](https://arxiv.org/abs/1909.10649), *arXiv 2019*.** Whether the CRF still helps on top of BERT.
- **`pytorch-crf`** — <https://github.com/kmkurn/pytorch-crf>. Reference PyTorch CRF layer.

## Span-based NER

- **Eberts & Ulges, ["Span-based Joint Entity and Relation Extraction with Transformer Pre-training"](https://arxiv.org/abs/1909.07755), *ECAI 2020*.** SpERT.
- **Yu, Bohnet & Poesio, ["Named Entity Recognition as Dependency Parsing"](https://arxiv.org/abs/2005.07150), *ACL 2020*.** Biaffine nested NER.
- **Wang, Shou, Chen, Chen & Chen, ["Pyramid: A Layered Model for Nested Named Entity Recognition"](https://aclanthology.org/2020.acl-main.525/), *ACL 2020*.**
- **Shen et al., ["Locate and Label: A Two-stage Identifier for Nested Named Entity Recognition"](https://arxiv.org/abs/2105.06804), *ACL 2021*.**
- **Zaratiana et al., ["GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"](https://arxiv.org/abs/2311.08526), *NAACL 2024*.** Zero-shot label-aware span NER.
- **Zhou et al., ["UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition"](https://arxiv.org/abs/2308.03279), *ICLR 2024*.**
- **Zhong & Chen, ["A Frustratingly Easy Approach for Entity and Relation Extraction"](https://arxiv.org/abs/2010.12812), *NAACL 2021*.** Introduced span-marker encoding for both NER and RE.

### Discontinuous NER

- **Dai, Karimi, Hachey & Paris, ["An Effective Transition-based Model for Discontinuous NER"](https://aclanthology.org/2020.acl-main.520/), *ACL 2020*.**
- **Fei, Ren & Ji, ["Rethinking Boundaries: End-to-End Recognition of Discontinuous Mentions with Pointer Networks"](https://arxiv.org/abs/2003.11080), *AAAI 2021*.**

## Long-context and document-level NER

- **Beltagy, Peters & Cohan, ["Longformer: The Long-Document Transformer"](https://arxiv.org/abs/2004.05150), *arXiv 2020*.**
- **Zaheer et al., ["Big Bird: Transformers for Longer Sequences"](https://arxiv.org/abs/2007.14062), *NeurIPS 2020*.**
- **Yang et al., ["Clinical-Longformer and Clinical-BigBird: Transformers for Long Clinical Sequences"](https://arxiv.org/abs/2201.11838), *arXiv 2022*.** Domain-adapted long-context encoders.

## Relation extraction

- **Zhang, Zhong, Chen, Angeli & Manning, ["Position-aware Attention and Supervised Data Improve Slot Filling"](https://aclanthology.org/D17-1004/), *EMNLP 2017*.** TACRED and the position-aware attention baseline.
- **Alt, Gabryszak & Hennig, ["TACRED Revisited: A Thorough Evaluation of the TACRED Relation Extraction Task"](https://aclanthology.org/2020.acl-main.142/), *ACL 2020*.**
- **Stoica, Platanios & Póczos, ["Re-TACRED: Addressing Shortcomings of the TACRED Dataset"](https://arxiv.org/abs/2104.08398), *AAAI 2021*.**
- **Zhong & Chen, ["A Frustratingly Easy Approach for Entity and Relation Extraction"](https://arxiv.org/abs/2010.12812), *NAACL 2021*.** Typed-marker pipeline.
- **Ye et al., ["Packed Levitated Marker for Entity and Relation Extraction"](https://arxiv.org/abs/2109.06067), *ACL 2022*.**
- **Wang et al., ["TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking"](https://arxiv.org/abs/2010.13415), *COLING 2020*.**
- **Wang, Shou, Chen, Chen & Chen, ["UniRE: A Unified Label Space for Entity Relation Extraction"](https://arxiv.org/abs/2107.04292), *ACL 2021*.**
- **Yao et al., ["DocRED: A Large-Scale Document-Level Relation Extraction Dataset"](https://arxiv.org/abs/1906.06127), *ACL 2019*.**
- **Zhou, Chen, Zhang, Peng, Yang & Huang, ["Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling"](https://arxiv.org/abs/2010.11304), *AAAI 2021*.** ATLOP.
- **Xu, Wang & Sun, ["Entity Structure Within and Throughout: Modeling Mention Dependencies for Document-Level Relation Extraction"](https://arxiv.org/abs/2102.10249), *AAAI 2021*.** SSAN.
- **Mintz, Bills, Snow & Jurafsky, ["Distant supervision for relation extraction without labeled data"](https://aclanthology.org/P09-1113/), *ACL 2009*.**
- **Riedel, Yao & McCallum, ["Modeling Relations and Their Mentions without Labeled Text"](https://link.springer.com/chapter/10.1007/978-3-642-15939-8_10), *ECML-PKDD 2010*.** Multi-instance learning.
- **Cabot & Navigli, ["REBEL: Relation Extraction By End-to-end Language generation"](https://aclanthology.org/2021.findings-emnlp.204/), *EMNLP Findings 2021*.**
- **Paolini et al., ["Structured Prediction as Translation between Augmented Natural Languages"](https://arxiv.org/abs/2101.05779), *ICLR 2021*.** TANL.

## Event extraction and slot filling

- **Doddington et al., ["The Automatic Content Extraction (ACE) Program – Tasks, Data, and Evaluation"](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf), *LREC 2004*.** ACE 2005 canonical ontology.
- **Baker, Fillmore & Lowe, ["The Berkeley FrameNet Project"](https://aclanthology.org/C98-1013/), *COLING 1998*.**
- **Palmer, Gildea & Kingsbury, ["The Proposition Bank: An Annotated Corpus of Semantic Roles"](https://aclanthology.org/J05-1004/), *Computational Linguistics 2005*.** PropBank.
- **Wang et al., ["MAVEN: A Massive General Domain Event Detection Dataset"](https://arxiv.org/abs/2004.13590), *EMNLP 2020*.**
- **Kim, Ohta, Pyysalo, Kano & Tsujii, ["Overview of BioNLP'09 Shared Task on Event Extraction"](https://aclanthology.org/W09-1401/), *BioNLP Workshop 2009*.**
- **Wadden, Wennberg, Luan & Hajishirzi, ["Entity, Relation, and Event Extraction with Contextualized Span Representations"](https://arxiv.org/abs/1909.03546), *EMNLP 2019*.** DYGIE++.
- **Lin, Ji, Huang & Wu, ["A Joint Neural Model for Information Extraction with Global Features"](https://aclanthology.org/2020.acl-main.713/), *ACL 2020*.** OneIE.
- **Hsu et al., ["DEGREE: A Data-Efficient Generation-Based Event Extraction Model"](https://arxiv.org/abs/2108.12724), *NAACL 2022*.**
- **Chen, Zhuo & Wang, ["BERT for Joint Intent Classification and Slot Filling"](https://arxiv.org/abs/1902.10909), *arXiv 2019*.**
- **FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual NLU Dataset"](https://arxiv.org/abs/2204.08582), *ACL 2023*.**
- **Namazifar, Papangelis, Hakkani-Tur & Tur, ["Language Model is All You Need: Natural Language Understanding as Question Answering"](https://arxiv.org/abs/2011.03023), *ICASSP 2021*.** QA framing for slot filling.

## Schema-driven structured extraction with LLMs

- **Willard & Louf, ["Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702), *arXiv 2023*.** Outlines.
- **Zhang et al., ["XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models"](https://arxiv.org/abs/2411.15100), *arXiv 2024*.**
- **Beurer-Kellner, Fischer & Vechev, ["Prompting Is Programming: A Query Language for Large Language Models"](https://arxiv.org/abs/2212.06094), *PLDI 2023*.** LMQL.
- **Wang et al., ["InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction"](https://arxiv.org/abs/2304.08085), *arXiv 2023*.**
- **Sainz et al., ["GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction"](https://arxiv.org/abs/2310.03668), *ICLR 2024*.**
- **Anthropic — tool use documentation** — <https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview>.
- **OpenAI — structured outputs documentation** — <https://platform.openai.com/docs/guides/structured-outputs>.
- **Outlines** — <https://github.com/dottxt-ai/outlines>.
- **`instructor`** — <https://github.com/instructor-ai/instructor>. Pydantic-first wrapper around structured-output APIs.
- **`xgrammar`** — <https://github.com/mlc-ai/xgrammar>.

## Entity linking

- **Wu, Petroni, Josifoski, Riedel & Zettlemoyer, ["Scalable Zero-shot Entity Linking with Dense Entity Retrieval"](https://aclanthology.org/2020.emnlp-main.519/), *EMNLP 2020*.** BLINK.
- **De Cao, Izacard, Riedel & Petroni, ["Autoregressive Entity Retrieval"](https://arxiv.org/abs/2010.00904), *ICLR 2021*.** GENRE.
- **De Cao et al., ["Multilingual Autoregressive Entity Linking"](https://arxiv.org/abs/2103.12528), *TACL 2022*.** mGENRE.
- **van Hulst, Hasibi, Dercksen, Balog & de Vries, ["REL: An Entity Linker Standing on the Shoulders of Giants"](https://arxiv.org/abs/2006.01969), *SIGIR 2020*.**
- **Neumann, King, Beltagy & Ammar, ["ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing"](https://arxiv.org/abs/1902.07669), *BioNLP 2019*.**
- **Liu, Shareghi, Meng, Basaldella & Collier, ["Self-Alignment Pretraining for Biomedical Entity Representations"](https://arxiv.org/abs/2010.11784), *NAACL 2021*.** SapBERT.
- **Hoffart et al., ["Robust Disambiguation of Named Entities in Text"](https://aclanthology.org/D11-1072/), *EMNLP 2011*.** AIDA-CoNLL benchmark.
- **Botha, Shan & Gillick, ["Entity Linking in 100 Languages"](https://arxiv.org/abs/2011.02690), *EMNLP 2020*.** XL-BEL.
- **Ji, Nothman & Hachey, ["Overview of TAC-KBP2015 Tri-lingual Entity Discovery and Linking"](https://tac.nist.gov/publications/2015/additional.papers/TAC2015.KBP_Tri-lingual_Entity_Discovery_and_Linking_overview.proceedings.pdf), *TAC 2015*.** NIL definitions.

### Knowledge bases

- **Wikidata** — <https://www.wikidata.org/>. Public knowledge graph; Q-ids.
- **UMLS Metathesaurus** — <https://www.nlm.nih.gov/research/umls/index.html>. Biomedical concepts; CUIs. Licence required.
- **MeSH** — <https://www.nlm.nih.gov/mesh/>.
- **GLEIF Legal Entity Identifier** — <https://www.gleif.org/en/lei-data/access-and-use-lei-data>.
- **ChEMBL** — <https://www.ebi.ac.uk/chembl/>.
- **UniProt** — <https://www.uniprot.org/>.

## Coreference resolution

- **Lee, He, Lewis & Zettlemoyer, ["End-to-end Neural Coreference Resolution"](https://arxiv.org/abs/1707.07045), *EMNLP 2017*.** Modern architecture.
- **Joshi, Levy, Zettlemoyer & Weld, ["BERT for Coreference Resolution: Baselines and Analysis"](https://arxiv.org/abs/1908.09091), *EMNLP 2019*.**
- **Xu & Choi, ["Revealing the Myth of Higher-Order Inference in Coreference Resolution"](https://arxiv.org/abs/2009.12013), *EMNLP 2020*.** coref-hoi.
- **Kirstain, Ram & Levy, ["Coreference Resolution without Span Representations"](https://arxiv.org/abs/2101.00434), *ACL-IJCNLP 2021*.** s2e-coref.
- **Dobrovolskii, ["Word-Level Coreference Resolution"](https://arxiv.org/abs/2109.04127), *EMNLP 2021*.** wl-coref.
- **Otmazgin, Cattan & Goldberg, ["F-COREF: Fast, Accurate and Easy to Use Coreference Resolution"](https://arxiv.org/abs/2209.04280), *AACL 2022*.**
- **Martinelli, Barba & Navigli, ["Maverick: Efficient and Accurate Coreference Resolution Defying Recent Trends"](https://arxiv.org/abs/2407.21489), *ACL 2024*.**
- **Xia, Sedoc & Van Durme, ["Incremental Neural Coreference Resolution in Constant Memory"](https://arxiv.org/abs/2005.00128), *EMNLP 2020*.**
- **Lee, Chang, Peirsman, Chambers, Surdeanu & Jurafsky, ["Deterministic Coreference Resolution Based on Entity-centric, Precision-ranked Rules"](https://aclanthology.org/J13-4004/), *Computational Linguistics 2013*.** Stanford multi-pass sieve.
- **Cattan, Eirew, Stanovsky, Joshi & Dagan, ["Streamlining Cross-Document Coreference Resolution: Evaluation and Modeling"](https://arxiv.org/abs/2009.11032), *arXiv 2020*.**
- **Caciularu et al., ["Cross-Document Language Modeling"](https://arxiv.org/abs/2101.00406), *EMNLP Findings 2021*.** CDLM.

### Coref metrics and bias

- **Vilain, Burger, Aberdeen, Connolly & Hirschman, ["A Model-Theoretic Coreference Scoring Scheme"](https://aclanthology.org/M95-1005/), *MUC-6 1995*.** MUC.
- **Bagga & Baldwin, ["Algorithms for Scoring Coreference Chains"](https://aclanthology.org/W98-1119/), *LREC 1998*.** B³.
- **Luo, ["On Coreference Resolution Performance Metrics"](https://aclanthology.org/H05-1004/), *HLT-EMNLP 2005*.** CEAF.
- **Recasens & Hovy, ["BLANC: Implementing the Rand index for coreference evaluation"](https://link.springer.com/article/10.1007/s10579-011-9145-0), *Natural Language Engineering 2011*.**
- **Pradhan, Ramshaw, Marcus, Palmer, Weischedel & Xue, ["CoNLL-2011 Shared Task: Modeling Unrestricted Coreference in OntoNotes"](https://aclanthology.org/W11-1901/), *CoNLL 2011*.**
- **Pradhan, Moschitti, Xue, Uryupina & Zhang, ["CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes"](https://aclanthology.org/W12-4501/), *EMNLP-CoNLL 2012*.**
- **Rudinger, Naradowsky, Leonard & Van Durme, ["Gender Bias in Coreference Resolution"](https://aclanthology.org/N18-2002/), *NAACL 2018*.** WinoGender.
- **Zhao, Wang, Yatskar, Ordonez & Chang, ["Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods"](https://arxiv.org/abs/1804.06876), *NAACL 2018*.** WinoBias.
- **Webster, Recasens, Axelrod & Baldridge, ["Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns"](https://arxiv.org/abs/1810.05201), *TACL 2018*.** GAP.
- **`coval`** — <https://github.com/ns-moosavi/coval>. Reference coref scorer.

## Active learning and weak supervision

- **Settles, ["Active Learning Literature Survey"](https://minds.wisconsin.edu/handle/1793/60660), *University of Wisconsin–Madison, 2010*.** Comprehensive reference.
- **Settles, Craven & Friedland, ["Active Learning with Real Annotation Costs"](https://web.eecs.umich.edu/~aparanjp/eecs486/papers/Real_Annotation_Costs.pdf), *NIPS Workshop 2008*.**
- **Shen, Yun, Lipton, Kronrod & Anandkumar, ["Deep Active Learning for Named Entity Recognition"](https://arxiv.org/abs/1707.05928), *ICLR 2018*.**
- **Zhdanov, ["Diverse mini-batch Active Learning"](https://arxiv.org/abs/1901.05954), *arXiv 2019*.**
- **Ratner, Bach, Ehrenberg, Fries, Wu & Ré, ["Snorkel: Rapid Training Data Creation with Weak Supervision"](https://arxiv.org/abs/1711.10160), *VLDB 2018*.**
- **Lison, Barnes, Hubin & Touileb, ["skweak: Weak Supervision Made Easy for NLP"](https://arxiv.org/abs/2104.09683), *ACL Demo 2021*.**
- **Zhang, Yu, Wang, Ré, Wang, Shao, Wang & Chen, ["WRENCH: A Comprehensive Benchmark for Weak Supervision"](https://arxiv.org/abs/2109.11377), *NeurIPS Datasets 2021*.**

## Datasets

### Newswire / general English NER

- **Tjong Kim Sang & De Meulder, ["Introduction to the CoNLL-2003 Shared Task"](https://aclanthology.org/W03-0419/), *CoNLL 2003*.** CoNLL-2003.
- **Weischedel et al., ["OntoNotes Release 5.0"](https://catalog.ldc.upenn.edu/LDC2013T19), *LDC 2013*.** OntoNotes 5.
- **Derczynski, Nichols, van Erp & Limsopatham, ["Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition"](https://aclanthology.org/W17-4418/), *WNUT 2017*.**

### Nested NER

- **Kim, Ohta, Tateisi & Tsujii, ["GENIA corpus — a semantically annotated corpus for bio-textmining"](https://academic.oup.com/bioinformatics/article/19/suppl_1/i180/227927), *Bioinformatics 2003*.**
- **Ringland, Dai, Hachey, Karimi, Paris & Curran, ["NNE: A Dataset for Nested Named Entity Recognition in English Newswire"](https://aclanthology.org/P19-1510/), *ACL 2019*.**
- **Doddington et al., ["The Automatic Content Extraction (ACE) Program"](http://www.lrec-conf.org/proceedings/lrec2004/pdf/5.pdf), *LREC 2004*.** ACE 2004/2005.

### Biomedical

- **Li et al., ["BioCreative V CDR task corpus: a resource for chemical disease relation extraction"](https://doi.org/10.1093/database/baw068), *Database 2016*.** BC5CDR.
- **Krallinger et al., ["Overview of the BioCreative VI chemical-protein interaction Track"](https://biocreative.bioinformatics.udel.edu/media/store/files/2017/chemprot_overview_v03.pdf), *BioCreative VI 2017*.** ChemProt.
- **Mohan & Li, ["MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts"](https://arxiv.org/abs/1902.09476), *AKBC 2019*.**
- **Uzuner, Solti & Cadag, ["Extracting Medication Information from Clinical Text"](https://pubmed.ncbi.nlm.nih.gov/20819857/), *JAMIA 2010*.** i2b2 medication extraction shared task.

### Scientific

- **Luan, He, Ostendorf & Hajishirzi, ["Multi-Task Identification of Entities, Relations, and Coreference for Scientific Knowledge Graph Construction"](https://aclanthology.org/D18-1360/), *EMNLP 2018*.** SciERC.

### Relation and event extraction

- **Zhang et al., ["Position-aware Attention and Supervised Data Improve Slot Filling"](https://aclanthology.org/D17-1004/), *EMNLP 2017*.** TACRED.
- **Hendrickx et al., ["SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals"](https://aclanthology.org/S10-1006/), *SemEval 2010*.**
- **Yao et al., ["DocRED: A Large-Scale Document-Level Relation Extraction Dataset"](https://arxiv.org/abs/1906.06127), *ACL 2019*.**
- **Wang et al., ["MAVEN: A Massive General Domain Event Detection Dataset"](https://arxiv.org/abs/2004.13590), *EMNLP 2020*.**

### Slot filling / conversational NLU

- **FitzGerald et al., ["MASSIVE: A 1M-Example Multilingual NLU Dataset"](https://arxiv.org/abs/2204.08582), *ACL 2023*.**
- **Coucke et al., ["Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces"](https://arxiv.org/abs/1805.10190), *arXiv 2018*.** SNIPS.
- **Hemphill, Godfrey & Doddington, ["The ATIS Spoken Language Systems Pilot Corpus"](https://aclanthology.org/H90-1021/), *Speech and Natural Language Workshop 1990*.**

### Entity linking

- **Hoffart et al., ["Robust Disambiguation of Named Entities in Text"](https://aclanthology.org/D11-1072/), *EMNLP 2011*.** AIDA-CoNLL.
- **Petroni et al., ["KILT: a Benchmark for Knowledge Intensive Language Tasks"](https://aclanthology.org/2021.naacl-main.200/), *NAACL 2021*.**

### Coreference

- **Pradhan et al., ["CoNLL-2012 Shared Task"](https://aclanthology.org/W12-4501/), *EMNLP-CoNLL 2012*.** OntoNotes coref.
- **Chen, Fan, Chen, Yu, Xu & Wang, ["PreCo: A Large-scale Dataset in Preschool Vocabulary for Coreference Resolution"](https://arxiv.org/abs/1810.09807), *EMNLP 2018*.**

### Legal, financial

- **Hendrycks, Burns, Chen & Ball, ["CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"](https://arxiv.org/abs/2103.06268), *NeurIPS Datasets 2021*.**
- **Malo, Sinha, Korhonen, Wallenius & Takala, ["Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts"](https://arxiv.org/abs/1307.5336), *JASIST 2014*.** Financial PhraseBank.

## Domain-adapted encoders

### Biomedical / clinical

- **Lee, Yoon, Kim, Kim, Kim, So & Kang, ["BioBERT: a pre-trained biomedical language representation model for biomedical text mining"](https://arxiv.org/abs/1901.08746), *Bioinformatics 2020*.**
- **Alsentzer et al., ["Publicly Available Clinical BERT Embeddings"](https://arxiv.org/abs/1904.03323), *ClinicalNLP 2019*.** Bio_ClinicalBERT.
- **Yang et al., ["A large language model for electronic health records"](https://www.nature.com/articles/s41746-022-00742-2), *npj Digital Medicine 2022*.** GatorTron.
- **Gu, Tinn, Cheng, Lucas, Usuyama, Liu, Naumann, Gao & Poon, ["Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing"](https://arxiv.org/abs/2007.15779), *ACM HEALTH 2021*.** PubMedBERT.
- **Beltagy, Lo & Cohan, ["SciBERT: A Pretrained Language Model for Scientific Text"](https://arxiv.org/abs/1903.10676), *EMNLP-IJCNLP 2019*.**

### Legal

- **Chalkidis, Fergadiotis, Malakasiotis, Aletras & Androutsopoulos, ["LEGAL-BERT: The Muppets straight out of Law School"](https://arxiv.org/abs/2010.02559), *EMNLP Findings 2020*.**

### Financial

- **Yang, Uy & Huang, ["FinBERT: A Pretrained Language Model for Financial Communications"](https://arxiv.org/abs/2006.08097), *arXiv 2020*.**

## Toolkits and libraries

### Core

- **Hugging Face `transformers`** — <https://github.com/huggingface/transformers>.
- **Hugging Face `datasets`** — <https://github.com/huggingface/datasets>.
- **Hugging Face `evaluate`** — <https://github.com/huggingface/evaluate>. Includes `seqeval`.
- **Hugging Face `Trainer` documentation** — <https://huggingface.co/docs/transformers/main/en/main_classes/trainer>.
- **`accelerate`** — <https://github.com/huggingface/accelerate>.
- **`seqeval`** — <https://github.com/chakki-works/seqeval>.

### NER / tagging

- **spaCy** — <https://spacy.io/>. Production-hardened NER with a transition-based decoder.
- **spaCy `spacy-transformers`** — <https://spacy.io/universe/project/spacy-transformers>.
- **flair** — <https://github.com/flairNLP/flair>.
- **SpanMarker** — <https://github.com/tomaarsen/SpanMarkerNER>.
- **SpERT** — <https://github.com/lavis-nlp/spert>.
- **GLiNER** — <https://github.com/urchade/GLiNER>.
- **AllenNLP** — <https://github.com/allenai/allennlp>.
- **Stanza** — <https://stanfordnlp.github.io/stanza/>. Includes biomedical models.
- **`pytorch-crf`** — <https://github.com/kmkurn/pytorch-crf>.

### Relation and event extraction

- **PL-Marker** — <https://github.com/thunlp/PL-Marker>.
- **OpenNRE** — <https://github.com/thunlp/OpenNRE>. Relation-extraction toolkit.
- **DYGIE++** — <https://github.com/dwadden/dygiepp>.

### Entity linking

- **BLINK** — <https://github.com/facebookresearch/BLINK>.
- **GENRE** — <https://github.com/facebookresearch/GENRE>.
- **REL** — <https://github.com/informagi/REL>.
- **scispaCy** — <https://github.com/allenai/scispacy>.

### Coreference

- **`fastcoref`** — <https://github.com/shon-otmazgin/fastcoref>.
- **`maverick-coref`** — <https://github.com/SapienzaNLP/maverick-coref>.
- **`coreferee`** — <https://github.com/msg-systems/coreferee>. spaCy pipeline component.
- **Stanford CoreNLP** — <https://stanfordnlp.github.io/CoreNLP/>. Java multi-pass sieve.

### Structured generation / LLM extraction

- **Outlines** — <https://github.com/dottxt-ai/outlines>.
- **xgrammar** — <https://github.com/mlc-ai/xgrammar>.
- **`instructor`** — <https://github.com/instructor-ai/instructor>.
- **`llama.cpp` GBNF grammars** — <https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md>.

### Active learning / weak supervision / labelling UIs

- **`prodigy`** — <https://prodi.gy/>. Commercial labelling UI.
- **Label Studio** — <https://labelstud.io/>.
- **`small-text`** — <https://github.com/webis-de/small-text>.
- **`modAL`** — <https://github.com/modAL-python/modAL>.
- **Snorkel** — <https://github.com/snorkel-team/snorkel>.
- **`skweak`** — <https://github.com/NorskRegnesentral/skweak>.
- **`wrench`** — <https://github.com/JieyuZ2/wrench>.

## Model cards worth reading in full

- **`microsoft/deberta-v3-base`** — <https://huggingface.co/microsoft/deberta-v3-base>.
- **`xlm-roberta-base`** — <https://huggingface.co/FacebookAI/xlm-roberta-base>.
- **`allenai/longformer-base-4096`** — <https://huggingface.co/allenai/longformer-base-4096>.
- **`allenai/scibert_scivocab_uncased`** — <https://huggingface.co/allenai/scibert_scivocab_uncased>.
- **`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`** — <https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract>.
- **`emilyalsentzer/Bio_ClinicalBERT`** — <https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT>.
- **`nlpaueb/legal-bert-base-uncased`** — <https://huggingface.co/nlpaueb/legal-bert-base-uncased>.
- **`ProsusAI/finbert`** — <https://huggingface.co/ProsusAI/finbert>.
- **`Babelscape/rebel-large`** — <https://huggingface.co/Babelscape/rebel-large>.
- **`urchade/gliner_medium-v2.1`** — <https://huggingface.co/urchade/gliner_medium-v2.1>.
- **`SapBERT` from `cambridgeltl`** — <https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext>.
