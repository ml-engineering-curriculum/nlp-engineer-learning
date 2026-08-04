# Resources for mod-101 · Tokenization & Text Foundations

Prefer primary sources: papers, standards, and official documentation. Blog posts are included only where they are the canonical explanation.

## Sub-word tokenization — primary papers

- **BPE (compression):** Philip Gage, "A New Algorithm for Data Compression", *The C Users Journal*, 12(2), 1994.
- **BPE for NMT:** Rico Sennrich, Barry Haddow, Alexandra Birch, "Neural Machine Translation of Rare Words with Subword Units", *ACL 2016*. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).
- **WordPiece (original):** Mike Schuster, Kaisuke Nakajima, "Japanese and Korean Voice Search", *ICASSP 2012*.
- **WordPiece (deployed at scale):** Yonghui Wu et al., "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation", 2016. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144).
- **BERT (WordPiece in practice):** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", *NAACL 2019*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- **Unigram LM + sub-word regularisation:** Taku Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates", *ACL 2018*. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959).
- **SentencePiece library:** Taku Kudo, John Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", *EMNLP 2018 System Demonstrations*. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226).
- **Byte-level BPE (GPT-2):** Alec Radford et al., "Language Models are Unsupervised Multitask Learners", 2019. [OpenAI paper (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- **BPE-dropout:** Ivan Provilkov, Dmitrii Emelianenko, Elena Voita, "BPE-Dropout: Simple and Effective Subword Regularization", *ACL 2020*. [arXiv:1910.13267](https://arxiv.org/abs/1910.13267).

## Tokenizer libraries — official docs

- **Hugging Face `tokenizers`** — <https://huggingface.co/docs/tokenizers/index>
- **Hugging Face NLP course, "Building a tokenizer"** — <https://huggingface.co/learn/nlp-course/en/chapter6/1>
- **Google `sentencepiece`** — <https://github.com/google/sentencepiece>
- **OpenAI `tiktoken`** — <https://github.com/openai/tiktoken>

## Unicode standards

- **UAX #29 · Unicode Text Segmentation** — <https://www.unicode.org/reports/tr29/>
- **UAX #15 · Unicode Normalization Forms** — <https://www.unicode.org/reports/tr15/>
- **UAX #9 · Unicode Bidirectional Algorithm** — <https://www.unicode.org/reports/tr9/>
- **UTS #10 · Unicode Collation Algorithm** — <https://www.unicode.org/reports/tr10/>
- **The Unicode Standard (current)** — <https://www.unicode.org/versions/latest/>
- **Character properties / General_Category values** — <https://www.unicode.org/reports/tr44/>

## Language tagging

- **BCP-47 · Tags for Identifying Languages** (RFC 5646) — <https://www.rfc-editor.org/info/bcp47>
- **ISO 15924 · Codes for the representation of names of scripts** — <https://www.unicode.org/iso15924/iso15924-codes.html>
- **W3C Language tags in HTML and XML** — <https://www.w3.org/International/articles/language-tags/>
- **`langcodes` (Python)** — <https://github.com/rspeer/langcodes>

## ICU and Python wrappers

- **International Components for Unicode (ICU)** — <https://icu.unicode.org/>
- **`icu4x` (Rust)** — <https://github.com/unicode-org/icu4x>
- **PyICU** — <https://gitlab.pyicu.org/main/pyicu>
- **`grapheme` (Python)** — <https://github.com/alvinlindstam/grapheme>
- **`uniseg` (Python)** — <https://github.com/emikanle/uniseg-python>

## Script-specific references

- **CJK / Han unification:** *The Unicode Standard*, ch. 18 "East Asia".
- **Arabic script:** *The Unicode Standard*, ch. 9 "Middle East-I: Ancient and Non-Modern Scripts", esp. §9.2 "Arabic". W3C, ["Arabic & Persian text layout"](https://www.w3.org/International/articles/arabic/).
- **Indic scripts:** *The Unicode Standard*, ch. 12 "South Asia-I". W3C, ["Language enablement (Indic scripts)"](https://www.w3.org/International/i18n-tests/indic).

## Transformer architectures — primary papers

- **The original transformer:** Ashish Vaswani et al., "Attention Is All You Need", *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- **Encoder-only pretraining:** Devlin et al., BERT (above); Yinhan Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019, [arXiv:1907.11692](https://arxiv.org/abs/1907.11692); Kevin Clark et al., "ELECTRA", *ICLR 2020*, [arXiv:2003.10555](https://arxiv.org/abs/2003.10555).
- **Encoder-decoder pretraining:** Colin Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5), *JMLR 2020*, [arXiv:1910.10683](https://arxiv.org/abs/1910.10683); Mike Lewis et al., "BART", *ACL 2020*, [arXiv:1910.13461](https://arxiv.org/abs/1910.13461); Jingqing Zhang et al., "PEGASUS", *ICML 2020*, [arXiv:1912.08777](https://arxiv.org/abs/1912.08777).
- **Decoder-only, in-context learning:** Alec Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1), 2018, [OpenAI paper (PDF)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf); Radford et al., GPT-2 (above); Tom Brown et al., "Language Models are Few-Shot Learners" (GPT-3), *NeurIPS 2020*, [arXiv:2005.14165](https://arxiv.org/abs/2005.14165); Hugo Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023, [arXiv:2302.13971](https://arxiv.org/abs/2302.13971); Hugo Touvron et al., "Llama 2", 2023, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288).

## KV-cache, attention efficiency, long context

- **Multi-Query Attention:** Noam Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150).
- **Grouped-Query Attention:** Joshua Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", *EMNLP 2023*, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
- **FlashAttention:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", *NeurIPS 2022*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135); Tri Dao, "FlashAttention-2", 2023, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).
- **PagedAttention / vLLM:** Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", *SOSP 2023*, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- **Longformer (sliding-window attention):** Iz Beltagy, Matthew E. Peters, Arman Cohan, "Longformer: The Long-Document Transformer", 2020, [arXiv:2004.05150](https://arxiv.org/abs/2004.05150).
- **ALiBi position bias:** Ofir Press, Noah A. Smith, Mike Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", *ICLR 2022*, [arXiv:2108.12409](https://arxiv.org/abs/2108.12409).
- **RoPE rotary positions:** Jianlin Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

## Multilingual and cross-lingual models

- **mBART:** Yinhan Liu et al., "Multilingual Denoising Pre-training for Neural Machine Translation", *TACL 2020*, [arXiv:2001.08210](https://arxiv.org/abs/2001.08210).
- **mT5:** Linting Xue et al., "mT5: A massively multilingual pre-trained text-to-text transformer", *NAACL 2021*, [arXiv:2010.11934](https://arxiv.org/abs/2010.11934).
- **XLM-R:** Alexis Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", *ACL 2020*, [arXiv:1911.02116](https://arxiv.org/abs/1911.02116).
- **NLLB (No Language Left Behind):** NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation", 2022, [arXiv:2207.04672](https://arxiv.org/abs/2207.04672).
- **IndicBERT:** Divyanshu Kakwani et al., "IndicNLPSuite / IndicBERT", *EMNLP Findings 2020*, [aclanthology.org/2020.findings-emnlp.445](https://aclanthology.org/2020.findings-emnlp.445/).

## Reference implementations to read

- **Hugging Face `tokenizers` (Rust)** — <https://github.com/huggingface/tokenizers>
- **SentencePiece (C++ / Python)** — <https://github.com/google/sentencepiece>
- **`tiktoken` (Rust / Python)** — <https://github.com/openai/tiktoken>
- **`transformers` model code (Python)** — <https://github.com/huggingface/transformers> — especially `modeling_bert.py`, `modeling_t5.py`, `modeling_llama.py`.
- **`vLLM` (Python / C++)** — <https://github.com/vllm-project/vllm> — canonical implementation of PagedAttention.
