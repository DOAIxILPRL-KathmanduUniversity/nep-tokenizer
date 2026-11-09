# Industry Findings, Baselines, and Tokenizer Adaptation

**Department of Artificial Intelligence (DOAI) & Information and Language Processing Research Lab (ILPRL)**  
**Kathmandu University, Dhulikhel, Nepal**

This document synthesizes key architectural findings across modern Indic and Devanagari tokenization research, specifies the standard 64k and 128k evaluation protocols, and details how standalone tokenizers relate to foundation model pretraining versus vocabulary expansion for existing models like Google Gemma or Meta LLaMA.

---

## 1. Evaluation Standards: The 64k and 128k Vocab Frontier

In modern Large Language Model (LLM) engineering for multilingual and Indic languages, vocabulary size selection represents a fundamental trade-off:
- **Compression Efficiency vs. Softmax Latency**: Larger vocabularies compress text into fewer sequence tokens (reducing attention complexity $\mathcal{O}(N^2)$ and inference prefill/decode passes), but increase the parameter count of the input embedding matrix ($W_e \in \mathbb{R}^{V \times d}$) and the final output LM projection head ($W_{\text{lm}} \in \mathbb{R}^{d \times V}$).
- **The Standard Evaluation Anchors**:
  - **32,000 (32k)**: The compact footprint tier for edge and resource-constrained environments (e.g. Nep-Tokenizer v2 32k, Yak v1.0 32k).
  - **64,000 (64k)**: The standard efficient frontier for regional and bilingual models (e.g. Nep-Tokenizer v2 64k, Yak 64k, sidskarki 64k, NyayaLM 64k, Himalaya AI v1 64k).
  - **128,000 (128k)**: The standard frontier for high-capacity multilingual and omni models (e.g. Yak 128k, LLaMA-3.1, Param2-17B).

All evaluation configurations in this repository prioritize **32,000**, **64,000**, and **128,000** vocabulary sizes to ensure direct comparability with published baselines.

---

## 2. Key Industry Findings & Architectural Insights

Recent independent studies across Indic tokenization (Param-1/BharatGen, KrutrimAI, Arkios, Yak, Sarvam, and Supernova) highlight several critical linguistic and computational considerations:

### A. The Pre-tokenization Ceiling Bug (Akshara Splitting)
Standard Western LLMs (GPT-2, GPT-4, LLaMA-3, Mistral) define words in their pre-tokenizers strictly as sequences of Unicode letters (`\p{L}+`):
- In Latin scripts, both vowels and consonants are letters (`\p{L}`).
- In **Devanagari and Indic abugidas**, consonants are letters (`\p{L}`), but dependent vowel signs (*matras*: `ा, ि, ी, ु, ू, े, ै, ो, ौ`) and virama (`्`) belong to Unicode categories `Mn` (non-spacing combining mark) or `Mc` (spacing combining mark).
- Consequently, `\p{L}+` forcefully splits Devanagari words at every vowel sign:
  - `नेपाली` under `\p{L}+` splits into `['न', 'े', 'प', 'ा', 'ल', 'ी']` before BPE merges even begin.
- **Mark-Aware Regex (`[\p{L}\p{M}]+`)**: Admitting Unicode marks (`\p{M}`) into the word pattern prevents artificial orthographic syllable splitting and allows words to remain whole.

### B. The Severed Matra Anomaly (Kathmandu Bug)
When words are severed at vowel signs, subsequent subword tokens often begin with a bare combining mark (e.g. `काठमाडौं` $\rightarrow$ `'क' + 'ाठ' + 'म' + 'ाड' + 'ौं'`, where `'ाठ'` and `'ाड'` start with an isolated matra `ा`).
- In broken regex tokenizers (such as Supernova V5 or Gemma-2), 20% to 44% of Nepali tokens begin with an isolated combining mark.
- Mark-aware tokenizers (Nep-Tokenizer v2, Yak, Arkios) ensure that dependent marks remain strictly bound to consonant stems, achieving **0.00% severed matra errors** (100.0% matra cohesion).

### C. Unicode Character Completeness & Silent Byte-Fallback
Byte-level BPE tokenizers avoid unknown token errors (`<unk>`) by falling back to UTF-8 raw bytes. However:
- If an everyday Devanagari character lacks a dedicated single-token ID in the vocabulary, it silently decomposes into **3 raw byte tokens** (`<0xE0>...`).
- Models trained with insufficient character coverage (< 0.9995) or lacking explicit Devanagari block initialization frequently suffer from this silent expansion on less frequent letters (e.g., `ऊ, ऋ, ॠ, ऐ, अं, अः, ॄ, ॅ, ॆ, ॉ, ॊ`).
- **Nep-Tokenizer v2** deterministically guarantees that **all 128 Unicode code points** in the Devanagari block (`U+0900` to `U+097F`) have atomic merges, completely eliminating byte fallback for any Devanagari phoneme.

### D. Isolated Digit Grouping (`[0-9०-९]{1,3}`)
Raw BPE merges digits arbitrarily across numbers based on corpus frequency (e.g. learning `"2023"` as 1 token but `"2024"` as 2 tokens).
- Pre-splitting digits into isolated chunks of 1 to 3 digits (for both ASCII `0-9` and Devanagari `०-९`) ensures that every number has a uniform, compositional subword representation, significantly aiding mathematical reasoning and financial data processing.

---

## 3. From-Scratch Pretraining vs. Adapting Pretrained Models (Gemma, LLaMA)

A crucial engineering distinction must be made regarding how this standalone tokenizer is applied:

### A. Direct Use: Training Foundation Models From Scratch
Nep-Tokenizer v2 is exported in native Hugging Face format (`PreTrainedTokenizerFast`) with `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`. It is designed as the native tokenizer for **training Himalayan foundation models from scratch**:
- The model architecture is initialized with $V = 64{,}053$ (or $V = 32{,}057$).
- The embedding matrix and LM head match the tokenizer's vocabulary index from step 0.
- Native Rust execution enables maximum data-loading throughput during pretraining.

### B. Adapting Existing Open-Weights Models (Gemma, LLaMA, Qwen)
**A standalone tokenizer cannot simply replace the tokenizer of an existing pretrained model.**
In models like Google Gemma (`google/gemma-2-9b`) or Meta LLaMA (`meta-llama/Llama-3.1-8B-Instruct`):
- The model weights ($W_e$ and $W_{\text{lm}}$) are rigidly tied to the base tokenizer's exact token IDs.
- Directly swapping tokenizers scrambles all token mappings, completely destroying pretrained representations.

### C. The Vocabulary Expansion Protocol for Gemma / LLaMA
To utilize the high-quality Nepali subwords learned by `nep-tokenizer` within an existing foundation model, practitioners must follow a **Vocabulary Expansion & Continual Pretraining** pipeline:

1. **Vocabulary Differencing ($\Delta V$)**:
   Extract high-frequency Nepali subwords from `nep-tokenizer` that do not exist in the base model's vocabulary:
   $$\Delta V = V_{\text{nep-tokenizer}} \setminus V_{\text{base}}$$

2. **Resize Embedding Matrix**:
   Extend the model's token embeddings from $V_{\text{base}}$ to $V_{\text{base}} + |\Delta V|$:
   ```python
   model.resize_token_embeddings(len(extended_tokenizer))
   ```

3. **Subword Decomposition Initialization (Piece-Pooling)**:
   Instead of initializing new token embeddings with random Gaussian noise (which destabilizes training and degrades language generation), initialize each new token embedding as the mean of its constituent subwords in the base tokenizer:
   $$E(w_{\text{new}}) = \frac{1}{|P_{\text{base}}(w_{\text{new}})|} \sum_{t \in P_{\text{base}}(w_{\text{new}})} E_{\text{base}}(t)$$

4. **Targeted Continual Pretraining**:
   - Freeze the majority of the transformer backbone (or apply LoRA/QLoRA).
   - Train the newly added token embeddings on high-quality Nepali text to calibrate representation vectors before unfreezing attention layers.
   - This prevents catastrophic forgetting of base reasoning and multilingual capabilities while drastically reducing sequence length on Nepali text.

---

## 4. Evaluated Baselines & Empirical Leaderboard

The benchmark harness in `scripts/benchmark_all_models.py` evaluates 17 models on the **exact same held-out multi-domain corpus** (775 documents across News, Literature, Legal Acts, Colloquial, and Technical domains):

| Model | Architecture | Vocab Size | Nepali Fertility ↓ | English Fertility ↓ | Nepali Tax ↓ | Matra Err% ↓ | Devanagari Cov% ↑ | Bytes/Token (BPT) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DOAIxILPRL Nep-Tokenizer v2 (64k)** | **Native HF Rust BPE** | **64,053** | **1.440** | **1.289** | **1.12x** | **0.00%** | **100.0% (128/128)** | **11.93** |
| **DOAIxILPRL Nep-Tokenizer v2 (32k)** | **Native HF Rust BPE** | **32,057** | **1.584** | **1.490** | **1.06x** | **0.00%** | **100.0% (128/128)** | **10.84** |
| **Ampixa Yak v2.0 (64k)** | Byte-Level BPE | 64,512 | 1.306 | 1.483 | 0.88x | 0.00% | 100.0% (128/128) | 13.15 |
| **Ampixa Yak v1.0 (32k)** | Byte-Level BPE | 31,744 | 1.448 | 1.658 | 0.87x | 0.00% | 100.0% (128/128) | 11.86 |
| **Ampixa Yak v2.0 (128k)** | Byte-Level BPE | 130,048 | 1.205 | 1.275 | 0.95x | 0.00% | 100.0% (128/128) | 14.26 |
| **Himalaya AI v1 (64k)** | SentencePiece BPE | 64,000 | 1.375 | 1.329 | 1.03x | 2.60% | 86.3% (63/73) | 12.49 |
| **Sajal Regmi Arkios (65k)** | Byte-Level BPE | 65,510 | 1.556 | 1.201 | 1.30x | 0.00% | 100.0% (128/128) | 11.04 |
| **sidskarki Nepali-BPE (64k)** | SentencePiece BPE | 64,000 | 1.450 | 2.772 | 0.52x | 0.00% | 100.0% (128/128) | 11.84 |
| **NyayaLM BPE (64k)** | SentencePiece BPE | 63,985 | 1.450 | 1.188 | 1.22x | 4.76% | 89.0% (65/73) | 11.84 |
| **AI4Bharat IndicBERT v2 (250k)** | Fast Tokenizer | 250,000 | 1.461 | 1.221 | 1.20x | 0.00% | 100.0% (128/128) | 11.76 |
| **BigScience BLOOM (250k)** | Fast Tokenizer | 250,680 | 1.625 | 1.228 | 1.32x | 0.00% | 100.0% (128/128) | 10.57 |
| **BharatGen Param2-17B** | Fast Tokenizer | 128,000 | 1.797 | 1.597 | 1.13x | 0.00% | 100.0% (128/128) | 9.56 |
| **IRIIS RoBERTa Nepali (50k)** | Fast Tokenizer | 50,256 | 1.994 | 7.893 | 0.25x | 0.00% | 100.0% (128/128) | 8.62 |
| **Sarvam AI Sarvam-105B (262k)** | Rust Tokenizer | 262,144 | 2.080 | 1.221 | 1.70x | 8.26% | 93.2% (68/73) | 8.26 |
| **Supernova V5 (48k)** | Custom Prefix Trie | 47,908 | 4.468 | 1.799 | 2.48x | 43.82% | 100.0% (128/128) | 3.85 |
| **Google Gemma-2 (256k)** | SentencePiece Unigram | 256,000 | 2.869 | 1.161 | 2.47x | 23.46% | 100.0% (128/128) | 5.99 |
| **Meta LLaMA-3.1 (128k)** | Tiktoken / BPE | 128,000 | 3.498 | 1.235 | 2.83x | 0.00% | 100.0% (128/128) | 4.91 |

---

## 5. Running the Pipeline

```bash
# 1. Prepare multi-source training corpus (~52 MB)
python scripts/prepare_training_corpus.py

# 2. Train native Hugging Face BPE (32k & 64k)
python scripts/train_hf_bpe.py --vocab-sizes 32000 64000

# 3. Run full 17-model benchmark on held-out multi-domain data
python scripts/benchmark_all_models.py

# 4. Generate publication plots (figures/)
python scripts/generate_plots.py

# 5. Run automated unit test suite
pytest tests/test_v2_tokenizer.py

# 6. Publish 64k model to Hugging Face Hub (DOAIxILPRL-KU/nep-tokenizer)
python scripts/push_v2_to_hf.py
```

---

## 6. References & Citations

```bibtex
@misc{doai_ilprl_nep_tokenizer_2026,
  author = {Department of Artificial Intelligence (DOAI) and Information and Language Processing Research Lab (ILPRL)},
  title = {Nep-Tokenizer v2: An Open, Lossless, Mark-Aware Byte-Level BPE Tokenizer for Nepali and Multilingual LLMs},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub repository},
  howpublished = {\url{https://huggingface.co/DOAIxILPRL-KU/nep-tokenizer}},
  institution = {Kathmandu University}
}

@article{regmi2026vowel,
  title={Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility},
  author={Regmi, Sajal and Pudasaini, Siddhartha and Pun, Chetan Phakami},
  journal={arXiv preprint arXiv:2608.26449},
  year={2026},
  url={https://arxiv.org/abs/2608.26449}
}

@misc{ampixa2026yak,
  title={Yak: An Open, Lossless, Nepali-First Tokenizer},
  author={{Ampixa Labs}},
  year={2026},
  howpublished={\url{https://ampixa.com/blog/yak-an-open-lossless-nepali-first-tokenizer/}},
  note={Hugging Face: \url{https://huggingface.co/ampixa/yak-tokenizer}}
}

@misc{karki2025nepalitokenizer,
  title={Nepali Tokenizer Benchmark & Strategy Brief},
  author={Karki, Siddhartha},
  year={2025},
  howpublished={\url{https://github.com/sidskarkii/nepali-tokenizer}}
}

@article{himalaya2024nepali,
  title={Nepali Tokenizer Pipeline: Scalable Multi-Source SentencePiece Sampling for Devanagari},
  author={{Himalaya AI}},
  year={2024},
  note={Hugging Face: \url{https://huggingface.co/himalaya-ai/nepali-tokenizer-bpe-64k-5b}}
}

@article{chaudhary2026merge,
  title={Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining},
  author={Chaudhary, Aarjan},
  journal={Naamche Labs Research / arXiv preprint},
  year={2026}
}

@inproceedings{sennrich2016neural,
  title={Neural Machine Translation of Rare Words with Subword Units},
  author={Sennrich, Rico and Haddow, Barry and Birch, Alexandra},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL)},
  pages={1715--1725},
  year={2016}
}
```
