# Tokenization Experiments & Linguistic Integrity Audit

This directory contains validation tests and integrity audits for evaluating SentencePiece and Hugging Face tokenizers across Nepali, Indic, and multilingual text.

## Test Suite Overview

`test_tokenization_integrity.py` validates the following criteria:

### 1. Phonetic Vowels & Matras Completeness:
Audits complete independent vowel sets and dependent vowel signs (matras):
- **Independent Vowels & Vocalics**:
  `अ [a], आ [aa], इ [i], ई [ii], उ [u], ऊ [uu], ऋ [ri], ॠ [rr], ए [e], ऐ [ai], ओ [o], औ [au], अं [am], अः [a:]`
- **Dependent Vowel Signs (Matras)**:
  `ा [aa], ि [i], ी [ii], ु [u], ू [uu], ृ [r], ॄ [rr], ॅ [candra e], ॆ [short e], े [e], ै [ai], ॉ [candra o], ॊ [short o], ो [o], ौ [au]`
Audits dedicated single-token representation versus byte-fallback decomposition.

### 2. Special Compound Conjuncts (विशेष संयुक्त वर्ण):
Audits common Nepali and Devanagari ligatures and virama clusters:
- `क्ष [ksha = क् + ष]`
- `त्र [tra = त् + र]`
- `ज्ञ [gya = ज् + ञ]`
- `श्र [shra = श् + र]`
- `द्य [dya = द् + य]`
- `द्व [dwa = द् + व]`
- `द्म [dma = द् + म]`
- `त्त [tta = त् + त]`
- `द्ध [ddha = द् + ध]`
- `ङ्ग [nga = ङ् + ग]`
- `ङ्ख [nkha = ङ् + ख]`
Ensures conjuncts do not produce `<unk>` tokens and verifies whether they encode as single subwords or phonemically valid clusters.

### 3. Base Devanagari Consonants (33 Stops/Nasals/Semivowels):
Audits all 33 base consonants:
`क, ख, ग, घ, ङ, च, छ, ज, झ, ञ, ट, ठ, ड, ढ, ण, त, थ, द, ध, न, प, फ, ब, भ, म, य, र, ल, व, श, ष, स, ह`

### 4. Matra & Akshara Cohesion (Kathmandu Bug Check):
Verifies that token pieces do not begin with an isolated combining mark (`\p{M}` or `U+093E`–`U+094D`). This checks that dependent vowel signs remain attached to consonant stems rather than severing into bare prefix fragments.

### 5. Lossless Round-Trip Reconstruction:
Tests exact bitwise and normalized reconstruction across:
- Formal Nepali legal, constitutional, and academic sentences
- Colloquial Devanagari and Romanized social media text
- Code snippets and mathematical expressions

### 6. Special Tokens Preservation:
Verifies handling of `<pad>`, `<bos>`, `<eos>` without accidental decomposition.

### 7. Fertility Sanity Check:
Verifies that tokens/word fertility on standard Nepali remains within expected bounds (< 2.0 for 64k/128k models, well below the > 3.2 broken regex baseline).

---

## Running Tests

### 1. Test a local SentencePiece model:

```bash
python experiments/test_tokenization_integrity.py \
  --model data/tokenizers/nep-tokenizer-64000.model
```

### 2. Test a Hugging Face baseline model:

```bash
# Himalayan AI Baseline (64k)
python experiments/test_tokenization_integrity.py \
  --model himalaya-ai/nepali-tokenizer-bpe-64k-5b

# Pretrained Arkios (65k)
python experiments/test_tokenization_integrity.py \
  --model sajalregmi4/arkios-tokenizer

# Pretrained Yak v2.0 (64k & 128k)
python experiments/test_tokenization_integrity.py \
  --model ampixa/yak-tokenizer \
  --subfolder v2.0/64k
```
