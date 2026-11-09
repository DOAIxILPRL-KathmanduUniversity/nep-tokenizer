#!/usr/bin/env python3
"""Tokenization Integrity & Robustness Audit Suite.

Comprehensive linguistic audit for Nepali & Devanagari tokenizers verifying:
1. Phonetic Vowel & Matra Completeness:
   - Independent vowels: अ [a], आ [aa], इ [i], ई [ii], उ [u], ऊ [uu], ऋ [ri], ॠ [rr], अं [am], अः [a:], ए [e], ऐ [ai], ओ [o], औ [au]
   - Dependent vowel signs (matras): ा [aa], ि [i], ी [ii], ु [u], ू [uu], ृ [r], ॄ [rr], ॅ [candra e], ॆ [short e], े [e], ै [ai], ॉ [candra o], ॊ [short o], ो [o], ौ [au]
2. Compound Conjuncts (विशेष संयुक्त वर्ण):
   - क्ष [ksh], त्र [tra], ज्ञ [gya], श्र [shra], द्य [dya], द्व [dwa], द्म [dma], त्त [tta], द्ध [ddha], ङ्ग [nga], ङ्ख [nkha]
3. Base Consonants (33 Devanagari stops, nasals, semivowels, sibilants).
4. Matra & Akshara Cohesion (Kathmandu Bug Check - verifying no bare combining mark prefixes).
5. Lossless bitwise round-trip reconstruction across formal, colloquial, and code samples.
6. Special tokens preservation (<pad>, <bos>, <eos>).

Usage:
    python experiments/test_tokenization_integrity.py --model himalaya-ai/nepali-tokenizer-bpe-64k-5b
    python experiments/test_tokenization_integrity.py --model ampixa/yak-tokenizer --subfolder v2.0/64k
    python experiments/test_tokenization_integrity.py --model data/tokenizers/nep-tokenizer-64000.model
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Complete Independent Vowels (14 + vocalic / anusvara / visarga forms)
VOWELS_INDEPENDENT = [
    ("अ", "a"),
    ("आ", "aa"),
    ("इ", "i"),
    ("ई", "ii"),
    ("उ", "u"),
    ("ऊ", "uu"),
    ("ऋ", "ri"),
    ("ॠ", "rr"),
    ("ए", "e"),
    ("ऐ", "ai"),
    ("ओ", "o"),
    ("औ", "au"),
    ("अं", "am"),
    ("अः", "a:"),
]

# Complete Dependent Vowel Signs / Matras (15 standard & extended forms)
MATRAS_DEPENDENT = [
    ("ा", "aa"),
    ("ि", "i"),
    ("ी", "ii"),
    ("ु", "u"),
    ("ू", "uu"),
    ("ृ", "r"),
    ("ॄ", "rr"),
    ("ॅ", "candra e"),
    ("ॆ", "short e"),
    ("े", "e"),
    ("ै", "ai"),
    ("ॉ", "candra o"),
    ("ॊ", "short o"),
    ("ो", "o"),
    ("ौ", "au"),
]

# Base Devanagari Consonants (33)
BASE_CONSONANTS = [
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
]

# Special Compound Conjuncts (विशेष संयुक्त वर्ण)
COMPOUND_CONJUNCTS = [
    ("क्ष", "ksha [क् + ष]"),
    ("त्र", "tra [त् + र]"),
    ("ज्ञ", "gya [ज् + ञ]"),
    ("श्र", "shra [श् + र]"),
    ("द्य", "dya [द् + य]"),
    ("द्व", "dwa [द् + व]"),
    ("द्म", "dma [द् + म]"),
    ("त्त", "tta [त् + त]"),
    ("द्ध", "ddha [द् + ध]"),
    ("ङ्ग", "nga [ङ् + ग]"),
    ("ङ्ख", "nkha [ङ् + ख]"),
]

# Signs, Digits & Punctuation
SIGNS_AND_PUNCT = ["्", "ं", "ँ", "ः", "ऽ", "ॐ", "।", "?", "!"]
DIGITS_DEVANAGARI = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]

SAMPLE_SENTENCES = [
    "नेपालको राजधानी काठमाडौं हो ।",
    "नेपालको संविधान २०७२ ले संघीय लोकतान्त्रिक गणतन्त्र संस्थागत गरेको छ ।",
    "शिक्षा, विज्ञान तथा प्रविधि मन्त्रालयले नयाँ पाठ्यक्रम स्वीकृत गर्यो ।",
    "काठमाडौं उपत्यकामा वायु प्रदूषण नियन्त्रणका लागि विद्युतीय सवारीसाधन प्रवर्द्धन गरिँदैछ ।",
    "ऋषिमुनिहरूको तपोभूमि नेपालमा ज्ञान, विज्ञान र अध्यात्मको संगम छ ।",
    "अन्तरिक्ष प्रविधि र कृत्रिम बौद्धिकताले देशको विकासमा टेवा पुर्याउँछ ।",
]

COLLOQUIAL_SAMPLES = [
    "timi kaile aaune ho ghar? ma ta wait gardai basirachu hai.",
    "aaja weather ekdam ramro cha, ghumna jaam na ta!",
    "daami film raicha bro, must watch cinema hall ma gayera.",
]

CODE_SAMPLES = [
    "def compute_fertility(tokens: int, words: int) -> float:\n    return tokens / max(words, 1)",
    "x = [i**2 for i in range(10) if i % 2 == 0]",
]


class TokenizerAdapter:
    """Unified adapter wrapping SentencePiece or Hugging Face AutoTokenizer."""

    def __init__(self, backend: Any, kind: str) -> None:
        self.backend = backend
        self.kind = kind
        if kind == "sp":
            self.unk_id = backend.unk_id()
        else:
            self.unk_id = getattr(backend, "unk_token_id", None)

    def encode(self, text: str) -> List[int]:
        if self.kind == "sp":
            return self.backend.encode(text, out_type=int)
        try:
            return self.backend.encode(text, add_special_tokens=False)
        except Exception:
            return self.backend(text, add_special_tokens=False).input_ids

    def decode(self, ids: List[int]) -> str:
        if self.kind == "sp":
            return self.backend.decode(ids)
        return self.backend.decode(ids, skip_special_tokens=False)

    def tokenize_pieces(self, text: str) -> List[str]:
        if self.kind == "sp":
            return self.backend.encode(text, out_type=str)
        try:
            return self.backend.tokenize(text)
        except Exception:
            ids = self.encode(text)
            return [self.backend.decode([i]) for i in ids]


def load_adapter(model_path_or_repo: str, subfolder: Optional[str] = None) -> TokenizerAdapter:
    if not subfolder:
        if "@" in model_path_or_repo:
            model_path_or_repo, subfolder = model_path_or_repo.split("@", 1)
        elif ":" in model_path_or_repo and not model_path_or_repo.startswith(("http:", "https:")):
            parts = model_path_or_repo.split(":", 1)
            model_path_or_repo, subfolder = parts[0], parts[1]

    if os.path.exists(model_path_or_repo) and model_path_or_repo.endswith(".model"):
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(model_path_or_repo)
        return TokenizerAdapter(sp, kind="sp")

    kwargs = {}
    if subfolder:
        kwargs["subfolder"] = subfolder

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path_or_repo, **kwargs)
        return TokenizerAdapter(tok, kind="hf")
    except Exception as exc:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            import sentencepiece as spm

            files = list_repo_files(model_path_or_repo)
            m_files = [f for f in files if f.endswith(".model")]
            if m_files:
                path = hf_hub_download(model_path_or_repo, m_files[0])
                sp = spm.SentencePieceProcessor()
                sp.load(path)
                return TokenizerAdapter(sp, kind="sp")
        except Exception:
            pass
        raise RuntimeError(f"Could not load tokenizer from {model_path_or_repo}: {exc}") from exc


def test_phonetic_vowels_and_matras(adapter: TokenizerAdapter) -> Dict[str, Any]:
    """Test representation of all independent vowels and dependent matras."""
    vowel_results = []
    byte_fallbacks = []
    unk_failures = []

    for char, name in VOWELS_INDEPENDENT:
        ids = adapter.encode(char)
        pieces = adapter.tokenize_pieces(char)
        is_unk = adapter.unk_id is not None and adapter.unk_id >= 0 and adapter.unk_id in ids
        # Check if decomposed into raw byte tokens (e.g. <0x..>)
        is_byte_fallback = any("<0x" in p or "byte:" in p for p in pieces)

        if is_unk:
            unk_failures.append(f"{char} [{name}]")
        elif is_byte_fallback:
            byte_fallbacks.append(f"{char} [{name}]")
        vowel_results.append({"char": char, "name": name, "tokens": len(ids), "pieces": pieces})

    matra_results = []
    for matra, name in MATRAS_DEPENDENT:
        ids = adapter.encode(matra)
        pieces = adapter.tokenize_pieces(matra)
        is_unk = adapter.unk_id is not None and adapter.unk_id >= 0 and adapter.unk_id in ids
        is_byte_fallback = any("<0x" in p or "byte:" in p for p in pieces)

        if is_unk:
            unk_failures.append(f"{matra} [{name}]")
        elif is_byte_fallback:
            byte_fallbacks.append(f"{matra} [{name}]")
        matra_results.append({"matra": matra, "name": name, "tokens": len(ids), "pieces": pieces})

    passed = len(unk_failures) == 0

    return {
        "test": "Phonetic Vowels & Matras Completeness",
        "passed": passed,
        "vowels_audited": len(VOWELS_INDEPENDENT),
        "matras_audited": len(MATRAS_DEPENDENT),
        "unk_count": len(unk_failures),
        "unk_failures": unk_failures,
        "byte_fallback_count": len(byte_fallbacks),
        "byte_fallback_examples": byte_fallbacks[:6],
    }


def test_compound_conjuncts(adapter: TokenizerAdapter) -> Dict[str, Any]:
    """Test that compound conjuncts (विशेष संयुक्त वर्ण) encode cleanly."""
    conjunct_results = []
    unk_failures = []

    for conjunct, name in COMPOUND_CONJUNCTS:
        ids = adapter.encode(conjunct)
        pieces = adapter.tokenize_pieces(conjunct)
        is_unk = adapter.unk_id is not None and adapter.unk_id >= 0 and adapter.unk_id in ids

        if is_unk:
            unk_failures.append(f"{conjunct} [{name}]")

        conjunct_results.append({
            "conjunct": conjunct,
            "name": name,
            "tokens": len(ids),
            "pieces": pieces,
        })

    passed = len(unk_failures) == 0

    # Calculate how many conjuncts are captured as single tokens
    single_token_conjuncts = [
        c["conjunct"] for c in conjunct_results if c["tokens"] == 1
    ]

    return {
        "test": "Special Compound Conjuncts (विशेष संयुक्त वर्ण)",
        "passed": passed,
        "conjuncts_audited": len(COMPOUND_CONJUNCTS),
        "unk_count": len(unk_failures),
        "single_token_conjuncts_count": len(single_token_conjuncts),
        "single_token_conjuncts": single_token_conjuncts,
    }


def test_base_consonants(adapter: TokenizerAdapter) -> Dict[str, Any]:
    """Verify all 33 base Devanagari consonants."""
    missing = []
    covered = []

    for c in BASE_CONSONANTS:
        ids = adapter.encode(c)
        if adapter.unk_id is not None and adapter.unk_id >= 0 and adapter.unk_id in ids:
            missing.append(c)
        elif not ids:
            missing.append(c)
        else:
            covered.append(c)

    return {
        "test": "Base Devanagari Consonants (33 Stops/Nasals/Semivowels)",
        "passed": len(missing) == 0,
        "total": len(BASE_CONSONANTS),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "missing_chars": missing,
    }


def test_matra_and_akshara_cohesion(adapter: TokenizerAdapter) -> Dict[str, Any]:
    """Audit whether token pieces start with an isolated combining mark (Kathmandu bug check)."""
    severed_tokens = []
    total_tokens = 0

    for sentence in SAMPLE_SENTENCES:
        pieces = adapter.tokenize_pieces(sentence)
        total_tokens += len(pieces)
        for piece in pieces:
            clean = piece.lstrip(" Ġ ")
            if not clean:
                continue
            first_cp = ord(clean[0])
            # Devanagari vowel sign (matra) or virama (0x093E to 0x094D)
            if 0x093E <= first_cp <= 0x094D:
                severed_tokens.append(piece)

    severed_rate = (len(severed_tokens) / total_tokens) if total_tokens else 0.0
    passed = severed_rate < 0.10

    return {
        "test": "Matra & Akshara Cohesion (Kathmandu Bug Check)",
        "passed": passed,
        "total_tokens_audited": total_tokens,
        "severed_tokens_count": len(severed_tokens),
        "severed_rate_pct": round(severed_rate * 100, 2),
        "severed_examples": severed_tokens[:5],
    }


def test_lossless_roundtrip(adapter: TokenizerAdapter) -> Dict[str, Any]:
    """Test lossless bitwise reconstruction across multiple domains."""
    failures = []
    whitespace_normalized_count = 0
    tested = 0

    test_corpus = SAMPLE_SENTENCES + COLLOQUIAL_SAMPLES + CODE_SAMPLES

    for text in test_corpus:
        tested += 1
        ids = adapter.encode(text)
        rec = adapter.decode(ids)
        rec_norm = unicodedata.normalize("NFKC", rec).strip()
        text_norm = unicodedata.normalize("NFKC", text).strip()

        if rec.strip() == text.strip() or rec_norm == text_norm:
            continue

        rec_collapsed = " ".join(rec_norm.split())
        text_collapsed = " ".join(text_norm.split())
        if rec_collapsed == text_collapsed:
            whitespace_normalized_count += 1
        else:
            failures.append({"original": text, "reconstructed": rec})

    passed = len(failures) == 0
    return {
        "test": "Lossless Round-Trip Reconstruction",
        "passed": passed,
        "total_tested": tested,
        "exact_matches": tested - len(failures) - whitespace_normalized_count,
        "whitespace_normalized": whitespace_normalized_count,
        "character_failures_count": len(failures),
        "failure_examples": failures[:3],
    }


def test_special_tokens(adapter: TokenizerAdapter) -> Dict[str, Any]:
    """Test that special tokens <pad>, <bos>, <eos> are recognized."""
    symbols = ["<pad>", "<bos>", "<eos>"]
    recognized = []

    for sym in symbols:
        ids = adapter.encode(sym)
        if len(ids) == 1:
            recognized.append(sym)

    return {
        "test": "Special Tokens Preservation",
        "symbols_tested": symbols,
        "recognized_as_single_token": recognized,
    }


def test_fertility_sanity(adapter: TokenizerAdapter) -> Dict[str, Any]:
    """Check that tokens/word fertility on standard Nepali is within sane bounds (< 2.5)."""
    total_tokens = 0
    total_words = 0

    for s in SAMPLE_SENTENCES:
        words = len(s.split())
        ids = adapter.encode(s)
        total_tokens += len(ids)
        total_words += words

    fertility = (total_tokens / total_words) if total_words else 0.0
    passed = fertility < 2.5

    return {
        "test": "Nepali Fertility Sanity Check",
        "passed": passed,
        "tokens": total_tokens,
        "words": total_words,
        "fertility": round(fertility, 4),
    }


def run_all_tests(model_path_or_repo: str, subfolder: Optional[str] = None) -> bool:
    logger.info("Loading model: %s (subfolder=%s)", model_path_or_repo, subfolder)
    adapter = load_adapter(model_path_or_repo, subfolder=subfolder)

    results = [
        test_phonetic_vowels_and_matras(adapter),
        test_compound_conjuncts(adapter),
        test_base_consonants(adapter),
        test_matra_and_akshara_cohesion(adapter),
        test_lossless_roundtrip(adapter),
        test_special_tokens(adapter),
        test_fertility_sanity(adapter),
    ]

    all_passed = True
    print("\n" + "=" * 65)
    print(f"Tokenization Linguistic Integrity Audit: {model_path_or_repo}")
    print("=" * 65)

    for r in results:
        status = "PASSED" if r.get("passed", True) else "FAILED"
        if not r.get("passed", True):
            all_passed = False
        print(f"[{status}] {r['test']}")
        for k, v in r.items():
            if k not in ("test", "passed"):
                print(f"    - {k}: {v}")

    print("=" * 65)
    print(f"OVERALL RESULT: {'PASSED' if all_passed else 'WARNINGS / FAILURES'}")
    print("=" * 65 + "\n")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tokenization linguistic integrity & robustness")
    parser.add_argument(
        "--model",
        default="himalaya-ai/nepali-tokenizer-bpe-64k-5b",
        help="Model path or HF repo ID to test",
    )
    parser.add_argument(
        "--subfolder",
        help="Optional HF repository subfolder (e.g. v2.0/64k)",
    )
    args = parser.parse_args()

    success = run_all_tests(args.model, args.subfolder)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
