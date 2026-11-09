#!/usr/bin/env python3
"""
scripts/benchmark_all_models.py

Standardized benchmark evaluation script across ALL Nepali, Indic, and frontier LLM tokenizers.
Affiliation: Department of Artificial Intelligence (DOAI) &
             Information and Language Processing Research Lab (ILPRL),
             Kathmandu University.

Evaluates on a consistent multi-domain benchmark:
- General News, Literature, Legal, Colloquial, Technical
- English Parity & Nepali Tax (F_Nepali / F_English)
- Severed Matra Anomaly Rate (The Kathmandu Bug check)
- Devanagari Character Coverage & Byte Fallback Rate (all 73 phonemes)
- Information Density: Bytes Per Token (BPT) & Tokens Per Char (TPC)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as RustTokenizer
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Devanagari character set for audit
VOWELS = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ए", "ऐ", "ओ", "औ", "अं", "अः"]
MATRAS = ["ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॅ", "ॆ", "े", "ै", "ॉ", "ॊ", "ो", "ौ"]
CONSONANTS = [
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह",
]
CONJUNCTS = ["क्ष", "त्र", "ज्ञ", "श्र", "द्य", "द्व", "द्म", "त्त", "द्ध", "ङ्ग", "ङ्ख"]

COMBINING_MARKS = {
    "\u0901", "\u0902", "\u0903", "\u093c", "\u093e", "\u093f", "\u0940",
    "\u0941", "\u0942", "\u0943", "\u0944", "\u0945", "\u0946", "\u0947",
    "\u0948", "\u0949", "\u094a", "\u094b", "\u094c", "\u094d", "\u0951",
    "\u0952", "\u0953", "\u0954", "\u0962", "\u0963",
}


# Custom Supernova Trie Loader
class SupernovaTrie:
    def encode(self, text: str) -> List[int]:
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            node = self.trie
            last_match = None
            last_j = i
            for j in range(i, n):
                ch = text[j]
                if ch in node:
                    node = node[ch]
                    if "__idx__" in node:
                        last_match = node["__idx__"]
                        last_j = j + 1
                else:
                    break
            if last_match is not None:
                tokens.append(last_match)
                i = last_j
            else:
                tokens.append(-1)
                i += 1
        return tokens


class SupernovaV5Tokenizer:
    def encode(self, text: str) -> List[int]:
        return self.trie_obj.encode(text)


def load_supernova():
    p_tok = hf_hub_download("Supernova11c/Supernova-NepaliFast-V5", "tokenizer_v5.pkl")
    # Patch module namespace for pickle
    sys.modules["__main__"].SupernovaTrie = SupernovaTrie
    sys.modules["__main__"].SupernovaV5Tokenizer = SupernovaV5Tokenizer
    with open(p_tok, "rb") as f:
        tok = pickle.load(f)
    return tok


class TokenizerWrapper:
    """Unified wrapper around SentencePiece, Hugging Face Tokenizers, and Fast Tokenizers."""

    def __init__(self, name: str, backend: Any, kind: str, vocab_size: int):
        self.name = name
        self.backend = backend
        self.kind = kind
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        if self.kind == "sp":
            return self.backend.encode(text, out_type=int)
        elif self.kind == "supernova":
            return self.backend.encode(text)
        elif self.kind == "rust_tokenizers":
            return self.backend.encode(text).ids
        try:
            return self.backend.encode(text, add_special_tokens=False)
        except Exception:
            return self.backend(text, add_special_tokens=False).input_ids

    def tokenize_pieces(self, text: str) -> List[str]:
        if self.kind == "sp":
            return self.backend.encode(text, out_type=str)
        elif self.kind == "supernova":
            ids = self.backend.encode(text)
            vocab = self.backend.trie_obj.vocab
            return [vocab.get(i, f"<{i}>") for i in ids]
        elif self.kind == "rust_tokenizers":
            return self.backend.encode(text).tokens
        try:
            return self.backend.tokenize(text)
        except Exception:
            ids = self.encode(text)
            return [self.backend.decode([i]) for i in ids]


def evaluate_dataset(
    tok: TokenizerWrapper,
    jsonl_path: Path,
) -> Dict[str, Any]:
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    domain_stats: Dict[str, Dict[str, int]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            domain = row.get("domain", "default")
            if not text.strip():
                continue

            words = len(text.split())
            chars = len(text)
            n_bytes = len(text.encode("utf-8"))
            tokens = len(tok.encode(text))

            total_tokens += tokens
            total_words += words
            total_chars += chars
            total_bytes += n_bytes

            if domain not in domain_stats:
                domain_stats[domain] = {"tokens": 0, "words": 0}
            domain_stats[domain]["tokens"] += tokens
            domain_stats[domain]["words"] += words

    fertility = total_tokens / max(total_words, 1)
    bpt = total_bytes / max(total_tokens, 1)
    tpc = total_tokens / max(total_chars, 1)

    domain_fertilities = {
        d: round(s["tokens"] / max(s["words"], 1), 3)
        for d, s in domain_stats.items()
    }

    return {
        "tokens": total_tokens,
        "words": total_words,
        "fertility": round(fertility, 3),
        "bpt": round(bpt, 2),
        "tpc": round(tpc, 3),
        "domain_fertilities": domain_fertilities,
    }


def evaluate_matra_cohesion(tok: TokenizerWrapper, samples: List[str]) -> Tuple[float, int, int]:
    total_pieces = 0
    severed_pieces = 0

    for s in samples:
        pieces = tok.tokenize_pieces(s)
        for p in pieces:
            clean_p = p.lstrip(" \u2581\u0120")
            if clean_p and clean_p[0] in COMBINING_MARKS:
                severed_pieces += 1
            total_pieces += 1

    rate = (severed_pieces / max(total_pieces, 1)) * 100.0
    return round(rate, 2), severed_pieces, total_pieces


def evaluate_devanagari_coverage(tok: TokenizerWrapper) -> Dict[str, Any]:
    all_chars = VOWELS + MATRAS + CONSONANTS + [c[0] for c in CONJUNCTS]
    total = len(all_chars)
    fallback_chars = []

    for ch in all_chars:
        pieces = tok.tokenize_pieces(ch)
        is_fallback = any(
            p.startswith("<0x") or p in ("<unk>", "[UNK]", "⁇", "\ufffd", "<-1>")
            for p in pieces
        )
        if is_fallback or len(pieces) > 2:
            fallback_chars.append(ch)

    covered = total - len(fallback_chars)
    coverage_pct = round((covered / total) * 100.0, 1)
    return {
        "total": total,
        "covered": covered,
        "coverage_pct": coverage_pct,
        "fallbacks": fallback_chars,
    }


def main():
    eval_ne = REPO_ROOT / "data" / "eval" / "eval_ne.jsonl"
    eval_en = REPO_ROOT / "data" / "eval" / "eval_en.jsonl"

    if not eval_ne.exists() or not eval_en.exists():
        logger.error("Evaluation files not found in data/eval/")
        sys.exit(1)

    cohesion_samples = [
        "काठमाडौं उपत्यकामा वायु प्रदूषण नियन्त्रणका लागि विद्युतीय सार्वजनिक यातायात प्रणाली विस्तार गर्नुपर्छ ।",
        "नेपालको संविधान २०७२ ले संघीय लोकतान्त्रिक गणतन्त्रलाई संस्थागत गरेको छ ।",
        "ऋषिमुनिहरूको तपोभूमि नेपालमा ज्ञान, विज्ञान र अध्यात्मको अनुपम संगम छ ।",
        "शिक्षा, विज्ञान तथा प्रविधि मन्त्रालयले नयाँ पाठ्यक्रम स्वीकृत गरेको छ ।",
        "प्रकृतिको मनोरम छटा, हरियाली डाँडाकाँडा र नदीनालाहरूले मन आनन्दित तुल्याउँछन् ।",
    ]

    # Full Model Evaluation Registry
    model_configs: List[Dict[str, Any]] = [
        # 1. DOAI x ILPRL Nep-Tokenizer v2 (Our hero models)
        {"name": "Nep-Tokenizer v2 (64k)", "type": "hf_dir", "path": str(REPO_ROOT / "hf_push" / "64k")},
        {"name": "Nep-Tokenizer v2 (32k)", "type": "hf_dir", "path": str(REPO_ROOT / "hf_push" / "32k")},

        # 2. Yak Tokenizer
        {"name": "Yak v2.0 (64k)", "type": "hf_hub", "repo": "ampixa/yak-tokenizer", "subfolder": "v2.0/64k"},
        {"name": "Yak v1.0 (32k)", "type": "hf_hub", "repo": "ampixa/yak-tokenizer", "subfolder": "v1.0/32k"},
        {"name": "Yak v2.0 (128k)", "type": "hf_hub", "repo": "ampixa/yak-tokenizer", "subfolder": "v2.0/128k"},

        # 3. Peer Nepali Tokenizers
        {"name": "Arkios (65k)", "type": "hf_hub", "repo": "sajalregmi4/arkios-1b-base"},
        {"name": "NyayaLM-BPE (64k)", "type": "hf_hub", "repo": "chhatramani/nyayalm-SentencePiece_BPE-64k"},
        {"name": "sidskarki-BPE (64k)", "type": "sp_hub", "repo": "sidskarki/nepali-bpe-tokenizer", "file": "nepali_bpe_64k.model"},
        {"name": "Supernova V5 (48k)", "type": "supernova"},
        {"name": "Himalaya AI v1 (64k)", "type": "sp_hub", "repo": "himalaya-ai/nepali-tokenizer-bpe-64k-5b", "file": "himalaya-ai_nepali-tokenizer-corpus-64000.model"},

        # 4. Regional Indic Foundations
        {"name": "IndicBERT v2 (250k)", "type": "hf_hub", "repo": "ai4bharat/IndicBERTv2-MLM-only"},
        {"name": "BharatGen Param2-17B", "type": "hf_hub", "repo": "bharatgenai/Param2-17B-A2.4B-Thinking"},
        {"name": "Sarvam-105b (262k)", "type": "rust_hub", "repo": "sarvamai/sarvam-105b", "file": "tokenizer.json"},
        {"name": "RoBERTa Nepali (50k)", "type": "hf_hub", "repo": "IRIIS-RESEARCH/RoBERTa_Nepali_125M"},
        {"name": "BLOOM (250k)", "type": "hf_hub", "repo": "bigscience/bloom"},

        # 5. Global Frontier Multilingual Baselines
        {"name": "LLaMA-3.1 (128k)", "type": "hf_hub", "repo": "unsloth/Meta-Llama-3.1-8B"},
        {"name": "Gemma-2 (256k)", "type": "hf_hub", "repo": "unsloth/gemma-2-9b"},
    ]

    loaded_tokenizers: List[TokenizerWrapper] = []

    for cfg in model_configs:
        name = cfg["name"]
        mtype = cfg["type"]
        logger.info(f"Loading {name} ({mtype})...")
        try:
            if mtype == "hf_dir":
                tok = AutoTokenizer.from_pretrained(cfg["path"])
                loaded_tokenizers.append(TokenizerWrapper(name, tok, "hf", tok.vocab_size))
            elif mtype == "hf_hub":
                sub = cfg.get("subfolder")
                tok = AutoTokenizer.from_pretrained(cfg["repo"], subfolder=sub, trust_remote_code=True) if sub else AutoTokenizer.from_pretrained(cfg["repo"], trust_remote_code=True)
                loaded_tokenizers.append(TokenizerWrapper(name, tok, "hf", tok.vocab_size))
            elif mtype == "sp_hub":
                p = hf_hub_download(cfg["repo"], cfg["file"])
                sp = spm.SentencePieceProcessor()
                sp.load(p)
                loaded_tokenizers.append(TokenizerWrapper(name, sp, "sp", sp.get_piece_size()))
            elif mtype == "rust_hub":
                p = hf_hub_download(cfg["repo"], cfg["file"])
                rtok = RustTokenizer.from_file(p)
                loaded_tokenizers.append(TokenizerWrapper(name, rtok, "rust_tokenizers", rtok.get_vocab_size()))
            elif mtype == "supernova":
                s_tok = load_supernova()
                loaded_tokenizers.append(TokenizerWrapper(name, s_tok, "supernova", 47908))
            logger.info(f"  -> Successfully loaded {name}")
        except Exception as e:
            logger.error(f"  -> FAILED to load {name}: {e}")

    # Benchmark Execution
    results = []
    print("\n" + "=" * 110)
    print(f"{'Model':<26} | {'Vocab':<8} | {'NE Fert':<8} | {'EN Fert':<8} | {'Tax':<6} | {'Matra Err%':<10} | {'Deva Cov%':<10} | {'BPT':<5}")
    print("=" * 110)

    for tok in loaded_tokenizers:
        ne_metrics = evaluate_dataset(tok, eval_ne)
        en_metrics = evaluate_dataset(tok, eval_en)

        tax = round(ne_metrics["fertility"] / max(en_metrics["fertility"], 0.01), 2)
        err_rate, _, _ = evaluate_matra_cohesion(tok, cohesion_samples)
        coverage = evaluate_devanagari_coverage(tok)

        res = {
            "name": tok.name,
            "vocab_size": tok.vocab_size,
            "ne_fertility": ne_metrics["fertility"],
            "en_fertility": en_metrics["fertility"],
            "nepali_tax": tax,
            "matra_severed_rate": err_rate,
            "matra_cohesion_rate": round(100.0 - err_rate, 2),
            "devanagari_coverage_pct": coverage["coverage_pct"],
            "fallbacks_count": len(coverage["fallbacks"]),
            "bpt": ne_metrics["bpt"],
            "tpc": ne_metrics["tpc"],
            "domains": ne_metrics["domain_fertilities"],
        }
        results.append(res)

        print(
            f"{tok.name:<26} | {tok.vocab_size:<8} | {ne_metrics['fertility']:<8.3f} | "
            f"{en_metrics['fertility']:<8.3f} | {tax:<6.2f}x | {err_rate:<10.2f} | "
            f"{coverage['coverage_pct']:<10.1f} | {ne_metrics['bpt']:<5.2f}"
        )

    print("=" * 110)

    # Save to disk
    out_file = REPO_ROOT / "data" / "benchmark_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
