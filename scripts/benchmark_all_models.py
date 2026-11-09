#!/usr/bin/env python3
"""Standardized benchmark evaluation script across ALL Nepali, Indic, and frontier LLM tokenizers.

Evaluates on a consistent multi-domain benchmark:
- General News, Literature, Legal, Colloquial, Technical
- English Parity (FLORES-style parallel reference)
- Severed Matra Anomaly Rate (The Kathmandu Bug check)
- Devanagari Character Coverage & Byte Fallback Rate (all 73 phonemes)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        try:
            return self.backend.encode(text, add_special_tokens=False)
        except Exception:
            return self.backend(text, add_special_tokens=False).input_ids

    def tokenize_pieces(self, text: str) -> List[str]:
        if self.kind == "sp":
            return self.backend.encode(text, out_type=str)
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
        # Check if pieces contain raw byte fallback e.g. <0xE0> or [UNK] or 
        is_fallback = any(
            p.startswith("<0x") or p in ("<unk>", "[UNK]", "⁇", "\ufffd")
            for p in pieces
        )
        if is_fallback:
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
    import sentencepiece as spm
    from transformers import AutoTokenizer

    eval_ne = REPO_ROOT / "data" / "eval" / "eval_ne.jsonl"
    eval_en = REPO_ROOT / "data" / "eval" / "eval_en.jsonl"

    if not eval_ne.exists() or not eval_en.exists():
        logger.error("Evaluation files not found in data/eval/")
        sys.exit(1)

    # Matra stress test samples
    cohesion_samples = [
        "काठमाडौं उपत्यकामा वायु प्रदूषण नियन्त्रणका लागि विद्युतीय सार्वजनिक यातायात प्रणाली विस्तार गर्नुपर्छ ।",
        "नेपालको संविधान २०७२ ले संघीय लोकतान्त्रिक गणतन्त्रलाई संस्थागत गरेको छ ।",
        "ऋषिमुनिहरूको तपोभूमि नेपालमा ज्ञान, विज्ञान र अध्यात्मको अनुपम संगम छ ।",
        "शिक्षा, विज्ञान तथा प्रविधि मन्त्रालयले नयाँ पाठ्यक्रम स्वीकृत गरेको छ ।",
        "प्रकृतिको मनोरम छटा, हरियाली डाँडाकाँडा र नदीनालाहरूले मन आनन्दित तुल्याउँछन् ।",
    ]

    model_registry: List[Tuple[str, str, str, Dict[str, Any]]] = [
        # 1. DOAI x ILPRL Nep-Tokenizer v2
        ("Nep-Tokenizer v2 (128k)", "sp", str(REPO_ROOT / "hf_push" / "128k" / "tokenizer.model"), {}),
        ("Nep-Tokenizer v2 (100k)", "sp", str(REPO_ROOT / "hf_push" / "100k" / "tokenizer.model"), {}),
        ("Nep-Tokenizer v2 (64k)", "sp", str(REPO_ROOT / "hf_push" / "64k" / "tokenizer.model"), {}),

        # 2. Himalayan AI v1 Baseline
        ("Himalaya AI v1 (128k)", "sp_hf", "himalaya-ai/nepali-tokenizer-bpe-128k-5b", {}),
        ("Himalaya AI v1 (100k)", "sp_hf", "himalaya-ai/nepali-tokenizer-bpe-100k-5b", {}),
        ("Himalaya AI v1 (64k)", "sp_hf", "himalaya-ai/nepali-tokenizer-bpe-64k-5b", {}),

        # 3. Yak v2.0
        ("Yak v2.0 (128k)", "hf", "ampixa/yak-tokenizer", {"subfolder": "v2.0/128k"}),
        ("Yak v2.0 (64k)", "hf", "ampixa/yak-tokenizer", {"subfolder": "v2.0/64k"}),

        # 4. Peer Nepali & Indic
        ("Arkios (65k)", "hf", "sajalregmi4/arkios-1b-base", {}),
        ("Qwen3-Nepali (166k)", "hf", "sidskarki/qwen3-nepali-tokenizer", {}),
        ("NepaliBPE (50k)", "hf", "Aananda-giri/NepaliBPE", {}),
        ("NepaliBERT (50k)", "hf", "Rajan/NepaliBERT", {}),
        ("BERT-Nepali (25k)", "hf", "raygx/BERT_Nepali_Tokenizer", {}),
        ("BharatGen Param-1 (256k)", "hf", "bharatgenai/Param-1-5B", {}),
        ("Sarvam-1 (68k)", "hf", "sarvamai/sarvam-1", {}),

        # 5. Global Frontier Baselines
        ("LLaMA-3.1 (128k)", "hf", "unsloth/Meta-Llama-3.1-8B", {}),
        ("Gemma-2 (256k)", "hf", "unsloth/gemma-2-9b", {}),
    ]

    results = []

    for name, kind, path_or_repo, kwargs in model_registry:
        logger.info("Evaluating: %s ...", name)
        try:
            if kind == "sp":
                if not Path(path_or_repo).exists():
                    logger.warning("File not found: %s", path_or_repo)
                    continue
                sp = spm.SentencePieceProcessor()
                sp.load(path_or_repo)
                tok = TokenizerWrapper(name, sp, "sp", sp.vocab_size())
            elif kind == "sp_hf":
                from huggingface_hub import hf_hub_download, list_repo_files
                files = list_repo_files(path_or_repo)
                m_file = [f for f in files if f.endswith(".model")][0]
                local_m = hf_hub_download(path_or_repo, m_file)
                sp = spm.SentencePieceProcessor()
                sp.load(local_m)
                tok = TokenizerWrapper(name, sp, "sp", sp.vocab_size())
            else:
                hf_tok = AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=True, **kwargs)
                tok = TokenizerWrapper(name, hf_tok, "hf", len(hf_tok))

            ne_metrics = evaluate_dataset(tok, eval_ne)
            en_metrics = evaluate_dataset(tok, eval_en)

            parity = round(ne_metrics["fertility"] / max(en_metrics["fertility"], 0.001), 2)
            severed_rate, sev_cnt, tot_cnt = evaluate_matra_cohesion(tok, cohesion_samples)
            cov_metrics = evaluate_devanagari_coverage(tok)

            res = {
                "name": name,
                "vocab_size": tok.vocab_size,
                "nepali_fertility": ne_metrics["fertility"],
                "bpt": ne_metrics["bpt"],
                "tpc": ne_metrics["tpc"],
                "english_fertility": en_metrics["fertility"],
                "english_parity": parity,
                "severed_matra_rate": severed_rate,
                "devanagari_coverage_pct": cov_metrics["coverage_pct"],
                "fallbacks_count": len(cov_metrics["fallbacks"]),
                "fallbacks": cov_metrics["fallbacks"],
                "domain_fertilities": ne_metrics["domain_fertilities"],
            }
            results.append(res)
            logger.info("  -> Fert: %s | BPT: %s | Parity: %sx | Severed: %s%% | Cov: %s%%",
                        res["nepali_fertility"], res["bpt"], res["english_parity"],
                        res["severed_matra_rate"], res["devanagari_coverage_pct"])

        except Exception as e:
            logger.error("Error evaluating %s: %s", name, e)

    out_file = REPO_ROOT / "data" / "benchmark_evaluation_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved evaluation results to %s", out_file)


if __name__ == "__main__":
    main()

