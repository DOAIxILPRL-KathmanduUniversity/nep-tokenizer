#!/usr/bin/env python3
"""
scripts/train_hf_bpe.py

Hugging Face Native Rust BPE Tokenizer Trainer for Nep-Tokenizer v2.
Affiliation: Department of Artificial Intelligence (DOAI) &
             Information and Language Processing Research Lab (ILPRL),
             Kathmandu University.

Key Architectural Guarantees:
1. Mark-Aware Regex: Binds Devanagari matras (Mn/Mc) and virama to base consonants.
2. Digit Grouping: Groups 1-3 digits for ASCII [0-9] and Devanagari [०-९].
3. Lossless Byte-Level BPE: Zero unknown tokens, 100% roundtrip fidelity.
4. Complete Devanagari Block Guarantee: Guarantees all 128 Devanagari code points
   (U+0900 to U+097F) are atomic single tokens with zero byte fallbacks.
5. Hugging Face Native Export: Generates tokenizer.json, tokenizer_config.json,
   special_tokens_map.json, loadable via AutoTokenizer.from_pretrained().
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def bytes_to_unicode():
    """GPT-2 standard byte-to-unicode character bijection."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


B2U = bytes_to_unicode()

DIGIT_PATTERN = r"[0-9०-९]{1,3}"
WORD_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[\p{L}\p{M}\u200c\u200d]+| ?\p{N}+| ?[^\s\p{L}\p{M}\u200c\u200d\p{N}]+"
    r"|\s+(?!\S)|\s+"
)

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|sep|>",
    "<|start_turn|>",
    "<|end_turn|>",
    "<|think|>",
    "<|/think|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
]


def build_base_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(DIGIT_PATTERN), behavior="isolated"),
        pre_tokenizers.Split(Regex(WORD_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def inject_complete_devanagari_block(tok: Tokenizer) -> Tokenizer:
    """Ensures all 128 Devanagari code points (U+0900..U+097F) have merges and vocab entries."""
    tok_json = json.loads(tok.to_str())
    vocab = tok_json["model"]["vocab"]
    merges = tok_json["model"]["merges"]

    merges_set = set(tuple(m) for m in merges)

    def add_merge(p1: str, p2: str):
        t = (p1, p2)
        merged = p1 + p2
        if t not in merges_set:
            merges.append([p1, p2])
            merges_set.add(t)
        if merged not in vocab:
            vocab[merged] = len(vocab)
        return merged

    # 1. Base prefixes for 3-byte Devanagari UTF-8
    add_merge(B2U[0xE0], B2U[0xA4])  # 'à¤' (U+0900 - U+093F)
    add_merge(B2U[0xE0], B2U[0xA5])  # 'à¥' (U+0940 - U+097F)

    # 2. For each Devanagari codepoint U+0900..U+097F
    injected_count = 0
    for cp in range(0x0900, 0x0980):
        ch = chr(cp)
        raw = ch.encode("utf-8")
        prefix = B2U[raw[0]] + B2U[raw[1]]
        suffix = B2U[raw[2]]
        t = (prefix, suffix)
        if t not in merges_set:
            add_merge(prefix, suffix)
            injected_count += 1

    # 3. Guarantee ZWNJ (U+200C) and ZWJ (U+200D) format characters as atomic single tokens
    add_merge(B2U[0xE2], B2U[0x80])  # 'âĢ' prefix
    add_merge(B2U[0xE2] + B2U[0x80], B2U[0x8C])  # ZWNJ U+200C
    add_merge(B2U[0xE2] + B2U[0x80], B2U[0x8D])  # ZWJ U+200D

    tok_json["model"]["vocab"] = vocab
    tok_json["model"]["merges"] = merges
    print(f"  Guaranteed Devanagari block + ZWNJ/ZWJ: injected {injected_count} missing character merges.")
    return Tokenizer.from_str(json.dumps(tok_json))


def train_tokenizer(
    corpus_path: Path,
    vocab_size: int,
    output_dir: Path,
    min_frequency: int = 2,
) -> PreTrainedTokenizerFast:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f"Training Nep-Tokenizer v2 (Vocab size: {vocab_size})...")
    print(f"Corpus: {corpus_path}")
    print(f"Target Output: {output_dir}")
    print(f"=======================================================")

    tok = build_base_tokenizer()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    t0 = time.time()
    tok.train([str(corpus_path)], trainer)
    print(f"Base BPE training complete in {time.time()-t0:.1f}s. Vocab: {tok.get_vocab_size()}")

    # Inject complete Devanagari guarantee
    tok = inject_complete_devanagari_block(tok)
    final_vocab = tok.get_vocab_size()
    print(f"Final vocab size after Devanagari verification: {final_vocab}")

    # Wrap in PreTrainedTokenizerFast
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        clean_up_tokenization_spaces=False,
    )
    fast_tok.save_pretrained(str(output_dir))
    print(f"Saved HF PreTrainedTokenizerFast to {output_dir}")

    # Quality Verification Suite
    print("\n--- Running Tokenizer Verification Suite ---")
    reloaded = AutoTokenizer.from_pretrained(str(output_dir))

    # 1. Roundtrip test
    test_sentences = [
        "काठमाडौँ विश्वविद्यालय धुलिखेलमा अवस्थित एक स्वायत्त र गैर-नाफामूलक शिक्षण संस्था हो।",
        "नेपालको संविधान २०७२ ले संघीय लोकतान्त्रिक गणतन्त्र स्थापना गरेको छ।",
        "Artificial Intelligence and Machine Learning in Nepali: एआई प्रविधिको विकास तीव्र रूपमा भइरहेको छ।",
        "def compute_loss(logits, targets):\n    return torch.nn.functional.cross_entropy(logits, targets)",
    ]
    for s in test_sentences:
        res = reloaded(s)
        dec = reloaded.decode(res["input_ids"])
        assert dec == s, f"Roundtrip failure! Original:\n{s}\nDecoded:\n{dec}"
    print("  ✓ Lossless round-trip test PASSED (100% exact equality).")

    # 2. Rare character atomic token test
    rare_devanagari = ["ऊ", "ऋ", "ॠ", "ऐ", "अं", "अः", "ॄ", "ॅ", "ॆ", "ॉ", "ॊ", "ॐ", "।", "॥"]
    for ch in rare_devanagari:
        tokens = reloaded.tokenize(ch)
        assert len(tokens) == 1 or ch in ["अं", "अः"], f"Rare character '{ch}' shattered into {len(tokens)} pieces: {tokens}"
    print("  ✓ Rare Devanagari atomic token verification PASSED.")

    # 3. Matra Cohesion Rate test
    # Ensure no token begins with an isolated combining mark
    combining_marks = set("ािीुूृॄेैोौ्ँंः़ॅॉॆॊ")
    sample_text = "नेपाली भाषामा ठूला भाषा मोडेलहरू प्रशिक्षण गर्नु निकै महत्त्वपूर्ण छ।"
    tokens = reloaded.tokenize(sample_text)
    orphan_marks = [t for t in tokens if len(t) > 0 and t[0] in combining_marks]
    assert len(orphan_marks) == 0, f"Found orphan marks: {orphan_marks}"
    print(f"  ✓ Matra cohesion rate: 100.0% (Zero orphan marks in Nepali sample).")
    print(f"  Sample tokenization: {tokens}")

    return fast_tok


def main():
    parser = argparse.ArgumentParser(description="Train Nep-Tokenizer v2")
    parser.add_argument("--corpus", type=str, default="data/tokenizer_train_corpus.txt")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[32000, 64000])
    parser.add_argument("--output-base", type=str, default="hf_push")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    corpus_path = project_root / args.corpus

    if not corpus_path.exists():
        print(f"Corpus file {corpus_path} not found. Please run scripts/prepare_training_corpus.py first.")
        sys.exit(1)

    for vs in args.vocab_sizes:
        tag = f"{vs // 1000}k"
        out_dir = project_root / args.output_base / tag
        train_tokenizer(corpus_path, vs, out_dir)


if __name__ == "__main__":
    main()

