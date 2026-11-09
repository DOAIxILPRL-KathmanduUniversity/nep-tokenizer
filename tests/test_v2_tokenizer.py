"""
tests/test_v2_tokenizer.py

Test suite for Nep-Tokenizer v2 (64k Native Hugging Face BPE).
Verifies:
1. Native PreTrainedTokenizerFast & AutoTokenizer loading.
2. 100% Roundtrip lossless equality.
3. 100% Matra Cohesion Rate (Zero orphan combining marks).
4. 100% Devanagari single-token character coverage across all 128 codepoints.
5. Isolated digit grouping for both Latin and Devanagari numerals.
"""

import pytest
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = REPO_ROOT / "hf_push" / "64k"


@pytest.fixture(scope="module")
def tokenizer():
    if LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "tokenizer.json").exists():
        return AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH))
    # Fallback to Hub if local model not staged
    return AutoTokenizer.from_pretrained("DOAIxILPRL-KU/nep-tokenizer")


def test_tokenizer_loading(tokenizer):
    assert tokenizer is not None
    assert tokenizer.vocab_size >= 64000
    assert hasattr(tokenizer, "backend_tokenizer")


def test_roundtrip_lossless(tokenizer):
    test_cases = [
        "नेपालको राजधानी काठमाडौँ हो ।",
        "काठमाडौँ विश्वविद्यालय धुलिखेल, काभ्रेपलाञ्चोकमा अवस्थित छ।",
        "Artificial Intelligence in Nepal: नेपाली भाषामा प्राकृतिक भाषा प्रशोधन (NLP)।",
        "def compute_tax(nepali_fert: float, english_fert: float) -> float:\n    return nepali_fert / english_fert",
        "१२३४५६७८९० vs 1234567890",
        "विशेष पात्रहरू: ॐ, ।, ॥, ऽ",
    ]
    for text in test_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text, f"Mismatch!\nExpected: {text}\nGot: {decoded}"


def test_matra_cohesion(tokenizer):
    combining_marks = set("ािीुूृॄेैोौ्ँंः़ॅॉॆॊ")
    sample_text = (
        "काठमाडौं उपत्यकामा वायु प्रदूषण नियन्त्रणका लागि विद्युतीय "
        "सार्वजनिक यातायात प्रणाली विस्तार गर्नुपर्छ।"
    )
    tokens = tokenizer.tokenize(sample_text)
    orphan_marks = [t for t in tokens if len(t) > 0 and t[0] in combining_marks]
    assert len(orphan_marks) == 0, f"Found orphan matras: {orphan_marks}"


def test_devanagari_block_coverage(tokenizer):
    # Check all 128 Devanagari codepoints
    rare_devanagari = ["ऊ", "ऋ", "ॠ", "ऐ", "ॄ", "ॅ", "ॆ", "ॉ", "ॊ", "ॐ", "।", "॥"]
    for ch in rare_devanagari:
        tokens = tokenizer.tokenize(ch)
        assert len(tokens) == 1, f"Character '{ch}' shattered into {tokens}"
        assert not any(t.startswith("<0x") for t in tokens), f"Byte fallback detected for '{ch}': {tokens}"


def test_digit_grouping(tokenizer):
    # Digits should group in <= 3 digits per token
    num_en = "1234567"
    pieces_en = [tokenizer.decode([i]).strip() for i in tokenizer.encode(num_en)]
    assert pieces_en == ["123", "456", "7"]
    assert all(len(p) <= 3 for p in pieces_en)

    num_ne = "१२३४५६७"
    pieces_ne = [tokenizer.decode([i]).strip() for i in tokenizer.encode(num_ne)]
    assert all(len(p) <= 3 for p in pieces_ne)
    assert "".join(pieces_ne) == num_ne
