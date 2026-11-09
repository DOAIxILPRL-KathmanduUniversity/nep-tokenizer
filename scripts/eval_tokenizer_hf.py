#!/usr/bin/env python3
"""Evaluate HF model tokenizers and baseline tokenizers on a fixed eval corpus."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "transformers is required. Install via pip install transformers"
    ) from exc

try:
    import sentencepiece as spm
except Exception:
    spm = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = "configs/eval_baselines.yml"

DEFAULT_MODELS = [
    "himalaya-ai/nepali-tokenizer-bpe-64k-5b",
    "sajalregmi4/arkios-tokenizer",
    "ampixa/yak-tokenizer@v2.0/64k",
    "ampixa/yak-tokenizer@v2.0/128k",
    "bharatgenai/Param2-17B-A2.4B-Thinking",
    "sarvamai/sarvam-105b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-27b-it",
]


class UnifiedTokenizer:
    """Unified wrapper exposing encode_ids and unk_id for HF and SentencePiece models."""

    def __init__(
        self,
        backend: Any,
        model_type: str = "hf",
        unk_id: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.model_type = model_type
        if unk_id is not None:
            self.unk_id = unk_id
        elif model_type == "sp":
            self.unk_id = backend.unk_id()
        else:
            self.unk_id = getattr(backend, "unk_token_id", None)

    def encode_ids(self, text: str) -> List[int]:
        if self.model_type == "sp":
            return self.backend.encode(text, out_type=int)
        try:
            return self.backend.encode(text, add_special_tokens=False)
        except Exception:
            return self.backend(text, add_special_tokens=False).input_ids


def load_tokenizer_entry(
    repo_or_path: str,
    subfolder: Optional[str] = None,
    model_type: Optional[str] = None,
    trust_remote_code: bool = False,
    use_fast: bool = True,
) -> UnifiedTokenizer:
    """Load a tokenizer from a Hugging Face repo, subfolder, or SentencePiece file."""
    # Support inline subfolder notation: repo@subfolder or repo:subfolder
    if not subfolder:
        if "@" in repo_or_path:
            repo_or_path, subfolder = repo_or_path.split("@", 1)
        elif ":" in repo_or_path and not repo_or_path.startswith(("http:", "https:")):
            parts = repo_or_path.split(":", 1)
            repo_or_path, subfolder = parts[0], parts[1]

    # Explicit SentencePiece or local .model file
    if model_type == "sentencepiece" or repo_or_path.endswith(".model"):
        if os.path.exists(repo_or_path):
            if spm is None:
                raise RuntimeError("sentencepiece is required to load .model files")
            sp = spm.SentencePieceProcessor()
            sp.load(repo_or_path)
            return UnifiedTokenizer(sp, model_type="sp")

    kwargs: Dict[str, Any] = {}
    if subfolder:
        kwargs["subfolder"] = subfolder

    # Try standard AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            repo_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            **kwargs,
        )
        return UnifiedTokenizer(tok, model_type="hf")
    except Exception as hf_err:
        # Fallback: check if the HF repository stores a SentencePiece .model file
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            files = list_repo_files(repo_or_path)
            m_files = [f for f in files if f.endswith(".model")]
            if m_files:
                if spm is None:
                    raise RuntimeError("sentencepiece is required to load .model files")
                downloaded = hf_hub_download(repo_or_path, m_files[0])
                sp = spm.SentencePieceProcessor()
                sp.load(downloaded)
                return UnifiedTokenizer(sp, model_type="sp")
        except Exception:
            pass
        raise hf_err


def list_eval_files(eval_dir: str) -> Dict[str, Path]:
    eval_path = Path(eval_dir)
    files = {}
    for path in eval_path.glob("eval_*.jsonl"):
        lang = path.stem.replace("eval_", "")
        files[lang] = path
    return files


def eval_file(tokenizer: UnifiedTokenizer, path: Path) -> Dict[str, Any]:
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    total_docs = 0
    unk_tokens = 0
    unk_id = tokenizer.unk_id

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            total_docs += 1
            total_chars += len(text)
            total_bytes += len(text.encode("utf-8"))
            words = len(text.split())
            total_words += words
            ids = tokenizer.encode_ids(text)
            total_tokens += len(ids)
            if unk_id is not None and unk_id >= 0:
                unk_tokens += sum(1 for i in ids if i == unk_id)

    fertility = (total_tokens / total_words) if total_words else 0.0
    token_word_ratio = fertility
    tokens_per_char = (total_tokens / total_chars) if total_chars else 0.0
    bytes_per_token = (total_bytes / total_tokens) if total_tokens else 0.0
    unk_rate = (unk_tokens / total_tokens) if total_tokens else 0.0

    return {
        "docs": total_docs,
        "tokens": total_tokens,
        "words": total_words,
        "chars": total_chars,
        "bytes": total_bytes,
        "fertility": fertility,
        "token_word_ratio": token_word_ratio,
        "tokens_per_char": tokens_per_char,
        "bytes_per_token": bytes_per_token,
        "unk_rate": unk_rate,
    }


def parse_model_specs(
    config_path: Optional[str],
    cli_models: str,
) -> List[Dict[str, Any]]:
    """Resolve model specifications from config file or CLI arguments."""
    specs: List[Dict[str, Any]] = []

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            raw_baselines = cfg.get("baselines", [])
            for item in raw_baselines:
                if isinstance(item, dict) and "id" in item:
                    specs.append(item)
                elif isinstance(item, str):
                    specs.append({"id": item, "name": item})

    if cli_models.strip():
        for m in cli_models.split(","):
            m = m.strip()
            if m:
                specs.append({"id": m})

    if not specs:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                for item in cfg.get("baselines", []):
                    if isinstance(item, dict) and "id" in item:
                        specs.append(item)
                    elif isinstance(item, str):
                        specs.append({"id": item, "name": item})
        else:
            specs = [{"id": m} for m in DEFAULT_MODELS]

    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HF and baseline tokenizers")
    parser.add_argument(
        "--eval-dir",
        default="data/eval",
        help="Directory with eval_*.jsonl files",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML baseline config (e.g. configs/eval_baselines.yml)",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated list of HF model repo IDs (optional override)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval_results_hf",
        help="Output directory for JSON/CSV results",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom tokenizer code from model repos",
    )
    parser.add_argument(
        "--use-slow",
        action="store_true",
        help="Use slow tokenizers instead of fast ones",
    )
    args = parser.parse_args()

    model_specs = parse_model_specs(args.config, args.models)

    eval_files = list_eval_files(args.eval_dir)
    if not eval_files:
        raise SystemExit("No eval_*.jsonl files found in eval dir")

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for spec in model_specs:
        repo_id = spec["id"]
        subfolder = spec.get("subfolder")
        model_type = spec.get("type")
        display_name = spec.get("name")

        # Determine clean key for results
        if display_name:
            model_key = display_name
        else:
            model_key = repo_id.replace("/", "__")
            if subfolder:
                model_key += f"__{subfolder.replace('/', '_')}"

        logger.info("Loading tokenizer: %s (key=%s)", repo_id, model_key)
        try:
            tokenizer = load_tokenizer_entry(
                repo_or_path=repo_id,
                subfolder=subfolder,
                model_type=model_type,
                trust_remote_code=args.trust_remote_code,
                use_fast=not args.use_slow,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s", repo_id, exc)
            continue

        results[model_key] = {}

        for lang, path in eval_files.items():
            metrics = eval_file(tokenizer, path)
            results[model_key][lang] = metrics
            csv_rows.append(
                {
                    "model": model_key,
                    "lang": lang,
                    **metrics,
                }
            )
            logger.info(
                "%s %s fertility=%.4f (bpt=%.2f)",
                model_key,
                lang,
                metrics["fertility"],
                metrics["bytes_per_token"],
            )

    if not csv_rows:
        raise SystemExit("No results were produced (all models failed to load?)")

    json_path = Path(args.output_dir) / "tokenizer_eval.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = Path(args.output_dir) / "tokenizer_eval.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("Saved eval results to %s and %s", json_path, csv_path)


if __name__ == "__main__":
    main()
