#!/usr/bin/env python3
"""Train SentencePiece BPE tokenizers from a streaming HF dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, get_token, login

try:
    import sentencepiece as spm
except Exception as exc:  # pragma: no cover
    raise SystemExit("sentencepiece is required. Install via pip install sentencepiece") from exc

# Ensure repo root is on path
import sys

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nepali_tokenizer.hf_utils import get_field_value
from nepali_tokenizer.quality_filters import FilterSpec, normalize_text, passes_quality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_texts(
    repo_id: str,
    split: str,
    field: str,
    config: Optional[str],
    max_samples: Optional[int],
    shuffle_buffer: Optional[int],
    seed: int,
    filter_spec: Optional[FilterSpec],
) -> Iterator[str]:
    if config:
        ds = load_dataset(repo_id, name=config, split=split, streaming=True)
    else:
        ds = load_dataset(repo_id, split=split, streaming=True)
    if shuffle_buffer and shuffle_buffer > 0:
        try:
            ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        except Exception:
            logger.warning("Shuffle failed; continuing without shuffle")
    seen = 0
    for row in ds:
        text_raw = get_field_value(row, field)
        if not isinstance(text_raw, str):
            continue
        text_norm = normalize_text(text_raw)
        if not text_norm:
            continue
        if filter_spec and not passes_quality(text_norm, filter_spec):
            continue
        yield text_norm
        seen += 1
        if max_samples and seen >= max_samples:
            break


def config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_repo(api: HfApi, repo_id: str, repo_type: str, token: str) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
    except Exception:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, token=token)


def upload_model_artifacts(
    repo_id: str,
    repo_type: str,
    model_prefix: Path,
    commit_message: str,
    subdir: Optional[str] = None,
) -> None:
    token = get_token() or os.getenv("HF_TOKEN")
    if not token:
        login()
        token = get_token()
    api = HfApi()
    ensure_repo(api, repo_id, repo_type, token)

    files = [
        model_prefix.with_suffix(".model"),
        model_prefix.with_suffix(".vocab"),
        model_prefix.with_suffix(".json"),
    ]
    for path in files:
        if not path.exists():
            logger.warning("Skipping missing artifact: %s", path)
            continue
        name = path.name
        path_in_repo = f"{subdir.rstrip('/')}/{name}" if subdir else name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizers")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", help="HF dataset config name")
    parser.add_argument("--field", default="text")
    parser.add_argument(
        "--train-config",
        default="configs/train_config.yml",
        help="YAML config with vocab sizes and SP options",
    )
    parser.add_argument(
        "--vocab-sizes",
        help="Comma-separated list of vocab sizes to override config (e.g. 64000,100000)",
    )
    parser.add_argument("--output-dir", default="data/tokenizers")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--user-defined-symbols",
        help=(
            "Comma-separated list of user defined symbols to override config. "
            "Use empty string to disable."
        ),
    )
    parser.add_argument(
        "--disable-user-defined-symbols",
        action="store_true",
        help="Ignore user_defined_symbols from config (and any CLI override).",
    )
    parser.add_argument(
        "--upload-repo",
        help="HF repo_id to upload tokenizer artifacts after each vocab size",
    )
    parser.add_argument(
        "--upload-repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repo type for uploads (default: model)",
    )
    parser.add_argument(
        "--upload-subdir",
        default="",
        help="Optional subdirectory inside the repo to place artifacts",
    )
    parser.add_argument(
        "--upload-commit-message",
        default="Add SentencePiece tokenizer artifacts",
        help="Commit message for HF uploads",
    )
    args = parser.parse_args()

    cfg = load_config(args.train_config)
    if args.vocab_sizes:
        vocab_sizes = [int(v.strip()) for v in args.vocab_sizes.split(",") if v.strip()]
    else:
        vocab_sizes = cfg.get("vocab_sizes", [64000])
    model_type = cfg.get("model_type", "bpe")
    normalization = cfg.get("normalization", "nfkc")
    character_coverage = float(cfg.get("character_coverage", 0.9995))
    byte_fallback = bool(cfg.get("byte_fallback", True))
    user_defined_symbols = cfg.get("user_defined_symbols") or []
    corpus_tag = cfg.get("corpus_tag")
    if args.disable_user_defined_symbols:
        user_defined_symbols = []
    elif args.user_defined_symbols is not None:
        if args.user_defined_symbols.strip() == "":
            user_defined_symbols = []
        else:
            user_defined_symbols = [
                s.strip() for s in args.user_defined_symbols.split(",") if s.strip()
            ]

    filter_spec = None
    if isinstance(cfg.get("filters"), dict):
        filter_spec = FilterSpec.from_dict(cfg["filters"])

    os.makedirs(args.output_dir, exist_ok=True)
    slug = args.source_repo.replace("/", "_")
    cfg_hash = config_hash(cfg)

    for vocab_size in vocab_sizes:
        model_prefix = Path(args.output_dir) / f"{slug}-{vocab_size}"
        logger.info("Training SentencePiece BPE vocab_size=%s", vocab_size)

        iterator = iter_texts(
            repo_id=args.source_repo,
            split=args.split,
            field=args.field,
            config=args.config,
            max_samples=args.max_samples,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            filter_spec=filter_spec,
        )

        spm.SentencePieceTrainer.train(
            sentence_iterator=iterator,
            model_prefix=str(model_prefix),
            vocab_size=int(vocab_size),
            model_type=model_type,
            normalization_rule_name=normalization,
            character_coverage=character_coverage,
            byte_fallback=byte_fallback,
            user_defined_symbols=user_defined_symbols,
        )

        meta = {
            "source_repo": args.source_repo,
            "split": args.split,
            "config": args.config,
            "field": args.field,
            "vocab_size": int(vocab_size),
            "model_type": model_type,
            "normalization": normalization,
            "character_coverage": character_coverage,
            "byte_fallback": byte_fallback,
            "user_defined_symbols": user_defined_symbols,
            "max_samples": args.max_samples,
            "shuffle_buffer": args.shuffle_buffer,
            "seed": args.seed,
            "config_hash": cfg_hash,
        }
        if corpus_tag:
            meta["corpus_tag"] = corpus_tag
        meta_path = Path(f"{model_prefix}.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved model + metadata: %s", model_prefix)

        if args.upload_repo:
            logger.info("Uploading artifacts for vocab_size=%s to %s", vocab_size, args.upload_repo)
            upload_model_artifacts(
                repo_id=args.upload_repo,
                repo_type=args.upload_repo_type,
                model_prefix=model_prefix,
                commit_message=args.upload_commit_message,
                subdir=args.upload_subdir or None,
            )


if __name__ == "__main__":
    main()
