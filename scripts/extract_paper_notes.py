#!/usr/bin/env python3
"""Extract tokenizer-related notes from Param-1 and KrutrimAI PDFs."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List

try:
    import PyPDF2
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyPDF2 is required. Install via pip install PyPDF2") from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_text(path: Path, max_pages: int | None = None) -> str:
    reader = PyPDF2.PdfReader(str(path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    chunks = []
    for page in pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def find_snippets(text: str, keywords: List[str], max_snippets: int = 6) -> List[str]:
    snippets: List[str] = []
    for line in text.splitlines():
        if any(k.lower() in line.lower() for k in keywords):
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned and cleaned not in snippets:
                snippets.append(cleaned)
        if len(snippets) >= max_snippets:
            break
    return snippets


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tokenizer notes from PDFs")
    parser.add_argument(
        "--param",
        required=True,
        help="Path to Param-1 PDF (2407.12481v2.pdf)",
    )
    parser.add_argument(
        "--krutrim",
        required=True,
        help="Path to KrutrimAI PDF (2507.13390v1.pdf)",
    )
    parser.add_argument(
        "--output",
        default="docs/tokenizer_paper_notes.md",
        help="Output markdown file",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    param_path = Path(args.param)
    krutrim_path = Path(args.krutrim)

    param_text = extract_text(param_path, args.max_pages)
    krutrim_text = extract_text(krutrim_path, args.max_pages)

    param_snips = find_snippets(param_text, ["fertility", "token", "vocab", "bharat", "tokenizer"])
    krutrim_snips = find_snippets(krutrim_text, ["token", "word", "ratio", "vocab", "tokenizer"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = [
        "# Tokenizer Paper Notes",
        "",
        "## Param-1 (2407.12481v2)",
        "",
        "**Reported BharatGen fertility (from baseline):**",
        "- 64k vocab: 1.87",
        "- 128k vocab: 1.60",
        "",
        "**Extracted snippets:**",
    ]
    if param_snips:
        content.extend([f"- {s}" for s in param_snips])
    else:
        content.append("- (No snippets found; check PDF parsing.)")

    content.extend(
        [
            "",
            "## KrutrimAI (2507.13390v1)",
            "",
            "**Extracted snippets:**",
        ]
    )
    if krutrim_snips:
        content.extend([f"- {s}" for s in krutrim_snips])
    else:
        content.append("- (No snippets found; check PDF parsing.)")

    out_path.write_text("\n".join(content), encoding="utf-8")
    logger.info("Wrote notes to %s", out_path)


if __name__ == "__main__":
    main()
