#!/usr/bin/env python3
"""Render a markdown comparison table from tokenizer_eval.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected eval format in {path}")
    return data  # {model: {lang: metrics}}


def normalize_model_name(name: str, pretty: bool) -> str:
    if pretty:
        return name.replace("__", "/")
    return name


def shorten_model_name(name: str, mode: str) -> str:
    if mode == "full":
        return name
    # Prefer HF-style repo/name, otherwise fall back to suffix after "__"
    if "/" in name:
        return name.split("/")[-1]
    if "__" in name:
        return name.split("__")[-1]
    return name


def resolve_input_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            p = p / "tokenizer_eval.json"
        paths.append(p)
    return paths


def merge_results(
    paths: List[Path], pretty_names: bool, name_mode: str, aliases: Dict[str, str]
) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    model_order: List[str] = []
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for path in paths:
        data = load_results(path)
        for model_name, per_lang in data.items():
            base = normalize_model_name(model_name, pretty_names)
            base = shorten_model_name(base, name_mode)
            base = aliases.get(base, base)
            name = base
            if name in results:
                suffix = 2
                while f"{base}#{suffix}" in results:
                    suffix += 1
                name = f"{base}#{suffix}"
            results[name] = per_lang
            model_order.append(name)
    return model_order, results


def render_markdown_table(
    model_order: List[str],
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    round_digits: int,
    lang_order: List[str] | None,
    bold_min: bool,
    align: str,
) -> str:
    if lang_order:
        langs = lang_order
    else:
        langs = sorted({lang for per_model in results.values() for lang in per_model})

    header = ["lang"] + model_order
    rows = []
    for lang in langs:
        row = [lang]
        numeric_vals: List[float] = []
        for model in model_order:
            metrics = results.get(model, {}).get(lang)
            if metrics and metric in metrics:
                numeric_vals.append(metrics[metric])
        min_val = min(numeric_vals) if numeric_vals else None
        for model in model_order:
            metrics = results.get(model, {}).get(lang)
            if not metrics or metric not in metrics:
                row.append("NA")
            else:
                val = metrics[metric]
                cell = f"{val:.{round_digits}f}"
                if bold_min and min_val is not None and val == min_val:
                    cell = f"**{cell}**"
                row.append(cell)
        rows.append(row)

    # Markdown table
    out_lines = []
    out_lines.append("| " + " | ".join(header) + " |")
    if align == "left":
        sep = ["---"] * len(header)
    elif align == "center":
        sep = ["---"] + [":---:"] * (len(header) - 1)
    else:
        # right align numeric columns
        sep = ["---"] + ["---:"] * (len(header) - 1)
    out_lines.append("| " + " | ".join(sep) + " |")
    for row in rows:
        out_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(out_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a markdown comparison table from tokenizer_eval.json files"
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated list of eval JSON files or directories containing tokenizer_eval.json",
    )
    parser.add_argument(
        "--metric",
        default="fertility",
        help="Metric key to display (default: fertility)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=4,
        help="Decimal places to round (default: 4)",
    )
    parser.add_argument(
        "--aliases",
        default="",
        help="Comma-separated aliases like old:new,old2:new2",
    )
    parser.add_argument(
        "--lang-order",
        default="",
        help="Comma-separated language order override (e.g. ne,hi,en,roman_ne,roman_hi)",
    )
    parser.add_argument(
        "--no-bold-min",
        action="store_true",
        help="Disable bolding the minimum (best) value per row",
    )
    parser.add_argument(
        "--align",
        default="right",
        choices=["left", "right", "center"],
        help="Column alignment for numeric columns",
    )
    args = parser.parse_args()

    inputs = [s.strip() for s in args.inputs.split(",") if s.strip()]
    paths = resolve_input_paths(inputs)

    alias_map: Dict[str, str] = {}
    if args.aliases:
        for pair in args.aliases.split(","):
            if ":" not in pair:
                continue
            old, new = pair.split(":", 1)
            alias_map[old.strip()] = new.strip()

    # Defaults: pretty names + short model names (no CLI flags needed).
    model_order, results = merge_results(paths, True, "model", alias_map)
    lang_order = [s.strip() for s in args.lang_order.split(",") if s.strip()] or None

    table = render_markdown_table(
        model_order=model_order,
        results=results,
        metric=args.metric,
        round_digits=args.round,
        lang_order=lang_order,
        bold_min=not args.no_bold_min,
        align=args.align,
    )
    print(table)


if __name__ == "__main__":
    main()
