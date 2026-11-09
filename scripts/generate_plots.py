#!/usr/bin/env python3
"""
scripts/generate_plots.py

Generate publication-quality evaluation figures for Nep-Tokenizer v2 across ALL evaluated models.
Affiliation: Department of Artificial Intelligence (DOAI) &
             Information and Language Processing Research Lab (ILPRL),
             Kathmandu University.

Loads empirical benchmark results directly from data/benchmark_evaluation_results.json.

Outputs:
  - figures/fig1_vocab_scaling_fertility.png
  - figures/fig2_domain_fertility_comparison.png
  - figures/fig3_matra_cohesion_comparison.png
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Publication-grade styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "font.family": "sans-serif",
})

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_benchmark_data() -> list[dict]:
    data_path = REPO_ROOT / "data" / "benchmark_evaluation_results.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark results not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fig1_vocab_scaling(data: list[dict], output_dir: Path) -> Path:
    """Figure 1: Vocab Size Scaling vs. Nepali Fertility and Information Density (BPT) across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Separate points for line connections vs scatters
    nep_v2_x, nep_v2_y, nep_v2_bpt = [], [], []
    yak_x, yak_y, yak_bpt = [], [], []
    others = []

    for r in data:
        name = r["name"]
        vk = r["vocab_size"] / 1000.0
        fert = r["ne_fertility"]
        bpt = r["bpt"]

        if "Nep-Tokenizer v2" in name:
            nep_v2_x.append(vk)
            nep_v2_y.append(fert)
            nep_v2_bpt.append(bpt)
        elif "Yak" in name:
            yak_x.append(vk)
            yak_y.append(fert)
            yak_bpt.append(bpt)
        else:
            others.append((name, vk, fert, bpt))

    # Sort lines
    idx = np.argsort(nep_v2_x)
    ax1.plot(np.array(nep_v2_x)[idx], np.array(nep_v2_y)[idx], "o-", color="#1f77b4", linewidth=2.8, markersize=9, label="DOAIxILPRL Nep-Tokenizer v2")
    ax2.plot(np.array(nep_v2_x)[idx], np.array(nep_v2_bpt)[idx], "o-", color="#1f77b4", linewidth=2.8, markersize=9, label="DOAIxILPRL Nep-Tokenizer v2")

    idx_y = np.argsort(yak_x)
    ax1.plot(np.array(yak_x)[idx_y], np.array(yak_y)[idx_y], "s--", color="#2ca02c", linewidth=2.0, markersize=8, label="Ampixa Yak")
    ax2.plot(np.array(yak_x)[idx_y], np.array(yak_bpt)[idx_y], "s--", color="#2ca02c", linewidth=2.0, markersize=8, label="Ampixa Yak")

    # Palette for other models
    palette = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#b15928", "#33a02c", "#fb9a99"]
    markers = ["^", "v", "D", "P", "X", "*", "H", "x", "+", "<", ">"]

    for i, (name, vk, fert, bpt) in enumerate(others):
        c = palette[i % len(palette)]
        m = markers[i % len(markers)]
        ax1.scatter([vk], [fert], color=c, marker=m, s=90, label=name, zorder=5)
        ax2.scatter([vk], [bpt], color=c, marker=m, s=90, label=name, zorder=5)

    ax1.set_xlabel("Vocabulary Size (k)")
    ax1.set_ylabel("Nepali Fertility (tokens / word)")
    ax1.set_title("(a) Nepali Subword Fertility vs. Vocab Scaling (Lower is Better)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Vocabulary Size (k)")
    ax2.set_ylabel("Information Density: Bytes Per Token (BPT)")
    ax2.set_title("(b) Information Density vs. Vocab Scaling (Higher is Better)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Place combined legend outside
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True)

    fig.suptitle("Nep-Tokenizer v2: Vocabulary Scaling & Multi-Domain Compression Benchmark", fontsize=14, weight="bold")
    plt.tight_layout()
    out_path = output_dir / "fig1_vocab_scaling_fertility.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    return out_path


def plot_fig2_domain_fertility(data: list[dict], output_dir: Path) -> Path:
    """Figure 2: Multi-Domain Compression across News, Literature, Legal, Colloquial, Technical."""
    selected_models = [
        "Nep-Tokenizer v2 (64k)",
        "Nep-Tokenizer v2 (32k)",
        "Yak v2.0 (64k)",
        "Arkios (65k)",
        "NyayaLM-BPE (64k)",
        "sidskarki-BPE (64k)",
        "IndicBERT v2 (250k)",
        "Supernova V5 (48k)",
        "Gemma-2 (256k)",
        "LLaMA-3.1 (128k)",
    ]

    filtered_data = [r for r in data if r["name"] in selected_models]
    # Sort in selected order
    filtered_data.sort(key=lambda r: selected_models.index(r["name"]) if r["name"] in selected_models else 99)

    domains = ["news", "literature", "legal", "colloquial", "technical"]
    domain_labels = ["News", "Literature", "Legal (Acts)", "Colloquial", "Technical / Wiki"]

    model_names = [r["name"] for r in filtered_data]
    n_models = len(model_names)
    n_domains = len(domains)

    matrix = np.zeros((n_models, n_domains))
    for i, r in enumerate(filtered_data):
        dom_dict = r.get("domains", {})
        for j, d in enumerate(domains):
            matrix[i, j] = dom_dict.get(d, 0.0)

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    x = np.arange(n_models)
    width = 0.16

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for j in range(n_domains):
        offset = (j - (n_domains - 1) / 2) * width
        rects = ax.bar(x + offset, matrix[:, j], width, label=domain_labels[j], color=colors[j], alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Domain Fertility (tokens / word)")
    ax.set_title("Multi-Domain Subword Fertility Across Nepali Domains (Lower is Better)", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9.5)
    ax.legend(title="Corpus Domain", frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / "fig2_domain_fertility_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    return out_path


def plot_fig3_matra_cohesion(data: list[dict], output_dir: Path) -> Path:
    """Figure 3: Matra Cohesion Rate & Devanagari Single-Token Coverage Comparison."""
    # Sort models by Matra Severed Error Rate (descending)
    sorted_data = sorted(data, key=lambda r: r.get("matra_severed_rate", 0.0), reverse=True)

    names = [r["name"] for r in sorted_data]
    err_rates = [r.get("matra_severed_rate", 0.0) for r in sorted_data]
    cohesion_rates = [r.get("matra_cohesion_rate", 100.0) for r in sorted_data]
    deva_cov = [r.get("devanagari_coverage_pct", 100.0) for r in sorted_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    y_pos = np.arange(len(names))
    colors1 = ["#d62728" if e > 5.0 else ("#ff7f0e" if e > 0 else "#2ca02c") for e in err_rates]

    # Panel A: Severed Matra Anomaly Rate
    bars1 = ax1.barh(y_pos, err_rates, color=colors1, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Severed Matra Error Rate (%) [The Kathmandu Severance Anomaly]")
    ax1.set_title("(a) Matra Severance Rate (Lower is Better; 0% = Perfect)", fontsize=11, weight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars1, err_rates):
        if val > 0:
            ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.5)

    # Panel B: Devanagari Character Coverage
    colors2 = ["#2ca02c" if c == 100.0 else "#ff7f0e" for c in deva_cov]
    bars2 = ax2.barh(y_pos, deva_cov, color=colors2, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(80, 103)
    ax2.set_xlabel("Devanagari Single-Token Character Coverage (%)")
    ax2.set_title("(b) Devanagari Coverage (100% = All Codepoints Preserved)", fontsize=11, weight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars2, deva_cov):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.5)

    fig.suptitle("Nep-Tokenizer v2 Architectural Integrity: Mark Cohesion & Character Completeness", fontsize=13, weight="bold")
    plt.tight_layout()
    out_path = output_dir / "fig3_matra_cohesion_comparison.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate publication evaluation plots")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_benchmark_data()
    print(f"Loaded benchmark data for {len(data)} models.")

    f1 = plot_fig1_vocab_scaling(data, out_dir)
    print(f"Generated Figure 1: {f1}")

    f2 = plot_fig2_domain_fertility(data, out_dir)
    print(f"Generated Figure 2: {f2}")

    f3 = plot_fig3_matra_cohesion(data, out_dir)
    print(f"Generated Figure 3: {f3}")

    # Also copy figures into hf_push/figures and brain artifacts
    hf_fig_dir = REPO_ROOT / "hf_push" / "figures"
    hf_fig_dir.mkdir(parents=True, exist_ok=True)
    brain_dir = Path("/Users/paramanada/.gemini/antigravity/brain/5e0fc54e-7ae1-4967-adb8-dd7d91772f0b")

    for p in [f1, f2, f3]:
        shutil.copy(p, hf_fig_dir / p.name)
        if brain_dir.exists():
            shutil.copy(p, brain_dir / p.name)
    print("Copied all publication figures to hf_push/figures/ and artifact directory.")


if __name__ == "__main__":
    main()
