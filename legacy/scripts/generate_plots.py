import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Use clean publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_pareto_efficiency(results_path: str = "results/benchmark_results.json", out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]

    methods = [row["Method"] for row in summary]
    accuracies = [row["Accuracy (%)"] for row in summary]
    tokens = [row["Avg Tokens"] for row in summary]

    # Figure 1: Accuracy vs Tokens (Pareto Frontier)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172D5"]
    markers = ["o", "s", "^", "*"]

    for i, m in enumerate(methods):
        size = 180 if "DESPIN" in m else 100
        ax.scatter(tokens[i], accuracies[i], color=colors[i % len(colors)], marker=markers[i % len(markers)], s=size, label=m, zorder=5)
        offset_y = 1.0 if "DESPIN" in m else -1.5
        ax.annotate(m, (tokens[i], accuracies[i] + offset_y), fontsize=9, ha="center", weight="bold" if "DESPIN" in m else "normal")

    ax.set_xlabel("Average Generated Tokens per Problem (Compute Cost)")
    ax.set_ylabel("Pass@1 Accuracy (%)")
    ax.set_title("Compute vs. Accuracy Trade-off on Reasoning Benchmarks")
    ax.grid(True, linestyle="--", alpha=0.5)

    out_path = os.path.join(out_dir, "fig1_pareto_efficiency.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    # Figure 2: Token Efficiency Score (TES)
    tes_scores = [row["Token Efficiency Score"] for row in summary]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.barh(methods, tes_scores, color=colors[:len(methods)], alpha=0.85, edgecolor="black")
    ax.set_xlabel("Token Efficiency Score (Accuracy / [Tokens/1000])")
    ax.set_title("Token Efficiency Comparison (Higher is Better)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, weight="bold")

    out_path = os.path.join(out_dir, "fig2_token_efficiency_score.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_pareto_efficiency()

