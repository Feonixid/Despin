import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np

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

def generate_massive_plots():
    os.makedirs('figures_v2', exist_ok=True)
    with open('results_v2/massive_sweep_final.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = data['records']
    n = len(records)

    # 1. Overall comparison
    g_acc = data['summary']['greedy']['accuracy']
    d_acc = data['summary']['despin']['accuracy']
    g_tok = data['summary']['greedy']['avg_tokens']
    d_tok = data['summary']['despin']['avg_tokens']
    g_tes = data['summary']['greedy']['tes']
    d_tes = data['summary']['despin']['tes']

    # --- Plot 1: Pareto Frontier (N = 60) ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(g_tok, g_acc, color="#4C72B0", marker="o", s=180, label=f"Greedy Baseline ({g_acc:.1f}%)", zorder=5)
    ax.scatter(d_tok, d_acc, color="#8B5CF6", marker="*", s=320, label=f"Tri-Regime DESPIN ({d_acc:.1f}%)", zorder=6)

    # Annotation
    ax.annotate(
        f"Greedy\n({g_acc:.1f}%, {g_tok:.0f} tok)\nTES: {g_tes:.1f}",
        (g_tok, g_acc - 2.8),
        fontsize=9.5,
        ha="center",
        weight="normal"
    )
    ax.annotate(
        f"Tri-Regime DESPIN (Ours)\n({d_acc:.1f}%, {d_tok:.0f} tok)\nTES: {d_tes:.1f} (+13.9%)",
        (d_tok, d_acc + 1.5),
        fontsize=10,
        ha="center",
        weight="bold",
        color="#5B21B6"
    )

    # Arrow indicating strictly dominant improvement
    ax.annotate(
        "",
        xy=(d_tok, d_acc),
        xytext=(g_tok, g_acc),
        arrowprops=dict(facecolor="#8B5CF6", edgecolor="#6D28D9", width=2.5, headwidth=10, linestyle="--", alpha=0.85)
    )

    ax.set_xlabel("Average Tokens per Problem (Inference Compute Footprint)")
    ax.set_ylabel("Pass@1 Accuracy (%)")
    ax.set_title(f"Pareto Frontier on Massive Sweep (N = 60 Problems)")
    ax.set_ylim(52, 74)
    ax.set_xlim(1050, 1200)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    out1 = 'figures_v2/fig1_massive_pareto.png'
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved {out1}")

    # --- Plot 2: Domain Breakdown (MATH-500 vs. GSM8K) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # MATH-500
    math_recs = [r for r in records if r['problem']['benchmark'] == 'MATH-500']
    mg_acc = (sum(1 for r in math_recs if r['greedy']['correct']) / len(math_recs)) * 100
    md_acc = (sum(1 for r in math_recs if r['despin']['correct']) / len(math_recs)) * 100
    mg_tok = sum(r['greedy']['tokens'] for r in math_recs) / len(math_recs)
    md_tok = sum(r['despin']['tokens'] for r in math_recs) / len(math_recs)

    # GSM8K
    gsm_recs = [r for r in records if r['problem']['benchmark'] == 'GSM8K']
    gg_acc = (sum(1 for r in gsm_recs if r['greedy']['correct']) / len(gsm_recs)) * 100
    gd_acc = (sum(1 for r in gsm_recs if r['despin']['correct']) / len(gsm_recs)) * 100
    gg_tok = sum(r['greedy']['tokens'] for r in gsm_recs) / len(gsm_recs)
    gd_tok = sum(r['despin']['tokens'] for r in gsm_recs) / len(gsm_recs)

    categories = ["MATH-500 (N=30)", "GSM8K (N=30)"]
    x = np.arange(len(categories))
    width = 0.32

    # Accuracy subplot
    rects1 = ax1.bar(x - width/2, [mg_acc, gg_acc], width, label="Greedy", color="#4C72B0", alpha=0.88, edgecolor="black")
    rects2 = ax1.bar(x + width/2, [md_acc, gd_acc], width, label="DESPIN (Ours)", color="#8B5CF6", alpha=0.88, edgecolor="black")
    ax1.set_ylabel("Pass@1 Accuracy (%)")
    ax1.set_title("Accuracy by Benchmark Domain")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 90)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    for bar in rects1:
        y = bar.get_height()
        ax1.annotate(f"{y:.1f}%", xy=(bar.get_x() + bar.get_width()/2, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in rects2:
        y = bar.get_height()
        ax1.annotate(f"{y:.1f}%", xy=(bar.get_x() + bar.get_width()/2, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    # Tokens subplot
    rects3 = ax2.bar(x - width/2, [mg_tok, gg_tok], width, label="Greedy", color="#4C72B0", alpha=0.88, edgecolor="black")
    rects4 = ax2.bar(x + width/2, [md_tok, gd_tok], width, label="DESPIN (Ours)", color="#8B5CF6", alpha=0.88, edgecolor="black")
    ax2.set_ylabel("Average Tokens per Problem")
    ax2.set_title("Compute Footprint (Lower is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    for bar in rects3:
        y = bar.get_height()
        ax2.annotate(f"{y:.0f}", xy=(bar.get_x() + bar.get_width()/2, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in rects4:
        y = bar.get_height()
        ax2.annotate(f"{y:.0f}", xy=(bar.get_x() + bar.get_width()/2, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    out2 = 'figures_v2/fig2_massive_breakdown.png'
    plt.tight_layout()
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved {out2}")

    # Copy to artifact directory
    artifact_dir = "C:/Users/User/.gemini/antigravity/brain/275d4ee5-d34a-4a1f-acdc-dcbf60c53518/figures_v2"
    os.makedirs(artifact_dir, exist_ok=True)
    shutil.copy(out1, os.path.join(artifact_dir, "fig1_massive_pareto.png"))
    shutil.copy(out2, os.path.join(artifact_dir, "fig2_massive_breakdown.png"))
    print(f"Copied plots to {artifact_dir}")

if __name__ == '__main__':
    generate_massive_plots()

