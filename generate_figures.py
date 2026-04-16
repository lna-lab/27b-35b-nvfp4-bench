#!/usr/bin/env python3
"""Generate benchmark figures for Dense 27B/31B NVFP4 report."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {"qwen": "#1E88E5", "qwopus": "#8E24AA", "gemma31": "#43A047", "gemma31_lyf": "#78909C"}
LABELS = {
    "qwen": "Qwen3.5-27B (Lna-Lab)",
    "qwopus": "Qwopus-27B (Lna-Lab)",
    "gemma31": "Gemma4-31B (Lna-Lab)",
    "gemma31_lyf": "Gemma4-31B (lyf)",
}

with open("data.json") as f:
    data = json.load(f)

summary = data["summary"]


# ── Fig 1: Quality scores by test (grouped bar) ─────────────────────────────

def fig_quality():
    tests = ["english_critique", "japanese", "math", "coding", "design"]
    test_labels = ["English\nCritique", "Japanese\nExpression", "Math\nReasoning", "Coding", "System\nDesign"]
    models = ["qwen", "qwopus", "gemma31", "gemma31_lyf"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tests))
    width = 0.2

    for i, m in enumerate(models):
        scores = []
        for t in tests:
            row = next((r for r in summary if r["model"] == m and r["test"] == t and r["concurrency"] == 1), None)
            scores.append(row["avg_quality"] if row else 0)
        bars = ax.bar(x + i * width, scores, width, label=LABELS[m], color=COLORS[m], alpha=0.85)
        for bar, s in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{s:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Quality Score (0-1)", fontsize=12)
    ax.set_title("Quality Comparison — Single Request (Concurrency = 1)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(test_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")
    plt.tight_layout()
    plt.savefig("figures/01_quality_comparison.png", dpi=150)
    plt.close()
    print("  01_quality_comparison.png")


# ── Fig 2: Per-request speed (x1 vs x4) ─────────────────────────────────────

def fig_speed_per_request():
    models = ["qwen", "qwopus", "gemma31", "gemma31_lyf"]
    tests = ["english_critique", "japanese", "math", "coding", "design"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, conc in enumerate([1, 4]):
        ax = axes[ax_idx]
        x = np.arange(len(tests))
        width = 0.2
        for i, m in enumerate(models):
            speeds = []
            for t in tests:
                row = next((r for r in summary if r["model"] == m and r["test"] == t and r["concurrency"] == conc), None)
                speeds.append(row["avg_tok_per_sec"] if row else 0)
            ax.bar(x + i * width, speeds, width, label=LABELS[m], color=COLORS[m], alpha=0.85)

        ax.set_ylabel("tok/s (per request)", fontsize=11)
        ax.set_title(f"Per-Request Speed — Concurrency = {conc}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(["Eng", "JPN", "Math", "Code", "Design"], fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/02_speed_per_request.png", dpi=150)
    plt.close()
    print("  02_speed_per_request.png")


# ── Fig 3: Aggregate throughput scaling ──────────────────────────────────────

def fig_throughput_scaling():
    models = ["qwen", "qwopus", "gemma31", "gemma31_lyf"]
    concurrencies = [1, 4]

    fig, ax = plt.subplots(figsize=(10, 6))

    for m in models:
        throughputs = []
        for c in concurrencies:
            # Average across all tests
            rows = [r for r in summary if r["model"] == m and r["concurrency"] == c and r["test"] != "tool_call"]
            avg = np.mean([r["total_throughput"] for r in rows]) if rows else 0
            throughputs.append(avg)
        ax.plot(concurrencies, throughputs, "o-", label=LABELS[m], color=COLORS[m],
                linewidth=2.5, markersize=10)
        for c, t in zip(concurrencies, throughputs):
            ax.annotate(f"{t:.0f}", (c, t), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    # Linear scaling reference
    base = np.mean([r["total_throughput"] for r in summary
                    if r["concurrency"] == 1 and r["test"] != "tool_call"])
    ax.plot(concurrencies, [base * c for c in concurrencies], "k--", alpha=0.3, label="Linear scaling")

    ax.set_xlabel("Concurrency", fontsize=12)
    ax.set_ylabel("Aggregate Throughput (tok/s)", fontsize=12)
    ax.set_title("Throughput Scaling — Average Across All Tests", fontsize=14, fontweight="bold")
    ax.set_xticks(concurrencies)
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 4.5)
    plt.tight_layout()
    plt.savefig("figures/03_throughput_scaling.png", dpi=150)
    plt.close()
    print("  03_throughput_scaling.png")


# ── Fig 4: Latency distribution ──────────────────────────────────────────────

def fig_latency():
    models = ["qwen", "qwopus", "gemma31", "gemma31_lyf"]
    results = data["results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, conc in enumerate([1, 4]):
        ax = axes[ax_idx]
        latencies_per_model = []
        labels = []
        colors = []
        for m in models:
            lats = [r["latency_ms"] / 1000 for r in results
                    if r["model"] == m and r["concurrency"] == conc and r["test"] != "tool_call" and r["latency_ms"] > 0]
            latencies_per_model.append(lats)
            labels.append(LABELS[m].split(" (")[0])
            colors.append(COLORS[m])

        bp = ax.boxplot(latencies_per_model, labels=labels, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        ax.set_ylabel("Latency (seconds)", fontsize=11)
        ax.set_title(f"Latency Distribution — Concurrency = {conc}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/04_latency_distribution.png", dpi=150)
    plt.close()
    print("  04_latency_distribution.png")


# ── Fig 5: Output length comparison ─────────────────────────────────────────

def fig_output_length():
    models = ["qwen", "qwopus", "gemma31", "gemma31_lyf"]
    tests = ["english_critique", "japanese", "math", "coding", "design"]
    test_labels = ["English", "Japanese", "Math", "Coding", "Design"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tests))
    width = 0.2

    for i, m in enumerate(models):
        tokens = []
        for t in tests:
            row = next((r for r in summary if r["model"] == m and r["test"] == t and r["concurrency"] == 1), None)
            tokens.append(row["avg_tokens_out"] if row else 0)
        bars = ax.bar(x + i * width, tokens, width, label=LABELS[m], color=COLORS[m], alpha=0.85)

    ax.set_ylabel("Average Output Tokens", fontsize=12)
    ax.set_title("Output Length — Single Request", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(test_labels, fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/05_output_length.png", dpi=150)
    plt.close()
    print("  05_output_length.png")


# ── Fig 6: Radar chart — overall profile ─────────────────────────────────────

def fig_radar():
    models = ["qwen", "qwopus", "gemma31", "gemma31_lyf"]
    tests = ["english_critique", "japanese", "math", "coding", "design"]
    test_labels = ["English", "Japanese", "Math", "Coding", "Design"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(tests), endpoint=False).tolist()
    angles += angles[:1]

    for m in models:
        scores = []
        for t in tests:
            row = next((r for r in summary if r["model"] == m and r["test"] == t and r["concurrency"] == 1), None)
            scores.append(row["avg_quality"] if row else 0)
        scores += scores[:1]
        ax.plot(angles, scores, "o-", label=LABELS[m], color=COLORS[m], linewidth=2)
        ax.fill(angles, scores, alpha=0.1, color=COLORS[m])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(test_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Profile — Quality Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/06_radar_profile.png", dpi=150)
    plt.close()
    print("  06_radar_profile.png")


if __name__ == "__main__":
    print("Generating figures...")
    fig_quality()
    fig_speed_per_request()
    fig_throughput_scaling()
    fig_latency()
    fig_output_length()
    fig_radar()
    print("Done!")
