"""Figures from frozen v2 controls and public benchmark diagnostics."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).resolve().parent.parent / "artifacts" / "v2"
    aggregate = json.loads((directory / "aggregate.json").read_text())
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    settings = ("noisy", "shifted", "linear", "cost_sensitive", "latency_sensitive")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    positions = np.arange(len(settings))
    for label, nuisance, estimator, color in (
        ("Nominal direct", "nominal_direct", "direct", "#a06b27"),
        ("Full-return direct", "full_direct", "direct", "#246bad"),
        ("Full-return DR", "full_direct", "dr", "#167e70"),
    ):
        values = [aggregate["synthetic"][name]["ope_errors"][nuisance][estimator]["mean"] for name in settings]
        axes[0].plot(positions, values, "o-", color=color, label=label, markersize=4)
    axes[0].set_xticks(positions, [name.replace("_", " ") for name in settings], rotation=20, ha="right")
    axes[0].set_ylabel("OPE MAE (identical target policies)")
    axes[0].set_title("Fair realized-return controls")
    axes[0].legend(fontsize=8)
    policies = ("always_abstain", "tfidf", "direct", "ips", "dr")
    balanced = [diagnostics["native_means"][name]["balanced_accuracy"] * 100 for name in policies]
    axes[1].bar(np.arange(len(policies)), balanced, color=["#8b9398", "#a06b27", "#246bad", "#aa435b", "#167e70"])
    axes[1].set_xticks(np.arange(len(policies)), [name.replace("_", " ") for name in policies], rotation=20, ha="right")
    axes[1].set(ylabel="Balanced function-selection accuracy (%)", ylim=(0, 100), title="BFCL-derived, held-out function groups")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(directory / "controls-and-public.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for index, name in enumerate(("qwen-0.5b", "qwen-1.5b")):
        model = diagnostics["models"][name]
        recall = [model["per_category"][category]["correct"] / model["per_category"][category]["tasks"] * 100 for category in ("multiple", "irrelevance")]
        axes[0].bar(np.arange(2) + (index - 0.5) * 0.3, recall, width=0.3, label=name, color=("#246bad", "#167e70")[index])
    model = diagnostics["models"]["qwen-0.5b"]
    axes[0].set_xticks([0, 1], [
        f"Should call ({model['per_category']['multiple']['tasks']} tasks)",
        f"Should abstain ({model['per_category']['irrelevance']['tasks']} tasks)",
    ])
    axes[0].set(ylabel="Correct function selection / class (%)", ylim=(0, 100), title="Real local LLMs: same 200 public tasks")
    axes[0].legend()
    names = ("dr", "blanket_support_abstain", "disagreement_fallback")
    contrasts = aggregate["public"]["support_gap"]["contrasts"]
    widths = [contrasts[name]["mean_identification_width"] for name in names]
    axes[1].bar(np.arange(3), widths, color=["#167e70", "#a06b27", "#246bad"])
    axes[1].set_xticks(np.arange(3), ["DR vs baseline", "Blanket abstain\nvs baseline", "Disagreement fallback\nvs baseline"])
    axes[1].set(ylabel="Mean identification interval width", title="Support gap: absolute values unidentified")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(directory / "llm-and-contrast.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()