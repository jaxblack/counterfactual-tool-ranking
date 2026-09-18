"""Derive operating curves from the frozen per-seed CSV artifacts."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def main() -> None:
    directory = Path(__file__).resolve().parent.parent / "artifacts" / "v1"
    aggregate = json.loads((directory / "aggregate.json").read_text())
    grouped = {}
    for seed in aggregate["manifest"]["seeds"]:
        with (directory / f"noisy-seed-{seed}" / "frontier.csv").open() as source:
            for row in csv.DictReader(source):
                if row["success_on_executed"]:
                    grouped.setdefault(float(row["threshold"]), []).append({
                        metric: float(row[metric])
                        for metric in ("coverage", "success_on_executed", "success_all_tasks", "cost_per_success", "unsafe_rate")
                    })
    points = [
        {"threshold": threshold, "nonempty_seeds": len(rows), **{
            metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]
        }}
        for threshold, rows in sorted(grouped.items())
    ]
    if not points or any(not 0 <= point["coverage"] <= 1 for point in points):
        raise ValueError("invalid operating curve")
    (directory / "operating-curves.json").write_text(json.dumps(points, indent=2) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    normalization = Normalize(0, 0.20)
    coverage = [point["coverage"] * 100 for point in points]
    costs = [point["cost_per_success"] for point in points]
    for axis, metric, label in (
        (axes[0], "success_on_executed", "Success / executed tasks (%)"),
        (axes[1], "unsafe_rate", "Unsafe outcomes / all tasks (%)"),
    ):
        values = [point[metric] * 100 for point in points]
        axis.plot(coverage, values, color="#555555", linewidth=1, alpha=0.7, label="Conservative DR threshold sweep")
        scatter = axis.scatter(coverage, values, c=costs, cmap="cividis", norm=normalization, s=38)
        for policy, marker in (("schema_match", "s"), ("rules", "^"), ("direct", "D"), ("dr", "X")):
            metrics = aggregate["settings"]["noisy"]["policies"][policy]
            axis.scatter(
                metrics["coverage"]["mean"] * 100, metrics[metric]["mean"] * 100,
                c=[metrics["cost_per_success"]["mean"]], cmap="cividis", norm=normalization,
                marker=marker, s=85, edgecolors="#222222", label=policy,
            )
        axis.set(xlabel="Executed / all tasks (%)", ylabel=label, xlim=(0, 100))
        axis.grid(alpha=0.2)
    axes[0].set_ylim(0, 102)
    axes[1].set_ylim(bottom=0)
    figure.colorbar(scatter, ax=axes, label="Simulated cost per successful task", shrink=0.8)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
    figure.suptitle("Noisy workload: sampled operating points, ACL enforced at every point")
    figure.savefig(directory / "operating-curves.png", dpi=170)
    plt.close(figure)
    print(json.dumps({"curve_points": len(points), "seed_count": len(aggregate["manifest"]["seeds"])}))


if __name__ == "__main__":
    main()