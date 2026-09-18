"""Frozen multi-seed experiment matrix and publication-safe aggregate artifacts."""

import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np

from run import POLICIES, parse_args, run_experiment, save_json


SETTINGS = {
    "clean": [],
    "noisy": ["--scenario", "noisy"],
    "small_data": ["--scenario", "noisy", "--train-size", "600"],
    "low_exploration": ["--scenario", "noisy", "--epsilon", "0.05"],
    "missing_support": [
        "--scenario", "noisy", "--exclude-logged-tool", "docs.batch_get",
        "--exclude-logged-tool", "tickets.close_checked", "--exclude-logged-tool", "crm.discount_checked",
    ],
    "shifted": ["--scenario", "noisy", "--test-scenario", "shifted"],
    "linear": ["--scenario", "noisy", "--model", "linear"],
    "cost_sensitive": ["--scenario", "noisy", "--cost-weight", "3"],
    "latency_sensitive": ["--scenario", "noisy", "--latency-weight", "0.5"],
}
DEFAULT_SEEDS = (7, 17, 23, 31, 47)
CORE_POLICIES = ("schema_match", "rules", "direct", "ips", "dr", "conservative_direct", "conservative_dr")


def descriptive(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "count": len(array),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    cells = {}
    for setting in dict.fromkeys(run["setting"] for run in runs):
        matching = [run for run in runs if run["setting"] == setting]
        policies = {}
        for policy in (*POLICIES, "oracle"):
            results = [run["summary"]["policies"][policy] for run in matching]
            metrics = {}
            for name in (
                "utility", "coverage", "success_all_tasks", "success_on_executed", "decision_accuracy",
                "cost_per_success", "unsafe_rate", "mean_latency_ms", "mean_excess_access",
            ):
                values = [result["actual"][name] for result in results if result["actual"][name] is not None]
                metrics[name] = descriptive(values) if values else None
            estimates = [result["ope"] for result in results if result.get("ope", {}).get("identifiable")]
            metrics["identified_runs"] = len(estimates)
            metrics["unauthorized_executions"] = sum(result["actual"]["unauthorized_executions"] for result in results)
            if estimates:
                metrics["ess"] = descriptive([estimate["ess"] for estimate in estimates])
                metrics["dr_absolute_error"] = descriptive([estimate["absolute_error"] for estimate in estimates])
                metrics["dr_interval_inclusion"] = sum(
                    result["ope"]["dr_ci95"][0] <= result["actual"]["utility"] <= result["ope"]["dr_ci95"][1]
                    for result in results if result.get("ope", {}).get("identifiable")
                ) / len(estimates)
            policies[policy] = metrics
        errors = {}
        for estimator in ("direct", "ips", "snips", "dr"):
            values = [
                abs(run["summary"]["policies"][policy]["ope"][estimator] - run["summary"]["policies"][policy]["actual"]["utility"])
                for run in matching for policy in CORE_POLICIES
                if run["summary"]["policies"][policy]["ope"]["identifiable"]
                and run["summary"]["policies"][policy]["ope"][estimator] is not None
            ]
            errors[estimator] = descriptive(values) if values else None
        differences = {}
        for first, second in (("dr", "direct"), ("dr", "ips"), ("conservative_dr", "conservative_direct"), ("direct", "rules")):
            differences[f"{first}-minus-{second}"] = descriptive([
                run["summary"]["policies"][first]["actual"]["utility"] - run["summary"]["policies"][second]["actual"]["utility"]
                for run in matching
            ])
        cells[setting] = {"policies": policies, "ope_absolute_errors": errors, "paired_seed_differences": differences}
    return {"settings": cells, "run_count": len(runs)}


def write_tables(output: Path, aggregate: dict) -> None:
    lines = [
        "# Frozen experiment results", "",
        "Mean +/- sample standard deviation across seeds. These are descriptive, not significance claims.", "",
        "| Setting | Rules utility | Direct | IPS | DR | Conservative direct | Conservative DR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting, cell in aggregate["settings"].items():
        values = [cell["policies"][policy]["utility"] for policy in ("rules", "direct", "ips", "dr", "conservative_direct", "conservative_dr")]
        lines.append(f"| {setting} | " + " | ".join(f"{value['mean']:.4f} +/- {value['std']:.4f}" for value in values) + " |")
    lines += [
        "", "## OPE mean absolute error", "",
        "Computed over the same identified target policies within a setting; abstain and hindsight oracle are excluded.", "",
        "| Setting | Direct | IPS | SNIPS | DR | Identified policy/seed cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting, cell in aggregate["settings"].items():
        estimates = cell["ope_absolute_errors"]
        values = [f"{estimates[estimator]['mean']:.4f}" if estimates[estimator] else "n/a" for estimator in ("direct", "ips", "snips", "dr")]
        count = estimates["dr"]["count"] if estimates["dr"] else 0
        lines.append(f"| {setting} | " + " | ".join(values) + f" | {count} |")
    (output / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("setting", "policy", "metric", "mean", "std", "seeds"))
        for setting, cell in aggregate["settings"].items():
            for policy, metrics in cell["policies"].items():
                for name, value in metrics.items():
                    if isinstance(value, dict) and "mean" in value:
                        writer.writerow((setting, policy, name, value["mean"], value["std"], value["count"]))


def write_figures(output: Path, aggregate: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = list(aggregate["settings"])
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    styles = (
        ("rules", "#777777"), ("direct", "#236dab"), ("ips", "#c28622"),
        ("dr", "#167e70"), ("conservative_dr", "#ab4054"),
    )
    positions = np.arange(len(settings))
    for policy_index, (policy, color) in enumerate(styles):
        for axis, metric, scale in ((axes[0], "utility", 1), (axes[1], "unsafe_rate", 100)):
            values = [aggregate["settings"][setting]["policies"][policy][metric] for setting in settings]
            axis.errorbar(
                positions + (policy_index - 2) * 0.12,
                [value["mean"] * scale for value in values],
                yerr=[value["std"] * scale for value in values],
                fmt="o", capsize=3, color=color, label=policy, markersize=4,
            )
            axis.set_xticks(positions, [setting.replace("_", " ") for setting in settings], rotation=20, ha="right")
            axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean utility (simulated cost units)")
    axes[1].set_ylabel("Unsafe outcomes / all tasks (%)")
    axes[0].legend(ncol=5, fontsize=8)
    figure.suptitle("Frozen settings: mean and standard deviation across seeds")
    figure.savefig(output / "policy-comparison.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 4.5), layout="constrained")
    for estimator, color in (("direct", "#236dab"), ("ips", "#c28622"), ("snips", "#ab4054"), ("dr", "#167e70")):
        values = [aggregate["settings"][setting]["ope_absolute_errors"][estimator] for setting in settings]
        axis.plot(positions, [value["mean"] if value else np.nan for value in values], "o-", label=estimator, color=color, markersize=4)
    axis.set_xticks(positions, [setting.replace("_", " ") for setting in settings], rotation=20, ha="right")
    axis.set_ylabel("OPE mean absolute error (identified targets only)")
    axis.legend(ncol=4)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "ope-comparison.png", dpi=170)
    plt.close(figure)


def run_suite(args) -> dict:
    output = args.out
    if output.exists() and any(output.iterdir()):
        raise ValueError("suite output must be empty; existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    seeds = [7] if args.quick else args.seeds
    settings = ["clean", "noisy", "missing_support"] if args.quick else args.settings
    manifest = {"schema_version": 1, "seeds": seeds, "settings": {name: SETTINGS[name] for name in settings}, "quick": args.quick}
    save_json(output / "manifest.json", manifest)
    runs = []
    for setting in settings:
        for seed in seeds:
            run_dir = output / f"{setting}-seed-{seed}"
            arguments = ["run", "--seed", str(seed), "--out", str(run_dir), *SETTINGS[setting]]
            if args.quick:
                arguments += ["--train-size", "120", "--calibration-size", "40", "--test-size", "40", "--bootstrap", "20"]
            started = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                summary = run_experiment(parse_args(arguments))
            runs.append({"setting": setting, "seed": seed, "summary": summary})
            print(f"{setting:20} seed={seed:2} direct={summary['policies']['direct']['actual']['utility']:.4f} "
                  f"dr={summary['policies']['dr']['actual']['utility']:.4f} seconds={time.perf_counter()-started:.2f}", flush=True)
    aggregate = aggregate_runs(runs)
    aggregate["manifest"] = manifest
    aggregate["runs"] = [
        {"setting": run["setting"], "seed": run["seed"], "summary_path": f"{run['setting']}-seed-{run['seed']}/summary.json"}
        for run in runs
    ]
    aggregate["selected_decision_count"] = sum(
        sum(run["summary"]["config"][key] for key in ("train_size", "calibration_size", "test_size")) for run in runs
    )
    aggregate["test_task_condition_count"] = sum(run["summary"]["config"]["test_size"] for run in runs)
    save_json(output / "aggregate.json", aggregate)
    write_tables(output, aggregate)
    write_figures(output, aggregate)
    if args.publish:
        publish_artifacts(output, args.publish, runs)
    return aggregate


def publish_artifacts(source: Path, destination: Path, runs: list[dict]) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("publication destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("aggregate.json", "manifest.json", "tables.md", "metrics.csv", "policy-comparison.png", "ope-comparison.png"):
        shutil.copyfile(source / name, destination / name)
    hashes = {}
    for run in runs:
        run_name = f"{run['setting']}-seed-{run['seed']}"
        run_source = source / run_name
        run_destination = destination / run_name
        run_destination.mkdir()
        summary = json.loads((run_source / "summary.json").read_text())
        summary["config"]["out"] = run_name
        save_json(run_destination / "summary.json", summary)
        shutil.copyfile(run_source / "frontier.csv", run_destination / "frontier.csv")
        for log_name in ("train.jsonl", "calibration.jsonl", "test.jsonl"):
            hashes[f"{run_name}/{log_name}"] = hashlib.sha256((run_source / log_name).read_bytes()).hexdigest()
    save_json(destination / "raw-log-hashes.json", hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--settings", choices=tuple(SETTINGS), nargs="+", default=list(SETTINGS))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or len(args.settings) != len(set(args.settings)):
        parser.error("seeds and settings must be unique")
    run_suite(args)


if __name__ == "__main__":
    main()