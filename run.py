"""Run or replay the local tool-ranking MVP. No network calls or credentials."""

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from learning import (
    Learner, bootstrap_mean, choose, collect, evaluate_ope,
    read_log, write_log,
)
from sandbox import ABSTAIN, TOOLS, Sandbox, authorize, make_tasks


POLICIES = ("abstain", "cheapest", "schema_match", "rules", "direct", "dr", "conservative_dr")
THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0)


def save_json(file_path: Path, value: object) -> None:
    file_path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def choices_for(tasks, rankings, name, threshold=0.0, min_count=5, uncertainty_weight=1.0):
    return [
        choose(task, ranking, name, threshold, min_count, uncertainty_weight)
        for task, ranking in zip(tasks, rankings)
    ]


def calibrate(learner, decisions, min_count, bootstrap, seed):
    tasks = [decision.task for decision in decisions]
    rankings = learner.rank(tasks)
    rows = []
    for uncertainty_weight in (0.0, 0.5, 1.0):
        for threshold in (0.0, 0.5, 0.8, 0.9, 1.0):
            choices = choices_for(tasks, rankings, "conservative_dr", threshold, min_count, uncertainty_weight)
            estimate = evaluate_ope(
                decisions, rankings, choices, learner.cost_weight, learner.risk_weight, bootstrap, seed,
            )
            rows.append({
                "threshold": threshold,
                "uncertainty_weight": uncertainty_weight,
                "ope": estimate,
            })
    supported = [row for row in rows if row["ope"]["identifiable"]]
    if not supported:
        return {"threshold": 2.0, "uncertainty_weight": 1.0, "fallback": "no identifiable calibration policy"}, rows
    selected = max(supported, key=lambda row: row["ope"]["dr_ci95"][0])
    return {"threshold": selected["threshold"], "uncertainty_weight": selected["uncertainty_weight"]}, rows


def execute_choices(tasks, rankings, choices, cache):
    outcomes = []
    for task, ranking, choice in zip(tasks, rankings, choices):
        action = ranking.actions[choice]
        key = (task.task_id, action)
        if key not in cache:
            with Sandbox(task) as sandbox:
                cache[key] = sandbox.execute(action)
        outcomes.append(cache[key])
    return outcomes


def metrics(tasks, rankings, choices, outcomes, cost_weight, risk_weight, bootstrap, seed):
    total = len(tasks)
    attempted = sum(not outcome.abstained for outcome in outcomes)
    successes = sum(outcome.success for outcome in outcomes)
    feasible = sum(task.should_act for task in tasks)
    correct_abstentions = sum(outcome.abstained and not task.should_act for task, outcome in zip(tasks, outcomes))
    unnecessary_abstentions = sum(outcome.abstained and task.should_act for task, outcome in zip(tasks, outcomes))
    costs = sum(outcome.cost for outcome in outcomes)
    rewards = np.array([outcome.utility(cost_weight, risk_weight) for outcome in outcomes])
    unauthorized_executions = sum(
        not authorize(task.policy, ranking.actions[choice])[0] and not outcome.denied and not outcome.abstained
        for task, ranking, choice, outcome in zip(tasks, rankings, choices, outcomes)
    )
    return {
        "tasks": total,
        "executed": attempted,
        "successes": successes,
        "coverage": attempted / total,
        "success_on_executed": successes / attempted if attempted else None,
        "success_all_tasks": successes / total,
        "success_on_feasible": successes / feasible if feasible else None,
        "decision_accuracy": (successes + correct_abstentions) / total,
        "correct_abstention_rate": correct_abstentions / (total - feasible) if total > feasible else None,
        "unnecessary_abstention_rate": unnecessary_abstentions / feasible if feasible else None,
        "mean_cost": costs / total,
        "cost_per_success": costs / successes if successes else None,
        "mean_latency_ms": sum(outcome.latency_ms for outcome in outcomes) / total,
        "runtime_p95_ms": float(np.percentile([outcome.runtime_ms for outcome in outcomes], 95)),
        "unsafe_rate": sum(outcome.unsafe for outcome in outcomes) / total,
        "mean_excess_access": sum(outcome.excess_access for outcome in outcomes) / total,
        "denied_at_execution": sum(outcome.denied for outcome in outcomes),
        "unauthorized_executions": unauthorized_executions,
        "utility": float(rewards.mean()),
        "utility_ci95": bootstrap_mean(rewards, bootstrap, seed),
    }, rewards


def format_value(value, percent=False):
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%" if percent else f"{value:.4f}"


def plot_frontier(output, rows, policies, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    valid = [row for row in rows if row["success_on_executed"] is not None]
    axes[0].plot(
        [row["coverage"] * 100 for row in valid],
        [row["success_on_executed"] * 100 for row in valid],
        "o-", color="#167d8d", label="conservative DR threshold sweep", markersize=4,
    )
    axes[1].plot(
        [row["success_all_tasks"] * 100 for row in valid],
        [row["cost_per_success"] for row in valid],
        "o-", color="#167d8d", markersize=4,
    )
    colors = ("#b95632", "#b09226", "#365caa", "#397f46", "#804b80", "#c33d44")
    for name, color in zip(POLICIES[1:], colors):
        result = policies[name]["actual"]
        if result["success_on_executed"] is None:
            continue
        axes[0].scatter(result["coverage"] * 100, result["success_on_executed"] * 100, label=name, color=color, marker="x", s=65)
        if result["cost_per_success"] is not None:
            axes[1].scatter(result["success_all_tasks"] * 100, result["cost_per_success"], color=color, marker="x", s=65)
    axes[0].set(xlabel="Executed / all tasks (%)", ylabel="Success / executed tasks (%)", xlim=(-2, 102), ylim=(-2, 104))
    axes[1].set(xlabel="Success / all tasks (%)", ylabel="Simulated cost / successful task", xlim=(-2, 102))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"Synthetic SQLite sandbox | seed {seed} | NOT production evidence")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
    figure.savefig(output / "frontier.png", dpi=160)
    plt.close(figure)


def write_report(output, summary):
    lines = [
        "# Counterfactual Tool Ranking: Local MVP",
        "",
        "Synthetic document and ticket tasks; eight executable tools; isolated SQLite state.",
        "Costs and service latency are simulated. No LLM, live MCP server, paid API, or production data was used.",
        "",
        f"Seed: {summary['config']['seed']}. Train/calibration/test: "
        f"{summary['config']['train_size']}/{summary['config']['calibration_size']}/{summary['config']['test_size']}.",
        f"Feasible test tasks: {summary['feasible_test_tasks']}/{summary['config']['test_size']}.",
        "",
        "## Actual independent test executions",
        "",
        "| Policy | Coverage | Success/executed | Success/all | Decision accuracy | Cost/success | Unsafe/all | Utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["policies"].items():
        actual = result["actual"]
        values = [format_value(actual[key], True) for key in ("coverage", "success_on_executed", "success_all_tasks", "decision_accuracy")]
        values += [format_value(actual["cost_per_success"]), format_value(actual["unsafe_rate"], True), format_value(actual["utility"])]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += [
        "", "## Offline estimates versus actual value", "",
        "| Policy | IPS | SNIPS | DR | Actual | DR absolute error | ESS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summary["policies"].items():
        estimate = result.get("ope")
        if estimate is None:
            continue
        if not estimate["identifiable"]:
            lines.append(f"| {name} | not identified | not identified | not identified | {result['actual']['utility']:.4f} | n/a | n/a |")
        else:
            values = [format_value(estimate[key]) for key in ("ips", "snips", "dr")]
            values += [format_value(result["actual"]["utility"]), format_value(estimate["absolute_error"]), format_value(estimate["ess"])]
            lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "## Paired utility differences (test tasks)", ""]
    for comparison, result in summary["paired_differences"].items():
        lines.append(f"- {comparison}: {result['mean']:.4f}; bootstrap 95% interval {result['ci95']}.")
    lines += [
        "", "![Coverage, success and simulated cost](frontier.png)",
        "", "## Interpretation and limits", "",
        "- All executable policies share the same deterministic full-action permission filter and execution recheck.",
        "- Unauthorized execution count is measured in this toy boundary, not a proof about external services.",
        "- The rules baseline knows these eight tool semantics; oracle is evaluation-only full feedback.",
        "- Direct and DR learners only receive the selected action's outcome. No oracle labels enter training or calibration.",
        "- Calibration selects the best bootstrap DR lower endpoint from a fixed grid on separate logged tasks.",
        "- Ensemble disagreement is a heuristic, NOT a calibrated action-level confidence bound or a safety guarantee.",
        "- Intervals are task bootstrap estimates. They are not uniform guarantees or rare-event security certificates.",
        "- One decision per task, fixed structured observations, IID synthetic splits, no language understanding or multi-step claim.",
        "- Approval status is explicitly observed. A policy should not assume it knows unobserved real-world approval state.",
        "- A small deterministic environment can favor hand-written rules and direct models. Equal performance is not evidence of DR superiority.",
        "- Zero-propensity targets are reported as unidentifiable, without silently dropping contexts or renormalizing.",
        "- The support/evidence fallback is a finite feature-stratum check; generalizing it to free-form contexts is future work.",
        "", "## Artifacts", "",
        "- `train.jsonl`, `calibration.jsonl`, `test.jsonl`: partial-feedback logs with exact post-mask propensities.",
        "- `test_decisions.csv`: selected action and actual outcome for every policy/task pair.",
        "- `frontier.csv`, `frontier.png`: held-out threshold sweep, not thresholds selected on test data.",
        "- `summary.json`: configuration, package/tool versions, metrics, calibration results and paired intervals.",
        "- Replay a log with `run.py replay --log PATH` to verify outcomes against freshly initialized SQLite state.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args):
    started = time.perf_counter()
    output = args.out or Path(__file__).parent / "results" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite a nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logger = {"name": "cheap_epsilon_greedy", "epsilon": args.epsilon, "excluded_tools": args.exclude_logged_tool}
    splits = {}
    print("Collecting selected-action logs...", flush=True)
    for split_index, (name, size) in enumerate((
        ("train", args.train_size), ("calibration", args.calibration_size), ("test", args.test_size),
    )):
        task_seed = args.seed + split_index
        logging_seed = args.seed + 100 + split_index
        tasks = make_tasks(size, task_seed, name)
        decisions = collect(tasks, args.epsilon, logging_seed, tuple(args.exclude_logged_tool))
        write_log(output / f"{name}.jsonl", decisions, {**logger, "seed": logging_seed, "task_seed": task_seed})
        splits[name] = decisions
    print("Training direct and cross-fitted DR models...", flush=True)
    learner = Learner(args.seed, args.cost_weight, args.risk_weight).fit(splits["train"])
    print("Selecting conservative threshold on calibration logs only...", flush=True)
    selected, calibration = calibrate(learner, splits["calibration"], args.min_count, args.bootstrap, args.seed)
    tasks = [decision.task for decision in splits["test"]]
    rankings = learner.rank(tasks)
    choices_by_policy = {
        name: choices_for(
            tasks, rankings, name, selected["threshold"] if name == "conservative_dr" else 0.0,
            args.min_count, selected["uncertainty_weight"],
        )
        for name in POLICIES
    }
    cache = {}
    oracle_choices = []
    print("Executing frozen policies in independent resettable test sandboxes...", flush=True)
    for task, ranking in zip(tasks, rankings):
        rewards = []
        for action in ranking.actions:
            with Sandbox(task) as sandbox:
                outcome = sandbox.execute(action)
            cache[(task.task_id, action)] = outcome
            rewards.append(outcome.utility(args.cost_weight, args.risk_weight))
        oracle_choices.append(int(np.argmax(rewards)))
    choices_by_policy["oracle"] = oracle_choices
    results = {}
    policy_rewards = {}
    with (output / "test_decisions.csv").open("w", newline="", encoding="utf-8") as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=(
            "policy", "task_id", "domain", "should_act", "tool", "resources", "success",
            "abstained", "unsafe", "denied", "cost", "excess_access", "utility", "error",
        ))
        writer.writeheader()
        for name, choices in choices_by_policy.items():
            outcomes = execute_choices(tasks, rankings, choices, cache)
            actual, rewards = metrics(
                tasks, rankings, choices, outcomes, args.cost_weight, args.risk_weight, args.bootstrap, args.seed,
            )
            policy_rewards[name] = rewards
            result = {"actual": actual}
            if name != "oracle":
                estimate = evaluate_ope(
                    splits["test"], rankings, choices, args.cost_weight, args.risk_weight, args.bootstrap, args.seed,
                )
                if estimate["identifiable"]:
                    estimate["absolute_error"] = abs(estimate["dr"] - actual["utility"])
                result["ope"] = estimate
            results[name] = result
            for task, ranking, choice, outcome, reward in zip(tasks, rankings, choices, outcomes, rewards):
                action = ranking.actions[choice]
                writer.writerow({
                    "policy": name, "task_id": task.task_id, "domain": task.domain,
                    "should_act": task.should_act, "tool": action.tool, "resources": json.dumps(action.resources),
                    "success": outcome.success, "abstained": outcome.abstained,
                    "unsafe": outcome.unsafe, "denied": outcome.denied, "cost": outcome.cost,
                    "excess_access": outcome.excess_access, "utility": float(reward), "error": outcome.error,
                })
    frontier = []
    for threshold in THRESHOLDS:
        choices = choices_for(tasks, rankings, "conservative_dr", threshold, args.min_count, selected["uncertainty_weight"])
        outcomes = execute_choices(tasks, rankings, choices, cache)
        actual, _ = metrics(tasks, rankings, choices, outcomes, args.cost_weight, args.risk_weight, args.bootstrap, args.seed)
        frontier.append({"threshold": threshold, **{key: value for key, value in actual.items() if key != "utility_ci95"}})
    with (output / "frontier.csv").open("w", newline="", encoding="utf-8") as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=list(frontier[0]))
        writer.writeheader()
        writer.writerows(frontier)
    differences = {}
    for baseline in ("schema_match", "rules", "direct", "dr"):
        differences[f"conservative_dr - {baseline}"] = {
            "mean": float((policy_rewards["conservative_dr"] - policy_rewards[baseline]).mean()),
            "ci95": bootstrap_mean(policy_rewards["conservative_dr"] - policy_rewards[baseline], args.bootstrap, args.seed),
        }
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version(), **{package: version(package) for package in ("numpy", "scikit-learn", "matplotlib")}},
        "tools": [asdict(tool) for tool in TOOLS.values()],
        "feature_names": list(learner.vectorizer.get_feature_names_out()),
        "feasible_test_tasks": sum(task.should_act for task in tasks),
        "selected_calibration": selected,
        "calibration_grid": calibration,
        "policies": results,
        "paired_differences": differences,
    }
    plot_frontier(output, frontier, results, args.seed)
    summary["elapsed_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    write_report(output, summary)
    print("\nPolicy              Coverage  Success/all  Cost/success  Unsafe/all  Utility")
    for name, result in results.items():
        actual = result["actual"]
        print(f"{name:20} {actual['coverage']:7.1%} {actual['success_all_tasks']:11.1%} "
              f"{format_value(actual['cost_per_success']):>13} {actual['unsafe_rate']:11.1%} {actual['utility']:8.4f}")
    print(f"\nCompleted in {summary['elapsed_seconds']:.2f}s. Report: {output / 'report.md'}")
    return summary


def replay(file_path: Path) -> int:
    decisions = read_log(file_path)
    mismatches = []
    for decision in decisions:
        with Sandbox(decision.task) as sandbox:
            outcome = sandbox.execute(decision.actions[decision.chosen])
        if replace(outcome, runtime_ms=0) != replace(decision.outcome, runtime_ms=0):
            mismatches.append(decision.task.task_id)
    print(json.dumps({"checked": len(decisions), "mismatches": len(mismatches), "examples": mismatches[:10]}))
    return int(bool(mismatches))


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="collect, learn, calibrate, execute and report")
    run.add_argument("--train-size", type=int, default=6000)
    run.add_argument("--calibration-size", type=int, default=1200)
    run.add_argument("--test-size", type=int, default=2000)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--epsilon", type=float, default=0.3)
    run.add_argument("--cost-weight", type=float, default=1.0)
    run.add_argument("--risk-weight", type=float, default=2.0)
    run.add_argument("--min-count", type=int, default=5)
    run.add_argument("--bootstrap", type=int, default=200)
    run.add_argument("--exclude-logged-tool", action="append", default=[], choices=tuple(TOOLS))
    run.add_argument("--out", type=Path)
    replay_parser = subparsers.add_parser("replay", help="verify logged results against fresh SQLite state")
    replay_parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args(arguments)
    if args.command == "run":
        if args.train_size < 12 or min(args.calibration_size, args.test_size, args.bootstrap, args.min_count) < 1:
            parser.error("train-size must be >= 12; other sizes, min-count and bootstrap must be positive")
        if (
            not all(np.isfinite(value) for value in (args.epsilon, args.cost_weight, args.risk_weight))
            or not 0 <= args.epsilon <= 1
            or min(args.cost_weight, args.risk_weight) < 0
        ):
            parser.error("epsilon must be in [0, 1] and penalty weights must be nonnegative")
        if not 0 <= args.seed < 2**32 - 1000:
            parser.error("seed must be in [0, 2**32 - 1000)")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.command == "replay":
            return replay(args.log)
        run_experiment(args)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())