"""Frozen v2 comparisons: realized-return controls and public BFCL evidence."""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np

from learning import collect, evaluate_ope
from policy_contrast import disagreement_fallback, evaluate_contrast
from public_data import download_bfcl, load_bfcl, split_by_tool_family
from public_learning import (
    PublicLearner, TextFeatures, collect_public, public_choices, public_metrics, public_ope,
)
from run import metrics, save_json
from sandbox import Sandbox, make_tasks
from v2_learning import FullReturnLearner, supported_argmax


SEEDS = (7, 17, 23, 31, 47)
SYNTHETIC_SETTINGS = {
    "clean": {"scenario": "clean"},
    "noisy": {"scenario": "noisy"},
    "shifted": {"scenario": "noisy", "test_scenario": "shifted"},
    "linear": {"scenario": "noisy", "model": "linear"},
    "cost_sensitive": {"scenario": "noisy", "cost_weight": 3.0},
    "latency_sensitive": {"scenario": "noisy", "latency_weight": 0.5},
}
PUBLIC_SETTINGS = ("native", "authorization", "support_gap")


def provenance():
    root = Path(__file__).parent
    return {file_name: hashlib.sha256((root / file_name).read_bytes()).hexdigest() for file_name in (
        "v2_learning.py", "v2_experiments.py", "public_data.py", "public_learning.py", "policy_contrast.py",
        "learning.py", "sandbox.py",
    )}


def synthetic_run(setting, seed, output, quick=False):
    configuration = {"cost_weight": 1.0, "risk_weight": 2.0, "latency_weight": 0.0, "model": "trees", **SYNTHETIC_SETTINGS[setting]}
    train_size, test_size = (240, 120) if quick else (6000, 2000)
    training = collect(make_tasks(train_size, seed, "train", configuration["scenario"]), 0.3, seed + 100)
    tasks = make_tasks(test_size, seed + 2, "test", configuration.get("test_scenario", configuration["scenario"]))
    test = collect(tasks, 0.3, seed + 102)
    learner = FullReturnLearner(seed, configuration["cost_weight"], configuration["risk_weight"], configuration["latency_weight"], configuration["model"]).fit(training)
    rankings, scores = learner.score_tasks(tasks)
    choices = {name: [supported_argmax(ranking, values[name]) for ranking, values in zip(rankings, scores, strict=True)] for name in scores[0]}
    cache = {}
    results = {}
    for name, selected in choices.items():
        outcomes = []
        for task, ranking, choice in zip(tasks, rankings, selected, strict=True):
            key = (task.task_id, choice)
            if key not in cache:
                with Sandbox(task) as sandbox:
                    cache[key] = sandbox.execute(ranking.actions[choice])
            outcomes.append(cache[key])
        actual, _ = metrics(tasks, rankings, selected, outcomes, configuration["cost_weight"], configuration["risk_weight"], 100, seed, configuration["latency_weight"])
        estimates = {}
        for nuisance in ("nominal_direct", "full_direct", "component_direct"):
            nuisance_rankings = [replace(ranking, direct_values=values[nuisance]) for ranking, values in zip(rankings, scores, strict=True)]
            estimate = evaluate_ope(test, nuisance_rankings, selected, configuration["cost_weight"], configuration["risk_weight"], 100, seed, configuration["latency_weight"])
            estimates[nuisance] = estimate
        results[name] = {"actual": actual, "ope": estimates}
    summary = {"kind": "synthetic_controls", "setting": setting, "seed": seed, "configuration": configuration,
               "train_size": train_size, "test_size": test_size, "policies": results, "source_sha256": provenance()}
    output.mkdir(parents=True)
    save_json(output / "summary.json", summary)
    return summary


def paired(decisions, predictions, target, baseline):
    return evaluate_contrast(
        [decision.probabilities for decision in decisions], [decision.chosen for decision in decisions],
        [decision.reward for decision in decisions], [values["direct"] for values in predictions], target, baseline,
        lower=-1.05, upper=1.0,
    )


def public_run(tasks, provenance_data, setting, seed, output):
    split_tasks, split_manifest = split_by_tool_family(tasks, seed)
    if min(map(len, split_tasks.values())) < 10:
        raise ValueError("a predeclared split has fewer than ten tasks; report rather than change its seed")
    feature_model = TextFeatures().fit([task.observation() for task in split_tasks["train"]])
    decisions = {
        name: collect_public(values, feature_model, split_manifest["assignments"], seed, setting, seed + 100 + index)
        for index, (name, values) in enumerate(split_tasks.items())
    }
    learner = PublicLearner(seed).fit(decisions["train"])
    test_predictions = learner.predict(decisions["test"])
    calibration_predictions = learner.predict(decisions["calibration"])
    choices = {name: public_choices(decisions["test"], test_predictions, name) for name in ("tfidf", "direct", "ips", "dr")}
    probabilities = [decision.probabilities for decision in decisions["test"]]
    choices["disagreement_fallback"] = disagreement_fallback(choices["dr"], choices["tfidf"], probabilities)
    choices["blanket_support_abstain"] = [
        choice if decision.probabilities[choice] > 0 else len(decision.probabilities) - 1
        for decision, choice in zip(decisions["test"], choices["dr"], strict=True)
    ]
    cal_baseline = public_choices(decisions["calibration"], calibration_predictions, "tfidf")
    cal_target = public_choices(decisions["calibration"], calibration_predictions, "dr")
    cal_fallback = disagreement_fallback(cal_target, cal_baseline, [decision.probabilities for decision in decisions["calibration"]])
    calibration = paired(decisions["calibration"], calibration_predictions, cal_fallback, cal_baseline)
    promote = calibration["conditional_hoeffding_interval"][0] > 0
    choices["calibration_switch"] = choices["disagreement_fallback"] if promote else choices["tfidf"]
    results, realized = {}, {}
    for name, selected in choices.items():
        actual, rewards = public_metrics(split_tasks["test"], decisions["test"], selected, setting)
        realized[name] = rewards
        results[name] = {"actual": actual, "ope": public_ope(decisions["test"], test_predictions, selected)}
    contrasts = {}
    for name in ("direct", "dr", "disagreement_fallback", "blanket_support_abstain"):
        contrast = paired(decisions["test"], test_predictions, choices[name], choices["tfidf"])
        contrast["actual_paired_gain"] = float((realized[name] - realized["tfidf"]).mean())
        contrast["absolute_target_identified"] = results[name]["ope"]["identified"]
        contrast["absolute_baseline_identified"] = results["tfidf"]["ope"]["identified"]
        contrasts[name] = contrast
    output.mkdir(parents=True)
    with (output / "feedback.jsonl").open("w") as destination:
        for split, rows in decisions.items():
            for decision in rows:
                destination.write(json.dumps({
                    "task_id": decision.task_id, "group": decision.group, "split": split,
                    "chosen": decision.chosen, "reward": decision.reward, "probabilities": decision.probabilities.tolist(),
                    "allowed": decision.allowed.tolist(), "features": decision.matrix.tolist(),
                    "logger": "controlled epsilon-greedy on public labels; NOT original BFCL trajectory",
                }) + "\n")
    save_json(output / "split.json", split_manifest)
    selected_rows = [
        {"task_id": task.task_id, "choices": {name: values[index] for name, values in choices.items()}}
        for index, task in enumerate(split_tasks["test"])
    ]
    save_json(output / "decisions.json", selected_rows)
    summary = {
        "kind": "public_bfcl_reduction", "setting": setting, "seed": seed, "source": provenance_data,
        "split": {key: value for key, value in split_manifest.items() if key != "assignments"},
        "policies": results, "contrasts": contrasts, "calibration": {"promoted": promote, "contrast": calibration},
        "source_sha256": provenance(), "feedback_sha256": hashlib.sha256((output / "feedback.jsonl").read_bytes()).hexdigest(),
        "scope": "function selection only; independent BFCL questions, controlled new bandit sampling; not an official BFCL end-to-end score",
    }
    save_json(output / "summary.json", summary)
    return summary


def describe(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "runs": len(values)}


def aggregate(summaries):
    result = {"synthetic": {}, "public": {}}
    for kind, label in (("synthetic_controls", "synthetic"), ("public_bfcl_reduction", "public")):
        for setting in dict.fromkeys(summary["setting"] for summary in summaries if summary["kind"] == kind):
            rows = [summary for summary in summaries if summary["kind"] == kind and summary["setting"] == setting]
            cell = {"policies": {}}
            for policy in rows[0]["policies"]:
                metrics_to_use = ("utility", "coverage", "success_all_tasks", "unsafe_rate", "cost_per_success") if label == "synthetic" else ("utility", "coverage", "selection_accuracy", "success_all", "wrong_call_rate")
                cell["policies"][policy] = {
                    metric: describe([row["policies"][policy]["actual"][metric] for row in rows if row["policies"][policy]["actual"][metric] is not None])
                    for metric in metrics_to_use
                }
            if label == "synthetic":
                cell["ope_errors"] = {}
                for nuisance in ("nominal_direct", "full_direct", "component_direct"):
                    cell["ope_errors"][nuisance] = {}
                    for estimator in ("direct", "dr"):
                        errors = [abs(policy["ope"][nuisance][estimator] - policy["actual"]["utility"]) for row in rows for policy in row["policies"].values() if policy["ope"][nuisance]["identifiable"]]
                        cell["ope_errors"][nuisance][estimator] = describe(errors)
            else:
                cell["contrasts"] = {}
                for policy in rows[0]["contrasts"]:
                    cell["contrasts"][policy] = {
                        "point_identified_runs": sum(row["contrasts"][policy]["point_identified"] for row in rows),
                        "absolute_identified_runs": sum(row["contrasts"][policy]["absolute_target_identified"] and row["contrasts"][policy]["absolute_baseline_identified"] for row in rows),
                        "mean_identification_width": float(np.mean([row["contrasts"][policy]["identification_width"] for row in rows])),
                        "mean_shared_unsupported": float(np.mean([row["contrasts"][policy]["shared_unsupported"] for row in rows])),
                    }
                cell["promoted_runs"] = sum(row["calibration"]["promoted"] for row in rows)
                cell["test_task_instances"] = sum(row["split"]["sizes"]["test"] for row in rows)
                cell["test_groups_per_run"] = [row["split"]["group_counts"]["test"] for row in rows]
            result[label][setting] = cell
    result["run_count"] = len(summaries)
    return result


def write_tables(output, result):
    lines = ["# Version-two results", "", "All listed seeds retained; values are means, not claims of significance.", "", "## Realized-return control policy utility", "",
             "| Setting | Nominal direct | Full direct | Component direct | Nominal DR | Full DR |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, cell in result["synthetic"].items():
        values = [cell["policies"][policy]["utility"]["mean"] for policy in ("nominal_direct", "full_direct", "component_direct", "nominal_dr", "full_dr")]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "## OPE MAE for identical target-policy cases", "", "| Setting | Nominal DM | Full DM | Component DM | Nominal DR | Full DR |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, cell in result["synthetic"].items():
        errors = cell["ope_errors"]
        values = [errors[nuisance][estimator]["mean"] for nuisance, estimator in (("nominal_direct", "direct"), ("full_direct", "direct"), ("component_direct", "direct"), ("nominal_direct", "dr"), ("full_direct", "dr"))]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "## BFCL-derived held-out function selection accuracy", "", "Native has no injected fees or permissions; other rows are explicit transformations. Not official BFCL scores.", "",
              "| Setting | TF-IDF | Direct | IPS | DR | Disagreement fallback | Blanket abstain |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, cell in result["public"].items():
        values = [cell["policies"][policy]["selection_accuracy"]["mean"] * 100 for policy in ("tfidf", "direct", "ips", "dr", "disagreement_fallback", "blanket_support_abstain")]
        lines.append(f"| {name} | " + " | ".join(f"{value:.2f}%" for value in values) + " |")
    (output / "tables.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--only", choices=("all", "synthetic", "public"), default="all")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    if args.publish and args.publish.exists() and any(args.publish.iterdir()):
        parser.error("publication directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [7] if args.quick else args.seeds
    save_json(args.out / "protocol.json", {"seeds": seeds, "synthetic_settings": SYNTHETIC_SETTINGS, "public_settings": PUBLIC_SETTINGS,
                                          "quick": args.quick, "only": args.only, "source_sha256": provenance()})
    summaries = []
    if args.only in ("all", "synthetic"):
        for setting in SYNTHETIC_SETTINGS:
            for seed in seeds:
                started = time.perf_counter()
                summary = synthetic_run(setting, seed, args.out / f"synthetic-{setting}-{seed}", args.quick)
                summaries.append(summary)
                print(f"synthetic {setting:18} seed={seed} seconds={time.perf_counter()-started:.2f}", flush=True)
    if args.only in ("all", "public"):
        cache = Path(__file__).parent / "data_cache" / "bfcl"
        download_bfcl(cache)
        tasks, source = load_bfcl(cache)
        save_json(args.out / "dataset.json", source)
        for setting in PUBLIC_SETTINGS:
            for seed in seeds:
                started = time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    summary = public_run(tasks, source, setting, seed, args.out / f"public-{setting}-{seed}")
                summaries.append(summary)
                print(f"public    {setting:18} seed={seed} test={summary['split']['sizes']['test']} seconds={time.perf_counter()-started:.2f}", flush=True)
    result = aggregate(summaries)
    result["source_sha256"] = provenance()
    save_json(args.out / "aggregate.json", result)
    write_tables(args.out, result)
    if args.publish:
        args.publish.mkdir(parents=True)
        for name in ("aggregate.json", "tables.md", "protocol.json", "dataset.json"):
            if (args.out / name).exists():
                shutil.copyfile(args.out / name, args.publish / name)
        for directory in args.out.iterdir():
            if directory.is_dir():
                (args.publish / directory.name).mkdir()
                for name in ("summary.json", "split.json", "decisions.json"):
                    if (directory / name).exists():
                        shutil.copyfile(directory / name, args.publish / directory.name / name)


if __name__ == "__main__":
    main()