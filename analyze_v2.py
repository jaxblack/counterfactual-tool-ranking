"""Post-run diagnostics on fixed public tasks; no model refitting or selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from public_data import load_bfcl
from public_learning import actual_outcome, allowed_mask
from run import save_json
from v2_experiments import SEEDS


def group_bootstrap(values, groups, seed, repetitions=2000):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = sorted(set(groups))
    sums = np.array([values[groups == group].sum() for group in unique])
    sizes = np.array([np.count_nonzero(groups == group) for group in unique])
    generator = np.random.default_rng(seed)
    sampled = generator.integers(len(unique), size=(repetitions, len(unique)))
    means = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def stratified_metrics(tasks, correct, executed, groups, seed):
    correct, executed = np.asarray(correct, dtype=bool), np.asarray(executed, dtype=bool)
    result = {"tasks": len(tasks), "groups": len(set(groups)), "accuracy": float(correct.mean()),
              "coverage": float(executed.mean()), "accuracy_group_bootstrap_ci95": group_bootstrap(correct, groups, seed)}
    recalls = []
    for category in ("multiple", "irrelevance"):
        mask = np.array([task.category == category for task in tasks])
        result[category] = {"tasks": int(mask.sum()), "correct": int(correct[mask].sum()), "accuracy": float(correct[mask].mean()) if mask.any() else None}
        if mask.any():
            recalls.append(float(correct[mask].mean()))
    result["balanced_accuracy"] = float(np.mean(recalls)) if len(recalls) == 2 else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--llm-small", type=Path, default=Path("results/llm-v2-full-context-0.5b/summary.json"))
    parser.add_argument("--llm-large", type=Path, default=Path("results/llm-v2-full-context-1.5b/summary.json"))
    args = parser.parse_args()
    root = Path(__file__).parent
    tasks, _ = load_bfcl(root / "data_cache/bfcl")
    indexed = {task.task_id: task for task in tasks}
    models = {name: json.loads(file_path.read_text()) for name, file_path in (("qwen-0.5b", args.llm_small), ("qwen-1.5b", args.llm_large))}
    result_sets = [{row["task_id"] for row in model["results"]} for model in models.values()]
    if result_sets[0] != result_sets[1]:
        raise ValueError("local models must be compared on identical task IDs")
    analysis = {"per_seed": {}, "models": {}, "scope": "post-run stratification; no retraining or threshold changes"}
    for name, model in models.items():
        save_json(args.artifacts / f"llm-{name}.json", model)
        rows = model["results"]
        local_tasks = [indexed[row["task_id"]] for row in rows]
        analysis["models"][name] = {
            "model": model["model"], "revision": model["revision"], "tasks": len(rows),
            "selection_accuracy": model["overall"]["selection_correct"],
            "format_validity": model["overall"]["format_valid"], "abstention_rate": model["overall"]["abstained"],
            "schema_validity_among_known_calls": model["argument_schema_valid_among_known_calls"],
            "per_category": {
                category: {"tasks": sum(task.category == category for task in local_tasks),
                           "correct": sum(task.category == category and row["score"]["selection_correct"] for task, row in zip(local_tasks, rows, strict=True))}
                for category in ("multiple", "irrelevance")
            },
            "generated_tokens": model["generated_tokens"], "prompt_tokens": model["prompt_tokens"],
            "total_generation_seconds": model["total_generation_seconds"],
        }
    for seed in SEEDS:
        directory = args.artifacts / f"public-native-{seed}"
        decisions = json.loads((directory / "decisions.json").read_text())
        assignments = json.loads((directory / "split.json").read_text())["assignments"]
        groups = [assignments[row["task_id"]]["component"] for row in decisions]
        test_tasks = [indexed[row["task_id"]] for row in decisions]
        metrics = {}
        correct_vectors = {}
        for policy in (*decisions[0]["choices"], "always_abstain"):
            choices = [row["choices"][policy] if policy != "always_abstain" else len(task.tools) for row, task in zip(decisions, test_tasks, strict=True)]
            outcomes = [actual_outcome(task, choice, allowed_mask(task, "native", seed), "native") for task, choice in zip(test_tasks, choices, strict=True)]
            correct = [outcome["correct"] for outcome in outcomes]
            correct_vectors[policy] = np.asarray(correct, dtype=float)
            metrics[policy] = stratified_metrics(test_tasks, correct, [not outcome["abstained"] for outcome in outcomes], groups, seed)
        multiple_mask = np.array([len(task.tools) >= 2 for task in test_tasks])
        multiple_tasks = [task for task, include in zip(test_tasks, multiple_mask, strict=True) if include]
        multiple_groups = np.asarray(groups)[multiple_mask].tolist()
        multiple_metrics = {}
        for policy in ("always_abstain", "tfidf", "direct", "ips", "dr"):
            executed = [
                policy != "always_abstain" and row["choices"][policy] != len(task.tools)
                for row, task, include in zip(decisions, test_tasks, multiple_mask, strict=True) if include
            ]
            multiple_metrics[policy] = stratified_metrics(multiple_tasks, correct_vectors[policy][multiple_mask], executed, multiple_groups, seed)
        matched = {}
        for model_name, model in models.items():
            llm_index = {row["task_id"]: row for row in model["results"]}
            mask = np.array([task.task_id in llm_index for task in test_tasks])
            matched_tasks = [task for task, included in zip(test_tasks, mask, strict=True) if included]
            matched_groups = np.asarray(groups)[mask].tolist()
            model_correct = np.array([llm_index[task.task_id]["score"]["selection_correct"] for task in matched_tasks])
            model_executed = [not llm_index[task.task_id]["score"]["abstained"] for task in matched_tasks]
            comparisons = {"llm": stratified_metrics(matched_tasks, model_correct, model_executed, matched_groups, seed)}
            for policy in ("tfidf", "direct", "dr", "always_abstain"):
                choices = [row["choices"].get(policy, len(task.tools)) for row, task, included in zip(decisions, test_tasks, mask, strict=True) if included]
                comparisons[policy] = stratified_metrics(
                    matched_tasks, correct_vectors[policy][mask], [choice != len(task.tools) for choice, task in zip(choices, matched_tasks, strict=True)], matched_groups, seed,
                )
                differences = correct_vectors[policy][mask] - model_correct.astype(float)
                comparisons[policy]["paired_accuracy_difference_vs_llm"] = float(differences.mean())
                comparisons[policy]["paired_group_bootstrap_ci95"] = group_bootstrap(differences, matched_groups, seed)
            matched[model_name] = comparisons
        analysis["per_seed"][str(seed)] = {"all_test_tasks": metrics, "multi_candidate_only": multiple_metrics, "matched_models": matched}
    policies = ("always_abstain", "tfidf", "direct", "ips", "dr")
    analysis["native_means"] = {
        policy: {metric: float(np.mean([analysis["per_seed"][str(seed)]["all_test_tasks"][policy][metric] for seed in SEEDS]))
                 for metric in ("accuracy", "balanced_accuracy", "coverage")}
        for policy in policies
    }
    analysis["matched_means"] = {
        model: {policy: {metric: float(np.mean([analysis["per_seed"][str(seed)]["matched_models"][model][policy][metric] for seed in SEEDS]))
                         for metric in ("accuracy", "balanced_accuracy", "coverage")}
                for policy in ("llm", "always_abstain", "tfidf", "direct", "dr")}
        for model in models
    }
    analysis["multi_candidate_means"] = {
        policy: {metric: float(np.mean([analysis["per_seed"][str(seed)]["multi_candidate_only"][policy][metric] for seed in SEEDS]))
                 for metric in ("accuracy", "balanced_accuracy", "coverage")}
        for policy in policies
    }
    analysis["candidate_count_audit"] = {
        "single_candidate_tasks": sum(len(task.tools) == 1 for task in tasks),
        "single_candidate_should_call": sum(len(task.tools) == 1 and task.category == "multiple" for task in tasks),
        "interpretation": "single-candidate rows are all irrelevance examples in this selected BFCL subset; report a multi-candidate cohort without retraining",
    }
    analysis["input_sha256"] = {
        name: hashlib.sha256(file_path.read_bytes()).hexdigest()
        for name, file_path in (("qwen-0.5b", args.llm_small), ("qwen-1.5b", args.llm_large))
    }
    analysis["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_json(args.artifacts / "diagnostics.json", analysis)
    lines = ["# Public-data diagnostics", "", "No refitting or threshold selection was performed for these diagnostics.", "",
             "## Native BFCL-derived selection, mean across fixed splits", "",
             "| Policy | Accuracy | Balanced accuracy | Coverage |", "| --- | ---: | ---: | ---: |"]
    for policy, values in analysis["native_means"].items():
        lines.append(f"| {policy} | " + " | ".join(f"{values[metric]*100:.2f}%" for metric in ("accuracy", "balanced_accuracy", "coverage")) + " |")
    lines += ["", "## Post-run candidate-count shortcut diagnostic", "",
              "All 508 single-candidate rows are should-abstain tasks. The following cohort contains only test tasks with at least two candidates; no model was retrained.", "",
              "| Policy | Accuracy | Balanced accuracy | Coverage |", "| --- | ---: | ---: | ---: |"]
    for policy, values in analysis["multi_candidate_means"].items():
        lines.append(f"| {policy} | " + " | ".join(f"{values[metric]*100:.2f}%" for metric in ("accuracy", "balanced_accuracy", "coverage")) + " |")
    for model_name, comparisons in analysis["matched_means"].items():
        lines += ["", f"## Matched task subset: {model_name}", "", "All compared methods use exactly the same held-out tasks within each seed. Task IDs can repeat across seeds.", "",
                  "| Policy | Accuracy | Balanced accuracy | Coverage |", "| --- | ---: | ---: | ---: |"]
        for policy, values in comparisons.items():
            lines.append(f"| {policy} | " + " | ".join(f"{values[metric]*100:.2f}%" for metric in ("accuracy", "balanced_accuracy", "coverage")) + " |")
    lines += ["", "Intervals in diagnostics.json resample connected tool/query groups, not individual rows.",
              "LLMs also generated parameters; only exact function selection is compared here. Small-model results do not establish frontier-model superiority."]
    (args.artifacts / "diagnostics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"native": analysis["native_means"], "models": analysis["models"]}, indent=2))


if __name__ == "__main__":
    main()