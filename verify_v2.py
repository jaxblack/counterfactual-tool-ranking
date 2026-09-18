"""Recompute v2 public results and provenance without raw private data or inference."""

import hashlib
import json
from pathlib import Path

from v2_experiments import PUBLIC_SETTINGS, SEEDS, SYNTHETIC_SETTINGS, aggregate
from verify_artifacts import same_statistics


def verify(root):
    directory = root / "artifacts" / "v2"
    protocol = json.loads((directory / "protocol.json").read_text())
    dataset = json.loads((directory / "dataset.json").read_text())
    if not dataset["accounting"].get("full_role_context_preserved"):
        raise ValueError("public artifact omits source message context")
    if protocol["quick"] or protocol["seeds"] != list(SEEDS) or protocol["synthetic_settings"] != SYNTHETIC_SETTINGS or protocol["public_settings"] != list(PUBLIC_SETTINGS):
        raise ValueError("v2 artifact does not match the frozen full protocol")
    source_count = 0
    for name, digest in protocol["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"v2 experimental source changed: {name}")
        source_count += 1
    summaries = []
    for label, settings in (("synthetic", SYNTHETIC_SETTINGS), ("public", PUBLIC_SETTINGS)):
        for setting in settings:
            for seed in SEEDS:
                run_dir = directory / f"{label}-{setting}-{seed}"
                summary = json.loads((run_dir / "summary.json").read_text())
                if summary["source_sha256"] != protocol["source_sha256"]:
                    raise ValueError("run source differs from protocol")
                summaries.append(summary)
                if label == "public":
                    split = json.loads((run_dir / "split.json").read_text())
                    assignments = split["assignments"]
                    component_splits = {}
                    for item in assignments.values():
                        component_splits.setdefault(item["component"], set()).add(item["split"])
                    if any(len(values) != 1 for values in component_splits.values()):
                        raise ValueError("a tool component crosses a data split")
                    selected = json.loads((run_dir / "decisions.json").read_text())
                    if any(assignments[row["task_id"]]["split"] != "test" for row in selected):
                        raise ValueError("published evaluation choices include training tasks")
                    if len(selected) != split["sizes"]["test"]:
                        raise ValueError("test accounting mismatch")
    published = json.loads((directory / "aggregate.json").read_text())
    rebuilt = aggregate(summaries)
    if not same_statistics(rebuilt, {key: value for key, value in published.items() if key != "source_sha256"}):
        raise ValueError("v2 aggregate differs from per-run statistics")
    model_ids = []
    model_runs = []
    for name in ("qwen-0.5b", "qwen-1.5b"):
        summary = json.loads((directory / f"llm-{name}.json").read_text())
        if summary["smoke_training_only"] or summary["tasks"] != 200:
            raise ValueError("unexpected LLM evaluation size or training-only run")
        if summary["source_sha256"] != hashlib.sha256((root / "public_llm.py").read_bytes()).hexdigest():
            raise ValueError("LLM evaluation source changed")
        if summary["input_adapter_sha256"] != protocol["source_sha256"]["public_data.py"]:
            raise ValueError("LLM inputs used a different public-data adapter")
        model_ids.append({row["task_id"] for row in summary["results"]})
        if len(model_ids[-1]) != 200:
            raise ValueError("repeated LLM test task")
        for row in summary["results"]:
            for seed in row["heldout_seeds"]:
                split = json.loads((directory / f"public-native-{seed}" / "split.json").read_text())
                if split["assignments"][row["task_id"]]["split"] != "test":
                    raise ValueError("LLM comparison is not held out for the stated seed")
        for metric, value in summary["overall"].items():
            expected = sum(row["score"][metric] for row in summary["results"]) / 200
            if not same_statistics(expected, value):
                raise ValueError("LLM aggregate differs from recorded outputs")
        model_runs.append({"model": name, "tasks": 200})
    if model_ids[0] != model_ids[1]:
        raise ValueError("LLM baselines were evaluated on different task IDs")
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    if diagnostics["source_sha256"] != hashlib.sha256((root / "analyze_v2.py").read_bytes()).hexdigest():
        raise ValueError("post-run diagnostic source changed")
    for policy, values in diagnostics["native_means"].items():
        if policy != "always_abstain" and not same_statistics(
            values["accuracy"], published["public"]["native"]["policies"][policy]["selection_accuracy"]["mean"],
        ):
            raise ValueError("diagnostic accuracy differs from published public experiment")
        for metric in ("accuracy", "balanced_accuracy", "coverage"):
            expected = sum(diagnostics["per_seed"][str(seed)]["all_test_tasks"][policy][metric] for seed in SEEDS) / len(SEEDS)
            if not same_statistics(expected, values[metric]):
                raise ValueError("diagnostic mean differs from per-seed data")
    for policy, values in diagnostics["multi_candidate_means"].items():
        for metric in ("accuracy", "balanced_accuracy", "coverage"):
            expected = sum(diagnostics["per_seed"][str(seed)]["multi_candidate_only"][policy][metric] for seed in SEEDS) / len(SEEDS)
            if not same_statistics(expected, values[metric]):
                raise ValueError("candidate-count cohort mean differs from per-seed data")
    claims = json.loads((root / "paper" / "claims.json").read_text())
    expected_claims = {
        "v2_linear_full_dm": f"{published['synthetic']['linear']['ope_errors']['full_direct']['direct']['mean']:.4f}",
        "v2_linear_full_dr": f"{published['synthetic']['linear']['ope_errors']['full_direct']['dr']['mean']:.4f}",
        "v2_shift_full_dm": f"{published['synthetic']['shifted']['ope_errors']['full_direct']['direct']['mean']:.4f}",
        "v2_shift_full_dr": f"{published['synthetic']['shifted']['ope_errors']['full_direct']['dr']['mean']:.4f}",
        "v2_public_direct_balanced": f"{diagnostics['native_means']['direct']['balanced_accuracy'] * 100:.2f}",
        "v2_public_dr_balanced": f"{diagnostics['native_means']['dr']['balanced_accuracy'] * 100:.2f}",
        "v2_public_tfidf_balanced": f"{diagnostics['native_means']['tfidf']['balanced_accuracy'] * 100:.2f}",
        "v2_multi_direct_balanced": f"{diagnostics['multi_candidate_means']['direct']['balanced_accuracy'] * 100:.2f}",
        "v2_multi_tfidf_balanced": f"{diagnostics['multi_candidate_means']['tfidf']['balanced_accuracy'] * 100:.2f}",
    }
    for key, expected in expected_claims.items():
        if claims[key] != expected:
            raise ValueError(f"v2 paper claim differs from evidence: {key}")
    if any(directory.rglob("*.jsonl")):
        raise ValueError("raw public prompts or model responses were unexpectedly published")
    return {"runs": len(summaries), "source_hash_checks": source_count, "models": model_runs,
            "split_group_check": "passed", "aggregate_check": "passed", "paper_claim_checks": len(expected_claims), "v1_modified": False}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent), indent=2))