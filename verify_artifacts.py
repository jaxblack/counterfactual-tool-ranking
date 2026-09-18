"""Verify published statistics and source provenance without re-running the suite."""

import hashlib
import json
import math
from pathlib import Path

from suite import SETTINGS, aggregate_runs


def same_statistics(first, second) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(same_statistics(value, second[key]) for key, value in first.items())
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
    return first == second


def verify(root: Path) -> dict:
    directory = root / "artifacts" / "v1"
    aggregate = json.loads((directory / "aggregate.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {(setting, seed) for setting in manifest["settings"] for seed in manifest["seeds"]}
    actual = {(run["setting"], run["seed"]) for run in aggregate["runs"]}
    if actual != expected or len(aggregate["runs"]) != len(expected):
        raise ValueError("missing or duplicate experiment cells")
    if manifest["settings"] != {name: SETTINGS[name] for name in manifest["settings"]}:
        raise ValueError("published experiment protocol differs from source")
    runs = []
    source_checks = 0
    for run in aggregate["runs"]:
        summary = json.loads((directory / run["summary_path"]).read_text())
        for file_name, expected_digest in summary["source_sha256"].items():
            digest = hashlib.sha256((root / file_name).read_bytes()).hexdigest()
            if digest != expected_digest:
                raise ValueError(f"source changed after experiment: {file_name}; regenerate artifacts explicitly")
            source_checks += 1
        if Path(summary["config"]["out"]).is_absolute():
            raise ValueError("published local absolute output path")
        for result in summary["policies"].values():
            if result["actual"]["unauthorized_executions"]:
                raise ValueError("unauthorized execution in published experiment")
        runs.append({**run, "summary": summary})
    rebuilt = aggregate_runs(runs)
    if not same_statistics(rebuilt["settings"], aggregate["settings"]) or rebuilt["run_count"] != aggregate["run_count"]:
        raise ValueError("aggregate does not match per-run observations")
    decisions = sum(sum(run["summary"]["config"][name] for name in ("train_size", "calibration_size", "test_size")) for run in runs)
    if decisions != aggregate["selected_decision_count"]:
        raise ValueError("logged-decision total does not match runs")
    if any(directory.rglob("*.jsonl")):
        raise ValueError("raw logs must not be included in published aggregate artifacts")
    paper = (root / "paper" / "paper.md").read_text()
    if "{{" in paper or f"{decisions:,}" not in paper:
        raise ValueError("paper was not rendered from the committed experiment results")
    claims = json.loads((root / "paper" / "claims.json").read_text())
    for setting in ("noisy", "small", "linear"):
        experiment_setting = "small_data" if setting == "small" else setting
        for policy in ("direct", "dr"):
            expected_number = f"{aggregate['settings'][experiment_setting]['policies'][policy]['utility']['mean']:.4f}"
            if claims[f"{setting}_{policy}"] != expected_number:
                raise ValueError("paper numeric claim differs from committed aggregate")
    return {"runs": len(runs), "source_hash_checks": source_checks, "logged_decision_instances": decisions, "paper_claims": "consistent"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent), indent=2))