"""Pinned, attributed BFCL data with label-free observations and grouped splits."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import urllib.request


BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
BFCL_FILES = (
    "BFCL_v3_live_multiple.json",
    "possible_answer/BFCL_v3_live_multiple.json",
    "BFCL_v3_live_irrelevance.json",
    "README.md",
)


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    category: str
    query: str
    tools: tuple[dict, ...]
    gold_names: tuple[str, ...]
    reference_arguments: dict

    def observation(self) -> dict:
        return {"query": self.query, "tools": list(self.tools)}


def download_bfcl(cache: Path) -> dict:
    directory = cache / BFCL_REVISION
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": BFCL_REPO, "revision": BFCL_REVISION, "license": "Apache-2.0", "files": {}}
    for file_name in BFCL_FILES:
        destination = directory / file_name
        url = f"https://huggingface.co/datasets/{BFCL_REPO}/resolve/{BFCL_REVISION}/{file_name}"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "counterfactual-tool-ranking-research/2"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(20_000_001)
            if len(payload) > 20_000_000:
                raise ValueError("unexpectedly large public data file")
            if file_name.endswith(".json"):
                for line in payload.decode("utf-8").splitlines():
                    json.loads(line)
            destination.write_bytes(payload)
        payload = destination.read_bytes()
        manifest["files"][file_name] = {"url": url, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest_path = directory / "provenance.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("cached dataset content differs from recorded provenance")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def read_jsonl(file_path: Path) -> list[dict]:
    return [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_records(multiple: list[dict], answers: list[dict], irrelevant: list[dict]) -> tuple[list[PublicTask], dict]:
    answer_index = {row["id"]: row["ground_truth"] for row in answers}
    if len(answer_index) != len(answers):
        raise ValueError("duplicate reference IDs")
    tasks = []
    skipped = []
    non_user_context = []
    seen = set()
    for category, records in (("multiple", multiple), ("irrelevance", irrelevant)):
        for row in records:
            task_id = row["id"]
            if task_id in seen:
                raise ValueError("duplicate public task ID")
            seen.add(task_id)
            if len(row["question"]) != 1:
                skipped.append({"id": task_id, "reason": "not single-turn"})
                continue
            messages = row["question"][0]
            if not any(message["role"] == "user" for message in messages) or any(
                not isinstance(message.get("content"), str) or not isinstance(message.get("role"), str)
                for message in messages
            ):
                skipped.append({"id": task_id, "reason": "missing user request or nontext message"})
                continue
            query = "\n".join(f"[{message['role']}]\n{message['content']}" for message in messages)
            if any(message["role"] != "user" for message in messages):
                non_user_context.append(task_id)
            tools = tuple(row["function"])
            names = [tool["name"] for tool in tools]
            if not query or not tools or len(names) != len(set(names)):
                skipped.append({"id": task_id, "reason": "missing query/tools or duplicate function names"})
                continue
            if category == "multiple" and task_id not in answer_index:
                skipped.append({"id": task_id, "reason": "no reference with exactly matching ID"})
                continue
            truth = answer_index[task_id] if category == "multiple" else []
            if category == "multiple" and (len(truth) != 1 or not isinstance(truth[0], dict) or len(truth[0]) != 1):
                skipped.append({"id": task_id, "reason": "not one reference function call"})
                continue
            gold = tuple(truth[0]) if truth else ()
            if not set(gold).issubset(names):
                raise ValueError(f"reference function missing from candidates: {task_id}")
            tasks.append(PublicTask(task_id, category, query, tools, gold, truth[0] if truth else {}))
    return tasks, {
        "input_multiple": len(multiple), "input_irrelevance": len(irrelevant), "retained": len(tasks), "skipped": skipped,
        "orphan_reference_ids": sorted(set(answer_index) - {row["id"] for row in multiple}),
        "full_role_context_preserved": True,
        "rows_with_non_user_context": len(non_user_context),
    }


def load_bfcl(cache: Path) -> tuple[list[PublicTask], dict]:
    directory = cache / BFCL_REVISION
    tasks, accounting = parse_records(
        read_jsonl(directory / BFCL_FILES[0]), read_jsonl(directory / BFCL_FILES[1]), read_jsonl(directory / BFCL_FILES[2]),
    )
    manifest = json.loads((directory / "provenance.json").read_text())
    for file_name, expected in manifest["files"].items():
        if hashlib.sha256((directory / file_name).read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("public data hash mismatch")
    return tasks, {"source": manifest, "accounting": accounting}


def family_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def split_by_tool_family(tasks: list[PublicTask], seed: int) -> tuple[dict[str, list[PublicTask]], dict]:
    parents = list(range(len(tasks)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, task in enumerate(tasks):
        keys = ["tool:" + family_name(tool["name"]) for tool in task.tools]
        keys.append("query:" + " ".join(task.query.lower().split()))
        for key in keys:
            if key in owners:
                parents[find(index)] = find(owners[key])
            else:
                owners[key] = index
    components = {}
    for index, task in enumerate(tasks):
        components.setdefault(find(index), []).append(task)
    splits = {"train": [], "calibration": [], "test": []}
    assignment = {}
    forced_training = []
    for component in components.values():
        identity = "|".join(sorted(task.task_id for task in component))
        digest = hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()
        percentile = int(digest[:8], 16) / 2**32
        split = "train" if percentile < 0.6 else "calibration" if percentile < 0.8 else "test"
        if len(component) > len(tasks) / 2:
            split = "train"
            forced_training.append({"component": digest, "size": len(component)})
        splits[split].extend(component)
        for task in component:
            assignment[task.task_id] = {"split": split, "component": digest}
    for values in splits.values():
        values.sort(key=lambda task: task.task_id)
    tools_by_split = {name: {family_name(tool["name"]) for task in values for tool in task.tools} for name, values in splits.items()}
    for first, second in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        if tools_by_split[first] & tools_by_split[second]:
            raise ValueError("function family leakage across splits")
    return splits, {
        "seed": seed, "components": len(components), "largest_component": max(map(len, components.values()), default=0),
        "sizes": {name: len(values) for name, values in splits.items()}, "assignments": assignment,
        "function_name_overlap": 0, "group_definition": "connected normalized function names or identical normalized queries",
        "forced_large_training_components": forced_training,
        "group_counts": {name: len({assignment[task.task_id]["component"] for task in values}) for name, values in splits.items()},
    }


if __name__ == "__main__":
    cache = Path(__file__).parent / "data_cache" / "bfcl"
    download_bfcl(cache)
    loaded, provenance = load_bfcl(cache)
    _, split_manifest = split_by_tool_family(loaded, 7)
    print(json.dumps({"revision": BFCL_REVISION, "accounting": provenance["accounting"], "split": {key: value for key, value in split_manifest.items() if key != "assignments"}}, indent=2))