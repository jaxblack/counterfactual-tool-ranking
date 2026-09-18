"""Semi-synthetic bandit reduction of public tool-selection annotations."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from learning import make_regressor


def tool_text(tool: dict) -> str:
    return json.dumps(tool, sort_keys=True, ensure_ascii=True)[:12000]


def stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / 2**32


class TextFeatures:
    def __init__(self):
        self.words = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, stop_words="english")
        self.characters = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=20000)

    def fit(self, observations):
        documents = [text for observation in observations for text in (observation["query"], *(tool_text(tool) for tool in observation["tools"]))]
        self.words.fit(documents)
        self.characters.fit(documents)
        return self

    def transform(self, observation):
        query, tools = observation["query"], observation["tools"]
        documents = [tool_text(tool) for tool in tools]
        word_sim = (self.words.transform(documents) @ self.words.transform([query]).T).toarray().ravel()
        char_sim = (self.characters.transform(documents) @ self.characters.transform([query]).T).toarray().ravel()
        maximum = float(word_sim.max())
        second = float(np.sort(word_sim)[-2]) if len(tools) > 1 else 0.0
        rows = []
        for index, tool in enumerate(tools):
            parameters = tool.get("parameters", {})
            rows.append([
                word_sim[index], char_sim[index], maximum, maximum - word_sim[index], maximum - second,
                len(parameters.get("required", [])), len(parameters.get("properties", {})),
                np.log1p(len(documents[index])), np.log1p(len(query)), len(tools), 0,
            ])
        rows.append([0, 0, maximum, maximum, maximum - second, 0, 0, 0, np.log1p(len(query)), len(tools), 1])
        return np.asarray(rows, dtype=float), word_sim


def allowed_mask(task, scenario, seed):
    allowed = np.ones(len(task.tools) + 1, dtype=bool)
    if scenario == "authorization":
        for index, tool in enumerate(task.tools):
            allowed[index] = stable_fraction(f"acl:{seed}:{tool['name']}") >= 0.2
    return allowed


def logging_distribution(task, similarities, allowed, scenario, seed, epsilon=0.3):
    if not 0 < epsilon <= 1:
        raise ValueError("public experiments require nonzero exploration")
    eligible = allowed.copy()
    if scenario == "support_gap":
        for index, tool in enumerate(task.tools):
            eligible[index] &= stable_fraction(f"support:{seed}:{tool['name']}") >= 0.25
    candidates = np.flatnonzero(eligible)
    executable = [index for index in candidates if index < len(task.tools)]
    favorite = max(executable, key=lambda index: similarities[index]) if executable else len(task.tools)
    probabilities = eligible.astype(float) * epsilon / len(candidates)
    probabilities[favorite] += 1 - epsilon
    return probabilities


def actual_outcome(task, action, allowed, scenario):
    abstain = action == len(task.tools)
    if not allowed[action]:
        return {"reward": 0.0, "correct": False, "abstained": False, "denied": True, "cost": 0.0, "wrong_call": False}
    available_gold = {tool["name"] for index, tool in enumerate(task.tools) if allowed[index]} & set(task.gold_names)
    if abstain:
        return {"reward": 0.0, "correct": not available_gold, "abstained": True, "denied": False, "cost": 0.0, "wrong_call": False}
    correct = task.tools[action]["name"] in available_gold
    cost = 0.0 if scenario == "native" else min(0.05, 0.005 + len(tool_text(task.tools[action])) / 200000)
    return {"reward": (1.0 if correct else -1.0) - cost, "correct": correct, "abstained": False, "denied": False, "cost": cost, "wrong_call": not correct}


@dataclass
class PublicDecision:
    task_id: str
    group: str
    matrix: np.ndarray
    probabilities: np.ndarray
    chosen: int
    reward: float
    allowed: np.ndarray
    similarities: np.ndarray


def collect_public(tasks, feature_model, assignments, seed, scenario, sampling_seed=None):
    generator = np.random.default_rng(seed + 100 if sampling_seed is None else sampling_seed)
    decisions = []
    for task in tasks:
        matrix, similarities = feature_model.transform(task.observation())
        allowed = allowed_mask(task, scenario, seed)
        probabilities = logging_distribution(task, similarities, allowed, scenario, seed)
        chosen = int(generator.choice(len(probabilities), p=probabilities))
        outcome = actual_outcome(task, chosen, allowed, scenario)
        decisions.append(PublicDecision(task.task_id, assignments[task.task_id]["component"], matrix, probabilities, chosen, outcome["reward"], allowed, similarities))
    return decisions


class PublicLearner:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, decisions):
        matrix = np.concatenate([decision.matrix for decision in decisions])
        offsets = np.cumsum([0] + [len(decision.matrix) for decision in decisions])
        selected = np.array([offsets[index] + decision.chosen for index, decision in enumerate(decisions)])
        rewards = np.array([decision.reward for decision in decisions])
        groups = [decision.group for decision in decisions]
        if len(set(groups)) < 3:
            raise ValueError("at least three independent training tool families are required")
        out_of_fold = np.zeros(len(matrix))
        for fold, (training, heldout) in enumerate(GroupKFold(n_splits=3).split(selected, groups=groups)):
            model = make_regressor(self.seed + fold)
            model.fit(matrix[selected[training]], rewards[training])
            heldout_rows = np.concatenate([np.arange(offsets[index], offsets[index + 1]) for index in heldout])
            out_of_fold[heldout_rows] = model.predict(matrix[heldout_rows])
        pseudo = out_of_fold.copy()
        ips = np.zeros(len(matrix))
        for index, decision in enumerate(decisions):
            row = selected[index]
            pseudo[row] += (rewards[index] - out_of_fold[row]) / decision.probabilities[decision.chosen]
            ips[row] = rewards[index] / decision.probabilities[decision.chosen]
        support = np.concatenate([decision.probabilities > 0 for decision in decisions])
        weights = np.concatenate([np.full(len(decision.matrix), 1 / len(decision.matrix)) for decision in decisions])
        self.direct = make_regressor(self.seed)
        self.dr = make_regressor(self.seed)
        self.ips = make_regressor(self.seed)
        self.direct.fit(matrix[selected], rewards)
        self.dr.fit(matrix[support], pseudo[support], sample_weight=weights[support])
        self.ips.fit(matrix[support], ips[support], sample_weight=weights[support])
        return self

    def predict(self, decisions):
        offsets = np.cumsum([0] + [len(decision.matrix) for decision in decisions])
        matrix = np.concatenate([decision.matrix for decision in decisions])
        vectors = {name: np.clip(getattr(self, name).predict(matrix), -1.05, 1.0) for name in ("direct", "ips", "dr")}
        predictions = []
        for index in range(len(decisions)):
            rows = {name: values[offsets[index]:offsets[index + 1]].copy() for name, values in vectors.items()}
            for values in rows.values():
                values[-1] = 0.0
            predictions.append(rows)
        return predictions


def public_choices(decisions, predictions, policy):
    choices = []
    for decision, values in zip(decisions, predictions, strict=True):
        if policy == "tfidf":
            scores = np.append(decision.similarities, 0.08)
        else:
            scores = values[policy].copy()
        scores[~decision.allowed] = -np.inf
        choices.append(int(np.argmax(scores)))
    return choices


def public_metrics(tasks, decisions, choices, scenario):
    outcomes = [actual_outcome(task, chosen, decision.allowed, scenario) for task, decision, chosen in zip(tasks, decisions, choices, strict=True)]
    count = len(outcomes)
    executed = sum(not row["abstained"] for row in outcomes)
    return {
        "tasks": count,
        "utility": float(np.mean([row["reward"] for row in outcomes])),
        "selection_accuracy": sum(row["correct"] for row in outcomes) / count,
        "coverage": executed / count,
        "success_all": sum(row["correct"] and not row["abstained"] for row in outcomes) / count,
        "correct_on_executed": sum(row["correct"] and not row["abstained"] for row in outcomes) / executed if executed else None,
        "wrong_call_rate": sum(row["wrong_call"] for row in outcomes) / count,
        "denied_attempts": sum(row["denied"] for row in outcomes),
        "mean_simulated_cost": float(np.mean([row["cost"] for row in outcomes])),
    }, np.array([row["reward"] for row in outcomes])


def public_ope(decisions, predictions, choices):
    if any(decision.probabilities[choice] == 0 for decision, choice in zip(decisions, choices, strict=True)):
        return {"identified": False, "reason": "target selects a zero-probability action"}
    weights = np.array([float(decision.chosen == choice) / decision.probabilities[choice] for decision, choice in zip(decisions, choices, strict=True)])
    rewards = np.array([decision.reward for decision in decisions])
    target_values = np.array([values["direct"][choice] for values, choice in zip(predictions, choices, strict=True)])
    observed_values = np.array([values["direct"][decision.chosen] for decision, values in zip(decisions, predictions, strict=True)])
    return {
        "identified": True, "direct": float(target_values.mean()), "ips": float(np.mean(weights * rewards)),
        "dr": float(np.mean(target_values + weights * (rewards - observed_values))),
        "ess": float(weights.sum()**2 / np.square(weights).sum()) if np.any(weights) else 0.0,
        "matched": int(np.count_nonzero(weights)),
    }