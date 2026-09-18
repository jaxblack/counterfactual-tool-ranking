"""Logged bandit feedback, cross-fitted DR learning, and overlap-aware OPE."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from sandbox import (
    ABSTAIN, TOOLS, Action, Outcome, Sandbox, Task, authorize, candidates,
    legal_actions, task_from_dict, task_to_dict,
)


@dataclass(frozen=True)
class Decision:
    task: Task
    actions: tuple[Action, ...]
    probabilities: tuple[float, ...]
    chosen: int
    outcome: Outcome


def catalog_digest() -> str:
    serialized = json.dumps([asdict(tool) for tool in TOOLS.values()], sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def candidate_snapshot(task: Task) -> list[dict]:
    return [
        {**asdict(action), "resources": list(action.resources), "allowed": allowed, "reason": reason}
        for action in candidates(task)
        for allowed, reason in (authorize(task.policy, action),)
    ]


def logging_probabilities(
    actions: tuple[Action, ...], epsilon: float, excluded_tools: tuple[str, ...] = (),
) -> tuple[float, ...]:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between zero and one")
    if ABSTAIN in excluded_tools or any(name not in TOOLS for name in excluded_tools):
        raise ValueError("only registered executable tools can be excluded")
    eligible = [index for index, action in enumerate(actions) if action.tool not in excluded_tools]
    if not eligible:
        raise ValueError("logger must retain abstain")
    executable = [index for index in eligible if actions[index].tool != ABSTAIN]
    favorite = min(executable, key=lambda index: TOOLS[actions[index].tool].cost) if executable else eligible[0]
    probabilities = [0.0] * len(actions)
    for index in eligible:
        probabilities[index] = epsilon / len(eligible)
    probabilities[favorite] += 1 - epsilon
    return tuple(probabilities)


def collect(
    tasks: list[Task], epsilon: float, seed: int, excluded_tools: tuple[str, ...] = (),
    executor=None,
) -> list[Decision]:
    generator = random.Random(seed)
    decisions = []
    for task in tasks:
        actions = legal_actions(task)
        probabilities = logging_probabilities(actions, epsilon, excluded_tools)
        chosen = generator.choices(range(len(actions)), weights=probabilities, k=1)[0]
        if executor is None:
            with Sandbox(task) as sandbox:
                outcome = sandbox.execute(actions[chosen])
        else:
            outcome = executor.execute(task, actions[chosen])
        decisions.append(Decision(task, actions, probabilities, chosen, outcome))
    return decisions


def write_log(file_path: Path, decisions: list[Decision], metadata: dict) -> None:
    with file_path.open("w", encoding="utf-8") as output:
        for decision in decisions:
            raw = {
                "schema_version": 2,
                "catalog_sha256": catalog_digest(),
                "logger": metadata,
                "task": task_to_dict(decision.task),
                "candidates": candidate_snapshot(decision.task),
                "actions": [asdict(action) for action in decision.actions],
                "probabilities": decision.probabilities,
                "chosen": decision.chosen,
                "propensity": decision.probabilities[decision.chosen],
                "outcome": asdict(decision.outcome),
            }
            output.write(json.dumps(raw, sort_keys=True) + "\n")


def read_log(file_path: Path) -> list[Decision]:
    decisions = []
    with file_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            raw = json.loads(line)
            if raw["schema_version"] != 2:
                raise ValueError(f"line {line_number}: unsupported schema")
            if raw["catalog_sha256"] != catalog_digest():
                raise ValueError(f"line {line_number}: tool catalog changed")
            task = task_from_dict(raw["task"])
            if raw["candidates"] != candidate_snapshot(task):
                raise ValueError(f"line {line_number}: candidate snapshot changed")
            actions = tuple(Action(action["tool"], tuple(action["resources"]), action.get("amount")) for action in raw["actions"])
            probabilities = tuple(float(value) for value in raw["probabilities"])
            chosen = raw["chosen"]
            if (
                actions != legal_actions(task)
                or len(probabilities) != len(actions)
                or not all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities)
                or not np.isclose(sum(probabilities), 1.0)
                or type(chosen) is not int
                or not 0 <= chosen < len(actions)
                or probabilities[chosen] <= 0
                or not np.isclose(raw["propensity"], probabilities[chosen])
            ):
                raise ValueError(f"line {line_number}: invalid action support or propensity")
            if raw["logger"].get("name") == "cheap_epsilon_greedy":
                expected = logging_probabilities(actions, raw["logger"]["epsilon"], tuple(raw["logger"]["excluded_tools"]))
                if not np.allclose(expected, probabilities, atol=1e-12, rtol=1e-12):
                    raise ValueError(f"line {line_number}: probabilities disagree with logger")
            outcome = raw["outcome"]
            if any(not np.isfinite(outcome[key]) or outcome[key] < 0 for key in ("cost", "latency_ms", "runtime_ms", "excess_access")):
                raise ValueError(f"line {line_number}: invalid outcome metric")
            if any(type(outcome[key]) is not bool for key in ("success", "unsafe", "abstained", "denied")):
                raise ValueError(f"line {line_number}: invalid outcome flag")
            outcome["changed_resources"] = tuple(outcome["changed_resources"])
            outcome["answer"] = tuple(tuple(pair) for pair in outcome["answer"])
            decisions.append(Decision(task, actions, probabilities, chosen, Outcome(**outcome)))
    return decisions


def features(task: Task, action: Action) -> dict:
    return {
        "domain": task.domain,
        "requested_field": task.field,
        "requires_current_data": int(task.fresh),
        "requested_count": len(task.resources),
        "observed_approval": int(task.observed_approval) if task.domain != "documents" else -1,
        "observed_limit": task.observed_limit if task.domain == "crm" else -1,
        "discount_amount": action.amount if action.amount is not None else -1,
        "observed_service": task.observed_service,
        "cache_age": task.cache_age if task.domain == "documents" else -1,
        "tool": action.tool,
        "action_resource_count": len(action.resources),
        "scopes": ",".join(sorted(task.policy.scopes)),
    }


def feature_key(task: Task, action: Action) -> tuple:
    return tuple(sorted(features(task, action).items()))


def action_penalty(task: Task, action: Action, cost_weight: float, latency_weight: float = 0.0) -> float:
    if action.tool == ABSTAIN:
        return 0.0
    return (
        cost_weight * TOOLS[action.tool].cost + 0.05 * len(set(action.resources) - set(task.resources))
        + latency_weight * TOOLS[action.tool].latency_ms / 100
    )


def make_regressor(seed: int, model: str = "trees") -> ExtraTreesRegressor | Ridge:
    if model == "linear":
        return Ridge(alpha=10)
    if model != "trees":
        raise ValueError("unknown model")
    return ExtraTreesRegressor(
        n_estimators=48, max_depth=12, min_samples_leaf=3, random_state=seed, n_jobs=1,
    )


@dataclass
class Ranking:
    actions: tuple[Action, ...]
    direct_values: np.ndarray
    dr_values: np.ndarray
    disagreement: np.ndarray
    supported: np.ndarray
    counts: np.ndarray
    unsafe_predictions: np.ndarray
    ips_values: np.ndarray | None = None
    direct_disagreement: np.ndarray | None = None


class Learner:
    def __init__(
        self, seed: int, cost_weight: float = 1.0, risk_weight: float = 2.0,
        latency_weight: float = 0.0, model: str = "trees",
    ):
        self.seed = seed
        self.cost_weight = cost_weight
        self.risk_weight = risk_weight
        self.latency_weight = latency_weight
        self.model = model
        self.vectorizer = DictVectorizer(sparse=False)
        self.outcome_model = make_regressor(seed, model)
        self.dr_model = make_regressor(seed + 1, model)
        self.ips_model = make_regressor(seed + 2, model)
        self.support: set[tuple] = set()
        self.counts: Counter = Counter()

    def fit(self, decisions: list[Decision]) -> "Learner":
        if len(decisions) < 12 or len({decision.task.task_id for decision in decisions}) != len(decisions):
            raise ValueError("training requires at least 12 distinct task decisions")
        self.support.clear()
        self.counts.clear()
        expanded = [
            (decision.task, action)
            for decision in decisions
            for action in decision.actions
        ]
        matrix = self.vectorizer.fit_transform([features(task, action) for task, action in expanded])
        offsets = np.cumsum([0] + [len(decision.actions) for decision in decisions])
        chosen_rows = np.array([offsets[index] + decision.chosen for index, decision in enumerate(decisions)])
        observed_outcomes = np.array([
            [float(decision.outcome.success), float(decision.outcome.unsafe)]
            for decision in decisions
        ])
        predictions = np.zeros((len(expanded), 2))
        folds = KFold(n_splits=3, shuffle=True, random_state=self.seed)
        for fold_index, (train_indices, heldout_indices) in enumerate(folds.split(decisions)):
            model = make_regressor(self.seed + 10 + fold_index, self.model)
            model.fit(matrix[chosen_rows[train_indices]], observed_outcomes[train_indices])
            heldout_rows = np.concatenate([
                np.arange(offsets[index], offsets[index + 1]) for index in heldout_indices
            ])
            predictions[heldout_rows] = np.clip(model.predict(matrix[heldout_rows]), 0, 1)
        penalties = np.array([action_penalty(task, action, self.cost_weight, self.latency_weight) for task, action in expanded])
        pseudo_values = predictions[:, 0] - self.risk_weight * predictions[:, 1] - penalties
        ips_values = np.zeros(len(expanded))
        supported_rows = np.array([
            probability > 0 for decision in decisions for probability in decision.probabilities
        ])
        for index, decision in enumerate(decisions):
            chosen_row = chosen_rows[index]
            reward = decision.outcome.utility(self.cost_weight, self.risk_weight, self.latency_weight)
            pseudo_values[chosen_row] += (
                reward - pseudo_values[chosen_row]
            ) / decision.probabilities[decision.chosen]
            ips_values[chosen_row] = reward / decision.probabilities[decision.chosen]
            self.counts[feature_key(decision.task, decision.actions[decision.chosen])] += 1
            for action, probability in zip(decision.actions, decision.probabilities, strict=True):
                if probability > 0:
                    self.support.add(feature_key(decision.task, action))
        self.outcome_model.fit(matrix[chosen_rows], observed_outcomes)
        context_weights = np.array([1 / len(decision.actions) for decision in decisions for _ in decision.actions])
        self.dr_model.fit(matrix[supported_rows], pseudo_values[supported_rows], sample_weight=context_weights[supported_rows])
        self.ips_model.fit(matrix[supported_rows], ips_values[supported_rows], sample_weight=context_weights[supported_rows])
        return self

    def rank(self, tasks: list[Task]) -> list[Ranking]:
        action_lists = [legal_actions(task) for task in tasks]
        pairs = [(task, action) for task, actions in zip(tasks, action_lists, strict=True) for action in actions]
        matrix = self.vectorizer.transform([features(task, action) for task, action in pairs])
        outcomes = np.clip(self.outcome_model.predict(matrix), 0, 1)
        penalties = np.array([action_penalty(task, action, self.cost_weight, self.latency_weight) for task, action in pairs])
        direct = outcomes[:, 0] - self.risk_weight * outcomes[:, 1] - penalties
        estimators = getattr(self.dr_model, "estimators_", [self.dr_model])
        tree_predictions = np.array([tree.predict(matrix) for tree in estimators])
        dr_values = tree_predictions.mean(axis=0)
        disagreement = tree_predictions.std(axis=0)
        direct_estimators = getattr(self.outcome_model, "estimators_", [self.outcome_model])
        direct_predictions = np.array([np.clip(tree.predict(matrix), 0, 1) for tree in direct_estimators])
        direct_disagreement = (direct_predictions[:, :, 0] - self.risk_weight * direct_predictions[:, :, 1]).std(axis=0)
        ips_values = self.ips_model.predict(matrix)
        offset = 0
        rankings = []
        for task, actions in zip(tasks, action_lists, strict=True):
            section = slice(offset, offset + len(actions))
            values = direct[section].copy()
            adjusted = dr_values[section].copy()
            uncertainty = disagreement[section].copy()
            unsafe = outcomes[section, 1].copy()
            ips_section = ips_values[section].copy()
            direct_uncertainty = direct_disagreement[section].copy()
            for index, action in enumerate(actions):
                if action.tool == ABSTAIN:
                    values[index] = adjusted[index] = uncertainty[index] = unsafe[index] = 0.0
                    ips_section[index] = direct_uncertainty[index] = 0.0
            rankings.append(Ranking(
                actions, values, adjusted, uncertainty,
                np.array([feature_key(task, action) in self.support for action in actions]),
                np.array([self.counts[feature_key(task, action)] for action in actions]),
                unsafe, ips_section, direct_uncertainty,
            ))
            offset += len(actions)
        return rankings


def choose(
    task: Task, ranking: Ranking, policy: str, threshold: float = 0.0,
    min_count: int = 5, uncertainty_weight: float = 1.0,
) -> int:
    actions = ranking.actions
    abstain = next(index for index, action in enumerate(actions) if action.tool == ABSTAIN)
    executable = [index for index, action in enumerate(actions) if action.tool != ABSTAIN]
    if not executable or policy == "abstain":
        return abstain
    if policy == "cheapest":
        return min(executable, key=lambda index: TOOLS[actions[index].tool].cost)
    if policy in ("schema_match", "rules"):
        preferred = {"documents": "docs.batch_get", "tickets": "tickets.close_checked", "crm": "crm.discount_checked"}[task.domain]
        if policy == "rules":
            if task.domain in ("tickets", "crm"):
                if not task.observed_approval:
                    return abstain
                if task.domain == "crm":
                    if task.amount > task.observed_limit:
                        return abstain
                    preferred = "crm.discount_one" if len(task.resources) == 1 else "crm.discount_checked"
                else:
                    preferred = "tickets.close_one" if len(task.resources) == 1 else "tickets.close_checked"
            elif task.field == "title":
                preferred = "docs.search"
            elif not task.fresh:
                preferred = "docs.cache"
            elif len(task.resources) == 1:
                preferred = "docs.get"
        return next((index for index in executable if actions[index].tool == preferred), abstain)
    if policy not in ("direct", "ips", "dr", "conservative_direct", "conservative_dr"):
        raise ValueError(f"unknown policy: {policy}")
    if policy in ("direct", "conservative_direct"):
        values = ranking.direct_values.copy()
    elif policy == "ips":
        if ranking.ips_values is None:
            raise ValueError("IPS model was not fitted")
        values = ranking.ips_values.copy()
    else:
        values = ranking.dr_values.copy()
    if policy != "direct":
        values[~ranking.supported] = -np.inf
    if policy in ("conservative_direct", "conservative_dr"):
        uncertainty = ranking.direct_disagreement if policy == "conservative_direct" else ranking.disagreement
        if uncertainty is not None:
            values -= uncertainty_weight * uncertainty
        values[ranking.counts < min_count] = -np.inf
    values[abstain] = threshold
    return int(np.argmax(values))


def bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> list[float]:
    if samples < 1 or len(values) < 1:
        raise ValueError("bootstrap needs observations and at least one resample")
    generator = np.random.default_rng(seed)
    means = [values[generator.integers(0, len(values), size=len(values))].mean() for _ in range(samples)]
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def evaluate_ope(
    decisions: list[Decision], rankings: list[Ranking], choices: list[int],
    cost_weight: float = 1.0, risk_weight: float = 2.0,
    bootstrap: int = 200, seed: int = 0,
    latency_weight: float = 0.0,
) -> dict:
    if not decisions or not len(decisions) == len(rankings) == len(choices):
        raise ValueError("OPE inputs must be nonempty and have equal length")
    if any(
        decision.actions != ranking.actions or not 0 <= choice < len(decision.actions)
        for decision, ranking, choice in zip(decisions, rankings, choices, strict=True)
    ):
        raise ValueError("OPE target actions must match logged candidate order")
    unsupported = sum(decision.probabilities[choice] == 0 for decision, choice in zip(decisions, choices, strict=True))
    if unsupported:
        return {
            "identifiable": False, "unsupported_fraction": unsupported / len(decisions),
            "reason": "target selects actions with zero logging probability; full value is not identified",
            "ips": None, "snips": None, "dr": None, "dr_ci95": None, "ess": 0.0,
        }
    weights = np.array([
        float(decision.chosen == choice) / decision.probabilities[choice]
        for decision, choice in zip(decisions, choices, strict=True)
    ])
    rewards = np.array([decision.outcome.utility(cost_weight, risk_weight, latency_weight) for decision in decisions])
    direct_target = np.array([ranking.direct_values[choice] for ranking, choice in zip(rankings, choices, strict=True)])
    direct_observed = np.array([
        ranking.direct_values[decision.chosen] for ranking, decision in zip(rankings, decisions, strict=True)
    ])
    dr_values = direct_target + weights * (rewards - direct_observed)
    ips_values = weights * rewards
    weight_sum = weights.sum()
    effective_sample_size = float(weight_sum ** 2 / np.square(weights).sum()) if weight_sum else 0.0
    return {
        "identifiable": True,
        "unsupported_fraction": 0.0,
        "direct": float(direct_target.mean()),
        "ips": float(ips_values.mean()),
        "snips": float(ips_values.sum() / weight_sum) if weight_sum else None,
        "dr": float(dr_values.mean()),
        "dr_ci95": bootstrap_mean(dr_values, bootstrap, seed),
        "ess": effective_sample_size,
        "matched": int(np.count_nonzero(weights)),
        "max_weight": float(weights.max()),
        "warning": (
            "no target-action matches; DR uses only the outcome model here" if not weight_sum
            else "low effective sample size; interval may be unstable" if effective_sample_size < 30 else ""
        ),
    }