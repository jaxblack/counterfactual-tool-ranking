"""Version-two controls that learn realized returns without altering v1 source."""

import numpy as np
from sklearn.model_selection import KFold

from learning import Learner, features, make_regressor
from sandbox import ABSTAIN, legal_actions


class FullReturnLearner(Learner):
    def fit(self, decisions):
        super().fit(decisions)
        pairs = [(decision.task, action) for decision in decisions for action in decision.actions]
        matrix = self.vectorizer.transform([features(task, action) for task, action in pairs])
        offsets = np.cumsum([0] + [len(decision.actions) for decision in decisions])
        selected = np.array([offsets[index] + decision.chosen for index, decision in enumerate(decisions)])
        returns = np.array([
            decision.outcome.utility(self.cost_weight, self.risk_weight, self.latency_weight)
            for decision in decisions
        ])
        components = np.array([
            [float(decision.outcome.success), float(decision.outcome.unsafe), decision.outcome.cost,
             decision.outcome.latency_ms / 100, decision.outcome.excess_access]
            for decision in decisions
        ])
        self.full_model = make_regressor(self.seed, self.model)
        self.component_model = make_regressor(self.seed, self.model)
        self.full_dr_model = make_regressor(self.seed, self.model)
        self.full_model.fit(matrix[selected], returns)
        self.component_model.fit(matrix[selected], components)
        out_of_fold = np.zeros(len(pairs))
        for fold, (training, heldout) in enumerate(KFold(n_splits=3, shuffle=True, random_state=self.seed).split(decisions)):
            model = make_regressor(self.seed + 10 + fold, self.model)
            model.fit(matrix[selected[training]], returns[training])
            heldout_rows = np.concatenate([np.arange(offsets[index], offsets[index + 1]) for index in heldout])
            out_of_fold[heldout_rows] = model.predict(matrix[heldout_rows])
        targets = out_of_fold.copy()
        supported = np.array([probability > 0 for decision in decisions for probability in decision.probabilities])
        for index, decision in enumerate(decisions):
            row = selected[index]
            targets[row] += (returns[index] - out_of_fold[row]) / decision.probabilities[decision.chosen]
        context_weights = np.array([1 / len(decision.actions) for decision in decisions for _ in decision.actions])
        self.full_dr_model.fit(matrix[supported], targets[supported], sample_weight=context_weights[supported])
        return self

    def score_tasks(self, tasks):
        action_lists = [legal_actions(task) for task in tasks]
        pairs = [(task, action) for task, actions in zip(tasks, action_lists, strict=True) for action in actions]
        matrix = self.vectorizer.transform([features(task, action) for task, action in pairs])
        components = self.component_model.predict(matrix)
        component_values = components @ np.array([1, -self.risk_weight, -self.cost_weight, -self.latency_weight, -0.05])
        vectors = {
            "full_direct": self.full_model.predict(matrix),
            "component_direct": component_values,
            "full_dr": self.full_dr_model.predict(matrix),
        }
        rankings = self.rank(tasks)
        scores = []
        offset = 0
        for ranking in rankings:
            section = slice(offset, offset + len(ranking.actions))
            current = {name: vector[section].copy() for name, vector in vectors.items()}
            current["nominal_direct"] = ranking.direct_values.copy()
            current["nominal_dr"] = ranking.dr_values.copy()
            current["ips"] = ranking.ips_values.copy()
            for index, action in enumerate(ranking.actions):
                if action.tool == ABSTAIN:
                    for values in current.values():
                        values[index] = 0.0
            scores.append(current)
            offset += len(ranking.actions)
        return rankings, scores


def supported_argmax(ranking, values):
    eligible = values.copy()
    eligible[~ranking.supported] = -np.inf
    abstain = next(index for index, action in enumerate(ranking.actions) if action.tool == ABSTAIN)
    eligible[abstain] = 0.0
    return int(np.argmax(eligible))