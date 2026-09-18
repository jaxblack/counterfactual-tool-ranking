from dataclasses import replace
from pathlib import Path
import unittest
from itertools import product

import numpy as np

from learning import collect, evaluate_ope
from public_data import PublicTask, parse_records, split_by_tool_family
from policy_contrast import disagreement_fallback, evaluate_contrast
from public_learning import TextFeatures, actual_outcome, allowed_mask, logging_distribution
from public_llm import normalize_schema, prompt_messages, score_response, select_tasks
from analyze_v2 import group_bootstrap
from verify_v2 import verify as verify_extension
from sandbox import ABSTAIN, legal_actions, make_tasks
from v2_learning import FullReturnLearner, supported_argmax


class FullReturnTests(unittest.TestCase):
    def test_realized_zero_returns_do_not_pay_nominal_fees(self):
        tasks = [
            replace(task, execution_policy=replace(task.policy, version="revoked", scopes=()))
            for task in make_tasks(160, 7, "revoked")
        ]
        decisions = collect(tasks, 0.5, 8)
        self.assertTrue(all(decision.outcome.utility() == 0 for decision in decisions))
        learner = FullReturnLearner(7).fit(decisions)
        rankings, scores = learner.score_tasks(tasks[:20])
        saw_nominal_penalty = False
        for ranking, values in zip(rankings, scores, strict=True):
            for name in ("full_direct", "component_direct", "full_dr"):
                np.testing.assert_allclose(values[name], 0, atol=1e-12)
            saw_nominal_penalty |= bool(np.any(values["nominal_direct"] < 0))
            self.assertIn(ranking.actions[supported_argmax(ranking, values["full_direct"])], ranking.actions)
        self.assertTrue(saw_nominal_penalty)

    def test_full_return_ope_uses_same_observed_reward_as_dr(self):
        training = collect(make_tasks(180, 4, "train", "noisy"), 0.5, 6)
        test = collect(make_tasks(30, 5, "test", "noisy"), 0.5, 7)
        learner = FullReturnLearner(4, latency_weight=0.5).fit(training)
        rankings, scores = learner.score_tasks([decision.task for decision in test])
        choices = [supported_argmax(ranking, values["full_direct"]) for ranking, values in zip(rankings, scores, strict=True)]
        full_rankings = [replace(ranking, direct_values=values["full_direct"]) for ranking, values in zip(rankings, scores, strict=True)]
        result = evaluate_ope(test, full_rankings, choices, bootstrap=20, latency_weight=0.5)
        self.assertTrue(result["identifiable"])
        self.assertTrue(np.isfinite(result["dr"]))

    def test_identical_support_boundary_for_all_new_controls(self):
        training = collect(make_tasks(200, 17, "train"), 0.4, 18, ("docs.batch_get",))
        learner = FullReturnLearner(17).fit(training)
        tasks = make_tasks(30, 19, "test")
        rankings, scores = learner.score_tasks(tasks)
        for task, ranking, values in zip(tasks, rankings, scores, strict=True):
            self.assertEqual(ranking.actions, legal_actions(task))
            for vector in values.values():
                action = ranking.actions[supported_argmax(ranking, vector)]
                self.assertNotEqual(action.tool, "docs.batch_get")
            abstain = next(index for index, action in enumerate(ranking.actions) if action.tool == ABSTAIN)
            self.assertTrue(all(vector[abstain] == 0 for vector in values.values()))


class PolicyContrastTests(unittest.TestCase):
    def test_identical_unsupported_action_has_exact_zero_difference(self):
        result = evaluate_contrast([np.array([1.0, 0.0])], [0], [0.4], [np.zeros(2)], [1], [1], 0, 1)
        self.assertTrue(result["point_identified"])
        self.assertEqual(result["shared_unsupported"], 1)
        self.assertEqual(result["conditional_hoeffding_interval"], [0.0, 0.0])

    def test_missing_disagreement_remains_a_partial_interval(self):
        result = evaluate_contrast([np.array([1.0, 0.0])], [0], [0.4], [np.zeros(2)], [1], [0], 0, 1)
        np.testing.assert_allclose(result["estimated_identification_interval"], [-0.4, 0.6])
        self.assertFalse(result["point_identified"])
        self.assertEqual(result["identification_width"], 1)

    def test_supported_randomization_recovers_true_policy_difference(self):
        result = evaluate_contrast(
            [np.array([0.5, 0.5]), np.array([0.5, 0.5])], [0, 1], [0.2, 0.8],
            [np.array([0.3, 0.4]), np.array([0.3, 0.4])], [1, 1], [0, 0], 0, 1,
        )
        np.testing.assert_allclose(result["estimated_identification_interval"], [0.6, 0.6])
        self.assertTrue(result["point_identified"])

    def test_fallback_only_changes_unsupported_disagreements(self):
        choices = disagreement_fallback([1, 1, 0], [1, 0, 1], [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.5])])
        self.assertEqual(choices, [1, 0, 0])

    def test_partial_intervals_contain_all_bounded_unsupported_outcomes(self):
        distribution = np.array([0.4, 0.6, 0.0, 0.0])
        for target, baseline in product(range(4), repeat=2):
            result = evaluate_contrast(
                [distribution] * 5, [0, 0, 1, 1, 1], [0.2, 0.2, 0.7, 0.7, 0.7],
                [np.array([0.1, 0.3, 0.4, 0.9])] * 5, [target] * 5, [baseline] * 5, 0, 1,
            )
            interval = result["estimated_identification_interval"]
            for third, fourth in product((0.0, 1.0), repeat=2):
                values = [0.2, 0.7, third, fourth]
                gain = values[target] - values[baseline]
                self.assertLessEqual(interval[0] - 1e-12, gain)
                self.assertGreaterEqual(interval[1] + 1e-12, gain)

    def test_group_bootstrap_preserves_constant_differences(self):
        self.assertEqual(group_bootstrap([0.25] * 6, ["a", "a", "a", "b", "c", "c"], 7), [0.25, 0.25])


class PublicDataTests(unittest.TestCase):
    def test_observation_contains_no_labels_or_task_id(self):
        task = PublicTask("test-1", "multiple", "read the report", ({"name": "read"},), ("read",), {"read": {}})
        self.assertEqual(set(task.observation()), {"query", "tools"})
        changed = replace(task, gold_names=(), reference_arguments={}, task_id="different", category="irrelevance")
        self.assertEqual(task.observation(), changed.observation())

    def test_connected_tool_names_and_duplicate_queries_stay_together(self):
        tasks = [
            PublicTask("one", "multiple", "request one", ({"name": "shared"}, {"name": "left"}), (), {}),
            PublicTask("two", "multiple", "request two", ({"name": "shared"}, {"name": "right"}), (), {}),
            PublicTask("three", "irrelevance", " REQUEST TWO ", ({"name": "another"},), (), {}),
        ]
        _, manifest = split_by_tool_family(tasks, 7)
        self.assertEqual(manifest["components"], 1)
        self.assertEqual(len({row["split"] for row in manifest["assignments"].values()}), 1)
        self.assertEqual(manifest["sizes"]["train"], 3)
        self.assertEqual(manifest["forced_large_training_components"][0]["size"], 3)

    def test_official_answer_shape_and_explicit_exclusions(self):
        question = [[{"role": "user", "content": "Use the reader"}]]
        rows = [{"id": "multiple-1", "question": question, "function": [{"name": "reader"}, {"name": "writer"}]}]
        answers = [{"id": "multiple-1", "ground_truth": [{"reader": {"file": ["report"]}}]}]
        tasks, counts = parse_records(rows, answers, [])
        self.assertEqual(tasks[0].gold_names, ("reader",))
        self.assertEqual(counts["retained"], 1)
        answers[0]["ground_truth"] = [{"reader": {}}, {"writer": {}}]
        tasks, counts = parse_records(rows, answers, [])
        self.assertFalse(tasks)
        self.assertEqual(len(counts["skipped"]), 1)

    def test_missing_reference_is_not_joined_by_row_number(self):
        row = {"id": "question", "question": [[{"role": "user", "content": "read"}]], "function": [{"name": "read"}]}
        tasks, counts = parse_records([row], [{"id": "different-id", "ground_truth": [{"read": {}}]}], [])
        self.assertFalse(tasks)
        self.assertIn("matching ID", counts["skipped"][0]["reason"])
        self.assertEqual(counts["orphan_reference_ids"], ["different-id"])

    def test_public_role_context_is_preserved_not_silently_dropped(self):
        messages = [
            {"role": "system", "content": "Use the approved currency."},
            {"role": "assistant", "content": "The account currency is EUR."},
            {"role": "user", "content": "Read my balance."},
        ]
        row = {"id": "context", "question": [messages], "function": [{"name": "balance"}]}
        tasks, counts = parse_records([row], [{"id": "context", "ground_truth": [{"balance": {}}]}], [])
        self.assertEqual(counts["rows_with_non_user_context"], 1)
        self.assertTrue(counts["full_role_context_preserved"])
        self.assertEqual(tasks[0].query, "[system]\nUse the approved currency.\n[assistant]\nThe account currency is EUR.\n[user]\nRead my balance.")

    def test_public_features_masks_and_propensities_ignore_reference_labels(self):
        task = PublicTask("task", "multiple", "read the report", ({"name": "read", "description": "Read a report"}, {"name": "delete", "description": "Delete a file"}), ("read",), {})
        model = TextFeatures().fit([task.observation()])
        matrix, similarity = model.transform(task.observation())
        altered = replace(task, gold_names=("delete",), category="irrelevance")
        np.testing.assert_allclose(matrix, model.transform(altered.observation())[0])
        allowed = allowed_mask(task, "authorization", 7)
        np.testing.assert_array_equal(allowed, allowed_mask(altered, "authorization", 7))
        probabilities = logging_distribution(task, similarity, allowed, "support_gap", 7)
        self.assertAlmostEqual(probabilities.sum(), 1)
        self.assertTrue(np.all(probabilities[~allowed] == 0))
        self.assertGreater(probabilities[-1], 0)

    def test_public_outcomes_distinguish_relevance_from_execution(self):
        task = PublicTask("task", "multiple", "read", ({"name": "read"}, {"name": "delete"}), ("read",), {})
        allowed = np.array([True, True, True])
        self.assertEqual(actual_outcome(task, 0, allowed, "native")["reward"], 1)
        self.assertEqual(actual_outcome(task, 1, allowed, "native")["reward"], -1)
        self.assertFalse(actual_outcome(task, 2, allowed, "native")["correct"])
        denied_gold = np.array([False, True, True])
        self.assertTrue(actual_outcome(task, 2, denied_gold, "authorization")["correct"])
        self.assertTrue(actual_outcome(task, 0, denied_gold, "authorization")["denied"])

    def test_llm_prompt_has_no_gold_and_invalid_json_is_not_abstention(self):
        task = PublicTask("hidden-id", "irrelevance", "read the report", ({"name": "read", "parameters": {"type": "dict", "properties": {"count": {"type": "int"}}, "required": ["count"]}},), (), {})
        self.assertEqual(prompt_messages(task), prompt_messages(replace(task, gold_names=("read",), reference_arguments={"secret": 1})))
        self.assertFalse(score_response(task, "not json")["selection_correct"])
        self.assertTrue(score_response(task, '{"name":null,"arguments":{}}')["selection_correct"])
        valid = score_response(replace(task, gold_names=("read",)), '{"name":"read","arguments":{"count":1}}')
        self.assertTrue(valid["selection_correct"])
        self.assertTrue(valid["argument_schema_valid"])
        invalid = score_response(task, '{"name":"read","arguments":{"count":"one"}}')
        self.assertFalse(invalid["argument_schema_valid"])

    def test_schema_normalization_does_not_rewrite_default_data(self):
        original = {"type": "dict", "properties": {"payload": {"type": "any", "default": {"type": "int"}}}}
        normalized = normalize_schema(original)
        self.assertEqual(normalized["type"], "object")
        self.assertNotIn("type", normalized["properties"]["payload"])
        self.assertEqual(normalized["properties"]["payload"]["default"], {"type": "int"})

    def test_llm_smoke_only_uses_common_training_tasks(self):
        tasks = [PublicTask(str(index), "multiple", f"query {index}", ({"name": "shared"},), (), {}) for index in range(10)]
        selected, memberships = select_tasks(tasks, 4, smoke=True)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(len(memberships[task.task_id]) == 5 for task in selected))


class VersionTwoArtifactTests(unittest.TestCase):
    def test_full_extension_provenance_and_paper_claims(self):
        result = verify_extension(Path(__file__).parent)
        self.assertEqual(result["runs"], 45)
        self.assertEqual(result["paper_claim_checks"], 9)
        self.assertEqual([model["tasks"] for model in result["models"]], [200, 200])


if __name__ == "__main__":
    unittest.main()