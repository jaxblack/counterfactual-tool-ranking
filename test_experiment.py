from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from learning import (
    Decision, Learner, Ranking, choose, collect, evaluate_ope,
    features, logging_probabilities, read_log, write_log,
)
from run import parse_args, replay, run_experiment
from sandbox import (
    ABSTAIN, TOOLS, Action, Grant, Outcome, Policy, Sandbox, authorize, candidates, legal_actions,
    make_tasks, task_from_dict, task_to_dict,
)


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.policy = Policy("v1", ("docs.read", "tickets.write"), ("tenant/team",))

    def test_inherited_allow_and_explicit_deny(self):
        action = Action("docs.get", ("tenant/team/folder/document",))
        self.assertTrue(authorize(self.policy, action)[0])
        denied = replace(self.policy, deny_prefixes=("tenant/team/folder",))
        self.assertFalse(authorize(denied, action)[0])

    def test_parameter_level_cross_tenant_and_path_boundary(self):
        for resource in ("other/team/document", "tenant/team-other/document", "tenant/team/../private"):
            with self.subTest(resource=resource):
                self.assertFalse(authorize(self.policy, Action("docs.get", (resource,)))[0])

    def test_unknown_and_missing_scope_fail_closed(self):
        for name in ("unknown", "docs.export"):
            self.assertFalse(authorize(self.policy, Action(name, ("tenant/team/document",)))[0])

    def test_numeric_parameters_and_encoded_paths_are_validated(self):
        policy = replace(self.policy, scopes=("crm.write",))
        for amount in (None, True, -1, 0, 101, "10"):
            self.assertFalse(authorize(policy, Action("crm.discount_one", ("tenant/team/customer",), amount))[0])
        for resource in ("tenant/team/%2e%2e/private", "tenant/team/child\\secret", "tenant/team/x\x00"):
            self.assertFalse(authorize(self.policy, Action("docs.get", (resource,)))[0])
        self.assertTrue(authorize(policy, Action("crm.discount_one", ("tenant/team/customer",), 10))[0])

    def test_abstain_is_always_available_without_resource_arguments(self):
        empty = Policy("v2", (), ())
        self.assertTrue(authorize(empty, Action(ABSTAIN))[0])
        self.assertFalse(authorize(empty, Action(ABSTAIN, ("tenant/team/document",)))[0])

    def test_group_and_user_grants_do_not_form_a_scope_resource_cross_product(self):
        policy = Policy(
            "v1", ("docs.read", "tickets.write"), ("tenant",), principal="alice", groups=("readers",),
            grants=(Grant("group:readers", "docs.read", "tenant/docs"), Grant("user:alice", "tickets.write", "tenant/tickets")),
        )
        self.assertTrue(authorize(policy, Action("docs.get", ("tenant/docs/report",)))[0])
        self.assertTrue(authorize(policy, Action("tickets.close_one", ("tenant/tickets/task",)))[0])
        self.assertFalse(authorize(policy, Action("docs.get", ("tenant/tickets/task",)))[0])
        self.assertFalse(authorize(replace(policy, groups=()), Action("docs.get", ("tenant/docs/report",)))[0])
        denied = replace(policy, grants=policy.grants + (Grant("user:alice", "docs.read", "tenant/docs/private", "deny"),))
        self.assertFalse(authorize(denied, Action("docs.get", ("tenant/docs/private/report",)))[0])


class SandboxTests(unittest.TestCase):
    def setUp(self):
        tasks = make_tasks(100, 17)
        self.document = next(task for task in tasks if task.domain == "documents" and task.should_act)
        self.ticket = next(task for task in tasks if task.domain == "tickets" and task.should_act)

    def test_fresh_body_uses_actual_document_content(self):
        task = replace(self.document, field="body", fresh=True)
        with Sandbox(task) as sandbox:
            cached = sandbox.execute(Action("docs.cache", task.resources))
            live = sandbox.execute(Action("docs.batch_get", task.resources))
        self.assertFalse(cached.success)
        self.assertTrue(live.success)
        self.assertEqual(dict(live.answer)[task.resources[0]], task.records[0].body)

    def test_single_get_cannot_solve_multi_document_request(self):
        task = replace(self.document, resources=tuple(record.resource_id for record in self.document.records[:3]))
        with Sandbox(task) as sandbox:
            self.assertFalse(sandbox.execute(Action("docs.get", task.resources)).success)
            self.assertTrue(sandbox.execute(Action("docs.batch_get", task.resources)).success)

    def test_revocation_between_ranking_and_execution_blocks_mutation(self):
        action = Action("tickets.close_checked", self.ticket.resources)
        self.assertTrue(authorize(self.ticket.policy, action)[0])
        revoked = replace(self.ticket.policy, version="v2", scopes=())
        with Sandbox(self.ticket) as sandbox:
            before = sandbox.states()
            outcome = sandbox.execute(action, latest_policy=revoked)
            self.assertTrue(outcome.denied)
            self.assertEqual(before, sandbox.states())

    def test_guarded_close_is_atomic_when_approval_is_missing(self):
        records = tuple(replace(record, approved=False) for record in self.ticket.records)
        task = replace(self.ticket, records=records)
        with Sandbox(task) as sandbox:
            before = sandbox.states()
            checked = sandbox.execute(Action("tickets.close_checked", task.resources))
            self.assertFalse(checked.success)
            self.assertEqual(before, sandbox.states())
            unchecked = sandbox.execute(Action("tickets.close_one", task.resources))
            self.assertTrue(unchecked.unsafe)
            self.assertNotEqual(before, sandbox.states())

    def test_checked_close_matches_database_end_state(self):
        with Sandbox(self.ticket) as sandbox:
            result = sandbox.execute(Action("tickets.close_checked", self.ticket.resources))
            self.assertTrue(result.success)
            for resource in self.ticket.resources:
                self.assertEqual(sandbox.states()[resource], "closed")

    def test_unknown_resource_and_wrong_domain_do_not_mutate(self):
        actions = (
            Action("tickets.close_checked", self.ticket.resources + ("tenant/team/missing",)),
            Action("docs.get", self.ticket.resources),
        )
        for action in actions:
            with Sandbox(self.ticket) as sandbox:
                before = sandbox.states()
                outcome = sandbox.execute(action)
                self.assertFalse(outcome.success)
                self.assertEqual(before, sandbox.states())

    def test_every_task_has_abstain_and_masks_foreign_arguments(self):
        for task in make_tasks(200, 29):
            actions = legal_actions(task)
            self.assertIn(Action(ABSTAIN), actions)
            self.assertTrue(all("other/team/private" not in action.resources for action in actions))
            self.assertGreater(len(candidates(task)), len(actions))

    def test_serialization_and_generation_are_reproducible(self):
        self.assertEqual(make_tasks(10, 2), make_tasks(10, 2))
        self.assertEqual(self.document, task_from_dict(task_to_dict(self.document)))

    def test_crm_limit_and_atomic_end_state(self):
        task = next(task for task in make_tasks(100, 18) if task.domain == "crm" and task.should_act)
        with Sandbox(task) as sandbox:
            valid = sandbox.execute(Action("crm.discount_checked", task.resources, task.amount))
            self.assertTrue(valid.success)
            self.assertTrue(all(sandbox.discounts()[resource] == task.amount for resource in task.resources))
        limited = replace(task, records=tuple(replace(record, discount_limit=1) for record in task.records))
        with Sandbox(limited) as sandbox:
            guarded = sandbox.execute(Action("crm.discount_checked", task.resources, task.amount))
            self.assertFalse(guarded.success)
            self.assertTrue(all(value == 0 for value in sandbox.discounts().values()))
            unguarded = sandbox.execute(Action("crm.discount_one", task.resources, task.amount))
            self.assertTrue(unguarded.unsafe)

    def test_injected_failure_is_replayable_and_cannot_mutate(self):
        task = replace(self.ticket, failed_tools=("tickets.close_checked",))
        with Sandbox(task) as sandbox:
            before = sandbox.states()
            outcome = sandbox.execute(Action("tickets.close_checked", task.resources))
            self.assertIn("injected service failure", outcome.error)
            self.assertEqual(before, sandbox.states())
        self.assertEqual(task, task_from_dict(task_to_dict(task)))

    def test_noisy_generation_contains_stale_signals_and_revocation(self):
        tasks = make_tasks(400, 19, scenario="shifted")
        self.assertEqual(tasks, make_tasks(400, 19, scenario="shifted"))
        self.assertTrue(any(task.failed_tools for task in tasks))
        self.assertTrue(any(task.execution_policy for task in tasks))
        self.assertTrue(any(task.observed_approval != task.records[0].approved for task in tasks))
        for task in tasks:
            self.assertEqual(task, task_from_dict(task_to_dict(task)))

class LearningTests(unittest.TestCase):
    def test_probabilities_are_post_mask_and_sum_to_one(self):
        for task in make_tasks(100, 5):
            actions = legal_actions(task)
            probabilities = logging_probabilities(actions, 0.3)
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertTrue(all(probability > 0 for probability in probabilities))
            self.assertTrue(all(authorize(task.policy, action)[0] for action in actions))

    def test_log_roundtrip_and_tampered_propensity(self):
        decisions = collect(make_tasks(10, 7), 0.3, 8)
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "log.jsonl"
            write_log(file_path, decisions, {"epsilon": 0.3})
            self.assertEqual(decisions, read_log(file_path))
            rows = file_path.read_text().splitlines()
            first = json.loads(rows[0])
            first["propensity"] = -1
            rows[0] = json.dumps(first)
            file_path.write_text("\n".join(rows))
            with self.assertRaisesRegex(ValueError, "propensity"):
                read_log(file_path)

    def test_training_features_do_not_read_hidden_content_or_task_id(self):
        task = make_tasks(1, 9)[0]
        action = legal_actions(task)[0]
        changed = replace(task, task_id="different", records=())
        self.assertEqual(features(task, action), features(changed, action))

    def test_logging_only_executes_selected_actions(self):
        tasks = make_tasks(30, 4)
        with patch("learning.Sandbox.execute", autospec=True, side_effect=Sandbox.execute) as execute:
            decisions = collect(tasks, 0.4, 5, ("docs.batch_get",))
        self.assertEqual(execute.call_count, len(tasks))
        for decision in decisions:
            self.assertNotEqual(decision.actions[decision.chosen].tool, "docs.batch_get")
            for action, probability in zip(decision.actions, decision.probabilities, strict=True):
                if action.tool == "docs.batch_get":
                    self.assertEqual(probability, 0)

    def test_ope_known_answer_and_zero_support(self):
        task = next(task for task in make_tasks(30, 12) if task.domain == "documents" and task.should_act)
        actions = (Action("docs.get", task.resources), Action(ABSTAIN))
        ranking = Ranking(
            actions, np.array([0.4, 0.0]), np.zeros(2), np.zeros(2),
            np.ones(2, dtype=bool), np.ones(2), np.zeros(2),
        )
        decisions = [
            Decision(task, actions, (0.5, 0.5), 0, Outcome(True, 0.08, 25)),
            Decision(task, actions, (0.5, 0.5), 1, Outcome(False, 0, 0, abstained=True)),
        ]
        result = evaluate_ope(decisions, [ranking, ranking], [0, 0], bootstrap=20)
        self.assertAlmostEqual(result["ips"], 0.92)
        self.assertAlmostEqual(result["snips"], 0.92)
        self.assertAlmostEqual(result["dr"], 0.92)
        self.assertAlmostEqual(result["ess"], 1.0)
        unsupported = [replace(decisions[1], probabilities=(0.0, 1.0))]
        result = evaluate_ope(unsupported, [ranking], [0], bootstrap=20)
        self.assertFalse(result["identifiable"])
        self.assertIsNone(result["dr"])

    def test_learner_runs_without_unobserved_outcomes(self):
        decisions = collect(make_tasks(300, 10, "train"), 0.4, 11)
        learner = Learner(12).fit(decisions)
        tasks = make_tasks(20, 13, "test")
        rankings = learner.rank(tasks)
        self.assertEqual(len(tasks), len(rankings))
        for task, ranking in zip(tasks, rankings, strict=True):
            for policy in ("direct", "ips", "dr", "conservative_direct", "conservative_dr", "rules", "schema_match"):
                chosen = choose(task, ranking, policy)
                self.assertTrue(authorize(task.policy, ranking.actions[chosen])[0])

    def test_unsupported_and_low_evidence_actions_fall_back(self):
        task = next(task for task in make_tasks(30, 12) if task.domain == "documents" and task.should_act)
        actions = (Action("docs.get", task.resources), Action(ABSTAIN))
        ranking = Ranking(
            actions, np.array([0.9, 0.0]), np.array([0.9, 0.0]), np.zeros(2),
            np.array([False, True]), np.array([0, 10]), np.zeros(2),
        )
        self.assertEqual(choose(task, ranking, "direct"), 0)
        self.assertEqual(choose(task, ranking, "dr"), 1)
        ranking.supported[0] = True
        self.assertEqual(choose(task, ranking, "dr"), 0)
        self.assertEqual(choose(task, ranking, "conservative_dr"), 1)

    def test_linear_models_and_latency_penalty(self):
        tasks = make_tasks(120, 8, scenario="noisy")
        learner = Learner(8, latency_weight=0.5, model="linear").fit(collect(tasks, 0.4, 9))
        rankings = learner.rank(tasks[:10])
        for task, ranking in zip(tasks[:10], rankings, strict=True):
            for name in ("direct", "ips", "dr", "conservative_direct", "conservative_dr"):
                self.assertTrue(authorize(task.policy, ranking.actions[choose(task, ranking, name)])[0])
        self.assertAlmostEqual(Outcome(True, 0.1, 50).utility(latency_weight=0.5), 0.65)

    def test_latent_failures_and_true_approval_never_enter_features(self):
        task = make_tasks(1, 4, scenario="noisy")[0]
        action = legal_actions(task)[0]
        changed = replace(task, failed_tools=tuple(TOOLS), records=(), execution_policy=replace(task.policy, scopes=()))
        self.assertEqual(features(task, action), features(changed, action))

    def test_rejects_mismatched_candidates_and_nonfinite_outcomes(self):
        decisions = collect(make_tasks(12, 7), 0.3, 8)
        rankings = Learner(7).fit(decisions).rank([decision.task for decision in decisions])
        with self.assertRaisesRegex(ValueError, "candidate order"):
            evaluate_ope(decisions, rankings, [len(decision.actions) for decision in decisions])
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "invalid.jsonl"
            invalid = replace(decisions[0], outcome=replace(decisions[0].outcome, cost=float("nan")))
            write_log(file_path, [invalid], {})
            with self.assertRaisesRegex(ValueError, "invalid outcome"):
                read_log(file_path)

    def test_log_rejects_changed_catalog_or_candidate_evidence(self):
        decisions = collect(make_tasks(2, 8), 0.3, 9)
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "log.jsonl"
            for target in ("catalog_sha256", "candidates"):
                write_log(file_path, decisions, {})
                rows = file_path.read_text().splitlines()
                raw = json.loads(rows[0])
                raw[target] = "tampered" if target == "catalog_sha256" else []
                rows[0] = json.dumps(raw)
                file_path.write_text("\n".join(rows))
                with self.assertRaisesRegex(ValueError, "changed"):
                    read_log(file_path)


class CommandLineTests(unittest.TestCase):
    def test_complete_pipeline_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = parse_args([
                "run", "--train-size", "120", "--calibration-size", "40", "--test-size", "40",
                "--bootstrap", "20", "--out", str(output),
            ])
            with redirect_stdout(io.StringIO()):
                summary = run_experiment(args)
                replay_code = replay(output / "train.jsonl")
            self.assertEqual(replay_code, 0)
            self.assertEqual(summary["config"]["test_size"], 40)
            self.assertGreater((output / "frontier.png").stat().st_size, 1000)
            self.assertTrue((output / "report.md").exists())
            self.assertEqual(json.loads((output / "summary.json").read_text())["schema_version"], 1)
            task_ids = [
                {decision.task.task_id for decision in read_log(output / f"{split}.jsonl")}
                for split in ("train", "calibration", "test")
            ]
            self.assertFalse(task_ids[0] & task_ids[1] or task_ids[0] & task_ids[2] or task_ids[1] & task_ids[2])
            for result in summary["policies"].values():
                self.assertEqual(result["actual"]["unauthorized_executions"], 0)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                run_experiment(args)

    def test_replay_detects_tampered_outcome(self):
        decisions = collect(make_tasks(4, 2), 0.3, 7)
        first = decisions[0]
        decisions[0] = replace(first, outcome=replace(first.outcome, success=not first.outcome.success))
        with tempfile.TemporaryDirectory() as directory:
            file_path = Path(directory) / "log.jsonl"
            write_log(file_path, decisions, {"epsilon": 0.3})
            with redirect_stdout(io.StringIO()):
                self.assertEqual(replay(file_path), 1)

    def test_nonfinite_weights_and_invalid_sizes_are_rejected(self):
        for arguments in (
            ["run", "--cost-weight", "nan"], ["run", "--risk-weight", "inf"],
            ["run", "--train-size", "0"], ["run", "--epsilon", "1.1"],
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(arguments)


if __name__ == "__main__":
    unittest.main()