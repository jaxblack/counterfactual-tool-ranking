from dataclasses import replace
import unittest

from execution import MCPExecutor
from sandbox import ABSTAIN, TOOLS, Action, Sandbox, legal_actions, make_tasks


class MCPIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = make_tasks(40, 17, "mcp")
        cls.revoked = next(task for task in cls.tasks if task.domain == "tickets" and task.should_act)
        overrides = {cls.revoked.task_id: replace(cls.revoked.policy, version="v2", scopes=())}
        cls.backend = MCPExecutor(cls.tasks, overrides)
        cls.backend.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.backend.__exit__(None, None, None)

    def test_real_subprocess_discovers_registered_tools(self):
        self.assertEqual(set(self.backend.metadata["tools"]), set(TOOLS) | {ABSTAIN})
        self.assertTrue(self.backend.metadata["protocol"])

    def test_protocol_matches_local_execution(self):
        checked = 0
        for task in self.tasks:
            if task.task_id == self.revoked.task_id:
                continue
            for action in legal_actions(task):
                with Sandbox(task) as sandbox:
                    expected = sandbox.execute(action)
                actual = self.backend.execute(task, action)
                self.assertEqual(replace(expected, runtime_ms=0), replace(actual, runtime_ms=0))
                checked += 1
        self.assertGreater(checked, 60)

    def test_server_checks_latest_trusted_policy(self):
        outcome = self.backend.execute(self.revoked, Action("tickets.close_checked", self.revoked.resources))
        self.assertTrue(outcome.denied)
        self.assertEqual(outcome.changed_resources, ())

    def test_server_rejects_cross_tenant_arguments(self):
        task = next(task for task in self.tasks if task.domain == "documents" and task.should_act)
        outcome = self.backend.execute(task, Action("docs.get", ("other/team/private",)))
        self.assertTrue(outcome.denied)
        self.assertFalse(outcome.answer)

    def test_schema_and_unknown_calls_return_errors(self):
        cases = (
            ("unknown.tool", {}),
            ("docs.get", {"task_id": "not-a-task", "resources": []}),
            ("docs.get", {"task_id": self.tasks[0].task_id, "resources": 123}),
            ("crm.discount_one", {"task_id": self.tasks[0].task_id, "resources": [], "amount": True}),
            ("crm.discount_one", {"task_id": self.tasks[0].task_id, "resources": [], "amount": "10"}),
            ("docs.get", {"task_id": self.tasks[0].task_id, "resources": [123]}),
        )
        for tool, arguments in cases:
            with self.subTest(tool=tool, arguments=arguments):
                self.assertTrue(self.backend.call(tool, arguments).is_error)

    def test_noisy_and_shifted_tasks_match_local_results(self):
        tasks = make_tasks(30, 24, "shifted-mcp", scenario="shifted")
        with MCPExecutor(tasks) as backend:
            for task in tasks:
                for action in legal_actions(task):
                    with Sandbox(task) as sandbox:
                        expected = sandbox.execute(action)
                    actual = backend.execute(task, action)
                    self.assertEqual(replace(expected, runtime_ms=0), replace(actual, runtime_ms=0))


if __name__ == "__main__":
    unittest.main()