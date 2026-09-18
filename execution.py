"""Interchangeable local and real stdio MCP execution with reset-per-call state."""

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time

from anyio.from_thread import start_blocking_portal
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from sandbox import ABSTAIN, TOOLS, Action, Outcome, Policy, Sandbox, Task, task_to_dict


def outcome_from_dict(payload: dict) -> Outcome:
    return Outcome(**{
        **payload,
        "changed_resources": tuple(payload["changed_resources"]),
        "answer": tuple(tuple(pair) for pair in payload["answer"]),
    })


class LocalExecutor:
    def __init__(self, tasks: list[Task], execution_policies: dict[str, Policy] | None = None):
        self.execution_policies = execution_policies or {}
        self.metadata = {"backend": "local", "calls": 0}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, task: Task, action: Action) -> Outcome:
        self.metadata["calls"] += 1
        with Sandbox(task) as sandbox:
            return sandbox.execute(action, self.execution_policies.get(task.task_id))


class MCPExecutor:
    def __init__(self, tasks: list[Task], execution_policies: dict[str, Policy] | None = None):
        self.tasks = tasks
        self.execution_policies = execution_policies or {}
        self.stack = ExitStack()
        self.metadata = {"backend": "mcp-stdio", "calls": 0}
        self.roundtrip_ms: list[float] = []

    def __enter__(self):
        try:
            directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="tool-ranking-")))
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({
                "tasks": [task_to_dict(task) for task in self.tasks],
                "execution_policies": {key: asdict(value) for key, value in self.execution_policies.items()},
            }), encoding="utf-8")
            self.portal = self.stack.enter_context(start_blocking_portal())
            parameters = StdioServerParameters(
                command=sys.executable,
                args=[str(Path(__file__).with_name("mcp_server.py")), "--manifest", str(manifest)],
            )
            self.client = self.stack.enter_context(self.portal.wrap_async_context_manager(
                Client(parameters, read_timeout_seconds=15),
            ))
            tools = self.portal.call(self.client.list_tools)
            catalog = {tool.name for tool in tools.tools}
            if catalog != set(TOOLS) | {ABSTAIN}:
                raise RuntimeError("MCP tool catalog differs from experiment registry")
            self.metadata.update({
                "protocol": self.client.protocol_version,
                "tools": sorted(catalog),
                "server_version": self.client.server_info.version if self.client.server_info else "unknown",
            })
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def call(self, name: str, arguments: dict):
        return self.portal.call(self.client.call_tool, name, arguments)

    def execute(self, task: Task, action: Action) -> Outcome:
        started = time.perf_counter()
        response = self.call(action.tool, {
            "task_id": task.task_id, "resources": list(action.resources), "amount": action.amount,
        })
        self.roundtrip_ms.append((time.perf_counter() - started) * 1000)
        self.metadata["calls"] += 1
        if response.is_error or not isinstance(response.structured_content, dict):
            raise RuntimeError(f"MCP transport/tool error for {action.tool}: {response.content}")
        return outcome_from_dict(response.structured_content)


def executor_for(name: str, tasks: list[Task]):
    if name == "local":
        return LocalExecutor(tasks)
    if name == "mcp":
        return MCPExecutor(tasks)
    raise ValueError(f"unknown execution backend: {name}")