"""Local-only MCP benchmark server; authority comes from a trusted manifest."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import StrictInt, StrictStr

from sandbox import ABSTAIN, TOOLS, Action, Sandbox, make_tasks, policy_from_dict, task_from_dict


def build_server(manifest: Path | None = None) -> MCPServer:
    payload = json.loads(manifest.read_text(encoding="utf-8")) if manifest else {}
    tasks = [task_from_dict(row) for row in payload["tasks"]] if manifest else make_tasks(24, 7, "demo")
    task_index = {task.task_id: task for task in tasks}
    if len(task_index) != len(tasks):
        raise ValueError("duplicate task IDs in trusted manifest")
    overrides = {}
    for task_id, value in payload.get("execution_policies", {}).items():
        if task_id not in task_index:
            raise ValueError("policy override for an unknown task")
        overrides[task_id] = policy_from_dict(value)
    server = MCPServer(
        "counterfactual-tool-ranking", version="0.2.0", log_level="ERROR",
        instructions="Synthetic local benchmark only. Every call resets task state. No production access.",
    )

    def register(tool_name: str) -> None:
        def invoke(task_id: StrictStr, resources: list[StrictStr], amount: StrictInt | None = None) -> dict[str, Any]:
            if task_id not in task_index:
                raise ToolError("unknown task ID")
            task = task_index[task_id]
            with Sandbox(task) as sandbox:
                outcome = sandbox.execute(Action(tool_name, tuple(resources), amount), overrides.get(task_id))
            return asdict(outcome)

        server.tool(
            name=tool_name, structured_output=True,
            description=f"Execute {tool_name} against an isolated synthetic task; resource ACL is enforced server-side.",
        )(invoke)

    for name in (*TOOLS, ABSTAIN):
        register(name)

    @server.resource("sandbox://tasks")
    def task_catalog() -> str:
        return json.dumps([
            {"task_id": task.task_id, "request": task.request, "resources": task.resources}
            for task in tasks
        ])

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    build_server(args.manifest).run(transport="stdio")


if __name__ == "__main__":
    main()