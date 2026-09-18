# Research Artifact Rules

- Keep all data synthetic unless an explicitly approved public dataset is added.
- Never call production services, upload credentials, or use private traces.
- Report negative results. Do not change test seeds or thresholds to improve a headline.
- Training sees selected-action outcomes only. Keep oracle results evaluation-only.
- Run `python -m unittest discover -p 'test_*.py' -v` after changes.
- MCP uses the official Python SDK v2: https://github.com/modelcontextprotocol/python-sdk
- Client and server documentation: https://py.sdk.modelcontextprotocol.io/client/
  and https://py.sdk.modelcontextprotocol.io/servers/.
- Authority is server-owned fixture state, never caller-supplied policy arguments.
