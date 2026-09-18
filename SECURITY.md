# Security Boundary

This is a local research artifact, not a production security product.

- MCP runs only over stdio. It does not expose a network listener, a generic shell,
  arbitrary file access, or an external-service credential.
- Authority and task state come from a trusted startup manifest. The model/client
  cannot replace the principal, group membership or policy via tool arguments.
- Authorization covers tool, scope, concrete resources and typed numeric fields.
  Explicit denies take precedence. Scope/resource grants stay paired.
- Execution rechecks the authoritative snapshot. Every call has immutable policy
  and isolated SQLite state; production concurrent revoke/write transactions are
  not implemented or proven here.
- Business approvals are separate from ACLs. An authorized but unapproved update
  is counted as unsafe. A weighted reward does not enforce a hard business-risk cap.
- Catalog hashes and replay checks detect changes; they are not signatures or
  guarantees against a malicious logger or a compromised host.
- A zero measured unauthorized-execution count is not a statistical proof of
  zero risk and does not apply to services outside this sandbox.

Do not point a production agent at this artifact as an authorization proxy.
Do not add real user records, API keys or private trajectories to fixtures or
published artifacts. Synthetic logs, downloaded public prompts, model weights
and raw LLM responses remain local by default; public outputs contain derived
task IDs and metrics. Tool descriptions in public data are untrusted text and
are never executed. Network access is limited to explicit public-data/model
downloads; inference remains local. The public permission/cost transforms are
research conditions, not actual BFCL authorization data. Approval escalation
and distributed identity need a separate reviewed implementation before production use.
