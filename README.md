# Counterfactual Tool Ranking

A reproducible study of tool selection under utility, cost, and privilege
constraints: deterministic full-action authorization, exact-propensity logging,
direct/IPS/DR learning, selective execution, and real local MCP transport.

**Read the first working paper:** [PDF](paper/paper.pdf) | [Markdown](paper/paper.md).
**Inspect the results:** [all tables](artifacts/v1/tables.md) | [protocol](EXPERIMENTS.md).

The frozen matrix has 45 runs across nine settings and five seeds, with 387,000
logged decision instances and 90,000 test task-condition instances. Conditions
reuse fixtures; these counts do not represent that many independent unique tasks.

## Run locally

Python 3.11+ is required. No credentials, paid APIs, or production data are needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m unittest discover -p 'test_*.py' -v
.venv/bin/python run.py run
```

Reports, figures, per-task outcomes, and partial-feedback logs are written to a
new directory under `results/`. Nonempty output directories are not overwritten.

```sh
.venv/bin/python run.py run --scenario noisy --seed 17 --epsilon 0.5
.venv/bin/python run.py run --exclude-logged-tool docs.batch_get
.venv/bin/python run.py replay --log results/RUN/train.jsonl
.venv/bin/python run.py run --backend mcp --scenario noisy \
  --train-size 600 --calibration-size 200 --test-size 400
.venv/bin/python suite.py --out results/reproduction
.venv/bin/python verify_artifacts.py
```

## What is implemented

- Eleven executable tools across document reads, ticket closure, and customer
  discounts. Typed resource/numeric arguments and SQLite terminal-state checks.
- User/group scope-resource grants, inherited prefixes, explicit deny precedence,
  server-owned authority, and execution-time revocation checks.
- Official MCP Python SDK 2.2.0 client and independent stdio server. The included
  VS Code MCP configuration exposes a synthetic demo with no production access.
- Exact randomized partial-feedback logs, tool/candidate snapshots, schema/hash
  validation, and semantic replay. Zero logging support is never silently imputed.
- Cross-fitted DR and IPS regression, direct models, matched conservative variants,
  independent calibration, ESS diagnostics, bootstrap intervals and hindsight oracle.
- Clean/noisy/shifted conditions, stale observations, injected failures, data and
  exploration budgets, linear-model ablation, cost and latency sensitivity.
- Fixed multi-seed suite, public per-run summaries, generated figures/tables,
  source hashes, artifact verification, unit/integration tests and GitHub Actions.

## What the results say

DR is useful for **evaluation**, not a universally better ranking algorithm.
With changed conditions, DR evaluation MAE is 0.0187 versus 0.0772 for direct
prediction. With linear outcome models it is 0.0193 versus 0.1039.

For noisy tree-model policy learning, direct utility is 0.5118 and DR is 0.5027.
The conservative evidence floor substantially reduces coverage and does not
provide a business-safety guarantee. Negative results are retained in full.

![Policy comparison](artifacts/v1/policy-comparison.png)

These are **synthetic enterprise-inspired tasks**, not production-agent results.
Real MCP transport does not make costs or workloads real. No LLM, paid API,
external enterprise account, or private trajectory is used. The first version
does not implement natural-language argument generation, multi-step offline RL,
human approval workflows, arbitrary MCP connectors, or large-catalog retrieval.
It is not a drop-in production authorization gateway. See [SECURITY.md](SECURITY.md).

## Build the paper

```sh
npm ci --prefix paper
npx --prefix paper playwright install chromium
npm run pdf --prefix paper
```

The build reads committed experiment JSON, fills the result tables and claims,
renders math offline, and checks the document in headless Chromium. It does not
open or switch a visible browser. See [paper/README.md](paper/README.md).

## Repository map

| File | Responsibility |
| --- | --- |
| `sandbox.py` | Domain state, tool actions, authorization, fixture generation |
| `learning.py` | Exact logger, features, learners, OPE and support diagnostics |
| `execution.py`, `mcp_server.py` | Local and real MCP transports |
| `run.py` | Single experiment and replay CLI |
| `suite.py` | Frozen matrix, aggregate statistics and public exports |
| `test_experiment.py`, `test_mcp.py` | Regression and subprocess protocol tests |
| `verify_artifacts.py` | Recompute published summaries and check source hashes |
| `artifacts/` | Small public metrics and figures; raw logs stay gitignored |
| `paper/` | Manuscript source, rendered first draft, PDF and build tooling |

The original deterministic prototype was developed independently of production
services. This standalone repository contains only research code and synthetic
aggregate data; it includes no application history or deployment credentials.
