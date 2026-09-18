# Counterfactual Tool Ranking

A reproducible research prototype for tool selection under utility, cost, and
privilege constraints. It combines full-action authorization, exact-propensity
logging, direct and doubly robust learning, selective execution, and independent
execution-based evaluation.

## Run locally

Python 3.11+ is required. No credentials, paid APIs, or production data are needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -p 'test_*.py' -v
.venv/bin/python run.py run
```

Reports, figures, per-task outcomes, and partial-feedback logs are written to a
new directory under `results/`. Nonempty output directories are not overwritten.

```sh
.venv/bin/python run.py run --seed 17 --epsilon 0.5
.venv/bin/python run.py run --exclude-logged-tool docs.batch_get
.venv/bin/python run.py replay --log results/RUN/train.jsonl
```

## Research status

The initial version has eight executable SQLite-backed tools in document and
ticket domains. It does not yet connect to an LLM or a live external service.
Costs and service latency are explicit simulated quantities, not API invoices.

The initial deterministic experiment finds that both direct prediction and DR
learning match a hand-written rule baseline. It does **not** establish that DR
is superior. Conservative evidence thresholds sometimes sacrifice useful coverage.
Targets outside the logging policy's support are explicitly unidentifiable.

Next: real local MCP transport, noisy and stale observations, stronger overlap
diagnostics, reproducible multi-seed experiments, and a results-grounded paper.
