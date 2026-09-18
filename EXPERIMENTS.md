# Version 0.2 Frozen Experiment Protocol

This protocol is specified before the complete matrix is run. Exploratory smoke
runs are not included in the paper's aggregate. Every listed seed is retained.

## Common configuration

- Seeds: 7, 17, 23, 31, 47. Split task seeds are seed, seed+1, seed+2;
  action sampling seeds are seed+100, seed+101, seed+102.
- 6,000 training, 1,200 calibration, 2,000 test task decisions per run.
- One independently reset decision per task. Three domains, eleven tools,
  and an always-available abstain action. Candidate templates are fixed.
- Epsilon-greedy logger: epsilon 0.3 over authorized actions including abstain.
- Cost/risk/latency weights: 1/2/0. Extra-resource penalty: 0.05 per resource.
- Three-fold cross-fitting. Trees: 48 estimators, depth 12, leaf minimum 3.
  Linear ablation: Ridge with alpha 10. IPS and DR regression give each context
  total weight one over its candidate-action rows.
- Conservative evidence floor: 5 selected observations per feature stratum.
  Disagreement weights: 0, 0.5, 1; thresholds: 0, 0.5, 0.8, 0.9, 1.
  Direct and DR conservative policies are independently selected using calibration
  DR lower bootstrap endpoints. This is a heuristic, not a uniform guarantee.
- Bootstrap: 200 task-level resamples per run. Paper tables report seed means
  and sample standard deviations, not a claim of statistical significance.

## Matrix

| Setting | Difference from common configuration |
| --- | --- |
| clean | Exact approval observations, deterministic service availability |
| noisy | Stale approval/limit observations, cache freshness and tool failures |
| small_data | Noisy; 600 training decisions |
| low_exploration | Noisy; epsilon 0.05 |
| missing_support | Noisy; three checked/batch tools have logging probability zero |
| shifted | Train/calibration noisy; test has more stale signals, outages and revocation |
| linear | Noisy; all learned score models use the same linear model family |
| cost_sensitive | Noisy; cost weight 3 |
| latency_sensitive | Noisy; latency weight 0.5 per 100 simulated milliseconds |

The frozen implementation is `suite.py`. Run:

```sh
.venv/bin/python suite.py --out results/paper-v1 --publish artifacts/v1
```

Raw logs stay local. The public artifact includes every per-run summary and
frontier, aggregate tables/figures, configuration, source hashes and raw-log hashes.
Wall-clock measurements make raw log hashes machine/run-specific; exact semantic
reproduction compares outcomes excluding runtime and regeneration uses the same
seeds. The public data contain only synthetic observations and aggregate metrics.

## Additional validation

MCP protocol parity is tested over a real subprocess. A noisy MCP end-to-end
run records negotiated protocol, tool catalog, server version and round-trip
latency separately from simulated service latency. Adversarial tests cover unknown
tools, numeric parameters, malformed paths, cross-tenant resources, deny precedence,
scope/resource cross-products, approval checks and authority revocation.

## Reporting rules

Do not discard poor seeds or unidentifiable targets. Report all-task success as
well as execution coverage and success conditional on execution. Compare OPE error
only on the same identified target-policy cases. Exclude the all-abstain policy
and the hindsight oracle from OPE method comparisons. The oracle knows realized
faults and latent state, so it is an upper bound, not a deployable competitor.

Costs, service latency, workloads and business records are synthetic. Real MCP
transport does not turn the workload into production evidence. No LLM, prompt
understanding, external benchmark result, large-catalog scaling claim or general
offline multi-step RL claim is implied by this experiment.
