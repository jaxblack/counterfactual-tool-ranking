# Counterfactual Tool Ranking under Utility, Cost, and Privilege Constraints

## A Reproducible Empirical Study with Executable Enterprise-Inspired Tools

Working paper, version 1. September 2026. Not peer reviewed.

Code and experimental artifacts: <https://github.com/jaxblack/counterfactual-tool-ranking>.

## Abstract

Tool-using agents must choose among actions that differ in utility, cost, latency,
and authority. Their execution logs contain outcomes for selected actions, not
for every available alternative. This suggests transferring contextual-bandit
methods from advertising and recommendation, but three distinctions matter:
authorization is not a statistical prediction; legal actions may lack historical
support; and reliable off-policy evaluation does not imply that an off-policy
learner produces the best policy. We present a reproducible implementation that
separates full-action authorization, exact-propensity logging, policy learning,
and selective execution. Eleven executable tools operate on resettable SQLite
state in document, ticket, and customer-discount domains. An official Model
Context Protocol (MCP) client and independent stdio server execute the same
workload over a real protocol boundary. A fixed matrix of nine conditions and
five seeds contains {{decisions}} logged decision instances and
{{task_conditions}} held-out task-condition instances. Doubly robust (DR)
evaluation reduces mean absolute error relative to direct prediction from
{{shifted_dm_error}} to {{shifted_dr_error}} under a shifted test distribution,
and from {{linear_dm_error}} to {{linear_dr_error}} with linear outcome models.
DR policy learning does not consistently outperform direct prediction: noisy
tree-model utilities are {{noisy_dr}} and {{noisy_direct}}, respectively.
Evidence-stratum abstention substantially reduces coverage and is not a safety
guarantee. Our contribution is an executable, support-aware experimental contract
and an empirical account of where established counterfactual methods help or
fail, not a new DR estimator or a production security claim.

**Keywords:** tool selection; contextual bandits; off-policy evaluation; doubly
robust estimation; least privilege; selective execution; Model Context Protocol.

## 1. Introduction

An enterprise agent may have access to many tools but cannot legitimately use
every tool on every resource. Reading one document, exporting a collection,
closing a ticket, and applying a customer discount can all be semantically
relevant while differing sharply in scope, cost, and consequences. Ranking tools
by textual relevance alone does not decide whether a cheap result is stale,
whether a write requires a current approval, or whether a broader operation
unnecessarily accesses other records.

There is a useful connection to advertising systems. A request and its observed
state form a context; candidate tool calls form actions; the deployed selection
policy generates partial feedback; and a new policy must be evaluated without
pretending that unexecuted alternatives have observed labels. The connection is
not an equivalence. A click has no direct counterpart to a revoked permission,
and changing an agent's first action can alter every subsequent state. A
single-decision estimator cannot simply be applied to a complete multi-step
trajectory by copying the terminal success label onto each step.

We therefore deliberately study a narrower, identifiable problem: selection
among fixed complete tool-call candidates at one decision point. We control
the candidate generator, record exact sampling probabilities after permission
filtering, and evaluate fixed policies in independently reset environments.
No language model is used to infer an authorization decision or to judge task
success. This isolates the decision-learning question from natural-language
parsing and long-horizon credit assignment.

The empirical questions are:

1. Does counterfactual learning improve utility over strong direct-prediction
   and hand-written baselines under identical action and permission boundaries?
2. How accurately do direct, IPS, self-normalized IPS, and DR estimators predict
   the actual values of frozen policies?
3. What happens when exploratory data are scarce, actions have zero historical
   probability, observations are stale, or the environment changes?
4. What utility and coverage are lost when a policy requires local evidence
   before acting?

The results reject a blanket claim that DR ranking is always superior. They
instead distinguish two uses of counterfactual reasoning: constructing a policy
and evaluating a fixed policy. In our experiments the strongest and most
consistent gains are in the second use, especially when the outcome model is
misspecified or transferred to changed conditions.

## 2. Related Work and Scope

Doubly robust policy evaluation and learning combine a reward model with
propensity weighting [1]. Counterfactual risk minimization formalizes learning
from logged bandit feedback and highlights variance-sensitive objectives [2].
Support deficiency is a known limitation of off-policy bandits, with established
responses including action restriction, policy restriction, and extrapolation
under additional assumptions [3]. Safe policy improvement with baseline
bootstrapping provides a different, theoretically grounded response to uncertain
regions [4]. Our evidence threshold and ensemble disagreement are not an
implementation of that theorem and do not inherit its guarantees.

Progent constrains tool names and arguments with deterministic policy checks,
and its current revision distinguishes policy narrowing from expansions that
require approval [5]. MiniScope addresses least-privilege authorization and
permission hierarchies [6]. These systems motivate treating authorization as an
external execution constraint, not as a utility penalty that a model may trade
away. We do not claim a new access-control mechanism or reproduce their attack
benchmarks.

Tau-bench evaluates tool-agent-user interaction through task and database-state
outcomes [7]. AgentAbstain studies paired should-act and should-abstain tasks
[8]. Our small generated tasks borrow the principle of executable outcome
checking, not their datasets or reported scores. We have not run either
external benchmark. Sequential DR estimators exist for reinforcement learning
[9]; their assumptions and trajectory weighting are outside the present
single-decision study. Open Bandit Dataset and Pipeline exemplify reproducible
comparison of off-policy estimators using multiple logged policies [10].

Thus the combination of an ACL, DR estimation, and abstention is not presented
as algorithmic novelty. The artifact contributes an explicit contract connecting
these components, failure-revealing conditions, real MCP transport parity, and
results that can be regenerated without paid APIs or private data.

## 3. Problem Formulation

### 3.1 Complete actions and feasible sets

Let $x$ contain the request's structured pre-action observations. Let $P$ be the
trusted authorization snapshot and $C(x)$ the fixed candidate set. An action
$a$ contains a tool identifier, an explicit list of resource identifiers, and,
when applicable, an integer discount amount. The authorized candidate set is

$$
A(x,P)=\{a\in C(x):\operatorname{Auth}(P,a)=1\}\cup\{\bot\},
$$

where $\bot$ denotes abstention. The ranking policy satisfies
$\pi(a\mid x,P,C)=0$ outside this set. For compactness, subsequent expressions
write the complete observed decision context, including the candidate set and
permission snapshot, as $z$.

The candidate generator is fixed for every compared policy. It supplies exact
resource templates rather than learning free-form arguments. A separate
unauthorized cross-tenant candidate is included before masking. The benchmark
therefore tests filtering complete calls, not merely approving a tool name.

### 3.2 Utility and non-negotiable authorization

For an executed action, the measured utility is

$$
r = s-\lambda_c c-\lambda_l\frac{\ell}{100}-\lambda_u u-0.05e,
$$

where $s$ is verified task success, $c$ is simulated service cost, $\ell$ is
simulated latency in milliseconds, $u$ indicates an unsafe business outcome,
and $e$ counts supplied resource arguments outside the requested resource set.
The default weights are $\lambda_c=1$, $\lambda_l=0$, and $\lambda_u=2$.
Abstention yields $s=r=0$. Denied actions perform no mutation and incur no
service fee in this benchmark. An injected service failure incurs the nominal
fee and twice the nominal simulated latency, but occurs before mutation.

Authorization and $u$ are different axes. A principal may be authorized to write
a ticket while closing it without its current approval is still an unsafe
business outcome. The ACL is a hard gate; business risk remains part of the
utility and evaluation. We do not allow a high predicted reward to bypass an
ACL denial. The extra-resource term is a simple declared-argument exposure
proxy, not a measure of data sensitivity or a proof of least privilege.

### 3.3 Partial feedback and support

Each log entry contains $(z_i,a_i,\mu_i,r_i)$, together with the complete
post-mask probability vector and reproducible environment state. Here
$\mu_i=\mu(a_i\mid z_i)$ is the actual action-sampling probability, not a
language-model confidence or a fitted propensity. The learning interface only
receives the selected action's result.

Identifying a target value requires overlap:

$$
\pi(a\mid z)>0\ \Longrightarrow\ \mu(a\mid z)>0.
$$

Permission does not imply overlap. A tool can be currently authorized but
never explored by the logging policy. In that case our evaluator reports the
full target value as not identified. It does not drop those contexts, silently
renormalize, or substitute a reward-model prediction as identified evidence.
Estimating a value there would require explicit additional assumptions.

## 4. System and Threat Model

### 4.1 Authorization path

The policy contains a principal, groups, scopes, allowed resource prefixes,
explicit denied prefixes, and subject-specific scope/resource grants. Prefix
inheritance respects path segment boundaries. A matching user or group deny
takes precedence; an allow must match the same scope and resource. Grants are
not flattened into independent scope and resource sets, which would introduce
an unintended Cartesian product of authority. Unknown tools and malformed
resource identifiers fail closed. Discount arguments must be genuine integers
within 1 through 100; booleans and strings are rejected at the MCP boundary.

Every candidate is filtered before ranking. Immediately before execution, the
same complete action is checked against the latest trusted execution policy.
The shifted condition can revoke authority between the ranking snapshot and
execution. Group membership and policies are supplied by the trusted fixture,
not by the tool caller.

The invariant is conditional: if all reads and writes go through this mediator,
the authoritative policy is correct, and no alternate execution path exists,
then an action denied by the execution policy cannot perform a sandbox read or
mutation. Tests inspect both returned outcomes and database state. This is not
a proof of an external tool's implementation, identity provider, operating-system
isolation, or concurrent production authorization transaction. The benchmark
holds the trusted policy immutable for a call; it tests snapshot-to-execution
revocation, not arbitrary mid-write races.

### 4.2 Real protocol boundary, synthetic services

Two backends share the same domain implementation. The local backend invokes
the sandbox directly. The MCP backend uses the official Python SDK 2.2.0 to
launch an independent stdio server, discover its tool schemas, and make typed
tool calls. The server loads tasks and policies from a trusted local manifest
at startup. Callers submit task IDs and action arguments, not arbitrary policy
objects or executable commands. No service listens on a network interface.

Each call creates fresh in-memory SQLite state. This makes policy comparisons
independent and replayable. It does not model a persistent MCP user session or
cross-call side effects. Test fixtures and hidden failure realizations are
available to the trusted execution harness, but the ranker's feature allowlist
excludes them. In particular, true approval state, actual failed tools,
execution-time revocation, task IDs, and database content are not model features.

### 4.3 Logging and reproducibility

The logger is epsilon-greedy over authorized actions: it chooses the cheapest
executable action with probability $1-\epsilon$ and explores uniformly over the
eligible authorized set, including abstain, with probability $\epsilon$.
If no executable action exists, it chooses abstain. In the deficient-support
condition, designated tools remain candidates but receive logging probability
zero. Only one action is executed per collection decision.

Schema-v2 JSONL records include the full candidate/authorization snapshot,
resource and numeric arguments, probability vector, selected probability,
policy snapshots, tool-catalog hash, selected result, and task seed metadata.
Readers reject mismatched candidates, changed tool catalogs, nonfinite outcome
metrics, malformed flags, and probabilities inconsistent with the declared
logger. Replays compare semantic outcomes while excluding measured runtime.
Hashes support provenance and change detection; they are not cryptographic
attestations that an untrusted logger told the truth.

## 5. Learning and Evaluation Methods

### 5.1 Direct and counterfactual score models

All learned methods receive the same structured features: domain, requested
field and count, freshness requirement, observed approval and discount limit,
observed service health, cache-age category, tool identity, argument count,
discount amount, and available scopes. No text embedding or hidden database
label is used. Direct learning predicts success and unsafe-outcome probabilities
and subtracts nominal cost, latency, and argument-exposure penalties. Values of
probability predictions are clipped to $[0,1]$.

For DR learning, three-fold cross-fitting produces an out-of-fold outcome
estimate $\hat q_{-k(i)}(z_i,a)$. Every supported candidate receives the target

$$
\tilde r_i^{\rm DR}(a)=\hat q_{-k(i)}(z_i,a)+
\frac{\mathbf 1\{a_i=a\}}{\mu(a_i\mid z_i)}
\left[r_i-\hat q_{-k(i)}(z_i,a_i)\right].
$$

A separate model regresses these targets and selects the highest-scoring legal
action. The IPS learner uses
$\tilde r_i^{\rm IPS}(a)=\mathbf 1\{a_i=a\}r_i/\mu(a_i\mid z_i)$.
Unsupported candidate rows are excluded. Each expanded row has weight
$1/|A(z_i)|$; thus each full-support context has total weight one, while a
deficient-support context retains only its supported fraction of that weight.
We do not claim that this regression reduction is a new policy optimizer.

Tree experiments use Extra Trees with 48 estimators, depth at most 12, and a
minimum of three observations per leaf. The linear ablation uses Ridge
regression with regularization 10. Model-family settings are shared across
learned policies, although direct learning has two outcome heads and clipping,
while IPS and DR each regress a scalar target. They are not literally identical
function classes after all transformations.

### 5.2 Evidence-aware selective execution

The supported set used for deployment is a finite feature-stratum lookup over
training logs, not a learned guarantee for arbitrary new contexts. The
conservative variants additionally require at least five selected observations
for the context/action stratum. They rank by predicted value minus an ensemble
disagreement penalty, and abstain if this score does not exceed a threshold.

Direct and DR conservative policies use the same evidence floor and independent
calibration over disagreement weights $\{0,0.5,1\}$ and thresholds
$\{0,0.5,0.8,0.9,1\}$. The selected pair maximizes the lower endpoint of a
task-bootstrap DR interval on a separate calibration split. A fully unsupported
calibration set falls back to abstention. Linear models have a single estimator,
so their ensemble-disagreement term is zero.

This procedure is intentionally described as a heuristic. Tree disagreement is
not a calibrated conditional confidence bound. Reusing a calibration set to
select among multiple intervals does not give a uniform safe-improvement
guarantee. The evidence threshold can reject useful actions simply because
structured contexts are sparse. Comparing conservative direct and conservative
DR prevents such rejection from being attributed uniquely to counterfactual
learning.

### 5.3 Off-policy estimators

For a frozen deterministic target $\pi$, let
$w_i=\mathbf 1\{a_i=\pi(z_i)\}/\mu(a_i\mid z_i)$.
We compare direct prediction, IPS, self-normalized IPS, and DR:

$$
\widehat V_{\rm DM}=\frac1n\sum_i\hat q(z_i,\pi(z_i)),\qquad
\widehat V_{\rm IPS}=\frac1n\sum_iw_ir_i,
$$

$$
\widehat V_{\rm SNIPS}=\frac{\sum_iw_ir_i}{\sum_iw_i},
$$

$$
\widehat V_{\rm DR}=\frac1n\sum_i\left[
\hat q(z_i,\pi(z_i))+w_i(r_i-\hat q(z_i,a_i))\right].
$$

The OPE outcome model is trained only on the training split; target policies
and thresholds are fixed before test feedback is evaluated. No propensity
clipping is applied. We report effective sample size
$\mathrm{ESS}=(\sum_iw_i)^2/\sum_iw_i^2$ and warn about low ESS or no matches.
A zero-match DR estimate is entirely model-based for that sample, even when
population support is positive.

Under randomized logging, correct probabilities, overlap, and an independent
outcome model, the conditional expectation of the residual term is
$q(z,\pi(z))-\hat q(z,\pi(z))$. It cancels the direct-model error. This is the
standard DR identity [1], not a new theorem. Finite-sample variance, regression
approximation, and the nonlinear maximization step explain why a good value
estimator need not produce the strongest learned ranking policy.

## 6. Experimental Design

### 6.1 Executable workloads

Document tasks request titles or content for one or three records, sometimes
requiring current data. Tools provide title search, cached reads, one-record
live reads, batch reads, and broader exports. Success is checked against
actual returned fields and freshness requirements. A one-record tool cannot
satisfy a three-record task.

Ticket tasks close one or three records subject to current approval. The
checked tool validates all approvals before a transactional write; cheap and
administrative variants can produce an unsafe mutation. Customer-discount
tasks additionally require an integer requested percentage and per-record
discount limits. Their checked variant is likewise atomic. Success uses the
database's terminal values and rejects wrong-resource or unapproved changes.

Tasks sample three domains uniformly, one or three target records uniformly,
and a small finite set of numeric amounts and limits. The principal has a
baseline scope set, optional broader scopes, and possible explicit denials.
Nominal costs range from 0.01 to 0.16 simulated units; nominal latency ranges
from 5 to 45 milliseconds. The all-abstain policy provides a zero-utility floor.
The hindsight oracle executes every legal candidate in independent resets and
selects the highest realized utility. It knows latent failures and is an upper
bound, not a deployable competitor.

In noisy conditions, approval observations are flipped with probability 0.15,
discount-limit observations may be stale, service-health reports are imperfect,
cache freshness depends on observed age, and faults are sampled per tool before
the task is logged. A shifted condition increases stale-observation probability
to 0.30 and service degradation probability from 0.25 to 0.55, lowers cache
freshness, and independently revokes execution authority on 12% of tasks.
Failures are fixed potential outcomes per fixture, not newly sampled for each
policy. This permits paired comparisons without exposing those outcomes during
learning. Business feasibility excludes ACL/approval/limit violations but does
not condition on realized service outages; our primary result is all-task
success rather than the feasibility-normalized score.

### 6.2 Frozen matrix and baselines

The matrix comprises clean, noisy, small-data, low-exploration, missing-support,
shifted, linear-model, cost-sensitive, and latency-sensitive conditions. Each
uses seeds 7, 17, 23, 31, and 47. Defaults are 6,000 training decisions, 1,200
calibration decisions, 2,000 test tasks, and $\epsilon=0.3$. Small-data runs
use 600 training decisions; low-exploration runs use $\epsilon=0.05$.
Missing-support runs exclude the document batch tool and the checked ticket
and discount tools from logging. Cost-sensitive runs use $\lambda_c=3$;
latency-sensitive runs use $\lambda_l=0.5$.

Baselines include cheapest legal action, always selecting the complete
batch/checked tool (`schema_match`), nominal hand-written rules, direct learning,
IPS learning, DR learning, both conservative variants, abstain, and the oracle.
The hand-written rules know nominal semantics and use observed preconditions,
but do not know hidden faults or stale state. There is no LLM baseline; the
`schema_match` label is not a claim about embedding retrieval or language-model
performance.

All five seeds and every condition are retained. The matrix yields
{{decisions}} selected-action log records and {{task_conditions}} held-out
task-condition instances. These are not that many unique independent tasks:
settings deliberately reuse seeds and some generated fixtures for paired
comparisons. Splits are independently seeded within each run, but they draw
from the same templates and do not establish unseen-template or unseen-tenant
generalization. Source hashes and per-run summaries are published.

### 6.3 Metrics and statistical reporting

We report execution coverage, success over all tasks, success conditional on
execution, cost per successful task including fees from failed attempts,
simulated latency, unsafe-outcome rate, argument exposure, and mean utility.
No successful execution is credited to abstention. Attempts denied at execution
are reported separately. Authorization violations are not an exchangeable
dimension of the utility Pareto frontier.

Per-run estimates use 200 task-level bootstrap resamples. Tables show mean
and sample standard deviation across five seeds. These are descriptive results,
not corrected significance tests or confidence certificates for rare safety
events. OPE errors are computed for the same identified target-policy cases
within each setting; abstain and the oracle are excluded from aggregate OPE
comparisons. Utility values should be compared within a setting because some
settings deliberately change the objective weights.

## 7. Results

### 7.1 Policy learning does not uniformly favor DR

Table 1 reports utility. Standard deviations across seeds appear in parentheses.

{{utility_table}}

In the noisy tree setting, direct learning achieves {{noisy_direct}}, compared
with {{noisy_dr}} for DR and {{noisy_rules}} for rules. With small training
data, direct and DR utilities are {{small_direct}} and {{small_dr}}. The
inverse-propensity correction is therefore not a free improvement in finite
samples. Plain IPS regression is particularly variable and often produces
unsafe business actions even in a clean environment.

The linear setting is an exception: DR reaches {{linear_dr}} versus
{{linear_direct}} for direct learning, a mean paired difference of
{{linear_paired}} across seeds. However, IPS reaches {{linear_ips}}, slightly
above DR. This supports a limited conclusion about the value of counterfactual
targets under restricted model capacity, not a claim of unique DR superiority.

![Figure 1. Policy utility and unsafe outcomes across the frozen matrix. Points and error bars show seed means and standard deviations; objectives differ in the last two settings.](../artifacts/v1/policy-comparison.png)

### 7.2 DR is more useful as a value estimator

Table 2 reports OPE mean absolute error against independent reset execution on
the same test tasks. Cases count identified target-policy/seed pairs, not
independent studies. The targets are the seven nontrivial policies listed in
the experiment protocol.

{{ope_table}}

Under shifted conditions, direct-model error is {{shifted_dm_error}}, while
DR error is {{shifted_dr_error}}. For the linear model, errors are
{{linear_dm_error}} and {{linear_dr_error}}. Small-data errors are
{{small_dm_error}} and {{small_dr_error}}. Across this fixed matrix, DR has
the lowest mean absolute error among the four compared estimators in every
condition. This is an empirical observation for these targets and generators,
not a universal ordering of estimators.

The shifted experiment evaluates target policies with held-out logs collected
under the shifted test distribution. It does not show that old-distribution
logs alone identify a new-distribution value without covariate-shift assumptions.
The outcome model remains trained on the original noisy distribution, making
residual correction useful while the current test logger supplies valid support.

![Figure 2. Off-policy estimation error on identical identified targets. Missing-support targets are reported as not identified rather than included using extrapolated labels.](../artifacts/v1/ope-comparison.png)

### 7.3 Abstention can lose utility without improving safety

Table 3 expands the noisy setting. Lower cost or risk must be interpreted
together with coverage and success, rather than as a standalone improvement.

{{noisy_table}}

Conservative DR falls to utility {{noisy_conservative}}. Its evidence floor
rejects many strata in a cross-product of domain, observed approval, scope set,
resource count, service state, cache age, amount, and limit. Ensemble disagreement
does not reliably detect stale approvals. The conservative variants consequently
do not establish either a monotonic risk reduction or dominance at matched
coverage. Their poor utility is a result, not an omitted failure.

The complete checked-tool baseline has zero unsafe business outcomes in this
construction because its implementation validates preconditions. Learning can
trade some of that conservative behavior for lower fees and higher coverage,
but that trade is not an authorization relaxation. A practical deployment must
specify a separate acceptable business-risk budget rather than assume that a
weighted reward alone enforces it.

Figure 3 plots the held-out conservative-DR threshold sweep with cost shown by
color and business risk shown separately. It is a set of sampled operating
points, not a proven global Pareto frontier. Where a threshold produces no
executions, conditional success is undefined; such seed points are excluded
from that conditional curve and the accompanying JSON records their count.
Thresholds were specified before test evaluation, not selected using this plot.

![Figure 3. Coverage, conditional success, simulated cost and unsafe business outcomes. All points retain the same hard authorization gate.](../artifacts/v1/operating-curves.png)

### 7.4 Missing support and protocol correctness

When three tools are never explored, full values for the rules and complete-tool
policies are not identified for any of the five seeds. The target actions are
legal; the problem is missing counterfactual evidence. Other restricted learned
policies remain evaluable, but their reduced attainable value is visible in
Table 1. We do not compare errors by inventing outcomes for the excluded tools.

A separate noisy MCP run uses 600 training, 200 calibration, and 400 test tasks.
It performs {{mcp_calls}} actual tool calls through the stdio subprocess and
negotiates protocol {{mcp_protocol}}. Measured round-trip median and 95th
percentile are {{mcp_p50}} and {{mcp_p95}} milliseconds on the development
machine. These are local transport timings, not production service latency or
API cost. Replaying all 600 selected training calls through MCP produces zero
semantic mismatches. Integration tests additionally compare local and MCP
results for clean and shifted fixtures, including server-side revocation.

No unauthorized execution is observed in the evaluated policies. This is
expected from the deterministic mediator and tested implementation; the policies
share that same gate. It is not evidence that the learner itself understands
authorization or that a real enterprise deployment has zero security risk.

## 8. Discussion and Limitations

**Decision learning and policy evaluation are different deliverables.** A
useful first deployment of this architecture may be better logging and policy
evaluation around an existing direct ranker, not replacement of that ranker
with DR pseudo-outcome regression. The experiment provides no reason to discard
the stronger direct baseline simply because the paper's motivation mentions
counterfactual learning.

**Sparse support is an engineering constraint, not an estimator detail.**
Exploration takes place only among approved sandbox actions. Neither a new ACL
grant nor a powerful outcome model creates evidence for a never-tried action.
Real deployments need an explicit exploration or evidence-acquisition policy,
and potentially a human approval path, before extending autonomous action sets.

**The conservative heuristic is inadequate as a safety mechanism.** Its evidence
floor is too coarse to represent all business risks and too fragmented to retain
useful coverage. A subsequent study should compare calibration under matched
coverage, risk-constrained policy selection, and state-dependent information
acquisition. Such changes should be specified before collecting the next test
set, rather than tuned against the current matrix.

**The environment is intentionally limited.** It contains eleven tools, finite
argument templates, three synthetic domains, and one decision per task. It does
not evaluate thousands of MCP tools, semantic retrieval, arbitrary parameter
generation, token-cost prediction, an LLM planner, interactive human approval,
or multi-step policy improvement. It provides local real protocol execution,
not external enterprise services. No claim about BFCL, tau-bench, AgentAbstain,
or a production agent leaderboard follows from these results.

**Authority is simplified.** The access-control model supports principal/group
grants, resource inheritance, deny precedence, and revocation snapshots. It does
not implement a full enterprise IAM system, transitive group graphs, distributed
revocation, or formal verification. The server manifest is trusted. A caller
who can replace that manifest is outside the threat model.

**Utility design is subjective.** Simulated fees, unsafe-outcome penalties,
exposure proxies, and a zero-utility abstain baseline encode particular operator
preferences. Sensitivity settings expose some consequences but do not establish
universal weights. Real deployments must measure costs and user utility, include
the cost of asking or escalating, and distinguish recoverable from irreversible
side effects.

**Statistical conclusions are preliminary.** Five seeds and 200 bootstrap
resamples support an inspectable first study, not a definitive method ranking.
Hyperparameter search is limited, classifiers are not probability-calibrated on
an independent action-level validation target, and interval coverage is not
uniformly guaranteed after selection. Reused task templates and coupled setting
seeds are explicit restrictions on generalization and statistical independence.

## 9. Reproducibility and Data Handling

The public repository contains the simulator, authorized candidate generation,
exact logger, three learners, conservative variants, four OPE estimators, MCP
server/client, tests, fixed experiment protocol, per-run summaries, and generated
tables and figures. The full experiment can be recreated without an API key:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m unittest discover -p 'test_*.py' -v
.venv/bin/python suite.py --out results/reproduction
```

For the protocol check:

```sh
.venv/bin/python run.py run --backend mcp --scenario noisy \
  --train-size 600 --calibration-size 200 --test-size 400 \
  --bootstrap 100 --seed 7 --out results/mcp-reproduction
.venv/bin/python run.py replay --backend mcp \
  --log results/mcp-reproduction/train.jsonl
```

Raw generated logs remain local by default. Published summaries remove local
output paths and expose only synthetic metadata and aggregate outcomes. Raw-log
hashes are recorded for provenance, but measured runtimes make byte-for-byte
hashes machine-specific; semantic replay ignores that one measurement. The
working paper's tables are generated from committed JSON summaries instead of
being independently transcribed. All figures are generated from the same data.

## 10. Conclusion

Counterfactual tool selection is useful only when its execution, logging, and
evaluation contracts are explicit. Our implementation keeps deterministic
authorization separate from learned value and separates legal availability
from historical support. In a reproducible enterprise-inspired workload, DR
evaluation is consistently more accurate than the compared estimators, while
DR learning is not consistently better than direct prediction. Evidence-based
abstention can severely damage coverage without reliably improving business
safety. These results support an evidence-first development path: trustworthy
action-level logs and honest OPE before stronger claims about autonomous
enterprise-agent optimization.

## References

1. Miroslav Dudik, John Langford, and Lihong Li. **Doubly Robust Policy Evaluation and Learning.** ICML, 2011. <https://arxiv.org/abs/1103.4601>.
2. Adith Swaminathan and Thorsten Joachims. **Counterfactual Risk Minimization: Learning from Logged Bandit Feedback.** ICML, 2015. <https://proceedings.mlr.press/v37/swaminathan15.html>.
3. Noveen Sachdeva, Yi Su, and Thorsten Joachims. **Off-policy Bandits with Deficient Support.** KDD, 2020. <https://arxiv.org/abs/2006.09438>.
4. Romain Laroche, Paul Trichelair, and Remi Tachet des Combes. **Safe Policy Improvement with Baseline Bootstrapping.** ICML, 2019. <https://arxiv.org/abs/1712.06924>.
5. Tianneng Shi, Jingxuan He, Zhun Wang, Hongwei Li, Linyu Wu, Wenbo Guo, and Dawn Song. **Progent: Securing AI Agents with Privilege Control.** arXiv:2504.11703, 2025; version 3 revised May 2026. <https://arxiv.org/abs/2504.11703v3>.
6. Jinhao Zhu, Kevin Tseng, Gil Vernik, Xiao Huang, Shishir G. Patil, Vivian Fang, and Raluca Ada Popa. **MiniScope: A Least Privilege Framework for Authorizing Tool Calling Agents.** arXiv:2512.11147, 2025. <https://arxiv.org/abs/2512.11147>.
7. Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. **Tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.** arXiv:2406.12045, 2024. <https://arxiv.org/abs/2406.12045>.
8. Xun Liu et al. **AgentAbstain: Do LLM Agents Know When Not to Act?** arXiv:2607.10059, 2026. <https://arxiv.org/abs/2607.10059>.
9. Nan Jiang and Lihong Li. **Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.** ICML, 2016. <https://proceedings.mlr.press/v48/jiang16.html>.
10. Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita. **Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation.** NeurIPS Datasets and Benchmarks, 2021. <https://arxiv.org/abs/2008.07146>.
