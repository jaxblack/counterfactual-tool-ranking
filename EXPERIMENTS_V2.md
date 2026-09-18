# Version 2: Fixed Controls and Independent Public Evidence

The original version-1 source and 45-run artifacts are unchanged. This protocol
is frozen before the full version-2 runs; small smoke runs are excluded.

## Questions and contribution

1. Does the OPE advantage survive direct regression of **complete realized
   utility**, or was part of it caused by nominal-cost misspecification?
2. Can learned selection generalize to held-out function families in an
   independently released dataset rather than only our own simulator?
3. Can incremental policy changes be identified when unsupported actions shared
   by target and baseline prevent identifying their separate absolute values?

The third question is studied using disagreement-support diagnostics, sharp
bounded-outcome identification intervals and an agreement-preserving fallback.
This is a concrete method extension to test, not a claim to have invented DR,
baseline bootstrapping, or partial identification. Existing safe-policy-improvement
and deficient-support work remain necessary prior art. Novelty and generality
cannot be established merely by writing a new method name.

## Realized-return controls

- Seeds 7, 17, 23, 31, 47; 6,000 training and 2,000 independent test tasks.
- Settings: clean, noisy, shifted, linear, cost weight 3, latency weight 0.5.
- All policies share the same legal and empirically supported candidate boundary.
- Nominal direct, complete realized-return direct, separately predicted realized
  components, nominal DR, complete-return DR, and IPS are retained.
- Three-fold cross-fitting; existing 48-tree/depth-12/leaf-3 model family or Ridge
  alpha 10. Full direct and full DR use the same scalar-return nuisance model.
- OPE compares nominal and full-return nuisance models on identical frozen
  target-policy cases, with actual reset execution as the comparison value.

## Public dataset

Source: [BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard),
revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`, Apache-2.0 as declared by its
dataset card. The live multiple and live irrelevance files contain 1,935 rows.
One question has no exactly matching reference ID, and four rows have duplicate
function names. They are transparently excluded; 1,930 remain. An orphan answer
is recorded rather than joined by row position.

Citation: Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G.
Patil, Ion Stoica, Joseph E. Gonzalez. Berkeley Function Calling Leaderboard,
2024. <https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html>.

Split connected components of normalized function names and duplicate normalized
queries. Every function-name family is disjoint across train/calibration/test.
One 1,544-row component exceeds half the data and is always assigned to training;
other groups use a seeded SHA256 60/20/20 assignment. With full role-labeled
context this yields only 57--132 test tasks and 20--31 test groups per seed.
Report these actual sizes. Do not claim
1,930 independent held-out tests or resample seeds to hide the grouping constraint.

Full public message context is preserved with explicit role labels, including
system and assistant messages. It is supplied as task data, not executed as host
instructions. Word/character TF-IDF is fitted on training observations only. Supervised labels,
dataset categories, task IDs and reference parameters are absent from features.
Cross-fitting uses GroupKFold over training function families. The logging policy
is newly generated epsilon-greedy (epsilon 0.3), with independent split-specific
sampling seeds. It reveals one chosen action label. These are **semi-synthetic
bandit logs**, not original production or BFCL interaction propensities.

Settings are native, authorization and support-gap. Native utility is +1 for a
correct function and -1 for an incorrect function; abstain has utility 0 and is
scored separately for selection accuracy. Native has no injected fee or ACL.
Authorization independently masks 20% of functions by a seeded name hash.
Support-gap gives 25% of functions zero logging probability. Those two derivative
settings add an explicit schema-length cost proxy capped at 0.05, not actual API
fees. No external endpoints in tool descriptions are executed.

Compare TF-IDF (fixed abstain similarity 0.08), full-return direct, IPS, DR,
disagreement fallback, blanket support abstention, and a conservative calibration
switch. The calibration switch promotes the supported-disagreement policy only
when a conditional Hoeffding lower bound on its improvement is positive. Its
guarantee concerns the finite calibration contexts and randomized logging, not
unseen deployment distributions; indecisive bounds and no promotions are results.

The task is **function selection**, not the complete BFCL AST/execution metric.
All tables must say this and must not be presented as official leaderboard scores.

## Real local LLM baselines

- Models: MLX-community Qwen2.5-0.5B-Instruct-4bit and
  Qwen2.5-1.5B-Instruct-4bit; both Apache-2.0.
- Pinned revisions are recorded in `public_llm.py` and each result summary.
- Exactly the same 200 SHA256-selected tasks from the union of the five test
  splits; compare learned methods only on the corresponding matched subset.
- Temperature 0; 192 output-token cap; 8,192 input-token cap; independent context
  for every request. Overlong inputs and malformed outputs are counted, not dropped.
- No reference answers, provider keys or production services are used for inference.
- Report strict JSON validity, exact function-selection accuracy, parameter-schema
  validity, token counts and local wall time separately. Schema validity is not
  argument semantic correctness. These small models are not frontier-model proxies.
- Pretraining overlap with BFCL cannot be ruled out.

## Reproduction

```sh
.venv/bin/python -m unittest test_v2 -v
.venv/bin/python v2_experiments.py --out results/v2 --publish artifacts/v2
.venv/bin/python -m pip install -r requirements-llm.txt
.venv/bin/python public_llm.py --model 0.5b --out results/llm-v2-0.5b
.venv/bin/python public_llm.py --model 1.5b --out results/llm-v2-1.5b
```

Raw dataset downloads, model weights, new feedback logs and raw model responses
remain local. Publish source revisions, hashes, split assignments, model choices,
per-run summaries and aggregate metrics. Do not alter v1 values after viewing v2.

## Input-audit correction before publication

The initial v2 adapter extracted only user messages. A publication audit found
86 source rows with system/assistant context. This was an input-fidelity defect,
not a reason to exclude those rows. The adapter now preserves all textual roles.
The initial runs and raw model responses remain in gitignored local directories;
the official v2 matrix and both model runs are regenerated with the same seeds,
models, prompt template and caps. No thresholds or model parameters are changed
in response to the first results. Published v2 artifacts use full-role context.
