# Public Data and Model Provenance

No production enterprise data or private trajectories are used. Public datasets
were explicitly requested for the version-2 extension.

## Berkeley Function Calling Leaderboard

- Publisher: Gorilla LLM / UC Berkeley.
- Dataset: <https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard>.
- Revision: `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`.
- Declared license: Apache-2.0 on the upstream dataset card.
- Files: live multiple questions and their `possible_answer` references, live
  irrelevance questions, and the upstream dataset card.
- Exact URLs, byte counts and SHA256 digests: `artifacts/v2/dataset.json`.
- Citation: Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang,
  Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez. Berkeley Function Calling
  Leaderboard, 2024. <https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html>.

The adapter does not redistribute the original question text or reference
arguments. It downloads the pinned source, validates hashes, and publishes
derived task-ID splits and aggregate results. The retained set is 1,930 of 1,935
question rows. Missing exact reference IDs and duplicate function names are
listed in the dataset accounting; no guessed row-order labels are used.

This is a **BFCL-derived function-selection study**, not an official BFCL AST,
parameter-semantic, execution or leaderboard result. Tool descriptions are not
executed. No API keys, URLs, shell instructions or filesystem requests found in
the public text are acted on. Original labels are visible only to the scorer.

Bandit feedback is newly sampled with controlled logged probabilities. Costs,
permission masks and deficient-support settings are explicit experimental
transformations; none is presented as original BFCL or production logging.

## Local Open-Weight Models

Both source and MLX-converted model cards declare Apache-2.0. Converted weights
are downloaded to a gitignored local cache; no weights are redistributed.

| Model | MLX revision |
| --- | --- |
| [Qwen2.5-0.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit) | `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` |
| [Qwen2.5-1.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit) | `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` |

Original model family: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct> and
<https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>. Inference implementation:
<https://github.com/ml-explore/mlx-lm>, version 0.31.3. Weight hashes, package
versions, prompts, task IDs, token counts and timing are in the model summaries.

Raw model text remains local. Model pretraining overlap with BFCL is unknown.
These small quantized models are real LLM baselines, not evidence about all
LLMs or frontier systems. JSON-schema validity is not parameter semantic accuracy.

## Other Candidates

Salesforce xLAM-function-calling-60k was examined but not used because its
repository requires acceptance of additional access conditions. No authenticated
dataset, enterprise account, or private API was accessed for version 2.
