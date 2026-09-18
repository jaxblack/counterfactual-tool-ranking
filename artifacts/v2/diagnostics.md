# Public-data diagnostics

No refitting or threshold selection was performed for these diagnostics.

## Native BFCL-derived selection, mean across fixed splits

| Policy | Accuracy | Balanced accuracy | Coverage |
| --- | ---: | ---: | ---: |
| always_abstain | 69.22% | 50.00% | 0.00% |
| tfidf | 70.47% | 69.23% | 43.00% |
| direct | 86.42% | 81.85% | 27.63% |
| ips | 84.44% | 79.63% | 28.06% |
| dr | 84.09% | 79.83% | 28.83% |

## Post-run candidate-count shortcut diagnostic

All 508 single-candidate rows are should-abstain tasks. The following cohort contains only test tasks with at least two candidates; no model was retrained.

| Policy | Accuracy | Balanced accuracy | Coverage |
| --- | ---: | ---: | ---: |
| always_abstain | 40.37% | 50.00% | 0.00% |
| tfidf | 63.76% | 64.89% | 64.19% |
| direct | 74.19% | 76.80% | 54.04% |
| ips | 71.47% | 74.17% | 53.12% |
| dr | 70.25% | 72.26% | 55.38% |

## Matched task subset: qwen-0.5b

All compared methods use exactly the same held-out tasks within each seed. Task IDs can repeat across seeds.

| Policy | Accuracy | Balanced accuracy | Coverage |
| --- | ---: | ---: | ---: |
| llm | 23.19% | 34.33% | 100.00% |
| always_abstain | 66.83% | 50.00% | 0.00% |
| tfidf | 70.90% | 70.41% | 43.64% |
| direct | 85.95% | 82.36% | 30.16% |
| dr | 83.76% | 80.44% | 29.67% |

## Matched task subset: qwen-1.5b

All compared methods use exactly the same held-out tasks within each seed. Task IDs can repeat across seeds.

| Policy | Accuracy | Balanced accuracy | Coverage |
| --- | ---: | ---: | ---: |
| llm | 27.68% | 41.73% | 100.00% |
| always_abstain | 66.83% | 50.00% | 0.00% |
| tfidf | 70.90% | 70.41% | 43.64% |
| direct | 85.95% | 82.36% | 30.16% |
| dr | 83.76% | 80.44% | 29.67% |

Intervals in diagnostics.json resample connected tool/query groups, not individual rows.
LLMs also generated parameters; only exact function selection is compared here. Small-model results do not establish frontier-model superiority.
