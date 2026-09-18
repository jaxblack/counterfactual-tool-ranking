# Version-two results

All listed seeds retained; values are means, not claims of significance.

## Realized-return control policy utility

| Setting | Nominal direct | Full direct | Component direct | Nominal DR | Full DR |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.5922 | 0.5918 | 0.5860 | 0.5852 | 0.5875 |
| noisy | 0.5117 | 0.5135 | 0.5074 | 0.5027 | 0.5029 |
| shifted | 0.3926 | 0.3952 | 0.3917 | 0.3814 | 0.3816 |
| linear | 0.4661 | 0.4658 | 0.4658 | 0.4778 | 0.4737 |
| cost_sensitive | 0.4243 | 0.4244 | 0.4261 | 0.4199 | 0.4212 |
| latency_sensitive | 0.4355 | 0.4377 | 0.4344 | 0.4332 | 0.4326 |

## OPE MAE for identical target-policy cases

| Setting | Nominal DM | Full DM | Component DM | Nominal DR | Full DR |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.0016 | 0.0016 | 0.0098 | 0.0010 | 0.0014 |
| noisy | 0.0179 | 0.0183 | 0.0155 | 0.0110 | 0.0118 |
| shifted | 0.0956 | 0.0948 | 0.0932 | 0.0262 | 0.0227 |
| linear | 0.0870 | 0.0139 | 0.0139 | 0.0240 | 0.0272 |
| cost_sensitive | 0.0138 | 0.0178 | 0.0127 | 0.0092 | 0.0080 |
| latency_sensitive | 0.0197 | 0.0166 | 0.0139 | 0.0114 | 0.0111 |

## BFCL-derived held-out function selection accuracy

Native has no injected fees or permissions; other rows are explicit transformations. Not official BFCL scores.

| Setting | TF-IDF | Direct | IPS | DR | Disagreement fallback | Blanket abstain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 70.47% | 86.42% | 84.44% | 84.09% | 84.09% | 84.09% |
| authorization | 75.35% | 89.37% | 87.18% | 84.26% | 84.26% | 84.26% |
| support_gap | 70.47% | 85.49% | 82.25% | 81.13% | 78.29% | 77.90% |
