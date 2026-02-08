# Thesis Prediction Validation (§10.1)

| ID | Prediction | Supported? | p-value | Evidence |
|---|---|---|---|---|
| P1 | v1 matches consortium quality on simple tasks | ❌ | N/A | Insufficient data |
| P2 | v3 achieves highest peak quality on complex tasks | ❌ | N/A | No complex tasks |
| P3 | v7 has lowest variance across repetitions | ❌ | N/A | Lowest CV: v1 (CV=0.0000), v7 CV=0.0 |
| P4 | v4 has highest variance across repetitions | ❌ | N/A | Highest CV: v8 (CV=0.0000), v4 CV=0.0 |
| P5 | v5 outperforms v2 on specialist dimensions (security, scalability) | ❌ | N/A | Insufficient dimension data |
| P6 | Rubric awareness matters more than topology | ❌ | 0.5588 | Rubric-aware mean=2.225, Non-rubric mean=2.450, U=8.0, p=0.5588 |
| P7 | Consortium variants (v2-v8) have higher coherence than single-LLM baseline (v1) | ❌ | N/A | Insufficient data: v1 n=1, consortium n=7 |
| P8 | Quality and coherence are positively correlated (r > 0.3) | ❌ | 0.4703 | Spearman r=-0.300, p=0.4703, n=8 designs |

**0/8 predictions supported by data.**