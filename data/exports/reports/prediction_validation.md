# Thesis Prediction Validation (§10.1)

| ID | Prediction | Supported? | p-value | Evidence |
|---|---|---|---|---|
| P1 | v1 matches consortium quality on simple tasks | ✅ | 0.1745 | Mann-Whitney U=12.5, p=0.1745 (NS means v1 competitive) |
| P2 | v3 achieves highest peak quality on complex tasks | ❌ | N/A | No complex tasks |
| P3 | v7 has lowest variance across repetitions | ❌ | N/A | Lowest CV: v1 (CV=0.0136), v7 CV=N/A |
| P4 | v4 has highest variance across repetitions | ❌ | N/A | Highest CV: v2 (CV=0.1053), v4 CV=N/A |
| P5 | v5 outperforms v2 on specialist dimensions (security, scalability) | ❌ | N/A | Missing v5 or v2 |
| P6 | Rubric awareness matters more than topology | ❌ | N/A | Insufficient data |
| P7 | Consortium variants (v2-v8) have higher coherence than single-LLM baseline (v1) | ❌ | 0.9427 | Consortium mean coherence=0.600, v1 mean=0.889, U=3.0, p=0.9427 |
| P8 | Quality and coherence are positively correlated (r > 0.3) | ✅ | 0.0195 | Spearman r=0.791, p=0.0195, n=8 designs |

**2/8 predictions supported by data.**