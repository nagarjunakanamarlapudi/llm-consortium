# Thesis Prediction Validation (§10.1)

| ID | Prediction | Supported? | p-value | Evidence |
|---|---|---|---|---|
| P1 | v1 matches consortium quality on simple tasks | ❌ | N/A | Insufficient data |
| P2 | v3 achieves highest peak quality on complex tasks | ❌ | N/A | No complex tasks |
| P3 | v7 has lowest variance across repetitions | ❌ | N/A | Lowest CV: v3 (CV=0.1676), v7 CV=0.19068089824035847 |
| P4 | v4 has highest variance across repetitions | ❌ | N/A | Highest CV: v1 (CV=0.2609), v4 CV=0.21325034272358154 |
| P5 | v5 outperforms v2 on specialist dimensions (security, scalability) | ❌ | 0.2688 | v5 specialist mean=3.116, v2 specialist mean=3.025, U=1847.5, p=0.2688 |
| P6 | Rubric awareness matters more than topology | ✅ | 0.0000 | Rubric-aware mean=2.834, Non-rubric mean=2.518, U=16146.0, p=0.0000 |
| P7 | Consortium variants (v2-v8) have higher coherence than single-LLM baseline (v1) | ✅ | 0.0401 | Consortium mean coherence=0.893, v1 mean=0.833, U=6389.5, p=0.0401 |
| P8 | Quality and coherence are positively correlated (r > 0.3) | ❌ | 0.0000 | Spearman r=-0.309, p=0.0000, n=320 designs |

**2/8 predictions supported by data.**