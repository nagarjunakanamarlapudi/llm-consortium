## humanevalplus — pass@1

| condition | problems | pass@1 | 95% CI | pass@k |
|---|---:|---:|---|---:|
| v1a_gpt52 | 164 | 90.9% | [86.0, 95.1] | 90.9% |
| v1a_gptoss | 164 | 93.3% | [89.0, 97.0] | 93.3% |
| v1a_sonnet | 164 | 94.5% | [90.9, 97.6] | 94.5% |
| v2b_xreview | 164 | 87.8% | [82.9, 92.7] | 87.8% |
| v4b_adv | 164 | 91.5% | [87.2, 95.7] | 91.5% |

### McNemar vs v1a_sonnet

| condition | both | neither | cond-only | base-only | p-value |
|---|---:|---:|---:|---:|---:|
| v1a_gpt52 | 145 | 5 | 4 | 10 | 0.1796 |
| v1a_gptoss | 148 | 4 | 5 | 7 | 0.7744 |
| v2b_xreview | 141 | 6 | 3 | 14 | 0.0127 |
| v4b_adv | 146 | 5 | 4 | 9 | 0.2668 |
