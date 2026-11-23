| task | model | predictor params | total params | median steps/s | relative speed |
|---|---|---|---|---|---|
| Path-Star G(7,7) | NextLat (baseline) | 590,592 | 21.9M | 11.45 | 1.00x |
| Path-Star G(7,7) | Attend, learned gate | 590,849 | 21.9M | 10.67 | 0.93x |
| Path-Star G(7,7) | Attend, uniform gate | 590,849 | 21.9M | 10.92 | 0.95x |
| Manhattan | NextLat (baseline) | 4,129,536 | 92.6M | 3.53 | 1.00x |
| Manhattan | Attend, learned gate, share 0.10 | 4,129,729 | 92.6M | 2.08 | 0.59x |
| Manhattan | Attend, learned gate, share 0.25 | 4,129,729 | 92.6M | 3.02 | 0.86x |
| Manhattan | Attend, learned gate, share 0.50 | 4,129,729 | 92.6M | 1.87 | 0.53x |

Steps/s = median logged training throughput (one H100 NVL per run, compiled, bf16); relative to the NextLat baseline on the same task. Manhattan wall-clock to 255k steps incl. in-training evals: 25.8 h (baseline), 29.3 h (learned gate).
