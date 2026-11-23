| model | step | valid trajectories | sequence compression | effective rank (↓) | detour robustness |
|---|---|---|---|---|---|
| ours: NextLat (baseline) | 255001 | 96.5% | 0.59 | 82.6 | 88.4% (n=1000) |
| ours: Attend, learned gate, share 0.10 | 255001 | 96.8% | 0.70 | 142.2 | 84.1% (n=1000) |
| ours: Attend, learned gate, share 0.25 | 255001 | 96.3% | 0.59 | 125.0 | 85.5% (n=1000) |
| ours: Attend, learned gate, share 0.50 | 255001 | 95.5% | 0.63 | 117.7 | 81.9% (n=1000) |
| paper: GPT | 6 epochs | 97.0% | 0.65 | 160.1 | 85.0% |
| paper: MTP | 6 epochs | 98.1% | 0.64 | 57.7 | 95.0% |
| paper: JTP | 6 epochs | 97.1% | 0.32 | 215.8 | 87.0% |
| paper: NextLat | 6 epochs | 98.7% | 0.71 | 52.7 | 95.0% |
