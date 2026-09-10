# Cached baseline rare-event sweep

- Completed: 10/10 cases
- Scores are unweighted case means; this is not a full DAVIS benchmark.

| Method | Cases | J&F | Visible J&F | Past backbone | Wall s |
|---|---:|---:|---:|---:|---:|
| Direct Copy | 10 | 0.278243 | 0.000000 | 0.00 | 24.893 |
| Target Reset (blank object slot) | 10 | 0.274672 | 0.000000 | 1.00 | 24.079 |
| Last-Mask | 10 | 0.468188 | 0.319945 | 1.00 | 21.482 |
| Replay-1 | 10 | 0.468188 | 0.319945 | 1.00 | 21.729 |
| Replay-2 | 10 | 0.538350 | 0.425541 | 2.00 | 22.744 |
| Replay-4 | 10 | 0.700364 | 0.623753 | 4.00 | 23.432 |
| Full Replay / Large-native | 10 | 0.603838 | 0.483535 | 30.50 | 33.545 |
