# Cached SAM 2 baseline suite

> One video/object/switch partial DAVIS evaluation; not a full benchmark.

- Sequence: `bike-packing`
- Object: `1`
- Switch frame: `14`
- Evaluated frames: `15–67`
- Last-Mask = Replay-1 sanity check: `True`

| Method | Prompt/input | Prefix frames | J&F | Native IoU | Wall s | Peak GiB | Past backbone | Future backbone |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Direct Copy | `translated_source_state` | 0 | 0.000000 | 0.000000 | 30.511 | 1.537 | 0 | 54 |
| Target Reset (blank object slot) | `blank_mask` | 1 | 0.000000 | 0.000000 | 29.672 | 2.519 | 1 | 54 |
| Last-Mask | `source_prediction_at_switch` | 1 | 0.857995 | 0.973530 | 28.439 | 2.524 | 1 | 54 |
| Replay-1 | `source_prediction_at_replay_start` | 1 | 0.857995 | 0.973530 | 28.417 | 2.528 | 1 | 54 |
| Replay-2 | `source_prediction_at_replay_start` | 2 | 0.859054 | 0.978674 | 28.847 | 2.533 | 2 | 54 |
| Replay-4 | `source_prediction_at_replay_start` | 4 | 0.857450 | 0.977736 | 29.684 | 2.537 | 4 | 54 |
| Full Replay / Large-native | `first_frame_ground_truth` | 15 | 0.861535 | 1.000000 | 34.490 | 2.543 | 15 | 54 |

`target_reset` is a SAM 2 runtime proxy: an all-zero mask registers the object on the switch frame, but no source object information or temporal memory is transferred.
