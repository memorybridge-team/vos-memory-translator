# Cached SAM 2 baseline suite

> One video/object/switch partial DAVIS evaluation; not a full benchmark.

- Sequence: `india`
- Object: `3`
- Switch frame: `35`
- Evaluated frames: `36–79`
- Last-Mask = Replay-1 sanity check: `True`

| Method | Prompt/input | Prefix frames | J&F | Visible J&F | Native IoU | Wall s | Peak GiB | Past backbone | Future backbone |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct Copy | `translated_source_state` | 0 | 0.204545 | 0.000000 | 0.222222 | 27.875 | 1.537 | 0 | 45 |
| Target Reset (blank object slot) | `blank_mask` | 1 | 0.204545 | 0.000000 | 0.222222 | 26.530 | 2.519 | 1 | 45 |
| Last-Mask | `source_prediction_at_switch` | 1 | 0.204545 | 0.000000 | 0.222222 | 25.954 | 2.524 | 1 | 45 |
| Replay-1 | `source_prediction_at_replay_start` | 1 | 0.204545 | 0.000000 | 0.222222 | 25.526 | 2.528 | 1 | 45 |
| Replay-2 | `source_prediction_at_replay_start` | 2 | 0.552702 | 0.437682 | 0.584968 | 26.243 | 2.533 | 2 | 45 |
| Replay-4 | `source_prediction_at_replay_start` | 4 | 0.868743 | 0.834992 | 0.956255 | 27.039 | 2.537 | 4 | 45 |
| Full Replay / Large-native | `first_frame_ground_truth` | 36 | 0.895240 | 0.868302 | 1.000000 | 39.630 | 2.543 | 36 | 45 |

`target_reset` is a SAM 2 runtime proxy: an all-zero mask registers the object on the switch frame, but no source object information or temporal memory is transferred.

Visible J&F averages only frames where the selected object exists in DAVIS GT. It prevents empty-GT/empty-prediction frames from making a failed reappearance look successful.
