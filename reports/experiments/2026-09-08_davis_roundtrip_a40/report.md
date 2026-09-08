# DAVIS same-checkpoint round-trip — A40

- Date: 2026-09-08 (KST)
- Dataset: DAVIS 2017 trainval 480p
- Sequence: `blackswan` (50 frames)
- Prompt: object ID 1 from frame 0 ground-truth annotation
- Switch frame: 10
- Compared continuation: frames 11–49 (39 frames)
- Device: NVIDIA A40
- SAM 2 upstream: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Seed: 7

## Results

| Model | Mean binary IoU vs native | Mean logit MSE | Max absolute error | Prefix backbone calls during injection | Future backbone calls | Wall time | Peak CUDA memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM 2.1 Tiny | 0.9998837611 | 0.0113338014 | 2.9432945 | 0 | 39 | 26.69 s | 755,715,584 B |
| SAM 2.1 Large | 0.9999313743 | 0.0010071985 | 0.9222040 | 0 | 39 | 45.48 s | 1,655,799,296 B |

## Interpretation

Both predictors resumed at frame 11 without replaying the prefix backbone.  The
binary masks remained almost identical to each model's native continuation over
all 39 future frames.  The logits were not bit-identical on this real sequence,
so the result supports behavioral continuation closure rather than exact tensor
identity.

This is a one-sequence, one-object round-trip validation.  It does not yet test
multi-object state, interactive continuation, cross-model translation, or DAVIS
J&F against ground truth.

