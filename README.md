# VOS Memory Translator

This repository currently implements the inspection milestone that precedes any
Tiny-to-Large translator training. It records SAM 2.1 compact video state and the
assembled tensors that are passed to memory attention.

The supported SAM 2 upstream revision is pinned in
[`SAM2_UPSTREAM_COMMIT`](SAM2_UPSTREAM_COMMIT). See
[`docs/memory_tensor_inventory.md`](docs/memory_tensor_inventory.md) for the
producer, storage, and consumer paths verified against that revision.

No checkpoint, dataset, prompt token, or tensor dump belongs in Git. Tensor
dumps are opt-in and ignored by `.gitignore`.

## Status

- Memory-flow inventory: complete for the pinned upstream source.
- Synthetic-state probe: implementation in progress.
- Tiny/Large checkpoint run: requires checkpoints and a CUDA runtime; no result
  is claimed yet.
- Translator fitting or state injection: intentionally out of scope.

