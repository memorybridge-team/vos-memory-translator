# RunPod GPU pilot and artifact sharing

This repository keeps code, configs, metrics, and selected visual previews in
Git. Checkpoints, datasets, raw tensors, and bulk masks stay outside Git.

## Recommended first Pod

- Use a PyTorch image with CUDA and at least 40 GB of persistent volume.
- Prefer one A40 48 GB GPU for the first Tiny→Large pilot. The pipeline currently
  loads source, target oracle, and target handoff models sequentially, so it does
  not require all three predictors to occupy VRAM at once.
- The official price shown on 2026-09-07 was USD 0.49/hour for A40. Reserve USD 2
  of a USD 12 balance for storage and mistakes; a USD 10 compute cap is roughly
  20.4 A40 hours. Availability and the launch-screen price must be checked again
  before creating the Pod.
- Clone this feature branch into `/workspace/CMMT`, then run:

```bash
cd /workspace
git clone --branch kim/exp-sam2-state-translator \
  https://github.com/memorybridge-team/vos-memory-translator.git CMMT
cd /workspace/CMMT
bash scripts/runpod_bootstrap.sh /workspace/CMMT
bash scripts/runpod_direct_smoke.sh /workspace/CMMT
```

Pricing reference: [RunPod GPU pricing](https://www.runpod.io/pricing).

The second command writes a small, Git-friendly result bundle under
`reports/experiments/<UTC-run-id>/`:

- `comparisons/*.png`: input, target-native mask, handoff mask, and error map
- `oracle_masks/*.png` and `candidate_masks/*.png`: binary masks
- `report.json`: full machine-readable metrics and provenance
- `report.md`: preview page that renders inside VS Code and GitHub

Before committing a run, inspect its size and commit only representative PNGs
and reports. Do not force-add checkpoints, datasets, state dumps, or large mask
collections.

## Budget-safe execution order

1. Run the three-frame smoke and open `report.md`.
2. Run a small DAVIS subset and collect paired Tiny/Large state offline.
3. Fit Direct/Ridge/Linear first; train the residual MLP only after data and
   evaluation checks pass.
4. Increase videos and switch points only when the pilot improves downstream
   masks over Direct and the required baselines.
5. Stop or terminate the Pod immediately after syncing the selected report
   bundle and any needed private raw artifacts.

A practical initial cap for A40 is 20 GPU-hours: at most 1 hour for setup/smoke,
6 hours for a small paired-state extraction, 2 hours for Direct/Ridge/Linear,
7 hours for residual-MLP and downstream evaluation, and 4 hours of retry margin.
These are planning limits, not measured runtimes; record the actual time and VRAM
from the first run before expanding the dataset.

Dataset download remains a separate action because the DAVIS downloader requires
explicit acceptance of its dataset terms.
