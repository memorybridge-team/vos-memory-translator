# VOS Memory Translator

This repository implements the CMMT inspection and translator-baseline
milestones. It records SAM 2.1 compact video state and the assembled tensors
passed to memory attention, normalizes nested/Hugging Face cache containers,
builds a runtime-validated canonical handoff state, and fits Direct, closed-form
Ridge/OLS, Linear, and residual two-layer MLP baselines. Target-owned positional
regeneration is mandatory during SAM 2 history materialization.

The supported SAM 2 upstream revision is pinned in
[`SAM2_UPSTREAM_COMMIT`](SAM2_UPSTREAM_COMMIT). See
[`docs/memory_tensor_inventory.md`](docs/memory_tensor_inventory.md) for the
producer, storage, and consumer paths verified against that revision.

No checkpoint, dataset, prompt token, or tensor dump belongs in Git. Tensor
dumps are opt-in and ignored by `.gitignore`.

## Project documents and layout

Read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) before changing the research
scope, state contract, baselines, or evaluation design.

| Path | Purpose |
| --- | --- |
| `docs/design/` | Translator, state-contract, and experiment design documents |
| `docs/memory_tensor_inventory.md` | Verified SAM 2 memory producer/storage/consumer paths |
| `docs/design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md` | KV paper analysis, SAM 2 state contract, implementation and smoke results |
| `docs/validation.md` | Commands and tests run for the current probe milestone |
| `references/` | Source papers and searchable page-level text extracts |
| `src/vos_memory_inspector/` | Inspection, canonical state, translators, metrics, and experiments |
| `tests/` | Synthetic unit and integration tests |
| `configs/` | Repository-owned experiment configuration files |
| `scripts/` | Reproducible command entry points |
| `data/`, `outputs/` | Local-only datasets and generated artifacts |

## What is measured

- Stored per-frame tensors: `maskmem_features`, every element of
  `maskmem_pos_enc`, `pred_masks`, `obj_ptr`, and `object_score_logits`.
- Actual next-frame consumer inputs: assembled `memory_attention.memory` and
  `memory_attention.memory_pos` captured by a forward pre-hook.
- Shape, dtype, device, mean, population standard deviation, norm, minimum,
  maximum, byte size, and exact change versus the preceding recorded frame.
- Conditioning versus non-conditioning storage and the precise state path.
- Matching Tiny/Large rows, including spatial resolution, channel, dtype, and
  byte-size compatibility.

Statistics-only mode is the default. `--dump-tensor` is required to write any
tensor, and dumps are moved to CPU before `torch.save`.

## Install

Clone and pin the official upstream separately. The probe refuses a different
commit by default and checks the expected private source contract before model
construction.

```bash
git clone https://github.com/facebookresearch/sam2.git
git -C sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
python -m pip install -e ./sam2

git clone https://github.com/memorybridge-team/vos-memory-translator.git
cd vos-memory-translator
git switch kim/cmmt-development
python -m pip install -e ".[dev]"
```

Download the SAM 2.1 Tiny and Large checkpoints from the official SAM 2
checkpoint links. Keep them outside this repository.

## Synthetic tests

```bash
python -m pytest -q
```

The tests use synthetic tensors only. They do not download checkpoints or
datasets.

Run the paired-state translator smoke explicitly:

```bash
cmmt-synthetic-experiment --output-dir outputs/synthetic_cmmt --seed 7
```

This command is clearly marked synthetic in its JSON/Markdown output. It is not
a SAM 2 checkpoint result.

## DAVIS 2017 val input

The downloader requires an explicit dataset-terms acknowledgement and extracts
the official 480p trainval archive with path-traversal checks:

```bash
sam2-davis-download \
  --destination /content/data \
  --accept-dataset-terms

sam2-davis-check \
  --root /content/data/DAVIS \
  --sequence bike-packing
```

The validator requires the sequence to appear in `ImageSets/2017/val.txt` and
returns the frame directory and matching first-frame annotation. DAVIS data is
ignored by Git and must not be copied into the repository.

For a checkpoint smoke run without DAVIS, create deterministic local RGB frames
and a legal synthetic prompt:

```bash
python scripts/make_synthetic_video.py --output-dir /content/smoke
```

## One-model probe

```bash
sam2-memory-probe \
  --sam2-repo /content/sam2 \
  --config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint /content/checkpoints/sam2.1_hiera_tiny.pt \
  --model-id sam2.1-hiera-tiny \
  --video-dir /content/data/DAVIS/JPEGImages/480p/bike-packing \
  --prompt-mask /content/data/DAVIS/Annotations/480p/bike-packing/00000.png \
  --object-id 1 \
  --switch-frame 5 \
  --jsonl /content/probe/tiny.jsonl \
  --csv /content/probe/tiny.csv \
  --canonical-state /content/probe/tiny_state.pt \
  --seed 7
```

To dump only selected CPU tensors, add both `--dump-dir /content/probe/dumps`
and one or more options such as `--dump-tensor maskmem_features`. Dumping is not
needed for compatibility inspection.

## Sequential Tiny/Large run on a Colab T4

The pair script always completes and releases Tiny before constructing Large;
video and state storage are CPU-offloaded for both models.

```bash
python scripts/run_tiny_large_probe.py \
  --sam2-repo /content/sam2 \
  --tiny-checkpoint /content/checkpoints/sam2.1_hiera_tiny.pt \
  --large-checkpoint /content/checkpoints/sam2.1_hiera_large.pt \
  --video-dir /content/data/DAVIS/JPEGImages/480p/bike-packing \
  --prompt-mask /content/data/DAVIS/Annotations/480p/bike-packing/00000.png \
  --object-id 1 \
  --switch-frame 5 \
  --output-dir /content/probe
```

It creates `tiny.jsonl`, `large.jsonl`, CSV equivalents, opt-in canonical
`tiny_state.pt`/`large_state.pt` artifacts,
`compatibility.json`, and `compatibility.md`. A shape-compatible row is labeled
as a direct-copy candidate with semantics explicitly unverified. A learned
linear map is only proposed when tensor rank and spatial resolution agree but
the mapped dimension differs.

After collecting multiple disjoint train/test video-switch pairs, fit the
offline tensor baselines with repeated `--train-source/--train-target` and
`--test-source/--test-target` arguments:

```bash
cmmt-paired-experiment \
  --train-source outputs/train01_tiny.pt --train-target outputs/train01_large.pt \
  --test-source outputs/test01_tiny.pt --test-target outputs/test01_large.pt \
  --output-dir outputs/paired_tiny_to_large --seed 7
```

This command evaluates serialized tensors only; it does not claim downstream
SAM 2 continuation quality.

## Current verification boundary

- Memory-flow inventory: verified from pinned upstream source.
- Synthetic-state and hook tests: implemented; see
  [`docs/validation.md`](docs/validation.md) for commands actually run.
- Tiny/Large checkpoint and DAVIS GPU run: not run in the local CPU-only
  environment; no compatibility result is claimed.
- Translator fitting: implemented and verified on synthetic paired state only.
- Target history materialization: implemented behind a required target
  positional factory; actual checkpoint next-frame injection is not yet run.

Private API risks and the translator candidate rationale are documented in
[`docs/memory_tensor_inventory.md`](docs/memory_tensor_inventory.md).
The exact KV-cache analysis, component policies, formulas, commands and current
blockers are in
[`docs/design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md`](docs/design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md).

