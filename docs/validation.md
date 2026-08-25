# Validation record

This file distinguishes source/synthetic verification from checkpoint and
dataset experiments.

## Environment used

- Date: 2026-08-24
- Host OS: Windows
- Test Python: 3.14
- PyTorch: 2.13.0 (CPU wheel)
- Official SAM 2 checkout:
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`

## Commands actually run

Repository state and branch:

```powershell
git remote -v
git status --short --branch
git switch -c feature/sam2-memory-inspection
```

Pinned upstream and private source contract:

```powershell
python -c "from vos_memory_inspector.upstream import verify_sam2_checkout; print(verify_sam2_checkout(r'C:\path\to\upstream-sam2'))"
```

Observed result:

```text
2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Synthetic tests:

```powershell
python -m pytest -q
```

Observed result after the initial implementation:

```text
4 passed in 8.39s
```

The suite was extended with DAVIS layout and safe-extraction tests after this
run. The final rerun observed:

```text
6 passed in 9.00s
```

Syntax/import compilation:

```powershell
python -m compileall -q src scripts
```

Observed result: exit code 0 and no output.

## Not run locally

- SAM 2.1 Tiny checkpoint inference on CUDA
- SAM 2.1 Large checkpoint inference on CUDA
- DAVIS 2017 val download or inference
- Tiny/Large numerical or behavioral compatibility evaluation

These require checkpoints, dataset storage, and a CUDA runtime. The scripts and
validators are present, but no GPU/DAVIS success is inferred from synthetic
tests.

## 2026-08-25 canonical-state/translator milestone

Environment used for this rerun:

- Windows 11 `10.0.26200`
- Python `3.13.0`
- PyTorch `2.13.0+cpu`
- CUDA available: `False`
- Isolated environment: `.venv`

Commands actually run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -c "from pathlib import Path; from vos_memory_inspector.paired_experiment import run_synthetic_experiment; run_synthetic_experiment(Path('outputs/synthetic_cmmt'), seed=7, epochs=120)"
```

Observed results:

```text
compileall: exit code 0
pytest: 12 passed in 5.64s (latest warm rerun; earlier cold run 13.05s)
synthetic Direct aggregate MSE: 1.447550
synthetic Ridge aggregate MSE: 0.030507 (97.89% improvement over Direct)
synthetic Linear aggregate MSE: 0.111864 (92.27% improvement over Direct)
synthetic residual MLP aggregate MSE: 0.019728 (98.64% improvement over Direct)
```

The paired-state run uses generated tensors with an explicitly labelled fixed
synthetic nonlinear relation. It validates schema, fitting, metrics, serialization
and shape-adapter paths only. No SAM 2 checkpoint, video, J&F evaluation or actual
next-frame injection was run.
