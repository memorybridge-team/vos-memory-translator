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
6 passed in 7.50s
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
