#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/runpod_direct_smoke.sh /workspace/CMMT"
  exit 2
fi

project_dir="$(cd "$1" && pwd)"
cd "${project_dir}"

run_id="$(date -u +%Y%m%dT%H%M%SZ)_tiny_to_large_direct"
work_dir="outputs/${run_id}"
artifact_dir="reports/experiments/${run_id}"

python scripts/make_synthetic_video.py \
  --output-dir "${work_dir}/video" \
  --frames 3 --height 192 --width 320

sam2-direct-handoff-smoke \
  --sam2-repo .external/sam2 \
  --source-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --source-checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --source-model-id sam2.1-hiera-tiny \
  --target-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --target-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --target-model-id sam2.1-hiera-large \
  --video-dir "${work_dir}/video/frames" \
  --prompt-mask "${work_dir}/video/00000.png" \
  --object-id 1 --switch-frame 1 --device cuda \
  --json "${artifact_dir}/cli_report.json" \
  --artifact-dir "${artifact_dir}"

echo "Open ${artifact_dir}/report.md in VS Code to inspect the result."
