#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/runpod_bootstrap.sh /workspace/CMMT"
  exit 2
fi

project_dir="$(cd "$1" && pwd)"
sam2_dir="${project_dir}/.external/sam2"
checkpoint_dir="${project_dir}/checkpoints"
sam2_commit="2b90b9f5ceec907a1c18123530e92e794ad901a4"

python -m pip install --upgrade pip
python -m pip install -e "${project_dir}[dev]"

if [[ ! -d "${sam2_dir}/.git" ]]; then
  git clone https://github.com/facebookresearch/sam2.git "${sam2_dir}"
fi
git -C "${sam2_dir}" fetch origin "${sam2_commit}"
git -C "${sam2_dir}" checkout --detach "${sam2_commit}"
SAM2_BUILD_ALLOW_ERRORS=0 python -m pip install -v -e "${sam2_dir}"

mkdir -p "${checkpoint_dir}"
for checkpoint in sam2.1_hiera_tiny.pt sam2.1_hiera_large.pt; do
  if [[ ! -f "${checkpoint_dir}/${checkpoint}" ]]; then
    curl -fL \
      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/${checkpoint}" \
      -o "${checkpoint_dir}/${checkpoint}"
  fi
done

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
