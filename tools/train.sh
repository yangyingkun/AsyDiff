#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
config="configs/AsyDiff/WV3/config/Trans_trans.yaml"
mkdir -p log_file
exec python train.py --base "$config" -t --gpus 0, "$@"
